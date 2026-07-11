# -*- coding: utf-8 -*-
"""
Mechanics fitting module for bulk MFAE analysis.

Provides three tiers of fitting:

1. VISCOELASTIC (ASP cells only)
   Fits Kelvin-Voigt (2 params), Jeffreys (3 params), and Burgers (4 params)
   to the full aspiration protrusion trace. Selects the winner by BIC.

2. MODEL-INDEPENDENT (both ASP and EP)
   Fits a linear (L = mt + b) and power-law (L = a*t^b) to the aspiration
   phase of the protrusion trace. For ASP this is the full trace; for EP it
   is the pre-pulse window only (the portion where the cell is aspirated
   before electroporation). This lets you compare slope and exponent directly
   between conditions without committing to a rheological model.
   Selects the better model by BIC.

3. PRE/POST-PULSE SLOPES (EP only)
   Linear slope over the 5 s before and 2 s after the pulse.

Additionally, fits an exponential uptake curve A*(1-exp(-t/tau)) to each
uptake region (Body, Protrusion, Total) and stores the time constant tau.

All physics functions are self-contained copies of Calculation_MFA.py /
config_schema.py equivalents so this module does not need to import from
the main pipeline tree.

Usage
-----
    import bulk_mechanics as bm

    r_eff        = bm.DEFAULT_R_EFF   # 6.7 x 5.0 um channel default
    mechanics_df = bm.run_all_mechanics(grouped_data, r_eff=r_eff)
    mechanics_df.to_csv(results_dir / "mechanics_results.csv", index=False)
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

import bulk_file_handling as bfh

logger = logging.getLogger(__name__)

# =============================================================================
# 1.  DEVICE GEOMETRY DEFAULTS
# =============================================================================

CHANNEL_WIDTH_UM  = 6.7   # microfluidic channel width  [um]
CHANNEL_HEIGHT_UM = 5.0   # microfluidic channel height [um]
HALFSPACE_C       = 1.0   # geometric half-space constant (Davidson convention)

# Minimum post-pulse slope (µm/s) required to call a protrusion as extending
# or retracting.  Below this threshold the protrusion is classified as stable.
# Adjust based on measurement noise floor.
EP_BEHAVIOR_SLOPE_THRESHOLD: float = 0.02

# Pre-pulse dye contamination threshold (MinMax scale, 0–1).
# If Protrusion_MinMax or Body_MinMax exceeds this value in any frame before
# the pulse, the cell is flagged as contaminated (either Pre_Leaky or
# Pre_Loaded -- see BLOCK D in _run_trap_mechanics).
#
# Empirical gap from the training set: clean cells peak at ≤ 0.04 pre-pulse;
# contaminated cells start at ≥ 0.43.  0.10 sits comfortably in the middle
# and is conservative enough to tolerate modest baseline drift without
# generating false positives.
PRE_PULSE_MINMAX_THRESHOLD: float = 0.10

# ---- Viscoelastic identifiability & bound-hit guards ----
# Post-hoc rejection of parameters that converge onto their bound.
# When a fitted parameter lies within this fraction of the (log-scale)
# distance between its lower and upper bound, it is treated as unidentified:
# the optimiser wandered in a flat direction until the wall stopped it.
# Any model with a bound-hitting parameter is dropped from BIC selection.
# Tune upward (e.g. 0.05) to be more aggressive, downward (e.g. 0.01) to
# tolerate parameters that genuinely lie near the boundary.
BOUND_PROXIMITY_FRAC: float = 0.02

# Pre-fit identifiability guard for the long-term flow term.
# Jeffreys and Burgers both contain a series dashpot that produces linear
# creep at long times. If the observed late-stage slope over the observation
# window is smaller than a few times the measurement noise floor, the flow
# parameter cannot be identified from the data; we therefore fit only
# Kelvin-Voigt for those cells. Increase noise_um if the raw traces are
# noisier than 0.3 um; decrease if cleaner.
IDENTIFIABILITY_NOISE_UM: float = 0.3    # length-noise floor [um]
IDENTIFIABILITY_LATE_FRAC: float = 0.33  # fraction of trace treated as "late"
IDENTIFIABILITY_FLOW_SNR: float = 3.0    # late slope * duration must exceed this * noise
IDENTIFIABILITY_PLATEAU_RATIO: float = 0.5  # late/early slope ratio below this = "elbowed"

# =============================================================================
# 2.  GEOMETRY -- Son (2007)
# =============================================================================

def _fstar(width: float, height: float) -> float:
    """
    Son (2007) Eq. 20: hydraulic shape factor for a rectangular duct.

    The sum converges in 11 terms (n = 1, 3, 5, ... 21) -- more than enough
    for floating-point precision.
    """
    dim_min = min(width, height)
    dim_max = max(width, height)
    x = dim_min / dim_max          # aspect ratio, always <= 1
    sum_term = sum(
        math.tanh(math.pi * n * x / 2) / n**5
        for n in range(1, 22, 2)
    )
    return 1.0 / ((1 + 1.0 / x)**2 * (1 - (192 / (math.pi**5 * x)) * sum_term))


def compute_reff(width: float = CHANNEL_WIDTH_UM,
                 height: float = CHANNEL_HEIGHT_UM) -> float:
    """
    Effective radius of a rectangular channel for Laplace / aspiration equations.

    Mirrors compute_reff() in Calculation_MFA.py without importing it.

    Parameters
    ----------
    width, height : channel dimensions [um]

    Returns
    -------
    r_eff : float [um]
    """
    fs  = _fstar(width, height)
    h_s = min(width, height)
    w_l = max(width, height)
    numerator   = (2.0 / (3.0 * math.pi)) * w_l * (h_s ** 3)
    denominator = (1 + h_s / w_l) ** 2 * fs
    return (numerator / denominator) ** 0.25


# Pre-computed for the standard device geometry; callers can override.
DEFAULT_R_EFF: float = compute_reff()

# =============================================================================
# 3.  VISCOELASTIC + EMPIRICAL MODEL FUNCTIONS
# =============================================================================

def _kelvin_voigt(t, r_eff, dp, C, E, eta):
    """Kelvin-Voigt (2 params): Spring E // Dashpot eta. No permanent flow."""
    tau = (3 * math.pi * eta) / (C * E)
    return (r_eff * dp) / (C * E) * (1 - np.exp(-t / tau))


def _jeffreys(t, r_eff, dp, C, E, eta1, eta2):
    """
    Jeffreys (3 params): (E // eta1) in series with eta2.
    Creep time constant: tau = 3*pi*eta1 / (C*E)
    Long-term flow slope: r_eff*dP / (3*pi*eta2)
    """
    tau     = (3 * math.pi * eta1) / (C * E)
    elastic = (r_eff * dp) / (C * E) * (1 - np.exp(-t / tau))
    viscous = (r_eff * dp) / (3 * math.pi * eta2) * t
    return elastic + viscous


def _burgers(t, r_eff, dp, C, E1, eta1, E2, eta2):
    """Burgers (4 params): Maxwell element (E1, eta1) + Kelvin-Voigt (E2, eta2)."""
    maxwell = (r_eff * dp / C) * (1 / E1 + t / (3 * math.pi * eta1))
    tau_kv  = (3 * math.pi * eta2) / (C * E2)
    kv      = (r_eff * dp / (C * E2)) * (1 - np.exp(-t / tau_kv))
    return maxwell + kv


def _linear(t, m, b):
    return m * t + b


def _power_law(t, a, b):
    """L = a * t^b  (epsilon avoids t=0 singularity)"""
    return a * (t + 1e-9) ** b


def _exp_uptake(t, A, tau):
    """A * (1 - exp(-t/tau))"""
    return A * (1 - np.exp(-t / tau))


# =============================================================================
# 4.  STATISTICS HELPERS
# =============================================================================

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def _bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """
    Bayesian Information Criterion.
    Lower is better. Penalises extra free parameters more strictly than AIC.
    Formula: n*ln(RSS/n) + k*ln(n)

    CAVEAT: This assumes i.i.d. Gaussian residuals. Time-series data from
    aspiration traces have autocorrelated residuals, which inflates n and
    can bias selection toward more complex models. Check the Durbin-Watson
    statistic (returned alongside BIC) to assess severity.
    """
    n   = len(y_true)
    if n == 0: return np.inf
    rss = max(float(np.sum((y_true - y_pred) ** 2)), 1e-12)
    return n * math.log(rss / n) + n_params * math.log(n)


def _durbin_watson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Durbin-Watson statistic for residual autocorrelation.

    Values near 2.0 = no autocorrelation (good).
    Values near 0.0 = strong positive autocorrelation (BIC overestimates
    effective sample size; model selection may favour complex models).
    Values near 4.0 = strong negative autocorrelation (unusual for creep).
    """
    residuals = y_true - y_pred
    if len(residuals) < 2:
        return np.nan
    diff = np.diff(residuals)
    ss_res = float(np.sum(residuals ** 2))
    if ss_res < 1e-15:
        return np.nan
    return float(np.sum(diff ** 2) / ss_res)

# =============================================================================
# 5.  DATA CLEANING
# =============================================================================

def _clean_trace(time: np.ndarray, length: np.ndarray,
                 window: int = 11, n_sigma: float = 2.5,
                 noise_floor_um: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes NaN, zero/negative lengths, and rolling-median outliers.

    Why MAD instead of rolling standard deviation?
    Standard deviation is inflated by the very spikes we are trying to
    remove. Median Absolute Deviation (MAD) is unaffected because medians
    are resistant to outliers. Multiplying MAD by 1.4826 converts it to an
    equivalent Gaussian sigma so the n_sigma threshold behaves as expected.

    Parameters
    ----------
    window        : rolling window in frames (forced to odd)
    n_sigma       : how many sigma from local median counts as outlier
    noise_floor_um: minimum scatter assumed even if the cell is perfectly still
    """
    if window % 2 == 0:
        window += 1

    mask = np.isfinite(time) & np.isfinite(length) & (length > 0)
    t, l = time[mask], length[mask]

    if len(t) < window:
        return t, l

    rolling_med = median_filter(l, size=window, mode='nearest')
    dev         = np.abs(l - rolling_med)
    local_mad   = median_filter(dev, size=window, mode='nearest')
    local_std   = np.maximum(local_mad * 1.4826, noise_floor_um)

    inliers = dev <= n_sigma * local_std
    return t[inliers], l[inliers]


# =============================================================================
# 6.  INITIAL PARAMETER ESTIMATION
# =============================================================================

def _estimate_params(t: np.ndarray, l: np.ndarray,
                     r_eff: float, dp: float,
                     C: float) -> Tuple[float, float, float]:
    """
    Heuristic starting point for (E, eta1, eta2).

    - E    : from the plateau (maximum aspiration length)
    - eta2 : from the late-stage slope (viscous flow term)
    - eta1 : from the 63% rise time (creep time constant)

    Mirrors estimate_initial_parameters() in Calculation_MFA.py.
    """
    max_l = max(float(np.max(l)), 1e-9)

    # Elasticity -- from asymptotic plateau of the KV term
    E_guess = (r_eff * dp) / (C * max_l)

    # Series viscosity -- from late-stage linear slope
    n_late = max(5, len(t) // 3)
    late_t, late_l = t[-n_late:], l[-n_late:]
    if len(late_t) > 2 and np.ptp(late_t) > 0:
        slope      = np.polyfit(late_t, late_l, 1)[0]
        eta2_guess = (r_eff * dp) / (3 * math.pi * slope) if slope > 0.01 else 15000.0
    else:
        eta2_guess = 15000.0

    # Parallel viscosity -- from time to reach 63% of max
    target     = 0.63 * max_l
    idx        = int(np.argmin(np.abs(l - target)))
    t_63       = float(t[idx])
    eta1_guess = t_63 * C * E_guess / (3 * math.pi) if t_63 > 0 else 5000.0

    # Clip to physically plausible bounds
    E_guess    = float(np.clip(E_guess,    1.0,   100_000.0))
    eta1_guess = float(np.clip(eta1_guess, 1.0,   500_000.0))
    eta2_guess = float(np.clip(eta2_guess, 500.0, 1_000_000.0))

    if eta2_guess < eta1_guess:
        eta2_guess = eta1_guess * 2.0

    return E_guess, eta1_guess, eta2_guess


# =============================================================================
# 7.  OPTIMISER
# =============================================================================

def _multi_start_fit(func, t: np.ndarray, l: np.ndarray,
                     bounds: Tuple, p0, n_starts: int = 7):
    """
    Runs n_starts Levenberg-Marquardt fits from randomly sampled starting
    points and returns the one with the best R².

    This is a WITHIN-model optimiser: every start uses the same function with
    the same number of free parameters, so R² and BIC rank identically.
    Model *selection* (across different k) is handled by BIC in
    fit_viscoelastic().

    Why multi-start?
    Viscoelastic parameters span several orders of magnitude (E ~ 100-100000 Pa,
    eta ~ 100-1000000 Pa*s). A single starting point frequently converges to a
    local minimum. Sampling log-uniformly (when all lower bounds > 0) covers
    the plausible range far more efficiently than plain uniform sampling.

    Returns
    -------
    (best_popt, best_r2) or (None, -inf) if every attempt failed.
    """
    best_r2, best_p = -np.inf, None
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    safe_lo = np.clip(lower, 1e-9, 1e6)
    safe_hi = np.clip(upper, 1e-9, 1e6)
    use_log = bool(np.all(lower > 0))

    for _ in range(n_starts):
        if use_log:
            guess = np.exp(np.random.uniform(np.log(safe_lo), np.log(safe_hi)))
        else:
            guess = np.random.uniform(safe_lo, safe_hi)
        try:
            popt, _ = curve_fit(func, t, l, p0=guess,
                                bounds=bounds, maxfev=10_000)
            r2 = _r_squared(l, func(t, *popt))
            if r2 > best_r2:
                best_r2, best_p = r2, popt
        except Exception:
            continue

    # Fallback: try the heuristic p0 if every random start failed
    if best_p is None:
        try:
            popt, _ = curve_fit(func, t, l, p0=p0,
                                bounds=bounds, maxfev=50_000)
            best_p  = popt
            best_r2 = _r_squared(l, func(t, *popt))
        except Exception:
            pass

    return best_p, best_r2


# =============================================================================
# 7b.  IDENTIFIABILITY & BOUND-HIT HELPERS
# =============================================================================

def _is_bound_hitting(popt, lower, upper,
                      frac: float = BOUND_PROXIMITY_FRAC) -> List[int]:
    """
    Return the indices of parameters that sit within `frac` of a bound.

    Log-scale comparison when both bounds are positive (matches how
    `_multi_start_fit` samples the space). This prevents a 2% linear
    tolerance from being triggered by, say, a fitted eta2 of 20 000 Pa·s
    just because the upper bound is 1 000 000.

    Parameters
    ----------
    popt   : fitted parameter vector
    lower  : per-parameter lower bounds (same order as popt)
    upper  : per-parameter upper bounds (same order as popt)
    frac   : proximity threshold as a fraction of the bound range

    Returns
    -------
    list of int : positional indices of pinned parameters (empty if none)
    """
    popt  = np.asarray(popt,  dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    pinned: List[int] = []
    for i, (p, lo, hi) in enumerate(zip(popt, lower, upper)):
        if not np.isfinite(p):
            pinned.append(i)
            continue
        if lo > 0 and hi > 0:
            log_range = math.log(hi) - math.log(lo)
            near_lo = (math.log(p) - math.log(lo)) < frac * log_range
            near_hi = (math.log(hi) - math.log(p)) < frac * log_range
        else:
            lin_range = hi - lo
            near_lo = (p - lo) < frac * lin_range
            near_hi = (hi - p) < frac * lin_range
        if near_lo or near_hi:
            pinned.append(i)
    return pinned


def _check_identifiability(t: np.ndarray, l: np.ndarray,
                           late_frac: float = IDENTIFIABILITY_LATE_FRAC,
                           noise_um: float = IDENTIFIABILITY_NOISE_UM,
                           snr: float = IDENTIFIABILITY_FLOW_SNR,
                           plateau_ratio: float = IDENTIFIABILITY_PLATEAU_RATIO
                           ) -> Dict[str, bool]:
    """
    Decide which viscoelastic elements the data can actually constrain.

    Both checks are geometry-agnostic. They compare observed length changes
    to the measurement noise floor over the observed duration, and make no
    assumptions about E, eta, r_eff, or delta_p. If the flow term is not
    identifiable, the caller should skip Jeffreys and Burgers (both contain
    a series dashpot).

    Returns
    -------
    dict with:
        'flow_ok'    -- late-stage slope * duration > snr * noise
                        (a viscous flow term is resolvable within the window)
        'plateau_ok' -- late slope << early slope
                        (the trace has visibly elbowed; a KV plateau lies
                        inside the observation window)
    """
    n = len(t)
    if n < 5:
        return {'flow_ok': False, 'plateau_ok': False}

    duration = float(t[-1] - t[0])
    if duration <= 0:
        return {'flow_ok': False, 'plateau_ok': False}

    n_side = max(5, int(round(late_frac * n)))

    # Late-stage linear fit
    t_late, l_late = t[-n_side:], l[-n_side:]
    try:
        late_slope = float(np.polyfit(t_late, l_late, 1)[0])
    except Exception:
        late_slope = 0.0

    # Early-stage linear fit
    t_early, l_early = t[:n_side], l[:n_side]
    try:
        early_slope = float(np.polyfit(t_early, l_early, 1)[0])
    except Exception:
        early_slope = 0.0

    # Flow identifiable if predicted flow over the window clears the noise floor
    flow_ok = abs(late_slope) * duration > snr * noise_um

    # Plateau identifiable if the trace has clearly decelerated
    if abs(early_slope) < 1e-9:
        plateau_ok = False
    else:
        plateau_ok = (abs(late_slope) / abs(early_slope)) < plateau_ratio

    return {'flow_ok': flow_ok, 'plateau_ok': plateau_ok}


# =============================================================================
# 8.  VISCOELASTIC FITTING  (ASP cells)
# =============================================================================

def fit_viscoelastic(time: np.ndarray, length: np.ndarray,
                     r_eff: float, delta_p: float,
                     C: float = HALFSPACE_C,
                     n_starts: int = 7,
                     min_points: int = 15) -> Dict:
    """
    Fits KV, Jeffreys, and Burgers to a protrusion trace; picks winner by BIC.

    Parameters
    ----------
    time      : [s], zeroed to pressure application onset
    length    : [um]
    r_eff     : effective channel radius [um]
    delta_p   : aspiration pressure [Pa]
    C         : half-space constant
    n_starts  : number of multi-start optimiser attempts per model
    min_points: minimum cleaned data points; returns failure dict if not met

    Returns
    -------
    dict with keys:
        'best_model'   -- winning model name (str) or None
        'best_params'  -- {param_name: value} or None
        'best_r2'      -- float or None
        'best_bic'     -- float or None
        'all_models'   -- per-model results dict
        'n_fit_points' -- int
    """
    FAIL = {
        'best_model': None, 'best_params': None,
        'best_r2': None, 'best_bic': None, 'best_dw': None,
        'all_models': {}, 'n_fit_points': 0
    }

    t, l = _clean_trace(time, length)
    if len(t) < min_points:
        logger.debug(f"  Viscoelastic fit skipped: {len(t)} clean points (need {min_points}).")
        return FAIL

    t = t - t[0]   # zero to first cleaned frame

    E0, eta1_0, eta2_0 = _estimate_params(t, l, r_eff, delta_p, C)

    # ---- Pre-fit identifiability guard ----
    # Skip Jeffreys/Burgers when the data cannot resolve a viscous flow term
    # within the observation window. Both models contain a series dashpot;
    # fitting them anyway causes eta_flow to run to its upper bound.
    ident = _check_identifiability(t, l)
    flow_ok = ident['flow_ok']

    E_lo,    E_hi    =   1.0,   100_000.0
    eta1_lo, eta1_hi =   1.0,   500_000.0
    eta2_lo, eta2_hi =   500.0, 1_000_000.0

    models = {
        "Kelvin-Voigt": {
            "func"  : lambda t, E, eta: _kelvin_voigt(t, r_eff, delta_p, C, E, eta),
            "p0"    : [E0, eta1_0],
            "bounds": ([E_lo, eta1_lo], [E_hi, eta1_hi]),
            "names" : ["E", "eta"],
            "k"     : 2
        },
    }
    if flow_ok:
        models["Jeffreys"] = {
            "func"  : lambda t, E, eta1, eta2: _jeffreys(t, r_eff, delta_p, C, E, eta1, eta2),
            "p0"    : [E0, eta1_0, eta2_0],
            "bounds": ([E_lo, eta1_lo, eta2_lo], [E_hi, eta1_hi, eta2_hi]),
            "names" : ["E", "eta1", "eta2"],
            "k"     : 3
        }
        models["Burgers"] = {
            "func"  : lambda t, E1, eta1, E2, eta2: _burgers(t, r_eff, delta_p, C, E1, eta1, E2, eta2),
            "p0"    : [E0, eta2_0, E0, eta1_0],
            "bounds": ([E_lo, eta2_lo, E_lo, eta1_lo],
                       [E_hi, eta2_hi, E_hi, eta1_hi]),
            "names" : ["E1", "eta1", "E2", "eta2"],
            "k"     : 4
        }
    else:
        logger.debug(
            "  Flow term not identifiable "
            f"(late-slope * duration < {IDENTIFIABILITY_FLOW_SNR}*noise); "
            "fitting Kelvin-Voigt only."
        )

    all_results: Dict = {}
    best_bic  = np.inf
    best_name = None

    for name, cfg in models.items():
        popt, _ = _multi_start_fit(cfg["func"], t, l, cfg["bounds"],
                                   cfg["p0"], n_starts=n_starts)
        if popt is None:
            all_results[name] = {"params": None, "r2": None, "bic": np.inf,
                                 "dw": None, "bound_hit": None}
            continue

        l_pred = cfg["func"](t, *popt)
        r2     = _r_squared(l, l_pred)
        bic    = _bic(l, l_pred, cfg["k"])
        dw     = _durbin_watson(l, l_pred)

        # ---- Post-hoc bound-hit check ----
        pinned_idx = _is_bound_hitting(popt, cfg["bounds"][0], cfg["bounds"][1])
        pinned_names = [cfg["names"][i] for i in pinned_idx]

        all_results[name] = {
            "params"   : dict(zip(cfg["names"], popt.tolist())),
            "r2"       : r2,
            "bic"      : bic,
            "dw"       : dw,
            "bound_hit": pinned_names,   # empty list = clean fit
        }

        # A bound-hit fit is not identifiable; exclude from selection.
        # It stays in all_results as an audit trail.
        if pinned_names:
            logger.debug(
                f"  {name}: parameter(s) {pinned_names} pinned to bound; "
                "excluded from BIC selection."
            )
            continue

        if bic < best_bic:
            best_bic  = bic
            best_name = name

    if best_name is None:
        return FAIL

    return {
        "best_model"   : best_name,
        "best_params"  : all_results[best_name]["params"],
        "best_r2"      : all_results[best_name]["r2"],
        "best_bic"     : best_bic,
        "best_dw"      : all_results[best_name]["dw"],
        "all_models"   : all_results,
        "n_fit_points" : len(t),
    }


# =============================================================================
# 9.  MODEL-INDEPENDENT FITTING  (both ASP and EP)
# =============================================================================

def fit_model_independent(time: np.ndarray, length: np.ndarray,
                           min_points: int = 5) -> Dict:
    """
    Fits a linear and a power-law model to a protrusion trace and selects the
    winner by BIC.

    These do not assume any rheological circuit -- they characterise curve
    shape. The power-law exponent b distinguishes regimes:
        b ~ 1    : purely viscous (linear)
        b ~ 0.5  : viscoelastic (Kelvin-Voigt prediction)
        0 < b < 0.5 : solid-like / strongly elastic

    Parameters
    ----------
    time, length : arrays, already zeroed to the window start
    min_points   : minimum cleaned points required

    Returns
    -------
    dict with sub-dicts 'linear' and 'power_law', each:
        {'params': {...}, 'r2': float, 'bic': float}  or  None if fit failed.
    Also includes 'best_model', 'window_duration_s', and 'n_fit_points'.
    """
    FAIL = {'linear': None, 'power_law': None, 'best_model': None,
            'window_duration_s': None, 'n_fit_points': 0}
    t, l = _clean_trace(time, length)
    if len(t) < min_points:
        return FAIL

    window_dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    result: Dict = {'window_duration_s': window_dur, 'n_fit_points': len(t)}

    # ---- Linear (2 free parameters: slope m, intercept b) ----
    try:
        slope_guess = (l[-1] - l[0]) / max(t[-1] - t[0], 1e-9)
        popt, _ = curve_fit(_linear, t, l,
                            p0=[slope_guess, float(l[0])],
                            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
                            maxfev=5000)
        l_pred = _linear(t, *popt)
        result['linear'] = {
            'params': {'slope': float(popt[0]), 'intercept': float(popt[1])},
            'r2'    : _r_squared(l, l_pred),
            'bic'   : _bic(l, l_pred, n_params=2),
        }
    except Exception:
        result['linear'] = None

    # ---- Power law (2 free parameters: amplitude a, exponent b) ----
    # Only valid for t >= 0; _power_law adds tiny epsilon for t=0.
    try:
        a0 = float(l[0]) if l[0] > 0 else 1.0
        popt, _ = curve_fit(_power_law, t, l,
                            p0=[a0, 0.5],
                            bounds=([0, 0], [np.inf, 2]),
                            maxfev=5000)
        l_pred = _power_law(t, *popt)
        result['power_law'] = {
            'params': {'a': float(popt[0]), 'exponent_b': float(popt[1])},
            'r2'    : _r_squared(l, l_pred),
            'bic'   : _bic(l, l_pred, n_params=2),
        }
    except Exception:
        result['power_law'] = None

    # ---- BIC-based model selection ----
    lin_bic = result['linear']['bic'] if result['linear'] else np.inf
    pl_bic  = result['power_law']['bic'] if result['power_law'] else np.inf

    if lin_bic <= pl_bic and result['linear']:
        result['best_model'] = 'Linear'
    elif result['power_law']:
        result['best_model'] = 'Power-Law'
    else:
        result['best_model'] = None

    return result


# =============================================================================
# 10.  UPTAKE FITTING  (both ASP and EP)
# =============================================================================

def fit_exponential_uptake(time: np.ndarray, uptake: np.ndarray,
                            pulse_time_s: float = 0.0,
                            min_points: int = 5) -> Dict:
    """
    Fits A*(1-exp(-t/tau)) to post-pulse uptake data after subtracting
    the pre-pulse baseline.

    The baseline subtraction is important: if there is a non-zero dF/F₀
    at the pulse moment (from background drift or passive permeability),
    forcing the model through zero distorts the fitted tau. The median
    of the pre-pulse window is a robust baseline estimator.

    For ASP cells pulse_frame=0 is correct (no electroporation; dye uptake
    is interpreted relative to the first frame, and baseline is 0 by
    construction of dF/F₀).

    Parameters
    ----------
    time        : raw time array, NOT yet zeroed to the pulse
    uptake      : normalised intensity (dF/F0)
    pulse_frame : index of the electroporation pulse frame

    Returns
    -------
    dict with keys 'A', 'tau', 'r2', 'baseline' (float) or all None
    if fitting failed. 'baseline' is the value subtracted before fitting.
    """
    FAIL = {'A': None, 'tau': None, 'r2': None, 'baseline': None}

    if len(time) < min_points:
        return FAIL

    t_zero = time - pulse_time_s

    # --- Compute and subtract pre-pulse baseline ---
    pre_mask = (t_zero <= 0) & np.isfinite(uptake)
    if np.sum(pre_mask) >= 2:
        baseline = float(np.nanmedian(uptake[pre_mask]))
    else:
        # Fallback for late camera triggers: use the first 3 valid recorded frames
        valid_mask = np.isfinite(uptake)
        if np.sum(valid_mask) >= 3:
            baseline = float(np.nanmedian(uptake[valid_mask][:3]))
        elif np.sum(valid_mask) > 0:
            baseline = float(uptake[valid_mask][0])
        else:
            baseline = 0.0

    uptake_corrected = uptake - baseline

    mask   = (t_zero > 0) & np.isfinite(uptake_corrected) & (uptake_corrected >= 0)
    t_fit, y_fit = t_zero[mask], uptake_corrected[mask]

    if len(t_fit) < min_points:
        return FAIL

    A0   = float(np.nanmax(y_fit))
    tau0 = float((t_fit[-1] - t_fit[0]) / 2.0)
    try:
        popt, _ = curve_fit(_exp_uptake, t_fit, y_fit,
                            p0=[A0, tau0],
                            bounds=([0, 0], [np.inf, np.inf]),
                            maxfev=5000)
        A, tau = float(popt[0]), float(popt[1])
        r2     = _r_squared(y_fit, _exp_uptake(t_fit, *popt))
        return {'A': A, 'tau': tau, 'r2': r2, 'baseline': baseline}
    except Exception:
        return FAIL


# =============================================================================
# 11.  COMMON TIME WINDOW  (NEW — for comparable ASP fits)
# =============================================================================

def compute_common_duration(traps: List[bfh.TrapData],
                            min_duration_fraction: float = 0.50) -> Optional[float]:
    """
    Finds the common time window for a group of ASP traces so that
    viscoelastic and model-independent fits are performed over the same
    duration, making parameters directly comparable across experiments.

    Some experiments record 300 s of aspiration, others 1500 s. Without a
    common window, a Jeffreys fit on a 1500 s trace will weigh the viscous
    flow term much more heavily than the same fit on a 300 s trace. By
    truncating all traces to the shortest surviving duration, we ensure
    the optimizer "sees" the same time horizon everywhere.

    Short-duration outlier filter
    -----------------------------
    Before taking the minimum, traces whose total duration is below
    `min_duration_fraction` of the group mean are excluded from the window
    computation. This is a purely statistical filter -- it does not
    identify ruptured cells, and it is not a substitute for the load-time
    `_ruptured` / `_irregular` filename filter in bulk_file_handling.py,
    which is the only place real rupture / annotation flags are honored.

    A trap can end up in the "short-duration outlier" bucket for many
    reasons that have nothing to do with membrane rupture: the cell
    slipped out of the trap, the recording ended before the cell did, the
    cell entered late so post-entry duration is short, etc. Those traps
    are still fitted on whatever data they have; they just don't get to
    set the common window.

    Parameters
    ----------
    traps                 : list of TrapData for one condition group.
    min_duration_fraction : fraction of the group mean below which a
                            trace is treated as a short-duration outlier
                            and excluded from window computation. Default
                            0.50 -- traces shorter than half the mean
                            (typically slipped-out cells or truncated
                            recordings) do not get to set the window, but
                            traces within a wide band around the mean do.
                            Raise this (e.g. to 0.80) for a tighter, more
                            homogeneous window; lower it (e.g. to 0.20)
                            to include almost every trace.

    Returns
    -------
    float : common window duration in seconds, or None if fewer than 2
            surviving traces exist.
    """
    durations = []
    for trap in traps:
        t = trap.protrusion_data.get('Time_s', np.array([]))
        if len(t) > 1:
            durations.append(float(t[-1] - t[0]))

    if len(durations) < 2:
        return None

    mean_dur   = float(np.mean(durations))
    threshold  = min_duration_fraction * mean_dur

    # Keep only traces whose total duration exceeds the outlier threshold.
    surviving_durations = [d for d in durations if d >= threshold]

    if len(surviving_durations) < 2:
        return None

    common = float(min(surviving_durations))
    n_excluded = len(durations) - len(surviving_durations)
    logger.info(
        f"  Common duration: {common:.1f} s "
        f"(mean={mean_dur:.1f}, {n_excluded} short-duration outlier(s) "
        f"excluded at <{min_duration_fraction:.0%} of mean)"
    )
    return common


def compute_common_ep_pre_duration(traps: List[bfh.TrapData],
                                   min_duration_fraction: float = 0.20) -> Optional[float]:
    """
    Finds the shortest common pre-pulse aspiration window (in seconds) across
    all EP traps in a group, so that model-independent fits use the same time
    horizon for every replicate.

    Different experiment folders in the same condition group may have slightly
    different pulse_frame values, which means the pre-pulse window in seconds
    can vary.  Truncating all traces to the same duration makes slope and
    power-law exponent directly comparable.

    Traces whose computed pre-pulse duration is less than
    `min_duration_fraction` of the group mean are excluded from setting the
    window (these are likely malformed recordings with an unusually early
    pulse trigger). This is a purely statistical filter and has no
    connection to rupture annotation.

    Returns None if fewer than 2 valid EP traps are found.
    """
    pre_durations = []
    for trap in traps:
        if trap.metadata.condition_type != "EP":
            continue
        t  = trap.protrusion_data.get('Time_s', np.array([]))
        pf = trap.metadata.pulse_frame
        if len(t) < 5 or not (0 < pf < len(t)):
            continue
        pre_dur = float(t[pf] - t[0])
        if pre_dur > 0:
            pre_durations.append(pre_dur)

    if len(pre_durations) < 2:
        return None

    mean_dur            = float(np.mean(pre_durations))
    threshold           = min_duration_fraction * mean_dur
    surviving_durations = [d for d in pre_durations if d >= threshold]

    if len(surviving_durations) < 2:
        return None

    common = float(min(surviving_durations))
    logger.info(
        f"  EP common pre-pulse duration: {common:.1f} s "
        f"(mean={mean_dur:.1f}, {len(pre_durations)-len(surviving_durations)} "
        f"short-duration outlier(s) excluded)"
    )
    return common


def compute_common_ep_post_duration(traps: List[bfh.TrapData],
                                    max_window_s: float = 5.0) -> Optional[float]:
    """
    Finds the shortest common post-pulse recording window (in seconds) across
    all EP traps in a group, capped at `max_window_s`.

    Used to ensure the post-pulse slope (and therefore the Extends / Stable /
    Retracts classification) is computed over the same time window for every
    replicate.  Traces that end before the pulse frame are skipped.

    Parameters
    ----------
    max_window_s : hard cap on the window (default 5 s).  Post-pulse dynamics
                   are typically assessed over the first 2–5 seconds; a longer
                   window would mix the immediate mechanical response with
                   slower membrane recovery.

    Returns None if fewer than 2 valid EP traps are found.
    """
    post_durations = []
    for trap in traps:
        if trap.metadata.condition_type != "EP":
            continue
        t  = trap.protrusion_data.get('Time_s', np.array([]))
        pf = trap.metadata.pulse_frame
        if len(t) < 5 or not (0 < pf < len(t)):
            continue
        post_dur = float(t[-1] - t[pf])
        if post_dur > 0:
            post_durations.append(post_dur)

    if len(post_durations) < 2:
        return None

    common = float(min(min(post_durations), max_window_s))
    logger.info(f"  EP common post-pulse window: {common:.1f} s")
    return common


# =============================================================================
# 12.  CONDITION LABEL HELPER
# =============================================================================

def _make_condition_label(meta: bfh.ExperimentMetadata) -> str:
    """Builds a composite condition string for the output CSV."""
    if meta.condition_type == "ASP":
        return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_ASP"
    return (f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_"
            f"{meta.voltage}V_{meta.duration_label}")


# =============================================================================
# 13.  TRAP-LEVEL RUNNER
# =============================================================================

def _run_trap_mechanics(trap: bfh.TrapData, r_eff: float, C: float,
                        r2_floor: float = 0.0,
                        common_duration_s: Optional[float] = None,
                        common_ep_pre_s: Optional[float] = None,
                        common_ep_post_s: Optional[float] = None) -> Dict:
    """
    Runs all applicable fits for one trap and returns a flat row dict.

    Parameters
    ----------
    trap              : TrapData object
    r_eff             : effective channel radius [um]
    C                 : half-space constant
    r2_floor          : minimum R² for a viscoelastic fit to be reported.
                        Fits below this threshold are flagged
                        (Visco_R2_Flag = False) but still stored so no data
                        is silently discarded.
    common_duration_s : if provided, ASP traces are truncated to this
                        duration (in seconds from cell entry) before
                        fitting. This makes fits comparable across
                        experiments with different recording lengths.
    common_ep_pre_s   : if provided, EP pre-pulse window for MI fitting is
                        truncated to this duration (seconds before the pulse).
    common_ep_post_s  : if provided, the post-pulse slope window for EP
                        behavior classification is capped at this value.

    Output columns
    --------------
    Identity    : Date, Cell_Type, Treatment, Experiment_Number, Pressure_Pa,
                  Voltage_V, Duration_ms, Duration_label, Condition_Type,
                  Condition, Trap_ID, Experiment_Folder, Son_Factor_fstar
    Viscoelastic: Best_Model, E_Pa, eta1_Pa_s, eta2_Pa_s, Tau_s,
                  Visco_R2, Visco_BIC, Visco_DW, Visco_R2_Flag,
                  Common_Duration_s
    Model-indep : MI_Best_Model, Linear_Slope, Linear_Intercept, Linear_R2,
                  Linear_BIC, PL_a, PL_b, PL_R2, PL_BIC,
                  MI_Window_Duration_s, MI_N_Points
    EP slopes   : Pre_Pulse_Slope, Post_Pulse_Slope, EP_Post_Pulse_Behavior
    Uptake      : Uptake_PrePulse_QC,
                  Uptake_Body_VolNorm_tau/R2/A,
                  Uptake_Prot_VolNorm_tau/R2/A,
                  Uptake_Total_VolNorm_tau/R2/A
    Morphology  : Cell_Body_Area_um2, Cell_Prot_Area_um2,
                  Cell_Body_Area_PrePulse_um2   (source: Uptake CSV),
                  Max_Prot_length_PrePulse_um   (source: full detection CSV)
    """
    meta    = trap.metadata
    pd_data = trap.protrusion_data
    ud_data = trap.uptake_data

    row = {
        'Date'              : meta.date,
        'Cell_Type'         : meta.cell_type,
        'Treatment'         : meta.treatment,
        'Experiment_Number' : meta.experiment_number,
        'Pressure_Pa'       : meta.pressure,
        'Voltage_V'         : meta.voltage,
        # FIX #10: Store numeric Duration_ms for sorting/filtering.
        'Duration_ms'       : meta.duration,
        'Duration_label'    : meta.duration_label,
        'Condition_Type'    : meta.condition_type,
        # FIX #11: Pre-built composite label for quick filtering in Excel/Prism.
        'Condition'         : _make_condition_label(meta),
        'Trap_ID'           : trap.trap_id,
        # Original folder name for traceability, e.g.
        # "260129_MDAMB231_WT_Chip1_Experiment7-1100Pa-100V-100us-frame35"
        'Experiment_Folder' : meta.full_path.name,
        # reference/traceability. It is NOT used to override compute_reff()
        # because the theoretical f* and the shear-CSV f* should agree for
        # the standard device geometry. If you use a non-standard device,
        # consider passing fstar to compute_reff() manually.
        'Son_Factor_fstar'  : meta.fstar,
        # Viscoelastic (ASP only)
        'Best_Model'        : None,
        'E_Pa'              : None,
        'E1_Pa'             : None,
        'eta1_Pa_s'         : None,
        'eta2_Pa_s'         : None,
        'Tau_s'             : None,
        'Visco_R2'          : None,
        'Visco_BIC'         : None,
        'Visco_DW'          : None,
        'Visco_R2_Flag'     : None,
        'Common_Duration_s' : common_duration_s,
        # Model-independent (both)
        'MI_Best_Model'         : None,
        'Linear_Slope'          : None,
        'Linear_Intercept'      : None,
        'Linear_R2'             : None,
        'Linear_BIC'            : None,
        'PL_a'                  : None,
        'PL_b'                  : None,
        'PL_R2'                 : None,
        'PL_BIC'                : None,
        'MI_Window_Duration_s'  : None,
        'MI_N_Points'           : None,
        # EP pre/post pulse slopes
        'Pre_Pulse_Slope'       : None,
        'Post_Pulse_Slope'      : None,
        # EP post-pulse protrusion behavior classification
        # 'Extends' | 'Stable' | 'Retracts'
        'EP_Post_Pulse_Behavior': None,
        # Pre-pulse dye contamination QC (EP only).
        # 'Clean'        : no significant dye before the pulse.
        # 'Pre_Leaky'    : MinMax rises into the threshold during the pre-pulse
        #                  window (membrane was already permeable; Trap 2 pattern).
        # 'Pre_Loaded'   : MinMax already above threshold at the first recorded
        #                  frame (cell contained dye before recording started;
        #                  Trap 7 pattern).  Also fires when recording began late
        #                  and the first frame is already contaminated (Trap 4
        #                  pattern -- the subtype is then ambiguous).
        # 'No_Data'      : uptake CSV missing or too short to evaluate.
        # 'N/A'          : ASP condition (no pulse; concept does not apply).
        'Uptake_PrePulse_QC'    : None,
        # Uptake exponential fit — volume-normalised intensity (ADU/μm³).
        # Dividing by cell volume removes the size confound (larger cells
        # accumulate more raw ADU regardless of membrane permeability).
        # Source columns: Body_VolNorm / Protrusion_VolNorm / Total_VolNorm
        # from UptakeQuantification.export_csv().
        'Uptake_Body_VolNorm_tau'  : None, 'Uptake_Body_VolNorm_R2'  : None, 'Uptake_Body_VolNorm_A'  : None,
        'Uptake_Prot_VolNorm_tau'  : None, 'Uptake_Prot_VolNorm_R2'  : None, 'Uptake_Prot_VolNorm_A'  : None,
        'Uptake_Total_VolNorm_tau' : None, 'Uptake_Total_VolNorm_R2' : None, 'Uptake_Total_VolNorm_A' : None,
        # Cell size proxies: median area over the 10 frames immediately before the
        # pulse (last 10 frames for ASP).  Captures the stable aspirated state
        # rather than the cell-entry transient.
        'Cell_Body_Area_um2': None,
        'Cell_Prot_Area_um2': None,
        # Pre-pulse body area: median of up to 10 frames immediately before the
        # pulse in the Uptake CSV.  Uses all available pre-pulse frames when
        # fewer than 10 exist.  Source: Uptake_Data.csv → Body_Area_um2.
        'Cell_Body_Area_PrePulse_um2': None,
        # Maximum protrusion length in the pre-pulse window.
        # Source: detection_full.csv → Protrusion_Length_um (NOT the filtered CSV,
        # which starts only from cell entry and misses early frames).
        'Max_Prot_length_PrePulse_um': None,
    }

    time_raw   = pd_data.get('Time_s',               np.array([]))
    length_raw = pd_data.get('Protrusion_Length_um', np.array([]))

    if len(time_raw) < 5 or len(length_raw) < 5:
        return row

    # --- Common time window truncation (ASP only) ---
    if meta.condition_type == "ASP" and common_duration_s is not None:
        t_from_entry = time_raw - time_raw[0]
        window_mask  = t_from_entry <= common_duration_s
        time_raw     = time_raw[window_mask]
        length_raw   = length_raw[window_mask]

        if len(time_raw) < 5:
            return row

    # --- Shared guard: compute physical pulse time in seconds ---
    pf = meta.pulse_frame
    pf_valid = (0 < pf < len(time_raw))
    pulse_time_s = float(time_raw[pf]) if pf_valid else 0.0

    if meta.condition_type == "EP" and not pf_valid:
        logger.warning(
            f"  Trap {trap.trap_id} ({meta.experiment_number}): "
            f"pulse_frame={pf} is out of range for array length {len(time_raw)}. "
            f"Skipping pulse-aligned analyses (pre/post slopes, uptake offset)."
        )

    # --- Cell size proxies ---
    if pf_valid:
        area_end   = pf
        area_start = max(0, area_end - 10)
    else:
        area_end   = None
        area_start = -10

    body_area_arr = pd_data.get('Body_Area_um2', np.array([]))
    if len(body_area_arr) > 0:
        window = body_area_arr[area_start:area_end]
        if len(window) > 0:
            row['Cell_Body_Area_um2'] = float(np.nanmedian(window))

    prot_area_arr = pd_data.get('Protrusion_Area_um2', np.array([]))
    if len(prot_area_arr) > 0:
        window = prot_area_arr[area_start:area_end]
        if len(window) > 0:
            row['Cell_Prot_Area_um2'] = float(np.nanmedian(window))

    # --- Pre-pulse body area ---
    ud_body_area = np.asarray(ud_data.get('Body_Area_um2', []), dtype=float)
    ud_time      = np.asarray(ud_data.get('Time_s',         []), dtype=float)

    if len(ud_body_area) > 0 and len(ud_body_area) == len(ud_time):
        if pf_valid:
            pre_mask   = ud_time < pulse_time_s
            pre_area   = ud_body_area[pre_mask]
            if len(pre_area) > 10:
                pre_area = pre_area[-10:]
        else:
            pre_area = ud_body_area[-10:]

        if len(pre_area) > 0:
            row['Cell_Body_Area_PrePulse_um2'] = float(np.nanmedian(pre_area))

    # --- Maximum protrusion length before the pulse ---
    if pf_valid:
        pre_pulse_lengths = length_raw[:pf]
    else:
        pre_pulse_lengths = length_raw

    valid_lengths = pre_pulse_lengths[np.isfinite(pre_pulse_lengths) & (pre_pulse_lengths > 0)]
    if len(valid_lengths) > 0:
        row['Max_Prot_length_PrePulse_um'] = float(np.max(valid_lengths))

    # ------------------------------------------------------------------
    # BLOCK A  --  Viscoelastic fitting (ASP only)
    # ------------------------------------------------------------------
    if meta.condition_type == "ASP":
        t_asp = time_raw - time_raw[0]
        visco = fit_viscoelastic(t_asp, length_raw, r_eff, meta.pressure, C)

        if visco['best_model']:
            row['Best_Model'] = visco['best_model']
            row['Visco_R2']   = visco['best_r2']
            row['Visco_BIC']  = visco['best_bic']
            row['Visco_DW']   = visco['best_dw']
            row['Visco_R2_Flag'] = (
                visco['best_r2'] >= r2_floor
                if visco['best_r2'] is not None else False
            )
            best_p = visco['best_params']

            # Normalise to common column names regardless of winning model.
            if visco['best_model'] == 'Kelvin-Voigt':
                E   = best_p['E']
                eta = best_p['eta']
                row.update({
                    'E_Pa'     : E,
                    'eta1_Pa_s': eta,
                    'Tau_s'    : (3 * math.pi * eta) / (C * E) if E else None,
                })
            elif visco['best_model'] == 'Jeffreys':
                E    = best_p['E']
                eta1 = best_p['eta1']
                eta2 = best_p['eta2']
                row.update({
                    'E_Pa'     : E,
                    'eta1_Pa_s': eta1,
                    'eta2_Pa_s': eta2,
                    'Tau_s'    : (3 * math.pi * eta1) / (C * E) if E else None,
                })
            elif visco['best_model'] == 'Burgers':
                E1_mw    = best_p.get('E1')
                E2_val   = best_p.get('E2')
                eta2_kv  = best_p.get('eta2')
                eta1_mw  = best_p.get('eta1')

                tau_val = None
                if E2_val and eta2_kv:
                    tau_val = (3 * math.pi * eta2_kv) / (C * E2_val)

                row.update({
                    'E_Pa'     : E2_val,
                    'E1_Pa'    : E1_mw,
                    'eta1_Pa_s': eta2_kv,
                    'eta2_Pa_s': eta1_mw,
                    'Tau_s'    : tau_val,
                })

    # ------------------------------------------------------------------
    # BLOCK B  --  Model-independent fitting (BIC-based selection)
    # ------------------------------------------------------------------
    if meta.condition_type == "ASP":
        t_mi   = time_raw - time_raw[0]
        mi_fit = fit_model_independent(t_mi, length_raw)
    else:
        mi_fit = {'linear': None, 'power_law': None, 'best_model': None,
                  'window_duration_s': None, 'n_fit_points': 0}
        if pf_valid:
            t_aligned = time_raw - time_raw[pf]
            pre_mask  = (t_aligned < 0) & (length_raw > 0)

            # Apply common pre-pulse window so MI fits across replicates use
            # the same time horizon (mirrors common_duration_s for ASP).
            if common_ep_pre_s is not None:
                pre_mask = pre_mask & (t_aligned >= -common_ep_pre_s)

            if np.sum(pre_mask) >= 5:
                t_pre  = t_aligned[pre_mask]
                t_pre  = t_pre - t_pre[0]
                l_pre  = length_raw[pre_mask]
                mi_fit = fit_model_independent(t_pre, l_pre)

    row['MI_Best_Model'] = mi_fit.get('best_model')
    if mi_fit['linear']:
        row['Linear_Slope']     = mi_fit['linear']['params']['slope']
        row['Linear_Intercept'] = mi_fit['linear']['params']['intercept']
        row['Linear_R2']        = mi_fit['linear']['r2']
        row['Linear_BIC']       = mi_fit['linear']['bic']
    if mi_fit['power_law']:
        row['PL_a']  = mi_fit['power_law']['params']['a']
        row['PL_b']  = mi_fit['power_law']['params']['exponent_b']
        row['PL_R2'] = mi_fit['power_law']['r2']
        row['PL_BIC'] = mi_fit['power_law']['bic']
    row['MI_Window_Duration_s'] = mi_fit.get('window_duration_s')
    row['MI_N_Points']          = mi_fit.get('n_fit_points')

    # ------------------------------------------------------------------
    # BLOCK C  --  Pre/post-pulse slopes (EP only)
    # ------------------------------------------------------------------
    if meta.condition_type == "EP" and pf_valid:
        t_aligned = time_raw - time_raw[pf]

        # Pre-pulse window: up to 5 s before pulse, or common_ep_pre_s if set.
        pre_window = common_ep_pre_s if common_ep_pre_s is not None else 5.0
        pre_mask  = (t_aligned >= -pre_window) & (t_aligned < 0) & (length_raw > 0)

        # Post-pulse window: capped at common_ep_post_s (default 2 s).
        post_window = common_ep_post_s if common_ep_post_s is not None else 2.0
        post_mask = (t_aligned >= 0) & (t_aligned <= post_window) & (length_raw > 0)

        if np.sum(pre_mask) > 2:
            row['Pre_Pulse_Slope'] = float(
                np.polyfit(t_aligned[pre_mask], length_raw[pre_mask], 1)[0])
        if np.sum(post_mask) > 2:
            post_slope = float(
                np.polyfit(t_aligned[post_mask], length_raw[post_mask], 1)[0])
            row['Post_Pulse_Slope'] = post_slope

            # Classify immediate post-pulse protrusion behavior.
            # Three outcomes:
            #   Extends  — positive slope above threshold (protrusion grows)
            #   Retracts — negative slope below threshold (protrusion shrinks)
            #   Stable   — slope within ±threshold (no clear movement)
            thr = EP_BEHAVIOR_SLOPE_THRESHOLD
            if post_slope > thr:
                row['EP_Post_Pulse_Behavior'] = 'Extends'
            elif post_slope < -thr:
                row['EP_Post_Pulse_Behavior'] = 'Retracts'
            else:
                row['EP_Post_Pulse_Behavior'] = 'Stable'

    # ------------------------------------------------------------------
    # BLOCK D  --  Pre-pulse dye contamination QC (EP only)
    # ------------------------------------------------------------------
    # DUAL-METRIC PHYSIOLOGICAL QC:
    # 1. Pre_Leaky: Detects cells that tear during aspiration by measuring the
    #    peak-to-peak amplitude (max - min) of the raw dF/F0 signal before the pulse.
    #    A >15% swing indicates significant active leakage.
    # 2. Pre_Loaded: Detects cells that enter the trap already saturated with dye.
    #    Since dF/F0 normalizes the baseline away, we mathematically reconstruct
    #    the absolute baseline intensity (F0 = dF / (dF/F0)) and apply a hard threshold.
    
    PRE_LEAKY_P2P_THRESHOLD = 0.15    # 15% peak-to-peak dF/F0 swing before pulse
    PRE_LOADED_F0_THRESHOLD = 1500.0  # Absolute F0 brightness threshold

    if meta.condition_type == "ASP":
        row['Uptake_PrePulse_QC'] = 'N/A'

    elif ud_data and 'Time_s' in ud_data:
        t_up     = np.asarray(ud_data.get('Time_s', []), dtype=float)
        prot_df  = np.asarray(ud_data.get('Protrusion_Normalized_dF_F0', []), dtype=float)
        body_df  = np.asarray(ud_data.get('Body_Normalized_dF_F0', []), dtype=float)
        prot_int = np.asarray(ud_data.get('Protrusion_Intensity', []), dtype=float)
        body_int = np.asarray(ud_data.get('Body_Intensity', []), dtype=float)

        if pf_valid and len(t_up) > 0 and len(t_up) == len(prot_df) == len(prot_int):
            pulse_time  = float(time_raw[pf])
            pre_mask_ud = t_up < pulse_time

            if pre_mask_ud.any():
                # --- 1. Peak-to-Peak Leaky Check ---
                # Using max - min accounts for traces that start below the baseline average
                prot_pre = prot_df[pre_mask_ud]
                body_pre = body_df[pre_mask_ud]
                
                prot_p2p = float(np.nanmax(prot_pre) - np.nanmin(prot_pre)) if len(prot_pre) > 0 else 0.0
                body_p2p = float(np.nanmax(body_pre) - np.nanmin(body_pre)) if len(body_pre) > 0 else 0.0

                # --- 2. Reconstruct Absolute F0 Loaded Check ---
                def _estimate_f0(df_array, int_array):
                    # Use the first 5 frames to robustly estimate F0
                    n_frames = min(5, len(df_array))
                    df_sub = df_array[:n_frames]
                    int_sub = int_array[:n_frames]
                    
                    # Avoid division by zero on mathematically flat arrays
                    valid = np.abs(df_sub) > 1e-4
                    if np.any(valid):
                        return float(np.nanmedian(int_sub[valid] / df_sub[valid]))
                    return 0.0

                f0_prot = _estimate_f0(prot_df, prot_int)
                f0_body = _estimate_f0(body_df, body_int)

                is_pre_loaded = (f0_prot > PRE_LOADED_F0_THRESHOLD or f0_body > PRE_LOADED_F0_THRESHOLD)
                is_pre_leaky  = (prot_p2p > PRE_LEAKY_P2P_THRESHOLD or body_p2p > PRE_LEAKY_P2P_THRESHOLD)

                if is_pre_loaded:
                    row['Uptake_PrePulse_QC'] = 'Pre_Loaded'
                elif is_pre_leaky:
                    row['Uptake_PrePulse_QC'] = 'Pre_Leaky'
                else:
                    row['Uptake_PrePulse_QC'] = 'Clean'
            else:
                row['Uptake_PrePulse_QC'] = 'No_Data'
        else:
            row['Uptake_PrePulse_QC'] = 'No_Data'
    else:
        row['Uptake_PrePulse_QC'] = 'No_Data'

    # ------------------------------------------------------------------
    # BLOCK E  --  Uptake exponential fitting (both conditions)
    # ------------------------------------------------------------------
    # For EP traps flagged as Pre_Leaky or Pre_Loaded, fitting
    # A*(1 - exp(-t/tau)) would mix pre-existing dye with genuine
    # electroporation-induced uptake.  The resulting tau is not
    # interpretable as a pure EP response, so we skip fitting entirely
    # for those traps.  ASP and Clean EP traps proceed normally.
    _qc = row.get('Uptake_PrePulse_QC')
    _contaminated = _qc in ('Pre_Leaky', 'Pre_Loaded')

    if ud_data and 'Time_s' in ud_data and not _contaminated:
        t_up = ud_data['Time_s']

        for col_base, region in [
            ('Uptake_Body_VolNorm',  'Body_VolNorm'),
            ('Uptake_Prot_VolNorm',  'Protrusion_VolNorm'),
            ('Uptake_Total_VolNorm', 'Total_VolNorm'),
        ]:
            if region in ud_data:
                fit = fit_exponential_uptake(t_up, ud_data[region],
                                             pulse_time_s=pulse_time_s)
                row[f'{col_base}_tau'] = fit['tau']
                row[f'{col_base}_R2']  = fit['r2']
                row[f'{col_base}_A']   = fit['A']

    return row

# =============================================================================
# 14.  BATCH RUNNER  (public API)
# =============================================================================

def run_all_mechanics(grouped_data: Dict,
                      r_eff: float = DEFAULT_R_EFF,
                      C: float = HALFSPACE_C,
                      r2_floor: float = 0.0) -> pd.DataFrame:
    """
    Runs mechanics fitting for every trap in grouped_data.

    Parameters
    ----------
    grouped_data : dict produced by BulkDataLoader.iter_groups()
    r_eff        : effective channel radius [um].
                   Default = compute_reff(6.7, 5.0) for the standard device.
    C            : half-space constant (default 1.0)
    r2_floor     : minimum R² for a viscoelastic fit to be flagged as reliable.

    Returns
    -------
    pd.DataFrame -- one row per trap, all columns as in _run_trap_mechanics.
    """
    rows  = []
    total = sum(len(v) for v in grouped_data.values())
    logger.info(
        f"Running mechanics fitting on {total} traps "
        f"(r_eff = {r_eff:.3f} um, C = {C}, R² floor = {r2_floor}) ..."
    )

    for key, traps in grouped_data.items():
        # --- Compute common time window for this group ---
        # Each grouped_data key is one condition, so a single scalar window
        # per condition is what we need (per-condition truncation).
        common_dur         = None
        common_ep_pre_dur  = None
        common_ep_post_dur = None

        if traps:
            ctype = traps[0].metadata.condition_type
            if ctype == "ASP":
                common_dur = compute_common_duration(traps)
            elif ctype == "EP":
                # For EP: align all pre-pulse MI windows and post-pulse slope
                # windows to the same duration across replicates.
                common_ep_pre_dur  = compute_common_ep_pre_duration(traps)
                common_ep_post_dur = compute_common_ep_post_duration(traps)

        for trap in traps:
            try:
                row = _run_trap_mechanics(
                    trap, r_eff, C, r2_floor=r2_floor,
                    common_duration_s=(
                        common_dur
                        if trap.metadata.condition_type == "ASP" else None
                    ),
                    common_ep_pre_s=common_ep_pre_dur,
                    common_ep_post_s=common_ep_post_dur,
                )
                rows.append(row)
            except Exception as e:
                logger.warning(
                    f"  Trap {trap.trap_id} "
                    f"({trap.metadata.experiment_number}): {e}"
                )

    df = pd.DataFrame(rows)

    # Log R² quality summary for ASP fits
    if 'Visco_R2_Flag' in df.columns:
        asp_mask = df['Condition_Type'] == 'ASP'
        n_asp = asp_mask.sum()
        n_pass = (df.loc[asp_mask, 'Visco_R2_Flag'] == True).sum()
        n_fitted = df.loc[asp_mask, 'Best_Model'].notna().sum()
        logger.info(
            f"Mechanics fitting complete -- {len(df)} rows total, "
            f"{n_fitted}/{n_asp} ASP traps fitted, "
            f"{n_pass}/{n_fitted} passed R² >= {r2_floor} threshold."
        )
    else:
        logger.info(f"Mechanics fitting complete -- {len(df)} rows.")

    return df