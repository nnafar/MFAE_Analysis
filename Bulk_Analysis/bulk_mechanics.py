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
        "Jeffreys": {
            "func"  : lambda t, E, eta1, eta2: _jeffreys(t, r_eff, delta_p, C, E, eta1, eta2),
            "p0"    : [E0, eta1_0, eta2_0],
            "bounds": ([E_lo, eta1_lo, eta2_lo], [E_hi, eta1_hi, eta2_hi]),
            "names" : ["E", "eta1", "eta2"],
            "k"     : 3
        },
        "Burgers": {
            "func"  : lambda t, E1, eta1, E2, eta2: _burgers(t, r_eff, delta_p, C, E1, eta1, E2, eta2),
            "p0"    : [E0, eta2_0, E0, eta1_0],
            "bounds": ([E_lo, eta2_lo, E_lo, eta1_lo],
                       [E_hi, eta2_hi, E_hi, eta1_hi]),
            "names" : ["E1", "eta1", "E2", "eta2"],
            "k"     : 4
        },
    }

    all_results: Dict = {}
    best_bic  = np.inf
    best_name = None

    for name, cfg in models.items():
        popt, _ = _multi_start_fit(cfg["func"], t, l, cfg["bounds"],
                                   cfg["p0"], n_starts=n_starts)
        if popt is None:
            all_results[name] = {"params": None, "r2": None, "bic": np.inf, "dw": None}
            continue

        l_pred = cfg["func"](t, *popt)
        r2     = _r_squared(l, l_pred)
        bic    = _bic(l, l_pred, cfg["k"])
        dw     = _durbin_watson(l, l_pred)
        all_results[name] = {
            "params": dict(zip(cfg["names"], popt.tolist())),
            "r2"    : r2,
            "bic"   : bic,
            "dw"    : dw,
        }
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
                            pulse_frame: int = 0,
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

    if len(time) <= pulse_frame or len(time) < min_points:
        return FAIL

    t_zero = time - time[pulse_frame]

    # --- Compute and subtract pre-pulse baseline ---
    pre_mask = (t_zero <= 0) & np.isfinite(uptake)
    if np.sum(pre_mask) >= 2:
        baseline = float(np.nanmedian(uptake[pre_mask]))
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
                            rupture_fraction: float = 0.95) -> Optional[float]:
    """
    Finds the common time window for a group of ASP traces so that
    viscoelastic and model-independent fits are performed over the same
    duration, making parameters directly comparable across experiments.

    Some experiments record 300 s of aspiration, others 1500 s. Without a
    common window, a Jeffreys fit on a 1500 s trace will weigh the viscous
    flow term much more heavily than the same fit on a 300 s trace. By
    truncating all traces to the shortest *non-ruptured* duration (e.g.
    300 s), we ensure the optimizer "sees" the same time horizon everywhere.

    Ruptured traces end abruptly at much shorter times. A trace is
    classified as ruptured if its duration is below `rupture_fraction` of
    the group median. Those short traces are excluded from setting the
    window (but they are still fitted on whatever data they have).

    Parameters
    ----------
    traps : list of TrapData for one condition group
    rupture_fraction : fraction of the median below which a trace is
                       considered ruptured and excluded from window
                       computation. Default 0.20 (20% of the median).

    Returns
    -------
    float : common window duration in seconds, or None if fewer than 2
            non-ruptured traces exist.
    """
    durations = []
    for trap in traps:
        t = trap.protrusion_data.get('Time_s', np.array([]))
        if len(t) > 1:
            durations.append(float(t[-1] - t[0]))

    if len(durations) < 2:
        return None

    median_dur = float(np.median(durations))
    threshold  = rupture_fraction * median_dur

    # Keep only traces whose total duration exceeds the rupture threshold.
    normal_durations = [d for d in durations if d >= threshold]

    if len(normal_durations) < 2:
        return None

    common = float(min(normal_durations))
    logger.info(
        f"  Common duration: {common:.1f} s "
        f"(median={median_dur:.1f}, {len(durations)-len(normal_durations)} "
        f"ruptured traces excluded)"
    )
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
                        common_duration_s: Optional[float] = None) -> Dict:
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

    Output columns
    --------------
    Identity    : Date, Cell_Type, Treatment, Experiment_Number, Pressure_Pa,
                  Voltage_V, Duration_ms, Duration_label, Condition_Type,
                  Condition, Trap_ID, Son_Factor_fstar
    Viscoelastic: Best_Model, E_Pa, eta1_Pa_s, eta2_Pa_s, Tau_s,
                  Visco_R2, Visco_BIC, Visco_DW, Visco_R2_Flag,
                  Common_Duration_s
    Model-indep : MI_Best_Model, Linear_Slope, Linear_Intercept, Linear_R2,
                  Linear_BIC, PL_a, PL_b, PL_R2, PL_BIC,
                  MI_Window_Duration_s, MI_N_Points
    EP slopes   : Pre_Pulse_Slope, Post_Pulse_Slope
    Uptake      : Uptake_Body_tau, Uptake_Body_R2,
                  Uptake_Prot_tau, Uptake_Prot_R2,
                  Uptake_Total_tau, Uptake_Total_R2
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
        # NOTE #12: fstar is loaded from the shear CSV and stored here for
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
        'Pre_Pulse_Slope'   : None,
        'Post_Pulse_Slope'  : None,
        # Uptake taus
        'Uptake_Body_tau'   : None, 'Uptake_Body_R2'  : None,
        'Uptake_Prot_tau'   : None, 'Uptake_Prot_R2'  : None,
        'Uptake_Total_tau'  : None, 'Uptake_Total_R2' : None,
    }

    time_raw   = pd_data.get('Time_s',               np.array([]))
    length_raw = pd_data.get('Protrusion_Length_um', np.array([]))

    if len(time_raw) < 5 or len(length_raw) < 5:
        return row

    # --- Common time window truncation (ASP only) ---
    # If a common_duration_s was computed for this group, truncate the
    # protrusion trace so that all ASP fits use the same time horizon.
    # This makes fitted parameters (E, eta, tau) directly comparable
    # across experiments that recorded for different durations.
    if meta.condition_type == "ASP" and common_duration_s is not None:
        t_from_entry = time_raw - time_raw[0]
        window_mask  = t_from_entry <= common_duration_s
        time_raw     = time_raw[window_mask]
        length_raw   = length_raw[window_mask]

        if len(time_raw) < 5:
            return row

    # --- Shared guard: validate pulse_frame is in range ---
    pf = meta.pulse_frame
    pf_valid = (0 < pf < len(time_raw))  # False for ASP (pf == 0) by design
    if meta.condition_type == "EP" and not pf_valid:
        logger.warning(
            f"  Trap {trap.trap_id} ({meta.experiment_number}): "
            f"pulse_frame={pf} is out of range for array length {len(time_raw)}. "
            f"Skipping pulse-aligned analyses (pre/post slopes, uptake offset)."
        )

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

        pre_mask  = (t_aligned >= -5) & (t_aligned < 0) & (length_raw > 0)
        post_mask = (t_aligned >= 0)  & (t_aligned <= 2) & (length_raw > 0)

        if np.sum(pre_mask) > 2:
            row['Pre_Pulse_Slope']  = float(
                np.polyfit(t_aligned[pre_mask], length_raw[pre_mask], 1)[0])
        if np.sum(post_mask) > 2:
            row['Post_Pulse_Slope'] = float(
                np.polyfit(t_aligned[post_mask], length_raw[post_mask], 1)[0])

    # ------------------------------------------------------------------
    # BLOCK D  --  Uptake exponential fitting (both conditions)
    # ------------------------------------------------------------------
    if ud_data and 'Time_s' in ud_data:
        t_up = ud_data['Time_s']
        # For EP with valid pulse_frame, use it; otherwise default to 0
        uptake_pf = pf if (meta.condition_type == "EP" and pf_valid) else 0

        for col_base, region in [
            ('Uptake_Body',  'Body_Normalized_dF_F0'),
            ('Uptake_Prot',  'Protrusion_Normalized_dF_F0'),
            ('Uptake_Total', 'Total_Normalized_dF_F0'),
        ]:
            if region in ud_data:
                fit = fit_exponential_uptake(t_up, ud_data[region],
                                             pulse_frame=uptake_pf)
                row[f'{col_base}_tau'] = fit['tau']
                row[f'{col_base}_R2']  = fit['r2']

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
        # --- Compute common time window for ASP groups ---
        common_dur = None
        if traps and traps[0].metadata.condition_type == "ASP":
            common_dur = compute_common_duration(traps)

        for trap in traps:
            try:
                row = _run_trap_mechanics(
                    trap, r_eff, C, r2_floor=r2_floor,
                    common_duration_s=common_dur
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