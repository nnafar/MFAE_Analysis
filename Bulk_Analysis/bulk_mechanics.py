# -*- coding: utf-8 -*-
"""
Mechanics fitting module for bulk MFAE analysis.
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

CHANNEL_WIDTH_UM  = 6.7   
CHANNEL_HEIGHT_UM = 5.0   
HALFSPACE_C       = 1.0   

EP_BEHAVIOR_SLOPE_THRESHOLD: float = 0.02
BOUND_PROXIMITY_FRAC: float = 0.02
IDENTIFIABILITY_NOISE_UM: float = 0.3    
IDENTIFIABILITY_LATE_FRAC: float = 0.33  
IDENTIFIABILITY_FLOW_SNR: float = 1.5    
IDENTIFIABILITY_PLATEAU_RATIO: float = 0.5  

# Fixed seed for the multi-start optimiser used by fit_viscoelastic. Any
# integer works; the point is that it never changes, so two runs on the same
# data produce the same fits. See _multi_start_fit for why this matters.
MULTISTART_SEED: int = 20250807

def _fstar(width: float, height: float) -> float:
    dim_min = min(width, height)
    dim_max = max(width, height)
    x = dim_min / dim_max          
    sum_term = sum(
        math.tanh(math.pi * n * x / 2) / n**5
        for n in range(1, 22, 2)
    )
    return 1.0 / ((1 + 1.0 / x)**2 * (1 - (192 / (math.pi**5 * x)) * sum_term))

def compute_reff(width: float = CHANNEL_WIDTH_UM,
                 height: float = CHANNEL_HEIGHT_UM) -> float:
    fs  = _fstar(width, height)
    h_s = min(width, height)
    w_l = max(width, height)
    numerator   = (2.0 / (3.0 * math.pi)) * w_l * (h_s ** 3)
    denominator = (1 + h_s / w_l) ** 2 * fs
    return (numerator / denominator) ** 0.25

DEFAULT_R_EFF: float = compute_reff()

def _kelvin_voigt(t, r_eff, dp, C, E, eta):
    tau = (3 * math.pi * eta) / (C * E)
    return (r_eff * dp) / (C * E) * (1 - np.exp(-t / tau))

def _jeffreys(t, r_eff, dp, C, E, eta1, eta2):
    tau     = (3 * math.pi * eta1) / (C * E)
    elastic = (r_eff * dp) / (C * E) * (1 - np.exp(-t / tau))
    viscous = (r_eff * dp) / (3 * math.pi * eta2) * t
    return elastic + viscous

def _burgers(t, r_eff, dp, C, E1, eta1, E2, eta2):
    maxwell = (r_eff * dp / C) * (1 / E1 + t / (3 * math.pi * eta1))
    tau_kv  = (3 * math.pi * eta2) / (C * E2)
    kv      = (r_eff * dp / (C * E2)) * (1 - np.exp(-t / tau_kv))
    return maxwell + kv

def _linear(t, m, b):
    return m * t + b

def _power_law(t, a, b, c):
    # Added c to the model as requested
    return a * (t) ** b + c

def _exp_uptake(t, A, tau):
    return A * (1 - np.exp(-t / tau))

def _biexp_uptake(t, A1, tau1, A2, tau2):
    """
    Sum of two saturating exponentials, each rising toward its own
    asymptote.  Total plateau = A1 + A2.

    We do NOT enforce tau1 < tau2 through the bounds — curve_fit does
    not support ordered-parameter constraints cleanly.  Instead the
    caller sorts (A, tau) pairs so that tau1 refers to the fast phase
    and tau2 to the slow phase.  Downstream identifiability guard
    lives in the caller as well.
    """
    return A1 * (1 - np.exp(-t / tau1)) + A2 * (1 - np.exp(-t / tau2))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

def _durbin_watson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residuals = y_true - y_pred
    if len(residuals) < 2:
        return np.nan
    diff = np.diff(residuals)
    ss_res = float(np.sum(residuals ** 2))
    if ss_res < 1e-15:
        return np.nan
    return float(np.sum(diff ** 2) / ss_res)

def _aicc(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    n = len(y_true)
    if n == 0: return np.inf
    rss = max(float(np.sum((y_true - y_pred) ** 2)), 1e-12)
    
    dw = _durbin_watson(y_true, y_pred)
    rho = 0.0 if np.isnan(dw) else 1.0 - (dw / 2.0)
    
    n_eff = n * ((1.0 - rho) / (1.0 + rho)) if rho != -1.0 else float(n)
    n_eff = max(float(n_params + 2), min(float(n), n_eff))
    
    penalty = (2.0 * n_params * (n_params + 1)) / (n_eff - n_params - 1)
    return n * math.log(rss / n) + 2.0 * n_params + penalty

def _clean_trace(time: np.ndarray, length: np.ndarray,
                 window: int = 11, n_sigma: float = 2.5,
                 noise_floor_um: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
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

def _estimate_params(t: np.ndarray, l: np.ndarray,
                     r_eff: float, dp: float,
                     C: float) -> Tuple[float, float, float]:
    max_l = max(float(np.max(l)), 1e-9)
    E_guess = (r_eff * dp) / (C * max_l)

    n_late = max(5, len(t) // 3)
    late_t, late_l = t[-n_late:], l[-n_late:]
    if len(late_t) > 2 and np.ptp(late_t) > 0:
        slope      = np.polyfit(late_t, late_l, 1)[0]
        eta2_guess = (r_eff * dp) / (3 * math.pi * slope) if slope > 0.01 else 15000.0
    else:
        eta2_guess = 15000.0

    target     = 0.63 * max_l
    idx        = int(np.argmin(np.abs(l - target)))
    t_63       = float(t[idx])
    eta1_guess = t_63 * C * E_guess / (3 * math.pi) if t_63 > 0 else 5000.0

    E_guess    = float(np.clip(E_guess,    1.0,   100_000.0))
    eta1_guess = float(np.clip(eta1_guess, 1.0,   500_000.0))
    eta2_guess = float(np.clip(eta2_guess, 500.0, 1_000_000.0))

    if eta2_guess < eta1_guess:
        eta2_guess = eta1_guess * 2.0

    return E_guess, eta1_guess, eta2_guess

def _multi_start_fit(func, t: np.ndarray, l: np.ndarray,
                     bounds: Tuple, p0, n_starts: int = 24,
                     seed: int = MULTISTART_SEED):
    """
    Multi-start curve fitting with a reproducible set of starting guesses.

    `curve_fit` is a local optimiser: it walks downhill from wherever you
    start it and stops at the first minimum it reaches. When the error
    surface has more than one minimum, the answer depends on the starting
    point. Trying many starts and keeping the best result is the standard
    way round this.

    Two properties matter for reproducibility:

    1. The guesses come from a private generator created inside this
       function with a fixed seed, rather than from numpy's global random
       state. Nothing else in the program can advance it, so the same cell
       receives the same guesses on every run.
    2. The heuristic guess `p0` is tried first, so the physically motivated
       starting point is always among the candidates.

    Parameters
    ----------
    func : callable
        Model function, called as func(t, *params).
    t, l : np.ndarray
        Time and protrusion length of the trace being fitted.
    bounds : tuple
        (lower, upper) sequences passed straight through to curve_fit.
    p0 : sequence
        Heuristic starting guess, tried first.
    n_starts : int
        Number of random starts in addition to p0. Raise this if the
        seed-sweep audit shows cells changing their selected model.
    seed : int
        Seed for the private random generator.

    Returns
    -------
    (best_params, best_r2), or (None, -inf) if every start failed.
    """
    best_r2, best_p = -np.inf, None
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    safe_lo = np.clip(lower, 1e-9, 1e6)
    safe_hi = np.clip(upper, 1e-9, 1e6)
    use_log = bool(np.all(lower > 0))

    # default_rng builds a generator that belongs to this call alone. Seeding
    # it here means the sequence of guesses is identical every run, whatever
    # else ran beforehand.
    rng = np.random.default_rng(seed)

    # Build the whole candidate list up front, heuristic guess first.
    starting_guesses = [np.asarray(p0, dtype=float)]
    for _ in range(n_starts):
        if use_log:
            # Log-uniform sampling gives each decade equal weight, which
            # suits parameters spanning several orders of magnitude such as
            # eta1 over [1, 500000] Pa s.
            starting_guesses.append(
                np.exp(rng.uniform(np.log(safe_lo), np.log(safe_hi)))
            )
        else:
            # Plain uniform sampling, needed when a bound reaches zero or
            # below and the logarithm is undefined.
            starting_guesses.append(rng.uniform(safe_lo, safe_hi))

    for guess in starting_guesses:
        try:
            popt, _ = curve_fit(func, t, l, p0=guess,
                                bounds=bounds, maxfev=10_000)
            r2 = _r_squared(l, func(t, *popt))
            if r2 > best_r2:
                best_r2, best_p = r2, popt
        except Exception:
            # A start that lands somewhere curve_fit cannot recover from is
            # expected and harmless; move on to the next one.
            continue

    if best_p is None:
        try:
            popt, _ = curve_fit(func, t, l, p0=p0,
                                bounds=bounds, maxfev=50_000)
            best_p  = popt
            best_r2 = _r_squared(l, func(t, *popt))
        except Exception:
            pass

    return best_p, best_r2

def _is_bound_hitting(popt, lower, upper,
                      frac: float = BOUND_PROXIMITY_FRAC) -> List[int]:
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
    n = len(t)
    if n < 5:
        return {'flow_ok': False, 'plateau_ok': False}

    duration = float(t[-1] - t[0])
    if duration <= 0:
        return {'flow_ok': False, 'plateau_ok': False}

    n_side = max(5, int(round(late_frac * n)))

    t_late, l_late = t[-n_side:], l[-n_side:]
    try:
        late_slope = float(np.polyfit(t_late, l_late, 1)[0])
    except Exception:
        late_slope = 0.0

    t_early, l_early = t[:n_side], l[:n_side]
    try:
        early_slope = float(np.polyfit(t_early, l_early, 1)[0])
    except Exception:
        early_slope = 0.0

    flow_ok = abs(late_slope) * duration > snr * noise_um

    if abs(early_slope) < 1e-9:
        plateau_ok = False
    else:
        plateau_ok = (abs(late_slope) / abs(early_slope)) < plateau_ratio

    return {'flow_ok': flow_ok, 'plateau_ok': plateau_ok}

def fit_viscoelastic(time: np.ndarray, length: np.ndarray,
                     r_eff: float, delta_p: float,
                     C: float = HALFSPACE_C,
                     n_starts: int = 24,
                     min_points: int = 15) -> Dict:
    FAIL = {
        'best_model': None, 'best_params': None,
        'best_r2': None, 'best_aicc': None, 'best_dw': None,
        'all_models': {}, 'n_fit_points': 0
    }

    t, l = _clean_trace(time, length)
    if len(t) < min_points:
        logger.debug(f"  Viscoelastic fit skipped: {len(t)} clean points (need {min_points}).")
        return FAIL

    t = t - t[0]   
    E0, eta1_0, eta2_0 = _estimate_params(t, l, r_eff, delta_p, C)

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
    best_aicc  = np.inf
    best_name = None

    for name, cfg in models.items():
        popt, _ = _multi_start_fit(cfg["func"], t, l, cfg["bounds"],
                                   cfg["p0"], n_starts=n_starts)
        if popt is None:
            all_results[name] = {"params": None, "r2": None, "aicc": np.inf,
                                 "dw": None, "bound_hit": None}
            continue

        l_pred = cfg["func"](t, *popt)
        r2     = _r_squared(l, l_pred)
        aicc   = _aicc(l, l_pred, cfg["k"])
        dw     = _durbin_watson(l, l_pred)

        pinned_idx = _is_bound_hitting(popt, cfg["bounds"][0], cfg["bounds"][1])
        pinned_names = [cfg["names"][i] for i in pinned_idx]

        all_results[name] = {
            "params"   : dict(zip(cfg["names"], popt.tolist())),
            "r2"       : r2,
            "aicc"     : aicc,
            "dw"       : dw,
            "bound_hit": pinned_names,   
        }

        if pinned_names:
            logger.debug(
                f"  {name}: parameter(s) {pinned_names} pinned to bound; "
                "excluded from AICc selection."
            )
            continue

        if aicc < best_aicc:
            best_aicc  = aicc
            best_name = name

    if best_name is None:
        return FAIL

    return {
        "best_model"   : best_name,
        "best_params"  : all_results[best_name]["params"],
        "best_r2"      : all_results[best_name]["r2"],
        "best_aicc"    : best_aicc,
        "best_dw"      : all_results[best_name]["dw"],
        "all_models"   : all_results,
        "n_fit_points" : len(t),
    }

def fit_model_independent(time: np.ndarray, length: np.ndarray,
                           min_points: int = 15) -> Dict:
    FAIL = {'linear': None, 'power_law': None, 'best_model': None,
            'window_duration_s': None, 'n_fit_points': 0}
    t, l = _clean_trace(time, length)
    if len(t) < min_points:
        return FAIL

    window_dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    result: Dict = {'window_duration_s': window_dur, 'n_fit_points': len(t)}

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
            'aicc'   : _aicc(l, l_pred, n_params=2),
        }
    except Exception:
        result['linear'] = None

    try:
        popt, _ = curve_fit(_power_law, t, l,
                            p0=[float(l[0]), 0.5, 0.0],
                            bounds=([0, 0, -np.inf], [np.inf, 2, np.inf]),
                            maxfev=10000)
        l_pred = _power_law(t, *popt)
        result['power_law'] = {
            'params': {'a': float(popt[0]), 'exponent_b': float(popt[1]), 'c': float(popt[2])},
            'r2'    : _r_squared(l, l_pred),
            'aicc'   : _aicc(l, l_pred, n_params=3), # AICc now uses 3 params
        }
    except Exception:
        result['power_law'] = None

    lin_aicc = result['linear']['aicc'] if result['linear'] else np.inf
    pl_aicc  = result['power_law']['aicc'] if result['power_law'] else np.inf

    if lin_aicc <= pl_aicc and result['linear']:
        result['best_model'] = 'Linear'
    elif result['power_law']:
        result['best_model'] = 'Power-Law'
    else:
        result['best_model'] = None

    return result

# Identifiability threshold for the bi-exponential uptake fit.  The two
# time constants must differ by at least this factor (tau2 / tau1 >= 3)
# for the split into fast + slow components to be treated as real.
# Fits that pass numerically but collapse below this ratio are excluded
# from AICc model selection and mono-exp wins by default.  Mirrors the
# philosophy of IDENTIFIABILITY_FLOW_SNR in the viscoelastic fit.
UPTAKE_TAU_RATIO_MIN: float = 3.0

# R2 floor for the uptake_R2_Flag column.  Mirrors the visco/MI R2
# convention (0.85) used elsewhere in Chapter 3.
UPTAKE_R2_FLAG_FLOOR: float = 0.85


def fit_exponential_uptake(time: np.ndarray, uptake: np.ndarray,
                            pulse_time_s: float = 0.0,
                            min_points: int = 15) -> Dict:
    """
    Fit dye-uptake time series with AICc model selection between a
    mono-exponential and a bi-exponential saturating rise.

    Returns a dictionary with a 'best_model' key ('Mono' or 'Bi') and
    parameters for BOTH models when they fit successfully, so a
    downstream sensitivity check (e.g. BIC comparison, agreement rate)
    can be run without re-fitting.

    Model selection rules
    ---------------------
    1. Both models are fitted; each contributes an AICc.
    2. Bi-exp is EXCLUDED from AICc selection when tau2 / tau1 <
       UPTAKE_TAU_RATIO_MIN (degenerate fast/slow split) or when either
       amplitude is <5% of the total, which indicates a collapsed
       component that is effectively mono.
    3. The R2 field on the return dict is the R2 of the SELECTED model.
    4. The `flag` field on the return dict is best_R2 >= 0.85, mirroring
       the visco R2 flag convention.

    Returned tau
    ------------
    For a mono winner:  tau = tau_mono.
    For a bi winner:    tau = amplitude-weighted mean of tau1 and tau2
                        (i.e. (A1*tau1 + A2*tau2) / (A1 + A2)).  This
                        keeps the existing `Uptake_*_VolNorm_tau` column
                        interpretable as a characteristic timescale and
                        lets legacy plots (e.g. `plot_uptake_tau_boxplot`)
                        keep working unchanged.
    """
    FAIL = {
        'best_model': None,
        'A': None, 'tau': None, 'r2': None, 'flag': False,
        'A1': None, 'tau1': None, 'A2': None, 'tau2': None,
        'mono_A': None, 'mono_tau': None, 'mono_r2': None, 'mono_aicc': None,
        'bi_A1': None, 'bi_tau1': None, 'bi_A2': None, 'bi_tau2': None,
        'bi_r2': None, 'bi_aicc': None,
        'baseline': None,
    }

    if len(time) < min_points:
        return FAIL

    # ---- Baseline: median of frames at or before the pulse time -----
    t_zero = time - pulse_time_s
    pre_mask = (t_zero <= 0) & np.isfinite(uptake)
    if np.sum(pre_mask) >= 2:
        baseline = float(np.nanmedian(uptake[pre_mask]))
    else:
        valid_mask = np.isfinite(uptake)
        if np.sum(valid_mask) >= 3:
            baseline = float(np.nanmedian(uptake[valid_mask][:3]))
        elif np.sum(valid_mask) > 0:
            baseline = float(uptake[valid_mask][0])
        else:
            baseline = 0.0

    uptake_corrected = uptake - baseline
    mask = (t_zero > 0) & np.isfinite(uptake_corrected) & (uptake_corrected >= 0)
    t_fit, y_fit = t_zero[mask], uptake_corrected[mask]

    if len(t_fit) < min_points:
        return FAIL

    A0   = float(np.nanmax(y_fit))
    T    = float(t_fit[-1] - t_fit[0])
    tau0 = T / 2.0 if T > 0 else 1.0

    # ---- Mono-exponential fit ---------------------------------------
    mono = {'A': None, 'tau': None, 'r2': None, 'aicc': np.inf}
    try:
        popt, _ = curve_fit(_exp_uptake, t_fit, y_fit,
                            p0=[A0, tau0],
                            bounds=([0, 0], [np.inf, np.inf]),
                            maxfev=5000)
        y_pred = _exp_uptake(t_fit, *popt)
        mono['A']    = float(popt[0])
        mono['tau']  = float(popt[1])
        mono['r2']   = _r_squared(y_fit, y_pred)
        mono['aicc'] = _aicc(y_fit, y_pred, n_params=2)
    except Exception:
        pass

    # ---- Bi-exponential fit -----------------------------------------
    # p0 uses a fast/slow split around T/10 and T/2, with amplitudes
    # each half of the observed max — a generic starting point that
    # avoids seeding the two components identically (which would
    # trigger a degenerate optimisation).
    bi = {'A1': None, 'tau1': None, 'A2': None, 'tau2': None,
          'r2': None, 'aicc': np.inf}
    try:
        p0_bi = [A0 / 2.0, max(tau0 / 5.0, 1e-3),
                 A0 / 2.0, max(tau0,       1e-3)]
        popt, _ = curve_fit(_biexp_uptake, t_fit, y_fit,
                            p0=p0_bi,
                            bounds=([0, 0, 0, 0],
                                    [np.inf, np.inf, np.inf, np.inf]),
                            maxfev=10000)
        A1, tau1_raw, A2, tau2_raw = (float(v) for v in popt)

        # Enforce tau1 < tau2 by sorting (tau, A) pairs so that
        # tau1 always refers to the FAST component.
        if tau1_raw > tau2_raw:
            A1, A2 = A2, A1
            tau1_raw, tau2_raw = tau2_raw, tau1_raw

        y_pred = _biexp_uptake(t_fit, A1, tau1_raw, A2, tau2_raw)
        bi['A1']   = A1
        bi['tau1'] = tau1_raw
        bi['A2']   = A2
        bi['tau2'] = tau2_raw
        bi['r2']   = _r_squared(y_fit, y_pred)
        bi['aicc'] = _aicc(y_fit, y_pred, n_params=4)
    except Exception:
        pass

    # ---- Identifiability guard on bi-exp -----------------------------
    # Exclude bi from selection when the split is degenerate.  The
    # numerical fit can still succeed on a near-mono trace, so we check
    # the tau ratio and the relative amplitudes explicitly.
    bi_identifiable = False
    if bi['tau1'] is not None and bi['tau2'] is not None:
        total_A = (bi['A1'] or 0.0) + (bi['A2'] or 0.0)
        if bi['tau1'] > 1e-9 and total_A > 1e-9:
            tau_ratio    = bi['tau2'] / bi['tau1']
            amp_min_frac = min(bi['A1'], bi['A2']) / total_A
            if tau_ratio >= UPTAKE_TAU_RATIO_MIN and amp_min_frac >= 0.05:
                bi_identifiable = True

    # ---- AICc selection ---------------------------------------------
    mono_aicc_eff = mono['aicc'] if mono['A'] is not None else np.inf
    bi_aicc_eff   = bi['aicc']   if bi_identifiable         else np.inf

    if mono_aicc_eff == np.inf and bi_aicc_eff == np.inf:
        return FAIL

    if bi_aicc_eff < mono_aicc_eff:
        best_model = 'Bi'
        best_A     = (bi['A1'] or 0.0) + (bi['A2'] or 0.0)
        best_tau   = (bi['A1'] * bi['tau1'] + bi['A2'] * bi['tau2']) / best_A \
                     if best_A > 1e-9 else np.nan
        best_r2    = bi['r2']
    else:
        best_model = 'Mono'
        best_A     = mono['A']
        best_tau   = mono['tau']
        best_r2    = mono['r2']

    return {
        'best_model': best_model,
        'A'   : best_A,
        'tau' : best_tau,
        'r2'  : best_r2,
        'flag': bool(best_r2 is not None and best_r2 >= UPTAKE_R2_FLAG_FLOOR),
        # Selected-model expanded fields (NaN when the other model won).
        'A1'  : bi['A1']   if best_model == 'Bi' else None,
        'tau1': bi['tau1'] if best_model == 'Bi' else None,
        'A2'  : bi['A2']   if best_model == 'Bi' else None,
        'tau2': bi['tau2'] if best_model == 'Bi' else None,
        # Full per-model records (populated whenever the fit converged),
        # kept so a supplementary BIC-agreement check can be run
        # post hoc without re-fitting.
        'mono_A'   : mono['A'],
        'mono_tau' : mono['tau'],
        'mono_r2'  : mono['r2'],
        'mono_aicc': mono['aicc'] if mono['aicc'] != np.inf else None,
        'bi_A1'   : bi['A1'],
        'bi_tau1' : bi['tau1'],
        'bi_A2'   : bi['A2'],
        'bi_tau2' : bi['tau2'],
        'bi_r2'   : bi['r2'],
        'bi_aicc' : bi['aicc'] if bi['aicc'] != np.inf else None,
        'baseline': baseline,
    }

def compute_common_duration(traps: List[bfh.TrapData],
                            min_duration_fraction: float = 0.50) -> Optional[float]:
    durations = []
    for trap in traps:
        t = trap.protrusion_data.get('Time_s', np.array([]))
        if len(t) > 1:
            durations.append(float(t[-1] - t[0]))

    if len(durations) < 2:
        return None

    mean_dur   = float(np.mean(durations))
    threshold  = min_duration_fraction * mean_dur
    surviving_durations = [d for d in durations if d >= threshold]

    if len(surviving_durations) < 2:
        return None

    common = float(min(surviving_durations))
    logger.info(
        f"  Common duration: {common:.1f} s "
        f"(mean={mean_dur:.1f}, {len(durations) - len(surviving_durations)} excluded)"
    )
    return common

def compute_common_ep_pre_duration(traps: List[bfh.TrapData],
                                   min_duration_fraction: float = 0.20) -> Optional[float]:
    pre_durations = []
    for trap in traps:
        if trap.metadata.condition_type != "EP":
            continue
        t  = trap.protrusion_data.get('Time_s', np.array([]))
        pf = trap.metadata.pulse_frame
        if len(t) < 10 or not (10 <= pf < len(t)):
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
    return common

def compute_common_ep_post_duration(traps: List[bfh.TrapData],
                                    max_window_s: float = 5.0) -> Optional[float]:
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
    return common

def _make_condition_label(meta: bfh.ExperimentMetadata) -> str:
    if meta.condition_type == "ASP":
        return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_ASP"
    return (f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_"
            f"{meta.voltage}V_{meta.duration_label}")

def _run_trap_mechanics(trap: bfh.TrapData, r_eff: float, C: float,
                        r2_floor: float = 0.0,
                        mi_r2_floor_asp: float = 0.85,
                        mi_r2_floor_ep: float = 0.80,
                        global_asp_dur: Optional[float] = None,
                        global_pre_dur: Optional[float] = None,
                        global_whole_dur: Optional[float] = None) -> Dict:
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
        'Duration_ms'       : meta.duration,
        'Duration_label'    : meta.duration_label,
        'Condition_Type'    : meta.condition_type,
        'Fate_Status'       : getattr(trap, 'fate_status', 'intact'),
        'Condition'         : _make_condition_label(meta),
        'Trap_ID'           : trap.trap_id,
        'Experiment_Folder' : meta.full_path.name,
        'Son_Factor_fstar'  : meta.fstar,
        'Best_Model'        : None,
        'E_Pa'              : None,
        'E1_Pa'             : None,
        'eta1_Pa_s'         : None,
        'eta2_Pa_s'         : None,
        'Tau_s'             : None,
        'Visco_R2'          : None,
        'Visco_AICc'        : None,
        'Visco_DW'          : None,
        'Visco_R2_Flag'     : None,
        # Pre-pulse-matched-window viscoelastic fits. For ASP cells the
        # window is truncated from aspiration onset to `global_pre_dur`
        # (matching the horizon available to EP-pre cells). For EP cells
        # the window is the pre-pulse segment up to `global_pre_dur`
        # before the pulse. Both cohorts therefore enter the pre-pulse
        # three-way comparison (ASP vs EP-intact vs EP-ruptured) with
        # matched fit horizons. The long-window ASP fits above remain
        # untouched and continue to serve the aspiration-only subsection.
        'PrePulse_Best_Model'  : None,
        'PrePulse_E_Pa'        : None,
        'PrePulse_E1_Pa'       : None,
        'PrePulse_eta1_Pa_s'   : None,
        'PrePulse_eta2_Pa_s'   : None,
        'PrePulse_Tau_s'       : None,
        'PrePulse_Visco_R2'    : None,
        'PrePulse_Visco_AICc'  : None,
        'PrePulse_Visco_DW'    : None,
        'PrePulse_Visco_R2_Flag': None,
        'PrePulse_Visco_Window_Duration_s': None,
        'PrePulse_Visco_N_Points': None,
        'MI_Whole_R2_Flag'  : None,
        'MI_Whole_Best_Model'         : None,
        'MI_Whole_Linear_Slope'       : None,
        'MI_Whole_Linear_Intercept'   : None,
        'MI_Whole_Linear_R2'          : None,
        'MI_Whole_Linear_AICc'        : None,
        'MI_Whole_PL_a'               : None,
        'MI_Whole_PL_b'               : None,
        'MI_Whole_PL_R2'              : None,
        'MI_Whole_PL_AICc'            : None,
        'MI_Whole_Window_Duration_s'  : None,
        'MI_Whole_N_Points'           : None,
        'Pre_Pulse_Slope'       : None,
        'Post_Pulse_Slope'      : None,
        'EP_Post_Pulse_Behavior': None,
        # Uptake fit columns.  Naming convention (per region):
        #   *_A         plateau of the selected model
        #                (mono: A; bi-exp: A1 + A2)
        #   *_tau       characteristic timescale of the selected model
        #                (mono: tau; bi-exp: amplitude-weighted mean)
        #   *_R2        R2 of the selected model
        #   *_R2_Flag   True iff R2 >= UPTAKE_R2_FLAG_FLOOR (0.85)
        #   *_Best_Model 'Mono' or 'Bi'
        #   *_A1, *_tau1, *_A2, *_tau2   populated when bi wins
        #   *_Mono_AICc, *_Bi_AICc       both AICcs (whenever fit converged)
        'Uptake_Body_VolNorm_Best_Model' : None,
        'Uptake_Body_VolNorm_A'          : None,
        'Uptake_Body_VolNorm_tau'        : None,
        'Uptake_Body_VolNorm_R2'         : None,
        'Uptake_Body_VolNorm_R2_Flag'    : False,
        'Uptake_Body_VolNorm_A1'         : None,
        'Uptake_Body_VolNorm_tau1'       : None,
        'Uptake_Body_VolNorm_A2'         : None,
        'Uptake_Body_VolNorm_tau2'       : None,
        'Uptake_Body_VolNorm_Mono_AICc'  : None,
        'Uptake_Body_VolNorm_Bi_AICc'    : None,

        'Uptake_Prot_VolNorm_Best_Model' : None,
        'Uptake_Prot_VolNorm_A'          : None,
        'Uptake_Prot_VolNorm_tau'        : None,
        'Uptake_Prot_VolNorm_R2'         : None,
        'Uptake_Prot_VolNorm_R2_Flag'    : False,
        'Uptake_Prot_VolNorm_A1'         : None,
        'Uptake_Prot_VolNorm_tau1'       : None,
        'Uptake_Prot_VolNorm_A2'         : None,
        'Uptake_Prot_VolNorm_tau2'       : None,
        'Uptake_Prot_VolNorm_Mono_AICc'  : None,
        'Uptake_Prot_VolNorm_Bi_AICc'    : None,

        'Uptake_Total_VolNorm_Best_Model': None,
        'Uptake_Total_VolNorm_A'         : None,
        'Uptake_Total_VolNorm_tau'       : None,
        'Uptake_Total_VolNorm_R2'        : None,
        'Uptake_Total_VolNorm_R2_Flag'   : False,
        'Uptake_Total_VolNorm_A1'        : None,
        'Uptake_Total_VolNorm_tau1'      : None,
        'Uptake_Total_VolNorm_A2'        : None,
        'Uptake_Total_VolNorm_tau2'      : None,
        'Uptake_Total_VolNorm_Mono_AICc' : None,
        'Uptake_Total_VolNorm_Bi_AICc'   : None,
        # Actin summary columns.  Values are per-region means over the
        # pre/post-pulse window, F0-normalised (each cell uses its own
        # F0_Body / F0_Prot as the reference).  For ASP cells the pre-pulse
        # column holds the whole-trace mean and the post-pulse column is
        # NaN.  For EP cells the split is at meta.pulse_frame (known
        # off-by-one; see comment in _apply_actin_summaries).
        'Actin_Body_PrePulse_F0Norm'  : None, 'Actin_Body_PostPulse_F0Norm' : None,
        'Actin_Prot_PrePulse_F0Norm'  : None, 'Actin_Prot_PostPulse_F0Norm' : None,
        # Raw F0 values (per-cell pre-aspiration baselines), passed
        # through from the actin CSV.  Enable between-condition raw-actin
        # comparisons that F0-normalisation collapses.
        'F0_Body': None, 'F0_Prot': None,
        'Cell_Body_Area_um2': None,
        'Cell_Prot_Area_um2': None,
        'Cell_Body_Area_PrePulse_um2': None,
        'Cell_Body_Volume_PrePulse_um3': None,
        'Max_Prot_length_PrePulse_um': None,
    }

    time_raw   = pd_data.get('Time_s',               np.array([]))
    length_raw = pd_data.get('Protrusion_Length_um', np.array([]))

    if len(time_raw) < 15 or len(length_raw) < 15:
        return row

    is_asp = (meta.condition_type == "ASP")
    # Post-pulse-entry cells are now excluded at load time (bulk_file_handling),
    # so is_asp_like collapses to is_asp and is_ep_standard collapses to
    # 'EP'.  The variables are kept under their old names to minimise churn
    # in the rest of this function.
    is_asp_like = is_asp
    is_ep_standard = (meta.condition_type == "EP")

    pf = meta.pulse_frame
    pf_valid = (0 < pf < len(time_raw))
    pulse_time_s = float(time_raw[pf]) if pf_valid else 0.0

    if meta.condition_type == "EP" and not pf_valid:
        logger.warning(
            f"  Trap {trap.trap_id} ({meta.experiment_number}): "
            f"pulse_frame={pf} is out of range. Skipping pulse-aligned analyses."
        )

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
            median_area = float(np.nanmedian(pre_area))
            row['Cell_Body_Area_PrePulse_um2'] = median_area
            # Sphere-equivalent volume from segmented area, matching the
            # convention used in bulk_file_handling._recompute_volumes_and_norms
            # for the body region.  V = (4/(3*sqrt(pi))) * A^(3/2)
            if np.isfinite(median_area) and median_area > 0:
                row['Cell_Body_Volume_PrePulse_um3'] = (
                    (4.0 / (3.0 * math.sqrt(math.pi))) * (median_area ** 1.5)
                )

    if pf_valid:
        pre_pulse_lengths = length_raw[:pf]
    else:
        pre_pulse_lengths = length_raw

    valid_lengths = pre_pulse_lengths[np.isfinite(pre_pulse_lengths) & (pre_pulse_lengths > 0)]
    if len(valid_lengths) > 0:
        row['Max_Prot_length_PrePulse_um'] = float(np.max(valid_lengths))

    if is_asp:
        t_asp = time_raw - time_raw[0]
        if global_asp_dur is not None:
            win_mask = t_asp <= global_asp_dur
        else:
            win_mask = np.ones_like(t_asp, dtype=bool)
            
        t_visco = time_raw[win_mask]
        l_visco = length_raw[win_mask]
        
        if len(t_visco) >= 15:
            visco = fit_viscoelastic(t_visco - t_visco[0], l_visco, r_eff, meta.pressure, C)

            if visco['best_model']:
                row['Best_Model'] = visco['best_model']
                row['Visco_R2']   = visco['best_r2']
                row['Visco_AICc'] = visco['best_aicc']
                row['Visco_DW']   = visco['best_dw']
                row['Visco_R2_Flag'] = (
                    visco['best_r2'] >= r2_floor
                    if visco['best_r2'] is not None else False
                )
                best_p = visco['best_params']

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
                    tau_val = (3 * math.pi * eta2_kv) / (C * E2_val) if E2_val and eta2_kv else None

                    row.update({
                        'E_Pa'     : E2_val,
                        'E1_Pa'    : E1_mw,
                        'eta1_Pa_s': eta2_kv,
                        'eta2_Pa_s': eta1_mw,
                        'Tau_s'    : tau_val,
                    })

    # -------------------------------------------------------------------
    # Pre-pulse-matched viscoelastic fits.
    # ASP cells are re-fit on the shorter `global_pre_dur` window so the
    # ASP viscoelastic parameters can be placed alongside EP-pre fits on
    # a matched horizon in the paired mechanical-electroporation
    # subsection. EP-pre cells are fit on the aspiration segment that
    # precedes the pulse (also capped at `global_pre_dur`). Results are
    # written to PrePulse_* columns and do not overwrite the long-window
    # ASP fits above.
    # -------------------------------------------------------------------
    t_pp_visco = None
    l_pp_visco = None

    if is_asp:
        t_asp_zero = time_raw - time_raw[0]
        if global_pre_dur is not None:
            pp_mask = t_asp_zero <= global_pre_dur
        else:
            pp_mask = np.ones_like(t_asp_zero, dtype=bool)

        if pp_mask.sum() >= 15:
            t_pp_visco = time_raw[pp_mask]
            l_pp_visco = length_raw[pp_mask]

    elif is_ep_standard and pf_valid:
        t_aligned_pp = time_raw - time_raw[pf]
        pre_mask_pp  = (t_aligned_pp < 0) & (length_raw > 0)
        if global_pre_dur is not None:
            pre_mask_pp = pre_mask_pp & (t_aligned_pp >= -global_pre_dur)

        if np.sum(pre_mask_pp) >= 15:
            t_pp_visco = time_raw[pre_mask_pp]
            l_pp_visco = length_raw[pre_mask_pp]

    if t_pp_visco is not None and len(t_pp_visco) >= 15:
        t_pp_zero = t_pp_visco - t_pp_visco[0]
        pp_visco = fit_viscoelastic(t_pp_zero, l_pp_visco, r_eff, meta.pressure, C)

        if pp_visco['best_model']:
            row['PrePulse_Best_Model'] = pp_visco['best_model']
            row['PrePulse_Visco_R2']   = pp_visco['best_r2']
            row['PrePulse_Visco_AICc'] = pp_visco['best_aicc']
            row['PrePulse_Visco_DW']   = pp_visco['best_dw']
            row['PrePulse_Visco_R2_Flag'] = (
                pp_visco['best_r2'] >= r2_floor
                if pp_visco['best_r2'] is not None else False
            )
            row['PrePulse_Visco_Window_Duration_s'] = float(t_pp_zero[-1] - t_pp_zero[0])
            row['PrePulse_Visco_N_Points'] = int(len(t_pp_zero))

            pp_best_p = pp_visco['best_params']
            if pp_visco['best_model'] == 'Kelvin-Voigt':
                E_pp   = pp_best_p['E']
                eta_pp = pp_best_p['eta']
                row['PrePulse_E_Pa']      = E_pp
                row['PrePulse_eta1_Pa_s'] = eta_pp
                row['PrePulse_Tau_s']     = (
                    (3 * math.pi * eta_pp) / (C * E_pp) if E_pp else None
                )
            elif pp_visco['best_model'] == 'Jeffreys':
                E_pp    = pp_best_p['E']
                eta1_pp = pp_best_p['eta1']
                eta2_pp = pp_best_p['eta2']
                row['PrePulse_E_Pa']      = E_pp
                row['PrePulse_eta1_Pa_s'] = eta1_pp
                row['PrePulse_eta2_Pa_s'] = eta2_pp
                row['PrePulse_Tau_s']     = (
                    (3 * math.pi * eta1_pp) / (C * E_pp) if E_pp else None
                )
            elif pp_visco['best_model'] == 'Burgers':
                E1_mw_pp   = pp_best_p.get('E1')
                E2_val_pp  = pp_best_p.get('E2')
                eta2_kv_pp = pp_best_p.get('eta2')
                eta1_mw_pp = pp_best_p.get('eta1')
                tau_val_pp = (
                    (3 * math.pi * eta2_kv_pp) / (C * E2_val_pp)
                    if E2_val_pp and eta2_kv_pp else None
                )
                row['PrePulse_E_Pa']      = E2_val_pp
                row['PrePulse_E1_Pa']     = E1_mw_pp
                row['PrePulse_eta1_Pa_s'] = eta2_kv_pp
                row['PrePulse_eta2_Pa_s'] = eta1_mw_pp
                row['PrePulse_Tau_s']     = tau_val_pp

    t_whole = time_raw - time_raw[0]
    if global_whole_dur is not None:
        win_mask = t_whole <= global_whole_dur
    else:
        win_mask = np.ones_like(t_whole, dtype=bool)
        
    if win_mask.sum() >= 15:
        mi_whole_fit = fit_model_independent(t_whole[win_mask], length_raw[win_mask])
        
        row['MI_Whole_Best_Model'] = mi_whole_fit.get('best_model')
        if mi_whole_fit['linear']:
            row['MI_Whole_Linear_Slope']     = mi_whole_fit['linear']['params']['slope']
            row['MI_Whole_Linear_Intercept'] = mi_whole_fit['linear']['params']['intercept']
            row['MI_Whole_Linear_R2']        = mi_whole_fit['linear']['r2']
            row['MI_Whole_Linear_AICc']      = mi_whole_fit['linear']['aicc']
        if mi_whole_fit['power_law']:
            row['MI_Whole_PL_a']  = mi_whole_fit['power_law']['params']['a']
            row['MI_Whole_PL_b']  = mi_whole_fit['power_law']['params']['exponent_b']
            row['MI_Whole_PL_c'] = mi_whole_fit['power_law']['params']['c']
            row['MI_Whole_PL_R2'] = mi_whole_fit['power_law']['r2']
            row['MI_Whole_PL_AICc'] = mi_whole_fit['power_law']['aicc']
        row['MI_Whole_Window_Duration_s'] = mi_whole_fit.get('window_duration_s')
        row['MI_Whole_N_Points']          = mi_whole_fit.get('n_fit_points')

    if is_ep_standard and pf_valid:
        t_aligned = time_raw - time_raw[pf]
        pre_mask  = (t_aligned >= -5.0) & (t_aligned < 0) & (length_raw > 0)
        post_mask = (t_aligned >= 0) & (t_aligned <= 2.0) & (length_raw > 0)

        if np.sum(pre_mask) > 2:
            row['Pre_Pulse_Slope'] = float(
                np.polyfit(t_aligned[pre_mask], length_raw[pre_mask], 1)[0])
        if np.sum(post_mask) > 2:
            post_slope = float(
                np.polyfit(t_aligned[post_mask], length_raw[post_mask], 1)[0])
            row['Post_Pulse_Slope'] = post_slope

            thr = EP_BEHAVIOR_SLOPE_THRESHOLD
            if post_slope > thr:
                row['EP_Post_Pulse_Behavior'] = 'Extends'
            elif post_slope < -thr:
                row['EP_Post_Pulse_Behavior'] = 'Retracts'
            else:
                row['EP_Post_Pulse_Behavior'] = 'Stable'
                
    # Evaluate MI Whole-Trace Flag
    mi_whole_best = row.get('MI_Whole_Best_Model')
    if mi_whole_best == 'Linear':
        mi_whole_r2 = row.get('MI_Whole_Linear_R2')
    elif mi_whole_best == 'Power-Law':
        mi_whole_r2 = row.get('MI_Whole_PL_R2')
    else:
        mi_whole_r2 = None

    if mi_whole_r2 is not None:
        thresh = mi_r2_floor_ep if meta.condition_type == 'EP' else mi_r2_floor_asp
        row['MI_Whole_R2_Flag'] = bool(mi_whole_r2 >= thresh)
    else:
        row['MI_Whole_R2_Flag'] = False

    if ud_data and 'Time_s' in ud_data:
        t_up = ud_data['Time_s']

        for col_base, region in [
            ('Uptake_Body_VolNorm',  'Body_VolNorm'),
            ('Uptake_Prot_VolNorm',  'Protrusion_VolNorm'),
            ('Uptake_Total_VolNorm', 'Total_VolNorm'),
        ]:
            if region in ud_data:
                fit = fit_exponential_uptake(t_up, ud_data[region],
                                             pulse_time_s=pulse_time_s)
                # Selected-model summary (drives all downstream plots).
                row[f'{col_base}_Best_Model'] = fit['best_model']
                row[f'{col_base}_A']          = fit['A']
                row[f'{col_base}_tau']        = fit['tau']
                row[f'{col_base}_R2']         = fit['r2']
                row[f'{col_base}_R2_Flag']    = fit['flag']
                # Bi-exp expanded fields (NaN when mono wins).
                row[f'{col_base}_A1']         = fit['A1']
                row[f'{col_base}_tau1']       = fit['tau1']
                row[f'{col_base}_A2']         = fit['A2']
                row[f'{col_base}_tau2']       = fit['tau2']
                # Both AICcs, for the supplementary sensitivity check.
                row[f'{col_base}_Mono_AICc']  = fit['mono_aicc']
                row[f'{col_base}_Bi_AICc']    = fit['bi_aicc']

    # ---------- Actin pre/post-pulse means ----------------------------
    # For ASP cells: pre = whole-trace mean, post = NaN.
    # For EP cells:  split at pulse_frame.  Known limitation: pulse_frame
    # indexing is off-by-one relative to the actual pulse timing
    # (Zeiss ZEN single-frame TIFFs strip per-frame timestamps, and the
    # index is reconstructed from the config frame interval).  One frame
    # of actin data straddles the boundary.  The paired within-cell
    # numbers should not be quoted in the thesis until this is resolved;
    # unpaired between-condition means are unaffected.
    _apply_actin_summaries(row, getattr(trap, 'actin_data', {}), meta)

    return row


def _apply_actin_summaries(row: dict,
                           actin_data: Dict[str, np.ndarray],
                           meta: 'bfh.ExperimentMetadata') -> None:
    """
    Populate Actin_{Body,Prot}_{Pre,Post}Pulse_F0Norm on `row` in place.

    Each region's mean intensity over the window is divided by that
    region's F0 (pre-aspiration baseline stored per-cell in the actin
    CSV as F0_Body / F0_Prot).  F0-normalisation makes traces comparable
    across cells and conditions when imaging settings are matched.

    ASP cells: pre = full-trace mean, post columns left as None.
    EP  cells: split at meta.pulse_frame; one-frame boundary error noted
               at module level in _run_trap_mechanics.

    Silent no-op if actin_data is missing or lacks the expected columns.
    """
    if not actin_data:
        return

    body = actin_data.get('Actin_Body_Mean')
    prot = actin_data.get('Actin_Prot_Mean')
    if body is None and prot is None:
        return

    # F0 columns hold a per-cell scalar broadcast across all rows.  Pick
    # the first finite value; treat non-positive / non-finite as missing.
    def _scalar_f0(key: str) -> Optional[float]:
        arr = actin_data.get(key)
        if arr is None or len(arr) == 0:
            return None
        for v in arr:
            if np.isfinite(v) and v > 0:
                return float(v)
        return None

    f0_body = _scalar_f0('F0_Body')
    f0_prot = _scalar_f0('F0_Prot')

    # Pass through raw F0 baselines so downstream plots have access to
    # between-condition baseline differences that F0-normalisation collapses.
    row['F0_Body'] = f0_body
    row['F0_Prot'] = f0_prot

    def _mean_norm(arr, sl, f0):
        if arr is None or f0 is None:
            return None
        window = arr[sl]
        if len(window) == 0:
            return None
        # Zero values are segmentation failures (empty mask), not real
        # zero-intensity measurements; drop them before averaging.
        valid = np.isfinite(window) & (window > 0)
        if valid.sum() == 0:
            return None
        val = float(np.mean(window[valid]))
        return val / f0

    n = len(body) if body is not None else len(prot)

    if meta.condition_type == 'ASP':
        row['Actin_Body_PrePulse_F0Norm']  = _mean_norm(body, slice(0, n), f0_body)
        row['Actin_Prot_PrePulse_F0Norm']  = _mean_norm(prot, slice(0, n), f0_prot)
        return

    # EP: split at pulse_frame.  Guard against out-of-range values.
    pf = meta.pulse_frame
    if pf <= 0 or pf >= n:
        row['Actin_Body_PrePulse_F0Norm'] = _mean_norm(body, slice(0, n), f0_body)
        row['Actin_Prot_PrePulse_F0Norm'] = _mean_norm(prot, slice(0, n), f0_prot)
        return

    row['Actin_Body_PrePulse_F0Norm']  = _mean_norm(body, slice(0, pf),  f0_body)
    row['Actin_Body_PostPulse_F0Norm'] = _mean_norm(body, slice(pf, n),  f0_body)
    row['Actin_Prot_PrePulse_F0Norm']  = _mean_norm(prot, slice(0, pf),  f0_prot)
    row['Actin_Prot_PostPulse_F0Norm'] = _mean_norm(prot, slice(pf, n),  f0_prot)

def run_all_mechanics(grouped_data: Dict,
                      r_eff: float = DEFAULT_R_EFF,
                      C: float = HALFSPACE_C,
                      r2_floor: float = 0.0,
                      mi_r2_floor_asp: float = 0.85,
                      mi_r2_floor_ep: float = 0.80,
                      global_asp_dur: Optional[float] = None,
                      global_pre_dur: Optional[float] = None,
                      global_whole_dur: Optional[float] = None) -> pd.DataFrame:
    rows  = []
    total = sum(len(v) for v in grouped_data.values())
    logger.info(
        f"Running mechanics fitting on {total} traps "
        f"(r_eff = {r_eff:.3f} um, C = {C}, R² floor = {r2_floor}) ..."
    )

    for key, traps in grouped_data.items():
        for trap in traps:
            try:
                row = _run_trap_mechanics(
                    trap, r_eff, C, r2_floor=r2_floor,
                    mi_r2_floor_asp=mi_r2_floor_asp,
                    mi_r2_floor_ep=mi_r2_floor_ep,
                    global_asp_dur=global_asp_dur,
                    global_pre_dur=global_pre_dur,
                    global_whole_dur=global_whole_dur
                )
                rows.append(row)
            except Exception as e:
                logger.warning(
                    f"  Trap {trap.trap_id} "
                    f"({trap.metadata.experiment_number}): {e}"
                )

    df = pd.DataFrame(rows)

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