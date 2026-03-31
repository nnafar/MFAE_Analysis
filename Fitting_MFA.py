# -*- coding: utf-8 -*-
"""
Performs multi-model fitting with rupture detection.

ROLE IN PIPELINE:
This module extracts physical parameters from the raw tracking data.
It fits the "Protrusion Length vs. Time" curve to various viscoelastic models.

KEY FEATURES:
1.  Data Cleaning: Automatically removes outliers (sudden spikes) and 
    initial entry artifacts before fitting.
2.  Multi-Model Competition: Fits the data to multiple models (Jeffreys, 
    Burgers, Power Law) and selects the winner based on R-squared.
3.  Parameter Estimation: Uses geometric data (channel size) to guess 
    initial parameters, improving fit convergence.
"""
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.ndimage import median_filter, generic_filter
import pandas as pd

import Calculation_MFA as calc

logger = logging.getLogger(__name__)


class FittingMFA:
    """
    Manages the multi-model fitting process for a single dataset (one trap).
    """

    def __init__(self, t_data: np.ndarray, l_data: np.ndarray, r_eff: float, delta_p: float, C: float = 1.0,
                 rupture_detected: bool = False, rupture_time: Optional[float] = None,
                 phase: str = 'full') -> None:
        """
        Initializes the fitter with experimental data and physical constants.
        
        Args:
            t_data: Time points [seconds]. Must already be truncated at the rupture
                    point by the caller before being passed in. For post-pulse fits,
                    pass time re-zeroed to the pulse frame (t - t_pulse).
            l_data: Protrusion lengths [microns]. Same truncation applies. For post-pulse
                    fits, pass lengths offset by L at the pulse (l - l_pulse), so the
                    window starts at ~0 and the physical model assumptions still hold.
            r_eff: Effective radius of the channel [microns] (Geometric factor)
            delta_p: Applied pressure [Pascals]
            C: Geometric correction factor (usually ~1.0 or incorporated into r_eff)
            rupture_detected: Whether LineDetectionMFA found a rupture event.
            rupture_time: Time (seconds) of the rupture, as reported by LineDetectionMFA.
            phase: Label for which portion of the trace this fit covers.
                   One of 'full', 'pre_pulse', or 'post_pulse'. Used for reporting only.
        """
        self.t_raw = np.asarray(t_data, dtype=float)
        self.l_raw = np.asarray(l_data, dtype=float)
        self.r_eff = r_eff
        self.delta_p = delta_p
        self.C = C
        self.phase = phase  # 'full', 'pre_pulse', or 'post_pulse'
        
        # This dictionary will be populated by the run() method
        self.params: Dict[str, Any] = {}
        self.debug_mode: bool = False

        # Processed data containers (Cleaned/Filtered data)
        self.t_fit: Optional[np.ndarray] = None
        self.l_fit: Optional[np.ndarray] = None

        # --- Multi-model fitting results ---
        # Stores result dicts for every model tried
        self.fit_results: Dict[str, Dict[str, Any]] = {}
        self.best_fit_model_name: Optional[str] = None
        self.best_fit_params: Optional[Dict[str, Any]] = None

        # Rupture info â€” set from LineDetectionMFA results passed in by the caller.
        # This fitter never re-detects rupture; data it receives is already truncated.
        self.rupture_detected: bool = rupture_detected
        self.rupture_time: Optional[float] = rupture_time
        self.rupture_index: Optional[int] = None

        # Processing metadata
        self.processing_log: List[str] = []
        
        self.initial_guess: Optional[Tuple[float, float, float]] = None
        self.models_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def run(self, params: Optional[Dict[str, Any]] = None) -> 'FittingMFA':
        """Executes the complete data processing and fitting pipeline."""
        self.params = params or {}
        self.debug_mode = self.params.get('debug_mode', False)
        
        try:
            # Step 1: Clean data (remove outliers, cut pre-entry noise)
            self._validate_and_prepare_data(self.params)
            # Step 2: Fit all models
            self._perform_multi_model_fitting()
            # Step 3: Pick the winner
            self._determine_best_fit()
        except Exception as e:
            if self.debug_mode: logger.error(f"Fitting process failed: {e}", exc_info=True)
            self.best_fit_model_name = None
        return self

    def _validate_and_prepare_data(self, params: Dict[str, Any]) -> None:
        """
        Validates, cleans, and processes data for fitting.
        Crucial for preventing bad fits due to tracking noise.
        """
        if len(self.t_raw) != len(self.l_raw):
            raise ValueError("Time and length arrays must have the same length.")
        
        # Pre-pulse windows are intentionally short (often < 15 frames).
        # Full and post-pulse fits require more points for statistical validity.
        min_points = 5 if self.phase == 'pre_pulse' else 15
        if len(self.t_raw) < min_points:
            raise ValueError(f"Insufficient data points ({len(self.t_raw)}) for phase '{self.phase}'. Need at least {min_points}.")

        # Filter out 0 or negative values (pre-entry phase)
        valid_mask = np.isfinite(self.t_raw) & np.isfinite(self.l_raw) & (self.l_raw > 0)
        
        sorted_indices = np.argsort(self.t_raw[valid_mask])
        t_processed = self.t_raw[valid_mask][sorted_indices]
        l_processed = self.l_raw[valid_mask][sorted_indices]
        
        min_after_filter = 4 if self.phase == 'pre_pulse' else 10
        if len(t_processed) < min_after_filter:
            raise ValueError(f"Insufficient non-zero data points ({len(t_processed)}) for phase '{self.phase}' after filtering.")

        # Remove the very beginning of entry if requested (often unstable)
        n_exclude = int(len(t_processed) * params.get('exclude_early_fraction', 0.01))
        # If n_exclude equals the length of the array (e.g., very short data), 
        # the array becomes empty and curve_fit will crash.
        if n_exclude >= len(t_processed) - 5:
            n_exclude = 0 # Safety fallback
        
        t_processed, l_processed = t_processed[n_exclude:], l_processed[n_exclude:]
        
        # --- STATISTICAL OUTLIER REJECTION ---
        # Uses a rolling window to find points that deviate > N sigma from local median.
        if params.get('enable_outlier_rejection', True):
            window_size = int(params.get('outlier_rejection_window', 11))
            if window_size % 2 == 0:
                window_size += 1 # Ensure window is odd
            std_dev_threshold = float(params.get('outlier_rejection_std_dev', 2.5))
            
            if len(l_processed) > window_size:
                # 1. Calculate local median (robust baseline)
                rolling_med = median_filter(l_processed, size=window_size, mode='nearest')
                
                # 2. Calculate local standard deviation
                rolling_std = pd.Series(l_processed).rolling(
                    window=window_size, center=True, min_periods=1
                ).std().values
                
                # Pandas rolling.std() returns NaN if it encounters identical values 
                # or a single point; convert these to 0.0 to prevent bounds from becoming NaN.
                rolling_std = np.nan_to_num(rolling_std, nan=0.0)
                
                # Define a physical noise floor (e.g., 0.1 pixels converted to microns)
                scale_factor = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
                min_noise_floor = 0.5 * scale_factor  # Increased from 0.1 to prevent bounds collapse
                
                # Prevent bounds from collapsing to zero if the cell is perfectly still
                rolling_std = np.maximum(rolling_std, min_noise_floor)
                
                # 3. Define the allowable range (Threshold)
                lower_bound = rolling_med - (std_dev_threshold * rolling_std)
                upper_bound = rolling_med + (std_dev_threshold * rolling_std)

                # 4. Filter: Remove points outside either bound.
                #    Lower bound removes downward spikes (cell retraction artifacts).
                #    Upper bound removes upward spikes (tracking jumps to spurious blobs).
                #    Both directions of tracking failure occur in real data.
                outlier_mask = (l_processed >= lower_bound) & (l_processed <= upper_bound)
                
                original_points = len(t_processed)
                t_processed = t_processed[outlier_mask]
                l_processed = l_processed[outlier_mask]
                filtered_points = len(t_processed)
                if self.debug_mode:
                    logger.debug(f"   - Outlier rejection filtered {original_points - filtered_points} points.")
        # --- END: OUTLIER REJECTION ---

        self.t_fit = t_processed
        self.l_fit = l_processed

    def _get_initial_guess(self) -> Tuple[float, float, float]:
        """
        Estimates starting parameters (E, eta1, eta2) from the curve shape.
        Good initial guesses are critical for the optimizer to converge.
        """
        if self.initial_guess is None:
            # Get bounds and defaults from the params dictionary
            clip_bounds = self.params.get('clipping_bounds_guess', {
                "E": (500.0, 25000.0), "eta1": (1000.0, 50000.0), "eta2": (5000.0, 100000.0)
            })
            default_guess = tuple(self.params.get('default_initial_guess', (3000.0, 5000.0, 15000.0)))
            
            if not self.t_fit is None and not self.l_fit is None:
                # Delegate to Calculation module
                self.initial_guess = calc.estimate_initial_parameters(
                    self.t_fit, self.l_fit, self.r_eff, self.delta_p, self.C,
                    clip_bounds=clip_bounds,
                    default_guess=default_guess
                )
            else:
                logger.warning("t_fit or l_fit is not set. Using default guess.")
                self.initial_guess = default_guess
                
        return self.initial_guess

    def _get_models_to_fit(self) -> Dict[str, Dict[str, Any]]:
        """
        Defines the suite of models to be fitted.
        Each entry contains the function, bounds, and initial guesses.
        """
        if self.models_cache is not None:
            return self.models_cache

        e_guess, eta1_guess, eta2_guess = self._get_initial_guess()
        
        # Define fitting bounds from params
        fit_bounds_config = self.params.get('fitting_bounds', {
            "E": (100.0, 100000.0), "eta1": (100.0, 500000.0), "eta2": (500.0, 1000000.0)
        })
        e_bounds = fit_bounds_config['E']
        eta1_bounds = fit_bounds_config['eta1']
        eta2_bounds = fit_bounds_config['eta2']
        
        # Dictionary mapping Model Name -> Configuration
        models = {
            "Jeffreys": {
                # 3-Parameter Standard Viscoelastic Liquid (Spring + Dashpot // Dashpot)
                "func": lambda t, E, eta1, eta2: calc.jeffreys_length(t, self.r_eff, self.delta_p, self.C, E, eta1, eta2),
                "p0": [e_guess, eta1_guess, eta2_guess],
                "bounds": ([e_bounds[0], eta1_bounds[0], eta2_bounds[0]], 
                           [e_bounds[1], eta1_bounds[1], eta2_bounds[1]]),
                "param_names": ['E', 'eta1', 'eta2'],
                "n_params": 3,
                "maxfev_key": 'jeffreys_maxfev'
            },
            "Burgers": {
                # 4-Parameter Model (Maxwell + Kelvin-Voigt)
                "func": lambda t, E1, eta1, E2, eta2: calc.burgers_length(t, self.r_eff, self.delta_p, self.C, E1, eta1, E2, eta2),
                # Note: p0[1] maps 'eta2_guess' (Slope/Flow) to Burgers 'eta1' (Maxwell/Flow)
                "p0": [e_guess, eta2_guess, e_guess, eta1_guess],
                "bounds": ([e_bounds[0], eta2_bounds[0], e_bounds[0], eta1_bounds[0]],
                           [e_bounds[1], eta2_bounds[1], e_bounds[1], eta1_bounds[1]]),
                "param_names": ['E1', 'eta1', 'E2', 'eta2'],
                "n_params": 4,
                "maxfev_key": 'jeffreys_maxfev' # Uses the same (higher) count
            },
            "Kelvin-Voigt": {
                # Viscoelastic Solid (Spring // Dashpot) - No permanent flow
                "func": lambda t, E, eta: calc.kelvin_voigt_length(t, self.r_eff, self.delta_p, self.C, E, eta),
                "p0": [e_guess, eta1_guess],
                "bounds": ([e_bounds[0], eta1_bounds[0]], 
                           [e_bounds[1], eta1_bounds[1]]),
                "param_names": ['E', 'eta'],
                "n_params": 2,
                "maxfev_key": 'jeffreys_maxfev'
            },
            "Linear": {
                "func": calc.linear_model, "p0": [1, self.l_fit[0]], "bounds": ([-np.inf, -np.inf], [np.inf, np.inf]),
                "param_names": ['m', 'b'],
                "n_params": 2,
                "maxfev_key": 'empirical_maxfev'
            },
            "Power Law": {
                # Empirical model: L = a * t^b
                "func": calc.power_law_model, "p0": [1, 0.5], "bounds": ([0, 0], [np.inf, 2]),
                "param_names": ['a', 'b'],
                "n_params": 2,
                "maxfev_key": 'empirical_maxfev'
            },
            "Power Law + C": {
                "func": calc.power_law_model_with_c, "p0": [1, 0.5, self.l_fit[0]], "bounds": ([0, 0, 0], [np.inf, 2, np.inf]),
                "param_names": ['a', 'b', 'c'],
                "n_params": 3,
                "maxfev_key": 'empirical_maxfev'
            }
        }
        self.models_cache = models
        return self.models_cache


    def _fit_with_multi_start(self, func, t, l, bounds, maxfev=2000, n_starts=7):
        """
        Multi-Start Levenberg-Marquardt.
    
        Uses log-uniform sampling when all lower bounds are strictly positive,
        because viscoelastic parameters (E, eta1, eta2) span multiple orders of
        magnitude. 
    
        Falls back to plain uniform sampling for models whose bounds include zero
        or negative values (e.g., the Linear model).
        """
        best_r2 = -np.inf
        best_p = None
        lower, upper = bounds
    
        # Sanitize bounds for the random generator to prevent np.inf crashes
        safe_lower = np.clip(lower, -1e6, 1e6)
        safe_upper = np.clip(upper, -1e6, 1e6)
    
        # Decide sampling strategy once, before the loop
        use_log_uniform = np.all(np.array(safe_lower) > 0)
    
        for _ in range(n_starts):
            if use_log_uniform:
                # Equal probability per decade — much better exploration of
                # wide positive ranges like [100, 500_000]
                log_lo = np.log(np.array(safe_lower, dtype=float))
                log_hi = np.log(np.array(safe_upper, dtype=float))
                guess = np.exp(np.random.uniform(log_lo, log_hi))
            else:
                # Fallback: plain uniform (required when bounds span zero)
                guess = np.random.uniform(safe_lower, safe_upper)
            try:
                popt, _ = curve_fit(func, t, l, p0=guess, bounds=bounds, maxfev=maxfev)
                r2 = self._calculate_r_squared(l, func(t, *popt))
                if r2 > best_r2:
                    best_r2, best_p = r2, popt
            except Exception:
                continue
    
        return best_p, best_r2

    def _perform_multi_model_fitting(self) -> None:
        """Iterates through models, performs fitting, and stores results."""
        if self.t_fit is None or self.l_fit is None or len(self.t_fit) < 10:
            raise RuntimeError("Insufficient data.")

        models_to_fit = self._get_models_to_fit()
        use_global = self.params.get('use_global_optimization', False)      
        
        for name, model_info in models_to_fit.items():
            try:
                n_params = model_info["n_params"]
                maxfev_val = self.params.get('fitting_parameters', {}).get(model_info["maxfev_key"], 2000)
                # Decide between the multi-start global approach or single local fit
                if use_global:
                    p_opt, r2 = self._fit_with_multi_start(
                        model_info["func"], self.t_fit, self.l_fit, model_info["bounds"], maxfev=maxfev_val
                    )
                    if p_opt is None:
                        raise ValueError(f"Multi-start optimizer failed to converge for {name}.")
                else:
                    # Fallback to standard local fit using the initial heuristic guess
                    p_opt, _ = curve_fit(
                        model_info["func"], self.t_fit, self.l_fit,
                        p0=model_info["p0"], bounds=model_info["bounds"], 
                        maxfev=50000
                    )
                    r2 = self._calculate_r_squared(self.l_fit, model_info["func"](self.t_fit, *p_opt))
                    
                l_pred = model_info["func"](self.t_fit, *p_opt)
                aic = self._calculate_aic(self.l_fit, l_pred, n_params)

                self.fit_results[name] = {
                    "params": p_opt,
                    "r_squared": r2,
                    "aic": aic,
                    "n_params": n_params,
                    "param_names": model_info["param_names"]
                }
            except Exception as e:
                logger.warning(f"   - Could not fit {name} model: {e}")
                self.fit_results[name] = {
                    "params": None, "r_squared": -np.inf, "aic": np.inf,
                    "n_params": model_info["n_params"], "param_names": model_info["param_names"]
                }
                
    def _calculate_r_squared(self, l_true: np.ndarray, l_pred: np.ndarray) -> float:
        """Calculates the coefficient of determination (R^2)."""
        ss_res = np.sum((l_true - l_pred) ** 2)
        ss_tot = np.sum((l_true - np.mean(l_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    def _calculate_aic(self, l_true: np.ndarray, l_pred: np.ndarray, n_params: int) -> float:
        """
        Calculates the Akaike Information Criterion (AIC).

        AIC penalizes models for each additional free parameter, preventing the
        optimizer from always preferring Burgers (4 params) over Jeffreys (3 params)
        simply because it has more degrees of freedom.

        Lower AIC = better model. The model with the lowest AIC is preferred.

        Formula:  AIC = n * ln(RSS / n) + 2 * k
            n      : number of data points
            RSS    : residual sum of squares (how badly the model misses)
            k      : number of free parameters

        The 2*k term is the penalty — every extra parameter costs 2 AIC units.
        Burgers only wins over Jeffreys if it reduces RSS enough to justify that cost.
        """
        n = len(l_true)
        if n == 0:
            return np.inf
        rss = np.sum((l_true - l_pred) ** 2)
        # Guard against perfect fit (RSS=0) causing log(0)
        rss = max(rss, 1e-12)
        return n * np.log(rss / n) + 2 * n_params

    def _determine_best_fit(self) -> None:
        """
        Selects the best model using AIC (Akaike Information Criterion).

        AIC is preferred over R² because R² always improves with more parameters,
        which would systematically favour Burgers (4 params) over Jeffreys (3 params)
        even when the extra parameters are fitting noise rather than biology.
        AIC penalises each additional parameter by 2 units, so Burgers only wins
        if its residual reduction genuinely justifies the cost.

        R² is still stored in fit_results for reporting in plots and CSVs.
        """
        if not self.fit_results:
            return

        best_model = None
        min_aic = np.inf

        for name, result in self.fit_results.items():
            if result['params'] is not None and result['aic'] < min_aic:
                min_aic = result['aic']
                best_model = name

        self.best_fit_model_name = best_model
        if best_model:
            self.best_fit_params = self.fit_results[best_model]

    def predict(self, model_name: str, times: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Predicts protrusion length using the fitted parameters of a specific model."""
        if model_name not in self.fit_results or self.fit_results[model_name]["params"] is None:
            return None
            
        model_info = self._get_models_to_fit()[model_name]
        params = self.fit_results[model_name]["params"]
        t_pred = times if times is not None else self.t_raw
        
        return model_info["func"](t_pred, *params)

    def get_best_fit_summary(self) -> Optional[Dict[str, Any]]:
        """Returns a comprehensive dictionary of the best fitting model results."""
        if not self.best_fit_model_name or not self.best_fit_params:
            return None
            
        if self.t_fit is None:
            logger.warning("Cannot get summary, t_fit is not set.")
            return None

        summary = {
            'best_model_name': self.best_fit_model_name,
            'r_squared': self.best_fit_params['r_squared'],
            'aic': self.best_fit_params['aic'],
            'phase': self.phase,
            'fit_successful': True,
            'num_fitting_points': len(self.t_fit),
            'parameters': dict(zip(self.best_fit_params['param_names'], self.best_fit_params['params'])),
            'rupture_detected': self.rupture_detected,
            'rupture_time_s': self.rupture_time,
        }
        return summary
        
    def _print_fitting_summary(self) -> None:
        """Prints a summary of the best fit to the console."""
        summary = self.get_best_fit_summary()
        if not summary:
            logger.warning("\nFITTING FAILED! No summary to print.")
            return
            
        logger.debug("\n" + "="*80 + "\nBEST FIT SUMMARY\n" + "="*80)
        logger.debug(f"  - Best Model: {summary['best_model_name']}")
        logger.debug(f"  - RÂ²: {summary['r_squared']:.4f}")
        logger.debug("  - Parameters:")
        for name, val in summary['parameters'].items():
            logger.debug(f"    - {name}: {val:.3e}")