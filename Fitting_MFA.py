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

import Calculation_MFA as calc

logger = logging.getLogger(__name__)


class FittingMFA:
    """
    Manages the multi-model fitting process for a single dataset (one trap).
    """

    def __init__(self, t_data: np.ndarray, l_data: np.ndarray, r_eff: float, delta_p: float, C: float = 1.0) -> None:
        """
        Initializes the fitter with experimental data and physical constants.
        
        Args:
            t_data: Time points [seconds]
            l_data: Protrusion lengths [microns]
            r_eff: Effective radius of the channel [microns] (Geometric factor)
            delta_p: Applied pressure [Pascals]
            C: Geometric correction factor (usually ~1.0 or incorporated into r_eff)
        """
        self.t_raw = np.asarray(t_data, dtype=float)
        self.l_raw = np.asarray(l_data, dtype=float)
        self.r_eff = r_eff
        self.delta_p = delta_p
        self.C = C
        
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

        # Rupture detection results
        self.rupture_detected: bool = False
        self.rupture_time: Optional[float] = None
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
        
        # Minimum data points required for a statistically valid fit
        if len(self.t_raw) < 15:
            raise ValueError(f"Insufficient data points ({len(self.t_raw)}). Need at least 15.")

        # Filter out 0 or negative values (pre-entry phase)
        valid_mask = np.isfinite(self.t_raw) & np.isfinite(self.l_raw) & (self.l_raw > 0)
        
        sorted_indices = np.argsort(self.t_raw[valid_mask])
        t_processed = self.t_raw[valid_mask][sorted_indices]
        l_processed = self.l_raw[valid_mask][sorted_indices]
        
        if len(t_processed) < 10:
             raise ValueError(f"Insufficient non-zero data points ({len(t_processed)}) for fitting.")

        # Remove the very beginning of entry if requested (often unstable)
        n_exclude = int(len(t_processed) * params.get('exclude_early_fraction', 0.01))
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
                def _std_func(values):
                    return np.std(values)
                rolling_std = generic_filter(l_processed, _std_func, size=window_size, mode='nearest')

                # 3. Define the allowable range (Threshold)
                lower_bound = rolling_med - (std_dev_threshold * rolling_std)
                
                # 4. Filter: Keep only points *above* this lower bound (removes downward spikes)
                outlier_mask = l_processed >= lower_bound
                
                original_points = len(t_processed)
                t_processed = t_processed[outlier_mask]
                l_processed = l_processed[outlier_mask]
                filtered_points = len(t_processed)
                if self.debug_mode:
                    logger.debug(f"   - Outlier rejection filtered {original_points - filtered_points} points.")
        # --- END: OUTLIER REJECTION ---
        
        # Redundant Rupture Check (Fallback if LineDetection missed it)
        if params.get('enable_rupture_detection', True):
            rupture_idx = self._detect_rupture(t_processed, l_processed, params)
            if rupture_idx is not None:
                self.rupture_detected = True
                self.rupture_index = rupture_idx
                self.rupture_time = t_processed[rupture_idx]
                t_processed, l_processed = t_processed[:rupture_idx], l_processed[:rupture_idx]

        self.t_fit = t_processed
        self.l_fit = l_processed

    def _detect_rupture(self, t: np.ndarray, l: np.ndarray, params: Dict[str, Any]) -> Optional[int]:
        """Identifies a membrane rupture by detecting a sharp, unphysical DROP in protrusion length."""
        window_size = params.get('rupture_window_size', 15)
        if len(l) < window_size + 5: return None

        drop_thresh = params.get('rupture_drop_threshold', 0.4)
        abs_thresh = params.get('rupture_absolute_threshold', 3.0)
        
        # Look for drops relative to the recent maximum
        rolling_max = np.array([np.max(l[max(0, i - window_size):i]) for i in range(1, len(l) + 1)])
        absolute_drops = rolling_max - l
        relative_drops = absolute_drops / (rolling_max + 1e-9)

        rupture_mask = (relative_drops >= drop_thresh) & (absolute_drops >= abs_thresh)
        rupture_candidates = np.where(rupture_mask)[0]

        return rupture_candidates[0] if len(rupture_candidates) > 0 else None

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
                "maxfev_key": 'jeffreys_maxfev' # Uses the same (higher) count
            },
            "Kelvin-Voigt": {
                # Viscoelastic Solid (Spring // Dashpot) - No permanent flow
                "func": lambda t, E, eta: calc.kelvin_voigt_length(t, self.r_eff, self.delta_p, self.C, E, eta),
                "p0": [e_guess, eta1_guess],
                "bounds": ([e_bounds[0], eta1_bounds[0]], 
                           [e_bounds[1], eta1_bounds[1]]),
                "param_names": ['E', 'eta'],
                "maxfev_key": 'jeffreys_maxfev'
            },
            "Linear": {
                "func": calc.linear_model, "p0": [1, self.l_fit[0]], "bounds": ([-np.inf, -np.inf], [np.inf, np.inf]),
                "param_names": ['m', 'b'],
                "maxfev_key": 'empirical_maxfev'
            },
            "Power Law": {
                # Empirical model: L = a * t^b
                "func": calc.power_law_model, "p0": [1, 0.5], "bounds": ([0, 0], [np.inf, 2]),
                "param_names": ['a', 'b'],
                "maxfev_key": 'empirical_maxfev'
            },
            "Power Law + C": {
                "func": calc.power_law_model_with_c, "p0": [1, 0.5, self.l_fit[0]], "bounds": ([0, 0, 0], [np.inf, 2, np.inf]),
                "param_names": ['a', 'b', 'c'],
                "maxfev_key": 'empirical_maxfev'
            }
        }
        self.models_cache = models
        return self.models_cache


    def _perform_multi_model_fitting(self) -> None:
        """Iterates through models, performs fitting (Local or Global), and stores results."""
        if self.t_fit is None or self.l_fit is None or len(self.t_fit) < 10:
            raise RuntimeError(f"Insufficient data.")

        models_to_fit = self._get_models_to_fit() # Note: You need to implement _get_models_to_fit as per original file
        
        # Strategy selection: Global vs Local optimization
        use_global = self.params.get('use_global_optimization', False)
        
        jeffreys_maxfev = self.params.get('jeffreys_maxfev', 50000)
        empirical_maxfev = self.params.get('empirical_maxfev', 5000)

        for name, model_info in models_to_fit.items():
            try:
                p_opt = None
                
                # STRATEGY A: GLOBAL OPTIMIZATION (Differential Evolution)
                # Slower, but less likely to get stuck in local minima.
                if use_global:
                    # Define cost function: Sum of Squared Residuals
                    def cost_func(params):
                        y_pred = model_info["func"](self.t_fit, *params)
                        return np.sum((self.l_fit - y_pred) ** 2)
                    
                    # Convert bounds format for DE: [(min, max), (min, max)...]
                    # Original bounds are ([mins], [maxs])
                    de_bounds = list(zip(model_info["bounds"][0], model_info["bounds"][1]))
                    
                    # Fix infinite bounds for DE (replace with large numbers)
                    de_bounds = [(max(-1e9, b[0]), min(1e9, b[1])) for b in de_bounds]
                    
                    res = differential_evolution(cost_func, de_bounds, maxiter=1000, polish=True)
                    if res.success:
                        p_opt = res.x
                    else:
                        raise RuntimeError(f"DE failed: {res.message}")

                # STRATEGY B: LOCAL OPTIMIZATION (Levenberg-Marquardt)
                # Standard curve fitting. Fast, but needs good initial guess.
                else:
                    maxfev = jeffreys_maxfev if 'jeffreys' in model_info['maxfev_key'] else empirical_maxfev
                    p_opt, p_cov = curve_fit(
                        model_info["func"], self.t_fit, self.l_fit,
                        p0=model_info["p0"], bounds=model_info["bounds"], 
                        maxfev=maxfev
                    )

                # Calculate Goodness of Fit (R-squared)
                l_pred = model_info["func"](self.t_fit, *p_opt)
                r_squared = self._calculate_r_squared(self.l_fit, l_pred)
                
                self.fit_results[name] = {
                    "params": p_opt,
                    "r_squared": r_squared,
                    "param_names": model_info["param_names"]
                }
            except Exception as e:
                logger.warning(f"   - Could not fit {name} model: {e}")
                self.fit_results[name] = {
                    "params": None, "r_squared": -np.inf, "param_names": model_info["param_names"]
                }
                
    def _calculate_r_squared(self, l_true: np.ndarray, l_pred: np.ndarray) -> float:
        """Calculates the coefficient of determination (R^2)."""
        ss_res = np.sum((l_true - l_pred) ** 2)
        ss_tot = np.sum((l_true - np.mean(l_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    def _determine_best_fit(self) -> None:
        """Determines the winner based on the highest R-squared value."""
        if not self.fit_results:
            return

        best_model = None
        max_r2 = -np.inf
        
        for name, result in self.fit_results.items():
            if result['r_squared'] > max_r2:
                max_r2 = result['r_squared']
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
        logger.debug(f"  - R²: {summary['r_squared']:.4f}")
        logger.debug("  - Parameters:")
        for name, val in summary['parameters'].items():
            logger.debug(f"    - {name}: {val:.3e}")