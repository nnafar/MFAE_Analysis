# -*- coding: utf-8 -*-
"""
Enhanced Standalone Bulk MFA Analysis Script with Selective Plotting
- Processes multiple experiment folders with trap timeseries data
- Generates individual trap plots for protrusion and dye uptake
- User-friendly selection of experiments and traps to analyze
- Splits analysis at user-specified timepoints (before/after)
- NEW: Generates ensemble plot for DYE UPTAKE (Normalized dF/F₀)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import re
import warnings
warnings.filterwarnings('ignore')
# =============================================================================
# PHYSICAL CONSTANTS AND MODEL PARAMETERS
# =============================================================================
E_PLAUSIBLE_RANGE_PA = (100, 50000)
ETA1_PLAUSIBLE_RANGE_PA_S = (500, 100000)
ETA2_PLAUSIBLE_RANGE_PA_S = (1000, 200000)
R_SQUARED_THRESHOLD = 0.9
UNITS = {
    'E': r'$\text{Pa}$',
    'E1': r'$\text{Pa}$',
    'E2': r'$\text{Pa}$',
    'eta': r'$\text{Pa}\cdot\text{s}$',
    'eta1': r'$\text{Pa}\cdot\text{s}$',
    'eta2': r'$\text{Pa}\cdot\text{s}$',
    'm': r'$\mu\text{m/s}$',
    'b': r'$\mu\text{m}$',
    'a': r'$\mu\text{m/s}^{\text{b}}$',
    'c': r'$\mu\text{m}$'
}
# =============================================================================
# CORE MATHEMATICAL MODELS
# =============================================================================
def jeffreys_model(t, r_eff, delta_p, C, E, eta1, eta2):
    """3-parameter Jeffreys viscoelastic model."""
    t = np.asarray(t, dtype=float)
    tau = (3 * np.pi * eta1) / (C * E)
    elastic_scale = (r_eff * delta_p) / (C * E)
    elastic_term = elastic_scale * (1 - np.exp(-t / tau))
    viscous_term = (r_eff * delta_p) / (3 * np.pi * eta2) * t
    return elastic_term + viscous_term

def kelvin_voigt_model(t, r_eff, delta_p, C, E, eta):
    """Kelvin-Voigt viscoelastic solid model."""
    t = np.asarray(t, dtype=float)
    tau = (3 * np.pi * eta) / (C * E)
    elastic_scale = (r_eff * delta_p) / (C * E)
    return elastic_scale * (1 - np.exp(-t / tau))

def burgers_model(t, r_eff, delta_p, C, E1, eta1, E2, eta2):
    """4-parameter Burgers model."""
    t = np.asarray(t, dtype=float)
    maxwell_term = (r_eff * delta_p / C) * (1 / E1 + t / (3 * np.pi * eta1))
    tau_kv = (3 * np.pi * eta2) / (C * E2)
    kelvin_voigt_term = (r_eff * delta_p / (C * E2)) * (1 - np.exp(-t / tau_kv))
    return maxwell_term + kelvin_voigt_term

def linear_model(t, m, b):
    """Linear model: y = m*t + b"""
    return m * t + b

def power_law_model(t, a, b):
    """Power law: y = a * t^b"""
    return a * (t + 1e-9)**b

def power_law_with_offset(t, a, b, c):
    """Power law with offset: y = a * t^b + c"""
    return a * (t + 1e-9)**b + c

def exponential_model(t, a, b):
    """Exponential: y = a * (1 - exp(-b*t))"""
    return a * (1 - np.exp(-b * t))

# =============================================================================
# PROTRUSION START DETECTION
# =============================================================================
def detect_protrusion_start(time_data, length_data, zero_threshold=0.1):
    """ Detect when protrusion actually starts (t_0). """
    for i in range(len(length_data) - 1):
        if length_data[i] > zero_threshold and length_data[i + 1] > zero_threshold:
            return i, time_data[i]
    return 0, time_data[0]

def shift_time_to_start(time_data, length_data, zero_threshold=0.1):
    """ Shift time axis so that t=0 corresponds to when protrusion actually starts. """
    start_idx, start_time = detect_protrusion_start(time_data, length_data, zero_threshold)
    shifted_time = time_data[start_idx:] - start_time
    shifted_length = length_data[start_idx:]
    return shifted_time, shifted_length, start_idx

# =============================================================================
# RUPTURE DETECTION
# =============================================================================
def detect_rupture(time_data, length_data, sensitivity=2.0):
    """Detect rupture point in protrusion data."""
    if len(length_data) < 10:
        return None, None
    try:
        window_length = min(9, len(length_data) if len(length_data) % 2 == 1 else len(length_data) - 1)
        if window_length < 5:
            window_length = 5
        smoothed = savgol_filter(length_data, window_length, 3)
        dt = np.diff(time_data)
        dl = np.diff(smoothed)
        velocity = dl / (dt + 1e-10)
        vel_std = np.std(velocity)
        vel_mean = np.mean(velocity)
        for i in range(len(velocity) - 1):
            if velocity[i] < (vel_mean - sensitivity * vel_std):
                return i + 1, time_data[i + 1]
        return None, None
    except:
        return None, None

# =============================================================================
# ENSEMBLE AVERAGING FUNCTIONS
# =============================================================================
def calculate_ensemble_average(all_raw_data: List[pd.DataFrame]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates ensemble average by interpolating onto a common time grid."""
    if not all_raw_data:
        return None
    
    # Use the max time from all datasets to define the common grid
    max_time = max(df['Time_s'].max() for df in all_raw_data)
    t_common = np.linspace(0, max_time, 1000)
    
    interpolated_lengths = []
    for df in all_raw_data:
        t_data = df['Time_s'].values
        l_data = df['Protrusion_Length_um'].values
        
        # Ensure only unique times are used for interpolation setup
        unique_times, unique_indices = np.unique(t_data, return_index=True)
        unique_lengths = l_data[unique_indices]
        
        if len(unique_times) < 2:
            continue
            
        # Interpolate the length data onto the common time grid
        f_interp = interp1d(unique_times, unique_lengths, kind='linear', 
                            bounds_error=False, 
                            fill_value=(unique_lengths[0], unique_lengths[-1]))
        l_interp = f_interp(t_common)
        interpolated_lengths.append(l_interp)

    if not interpolated_lengths:
        return None
        
    L_array = np.array(interpolated_lengths)
    L_mean = np.mean(L_array, axis=0)
    L_std = np.std(L_array, axis=0)
    
    return t_common, L_mean, L_std

def calculate_dye_ensemble_average(all_dye_data: List[pd.DataFrame]) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Calculates ensemble average for specific normalized dye columns."""
    if not all_dye_data:
        return None
    
    max_time = max(df['Time_s'].max() for df in all_dye_data) if all_dye_data else 0
    t_common = np.linspace(0, max_time, 500) # Common time grid for interpolation
    
    columns_to_average = ['Total_Normalized_dF_F0', 'Protrusion_Normalized_dF_F0', 'Body_Normalized_dF_F0']
    results = {}
    
    for column in columns_to_average:
        interpolated_values = []
        for df in all_dye_data:
            if column in df.columns:
                t_data = df['Time_s'].values
                y_data = df[column].values
                
                # Filter out NaNs/Infs and ensure at least 2 unique points for interpolation
                valid_mask = np.isfinite(t_data) & np.isfinite(y_data)
                t_data, y_data = t_data[valid_mask], y_data[valid_mask]
                
                if len(t_data) >= 2:
                    f_interp = interp1d(t_data, y_data, kind='linear', bounds_error=False, fill_value=np.nan)
                    y_interp = f_interp(t_common)
                    interpolated_values.append(y_interp)

        if interpolated_values:
            L_array = np.array([arr for arr in interpolated_values if np.any(np.isfinite(arr))])
            
            # Use nanmean and nanstd to handle missing data from different max lengths
            L_mean = np.nanmean(L_array, axis=0)
            L_std = np.nanstd(L_array, axis=0)
            
            # Only keep points where we have real data (not all NaNs)
            finite_mask = np.isfinite(L_mean)
            if np.any(finite_mask):
                results[column] = (t_common[finite_mask], L_mean[finite_mask], L_std[finite_mask])
            
    return results if results else None


# =============================================================================
# FITTING ENGINE
# =============================================================================
class ModelFitter:
    """Handles fitting of multiple models to protrusion data."""
    def __init__(self, time_data, length_data, pressure_pa, channel_width_m, pore_width_m, is_bulk=False):
        self.t_raw = np.asarray(time_data, dtype=float)
        self.l_raw = np.asarray(length_data, dtype=float)
        self.pressure_pa = pressure_pa
        self.delta_p = pressure_pa
        self.channel_width_m = channel_width_m
        self.pore_width_m = pore_width_m
        self.is_bulk = is_bulk
        self.r_eff = (channel_width_m * 1e6)
        self.C = 1.0
        
        if not is_bulk:
            self.rupture_index, self.rupture_time = detect_rupture(self.t_raw, self.l_raw)
        else:
            self.rupture_index, self.rupture_time = None, None
            
        if self.rupture_index is not None:
            self.t_fit = self.t_raw[:self.rupture_index]
            self.l_fit = self.l_raw[:self.rupture_index]
        else:
            self.t_fit = self.t_raw
            self.l_fit = self.l_raw
            
        self.fit_results = {}
        self.best_model = None
        self.best_r_squared = -np.inf

    def calculate_r_squared(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def fit_jeffreys(self):
        try:
            max_l = np.max(self.l_fit)
            E_guess = (self.r_eff * self.delta_p) / (self.C * max_l) if max_l > 0 else 3000
            eta1_guess = 5000
            eta2_guess = 15000
            
            def model_func(t, E, eta1, eta2):
                return jeffreys_model(t, self.r_eff, self.delta_p, self.C, E, eta1, eta2)
            
            bounds = ([100, 500, 1000], [50000, 100000, 200000])
            popt, pcov = curve_fit(model_func, self.t_fit, self.l_fit, p0=[E_guess, eta1_guess, eta2_guess], bounds=bounds, maxfev=5000)
            y_pred = model_func(self.t_fit, *popt)
            r2 = self.calculate_r_squared(self.l_fit, y_pred)
            
            self.fit_results['Jeffreys'] = {
                'params': {'E': popt[0], 'eta1': popt[1], 'eta2': popt[2]},
                'r_squared': r2,
                'success': True
            }
            if r2 > self.best_r_squared:
                self.best_r_squared = r2
                self.best_model = 'Jeffreys'
        except Exception:
            self.fit_results['Jeffreys'] = {'params': None, 'r_squared': 0, 'success': False}

    def fit_kelvin_voigt(self):
        try:
            max_l = np.max(self.l_fit)
            E_guess = (self.r_eff * self.delta_p) / (self.C * max_l) if max_l > 0 else 3000
            eta_guess = 10000
            
            def model_func(t, E, eta):
                return kelvin_voigt_model(t, self.r_eff, self.delta_p, self.C, E, eta)
            
            bounds = ([100, 500], [50000, 200000])
            popt, pcov = curve_fit(model_func, self.t_fit, self.l_fit, p0=[E_guess, eta_guess], bounds=bounds, maxfev=5000)
            y_pred = model_func(self.t_fit, *popt)
            r2 = self.calculate_r_squared(self.l_fit, y_pred)
            
            self.fit_results['Kelvin-Voigt'] = {
                'params': {'E': popt[0], 'eta': popt[1]},
                'r_squared': r2,
                'success': True
            }
            if r2 > self.best_r_squared:
                self.best_r_squared = r2
                self.best_model = 'Kelvin-Voigt'
        except Exception:
            self.fit_results['Kelvin-Voigt'] = {'params': None, 'r_squared': 0, 'success': False}

    def fit_burgers(self):
        try:
            def model_func(t, E1, eta1, E2, eta2):
                return burgers_model(t, self.r_eff, self.delta_p, self.C, E1, eta1, E2, eta2)
                
            bounds = ([100, 500, 100, 1000], [50000, 100000, 50000, 200000])
            popt, pcov = curve_fit(model_func, self.t_fit, self.l_fit, p0=[3000, 5000, 3000, 15000], bounds=bounds, maxfev=5000)
            y_pred = model_func(self.t_fit, *popt)
            r2 = self.calculate_r_squared(self.l_fit, y_pred)
            
            self.fit_results['Burgers'] = {
                'params': {'E1': popt[0], 'eta1': popt[1], 'E2': popt[2], 'eta2': popt[3]},
                'r_squared': r2,
                'success': True
            }
            if r2 > self.best_r_squared:
                self.best_r_squared = r2
                self.best_model = 'Burgers'
        except Exception:
            self.fit_results['Burgers'] = {'params': None, 'r_squared': 0, 'success': False}

    def fit_linear(self):
        try:
            popt, _ = curve_fit(linear_model, self.t_fit, self.l_fit, p0=[1, 0])
            y_pred = linear_model(self.t_fit, *popt)
            r2 = self.calculate_r_squared(self.l_fit, y_pred)
            
            self.fit_results['Linear'] = {
                'params': {'m': popt[0], 'b': popt[1]},
                'r_squared': r2,
                'success': True
            }
            if r2 > self.best_r_squared:
                self.best_r_squared = r2
                self.best_model = 'Linear'
        except Exception:
            self.fit_results['Linear'] = {'params': None, 'r_squared': 0, 'success': False}

    def fit_power_law(self):
        try:
            # We must shift or offset t to avoid log(0) issues, handled by t + 1e-9 in the model definition
            popt, _ = curve_fit(power_law_model, self.t_fit, self.l_fit, p0=[1, 0.5], bounds=([0, 0], [100, 2]))
            y_pred = power_law_model(self.t_fit, *popt)
            r2 = self.calculate_r_squared(self.l_fit, y_pred)
            
            self.fit_results['Power-Law'] = {
                'params': {'a': popt[0], 'b': popt[1]},
                'r_squared': r2,
                'success': True
            }
            if r2 > self.best_r_squared:
                self.best_r_squared = r2
                self.best_model = 'Power-Law'
        except Exception:
            self.fit_results['Power-Law'] = {'params': None, 'r_squared': 0, 'success': False}

    def fit_exponential(self):
        try:
            max_l = np.max(self.l_fit)
            # Use exponential model for approach to steady state (L = L_max * (1 - exp(-b*t)))
            popt, _ = curve_fit(exponential_model, self.t_fit, self.l_fit, p0=[max_l, 0.1], bounds=([0, 0], [max_l*2, 10]))
            y_pred = exponential_model(self.t_fit, *popt)
            r2 = self.calculate_r_squared(self.l_fit, y_pred)
            
            self.fit_results['Exponential'] = {
                'params': {'a': popt[0], 'b': popt[1]},
                'r_squared': r2,
                'success': True
            }
            if r2 > self.best_r_squared:
                self.best_r_squared = r2
                self.best_model = 'Exponential'
        except Exception:
            self.fit_results['Exponential'] = {'params': None, 'r_squared': 0, 'success': False}

    def fit_all_models(self):
        if not self.is_bulk:
            print(" Fitting models...", end=" ")
        self.fit_jeffreys()
        self.fit_kelvin_voigt()
        self.fit_burgers()
        self.fit_linear()
        self.fit_power_law()
        self.fit_exponential()
        
        if not self.is_bulk:
            if self.best_model:
                print(f"Best: {self.best_model} (R²={self.best_r_squared:.4f})")
            else:
                print("No models fitted successfully.")

    def predict(self, model_name, t):
        if model_name not in self.fit_results or self.fit_results[model_name]['params'] is None:
            return None
            
        params = self.fit_results[model_name]['params']
        
        if model_name == 'Jeffreys':
            return jeffreys_model(t, self.r_eff, self.delta_p, self.C, params['E'], params['eta1'], params['eta2'])
        elif model_name == 'Kelvin-Voigt':
            return kelvin_voigt_model(t, self.r_eff, self.delta_p, self.C, params['E'], params['eta'])
        elif model_name == 'Burgers':
            return burgers_model(t, self.r_eff, self.delta_p, self.C, params['E1'], params['eta1'], params['E2'], params['eta2'])
        elif model_name == 'Linear':
            return linear_model(t, params['m'], params['b'])
        elif model_name == 'Power-Law':
            return power_law_model(t, params['a'], params['b'])
        elif model_name == 'Exponential':
            return exponential_model(t, params['a'], params['b'])
            
        return None

    def calculate_tau(self, model_name):
        if model_name not in self.fit_results or self.fit_results[model_name]['params'] is None:
            return None
        
        params = self.fit_results[model_name]['params']
        try:
            if model_name == 'Jeffreys' and 'E' in params and 'eta1' in params and params['E'] > 0:
                return (3 * np.pi * params['eta1']) / (self.C * params['E'])
            elif model_name == 'Kelvin-Voigt' and 'E' in params and 'eta' in params and params['E'] > 0:
                return (3 * np.pi * params['eta']) / (self.C * params['E'])
            elif model_name == 'Burgers' and 'E2' in params and 'eta2' in params and params['E2'] > 0:
                return (3 * np.pi * params['eta2']) / (self.C * params['E2'])
        except:
            pass
        return None
# =============================================================================
# INDIVIDUAL TRAP PLOTTING FUNCTIONS
# =============================================================================
def plot_individual_protrusion(time_data, length_data, time_data_original, length_data_original, exp_id, trap_number, output_dir, pulse_time=None):
    """ Plots protrusion length in two formats: 1. Absolute protrusion length (shifted time) 2. Relative protrusion length (L - L0) """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{exp_id} - Trap {trap_number}: Protrusion Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Absolute protrusion length
    axes[0].plot(time_data, length_data, 'b-', linewidth=2, label='Protrusion Length')
    if pulse_time is not None:
        axes[0].axvline(x=pulse_time, color='cyan', linestyle='--', linewidth=2, label='Pulse')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel(r'Protrusion Length ($\mu\text{m}$)', fontsize=12)
    axes[0].set_title('Absolute Protrusion Length', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Relative protrusion length (L - L0)
    if len(length_data) > 0:
        L0 = length_data[0]
        relative_length = length_data - L0
        axes[1].plot(time_data, relative_length, 'r-', linewidth=2, label='L - L₀')
    if pulse_time is not None:
        axes[1].axvline(x=pulse_time, color='cyan', linestyle='--', linewidth=2, label='Pulse')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel(r'Relative Length ($\mu\text{m}$)', fontsize=12)
    axes[1].set_title('Relative Protrusion (L - L₀)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f'{exp_id}_Trap_{trap_number}_Protrusion.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved protrusion plot: {save_path.name}")

def plot_individual_dye_uptake(dye_data_path, exp_id, trap_number, output_dir, pulse_time=None):
    """ Plots three dye uptake graphs: 1. Absolute intensity 2. Min-Max normalized 3. Baseline normalized (ΔF/F₀) """
    try:
        df = pd.read_csv(dye_data_path, comment='#')
        if 'Time_s' not in df.columns:
            print(f"  Warning: No Time_s column in dye data for Trap {trap_number}")
            return
            
        time = df['Time_s'].values
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{exp_id} - Trap {trap_number}: Dye Uptake', fontsize=14, fontweight='bold')
        
        # Plot 1: Absolute intensity
        if 'Total_Mean' in df.columns:
            axes[0].plot(time, df['Total_Mean'], 'k-', linewidth=2, label='Total')
        if 'Protrusion_Mean' in df.columns:
            axes[0].plot(time, df['Protrusion_Mean'], 'r-', linewidth=2, label='Protrusion')
        if 'Body_Mean' in df.columns:
            axes[0].plot(time, df['Body_Mean'], 'lightblue', linewidth=2, label='Body')
        if pulse_time is not None:
            axes[0].axvline(x=pulse_time, color='cyan', linestyle='--', linewidth=2, label='Pulse')
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel('Intensity (a.u.)', fontsize=12)
        axes[0].set_title('Dye Uptake (Absolute)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Min-Max Normalized
        if 'Total_MinMax' in df.columns:
            axes[1].plot(time, df['Total_MinMax'], 'k-', linewidth=2, label='Total')
        if 'Protrusion_MinMax' in df.columns:
            axes[1].plot(time, df['Protrusion_MinMax'], 'r-', linewidth=2, label='Protrusion')
        if 'Body_MinMax' in df.columns:
            axes[1].plot(time, df['Body_MinMax'], 'lightblue', linewidth=2, label='Body')
        if pulse_time is not None:
            axes[1].axvline(x=pulse_time, color='cyan', linestyle='--', linewidth=2, label='Pulse')
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('Normalized Intensity (0-1)', fontsize=12)
        axes[1].set_title('Min-Max Normalized Uptake', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Baseline Normalized (ΔF/F₀)
        if 'Total_Normalized_dF_F0' in df.columns:
            axes[2].plot(time, df['Total_Normalized_dF_F0'], 'k-', linewidth=2, label='Total')
        if 'Protrusion_Normalized_dF_F0' in df.columns:
            axes[2].plot(time, df['Protrusion_Normalized_dF_F0'], 'r-', linewidth=2, label='Protrusion')
        if 'Body_Normalized_dF_F0' in df.columns:
            axes[2].plot(time, df['Body_Normalized_dF_F0'], 'lightblue', linewidth=2, label='Body')
        if pulse_time is not None:
            axes[2].axvline(x=pulse_time, color='cyan', linestyle='--', linewidth=2, label='Pulse')
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel(r'Normalized Fluorescence ($\Delta F/F_0$)', fontsize=12)
        axes[2].set_title('Normalized Uptake (Baseline)', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / f'{exp_id}_Trap_{trap_number}_DyeUptake.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved dye uptake plot: {save_path.name}")
        
    except Exception as e:
        print(f"  Error plotting dye uptake for Trap {trap_number}: {e}")

def run_bulk_fit_and_plot(all_raw_data: List[pd.DataFrame], analyzer: Any, output_dir: Path, n_trajectories: int, phase_label: str = "Full"):
    """Performs bulk analysis, fitting, and plotting on ensemble protrusion data."""
    print(f"\n--- Starting Bulk Ensemble Protrusion Analysis ({phase_label}) ---")
    
    # Constants related to analysis window (used only for plotting limits)
    MAX_ANALYSIS_TIME = 40.0 
    SPLIT_TIME_S = 19.0
    
    if phase_label == "Before_Split":
        START_TIME_S = 0.0
        END_TIME_S = SPLIT_TIME_S
    elif phase_label == "After_Split":
        START_TIME_S = SPLIT_TIME_S
        END_TIME_S = MAX_ANALYSIS_TIME
    else:
        START_TIME_S = 0.0
        END_TIME_S = MAX_ANALYSIS_TIME

    ensemble_data = calculate_ensemble_average(all_raw_data)
    if ensemble_data is None:
        print(f"[BULK-{phase_label}] Warning: No data available for ensemble average.")
        return None
        
    t_common, l_mean, l_std = ensemble_data
    
    # Filter common data to the current analysis window
    target_mask = (t_common >= START_TIME_S) & (t_common <= END_TIME_S)
    t_subset = t_common[target_mask]
    l_subset = l_mean[target_mask]
    l_std_subset = l_std[target_mask]

    if len(t_subset) == 0:
        print(f"[BULK-{phase_label}] Error: Masked data is empty for analysis window ({START_TIME_S}s to {END_TIME_S}s).")
        return None
        
    # Apply offset/shift for 'After_Split' analysis
    if phase_label == "After_Split":
        time_shift = t_subset[0]
        length_offset = l_subset[0]
        t_fit_data = t_subset - time_shift
        l_fit_data = l_subset - length_offset
        l_fit_data = np.clip(l_fit_data, a_min=0.0, a_max=None)
        t_plot_ens, l_mean_plot, l_std_plot = t_subset, l_subset, l_std_subset
        print(f"[BULK-{phase_label}] Fitting span: t={t_subset.min():.2f}s to {t_subset.max():.2f}s. (Length offset: {length_offset:.2f}μm)")
    else:
        time_shift = 0.0
        length_offset = 0.0
        t_fit_data = t_subset
        l_fit_data = l_subset
        t_plot_ens, l_mean_plot, l_std_plot = t_subset, l_subset, l_std_subset
        
    # Perform Bulk Fitting
    bulk_fitter = ModelFitter(t_fit_data, l_fit_data, analyzer.pressure_pa, analyzer.channel_width_m, analyzer.pore_width_m, is_bulk=True)
    bulk_fitter.fit_all_models()

    if bulk_fitter.best_model is None:
        print(f"[BULK-{phase_label}] Warning: Could not fit models to ensemble average.")
        return None

    best_model = bulk_fitter.best_model
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Bulk MFA Analysis ({phase_label}): {analyzer.experiment_prefix}', fontsize=16, fontweight='bold')
    
    # Plot individual grey lines (for visual reference)
    for df in all_raw_data:
        df_plot = df[(df['Time_s'] >= START_TIME_S) & (df['Time_s'] <= END_TIME_S)]
        ax.plot(df_plot['Time_s'], df_plot['Protrusion_Length_um'], color='gray', alpha=0.15, linewidth=1, zorder=1)
        
    # Plot Ensemble Average
    ax.plot(t_plot_ens, l_mean_plot, color='darkblue', linewidth=3, label=f'Ensemble Average (N={n_trajectories})', zorder=3)
    ax.fill_between(t_plot_ens, l_mean_plot - l_std_plot, l_mean_plot + l_std_plot, color='lightblue', alpha=0.3, label='Standard Deviation', zorder=2)
    
    # Plot Best Fit
    t_predict_shifted = np.linspace(t_fit_data.min(), t_fit_data.max(), 300)
    y_pred_shifted = bulk_fitter.predict(best_model, t_predict_shifted)

    if y_pred_shifted is not None:
        # Re-shift prediction back to original time/length domain for plotting
        t_plot = t_predict_shifted + time_shift
        y_plot = y_pred_shifted + length_offset
        ax.plot(t_plot, y_plot, '--', color='red', linewidth=2.5, label=f'Best Bulk Fit ({best_model})', zorder=4)
        
    # Annotate with Fit Parameters
    params = bulk_fitter.fit_results[best_model]['params']
    param_text = f"Best Bulk Fit: {best_model}\n"
    param_text += f"R² = {bulk_fitter.best_r_squared:.4f}\n"
    param_lines = []
    
    for name, val in params.items():
        unit = UNITS.get(name, '')
        param_lines.append(f"{name} = {val:.2e} {unit}")
    param_text += "\n".join(param_lines)

    tau = bulk_fitter.calculate_tau(best_model)
    if tau is not None:
        if best_model in ['Burgers', 'Kelvin-Voigt']:
            tau_label = r'$\tau_{\text{retardation}}$'
        else:
            tau_label = r'$\tau_{\text{relaxation}}$'
        param_text += f"\n{tau_label} = {tau:.2f} s"
        
    print(f"\n[BULK-{phase_label} RESULTS]")
    print(param_text)
    
    ax.text(0.98, 0.05, param_text, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Final Plot Formatting
    ax.set_xlabel(r'Time ($\text{s}$)', fontsize=12)
    ax.set_ylabel(r'Protrusion Length ($\mu\text{m}$)', fontsize=12)
    ax.set_title(f'Ensemble Protrusion ({phase_label}): {analyzer.experiment_prefix}', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(False)
    ax.set_xlim(left=START_TIME_S, right=END_TIME_S)

    bulk_plot_path = output_dir / f'Bulk_Ensemble_Fit_{phase_label}_{analyzer.experiment_prefix}.png'
    plt.savefig(bulk_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[BULK-{phase_label}] Successfully saved bulk plot to: {bulk_plot_path.name}")
    
    return bulk_fitter.fit_results[best_model]

def plot_dye_ensemble_average(ensemble_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                              output_dir: Path, analyzer: Any, n_trajectories: int):
    """Plots the ensemble average and standard deviation for baseline-normalized dye uptake."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fig.suptitle(f'Bulk Ensemble Dye Uptake ($\Delta F/F_0$): {analyzer.experiment_prefix} (N={n_trajectories})', 
                 fontsize=16, fontweight='bold')
    
    colors = {
        'Total_Normalized_dF_F0': ('black', 'gray', 'Total Uptake'),
        'Protrusion_Normalized_dF_F0': ('red', 'lightcoral', 'Protrusion Region'),
        'Body_Normalized_dF_F0': ('darkblue', 'lightblue', 'Body Region')
    }
    
    handles = []
    
    # Plotting each category
    for key, (color, fill_color, label) in colors.items():
        if key in ensemble_results:
            t, mean, std = ensemble_results[key]
            
            line, = ax.plot(t, mean, color=color, linewidth=2, label=label, zorder=3)
            ax.fill_between(t, mean - std, mean + std, color=fill_color, alpha=0.3, zorder=2)
            handles.append(line)
            
    # Add pulse time marker
    if analyzer.pulse_time is not None:
        # Plotting a visible line
        ax.axvline(x=analyzer.pulse_time, color='cyan', linestyle='--', linewidth=2, zorder=1)
        # Adding a handle specifically for the pulse marker in the legend
        handles.append(plt.Line2D([0], [0], color='cyan', linestyle='--', linewidth=2, label='Pulse Time'))
        
    ax.set_xlabel(r'Time ($\text{s}$)', fontsize=12)
    ax.set_ylabel(r'Normalized Fluorescence ($\Delta F/F_0$)', fontsize=12)
    ax.set_title(r'Ensemble Average Dye Uptake ($\Delta F/F_0$) Over Time', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    save_path = output_dir / f'Bulk_Ensemble_DyeUptake_{analyzer.experiment_prefix}.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[BULK-DYE] Successfully saved bulk dye uptake plot to: {save_path.name}")

def _common_trap_number(selected_traps: Optional[Dict[str, List[int]]]) -> Optional[int]:
    """
    Returns the trap number if *all* experiments in `selected_traps`
    contain **exactly the same single** trap ID.
    Otherwise returns None.
    """
    if not selected_traps:
        return None
    # Collect every trap list
    trap_lists = list(selected_traps.values())
    # All lists must have length 1
    if any(len(l) != 1 for l in trap_lists):
        return None
    # Extract the sole element from each list
    first = trap_lists[0][0]
    # Verify that every experiment uses the same number
    if all(l[0] == first for l in trap_lists):
        return first
    return None

# =============================================================================
# BULK ANALYZER WITH SELECTIVE PLOTTING
# =============================================================================
class BulkMFAAnalyzer:
    def __init__(self, base_folder: str, experiment_prefix: str = "exp", pressure_pa: float = 1000.0, channel_width_m: float = 100e-6, pore_width_m: float = 10e-6, split_timepoint: Optional[float] = None, selected_traps: Optional[Dict[str, List[int]]] = None, pulse_time: Optional[float] = None):
        self.base_folder = Path(base_folder)
        self.experiment_prefix = experiment_prefix
        self.pressure_pa = pressure_pa
        self.channel_width_m = channel_width_m
        self.pore_width_m = pore_width_m
        self.split_timepoint = split_timepoint
        self.selected_traps = selected_traps
        self.pulse_time = pulse_time
        self.protrusion_dir_name = "Full protrusion detection"
        self.dye_uptake_dir_name = "Dye Uptake"
        self.all_raw_data = [] # Protrusion data (Full Timeseries)
        self.all_raw_data_before = [] # Protrusion data (Before Split)
        self.all_raw_data_after = [] # Protrusion data (After Split)
        self.all_raw_dye_data = [] # NEW: Dye data (Full Timeseries)
        self.all_fit_summaries = []
        self.common_trap = _common_trap_number(selected_traps)   # ← new line


    def should_process_trap(self, exp_id: str, trap_number: int) -> bool:
        """Check if a trap should be processed based on selected_traps configuration."""
        if self.selected_traps is None:
            return True
        if exp_id in self.selected_traps:
            return trap_number in self.selected_traps[exp_id]
        return False

    def find_experiment_folders(self) -> List[Path]:
        folders = []
        for item in self.base_folder.iterdir():
            if item.is_dir() and item.name.startswith(self.experiment_prefix):
                folders.append(item)
        
        def sort_key(p):
            match = re.search(r'(\d+)', p.name)
            return int(match.group(1)) if match else 0
        
        folders.sort(key=sort_key)
        print(f"Found {len(folders)} experiment folders matching '{self.experiment_prefix}'")
        return folders

    def find_trap_files(self, experiment_folder: Path) -> List[Path]:
        protrusion_folder = experiment_folder / self.protrusion_dir_name
        if not protrusion_folder.is_dir():
            return []
            
        trap_files = []
        file_pattern = re.compile(r'trap_\d+', re.IGNORECASE)
        for ext in ['*.xlsx', '*.xls', '*.csv']:
            for file in protrusion_folder.glob(ext):
                file_name_lower = file.name.lower()
                if 'summary' in file_name_lower or 'fits' in file_name_lower:
                    continue
                if file_pattern.search(file_name_lower):
                    trap_files.append(file)
        
        def sort_key(p):
            match = re.search(r'trap_(\d+)', p.name)
            return int(match.group(1)) if match else 0
        
        trap_files.sort(key=sort_key)
        return trap_files

    def extract_trap_number(self, filename: str) -> int:
        match = re.search(r'trap_(\d+)', filename)
        return int(match.group(1)) if match else -1

    def load_dye_data_df(self, experiment_folder: Path, trap_number: int) -> Optional[pd.DataFrame]:
        """Loads the full dye uptake CSV file for ensemble averaging."""
        dye_folder = experiment_folder / self.dye_uptake_dir_name
        expected_filename = f"Trap_{trap_number:02d}_Uptake_Data.csv"
        file_path = dye_folder / expected_filename
        
        if not file_path.exists():
            dye_files = list(dye_folder.glob(f"Trap_{trap_number}*_Uptake_Data.*"))
            if not dye_files:
                return None
            file_path = dye_files[0]
            
        try:
            df = pd.read_csv(file_path, comment='#')
            
            # Ensure necessary columns are numeric and add metadata
            for col in ['Time_s', 'Total_Normalized_dF_F0', 'Protrusion_Normalized_dF_F0', 'Body_Normalized_dF_F0']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Trap_Number'] = trap_number
            df['Experiment_ID'] = experiment_folder.name
            return df
        except Exception:
            return None

    def load_and_summarize_dye_uptake(self, experiment_folder: Path, trap_number: int) -> Dict[str, Any]:
        """Loads processed dye uptake data (CSV) and returns key summary metrics."""
        dye_folder = experiment_folder / self.dye_uptake_dir_name
        expected_filename = f"Trap_{trap_number:02d}_Uptake_Data.csv"
        file_path = dye_folder / expected_filename
        
        if not file_path.is_file():
            dye_files = list(dye_folder.glob(f"Trap_{trap_number}*_Uptake_Data.*"))
            if not dye_files:
                return {'Max_dF_F0_Total': np.nan, 'Max_dF_F0_Protrusion': np.nan, 'Max_MinMax_Total': np.nan, 'Mean_MinMax_Std_Total': np.nan, 'Dye_Data_Available': False}
            file_path = dye_files[0]

        try:
            df = pd.read_csv(file_path, comment='#')
            
            max_dff0_total = df['Total_Normalized_dF_F0'].max() if 'Total_Normalized_dF_F0' in df.columns else np.nan
            max_dff0_prot = df['Protrusion_Normalized_dF_F0'].max() if 'Protrusion_Normalized_dF_F0' in df.columns else np.nan
            max_minmax_total = df['Total_MinMax'].max() if 'Total_MinMax' in df.columns else np.nan
            mean_minmax_std_total = df['Total_MinMax_Std'].mean() if 'Total_MinMax_Std' in df.columns else np.nan
            
            return {
                'Max_dF_F0_Total': max_dff0_total,
                'Max_dF_F0_Protrusion': max_dff0_prot,
                'Max_MinMax_Total': max_minmax_total,
                'Mean_MinMax_Std_Total': mean_minmax_std_total,
                'Dye_Data_Available': True
            }
        except Exception as e:
            # print(f"Warning: Failed to load/summarize dye uptake data for Trap {trap_number}. Error: {e}")
            return {'Max_dF_F0_Total': np.nan, 'Max_dF_F0_Protrusion': np.nan, 'Max_MinMax_Total': np.nan, 'Mean_MinMax_Std_Total': np.nan, 'Dye_Data_Available': False}

    def process_single_trap(self, file_path: Path, exp_folder: Path, trap_number: int, plots_folder: Path, generate_plots: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        exp_id = exp_folder.name
        print(f" Processing Exp: {exp_id}, Trap: {trap_number}...", end=" ")
        
        # --- Load Protrusion Data ---
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                print(f"FAILED: Unsupported file format {file_path.suffix}.")
                return None, None
                
            time_col = None
            length_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'time' in col_lower and time_col is None:
                    time_col = col
                if ('length' in col_lower or 'protrusion' in col_lower) and length_col is None:
                    length_col = col
                    
            if time_col is None or length_col is None:
                print(f"FAILED: Missing Time/Length data in columns.")
                return None, None
                
            time_data = df[time_col].values
            length_data = df[length_col].values
            
            time_data_original = time_data.copy()
            length_data_original = length_data.copy()
            
            shifted_time, shifted_length, start_idx = shift_time_to_start(time_data, length_data)
            
            if len(shifted_time) < 5:
                print(f"FAILED: Insufficient data after start detection.")
                return None, None

        except Exception as e:
            print(f"ERROR reading/processing protrusion data: {e}")
            return None, None
        
        # --- Dye Uptake Processing ---
        dye_data_df = self.load_dye_data_df(exp_folder, trap_number)
        dye_summary = self.load_and_summarize_dye_uptake(exp_folder, trap_number)

        # Plot individual dye uptake if required
        if generate_plots and dye_summary.get('Dye_Data_Available'):
            # Find the path robustly again to pass to the plotting function
            dye_folder = exp_folder / self.dye_uptake_dir_name
            dye_file_path = dye_folder / f"Trap_{trap_number:02d}_Uptake_Data.csv"
            if not dye_file_path.exists():
                dye_files = list(dye_folder.glob(f"Trap_{trap_number}*_Uptake_Data.*"))
                if dye_files:
                    dye_file_path = dye_files[0]

            if dye_file_path.exists():
                plot_individual_dye_uptake(dye_file_path, exp_id, trap_number, plots_folder, self.pulse_time)

        # Collect full dye data for ensemble averaging later
        if dye_data_df is not None and dye_summary.get('Dye_Data_Available'):
            self.all_raw_dye_data.append(dye_data_df)
            
        # --- Fitting Protrusion Data ---
        if generate_plots:
             plot_individual_protrusion(shifted_time, shifted_length, time_data_original, length_data_original, exp_id, trap_number, plots_folder, self.pulse_time)
             
        fitter = ModelFitter(shifted_time, shifted_length, self.pressure_pa, self.channel_width_m, self.pore_width_m)
        fitter.fit_all_models()
        
        if fitter.best_model is None:
            print("FAILED: No model fitted successfully.")
            return None, None
            
        if fitter.best_r_squared < R_SQUARED_THRESHOLD:
            print(f"REJECTED: R² ({fitter.best_r_squared:.4f}) below threshold.")
            return None, None
            
        # --- Aggregate Raw Data and Summary ---
        raw_data_slice = pd.DataFrame({
            'Experiment_ID': exp_id,
            'Trap_Number': trap_number,
            'Time_s': shifted_time,
            'Protrusion_Length_um': shifted_length
        })

        if self.split_timepoint is not None:
            split_idx = np.argmin(np.abs(shifted_time - self.split_timepoint))
            if split_idx > 0 and split_idx < len(shifted_time):
                # Ensure 'Before' ends before or at the split time
                raw_data_before = pd.DataFrame({
                    'Experiment_ID': exp_id,
                    'Trap_Number': trap_number,
                    'Time_s': shifted_time[:split_idx],
                    'Protrusion_Length_um': shifted_length[:split_idx]
                })
                # Ensure 'After' starts at the split time
                raw_data_after = pd.DataFrame({
                    'Experiment_ID': exp_id,
                    'Trap_Number': trap_number,
                    'Time_s': shifted_time[split_idx:],
                    'Protrusion_Length_um': shifted_length[split_idx:]
                })
                self.all_raw_data_before.append(raw_data_before)
                self.all_raw_data_after.append(raw_data_after)
                
        # Build summary dictionary
        summary_dict = {
            'Experiment_ID': exp_id,
            'Trap_Number': trap_number,
            'Start_Index': start_idx,
            'Original_Start_Time_s': time_data[start_idx],
            'Rupture_Detected': fitter.rupture_time is not None,
            'Rupture_Time_s': fitter.rupture_time if fitter.rupture_time is not None else np.nan,
            'Best_Model': fitter.best_model,
            'Best_R_Squared': fitter.best_r_squared,
        }
        
        best_result = fitter.fit_results[fitter.best_model]
        for name, val in best_result['params'].items():
            unit = UNITS.get(name, '')
            summary_dict[f'Param_{name}_{unit}'] = val
        
        tau = fitter.calculate_tau(fitter.best_model)
        if tau is not None:
            if fitter.best_model == 'Jeffreys':
                summary_dict['Tau_Relaxation_s'] = tau
            elif fitter.best_model == 'Kelvin-Voigt' or fitter.best_model == 'Burgers':
                summary_dict['Tau_Retardation_s'] = tau
                
        summary_dict.update(dye_summary)
        
        print(f"SUCCESS (R²={fitter.best_r_squared:.4f}", end="")
        if dye_summary.get('Dye_Data_Available'):
            print(f", Max dF/F0={dye_summary['Max_dF_F0_Total']:.2f})")
        else:
            print(")")
            
        return raw_data_slice, summary_dict

    def run_analysis(self):
        """Execute the bulk analysis pipeline with selective trap processing."""
        print("="*80)
        print("ENHANCED BULK MFA ANALYSIS WITH SELECTIVE PLOTTING")
        print("="*80)
        
        if self.split_timepoint is not None:
            print(f"Split timepoint: {self.split_timepoint}s (will analyze before and after)")

        if self.selected_traps is not None:
            print("\n SELECTED TRAPS FOR ANALYSIS:")
            for exp_id, traps in self.selected_traps.items():
                print(f" {exp_id}: Traps {traps}")
        else:
            print("\n Processing ALL available traps")

        # ---- ORIGINAL LINE ----
        # output_dir_name = f"MFA_Bulk_Results_{self.experiment_prefix}"
        
        # ---- NEW LOGIC ----
        base_name = f"MFA_Bulk_Results_{self.experiment_prefix}"
        # If we are analysing a *single identical* trap across experiments,
        # append the trap number so the folder is self‑describing.
        if self.common_trap is not None:
            output_dir_name = f"{base_name}_Trap_{self.common_trap}"
        else:
            output_dir_name = base_name

        # Store as instance variable so it can be accessed later
        self.output_dir = self.base_folder / output_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.output_dir / "Individual_Plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        print(f" Output directory: {self.output_dir}")          # existing line
        if self.common_trap is not None:
            print(f" (Folder includes common trap number: {self.common_trap})")

        print(f" Plots directory: {plots_dir}")

        experiment_folders = self.find_experiment_folders()
        if not experiment_folders:
            print("\n No matching experiment folders found. Exiting.")
            return

        for exp_folder in experiment_folders:
            exp_id = exp_folder.name
            trap_files = self.find_trap_files(exp_folder)
            
            if not trap_files:
                print(f"\n--- Skipping {exp_id}: No trap timeseries files found in '{self.protrusion_dir_name}'. ---")
                continue

            print(f"\n{'='*80}")
            print(f"PROCESSING EXPERIMENT: {exp_id}")
            print(f"{'='*80}")
            print(f"Found {len(trap_files)} trap files")
            
            for file_path in trap_files:
                trap_number = self.extract_trap_number(file_path.name)
                
                if not self.should_process_trap(exp_id, trap_number):
                    print(f" Skipping Trap {trap_number} (not in selection)")
                    continue
                    
                generate_plots = self.should_process_trap(exp_id, trap_number)
                raw_data, summary = self.process_single_trap(file_path, exp_folder, trap_number, plots_dir, generate_plots=generate_plots)
                
                if raw_data is not None:
                    self.all_raw_data.append(raw_data)
                if summary is not None:
                    self.all_fit_summaries.append(summary)

        print(f"\n{'='*80}")
        print("SAVING CONSOLIDATED RESULTS")
        print(f"{'='*80}")
        
        if self.all_raw_data:
            final_raw_df = pd.concat(self.all_raw_data, ignore_index=True)
            raw_output_path = self.output_dir / 'Consolidated_Raw_Timeseries_Data.csv'
            final_raw_df.to_csv(raw_output_path, index=False)
            print(f" Saved consolidated raw data (Protrusion) to: {raw_output_path.name}")
        else:
            print(" Warning: No protrusion raw data aggregated.")

        if self.all_fit_summaries:
            final_summary_df = pd.DataFrame(self.all_fit_summaries)
            summary_output_path = self.output_dir / 'Summarized_Fit_Parameters_and_Dye_Uptake.csv'
            final_summary_df.to_csv(summary_output_path, index=False)
            print(f" Saved summary (MFA + Dye Uptake) to: {summary_output_path.name}")
        else:
            print(" Warning: No fit parameters aggregated.")

        if self.all_raw_data:
            print(f"\n{'='*80}")
            print("GENERATING BULK ENSEMBLE PLOTS")
            print(f"{'='*80}")
            
            # 1. Protrusion Ensemble Plot (Full)
            print("\n Generating FULL timeseries bulk protrusion plot")
            run_bulk_fit_and_plot(self.all_raw_data, self, self.output_dir, len(self.all_raw_data), phase_label="Full")

            # 2. Dye Uptake Ensemble Plot (NEW)
            if self.all_raw_dye_data:
                print("\n Generating BULK DYE UPTAKE PLOT")
                ensemble_dye_results = calculate_dye_ensemble_average(self.all_raw_dye_data)
                if ensemble_dye_results:
                    plot_dye_ensemble_average(ensemble_dye_results, self.output_dir, self, len(self.all_raw_dye_data))
                else:
                    print(" Warning: Dye data was collected but failed to generate ensemble average (missing necessary columns).")
            else:
                print(" Warning: No dye uptake data collected for bulk analysis.")
            
            # 3. Protrusion Split Analysis
            if self.split_timepoint is not None:
                if self.all_raw_data_before:
                    print("\n Generating BEFORE split bulk plot")
                    run_bulk_fit_and_plot(self.all_raw_data_before, self, self.output_dir, len(self.all_raw_data_before), phase_label="Before_Split")
                else:
                    print("\n Warning: No 'before' data available for split analysis.")
                    
                if self.all_raw_data_after:
                    print("\n Generating AFTER split bulk plot")
                    run_bulk_fit_and_plot(self.all_raw_data_after, self, self.output_dir, len(self.all_raw_data_after), phase_label="After_Split")
                else:
                    print("\n Warning: No 'after' data available for split analysis.")

        print(f"\n{'='*80}")
        print(" ENHANCED BULK MFA ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nResults saved to: {self.output_dir}")
        
        if self.selected_traps:
            print(f"Processed {len(self.all_raw_data)} selected traps")
        print(f"Individual plots saved to: {plots_dir}")
        
        if self.split_timepoint is not None:
            print(f"Split analysis performed at t = {self.split_timepoint} seconds")
            print("Generated plots: Full, Before_Split, After_Split, and Dye Uptake Ensemble")
        else:
            print("Full timeseries analysis and Dye Uptake Ensemble plots generated.")
        print(f"{'='*80}\n")
# =============================================================================
# MAIN EXECUTION WITH USER-FRIENDLY CONFIGURATION
# =============================================================================
if __name__ == '__main__':
    # =========================================================================
    # CONFIGURATION SECTION - ADJUST THESE PARAMETERS
    # =========================================================================
    # Base folder containing experiment directories
    BASE_FOLDER_PATH = r"C:\GitHub\MFAE_Analysis\Output"

    # Physical parameters
    ASPIRATION_PRESSURE_PA = 3100.0 # Pa
    CHANNEL_WIDTH_M = 20e-6 # meters
    PORE_WIDTH_M = 6.7e-6 # meters

    # Experiment prefix to match folders
    EXPERIMENT_PREFIX = "100V_5ms"

    # =========================================================================
    # SPLIT TIMEPOINT CONFIGURATION
    # =========================================================================
    # Set to None for no split analysis (full timeseries only)
    # Set to a specific time value (in seconds) to split analysis before/after
    SPLIT_TIMEPOINT = 10.0

    # =========================================================================
    # PULSE TIME FOR VISUALIZATION
    # =========================================================================
    # Time when electrical pulse was applied (will be marked on plots)
    # Set to None to not show pulse line
    PULSE_TIME = 17.0 # seconds (adjust based on your experiment)

    # =========================================================================
    # SELECTIVE TRAP ANALYSIS CONFIGURATION
    # =========================================================================
    # Option 1: Analyze ALL traps (set to None)
    #SELECTED_TRAPS = None
    
    # Option 2: Select specific traps for each experiment
    # SELECTED_TRAPS = {
    #     "100V_5ms_pulse_Experiment1": [1, 2, 3, 5, 7],
    #     "100V_5ms_pulse_Experiment2": [1, 2, 4, 6],
    #     "100V_5ms_pulse_Experiment3": [1, 3, 5, 8, 10]
    # }

    # Option 3: Analyze only one trap from multiple experiments
    SELECTED_TRAPS = {
    "100V_5ms_pulse_Experiment3": [16],
    "100V_5ms_pulse_Experiment4": [17],
    "100V_5ms_pulse_Experiment5": [17]
    }

    # =========================================================================
    # RUN ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("MFA ANALYSIS CONFIGURATION")
    print("="*80)
    print(f"Base folder: {BASE_FOLDER_PATH}")
    print(f"Experiment prefix: {EXPERIMENT_PREFIX}")
    print(f"Pressure: {ASPIRATION_PRESSURE_PA} Pa")
    print(f"Channel width: {CHANNEL_WIDTH_M*1e6} μm")
    print(f"Pore width: {PORE_WIDTH_M*1e6} μm")
    
    if SPLIT_TIMEPOINT:
        print(f"Split analysis: YES (at t={SPLIT_TIMEPOINT}s)")
    else:
        print("Split analysis: NO")
        
    if PULSE_TIME:
        print(f"Pulse time: {PULSE_TIME}s (will be marked on plots)")

    if SELECTED_TRAPS:
        print("\nProcessing SELECTED traps only:")
        for exp, traps in SELECTED_TRAPS.items():
            print(f" - {exp}: {len(traps)} traps")
    else:
        print("\nProcessing ALL available traps")
    print("="*80 + "\n")
    
    # Initialize and run analyzer
    analyzer = BulkMFAAnalyzer(
        base_folder=BASE_FOLDER_PATH,
        experiment_prefix=EXPERIMENT_PREFIX,
        pressure_pa=ASPIRATION_PRESSURE_PA,
        channel_width_m=CHANNEL_WIDTH_M,
        pore_width_m=PORE_WIDTH_M,
        split_timepoint=SPLIT_TIMEPOINT,
        selected_traps=SELECTED_TRAPS,
        pulse_time=PULSE_TIME
    )
    analyzer.run_analysis()
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {analyzer.base_folder / f'MFA_Bulk_Results_{EXPERIMENT_PREFIX}'}")
    print(f"Individual plots saved to: {analyzer.base_folder / f'MFA_Bulk_Results_{EXPERIMENT_PREFIX}' / 'Individual_Plots'}")
    
    if SELECTED_TRAPS:
        total_traps = sum(len(traps) for traps in SELECTED_TRAPS.values())
        print(f"Processed {total_traps} selected traps")
        
    if SPLIT_TIMEPOINT:
        print(f"Split analysis performed at t = {SPLIT_TIMEPOINT} seconds")
        print("Generated plots: Full, Before_Split, After_Split, and Dye Uptake Ensemble")
    else:
        print("Full timeseries analysis and Dye Uptake Ensemble plots generated.")
    print("="*80 + "\n")
