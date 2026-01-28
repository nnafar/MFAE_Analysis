# -*- coding: utf-8 -*-
"""
Bulk Protrusion Analysis with Model Fitting
- Processes multiple experiment folders with trap timeseries data
- Generates ensemble average protrusion plot with model fitting
- Marks electroporation pulse time on plot
- Publication-quality visualization
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS AND MODEL PARAMETERS
# =============================================================================
E_PLAUSIBLE_RANGE_PA = (100, 50000)
ETA1_PLAUSIBLE_RANGE_PA_S = (500, 100000)
ETA2_PLAUSIBLE_RANGE_PA_S = (1000, 200000)
R_SQUARED_THRESHOLD = 0.1

UNITS = {
    'E': r'Pa',
    'E1': r'Pa',
    'E2': r'Pa',
    'eta': r'Pa·s',
    'eta1': r'Pa·s',
    'eta2': r'Pa·s',
    'm': r'µm/s',
    'b': r'µm',
    'a': r'µm/s^b',
    'c': r'µm'
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

def exponential_model(t, a, b):
    """Exponential: y = a * (1 - exp(-b*t))"""
    return a * (1 - np.exp(-b * t))

# =============================================================================
# PROTRUSION START DETECTION
# =============================================================================
def detect_protrusion_start(time_data, length_data, zero_threshold=0.1):
    """Detect when protrusion actually starts (t_0)."""
    for i in range(len(length_data) - 1):
        if length_data[i] > zero_threshold and length_data[i + 1] > zero_threshold:
            return i, time_data[i]
    return 0, time_data[0]

def shift_time_to_start(time_data, length_data, zero_threshold=0.1):
    """Shift time axis so that t=0 corresponds to when protrusion actually starts."""
    start_idx, start_time = detect_protrusion_start(time_data, length_data, zero_threshold)
    shifted_time = time_data[start_idx:] - start_time
    shifted_length = length_data[start_idx:]
    return shifted_time, shifted_length, start_idx

# =============================================================================
# ENSEMBLE AVERAGING
# =============================================================================
def calculate_ensemble_average(all_raw_data: List[pd.DataFrame]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates ensemble average by interpolating onto a common time grid."""
    if not all_raw_data:
        return None
    
    max_time = max(df['Time_s'].max() for df in all_raw_data)
    t_common = np.linspace(0, max_time, 1000)
    
    interpolated_lengths = []
    for df in all_raw_data:
        t_data = df['Time_s'].values
        l_data = df['Protrusion_Length_um'].values
        
        unique_times, unique_indices = np.unique(t_data, return_index=True)
        unique_lengths = l_data[unique_indices]
        
        if len(unique_times) < 2:
            continue
            
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
    L_sem = L_std / np.sqrt(len(interpolated_lengths))
    
    return t_common, L_mean, L_sem

# =============================================================================
# MODEL FITTING ENGINE
# =============================================================================
class ModelFitter:
    """Handles fitting of multiple models to protrusion data."""
    def __init__(self, time_data, length_data, pressure_pa, channel_width_m, pore_width_m):
        self.t_raw = np.asarray(time_data, dtype=float)
        self.l_raw = np.asarray(length_data, dtype=float)
        self.pressure_pa = pressure_pa
        self.delta_p = pressure_pa
        self.channel_width_m = channel_width_m
        self.pore_width_m = pore_width_m
        self.r_eff = (channel_width_m * 1e6)
        self.C = 1.0
        
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
            popt, _ = curve_fit(model_func, self.t_fit, self.l_fit, p0=[E_guess, eta1_guess, eta2_guess], bounds=bounds, maxfev=5000)
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
            popt, _ = curve_fit(model_func, self.t_fit, self.l_fit, p0=[E_guess, eta_guess], bounds=bounds, maxfev=5000)
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
            popt, _ = curve_fit(model_func, self.t_fit, self.l_fit, p0=[3000, 5000, 3000, 15000], bounds=bounds, maxfev=5000)
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
        self.fit_jeffreys()
        self.fit_kelvin_voigt()
        self.fit_burgers()
        self.fit_linear()
        self.fit_power_law()
        self.fit_exponential()

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
# PUBLICATION-QUALITY PLOTTING
# =============================================================================
def plot_ensemble_protrusion(all_raw_data, analyzer, output_dir, n_trajectories, pulse_time=None):
    """Creates publication-quality ensemble protrusion plot with model fit."""
    
    # Set publication-quality plot parameters
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black'
    })
    
    ensemble_data = calculate_ensemble_average(all_raw_data)
    if ensemble_data is None:
        print("Warning: No data available for ensemble average.")
        return None
        
    t_common, l_mean, l_sem = ensemble_data
    
    # Perform fitting
    fitter = ModelFitter(t_common, l_mean, analyzer.pressure_pa, 
                        analyzer.channel_width_m, analyzer.pore_width_m)
    fitter.fit_all_models()

    if fitter.best_model is None:
        print("Warning: Could not fit models to ensemble average.")
        return None

    best_model = fitter.best_model
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot individual traces (light gray, thin)
    for df in all_raw_data:
        ax.plot(df['Time_s'], df['Protrusion_Length_um'], 
               color='lightgray', alpha=0.3, linewidth=0.8, zorder=1)
    
    # Plot ensemble average with SEM
    ax.plot(t_common, l_mean, color='#2E86AB', linewidth=2.5, 
           label=f'Ensemble Average (n={n_trajectories})', zorder=3)
    ax.fill_between(t_common, l_mean - l_sem, l_mean + l_sem, 
                    color='#2E86AB', alpha=0.2, zorder=2)
    
    # Plot best fit
    t_predict = np.linspace(t_common.min(), t_common.max(), 500)
    y_pred = fitter.predict(best_model, t_predict)
    if y_pred is not None:
        ax.plot(t_predict, y_pred, '--', color='#A23B72', linewidth=2, 
               label=f'Best Fit: {best_model}', zorder=4)
    
    # Mark pulse time
    if pulse_time is not None:
        ax.axvline(x=pulse_time, color='#F18F01', linestyle='--', 
                  linewidth=2, label='Electroporation Pulse', zorder=5)
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Protrusion Length (µm)', fontsize=13, fontweight='bold')
    ax.set_title(f'Ensemble Protrusion Analysis: {analyzer.experiment_prefix}', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Add text box with fit parameters
    params = fitter.fit_results[best_model]['params']
    param_text = f"Best Fit: {best_model}\nR² = {fitter.best_r_squared:.4f}\n\n"
    
    for name, val in params.items():
        unit = UNITS.get(name, '')
        param_text += f"{name} = {val:.2e} {unit}\n"
    
    tau = fitter.calculate_tau(best_model)
    if tau is not None:
        param_text += f"\nτ = {tau:.2f} s"
    
    # Position text box
    ax.text(0.98, 0.02, param_text, transform=ax.transAxes,
           verticalalignment='bottom', horizontalalignment='right',
           fontsize=9, bbox=dict(boxstyle='round,pad=0.6', 
           facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f'Ensemble_Protrusion_Analysis_{analyzer.experiment_prefix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved ensemble plot: {save_path.name}")
    print(f"Best model: {best_model} (R² = {fitter.best_r_squared:.4f})")
    
    return fitter.fit_results[best_model]

# =============================================================================
# BULK ANALYZER
# =============================================================================
class BulkProtrusionAnalyzer:
    def __init__(self, base_folder, experiment_prefix, pressure_pa, channel_width_m, pore_width_m, selected_traps, pulse_time=None, trap_remap = None):
        self.base_folder = Path(base_folder)
        self.experiment_prefix = experiment_prefix
        self.pressure_pa = pressure_pa
        self.channel_width_m = channel_width_m
        self.pore_width_m = pore_width_m
        self.selected_traps = selected_traps
        self.pulse_time = pulse_time
        self.trap_remap = trap_remap or {}
        self.protrusion_dir_name = "Full protrusion detection"
        self.all_raw_data = []
        self.all_fit_summaries = []
        self.trap_raw_data = defaultdict(list)
        

    def should_process_trap(self, exp_id, trap_number):
        if self.selected_traps is None:
            return True
        if exp_id in self.selected_traps:
            return trap_number in self.selected_traps[exp_id]
        return False

    def find_experiment_folders(self):
        folders = []
        for item in self.base_folder.iterdir():
            if item.is_dir() and item.name.startswith(self.experiment_prefix):
                folders.append(item)
        def sort_key(p):
            match = re.search(r'(\d+)', p.name)
            return int(match.group(1)) if match else 0
        folders.sort(key=sort_key)
        return folders

    def find_trap_files(self, experiment_folder):
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

    def extract_trap_number(self, filename):
        match = re.search(r'trap_(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    def get_canonical_trap(self, exp_id, trap_number):
        """
        Maps experiment-specific trap labels to a canonical trap ID.
        """
        if exp_id in self.trap_remap:
            return self.trap_remap[exp_id].get(trap_number, trap_number)
        return trap_number


    def process_single_trap(self, file_path, exp_folder, trap_number):
        exp_id = exp_folder.name
        print(f" Processing {exp_id} - Trap {trap_number}...", end=" ")
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                print("FAILED: Unsupported format")
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
                print("FAILED: Missing columns")
                return None, None

            time_data = df[time_col].values
            length_data = df[length_col].values
            shifted_time, shifted_length, start_idx = shift_time_to_start(time_data, length_data)

            if len(shifted_time) < 5:
                print("FAILED: Insufficient data")
                return None, None

        except Exception as e:
            print(f"ERROR: {e}")
            return None, None

        # Fit models
        fitter = ModelFitter(shifted_time, shifted_length, self.pressure_pa, self.channel_width_m, self.pore_width_m)
        fitter.fit_all_models()

        if fitter.best_model is None:
            print("FAILED: No model fitted")
            return None, None

        if fitter.best_r_squared < R_SQUARED_THRESHOLD:
            print(f"REJECTED: R² ({fitter.best_r_squared:.4f}) too low")
            return None, None

        raw_data_slice = pd.DataFrame({
            'Experiment_ID': exp_id,
            'Trap_Number': trap_number,
            'Time_s': shifted_time,
            'Protrusion_Length_um': shifted_length
        })

        summary_dict = {
            'Experiment_ID': exp_id,
            'Trap_Number': trap_number,
            'Best_Model': fitter.best_model,
            'Best_R_Squared': fitter.best_r_squared,
        }
        best_result = fitter.fit_results[fitter.best_model]
        for name, val in best_result['params'].items():
            summary_dict[f'Param_{name}'] = val
        tau = fitter.calculate_tau(fitter.best_model)
        if tau is not None:
            summary_dict['Tau_s'] = tau

        print(f"SUCCESS (R²={fitter.best_r_squared:.4f})")
        return raw_data_slice, summary_dict

    def run_analysis(self):
        print("="*80)
        print("BULK PROTRUSION ANALYSIS WITH MODEL FITTING")
        print("="*80)

        if self.selected_traps is not None:
            print("\nSELECTED TRAPS:")
            for exp_id, traps in self.selected_traps.items():
                print(f" {exp_id}: Traps {traps}")

        # ── CREATE THE ROOT RESULTS FOLDER ──────────────────────────────────────
        root_output_dir = (
            self.base_folder / f"Bulk_Protrusion_Results_{self.experiment_prefix}"
        )
        root_output_dir.mkdir(parents=True, exist_ok=True) # <-- creates if missing
        print(f"\nOutput directory: {root_output_dir}")

        experiment_folders = self.find_experiment_folders() # FIX: Assign the result here
        if not experiment_folders:
            print("No matching experiment folders found.")
            return

        for exp_folder in experiment_folders:
            exp_id = exp_folder.name
            trap_files = self.find_trap_files(exp_folder)
            if not trap_files:
                continue

            print(f"\n{'='*80}")
            print(f"PROCESSING: {exp_id}")
            print(f"{'='*80}")

            for file_path in trap_files:
                trap_number = self.extract_trap_number(file_path.name)
            
                canonical_trap = self.get_canonical_trap(exp_id, trap_number)  # ← ADD
            
                if not self.should_process_trap(exp_id, trap_number):
                    continue


                # ── PROCESS ONE TRAP ────────────────────────────────────────
                raw_data, summary = self.process_single_trap(
                    file_path, exp_folder, trap_number
                )
                if raw_data is None:
                    continue

                # ── CREATE A SUB‑FOLDER FOR THIS EXPERIMENT/TRAP ───────────────
                trap_folder = root_output_dir / f"Trap_{canonical_trap:02d}"
                trap_folder.mkdir(parents=True, exist_ok=True)
                
                sub_folder = trap_folder / exp_id
                sub_folder.mkdir(parents=True, exist_ok=True)


                # ── SAVE THE PER‑TRAP CSVs ───────────────────────────────────
                raw_path = sub_folder / f"Raw_Protrusion_Exp{exp_id}.csv"
                sum_path = sub_folder / f"Fit_Summary_Exp{exp_id}.csv"
                raw_data.to_csv(raw_path, index=False)
                pd.DataFrame([summary]).to_csv(sum_path, index=False)

                # keep the data in the in‑memory lists for the final ensemble plot
                raw_data["Canonical_Trap"] = canonical_trap
                summary["Canonical_Trap"] = canonical_trap
                
                self.all_raw_data.append(raw_data)
                self.all_fit_summaries.append(summary)
                self.trap_raw_data[canonical_trap].append(raw_data)


        print(f"\n{'='*80}")
        print("SAVING RESULTS AND GENERATING PLOTS")
        print(f"{'='*80}")

        # ── AFTER ALL TRAPS HAVE BEEN PROCESSED ───────────────────────────────────
        if self.all_raw_data:
            final_raw_df = pd.concat(self.all_raw_data, ignore_index=True)
            all_raw_path = root_output_dir / 'Consolidated_Protrusion_Data.csv'
            final_raw_df.to_csv(all_raw_path, index=False)
            print(f"\nSaved raw data: {all_raw_path.name}")

        if self.all_fit_summaries:
            final_summary_df = pd.DataFrame(self.all_fit_summaries)
            all_sum_path = root_output_dir / 'Fit_Parameters_Summary.csv'
            final_summary_df.to_csv(all_sum_path, index=False)
            print(f"Saved fit summary: {all_sum_path.name}")

        if self.all_raw_data:
            print("\nGenerating ensemble plot...")
            # Generate the ensemble plot in the root folder (or you could also
            # place it in a sub‑folder if you prefer)
            plot_ensemble_protrusion(
                self.all_raw_data, self, root_output_dir, len(self.all_raw_data), self.pulse_time
            )

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {root_output_dir}")
        print(f"Processed {len(self.all_raw_data)} traps")
        print(f"{'='*80}\n")
        
        for canonical_trap, trap_data in self.trap_raw_data.items():
            if not trap_data:
                continue
        
            trap_folder = root_output_dir / f"Trap_{canonical_trap:02d}"
            trap_folder.mkdir(parents=True, exist_ok=True)  # ← IMPORTANT SAFETY LINE
        
            print(f"Generating ensemble plot for Trap {canonical_trap}...")
        
            plot_ensemble_protrusion(
                trap_data,
                self,
                trap_folder,
                len(trap_data),
                self.pulse_time
            )

            
            


    

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Configuration
    BASE_FOLDER_PATH = r"C:\GitHub\MFAE_Analysis\Output"
    ASPIRATION_PRESSURE_PA = 3100.0
    CHANNEL_WIDTH_M = 20e-6
    PORE_WIDTH_M = 6.7e-6
    EXPERIMENT_PREFIX = "100V_5ms"
    PULSE_TIME = 17.0  # Time when electroporation pulse was applied (seconds)
    
    # Select specific traps
    SELECTED_TRAPS = {
        "100V_5ms_pulse_Experiment3": [16],
        "100V_5ms_pulse_Experiment4": [17],
        "100V_5ms_pulse_Experiment5": [17]
    }
    
    trap_remap = {
    "100V_5ms_pulse_Experiment3": {16: 17},
    "100V_5ms_pulse_Experiment4": {17: 17},
    "100V_5ms_pulse_Experiment5": {17: 17},
}

    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Base folder: {BASE_FOLDER_PATH}")
    print(f"Experiment prefix: {EXPERIMENT_PREFIX}")
    print(f"Pressure: {ASPIRATION_PRESSURE_PA} Pa")
    print(f"Channel width: {CHANNEL_WIDTH_M*1e6} µm")
    print(f"Pulse time: {PULSE_TIME}s")
    print("="*80)
    
    analyzer = BulkProtrusionAnalyzer(
        base_folder=BASE_FOLDER_PATH,
        experiment_prefix=EXPERIMENT_PREFIX,
        pressure_pa=ASPIRATION_PRESSURE_PA,
        channel_width_m=CHANNEL_WIDTH_M,
        pore_width_m=PORE_WIDTH_M,
        selected_traps=SELECTED_TRAPS,
        pulse_time=PULSE_TIME,
        trap_remap=trap_remap
    )
    
    analyzer.run_analysis()