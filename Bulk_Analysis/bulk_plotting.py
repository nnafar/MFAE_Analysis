# -*- coding: utf-8 -*-
"""
Plotting Module for Bulk MFAE Analysis.
=============================================================================
This module handles all data visualization for the pipeline.

It generates three categories of plots:
1. Statistical Summaries: Global trends (e.g., "Do higher voltages cause more ruptures?").
2. Time-Series Dynamics: Averaged behavior over time (e.g., "How fast does dye enter?").
3. Verification Panels: Multipanel grids to inspect every single trap individually.

Key Features:
- Traps are sorted numerically (T1, T2, T3...) for consistent reading.
- Time is always aligned so t=0 is the moment of the electric pulse.
- Ruptured cells are excluded from delicate kinetic measurements (like Tau or Recoil) 
  to prevent skewing the data, but included in visual panels for inspection.

Dependencies:
    - seaborn, matplotlib: For plotting
    - scipy: For curve fitting and interpolation
    - numpy, pandas: For data manipulation
    - bulk_file_handling: For custom data structures (TrapData)
"""

import logging
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Internal Imports
import bulk_file_handling as bfh

# Setup Logging
logger = logging.getLogger(__name__)

# =============================================================================
# 1. VISUAL CONFIGURATION
# =============================================================================
# Centralized settings. Edit these variables to change the look of ALL plots.

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=0.8)

# Color Palettes
# Intact = Blue, Ruptured = Red (Standard safety colors)
PALETTE_STATUS = {"Intact": "#1f77b4", "Ruptured": "#d62728"}

# Body = Blue, Protrusion = Orange (Distinguishes regions clearly)
PALETTE_REGION = {"Body": "#1f77b4", "Protrusion": "#ff7f0e"}

# Linear Fit Lines
COLOR_FIT_PRE = 'green'    # Pre-pulse baseline
COLOR_FIT_POST = 'magenta' # Immediate post-pulse reaction

# Output Resolution
SAVE_DPI = 300       # High resolution for papers/presentations
PANEL_DPI = 150      # Lower resolution for massive grids (saves disk space)


# =============================================================================
# 2. LOGIC HELPER FUNCTIONS
# =============================================================================

def determine_status(trap: bfh.TrapData) -> str:
    """
    Classifies a trap as 'Ruptured' or 'Intact' based on tracking duration.

    Logic:
        If the protrusion tracking stops significantly earlier (>5%) than the 
        full experiment duration, it implies the cell ruptured, the membrane 
        disappeared, and tracking was lost.

    Args:
        trap (TrapData): The object containing data for one cell.

    Returns:
        str: "Intact", "Ruptured", "Empty" (no data), or "Unknown" (missing full file).
    """
    prot_time = trap.protrusion_data.get('Time_s', np.array([]))
    
    if len(prot_time) == 0:
        return "Empty"
    
    # Last timestamp recorded for the filtered protrusion
    t_filtered_max = prot_time[-1]
    
    # Total duration of the experiment (from the 'Full' CSV loaded in file_handling)
    t_full_max = trap.full_experiment_duration
    
    if t_full_max == 0:
        return "Unknown" # Occurs if the 'Full' CSV wasn't found

    # Threshold: Did tracking survive at least 95% of the experiment?
    if t_filtered_max < (0.95 * t_full_max):
        return "Ruptured"
    else:
        return "Intact"


# =============================================================================
# 3. MATH & PHYSICS HELPER FUNCTIONS
# =============================================================================

def align_time_to_pulse(time_array: np.ndarray, pulse_frame: int) -> np.ndarray:
    """
    Shifts the time axis so that t=0 corresponds to the moment the electric pulse occurs.
    
    Why?
    Experiments might start recording at different times relative to the pulse. 
    Aligning them allows us to average multiple experiments together.

    Args:
        time_array (np.array): Original time vector (starts at 0).
        pulse_frame (int): The frame index where the pulse occurred.

    Returns:
        np.array: Shifted time vector (negative values = pre-pulse).
    """
    # Safety check: If data is shorter than the pulse frame (recording stopped early)
    if len(time_array) <= pulse_frame:
        if len(time_array) > 0:
            # Estimate time per frame (dt) to guess where 0 should be
            dt = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
            return time_array - (pulse_frame * dt)
        return time_array
    
    # Standard Alignment: Subtract the time at the pulse frame
    t_pulse = time_array[pulse_frame]
    return time_array - t_pulse


def calculate_slope(time: np.ndarray, data: np.ndarray) -> Tuple[float, float]:
    """
    Performs a linear regression (y = mx + c) to find velocity/rate.

    Args:
        time (np.array): X-axis data.
        data (np.array): Y-axis data.

    Returns:
        (slope, intercept): Returns (nan, nan) if insufficient data points.
    """
    if len(time) < 2:
        return np.nan, np.nan
    
    # np.polyfit(x, y, 1) returns [slope, intercept]
    slope, intercept = np.polyfit(time, data, 1)
    return slope, intercept


def exponential_uptake_func(t: float, A: float, tau: float) -> float:
    """
    The physical model for dye uptake: First-order kinetics.
    
    Formula: 
        I(t) = A * (1 - e^(-t/tau))
        
    Where:
        A   = Maximum saturation intensity (plateau)
        tau = Time constant (time to reach ~63% of A)
    """
    return A * (1 - np.exp(-t / tau))


def fit_uptake_curve(time: np.ndarray, data: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits the uptake data to the exponential model.

    Args:
        time (np.array): Aligned time vector.
        data (np.array): Normalized intensity data.

    Returns:
        (A, tau, R_squared): Fitted parameters and goodness of fit metric.
    """
    # 1. Filter: Only fit data after the pulse (t > 0) and remove NaNs
    mask = (time > 0) & (~np.isnan(data))
    t_fit = time[mask]
    y_fit = data[mask]
    
    # Need enough points (at least 5) to generate a reliable fit
    if len(t_fit) < 5:
        return np.nan, np.nan, np.nan
    
    # 2. Initial Guess (Heuristics to help the optimizer converge)
    y_max = np.max(y_fit)
    tau_guess = (t_fit[-1] - t_fit[0]) / 2 # Guess tau is half the duration
    p0 = [y_max, tau_guess]
    
    try:
        # 3. Perform Non-Linear Least Squares Fit
        popt, _ = curve_fit(exponential_uptake_func, t_fit, y_fit, p0=p0, maxfev=5000)
        A, tau = popt
        
        # 4. Calculate R-Squared (Goodness of fit)
        residuals = y_fit - exponential_uptake_func(t_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return A, tau, r2
    except RuntimeError:
        # Fitting failed to converge
        return np.nan, np.nan, np.nan


def interpolate_to_common_time(time_series_list: List[np.ndarray], 
                               data_series_list: List[np.ndarray], 
                               dt: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples a list of jagged time series (different lengths) onto a single shared time axis.
    This is required to calculate the Mean +/- SEM line for Plot 3.
    """
    valid_pairs = [(t, y) for t, y in zip(time_series_list, data_series_list) if len(t) >= 2]
    
    if not valid_pairs:
        return np.array([]), np.array([])

    # Determine the global start and end times
    t_min = min(t[0] for t, _ in valid_pairs)
    t_max = max(t[-1] for t, _ in valid_pairs)
    
    # Create the master time axis
    common_time = np.arange(np.floor(t_min), np.ceil(t_max), dt)
    
    # Interpolate every trace onto this master axis
    interpolated_rows = []
    for t, y in valid_pairs:
        f = interp1d(t, y, bounds_error=False, fill_value=np.nan)
        interpolated_rows.append(f(common_time))
        
    return common_time, np.array(interpolated_rows)


# =============================================================================
# 4. PLOTTING FUNCTIONS: STATISTICAL DISTRIBUTIONS
# =============================================================================

def plot_max_protrusion_distribution(grouped_data, output_dir: Path):
    """
    Plot 1: Box Plot of Maximum Protrusion Length.
    Groups: Intact vs Ruptured side-by-side.
    """
    logger.info("Generating Plot 1: Max Protrusion...")
    records = []
    
    # Loop over sorted keys to keep plots ordered (e.g. 5us -> 5ms -> 5s)
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        # Use the specific label from the folder (e.g. "5us")
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
        for trap in traps:
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            
            p_data = trap.protrusion_data
            if 'Protrusion_Length_um' in p_data and len(p_data['Protrusion_Length_um']) > 0:
                records.append({
                    'Condition': cond_label, 
                    'Max_Length_um': np.max(p_data['Protrusion_Length_um']), 
                    'Status': status
                })
    
    if not records: return
    df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 7))
    # Box plot shows statistics (Median, IQR)
    sns.boxplot(data=df, x='Condition', y='Max_Length_um', hue='Status', palette=PALETTE_STATUS)
    # Strip plot adds raw data points for transparency
    sns.stripplot(data=df, x='Condition', y='Max_Length_um', hue='Status', 
                  dodge=True, palette='dark:black', alpha=0.5, legend=False)
    
    plt.title("Max Protrusion Length")
    plt.ylabel("Length (µm)")
    plt.savefig(output_dir / "Max_Protrusion_Split.png", dpi=SAVE_DPI)
    plt.close()


def plot_per_trap_protrusion_distribution(grouped_data, output_dir: Path):
    """
    Plot 2: Box Plot of Protrusion Length over time for EACH trap individually.
    
    Sorting:
    Traps are sorted numerically (T1, T2, T3) to make finding specific traps easy.
    If T1 is intact and T2 is ruptured, they appear in order 1, 2.
    """
    logger.info("Generating Plot 2: Per-Trap Protrusion...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}" # Underscore for safe filename
        
        records = []
        # --- SORTING LOGIC ---
        traps_sorted = sorted(traps, key=lambda t: t.trap_id)

        for trap in traps_sorted:
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            
            p_data = trap.protrusion_data
            if 'Protrusion_Length_um' in p_data:
                # We add every time point to the boxplot to show the range of motion
                for val in p_data['Protrusion_Length_um']:
                    records.append({
                        'Trap': f"T{trap.trap_id}", 
                        'Length_um': val, 
                        'Status': status
                    })
        
        if not records: continue
        df = pd.DataFrame(records)
        
        plt.figure(figsize=(14, 6))
        # hue='Status' will color T1 blue if intact, or red if ruptured
        sns.boxplot(data=df, x='Trap', y='Length_um', hue='Status', palette=PALETTE_STATUS, dodge=False)
        plt.title(f"Protrusion Stability ({volt}V {label})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"Protrusion_Dist_{cond_label}.png", dpi=SAVE_DPI)
        plt.close()


def plot_rupture_probability(grouped_data, output_dir: Path):
    """
    Plot 5: Bar chart showing percentage of ruptured cells per condition.
    Useful for determining the "lethality" of the electric field settings.
    """
    logger.info("Generating Plot 5: Rupture Probability...")
    summary = []
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        
        total = sum(1 for t in traps if determine_status(t) not in ["Empty", "Unknown"])
        if total == 0: continue
        
        ruptured = sum(1 for t in traps if determine_status(t) == "Ruptured")
        summary.append({
            'Condition': f"{volt}V {label}", 
            'Rupture_Percent': (ruptured / total) * 100, 
            'Count': total
        })
    
    if not summary: return
    df = pd.DataFrame(summary)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Condition', y='Rupture_Percent', color='darkred', alpha=0.7)
    
    # Label bars with sample size 'n='
    for i, row in df.iterrows():
        plt.text(i, row.Rupture_Percent + 1, f"n={int(row.Count)}", ha="center")
        
    plt.title("Rupture Probability by Condition")
    plt.ylabel("Rupture Rate (%)")
    plt.ylim(0, 110)
    plt.savefig(output_dir / "Rupture_Probability.png", dpi=SAVE_DPI)
    plt.close()


# =============================================================================
# 5. PLOTTING FUNCTIONS: TIME SERIES DYNAMICS
# =============================================================================

def plot_uptake_dynamics(grouped_data, output_dir: Path):
    """
    Plot 3: Mean Uptake Trace (+/- SEM) aligned to Pulse.
    Plots Body (solid) and Protrusion (dashed) on the same graph.
    """
    logger.info("Generating Plot 3: Uptake Dynamics...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
        data_storage = {"Intact": {'t': [], 'b': [], 'p': []}, "Ruptured": {'t': [], 'b': [], 'p': []}}
        
        for trap in traps:
            status = determine_status(trap)
            if status not in data_storage: continue
            ud = trap.uptake_data
            if 'Time_s' in ud and 'Body_Normalized_dF_F0' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                data_storage[status]['t'].append(at)
                data_storage[status]['b'].append(ud['Body_Normalized_dF_F0'])
                data_storage[status]['p'].append(ud['Protrusion_Normalized_dF_F0'])
        
        fig, ax = plt.subplots(figsize=(12, 7))
        has_data = False
        
        for status, color in PALETTE_STATUS.items():
            t_list = data_storage[status]['t']
            if not t_list: continue
            
            # Interpolate to common time axis for averaging
            ct, b_mat = interpolate_to_common_time(t_list, data_storage[status]['b'], 0.5)
            _,  p_mat = interpolate_to_common_time(t_list, data_storage[status]['p'], 0.5)
            
            if len(ct) == 0: continue
            has_data = True
            
            # Suppress "Mean of empty slice" warnings (happens at the tail end of uneven traces)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                b_mean = np.nanmean(b_mat, axis=0)
                b_sem = np.nanstd(b_mat, axis=0) / np.sqrt(len(t_list))
                p_mean = np.nanmean(p_mat, axis=0)
                p_sem = np.nanstd(p_mat, axis=0) / np.sqrt(len(t_list))
            
            ax.plot(ct, b_mean, color=color, ls='-', lw=2, label=f"{status} Body")
            ax.fill_between(ct, b_mean - b_sem, b_mean + b_sem, color=color, alpha=0.2)
            ax.plot(ct, p_mean, color=color, ls='--', lw=2, label=f"{status} Prot")
            ax.fill_between(ct, p_mean - p_sem, p_mean + p_sem, color=color, alpha=0.1)
            
        if not has_data: plt.close(); continue
        
        ax.axvline(0, color='black', ls=':', label="Pulse")
        ax.set_xlim(left=-5)
        ax.set_title(f"Dye Uptake Dynamics ({cond_label})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"Uptake_Dynamics_Aligned_{cond_label}.png", dpi=SAVE_DPI)
        plt.close()


def plot_protrusion_recoil_velocity(grouped_data, output_dir: Path):
    """
    Plot 6: Boxplots of Recoil Velocity (dL/dt) Pre- and Post-Pulse.
    
    FILTER:
    *** EXCLUDES RUPTURED CELLS *** Ruptured cells often have chaotic motion (collapsing) rather than elastic recoil.
    To get a clean measurement of membrane mechanics, we only plot Intact cells here.
    """
    logger.info("Generating Plot 6: Recoil Velocity Boxplots (Intact Only)...")
    records = []
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
        for trap in traps:
            status = determine_status(trap)
            if status != "Intact": continue # Filter applied here
            
            pd_data = trap.protrusion_data
            if 'Time_s' in pd_data and len(pd_data['Time_s']) > 5:
                at = align_time_to_pulse(pd_data['Time_s'], trap.metadata.pulse_frame)
                length = pd_data['Protrusion_Length_um']
                
                # Pre-Pulse Window (-5s to 0s)
                mask_pre = (at >= -5.0) & (at < 0)
                s_pre, _ = calculate_slope(at[mask_pre], length[mask_pre])
                
                # Post-Pulse Window (0s to 2s)
                mask_post = (at >= 0) & (at <= 2.0)
                s_post, _ = calculate_slope(at[mask_post], length[mask_post])
                
                if not np.isnan(s_pre): records.append({'Condition':cond_label, 'Slope':s_pre, 'Period':'Pre-Pulse'})
                if not np.isnan(s_post): records.append({'Condition':cond_label, 'Slope':s_post, 'Period':'Post-Pulse'})
                    
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Slope', hue='Period', palette="viridis")
    plt.axhline(0, color='black', ls=':')
    plt.title("Protrusion Velocity (dL/dt) - Intact Cells Only")
    plt.ylabel("Velocity (µm/s)")
    plt.savefig(output_dir / "Protrusion_Recoil_Velocity.png", dpi=SAVE_DPI)
    plt.close()


# =============================================================================
# 6. PLOTTING FUNCTIONS: CORRELATIONS & FITS
# =============================================================================

def plot_uptake_exponential_fit(grouped_data, output_dir: Path):
    """
    Plot 7: Boxplots of Time Constant (tau) from exponential fit.
    
    FILTER:
    *** EXCLUDES RUPTURED CELLS ***
    We only want to measure the permeabilization kinetics of cells that survive.
    """
    logger.info("Generating Plot 7: Tau Stats (Intact Only)...")
    records = []
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
        for trap in traps:
            status = determine_status(trap)
            if status != "Intact": continue # Filter applied here
            ud = trap.uptake_data
            
            if 'Time_s' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                
                # Fit Body
                if 'Body_Normalized_dF_F0' in ud:
                    _, tau, r2 = fit_uptake_curve(at, ud['Body_Normalized_dF_F0'])
                    # Filter poor fits (R2 > 0.5) and outliers (tau < 300s)
                    if r2 > 0.5 and tau < 300: 
                        records.append({'Condition':cond_label, 'Tau':tau, 'Region':'Body'})
                
                # Fit Protrusion
                if 'Protrusion_Normalized_dF_F0' in ud:
                    _, tau, r2 = fit_uptake_curve(at, ud['Protrusion_Normalized_dF_F0'])
                    if r2 > 0.5 and tau < 300: 
                        records.append({'Condition':cond_label, 'Tau':tau, 'Region':'Protrusion'})
                        
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Tau', hue='Region', palette=PALETTE_REGION)
    plt.yscale('log') # Log scale is standard for rate constants
    plt.title("Uptake Time Constant (τ) - Intact Cells Only")
    plt.ylabel("Time Constant (s)")
    plt.savefig(output_dir / "Uptake_Exponential_TimeConstant.png", dpi=SAVE_DPI)
    plt.close()


def plot_correlation_length_vs_uptake(grouped_data, output_dir: Path):
    """
    Plot 8: Scatter plot correlating mechanics (Length) vs electroporation (Uptake).
    """
    logger.info("Generating Plot 8: Correlation...")
    records = []
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
        for trap in traps:
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            
            ml = np.max(trap.protrusion_data.get('Protrusion_Length_um', [0]))
            mu = np.max(trap.uptake_data.get('Body_Normalized_dF_F0', [0]))
            
            records.append({'Condition': cond_label, 'Max_Length_um': ml, 'Max_Uptake': mu, 'Status': status})
    
    if not records: return
    df = pd.DataFrame(records)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Max_Length_um', y='Max_Uptake', hue='Status', 
                    style='Condition', palette=PALETTE_STATUS, s=100, alpha=0.8)
    
    plt.title("Correlation: Protrusion Length vs. Dye Uptake")
    plt.xlabel("Max Protrusion Length (µm)")
    plt.ylabel("Max Norm. Uptake (dF/F0)")
    plt.savefig(output_dir / "Correlation_Length_vs_Uptake.png", dpi=SAVE_DPI)
    plt.close()


# =============================================================================
# 7. PLOTTING FUNCTIONS: MULTIPANEL VERIFICATION (INCLUDES ALL TRAPS)
# =============================================================================

def plot_per_trap_uptake_distribution(grouped_data, output_dir: Path):
    """
    Plot 4: Box Plot per Trap for Post-Pulse Intensities (Side-by-side Body/Prot).
    Sorted by Trap ID.
    """
    logger.info("Generating Plot 4: Body vs Protrusion Uptake...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        records = []
        # Sort traps numerically
        for trap in sorted(traps, key=lambda t: t.trap_id):
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            ud = trap.uptake_data
            if 'Time_s' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                mask = at > 0 # Post-pulse data only
                for v in ud['Body_Normalized_dF_F0'][mask]: 
                    records.append({'Trap':f"T{trap.trap_id}", 'I':v, 'Region':'Body', 'Status':status})
                for v in ud['Protrusion_Normalized_dF_F0'][mask]: 
                    records.append({'Trap':f"T{trap.trap_id}", 'I':v, 'Region':'Protrusion', 'Status':status})
        
        if not records: continue
        df = pd.DataFrame(records)
        plt.figure(figsize=(16, 7))
        sns.boxplot(data=df, x='Trap', y='I', hue='Region', palette=PALETTE_REGION)
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(output_dir / f"Uptake_Dist_BodyVsProt_{cond_label}.png", dpi=SAVE_DPI); plt.close()

def plot_uptake_fits_multipanel(grouped_data, output_dir: Path):
    """
    Plot 9: Grid view showing raw uptake data + exponential fit curve.
    
    Purpose: Verification. Allows you to check if the 'Tau' values in Plot 7 
    actually match the data, or if the fitting algorithm failed.
    Includes Ruptured cells.
    """
    logger.info("Generating Plot 9: Multipanel Uptake Fit Verification...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        traps_sorted = sorted(traps, key=lambda t: t.trap_id)
        
        # Grid Layout Calculation
        n = len(traps_sorted)
        if n == 0: continue
        cols = 5
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        axes = axes.flatten()
        
        for i, trap in enumerate(traps_sorted):
            ax = axes[i]
            status = determine_status(trap)
            ud = trap.uptake_data
            ax.set_title(f"Trap {trap.trap_id} ({status})", fontsize=10, fontweight='bold')
            
            if 'Time_s' not in ud: continue
            at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
            
            # Plot Body (Blue)
            if 'Body_Normalized_dF_F0' in ud:
                y = ud['Body_Normalized_dF_F0']
                ax.plot(at, y, '.', ms=3, color=PALETTE_REGION['Body'], alpha=0.4)
                A, tau, r2 = fit_uptake_curve(at, y)
                if not np.isnan(tau):
                    ax.plot(at[at>0], exponential_uptake_func(at[at>0], A, tau), color=PALETTE_REGION['Body'])

            # Plot Protrusion (Orange)
            if 'Protrusion_Normalized_dF_F0' in ud:
                y = ud['Protrusion_Normalized_dF_F0']
                ax.plot(at, y, '.', ms=3, color=PALETTE_REGION['Protrusion'], alpha=0.4)
                A, tau, r2 = fit_uptake_curve(at, y)
                if not np.isnan(tau):
                    ax.plot(at[at>0], exponential_uptake_func(at[at>0], A, tau), color=PALETTE_REGION['Protrusion'])
                    
            ax.axvline(0, color='black', linestyle=':', linewidth=0.8)
            
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"Fit_Quality_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()

def plot_recoil_fits_multipanel(grouped_data, output_dir: Path):
    """
    Plot 10: Grid view of Protrusion Recoil with Pre/Post linear fits.
    
    Purpose: Verification. Visually confirms if the slopes (velocities) calculated 
    in Plot 6 correspond to real movement.
    """
    logger.info("Generating Plot 10: Multipanel Recoil Verification...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        traps_sorted = sorted(traps, key=lambda t: t.trap_id)
        n = len(traps_sorted); cols = 5; rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        axes = axes.flatten()
        
        for i, trap in enumerate(traps_sorted):
            ax = axes[i]; status = determine_status(trap); pd_data = trap.protrusion_data
            ax.set_title(f"Trap {trap.trap_id} ({status})", fontsize=10, fontweight='bold')
            
            if 'Time_s' not in pd_data: continue
            time_raw = pd_data['Time_s']
            length = pd_data.get('Protrusion_Length_um', [])
            if len(time_raw) < 5: continue
            
            at = align_time_to_pulse(time_raw, trap.metadata.pulse_frame)
            ax.plot(at[(at>=-10)&(at<=10)], length[(at>=-10)&(at<=10)], 'o', c='gray', ms=3, alpha=0.5)
            
            # Draw Fit Lines
            mask_pre = (at >= -5.0) & (at < 0)
            if np.sum(mask_pre) > 1:
                t, l = at[mask_pre], length[mask_pre]; s, int_ = calculate_slope(t, l)
                ax.plot(t, s*t + int_, color=COLOR_FIT_PRE, lw=2)
                ax.text(0.05, 0.9, f"Pre:{s:.2f}", transform=ax.transAxes, fontsize=8, color=COLOR_FIT_PRE, fontweight='bold')
                
            mask_post = (at >= 0) & (at <= 2.0)
            if np.sum(mask_post) > 1:
                t, l = at[mask_post], length[mask_post]; s, int_ = calculate_slope(t, l)
                ax.plot(t, s*t + int_, color=COLOR_FIT_POST, lw=2)
                ax.text(0.05, 0.8, f"Post:{s:.2f}", transform=ax.transAxes, fontsize=8, color=COLOR_FIT_POST, fontweight='bold')
                
            ax.axvline(0, color='black', linestyle=':', linewidth=1)
            
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"Recoil_Fits_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()

def plot_uptake_traces_multipanel(grouped_data, output_dir: Path):
    """
    Plot 11: Multipanel showing raw uptake traces (Body vs Prot) per trap.
    
    Purpose: Verification. Shows the raw trajectory without fitting lines.
    Useful for spotting weird artifacts or signal noise.
    """
    logger.info("Generating Plot 11: Multipanel Uptake Traces...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        traps_sorted = sorted(traps, key=lambda t: t.trap_id)
        n = len(traps_sorted); cols = 5; rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        axes = axes.flatten()
        
        for i, trap in enumerate(traps_sorted):
            ax = axes[i]; status = determine_status(trap); ud = trap.uptake_data
            ax.set_title(f"Trap {trap.trap_id} ({status})", fontsize=10, fontweight='bold')
            
            if 'Time_s' not in ud: continue
            at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
            
            if 'Body_Normalized_dF_F0' in ud:
                ax.plot(at, ud['Body_Normalized_dF_F0'], '-', color=PALETTE_REGION['Body'], lw=1.5, label='Body', alpha=0.8)
            if 'Protrusion_Normalized_dF_F0' in ud:
                ax.plot(at, ud['Protrusion_Normalized_dF_F0'], '-', color=PALETTE_REGION['Protrusion'], lw=1.5, label='Prot', alpha=0.8)
            ax.axvline(0, color='black', ls=':', lw=0.8)
            
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"Uptake_Traces_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()