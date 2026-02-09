# -*- coding: utf-8 -*-
"""
Plotting Module for Bulk MFAE Analysis.
=============================================================================
This module handles all data visualization for the pipeline.

It generates three categories of plots:
1. Statistical Summaries: Global trends.
2. Time-Series Dynamics: Averaged behavior.
3. Verification Panels: Multipanel grids with FITS and ANNOTATIONS.

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
import matplotlib.lines as mlines
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import bulk_file_handling as bfh

logger = logging.getLogger(__name__)

# =============================================================================
# 1. VISUAL CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=0.8)

# Color Palettes
PALETTE_STATUS = {"Intact": "#1f77b4", "Ruptured": "#d62728"} # Blue / Red
# Updated Region Palette to include Total
PALETTE_REGION = {
    "Body": "#1f77b4",       # Blue
    "Protrusion": "#ff7f0e", # Orange
    "Total": "#333333"       # Dark Grey/Black
}

# Line Styles for Multipanels
STYLE_INTACT = '-'      # Solid
STYLE_RUPTURED = ':'    # Dotted

# Recoil Fit Lines
COLOR_FIT_PRE = 'green'
COLOR_FIT_POST = 'magenta'

SAVE_DPI = 300
PANEL_DPI = 150

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def determine_status(trap: bfh.TrapData) -> str:
    prot_time = trap.protrusion_data.get('Time_s', np.array([]))
    if len(prot_time) == 0: return "Empty"
    t_filtered_max = prot_time[-1]
    t_full_max = trap.full_experiment_duration
    if t_full_max == 0: return "Unknown"
    if t_filtered_max < (0.95 * t_full_max): return "Ruptured"
    else: return "Intact"

def align_time_to_pulse(time_array: np.ndarray, pulse_frame: int) -> np.ndarray:
    if len(time_array) <= pulse_frame:
        if len(time_array) > 0:
            dt = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
            return time_array - (pulse_frame * dt)
        return time_array
    return time_array - time_array[pulse_frame]

def calculate_slope(time: np.ndarray, data: np.ndarray) -> Tuple[float, float]:
    if len(time) < 2: return np.nan, np.nan
    slope, intercept = np.polyfit(time, data, 1)
    return slope, intercept

def exponential_uptake_func(t: float, A: float, tau: float) -> float:
    return A * (1 - np.exp(-t / tau))

def fit_uptake_curve(time: np.ndarray, data: np.ndarray) -> Tuple[float, float, float]:
    mask = (time > 0) & (~np.isnan(data))
    t_fit = time[mask]
    y_fit = data[mask]
    if len(t_fit) < 5: return np.nan, np.nan, np.nan
    p0 = [np.max(y_fit), (t_fit[-1] - t_fit[0]) / 2]
    try:
        popt, _ = curve_fit(exponential_uptake_func, t_fit, y_fit, p0=p0, maxfev=5000)
        A, tau = popt
        residuals = y_fit - exponential_uptake_func(t_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return A, tau, r2
    except RuntimeError:
        return np.nan, np.nan, np.nan

def interpolate_to_common_time(time_series_list, data_series_list, dt=1.75):
    valid_pairs = [(t, y) for t, y in zip(time_series_list, data_series_list) if len(t) >= 2]
    if not valid_pairs: return np.array([]), np.array([])
    t_min = min(t[0] for t, _ in valid_pairs)
    t_max = max(t[-1] for t, _ in valid_pairs)
    common_time = np.arange(np.floor(t_min), np.ceil(t_max), dt)
    interpolated_rows = []
    for t_src, y_src in valid_pairs:
        f = interp1d(t_src, y_src, kind='linear', bounds_error=False, fill_value=np.nan)
        interpolated_rows.append(f(common_time))
    return common_time, np.array(interpolated_rows)

def _group_by_trap_id(traps: List[bfh.TrapData]) -> Dict[int, List[bfh.TrapData]]:
    grouped = {}
    for t in traps:
        if t.trap_id not in grouped: grouped[t.trap_id] = []
        grouped[t.trap_id].append(t)
    return grouped

def _add_global_legend(fig, mode='uptake'):
    handles = []
    if mode == 'uptake':
        # Regions (Color)
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Body'], label='Body', lw=2))
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Protrusion'], label='Protrusion', lw=2))
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Total'], label='Total', lw=2))
        # Status (Style)
        handles.append(mlines.Line2D([], [], color='black', linestyle=STYLE_INTACT, label='Intact'))
        handles.append(mlines.Line2D([], [], color='black', linestyle=STYLE_RUPTURED, label='Ruptured'))
    elif mode == 'recoil':
        handles.append(mlines.Line2D([], [], color=PALETTE_STATUS['Intact'], label='Intact (Data)', lw=0, marker='o'))
        handles.append(mlines.Line2D([], [], color=PALETTE_STATUS['Ruptured'], label='Ruptured (Data)', lw=0, marker='o'))
        handles.append(mlines.Line2D([], [], color=COLOR_FIT_PRE, label='Pre-Pulse Fit', lw=2))
        handles.append(mlines.Line2D([], [], color=COLOR_FIT_POST, label='Post-Pulse Fit', lw=2))

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
               ncol=len(handles), frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# 3. STATISTICAL PLOTS
# =============================================================================

def plot_max_protrusion_distribution(grouped_data, output_dir: Path):
    logger.info("Generating Plot 1: Max Protrusion...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V {traps[0].metadata.duration_label}"
        for trap in traps:
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            p_data = trap.protrusion_data
            if 'Protrusion_Length_um' in p_data and len(p_data['Protrusion_Length_um']) > 0:
                records.append({'Condition': cond_label, 'Max_Length_um': np.max(p_data['Protrusion_Length_um']), 'Status': status})
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Max_Length_um', hue='Status', palette=PALETTE_STATUS)
    sns.stripplot(data=df, x='Condition', y='Max_Length_um', hue='Status', dodge=True, palette='dark:black', alpha=0.5, legend=False)
    plt.title("Max Protrusion Length"); plt.ylabel("Length (µm)")
    plt.savefig(output_dir / "Max_Protrusion_Split.png", dpi=SAVE_DPI); plt.close()

def plot_per_trap_protrusion_distribution(grouped_data, output_dir: Path):
    """
    REPLACED: Plots the distribution of MAX Protrusion Lengths per trap.
    
    - Each Point = One Experiment (Replicate).
    - Solves "Two Medians" artifact by not plotting raw time-series.
    - Allows visibility of stochastic rupture (Blue dot vs Red dot on same Trap).
    """
    logger.info("Generating Plot 2: Per-Trap Protrusion Stability (Max Length)...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        meta = traps[0].metadata
        cond_label = f"{meta.voltage}V_{meta.duration_label}"
        
        # 1. Aggregate Data per Experiment (Replicate)
        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            
            p_data = trap.protrusion_data
            if 'Protrusion_Length_um' in p_data and len(p_data['Protrusion_Length_um']) > 0:
                # We take the MAX length to represent the protrusion size for this experiment
                val = np.max(p_data['Protrusion_Length_um'])
                records.append({
                    'Trap': f"T{trap.trap_id}", 
                    'Max_Length_um': val, 
                    'Status': status,
                    'ExpID': trap.metadata.experiment_id # Track ID to ensure dots are distinct
                })
        
        if not records: continue
        df = pd.DataFrame(records)
        
        # 2. Plotting
        plt.figure(figsize=(16, 7))
        
        # A. Boxplot (Background - shows distribution of replicates)
        # We turn off outliers in the boxplot because we will show them as dots
        sns.boxplot(
            data=df, x='Trap', y='Max_Length_um', 
            color='lightgray', showfliers=False, 
            boxprops=dict(alpha=0.4) # Make boxes faint
        )
        
        # B. Strip Plot (Foreground - shows individual experiments)
        # Jitter=True spreads dots so they don't overlap perfectly
        sns.stripplot(
            data=df, x='Trap', y='Max_Length_um', hue='Status', 
            palette=PALETTE_STATUS, size=8, jitter=True,
            edgecolor='black', linewidth=1, alpha=0.9
        )
        
        plt.title(f"Protrusion Length Consistency ({cond_label})")
        plt.ylabel("Max Protrusion Length (µm)")
        plt.xlabel("Trap ID")
        plt.xticks(rotation=45)
        
        # Move legend to best position
        plt.legend(title='Experiment Outcome', loc='upper right')
        
        plt.tight_layout()
        
        save_name = f"Protrusion_Dist_MaxPerExp_{cond_label}.png"
        plt.savefig(output_dir / save_name, dpi=SAVE_DPI)
        plt.close()
        logger.info(f"   -> Saved {save_name}")

def plot_rupture_probability(grouped_data, output_dir: Path):
    logger.info("Generating Plot 5: Rupture Probability...")
    summary = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V {traps[0].metadata.duration_label}"
        total = sum(1 for t in traps if determine_status(t) not in ["Empty", "Unknown"])
        if total == 0: continue
        ruptured = sum(1 for t in traps if determine_status(t) == "Ruptured")
        summary.append({'Condition': cond_label, 'Rupture_Percent': (ruptured/total)*100, 'Count': total})
    if not summary: return
    df = pd.DataFrame(summary)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Condition', y='Rupture_Percent', color='darkred', alpha=0.7)
    for i, r in df.iterrows(): plt.text(i, r.Rupture_Percent+1, f"n={int(r.Count)}", ha="center")
    plt.title("Rupture Probability"); plt.ylim(0, 110); plt.savefig(output_dir / "Rupture_Probability.png", dpi=SAVE_DPI); plt.close()

def plot_uptake_dynamics(grouped_data, output_dir: Path):
    """
    REPLACED: Generates a multipanel grid per Condition.
    - Panel = Trap ID.
    - Data = Mean ± SD (Standard Deviation) of Uptake Intensity.
    - Regions: Body, Protrusion, AND Total.
    - Filter = INTACT cells only.
    - Scaling = Uniform Y-axis across all traps in the condition.
    """
    logger.info("Generating Plot 3: Uptake Dynamics (Intact Only, Mean ± SD, incl. Total)...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        meta = traps[0].metadata
        cond_label = f"{meta.voltage}V_{meta.duration_label}"
        
        # 1. Group by Trap ID
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids)
        if n == 0: continue
        
        cols = 5
        rows = math.ceil(n / cols)
        
        # 2. Process Data & Find Global Y-Max
        global_max_y = 0.0
        processed_trap_data = {} 

        for tid in sorted_ids:
            trap_list = trap_groups[tid]
            
            # Container for raw arrays (INTACT ONLY)
            raw = {'t':[], 'b':[], 'p':[], 'tot':[]}
            
            for trap in trap_list:
                if determine_status(trap) != 'Intact': continue
                
                ud = trap.uptake_data
                if 'Time_s' in ud and 'Body_Normalized_dF_F0' in ud:
                    at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                    raw['t'].append(at)
                    raw['b'].append(ud['Body_Normalized_dF_F0'])
                    
                    # Protrusion
                    if 'Protrusion_Normalized_dF_F0' in ud:
                        raw['p'].append(ud['Protrusion_Normalized_dF_F0'])
                    else:
                        raw['p'].append(np.full_like(ud['Body_Normalized_dF_F0'], np.nan))
                        
                    # Total
                    if 'Total_Normalized_dF_F0' in ud:
                        raw['tot'].append(ud['Total_Normalized_dF_F0'])
                    else:
                        raw['tot'].append(np.full_like(ud['Body_Normalized_dF_F0'], np.nan))

            if not raw['t']:
                processed_trap_data[tid] = None
                continue
            
            # Interpolate all regions to common time
            ct, bm = interpolate_to_common_time(raw['t'], raw['b'], 0.5)
            _,  pm = interpolate_to_common_time(raw['t'], raw['p'], 0.5)
            _,  tm = interpolate_to_common_time(raw['t'], raw['tot'], 0.5)
            
            if len(ct) == 0:
                processed_trap_data[tid] = None
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                stats = {
                    'time': ct,
                    'body_mean': np.nanmean(bm, axis=0),
                    'body_std': np.nanstd(bm, axis=0),
                    'prot_mean': np.nanmean(pm, axis=0),
                    'prot_std': np.nanstd(pm, axis=0),
                    'total_mean': np.nanmean(tm, axis=0),
                    'total_std': np.nanstd(tm, axis=0)
                }
            
            processed_trap_data[tid] = stats
            
            # Update Global Max (Check all 3 regions)
            current_max = np.nanmax([
                np.nanmax(stats['body_mean'] + stats['body_std']),
                np.nanmax(stats['prot_mean'] + stats['prot_std']),
                np.nanmax(stats['total_mean'] + stats['total_std'])
            ])
            if current_max > global_max_y: global_max_y = current_max

        # Uniform Y-Limit
        y_limit = global_max_y * 1.1 if global_max_y > 0 else 1.0

        # 3. Create Figure
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        # Manual Legend
        handles = []
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Body'], label='Body', lw=2))
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Protrusion'], label='Protrusion', lw=2))
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Total'], label='Total', lw=2))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=12)
        
        # 4. Plot Each Trap
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            ax.set_ylim(-0.1, y_limit)
            ax.set_xlim(-5, 120)
            
            d = processed_trap_data.get(tid)
            
            if d is not None:
                # Plot Total (Grey - often better to plot first so colors pop on top, or last to be distinct)
                # Here plotting Total first (background layer) or last depends on preference. 
                # Usually Total is in between Body and Prot, so order matters less if alpha is used.
                
                # 1. Total (Grey)
                ax.plot(d['time'], d['total_mean'], color=PALETTE_REGION['Total'], lw=1.5, alpha=0.8)
                ax.fill_between(d['time'], 
                                d['total_mean'] - d['total_std'], 
                                d['total_mean'] + d['total_std'],
                                color=PALETTE_REGION['Total'], alpha=0.15, edgecolor=None)

                # 2. Body (Blue)
                ax.plot(d['time'], d['body_mean'], color=PALETTE_REGION['Body'], lw=1.5)
                ax.fill_between(d['time'], 
                                d['body_mean'] - d['body_std'], 
                                d['body_mean'] + d['body_std'],
                                color=PALETTE_REGION['Body'], alpha=0.2, edgecolor=None)
                
                # 3. Protrusion (Orange)
                ax.plot(d['time'], d['prot_mean'], color=PALETTE_REGION['Protrusion'], lw=1.5)
                ax.fill_between(d['time'], 
                                d['prot_mean'] - d['prot_std'], 
                                d['prot_mean'] + d['prot_std'],
                                color=PALETTE_REGION['Protrusion'], alpha=0.2, edgecolor=None)
            else:
                ax.text(0.5, 0.5, "No Intact Data", ha='center', va='center', fontsize=8, color='gray')

            ax.axvline(0, color='black', ls=':', lw=0.8)

        # 5. Cleanup
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=16)
        fig.text(0.01, 0.5, 'Normalized Intensity (dF/F0)', va='center', rotation='vertical', fontsize=16)
        
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        
        save_name = f"Uptake_Intact_MeanSD_{cond_label}.png"
        plt.savefig(output_dir / save_name, dpi=PANEL_DPI)
        plt.close()
        logger.info(f"   -> Saved {save_name}")

def plot_protrusion_recoil_velocity(grouped_data, output_dir: Path):
    logger.info("Generating Plot 6: Recoil Velocity...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V {traps[0].metadata.duration_label}"
        for trap in traps:
            status = determine_status(trap)
            if status != "Intact": continue
            pd_data = trap.protrusion_data
            if 'Time_s' in pd_data and len(pd_data['Time_s']) > 5:
                at = align_time_to_pulse(pd_data['Time_s'], trap.metadata.pulse_frame)
                s_pre, _ = calculate_slope(at[(at>=-5)&(at<0)], pd_data['Protrusion_Length_um'][(at>=-5)&(at<0)])
                s_post, _ = calculate_slope(at[(at>=0)&(at<=2)], pd_data['Protrusion_Length_um'][(at>=0)&(at<=2)])
                if not np.isnan(s_pre): records.append({'Condition':cond_label, 'Slope':s_pre, 'Period':'Pre-Pulse'})
                if not np.isnan(s_post): records.append({'Condition':cond_label, 'Slope':s_post, 'Period':'Post-Pulse'})
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Slope', hue='Period', palette="viridis")
    plt.axhline(0, c='k', ls=':'); plt.title("Protrusion Velocity (Intact Only)")
    plt.savefig(output_dir / "Protrusion_Recoil_Velocity.png", dpi=SAVE_DPI); plt.close()

def plot_uptake_exponential_fit(grouped_data, output_dir: Path):
    logger.info("Generating Plot 7: Tau Stats...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V {traps[0].metadata.duration_label}"
        for trap in traps:
            status = determine_status(trap)
            if status != "Intact": continue
            ud = trap.uptake_data
            if 'Time_s' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                
                # Fit Body
                if 'Body_Normalized_dF_F0' in ud:
                    _, tau, r2 = fit_uptake_curve(at, ud['Body_Normalized_dF_F0'])
                    if r2 > 0.5 and tau < 300: records.append({'Condition':cond_label, 'Tau':tau, 'Region':'Body'})
                # Fit Prot
                if 'Protrusion_Normalized_dF_F0' in ud:
                    _, tau, r2 = fit_uptake_curve(at, ud['Protrusion_Normalized_dF_F0'])
                    if r2 > 0.5 and tau < 300: records.append({'Condition':cond_label, 'Tau':tau, 'Region':'Protrusion'})
                # Fit Total
                if 'Total_Normalized_dF_F0' in ud:
                    _, tau, r2 = fit_uptake_curve(at, ud['Total_Normalized_dF_F0'])
                    if r2 > 0.5 and tau < 300: records.append({'Condition':cond_label, 'Tau':tau, 'Region':'Total'})
                    
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Tau', hue='Region', palette=PALETTE_REGION)
    plt.yscale('log'); plt.title("Uptake Time Constant (Intact Only)")
    plt.savefig(output_dir / "Uptake_Exponential_TimeConstant.png", dpi=SAVE_DPI); plt.close()

def plot_correlation_length_vs_uptake(grouped_data, output_dir: Path):
    logger.info("Generating Plot 8: Correlation...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V {traps[0].metadata.duration_label}"
        for trap in traps:
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            ml = np.max(trap.protrusion_data.get('Protrusion_Length_um', [0]))
            mu = np.max(trap.uptake_data.get('Total_Normalized_dF_F0', trap.uptake_data.get('Body_Normalized_dF_F0', [0])))
            records.append({'Condition': cond_label, 'Max_Length_um': ml, 'Max_Uptake': mu, 'Status': status})
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Max_Length_um', y='Max_Uptake', hue='Status', style='Condition', palette=PALETTE_STATUS, s=100, alpha=0.8)
    plt.title("Correlation: Length vs Uptake"); plt.savefig(output_dir / "Correlation_Length_vs_Uptake.png", dpi=SAVE_DPI); plt.close()

def plot_per_trap_uptake_distribution(grouped_data, output_dir: Path):
    """
    REPLACED: Plots the distribution of MAX Uptake Intensity per trap (Intact Only).
    
    - Metric: Maximum Normalized Intensity (dF/F0) reached after the pulse.
    - One Dot = One Experiment (Replicate).
    - Consistency: Matches the logic used for Protrusion Stability plots.
    """
    logger.info("Generating Plot 4: Body vs Protrusion Uptake (Intact Max per Exp)...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        meta = traps[0].metadata
        cond_label = f"{meta.voltage}V_{meta.duration_label}"
        
        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            status = determine_status(trap)
            
            # FILTER: Intact Only
            if status != "Intact": continue
            
            ud = trap.uptake_data
            if 'Time_s' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                
                # We only care about the MAX intensity reached AFTER the pulse
                mask = at > 0
                if not np.any(mask): continue # Skip if no post-pulse data
                
                # Helper to get max
                def get_max(col_name):
                    if col_name in ud and len(ud[col_name]) > 0:
                        valid_data = ud[col_name][mask]
                        if len(valid_data) > 0:
                            return np.nanmax(valid_data)
                    return None

                # 1. Body Max
                val_b = get_max('Body_Normalized_dF_F0')
                if val_b is not None:
                    records.append({'Trap': f"T{trap.trap_id}", 'Max_I': val_b, 'Region': 'Body'})
                
                # 2. Protrusion Max
                val_p = get_max('Protrusion_Normalized_dF_F0')
                if val_p is not None:
                    records.append({'Trap': f"T{trap.trap_id}", 'Max_I': val_p, 'Region': 'Protrusion'})
                
                # 3. Total Max
                val_t = get_max('Total_Normalized_dF_F0')
                if val_t is not None:
                    records.append({'Trap': f"T{trap.trap_id}", 'Max_I': val_t, 'Region': 'Total'})

        if not records: continue
        df = pd.DataFrame(records)
        
        # Plotting
        plt.figure(figsize=(16, 7))
        
        # A. Background Boxplot (Distribution summary)
        sns.boxplot(
            data=df, x='Trap', y='Max_I', hue='Region', 
            palette=PALETTE_REGION, showfliers=False,
            boxprops=dict(alpha=0.4) # Faint boxes
        )
        
        # B. Foreground Stripplot (Individual Experiments)
        # dodge=True ensures the dots align with the hue (Region) of the boxplots
        sns.stripplot(
            data=df, x='Trap', y='Max_I', hue='Region', 
            palette=PALETTE_REGION, size=6, jitter=True, 
            dodge=True, edgecolor='black', linewidth=0.8, alpha=0.9
        )
        
        # Clean up legend (remove duplicates from stripplot)
        handles, labels = plt.gca().get_legend_handles_labels()
        # The first 3 are boxplot, next 3 are stripplot. We only need 3.
        plt.legend(handles[:3], labels[:3], title='Region', loc='upper right')
        
        plt.title(f"Max Uptake Intensity per Experiment (Intact Only) - {cond_label}")
        plt.ylabel("Max Normalized Intensity (dF/F0)")
        plt.xlabel("Trap ID")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_name = f"Uptake_Dist_BodyVsProt_IntactMax_{cond_label}.png"
        plt.savefig(output_dir / save_name, dpi=SAVE_DPI)
        plt.close()
        logger.info(f"   -> Saved {save_name}")

# =============================================================================
# 7. MULTIPANEL PLOTS
# =============================================================================

def plot_uptake_fits_multipanel(grouped_data, output_dir: Path):
    logger.info("Generating Plot 9: Multipanel Uptake Fits...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V_{traps[0].metadata.duration_label}"
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        axes = axes.flatten()
        _add_global_legend(fig, mode='uptake')
        
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Normalized Intensity (dF/F0)', va='center', rotation='vertical', fontsize=14)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            for trap in trap_list:
                status = determine_status(trap)
                style = STYLE_RUPTURED if status == "Ruptured" else STYLE_INTACT
                ud = trap.uptake_data
                if 'Time_s' not in ud: continue
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                
                # Helper for fit+plot
                def _plot_fit(y_data, region_name):
                    color = PALETTE_REGION[region_name]
                    # Data points
                    ax.plot(at, y_data, '.', ms=2, color=color, alpha=0.3)
                    # Fit line
                    A, tau, r2 = fit_uptake_curve(at, y_data)
                    if not np.isnan(tau):
                        ax.plot(at[at>0], exponential_uptake_func(at[at>0], A, tau), 
                                ls=style, color=color, lw=1.5)
                        # Annotation
                        if status == 'Intact':
                            y_pos = 0.9 if region_name == 'Body' else (0.8 if region_name == 'Protrusion' else 0.7)
                            ax.text(0.05, y_pos, f"{region_name[0]}: τ={tau:.1f}", 
                                    transform=ax.transAxes, fontsize=7, color=color)

                if 'Body_Normalized_dF_F0' in ud: _plot_fit(ud['Body_Normalized_dF_F0'], 'Body')
                if 'Protrusion_Normalized_dF_F0' in ud: _plot_fit(ud['Protrusion_Normalized_dF_F0'], 'Protrusion')
                if 'Total_Normalized_dF_F0' in ud: _plot_fit(ud['Total_Normalized_dF_F0'], 'Total')

            ax.axvline(0, color='black', linestyle=':', linewidth=0.8)
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.savefig(output_dir / f"Fit_Quality_Panel_{cond_label}.png", dpi=PANEL_DPI); plt.close()

def plot_recoil_fits_multipanel(grouped_data, output_dir: Path):
    logger.info("Generating Plot 10: Multipanel Recoil Fits...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V_{traps[0].metadata.duration_label}"
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        axes = axes.flatten()
        _add_global_legend(fig, mode='recoil')
        
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Protrusion Length (µm)', va='center', rotation='vertical', fontsize=14)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            for trap in trap_list:
                status = determine_status(trap)
                color = PALETTE_STATUS.get(status, 'gray')
                pd_data = trap.protrusion_data
                if 'Time_s' not in pd_data: continue
                time_raw = pd_data['Time_s']; length = pd_data.get('Protrusion_Length_um', [])
                if len(time_raw) < 5: continue
                at = align_time_to_pulse(time_raw, trap.metadata.pulse_frame)
                ax.plot(at[(at>=-10)&(at<=10)], length[(at>=-10)&(at<=10)], 'o', c=color, ms=2, alpha=0.4)
                
                mp = (at>=-5)&(at<0)
                if np.sum(mp)>1: 
                    s, i_ = calculate_slope(at[mp], length[mp])
                    ax.plot(at[mp], s*at[mp]+i_, c=COLOR_FIT_PRE, lw=1.5)
                    if status == 'Intact':
                        ax.text(0.05, 0.9, f"Pre:{s:.2f}", transform=ax.transAxes, fontsize=7, color=COLOR_FIT_PRE, fontweight='bold')
                
                mpo = (at>=0)&(at<=2)
                if np.sum(mpo)>1: 
                    s, i_ = calculate_slope(at[mpo], length[mpo])
                    ax.plot(at[mpo], s*at[mpo]+i_, c=COLOR_FIT_POST, lw=1.5)
                    if status == 'Intact':
                        ax.text(0.05, 0.8, f"Post:{s:.2f}", transform=ax.transAxes, fontsize=7, color=COLOR_FIT_POST, fontweight='bold')
            
            ax.axvline(0, color='black', linestyle=':', linewidth=1)
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.savefig(output_dir / f"Recoil_Fits_Panel_{cond_label}.png", dpi=PANEL_DPI); plt.close()

def plot_uptake_traces_multipanel(grouped_data, output_dir: Path):
    logger.info("Generating Plot 11: Multipanel Uptake Traces...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = f"{traps[0].metadata.voltage}V_{traps[0].metadata.duration_label}"
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        
        # 1. Calculate Global Y-Max for this condition to standardize axes
        global_max_y = 0.0
        for trap in traps:
            ud = trap.uptake_data
            if 'Time_s' not in ud: continue
            # Check all relevant columns for the max value
            for col in ['Body_Normalized_dF_F0', 'Protrusion_Normalized_dF_F0', 'Total_Normalized_dF_F0']:
                if col in ud and len(ud[col]) > 0:
                    current_max = np.nanmax(ud[col])
                    if current_max > global_max_y:
                        global_max_y = current_max
        
        # Add 10% headroom, default to 1.0 if no data
        y_limit = global_max_y * 1.1 if global_max_y > 0 else 1.0

        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        axes = axes.flatten()
        _add_global_legend(fig, mode='uptake')
        
        # 2. Adjusted Text Placement and Layout
        # Shift x to 0.02 and use 'rect' in tight_layout to reserve space
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=16)
        fig.text(0.02, 0.5, 'Normalized Intensity (dF/F0)', va='center', rotation='vertical', fontsize=16)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            
            # Apply common Y-limit
            ax.set_ylim(-0.1, y_limit)
            
            for trap in trap_list:
                status = determine_status(trap)
                style = STYLE_RUPTURED if status == "Ruptured" else STYLE_INTACT
                ud = trap.uptake_data
                if 'Time_s' not in ud: continue
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                
                if 'Body_Normalized_dF_F0' in ud: 
                    ax.plot(at, ud['Body_Normalized_dF_F0'], ls=style, color=PALETTE_REGION['Body'], lw=1.5, alpha=0.7)
                if 'Protrusion_Normalized_dF_F0' in ud: 
                    ax.plot(at, ud['Protrusion_Normalized_dF_F0'], ls=style, color=PALETTE_REGION['Protrusion'], lw=1.5, alpha=0.7)
                if 'Total_Normalized_dF_F0' in ud:
                    ax.plot(at, ud['Total_Normalized_dF_F0'], ls=style, color=PALETTE_REGION['Total'], lw=1.5, alpha=0.7)
                    
            ax.axvline(0, color='black', ls=':', lw=0.8)
            
        # Hide unused axes
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        
        # 3. Reserve space on left (0.05) and bottom (0.05) for global labels
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        
        plt.savefig(output_dir / f"Uptake_Traces_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()