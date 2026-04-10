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
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

import bulk_file_handling as bfh

logger = logging.getLogger(__name__)

# =============================================================================
# 1. VISUAL CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=0.8)

PALETTE_REGION = {
    "Body": "#1f77b4",       # Blue
    "Protrusion": "#ff7f0e", # Orange
    "Total": "#333333"       # Dark Grey/Black
}

MI_MODEL_PALETTE = {
    "Linear":    "#e377c2",  # Pink
    "Power-Law": "#17becf",  # Cyan
}

# Line Styles for Multipanels
STYLE_DEFAULT = '-'      

# Recoil Fit Lines
COLOR_FIT_PRE = 'green'
COLOR_FIT_POST = 'magenta'

SAVE_DPI = 300
PANEL_DPI = 150

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_cond_label(meta: bfh.ExperimentMetadata) -> str:
    if meta.condition_type == "ASP":
        return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_ASP"
    return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_{meta.voltage}V_{meta.duration_label}"

def align_time_to_pulse(time_array: np.ndarray, pulse_frame: int, condition_type: str) -> np.ndarray:
    if len(time_array) == 0:
        return time_array
    if condition_type == "ASP":
        return time_array - time_array[0]
    if pulse_frame < len(time_array):
        return time_array - time_array[pulse_frame]
    dt = np.mean(np.diff(time_array)) if len(time_array) > 1 else 1.0
    return time_array - (time_array[0] + pulse_frame * dt)

def calculate_slope(time: np.ndarray, data: np.ndarray) -> Tuple[float, float]:
    if len(time) < 2: return np.nan, np.nan
    slope, intercept = np.polyfit(time, data, 1)
    return slope, intercept

def interpolate_to_common_time(time_series_list, data_series_list, dt=1.75):
    """
    Interpolates multiple time-series onto a shared time grid for averaging.

    Uses Pchip (monotone cubic Hermite) instead of plain cubic splines.
    Pchip preserves local monotonicity, which prevents the artificial
    negative dF/F₀ values that cubic overshoot can produce near plateaus.
    """
    valid_pairs = [(t, y) for t, y in zip(time_series_list, data_series_list) if len(t) >= 4]
    if not valid_pairs: return np.array([]), np.array([])
    t_min = min(t[0] for t, _ in valid_pairs)
    t_max = max(t[-1] for t, _ in valid_pairs)
    common_time = np.arange(np.floor(t_min), np.ceil(t_max), dt)
    interpolated_rows = []
    for t_src, y_src in valid_pairs:
        _, unique_idx = np.unique(t_src, return_index=True)
        t_u, y_u = t_src[unique_idx], y_src[unique_idx]
        if len(t_u) < 2:
            interpolated_rows.append(np.full_like(common_time, np.nan))
            continue
        f = PchipInterpolator(t_u, y_u, extrapolate=False)
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
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Body'], label='Body', lw=2))
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Protrusion'], label='Protrusion', lw=2))
        handles.append(mlines.Line2D([], [], color=PALETTE_REGION['Total'], label='Total', lw=2))
    elif mode == 'recoil':
        handles.append(mlines.Line2D([], [], color='gray', label='Data', lw=0, marker='o', alpha=0.5))
        handles.append(mlines.Line2D([], [], color=COLOR_FIT_PRE, label='Pre-Pulse Fit', lw=2))
        handles.append(mlines.Line2D([], [], color=COLOR_FIT_POST, label='Post-Pulse Fit', lw=2))

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
               ncol=len(handles), frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def _safe_max(arr: np.ndarray, default: float = 0.0) -> float:
    """
    FIX #4: Safe wrapper around np.max that handles empty arrays.
    Returns `default` if the array is empty or all-NaN, instead of
    raising a ValueError.
    """
    if arr is None or (isinstance(arr, np.ndarray) and len(arr) == 0):
        return default
    if isinstance(arr, (list, tuple)) and len(arr) == 0:
        return default
    try:
        val = float(np.nanmax(arr))
        return val if np.isfinite(val) else default
    except (ValueError, TypeError):
        return default


# =============================================================================
# 3. STATISTICAL PLOTS
# =============================================================================

def plot_max_protrusion_distribution(grouped_data, output_dir: Path):
    logger.info("Generating Plot: Max Protrusion...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        for trap in traps:
            p_data = trap.protrusion_data
            if 'Protrusion_Length_um' in p_data and len(p_data['Protrusion_Length_um']) > 0:
                records.append({'Condition': cond_label, 'Max_Length_um': np.max(p_data['Protrusion_Length_um'])})
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Max_Length_um', color='lightgray')
    sns.stripplot(data=df, x='Condition', y='Max_Length_um', dodge=True, color='black', alpha=0.5)
    plt.title("Max Protrusion Length")
    plt.ylabel("Length (µm)")
    plt.savefig(output_dir / "Max_Protrusion.png", dpi=SAVE_DPI)
    plt.close()

def plot_per_trap_protrusion_distribution(grouped_data, output_dir: Path):
    logger.info("Generating Plot: Per-Trap Protrusion Stability (Max Length)...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        
        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            p_data = trap.protrusion_data
            if 'Protrusion_Length_um' in p_data and len(p_data['Protrusion_Length_um']) > 0:
                val = np.max(p_data['Protrusion_Length_um'])
                records.append({
                    'Trap': f"T{trap.trap_id}", 
                    'Max_Length_um': val, 
                    'ExpID': trap.metadata.experiment_number
                })
        
        if not records: continue
        df = pd.DataFrame(records)
        
        plt.figure(figsize=(16, 7))
        sns.boxplot(data=df, x='Trap', y='Max_Length_um', color='lightgray', showfliers=False, boxprops=dict(alpha=0.4))
        sns.stripplot(data=df, x='Trap', y='Max_Length_um', color='black', size=8, jitter=True, edgecolor='black', linewidth=1, alpha=0.6)
        
        plt.title(f"Protrusion Length Consistency ({cond_label})")
        plt.ylabel("Max Protrusion Length (µm)")
        plt.xlabel("Trap ID")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_name = f"Protrusion_Dist_MaxPerExp_{cond_label}.png"
        plt.savefig(output_dir / save_name, dpi=SAVE_DPI)
        plt.close()

def plot_uptake_dynamics(grouped_data, output_dir: Path):
    logger.info("Generating Plot: Uptake Dynamics (Mean ± SD, incl. Total)...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids)
        if n == 0: continue
        
        cols = 5
        rows = math.ceil(n / cols)
        global_max_y = 0.0
        processed_trap_data = {} 

        for tid in sorted_ids:
            trap_list = trap_groups[tid]
            raw = {'t':[], 'b':[], 'p':[], 'tot':[]}
            
            for trap in trap_list:
                ud = trap.uptake_data
                if 'Time_s' in ud and 'Body_Normalized_dF_F0' in ud:
                    at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame, trap.metadata.condition_type)
                    raw['t'].append(at)
                    raw['b'].append(ud['Body_Normalized_dF_F0'])
                    if 'Protrusion_Normalized_dF_F0' in ud:
                        raw['p'].append(ud['Protrusion_Normalized_dF_F0'])
                    else:
                        raw['p'].append(np.full_like(ud['Body_Normalized_dF_F0'], np.nan))
                    if 'Total_Normalized_dF_F0' in ud:
                        raw['tot'].append(ud['Total_Normalized_dF_F0'])
                    else:
                        raw['tot'].append(np.full_like(ud['Body_Normalized_dF_F0'], np.nan))

            if not raw['t']:
                processed_trap_data[tid] = None
                continue
            
            ct, bm_arr = interpolate_to_common_time(raw['t'], raw['b'], 0.5)
            _,  pm = interpolate_to_common_time(raw['t'], raw['p'], 0.5)
            _,  tm = interpolate_to_common_time(raw['t'], raw['tot'], 0.5)
            
            if len(ct) == 0:
                processed_trap_data[tid] = None
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stats = {
                    'time': ct,
                    'body_mean': np.nanmean(bm_arr, axis=0),
                    'body_std': np.nanstd(bm_arr, axis=0),
                    'prot_mean': np.nanmean(pm, axis=0),
                    'prot_std': np.nanstd(pm, axis=0),
                    'total_mean': np.nanmean(tm, axis=0),
                    'total_std': np.nanstd(tm, axis=0)
                }
            processed_trap_data[tid] = stats
            
            current_max = np.nanmax([
                np.nanmax(stats['body_mean'] + stats['body_std']),
                np.nanmax(stats['prot_mean'] + stats['prot_std']),
                np.nanmax(stats['total_mean'] + stats['total_std'])
            ])
            if current_max > global_max_y: global_max_y = current_max

        y_limit = global_max_y * 1.1 if global_max_y > 0 else 1.0
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        _add_global_legend(fig, mode='uptake')
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            ax.set_ylim(-0.1, y_limit)
            ax.set_xlim(-5, 120)
            d = processed_trap_data.get(tid)
            
            if d is not None:
                ax.plot(d['time'], d['total_mean'], color=PALETTE_REGION['Total'], lw=1.5, alpha=0.8)
                ax.fill_between(d['time'], d['total_mean'] - d['total_std'], d['total_mean'] + d['total_std'], color=PALETTE_REGION['Total'], alpha=0.15, edgecolor=None)
                ax.plot(d['time'], d['body_mean'], color=PALETTE_REGION['Body'], lw=1.5)
                ax.fill_between(d['time'], d['body_mean'] - d['body_std'], d['body_mean'] + d['body_std'], color=PALETTE_REGION['Body'], alpha=0.2, edgecolor=None)
                ax.plot(d['time'], d['prot_mean'], color=PALETTE_REGION['Protrusion'], lw=1.5)
                ax.fill_between(d['time'], d['prot_mean'] - d['prot_std'], d['prot_mean'] + d['prot_std'], color=PALETTE_REGION['Protrusion'], alpha=0.2, edgecolor=None)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=8, color='gray')

            ax.axvline(0, color='black', ls=':', lw=0.8)

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=16)
        fig.text(0.01, 0.5, 'Normalized Intensity (dF/F0)', va='center', rotation='vertical', fontsize=16)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        save_name = f"Uptake_MeanSD_{cond_label}.png"
        plt.savefig(output_dir / save_name, dpi=PANEL_DPI)
        plt.close()


# FIX #8: Refactored to read from mechanics_df instead of recalculating slopes.
# This eliminates duplication with bulk_mechanics pre/post slope logic.
def plot_protrusion_recoil_velocity(mechanics_df: pd.DataFrame, output_dir: Path):
    """
    Boxplot of pre-pulse vs post-pulse protrusion slope for EP cells.
    Reads pre-computed slopes from mechanics_results.csv.
    """
    logger.info("Generating Plot: Recoil Velocity...")

    df_ep = mechanics_df[mechanics_df['Condition_Type'] == 'EP'].copy()
    if df_ep.empty:
        logger.info("  No EP data — skipping recoil velocity plot.")
        return

    df_ep['Cond'] = df_ep['Condition']

    records = []
    for _, row in df_ep.iterrows():
        if pd.notna(row.get('Pre_Pulse_Slope')):
            records.append({'Condition': row['Cond'], 'Slope': row['Pre_Pulse_Slope'], 'Period': 'Pre-Pulse'})
        if pd.notna(row.get('Post_Pulse_Slope')):
            records.append({'Condition': row['Cond'], 'Slope': row['Post_Pulse_Slope'], 'Period': 'Post-Pulse'})

    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Slope', hue='Period', palette="viridis")
    plt.axhline(0, c='k', ls=':')
    plt.title("Protrusion Velocity")
    plt.ylabel("Slope [µm/s]")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "Protrusion_Recoil_Velocity.png", dpi=SAVE_DPI)
    plt.close()


# FIX #3: Rewritten to use bm.fit_exponential_uptake() — the same function
# that produces the tau values in mechanics_results.csv.  The old version
# used a local fit_uptake_curve() helper that did NOT subtract a pre-pulse
# baseline and handled pulse_frame differently, causing the annotated tau
# values on this plot to disagree with the stored CSV values.
def plot_uptake_exponential_fit(grouped_data, output_dir: Path):
    import bulk_mechanics as bm

    logger.info("Generating Plot: Tau Stats...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        for trap in traps:
            ud = trap.uptake_data
            if 'Time_s' not in ud:
                continue
            pf = trap.metadata.pulse_frame if trap.metadata.condition_type == "EP" else 0
            for col_name, region_name in [
                ('Body_Normalized_dF_F0', 'Body'),
                ('Protrusion_Normalized_dF_F0', 'Protrusion'),
                ('Total_Normalized_dF_F0', 'Total'),
            ]:
                if col_name in ud:
                    fit = bm.fit_exponential_uptake(
                        ud['Time_s'], ud[col_name], pulse_frame=pf
                    )
                    if fit['r2'] is not None and fit['r2'] > 0.5 and fit['tau'] is not None:
                        records.append({
                            'Condition': cond_label,
                            'Tau': fit['tau'],
                            'Region': region_name,
                        })
                    
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Condition', y='Tau', hue='Region', palette=PALETTE_REGION)
    plt.yscale('log')
    plt.title("Uptake Time Constant")
    plt.ylabel("τ (s)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "Uptake_Exponential_TimeConstant.png", dpi=SAVE_DPI)
    plt.close()

def plot_correlation_length_vs_uptake(grouped_data, output_dir: Path):
    """FIX #4: Uses _safe_max() to handle empty arrays without crashing."""
    logger.info("Generating Plot: Correlation...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        for trap in traps:
            ml = _safe_max(trap.protrusion_data.get('Protrusion_Length_um', np.array([])))
            # Try Total first, fall back to Body
            total_arr = trap.uptake_data.get('Total_Normalized_dF_F0', np.array([]))
            body_arr  = trap.uptake_data.get('Body_Normalized_dF_F0', np.array([]))
            mu = _safe_max(total_arr) if len(total_arr) > 0 else _safe_max(body_arr)
            records.append({'Condition': cond_label, 'Max_Length_um': ml, 'Max_Uptake': mu})
    if not records: return
    df = pd.DataFrame(records)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Max_Length_um', y='Max_Uptake', hue='Condition', s=100, alpha=0.8)
    plt.title("Correlation: Length vs Uptake")
    plt.xlabel("Max Protrusion Length (µm)")
    plt.ylabel("Max Uptake (dF/F₀)")
    plt.tight_layout()
    plt.savefig(output_dir / "Correlation_Length_vs_Uptake.png", dpi=SAVE_DPI)
    plt.close()

def plot_per_trap_uptake_distribution(grouped_data, output_dir: Path):
    logger.info("Generating Plot: Body vs Protrusion Uptake (Max per Exp)...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        
        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            ud = trap.uptake_data
            if 'Time_s' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame, trap.metadata.condition_type)
                mask = at > 0
                if not np.any(mask): continue
                
                def get_max(col_name):
                    if col_name in ud and len(ud[col_name]) > 0:
                        valid_data = ud[col_name][mask]
                        if len(valid_data) > 0: return float(np.nanmax(valid_data))
                    return None

                val_b = get_max('Body_Normalized_dF_F0')
                if val_b is not None: records.append({'Trap': f"T{trap.trap_id}", 'Max_I': val_b, 'Region': 'Body'})
                val_p = get_max('Protrusion_Normalized_dF_F0')
                if val_p is not None: records.append({'Trap': f"T{trap.trap_id}", 'Max_I': val_p, 'Region': 'Protrusion'})
                val_t = get_max('Total_Normalized_dF_F0')
                if val_t is not None: records.append({'Trap': f"T{trap.trap_id}", 'Max_I': val_t, 'Region': 'Total'})

        if not records: continue
        df = pd.DataFrame(records)
        
        plt.figure(figsize=(16, 7))
        sns.boxplot(data=df, x='Trap', y='Max_I', hue='Region', palette=PALETTE_REGION, showfliers=False, boxprops=dict(alpha=0.4))
        sns.stripplot(data=df, x='Trap', y='Max_I', hue='Region', palette=PALETTE_REGION, size=6, jitter=True, dodge=True, edgecolor='black', linewidth=0.8, alpha=0.9)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:3], labels[:3], title='Region', loc='upper right')
        
        plt.title(f"Max Uptake Intensity per Experiment - {cond_label}")
        plt.ylabel("Max Normalized Intensity (dF/F0)")
        plt.xlabel("Trap ID")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_name = f"Uptake_Dist_BodyVsProt_{cond_label}.png"
        plt.savefig(output_dir / save_name, dpi=SAVE_DPI)
        plt.close()

# =============================================================================
# 7. MULTIPANEL PLOTS
# =============================================================================

def plot_uptake_fits_multipanel(grouped_data, output_dir: Path):
    """
    Multipanel grid showing uptake data with exponential fit overlays.
    Uses bulk_mechanics.fit_exponential_uptake() for consistency with CSV.
    """
    import bulk_mechanics as bm

    logger.info("Generating Plot: Multipanel Uptake Fits...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        if n == 0: continue
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        _add_global_legend(fig, mode='uptake')
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Normalized Intensity (dF/F0)', va='center', rotation='vertical', fontsize=14)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            for trap in trap_list:
                ud = trap.uptake_data
                if 'Time_s' not in ud: continue
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame, trap.metadata.condition_type)
                pf = trap.metadata.pulse_frame if trap.metadata.condition_type == "EP" else 0
                
                def _plot_fit(col_name, region_name):
                    if col_name not in ud: return
                    y_data = ud[col_name]
                    color = PALETTE_REGION[region_name]
                    ax.plot(at, y_data, '.', ms=2, color=color, alpha=0.3)
                    fit = bm.fit_exponential_uptake(ud['Time_s'], y_data, pulse_frame=pf)
                    if fit['tau'] is not None:
                        t_post = at[at > 0]
                        baseline = fit.get('baseline', 0.0) or 0.0
                        curve = baseline + bm._exp_uptake(t_post, fit['A'], fit['tau'])
                        ax.plot(t_post, curve, ls=STYLE_DEFAULT, color=color, lw=1.5)
                        y_pos = 0.9 if region_name == 'Body' else (0.8 if region_name == 'Protrusion' else 0.7)
                        ax.text(0.05, y_pos, f"{region_name[0]}: τ={fit['tau']:.1f}",
                                transform=ax.transAxes, fontsize=7, color=color)

                _plot_fit('Body_Normalized_dF_F0', 'Body')
                _plot_fit('Protrusion_Normalized_dF_F0', 'Protrusion')
                _plot_fit('Total_Normalized_dF_F0', 'Total')

            ax.axvline(0, color='black', linestyle=':', linewidth=0.8)
        for j in range(len(sorted_ids), len(axes)): axes[j].axis('off')
        plt.savefig(output_dir / f"Fit_Quality_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()


def plot_recoil_fits_multipanel(grouped_data, output_dir: Path):
    """Multipanel recoil fit grid. Only meaningful for EP data."""
    logger.info("Generating Plot: Multipanel Recoil Fits...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        # Skip ASP groups — recoil is an EP concept
        if traps[0].metadata.condition_type == "ASP":
            continue

        cond_label = get_cond_label(traps[0].metadata)
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        if n == 0: continue
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        _add_global_legend(fig, mode='recoil')
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Protrusion Length (µm)', va='center', rotation='vertical', fontsize=14)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            for trap in trap_list:
                pd_data = trap.protrusion_data
                if 'Time_s' not in pd_data: continue
                time_raw = pd_data['Time_s']; length = pd_data.get('Protrusion_Length_um', np.array([]))
                if len(time_raw) < 5 or len(length) < 5: continue
                at = align_time_to_pulse(time_raw, trap.metadata.pulse_frame, trap.metadata.condition_type)
                
                ax.plot(at[(at>=-10)&(at<=10)], length[(at>=-10)&(at<=10)], 'o', c='gray', ms=2, alpha=0.4)
                
                mp = (at>=-5)&(at<0)&(length>0)
                if np.sum(mp)>1: 
                    s, i_ = calculate_slope(at[mp], length[mp])
                    ax.plot(at[mp], s*at[mp]+i_, c=COLOR_FIT_PRE, lw=1.5)
                    ax.text(0.05, 0.9, f"Pre:{s:.2f}", transform=ax.transAxes, fontsize=7, color=COLOR_FIT_PRE, fontweight='bold')
                
                mpo = (at>=0)&(at<=2)&(length>0)
                if np.sum(mpo)>1: 
                    s, i_ = calculate_slope(at[mpo], length[mpo])
                    ax.plot(at[mpo], s*at[mpo]+i_, c=COLOR_FIT_POST, lw=1.5)
                    ax.text(0.05, 0.8, f"Post:{s:.2f}", transform=ax.transAxes, fontsize=7, color=COLOR_FIT_POST, fontweight='bold')
            
            ax.axvline(0, color='black', linestyle=':', linewidth=1)
        for j in range(len(sorted_ids), len(axes)): axes[j].axis('off')
        plt.savefig(output_dir / f"Recoil_Fits_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()

def plot_uptake_traces_multipanel(grouped_data, output_dir: Path):
    logger.info("Generating Plot: Multipanel Uptake Traces...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        if n == 0: continue
        
        global_max_y = 0.0
        for trap in traps:
            ud = trap.uptake_data
            if 'Time_s' not in ud: continue
            for col in ['Body_Normalized_dF_F0', 'Protrusion_Normalized_dF_F0', 'Total_Normalized_dF_F0']:
                if col in ud and len(ud[col]) > 0:
                    current_max = np.nanmax(ud[col])
                    if current_max > global_max_y: global_max_y = current_max
        
        y_limit = global_max_y * 1.1 if global_max_y > 0 else 1.0

        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        _add_global_legend(fig, mode='uptake')
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=16)
        fig.text(0.02, 0.5, 'Normalized Intensity (dF/F0)', va='center', rotation='vertical', fontsize=16)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            ax.set_ylim(-0.1, y_limit)
            
            for trap in trap_list:
                ud = trap.uptake_data
                if 'Time_s' not in ud: continue
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame, trap.metadata.condition_type)
                
                if 'Body_Normalized_dF_F0' in ud: 
                    ax.plot(at, ud['Body_Normalized_dF_F0'], ls=STYLE_DEFAULT, color=PALETTE_REGION['Body'], lw=1.5, alpha=0.7)
                if 'Protrusion_Normalized_dF_F0' in ud: 
                    ax.plot(at, ud['Protrusion_Normalized_dF_F0'], ls=STYLE_DEFAULT, color=PALETTE_REGION['Protrusion'], lw=1.5, alpha=0.7)
                if 'Total_Normalized_dF_F0' in ud:
                    ax.plot(at, ud['Total_Normalized_dF_F0'], ls=STYLE_DEFAULT, color=PALETTE_REGION['Total'], lw=1.5, alpha=0.7)
                    
            ax.axvline(0, color='black', ls=':', lw=0.8)
            
        for j in range(len(sorted_ids), len(axes)): axes[j].axis('off')
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.savefig(output_dir / f"Uptake_Traces_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()


# =============================================================================
# CONDITION LABEL HELPERS (for mechanics_df-based plots)
# =============================================================================

def _cond_label_from_row(row: pd.Series) -> str:
    """Build condition label from a mechanics_df row."""
    # Use pre-built Condition column if available (FIX #11)
    if 'Condition' in row.index and pd.notna(row.get('Condition')):
        return row['Condition']
    if row.get('Condition_Type') == 'ASP':
        return f"{row['Cell_Type']}_{row['Treatment']}_{row['Pressure_Pa']}Pa_ASP"
    return (f"{row['Cell_Type']}_{row['Treatment']}_"
            f"{row['Pressure_Pa']}Pa_{row['Voltage_V']}V_{row['Duration_label']}")


# =============================================================================
# 8. VISCOELASTIC PLOTS
# =============================================================================

def plot_viscoelastic_fits_multipanel(grouped_data: Dict, output_dir: Path, r_eff: float, C: float = 1.0) -> None:
    logger.info("Generating Plot: Multipanel Viscoelastic Fits (ASP only)...")
    import bulk_mechanics as bm
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        meta = traps[0].metadata
        if meta.condition_type != "ASP": continue
        
        cond_label = get_cond_label(meta)
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids)
        if n == 0: continue

        # Compute common duration for this group (same logic as mechanics)
        common_dur = bm.compute_common_duration(traps)

        cols = 5
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        handles = [
            mlines.Line2D([], [], color='black', marker='o', lw=0, label='Cleaned Data', alpha=0.4),
            mlines.Line2D([], [], color='#1f77b4', lw=2, label='Kelvin-Voigt'),
            mlines.Line2D([], [], color='#ff7f0e', lw=2, label='Jeffreys'),
            mlines.Line2D([], [], color='#2ca02c', lw=2, label='Burgers')
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=12)
        dur_note = f"  [window: {common_dur:.0f} s]" if common_dur else ""
        fig.text(0.5, 0.01, f'Time (s){dur_note}', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Protrusion Length (µm)', va='center', rotation='vertical', fontsize=14)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            trap_list = trap_groups[tid]
            
            for trap in trap_list:
                pd_data = trap.protrusion_data
                if 'Time_s' not in pd_data or 'Protrusion_Length_um' not in pd_data: continue
                
                t_raw = pd_data['Time_s']
                l_raw = pd_data['Protrusion_Length_um']

                # Apply common duration window (matching mechanics pipeline)
                if common_dur is not None:
                    t_from_entry = t_raw - t_raw[0]
                    win_mask = t_from_entry <= common_dur
                    t_raw = t_raw[win_mask]
                    l_raw = l_raw[win_mask]

                t_clean, l_clean = bm._clean_trace(t_raw, l_raw)
                if len(t_clean) < 5: continue
                t_zeroed = t_clean - t_clean[0]
                
                ax.plot(t_zeroed, l_clean, 'o', c='black', ms=3, alpha=0.4)
                visco = bm.fit_viscoelastic(t_zeroed, l_clean, r_eff, meta.pressure, C, n_starts=3, min_points=5)
                if not visco['best_model']: continue
                
                t_smooth = np.linspace(0, t_zeroed[-1], 200)
                models_info = visco['all_models']
                
                if 'Kelvin-Voigt' in models_info and models_info['Kelvin-Voigt']['params']:
                    p = models_info['Kelvin-Voigt']['params']
                    l_pred = bm._kelvin_voigt(t_smooth, r_eff, meta.pressure, C, p['E'], p['eta'])
                    ax.plot(t_smooth, l_pred, color='#1f77b4', lw=1.5, alpha=0.8)
                if 'Jeffreys' in models_info and models_info['Jeffreys']['params']:
                    p = models_info['Jeffreys']['params']
                    l_pred = bm._jeffreys(t_smooth, r_eff, meta.pressure, C, p['E'], p['eta1'], p['eta2'])
                    ax.plot(t_smooth, l_pred, color='#ff7f0e', lw=1.5, alpha=0.8)
                if 'Burgers' in models_info and models_info['Burgers']['params']:
                    p = models_info['Burgers']['params']
                    l_pred = bm._burgers(t_smooth, r_eff, meta.pressure, C, p['E1'], p['eta1'], p['E2'], p['eta2'])
                    ax.plot(t_smooth, l_pred, color='#2ca02c', lw=1.5, alpha=0.8)
                    
                best = visco['best_model']
                color_map = {'Kelvin-Voigt': '#1f77b4', 'Jeffreys': '#ff7f0e', 'Burgers': '#2ca02c'}
                ax.text(0.05, 0.9, f"Winner: {best}", transform=ax.transAxes, fontsize=8, fontweight='bold', color=color_map.get(best, 'black'))
                
        for j in range(len(sorted_ids), len(axes)): axes[j].axis('off')
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"Viscoelastic_Fits_Panel_{cond_label}.png", dpi=PANEL_DPI)
        plt.close()


# =============================================================================
# 9. ASP-SPECIFIC PLOTS  (reduced grouping: CellType × Treatment × Pressure)
# =============================================================================

def _asp_category_label(meta: bfh.ExperimentMetadata) -> str:
    return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa"


def _asp_category_label_from_row(row: pd.Series) -> str:
    return f"{row['Cell_Type']}_{row['Treatment']}_{row['Pressure_Pa']}Pa"


def plot_asp_best_fit_multipanel(
        all_grouped_data: Dict, output_dir: Path,
        r_eff: float, C: float = 1.0) -> None:
    """
    Multipanel grid for ASP cells: cleaned data + BEST viscoelastic model only.
    Applies common time window for comparable fits.
    """
    import bulk_mechanics as bm

    logger.info("Generating Plot: ASP Best-Fit Multipanel (reduced grouping)...")

    asp_pools: Dict[str, List[bfh.TrapData]] = {}
    for key, traps in all_grouped_data.items():
        if not traps: continue
        meta = traps[0].metadata
        if meta.condition_type != "ASP": continue
        cat = _asp_category_label(meta)
        if cat not in asp_pools: asp_pools[cat] = []
        asp_pools[cat].extend(traps)

    if not asp_pools:
        logger.info("  No ASP data found — skipping.")
        return

    model_color = {
        'Kelvin-Voigt': '#1f77b4',
        'Jeffreys':     '#ff7f0e',
        'Burgers':      '#2ca02c',
    }

    for cat_label in sorted(asp_pools.keys()):
        trap_list_all = asp_pools[cat_label]

        # Compute common duration for the pool
        common_dur = bm.compute_common_duration(trap_list_all)

        trap_groups = _group_by_trap_id(trap_list_all)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids)
        if n == 0: continue

        cols = 5
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()

        handles = [
            mlines.Line2D([], [], color='black', marker='o', lw=0, label='Data', alpha=0.4, ms=4),
            mlines.Line2D([], [], color=model_color['Kelvin-Voigt'], lw=2, label='Kelvin-Voigt'),
            mlines.Line2D([], [], color=model_color['Jeffreys'], lw=2, label='Jeffreys'),
            mlines.Line2D([], [], color=model_color['Burgers'], lw=2, label='Burgers'),
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=12)
        dur_note = f"  [window: {common_dur:.0f} s]" if common_dur else ""
        fig.text(0.5, 0.01, f'Time from entry (s){dur_note}', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Protrusion Length (µm)', va='center', rotation='vertical', fontsize=14)

        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')

            for trap in trap_groups[tid]:
                pd_data = trap.protrusion_data
                if ('Time_s' not in pd_data or 'Protrusion_Length_um' not in pd_data):
                    continue

                t_raw = pd_data['Time_s']
                l_raw = pd_data['Protrusion_Length_um']

                # Apply common duration window
                if common_dur is not None:
                    t_from_entry = t_raw - t_raw[0]
                    win_mask = t_from_entry <= common_dur
                    t_raw = t_raw[win_mask]
                    l_raw = l_raw[win_mask]

                t_clean, l_clean = bm._clean_trace(t_raw, l_raw)
                if len(t_clean) < 15: continue
                t_zeroed = t_clean - t_clean[0]

                ax.plot(t_zeroed, l_clean, 'o', c='black', ms=2, alpha=0.35)

                pressure = trap.metadata.pressure
                visco = bm.fit_viscoelastic(
                    t_zeroed, l_clean, r_eff, pressure, C,
                    n_starts=3, min_points=10)

                best = visco['best_model']
                if best is None:
                    ax.text(0.05, 0.9, "Fit failed", transform=ax.transAxes, fontsize=7, color='red')
                    continue

                bp_params = visco['best_params']
                t_smooth = np.linspace(0, t_zeroed[-1], 200)
                color = model_color.get(best, 'gray')

                if best == 'Kelvin-Voigt':
                    l_pred = bm._kelvin_voigt(t_smooth, r_eff, pressure, C, bp_params['E'], bp_params['eta'])
                elif best == 'Jeffreys':
                    l_pred = bm._jeffreys(t_smooth, r_eff, pressure, C, bp_params['E'], bp_params['eta1'], bp_params['eta2'])
                elif best == 'Burgers':
                    l_pred = bm._burgers(t_smooth, r_eff, pressure, C, bp_params['E1'], bp_params['eta1'], bp_params['E2'], bp_params['eta2'])
                else:
                    continue

                ax.plot(t_smooth, l_pred, color=color, lw=2, alpha=0.85)
                r2_str = f"R²={visco['best_r2']:.2f}" if visco['best_r2'] is not None else ""
                ax.text(0.05, 0.9, f"{best}  {r2_str}", transform=ax.transAxes, fontsize=7, fontweight='bold', color=color)

        for j in range(n, len(axes)): axes[j].axis('off')
        plt.suptitle(f"ASP Best-Fit — {cat_label}", fontweight='bold', fontsize=13, y=1.01)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"ASP_BestFit_Panel_{cat_label}.png", dpi=PANEL_DPI, bbox_inches='tight')
        plt.close()


def plot_asp_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Boxplots of viscoelastic parameters for ASP cells, grouped by the reduced
    key (CellType × Treatment × Pressure). Points coloured by winning model.
    Also produces a model selection frequency bar chart.
    """
    logger.info("Generating: ASP Parameter Boxplots (reduced grouping)...")

    df = mechanics_df[
        (mechanics_df['Condition_Type'] == 'ASP') &
        mechanics_df['Best_Model'].notna() &
        mechanics_df['E_Pa'].notna()
    ].copy()

    if df.empty:
        logger.info("  No fitted ASP data — skipping.")
        return

    df['Category'] = df.apply(_asp_category_label_from_row, axis=1)
    sorted_cats = sorted(df['Category'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    panels = [
        ('E_Pa',      'E / E₂ (Pa)',  'KV Spring Modulus (E/E₂)'),
        ('E1_Pa',     'E₁ (Pa)',      'Maxwell Spring E₁\n(Burgers only)'),
        ('Tau_s',     'τ (s)',        'Creep Time Constant τ'),
        ('eta1_Pa_s', 'η₁ (Pa·s)',    'Parallel Viscosity η₁'),
        ('eta2_Pa_s', 'η₂ (Pa·s)',    'Flow Viscosity η₂\n(Jeffreys / Burgers)'),
    ]

    for i, ax in enumerate(axes):
        if i >= len(panels):
            ax.set_visible(False)
            continue
            
        col, ylabel, title = panels[i]
        sub = df[df[col].notna()]
        if sub.empty:
            ax.set_visible(False)
            continue
            
        sns.boxplot(data=sub, x='Category', y=col, order=sorted_cats, ax=ax, showfliers=False, color='lightgray')
        sns.stripplot(data=sub, x='Category', y=col, order=sorted_cats,
                      hue='Best_Model',
                      palette={'Kelvin-Voigt': '#1f77b4', 'Jeffreys': '#ff7f0e', 'Burgers': '#2ca02c'},
                      dodge=False, alpha=0.6, ax=ax, size=5)
                      
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        
        if i != 0:
            legend = ax.get_legend()
            if legend: legend.remove()

    plt.suptitle(
        "Viscoelastic Parameters — ASP Cells\n"
        "(grouped by Cell Type × Treatment × Pressure)",
        fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "ASP_Parameter_Boxplots.png", dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

    # --- Model selection frequency bar chart ---
    model_counts = (df.groupby(['Category', 'Best_Model'])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(sorted_cats, fill_value=0))
    model_colors = {"Kelvin-Voigt": "#1f77b4", "Jeffreys": "#ff7f0e", "Burgers": "#2ca02c"}
    cols_present = [c for c in ["Kelvin-Voigt", "Jeffreys", "Burgers"] if c in model_counts.columns]

    model_counts[cols_present].plot(
        kind='bar', stacked=True,
        figsize=(max(6, len(sorted_cats) * 1.5), 5),
        color=[model_colors[c] for c in cols_present])
    plt.title("Best Model Selection Frequency (ASP)", fontweight='bold')
    plt.ylabel("Number of Traps")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "ASP_Model_Selection_Frequency.png", dpi=SAVE_DPI)
    plt.close()


# FIX #6/#7: Removed the duplicate model-selection frequency bar chart and
# the duplicate parameter boxplots that existed in the old
# plot_viscoelastic_parameters(). The ASP parameter boxplots above (reduced
# grouping) are the correct version. For ASP cells, the full-condition
# grouping is identical to the reduced grouping (voltage/duration are always
# 0), so the old function was producing a duplicate chart.


# =============================================================================
# 10. MODEL-INDEPENDENT PLOTS
# =============================================================================

def plot_model_independent_fits_multipanel(
        grouped_data: Dict, output_dir: Path) -> None:
    """
    NEW: Multipanel grid showing cleaned protrusion data with BOTH linear
    and power-law model curves overlaid. Labels the BIC winner on each panel.

    Works for both ASP and EP (EP uses pre-pulse window).
    """
    import bulk_mechanics as bm

    logger.info("Generating Plot: Model-Independent Fits Multipanel...")

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        meta = traps[0].metadata

        # For ASP groups, compute common duration window
        common_dur = None
        if meta.condition_type == "ASP":
            common_dur = bm.compute_common_duration(traps)

        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids)
        if n == 0: continue

        cols = 5
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()

        handles = [
            mlines.Line2D([], [], color='black', marker='o', lw=0, label='Data', alpha=0.4, ms=4),
            mlines.Line2D([], [], color=MI_MODEL_PALETTE['Linear'], lw=2, label='Linear'),
            mlines.Line2D([], [], color=MI_MODEL_PALETTE['Power-Law'], lw=2, label='Power-Law'),
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=3, frameon=False, fontsize=12)
        fig.text(0.5, 0.01, 'Time (s)', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Protrusion Length (µm)', va='center', rotation='vertical', fontsize=14)

        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')

            for trap in trap_groups[tid]:
                pd_data = trap.protrusion_data
                if 'Time_s' not in pd_data or 'Protrusion_Length_um' not in pd_data:
                    continue

                t_raw = pd_data['Time_s'].copy()
                l_raw = pd_data['Protrusion_Length_um'].copy()

                # --- Extract the fitting window ---
                if meta.condition_type == "ASP":
                    # Apply common duration window
                    if common_dur is not None:
                        t_from_entry = t_raw - t_raw[0]
                        win_mask = t_from_entry <= common_dur
                        t_raw = t_raw[win_mask]
                        l_raw = l_raw[win_mask]
                    t_fit_raw = t_raw - t_raw[0]
                    l_fit_raw = l_raw
                else:
                    # EP: use pre-pulse window
                    pf = trap.metadata.pulse_frame
                    if not (0 < pf < len(t_raw)):
                        continue
                    t_aligned = t_raw - t_raw[pf]
                    pre_mask = (t_aligned < 0) & (l_raw > 0)
                    if np.sum(pre_mask) < 5:
                        continue
                    t_fit_raw = t_aligned[pre_mask]
                    t_fit_raw = t_fit_raw - t_fit_raw[0]
                    l_fit_raw = l_raw[pre_mask]

                t_clean, l_clean = bm._clean_trace(t_fit_raw, l_fit_raw)
                if len(t_clean) < 5:
                    continue

                ax.plot(t_clean, l_clean, 'o', c='black', ms=2, alpha=0.35)

                mi_fit = bm.fit_model_independent(t_clean, l_clean)
                t_smooth = np.linspace(t_clean[0], t_clean[-1], 200)

                # Plot linear fit
                if mi_fit['linear']:
                    p = mi_fit['linear']['params']
                    ax.plot(t_smooth, bm._linear(t_smooth, p['slope'], p['intercept']),
                            color=MI_MODEL_PALETTE['Linear'], lw=1.5, alpha=0.8)

                # Plot power-law fit
                if mi_fit['power_law']:
                    p = mi_fit['power_law']['params']
                    ax.plot(t_smooth, bm._power_law(t_smooth, p['a'], p['exponent_b']),
                            color=MI_MODEL_PALETTE['Power-Law'], lw=1.5, alpha=0.8)

                # Label BIC winner
                winner = mi_fit.get('best_model')
                if winner:
                    color = MI_MODEL_PALETTE.get(winner, 'black')
                    ax.text(0.05, 0.9, f"BIC: {winner}",
                            transform=ax.transAxes, fontsize=7,
                            fontweight='bold', color=color)

        for j in range(n, len(axes)): axes[j].axis('off')
        plt.suptitle(f"Model-Independent Fits — {cond_label}", fontweight='bold', fontsize=13, y=1.01)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"MI_Fits_Panel_{cond_label}.png", dpi=PANEL_DPI, bbox_inches='tight')
        plt.close()


def plot_model_independent_comparison(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    UPDATED: Model-independent fit parameter comparison.

    Produces:
      1. Two-panel boxplot (Linear Slope, Power-Law exponent) with points
         coloured by the BIC-winning model (Linear or Power-Law).
      2. MI model selection frequency bar chart.
      3. EP pre-vs-post slope comparison (unchanged from original).
    """
    logger.info("Generating: Model-Independent Fit Comparison...")
    if mechanics_df is None or mechanics_df.empty: return

    df = mechanics_df.copy()
    df['Cond'] = df.apply(_cond_label_from_row, axis=1)
    sorted_conds = sorted(df['Cond'].unique())

    df_lin = df[df['Linear_Slope'].notna()]
    df_pl  = df[df['PL_b'].notna()]

    if df_lin.empty and df_pl.empty: return

    # --- Panel 1: Boxplots with points coloured by MI winner ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if not df_lin.empty:
        sns.boxplot(data=df_lin, x='Cond', y='Linear_Slope', order=sorted_conds,
                    showfliers=False, ax=axes[0], color='lightgray')
        if 'MI_Best_Model' in df_lin.columns:
            sns.stripplot(data=df_lin, x='Cond', y='Linear_Slope', order=sorted_conds,
                          hue='MI_Best_Model', palette=MI_MODEL_PALETTE,
                          dodge=False, alpha=0.6, ax=axes[0], size=5)
        else:
            sns.stripplot(data=df_lin, x='Cond', y='Linear_Slope', order=sorted_conds,
                          color='black', alpha=0.5, ax=axes[0], size=5)
        axes[0].axhline(0, color='black', lw=0.8, ls=':')
    axes[0].set_title("Linear Slope (aspiration phase)", fontweight='bold')
    axes[0].set_ylabel("Slope (µm/s)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)

    if not df_pl.empty:
        sns.boxplot(data=df_pl, x='Cond', y='PL_b', order=sorted_conds,
                    showfliers=False, ax=axes[1], color='lightgray')
        if 'MI_Best_Model' in df_pl.columns:
            sns.stripplot(data=df_pl, x='Cond', y='PL_b', order=sorted_conds,
                          hue='MI_Best_Model', palette=MI_MODEL_PALETTE,
                          dodge=False, alpha=0.6, ax=axes[1], size=5)
        else:
            sns.stripplot(data=df_pl, x='Cond', y='PL_b', order=sorted_conds,
                          color='black', alpha=0.5, ax=axes[1], size=5)
        axes[1].axhline(0.5, color='gray', lw=0.8, ls='--', label='b=0.5 (KV prediction)')
        axes[1].legend(fontsize=9)
    axes[1].set_title("Power-Law Exponent b  (L = a·tᵇ)", fontweight='bold')
    axes[1].set_ylabel("Exponent b")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle("Model-Independent Fit Parameters (ASP & EP)", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "Model_Independent_Comparison.png", dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

    # --- Panel 2: MI model selection frequency bar chart ---
    if 'MI_Best_Model' in df.columns:
        df_mi = df[df['MI_Best_Model'].notna()]
        if not df_mi.empty:
            mi_counts = (df_mi.groupby(['Cond', 'MI_Best_Model'])
                         .size()
                         .unstack(fill_value=0)
                         .reindex(sorted_conds, fill_value=0))
            mi_cols = [c for c in ['Linear', 'Power-Law'] if c in mi_counts.columns]
            if mi_cols:
                mi_counts[mi_cols].plot(
                    kind='bar', stacked=True,
                    figsize=(max(6, len(sorted_conds) * 1.5), 5),
                    color=[MI_MODEL_PALETTE[c] for c in mi_cols])
                plt.title("MI Best Model Selection Frequency", fontweight='bold')
                plt.ylabel("Number of Traps")
                plt.xlabel("")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_dir / "MI_Model_Selection_Frequency.png", dpi=SAVE_DPI)
                plt.close()

    # --- Panel 3: EP pre-vs-post slope comparison (unchanged) ---
    df_ep = df[(df['Condition_Type'] == 'EP') &
               (df['Pre_Pulse_Slope'].notna() | df['Post_Pulse_Slope'].notna())]
    if not df_ep.empty:
        df_melt = df_ep.melt(
            id_vars=['Cond'], value_vars=['Pre_Pulse_Slope', 'Post_Pulse_Slope'],
            var_name='Phase', value_name='Slope').dropna(subset=['Slope'])
        phase_palette = {'Pre_Pulse_Slope': COLOR_FIT_PRE, 'Post_Pulse_Slope': COLOR_FIT_POST}
        ep_conds = sorted(df_melt['Cond'].unique())

        plt.figure(figsize=(max(8, len(ep_conds) * 2), 6))
        sns.boxplot(data=df_melt, x='Cond', y='Slope', hue='Phase',
                    palette=phase_palette, order=ep_conds, showfliers=False)
        sns.stripplot(data=df_melt, x='Cond', y='Slope', hue='Phase',
                      dodge=True, palette='dark:black', alpha=0.5,
                      legend=False, order=ep_conds, size=5)
        plt.axhline(0, color='black', lw=0.8, ls=':')
        plt.title("Pre vs Post-Pulse Protrusion Slope (EP Cells)", fontweight='bold')
        plt.ylabel("Slope (µm/s)")
        plt.xlabel("")
        plt.xticks(rotation=45, ha='right')
        handles = [
            mlines.Line2D([], [], color=COLOR_FIT_PRE, lw=2, label='Pre-Pulse'),
            mlines.Line2D([], [], color=COLOR_FIT_POST, lw=2, label='Post-Pulse')
        ]
        plt.legend(handles=handles, title='Phase')
        plt.tight_layout()
        plt.savefig(output_dir / "EP_PrePost_Slope_Comparison.png", dpi=SAVE_DPI)
        plt.close()


def plot_uptake_tau_boxplot(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating: Uptake Tau Boxplots...")
    tau_col_map = {'Uptake_Body_tau': 'Body', 'Uptake_Prot_tau': 'Protrusion', 'Uptake_Total_tau': 'Total'}
    df = mechanics_df.copy()
    df['Cond'] = df.apply(_cond_label_from_row, axis=1)

    df_melt = df.melt(id_vars=['Cond', 'Condition_Type'],
                      value_vars=list(tau_col_map.keys()),
                      var_name='Region_raw', value_name='Uptake_Tau_s').dropna(subset=['Uptake_Tau_s'])
    df_melt['Region'] = df_melt['Region_raw'].map(tau_col_map)
    med_tau = df_melt['Uptake_Tau_s'].median()
    df_melt = df_melt[df_melt['Uptake_Tau_s'] <= med_tau * 20]

    if df_melt.empty: return
    sorted_conds = sorted(df_melt['Cond'].unique())

    plt.figure(figsize=(max(10, len(sorted_conds) * 2.5), 6))
    sns.boxplot(data=df_melt, x='Cond', y='Uptake_Tau_s', hue='Region',
                palette=PALETTE_REGION, order=sorted_conds, showfliers=False)
    sns.stripplot(data=df_melt, x='Cond', y='Uptake_Tau_s', hue='Region',
                  dodge=True, palette='dark:black', alpha=0.4, legend=False,
                  order=sorted_conds, size=4)
    plt.title("Uptake Time Constant τ by Condition", fontweight='bold')
    plt.ylabel("τ (s)")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Region', loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "Uptake_Tau_Boxplot.png", dpi=SAVE_DPI)
    plt.close()

    ctype_palette = {'ASP': '#1f77b4', 'EP': '#ff7f0e'}
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_melt, x='Region', y='Uptake_Tau_s', hue='Condition_Type',
                palette=ctype_palette, order=['Body', 'Protrusion', 'Total'], showfliers=False)
    sns.stripplot(data=df_melt, x='Region', y='Uptake_Tau_s', hue='Condition_Type',
                  dodge=True, palette='dark:black', alpha=0.4, legend=False,
                  order=['Body', 'Protrusion', 'Total'], size=4)
    plt.title("Uptake τ: ASP vs EP Comparison", fontweight='bold')
    plt.ylabel("τ (s)")
    plt.xlabel("Region")
    plt.legend(title='Condition Type', loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "Uptake_Tau_ASP_vs_EP.png", dpi=SAVE_DPI)
    plt.close()

def plot_spearman_correlation(df_scalars: pd.DataFrame, output_dir: Path) -> None:
    cols_to_drop = ['Condition', 'Trap_ID']
    df_numeric = df_scalars.drop(columns=[c for c in cols_to_drop if c in df_scalars.columns])
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]
    if df_numeric.shape[1] < 2: return
        
    corr_matrix = df_numeric.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Spearman ρ"}, ax=ax)
    
    ax.set_title("Spearman Correlation of Morphological and Fluorescent Metrics",
                 pad=20, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "spearman_correlation_heatmap.png", dpi=300)
    plt.close(fig)
    corr_matrix.to_csv(output_dir / "spearman_correlation_matrix.csv")

def plot_slope_comparison(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """Three-panel slope comparison: full-trace slope, EP pre-vs-post, and delta."""
    logger.info("Generating: Slope Comparison (ASP vs EP)...")
    if mechanics_df is None or mechanics_df.empty: return

    df = mechanics_df.copy()
    df['Cond'] = df.apply(_cond_label_from_row, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Empirical Creep Slope Comparison", fontsize=14, fontweight='bold')
    palette = {'ASP': '#1f77b4', 'EP': '#d62728'}

    ax = axes[0]
    valid = df['Linear_Slope'].notna()
    if valid.any():
        sns.boxplot(data=df[valid], x='Cond', y='Linear_Slope', hue='Condition_Type',
                    palette=palette, showfliers=False, ax=ax)
        sns.stripplot(data=df[valid], x='Cond', y='Linear_Slope', hue='Condition_Type',
                      palette=palette, dodge=True, alpha=0.6, jitter=True, ax=ax, legend=False)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("Full-trace creep slope")
    ax.set_ylabel("Slope [µm/s]")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    df_ep = df[df['Condition_Type'] == 'EP'].copy()
    if not df_ep.empty:
        pre_post_records = []
        for _, row in df_ep.iterrows():
            if pd.notna(row.get('Pre_Pulse_Slope')):
                pre_post_records.append({'Condition': row['Cond'], 'Window': 'Pre-pulse', 'Slope_umps': row['Pre_Pulse_Slope']})
            if pd.notna(row.get('Post_Pulse_Slope')):
                pre_post_records.append({'Condition': row['Cond'], 'Window': 'Post-pulse', 'Slope_umps': row['Post_Pulse_Slope']})
        if pre_post_records:
            df_pp = pd.DataFrame(pre_post_records)
            sns.boxplot(data=df_pp, x='Condition', y='Slope_umps', hue='Window',
                        palette={'Pre-pulse': '#ff7f0e', 'Post-pulse': '#9467bd'},
                        showfliers=False, ax=ax)
            sns.stripplot(data=df_pp, x='Condition', y='Slope_umps', hue='Window',
                          palette={'Pre-pulse': '#ff7f0e', 'Post-pulse': '#9467bd'},
                          dodge=True, alpha=0.6, jitter=True, ax=ax, legend=False)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("EP: pre-pulse vs post-pulse slope")
    ax.set_ylabel("Slope [µm/s]")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)

    ax = axes[2]
    if not df_ep.empty and 'Pre_Pulse_Slope' in df_ep.columns:
        df_ep = df_ep.copy()
        df_ep['Delta_Slope'] = df_ep['Post_Pulse_Slope'] - df_ep['Pre_Pulse_Slope']
        valid_d = df_ep['Delta_Slope'].notna()
        if valid_d.any():
            sns.boxplot(data=df_ep[valid_d], x='Cond', y='Delta_Slope', color='#2ca02c',
                        showfliers=False, ax=ax)
            sns.stripplot(data=df_ep[valid_d], x='Cond', y='Delta_Slope', color='black',
                          alpha=0.6, jitter=True, ax=ax)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("EP: Δslope (post − pre)")
    ax.set_ylabel("ΔSlope [µm/s]")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "Slope_Comparison_ASP_vs_EP.png", dpi=SAVE_DPI)
    plt.close()


# FIX #13: Added Protrusion panel to the ASP-vs-EP uptake overlay.
def plot_uptake_asp_vs_ep(grouped_data: Dict, output_dir: Path) -> None:
    logger.info("Generating: Uptake Comparison ASP vs EP...")
    base_conditions: Dict[Tuple, Dict[str, List[bfh.TrapData]]] = {}

    for (cell_type, treatment, pressure_pa, voltage, duration_ms), traps in grouped_data.items():
        base_key = (cell_type, treatment, pressure_pa)
        if base_key not in base_conditions: base_conditions[base_key] = {'ASP': [], 'EP': []}
        ctype = traps[0].metadata.condition_type if traps else None
        if ctype in ('ASP', 'EP'): base_conditions[base_key][ctype].extend(traps)

    for (cell_type, treatment, pressure_pa), groups in base_conditions.items():
        asp_traps = groups['ASP']; ep_traps  = groups['EP']
        if not asp_traps and not ep_traps: continue

        # 3 panels: Body, Protrusion, Total
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(f"PI Dye Uptake: ASP vs EP\n{cell_type} | {treatment} | {pressure_pa} Pa",
                     fontsize=13, fontweight='bold')

        panel_defs = [
            (axes[0], 'Body_Normalized_dF_F0',       'Body'),
            (axes[1], 'Protrusion_Normalized_dF_F0',  'Protrusion'),
            (axes[2], 'Total_Normalized_dF_F0',       'Total'),
        ]

        for ax, col, region_label in panel_defs:
            for ctype, trap_list, color in [('ASP', asp_traps, '#1f77b4'), ('EP',  ep_traps,  '#d62728')]:
                if not trap_list: continue

                times_list, data_list = [], []
                for trap in trap_list:
                    ud = trap.uptake_data
                    if 'Time_s' not in ud or col not in ud: continue
                    pf = trap.metadata.pulse_frame
                    t_raw = ud['Time_s']
                    t_aligned = align_time_to_pulse(t_raw, pf, trap.metadata.condition_type)
                    times_list.append(t_aligned)
                    data_list.append(ud[col])

                if not times_list: continue
                common_t, matrix = interpolate_to_common_time(times_list, data_list, dt=1.75)
                if len(common_t) == 0: continue

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mean = np.nanmean(matrix, axis=0)
                    sd   = np.nanstd(matrix, axis=0)

                n = len(trap_list)
                ax.plot(common_t, mean, color=color, lw=2, label=f"{ctype} (n={n})")
                ax.fill_between(common_t, mean - sd, mean + sd, color=color, alpha=0.15, edgecolor=None)

            ax.axvline(0, color='black', linestyle=':', linewidth=0.8, label='Pulse')
            ax.set_xlabel("Time from pulse [s]")
            ax.set_ylabel(f"{region_label} dF/F₀")
            ax.set_title(f"{region_label} Uptake")
            ax.legend(fontsize=9)

        plt.tight_layout()
        safe_label = f"{cell_type}_{treatment}_{pressure_pa}Pa"
        plt.savefig(output_dir / f"Uptake_ASP_vs_EP_{safe_label}.png", dpi=SAVE_DPI)
        plt.close()