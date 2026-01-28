# -*- coding: utf-8 -*-
"""
Plotting Module for Bulk MFAE Analysis.
=============================================================================
This module handles all data visualization for the pipeline.
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

import bulk_file_handling as bfh

logger = logging.getLogger(__name__)

# =============================================================================
# 1. VISUAL CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=0.8)

PALETTE_STATUS = {"Intact": "#1f77b4", "Ruptured": "#d62728"}
PALETTE_REGION = {"Body": "#1f77b4", "Protrusion": "#ff7f0e"}
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

def interpolate_to_common_time(time_series_list, data_series_list, dt=0.5):
    valid_pairs = [(t, y) for t, y in zip(time_series_list, data_series_list) if len(t) >= 2]
    if not valid_pairs: return np.array([]), np.array([])
    t_min = min(t[0] for t, _ in valid_pairs)
    t_max = max(t[-1] for t, _ in valid_pairs)
    common_time = np.arange(np.floor(t_min), np.ceil(t_max), dt)
    rows = [interp1d(t, y, bounds_error=False, fill_value=np.nan)(common_time) for t, y in valid_pairs]
    return common_time, np.array(rows)

# =============================================================================
# 3. PLOTTING FUNCTIONS
# =============================================================================

def plot_max_protrusion_distribution(grouped_data, output_dir: Path):
    logger.info("Generating Plot 1: Max Protrusion...")
    records = []
    # Loop over sorted keys to keep plots ordered: 5us -> 5ms -> 5s
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        # USE LABEL FROM METADATA
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
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
    plt.title("Max Protrusion Length")
    plt.savefig(output_dir / "Max_Protrusion_Split.png", dpi=SAVE_DPI); plt.close()

def plot_per_trap_protrusion_distribution(grouped_data, output_dir: Path):
    logger.info("Generating Plot 2: Per-Trap Protrusion...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}" # Underscore for filename
        
        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            p_data = trap.protrusion_data
            if 'Protrusion_Length_um' in p_data:
                for val in p_data['Protrusion_Length_um']: records.append({'Trap': f"T{trap.trap_id}", 'Length_um': val, 'Status': status})
        if not records: continue
        df = pd.DataFrame(records)
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x='Trap', y='Length_um', hue='Status', palette=PALETTE_STATUS, dodge=False)
        plt.title(f"Protrusion Stability ({volt}V {label})")
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(output_dir / f"Protrusion_Dist_{cond_label}.png", dpi=SAVE_DPI); plt.close()

def plot_rupture_probability(grouped_data, output_dir: Path):
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
        summary.append({'Condition': f"{volt}V {label}", 'Rupture_Percent': (ruptured/total)*100, 'Count': total})
    if not summary: return
    df = pd.DataFrame(summary)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Condition', y='Rupture_Percent', color='darkred', alpha=0.7)
    for i, r in df.iterrows(): plt.text(i, r.Rupture_Percent+1, f"n={int(r.Count)}", ha="center")
    plt.title("Rupture Probability"); plt.ylim(0, 110)
    plt.savefig(output_dir / "Rupture_Probability.png", dpi=SAVE_DPI); plt.close()

def plot_uptake_dynamics(grouped_data, output_dir: Path):
    logger.info("Generating Plot 3: Uptake Dynamics...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
        data = {"Intact": {'t':[],'b':[],'p':[]}, "Ruptured": {'t':[],'b':[],'p':[]}}
        for trap in traps:
            status = determine_status(trap)
            if status not in data: continue
            ud = trap.uptake_data
            if 'Time_s' in ud and 'Body_Normalized_dF_F0' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                data[status]['t'].append(at)
                data[status]['b'].append(ud['Body_Normalized_dF_F0'])
                data[status]['p'].append(ud['Protrusion_Normalized_dF_F0'])
        
        fig, ax = plt.subplots(figsize=(12, 7))
        has_data = False
        for s, c in PALETTE_STATUS.items():
            if not data[s]['t']: continue
            ct, bm = interpolate_to_common_time(data[s]['t'], data[s]['b'], 0.5)
            _, pm = interpolate_to_common_time(data[s]['t'], data[s]['p'], 0.5)
            if len(ct)==0: continue
            has_data = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bm_m = np.nanmean(bm, axis=0); bm_sem = np.nanstd(bm, axis=0)/np.sqrt(len(data[s]['t']))
                pm_m = np.nanmean(pm, axis=0); pm_sem = np.nanstd(pm, axis=0)/np.sqrt(len(data[s]['t']))
            ax.plot(ct, bm_m, c=c, ls='-', lw=2, label=f"{s} Body")
            ax.fill_between(ct, bm_m-bm_sem, bm_m+bm_sem, color=c, alpha=0.2)
            ax.plot(ct, pm_m, c=c, ls='--', lw=2, label=f"{s} Prot")
            ax.fill_between(ct, pm_m-pm_sem, pm_m+pm_sem, color=c, alpha=0.1)
        if not has_data: plt.close(); continue
        ax.axvline(0, c='k', ls=':'); ax.set_xlim(left=-5)
        ax.set_title(f"Dye Uptake Dynamics ({cond_label})")
        ax.legend(); plt.tight_layout()
        plt.savefig(output_dir / f"Uptake_Dynamics_Aligned_{cond_label}.png", dpi=SAVE_DPI); plt.close()

def plot_protrusion_recoil_velocity(grouped_data, output_dir: Path):
    logger.info("Generating Plot 6: Recoil Velocity...")
    records = []
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
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
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V {label}"
        
        for trap in traps:
            status = determine_status(trap)
            if status != "Intact": continue
            ud = trap.uptake_data
            if 'Time_s' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                if 'Body_Normalized_dF_F0' in ud:
                    _, tau, r2 = fit_uptake_curve(at, ud['Body_Normalized_dF_F0'])
                    if r2 > 0.5 and tau < 300: records.append({'Condition':cond_label, 'Tau':tau, 'Region':'Body'})
                if 'Protrusion_Normalized_dF_F0' in ud:
                    _, tau, r2 = fit_uptake_curve(at, ud['Protrusion_Normalized_dF_F0'])
                    if r2 > 0.5 and tau < 300: records.append({'Condition':cond_label, 'Tau':tau, 'Region':'Protrusion'})
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
    sns.scatterplot(data=df, x='Max_Length_um', y='Max_Uptake', hue='Status', style='Condition', palette=PALETTE_STATUS, s=100, alpha=0.8)
    plt.title("Correlation: Length vs Uptake"); plt.savefig(output_dir / "Correlation_Length_vs_Uptake.png", dpi=SAVE_DPI); plt.close()

def plot_per_trap_uptake_distribution(grouped_data, output_dir: Path):
    logger.info("Generating Plot 4: Body vs Protrusion Uptake...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            status = determine_status(trap)
            if status in ["Empty", "Unknown"]: continue
            ud = trap.uptake_data
            if 'Time_s' in ud:
                at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
                mask = at > 0
                for v in ud['Body_Normalized_dF_F0'][mask]: records.append({'Trap':f"T{trap.trap_id}", 'I':v, 'Region':'Body', 'Status':status})
                for v in ud['Protrusion_Normalized_dF_F0'][mask]: records.append({'Trap':f"T{trap.trap_id}", 'I':v, 'Region':'Protrusion', 'Status':status})
        if not records: continue
        df = pd.DataFrame(records)
        plt.figure(figsize=(16, 7))
        sns.boxplot(data=df, x='Trap', y='I', hue='Region', palette=PALETTE_REGION)
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(output_dir / f"Uptake_Dist_BodyVsProt_{cond_label}.png", dpi=SAVE_DPI); plt.close()

def plot_uptake_fits_multipanel(grouped_data, output_dir: Path):
    logger.info("Generating Plot 9: Multipanel Fits...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        traps_sorted = sorted(traps, key=lambda t: t.trap_id)
        n = len(traps_sorted); cols = 5; rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5*rows))
        axes = axes.flatten()
        for i, trap in enumerate(traps_sorted):
            ax = axes[i]; ud = trap.uptake_data; status = determine_status(trap)
            ax.set_title(f"Trap {trap.trap_id} ({status})", fontsize=10, fontweight='bold')
            if 'Time_s' not in ud: continue
            at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
            if 'Body_Normalized_dF_F0' in ud:
                y = ud['Body_Normalized_dF_F0']; ax.plot(at, y, '.', ms=3, c=PALETTE_REGION['Body'], alpha=0.4)
                A, tau, _ = fit_uptake_curve(at, y)
                if not np.isnan(tau): ax.plot(at[at>0], exponential_uptake_func(at[at>0], A, tau), c=PALETTE_REGION['Body'])
            if 'Protrusion_Normalized_dF_F0' in ud:
                y = ud['Protrusion_Normalized_dF_F0']; ax.plot(at, y, '.', ms=3, c=PALETTE_REGION['Protrusion'], alpha=0.4)
                A, tau, _ = fit_uptake_curve(at, y)
                if not np.isnan(tau): ax.plot(at[at>0], exponential_uptake_func(at[at>0], A, tau), c=PALETTE_REGION['Protrusion'])
            ax.axvline(0, c='k', ls=':', lw=0.8)
        for j in range(i+1, len(axes)): axes[j].axis('off')
        plt.tight_layout(); plt.savefig(output_dir / f"Fit_Quality_Panel_{cond_label}.png", dpi=PANEL_DPI); plt.close()

def plot_recoil_fits_multipanel(grouped_data, output_dir: Path):
    logger.info("Generating Plot 10: Multipanel Recoil...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        traps_sorted = sorted(traps, key=lambda t: t.trap_id)
        n = len(traps_sorted); cols = 5; rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5*rows))
        axes = axes.flatten()
        for i, trap in enumerate(traps_sorted):
            ax = axes[i]; status = determine_status(trap); pd = trap.protrusion_data
            ax.set_title(f"Trap {trap.trap_id} ({status})", fontsize=10, fontweight='bold')
            if 'Time_s' not in pd: continue
            at = align_time_to_pulse(pd['Time_s'], trap.metadata.pulse_frame)
            l = pd.get('Protrusion_Length_um', [])
            ax.plot(at[(at>=-10)&(at<=10)], l[(at>=-10)&(at<=10)], 'o', c='gray', ms=3, alpha=0.5)
            mp = (at>=-5)&(at<0); mpo = (at>=0)&(at<=2)
            if np.sum(mp)>1: s, i_ = calculate_slope(at[mp], l[mp]); ax.plot(at[mp], s*at[mp]+i_, c=COLOR_FIT_PRE, lw=2); ax.text(0.05, 0.9, f"{s:.2f}", transform=ax.transAxes, c=COLOR_FIT_PRE, fontsize=8)
            if np.sum(mpo)>1: s, i_ = calculate_slope(at[mpo], l[mpo]); ax.plot(at[mpo], s*at[mpo]+i_, c=COLOR_FIT_POST, lw=2); ax.text(0.05, 0.8, f"{s:.2f}", transform=ax.transAxes, c=COLOR_FIT_POST, fontsize=8)
            ax.axvline(0, c='k', ls=':', lw=1)
        for j in range(i+1, len(axes)): axes[j].axis('off')
        plt.tight_layout(); plt.savefig(output_dir / f"Recoil_Fits_Panel_{cond_label}.png", dpi=PANEL_DPI); plt.close()

def plot_uptake_traces_multipanel(grouped_data, output_dir: Path):
    logger.info("Generating Plot 11: Multipanel Traces...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        
        volt = traps[0].metadata.voltage
        label = traps[0].metadata.duration_label
        cond_label = f"{volt}V_{label}"
        
        traps_sorted = sorted(traps, key=lambda t: t.trap_id)
        n = len(traps_sorted); cols = 5; rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5*rows))
        axes = axes.flatten()
        for i, trap in enumerate(traps_sorted):
            ax = axes[i]; status = determine_status(trap); ud = trap.uptake_data
            ax.set_title(f"Trap {trap.trap_id} ({status})", fontsize=10, fontweight='bold')
            if 'Time_s' not in ud: continue
            at = align_time_to_pulse(ud['Time_s'], trap.metadata.pulse_frame)
            if 'Body_Normalized_dF_F0' in ud: ax.plot(at, ud['Body_Normalized_dF_F0'], c=PALETTE_REGION['Body'], lw=1.5, alpha=0.8)
            if 'Protrusion_Normalized_dF_F0' in ud: ax.plot(at, ud['Protrusion_Normalized_dF_F0'], c=PALETTE_REGION['Protrusion'], lw=1.5, alpha=0.8)
            ax.axvline(0, c='k', ls=':', lw=0.8)
        for j in range(i+1, len(axes)): axes[j].axis('off')
        plt.tight_layout(); plt.savefig(output_dir / f"Uptake_Traces_Panel_{cond_label}.png", dpi=PANEL_DPI); plt.close()