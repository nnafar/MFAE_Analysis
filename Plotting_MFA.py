# -*- coding: utf-8 -*-
"""
Centralized plotting module for MFA analysis.

ROLE IN PIPELINE:
This module generates all static figures (PNGs). It ensures consistent styling
(fonts, line widths, colors) across all outputs.
"""
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import cv2

import Utils_MFA as utils

if TYPE_CHECKING:
    from Fitting_MFA import FittingMFA

logger = logging.getLogger(__name__)

# Standard units for axis labels
UNITS = {'E': 'Pa', 'E1': 'Pa', 'E2': 'Pa', 'eta': 'Pa·s', 'eta1': 'Pa·s', 'eta2': 'Pa·s', 'm': 'μm/s', 'b': 'μm', 'a': 'μm/sᵇ', 'c': 'μm'}

# --- 1. DYE UPTAKE DASHBOARD ---
def plot_dye_uptake_dashboard(results: Dict[str, Any], trap_idx: int, output_dir: Path, params: Dict[str, Any], pipette_x: Optional[float] = None):
    """
    Generates the suite of Dye Uptake plots with Shaded SEM for ALL metrics including Min-Max.
    """
    utils.set_paper_style()
    output_dir = Path(output_dir)
    t = np.array(results['time_s'])
    if len(t) == 0: return

    colors = utils.MFA_COLORS
    pulse_frame = params.get('dye_uptake_parameters', {}).get('pulse_frame', 10) - 1
    pulse_time = t[min(pulse_frame, len(t)-1)]
    
    if pipette_x is None: pipette_x = results.get('pipette_x_px', 0)

    # --- Common Plotting Settings ---
    marker_style = 'o-'
    ms = 4

    # --- Helper: Calculate SEM ---
    def get_sem(std_key, count_key):
        if std_key in results and count_key in results:
            std = np.array(results[std_key])
            count = np.array(results[count_key])
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                sem = std / np.sqrt(count)
                sem[count == 0] = 0
            return sem
        return None

    # def plot_3panel_metric(fig_name: str, title: str, ylabel: str, suffix: str):
    #     """Helper to create a 1x3 horizontal panel plot with shared Y-axis."""
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        
    #     for ax in [ax1, ax2, ax3]:
    #         ax.axvline(pulse_time, color=colors['pulse'], linestyle='--', linewidth=2.5, label='Pulse')

    #     # --- Panel 1: Protrusion (and Tip) ---
    #     m_p = np.array(results.get(f'uptake_protrusion{suffix}', []))
    #     s_p = get_sem(f'uptake_protrusion{suffix}_std', 'count_protrusion')
    #     if len(m_p) > 0:
    #         ax1.plot(t, m_p, marker_style, color=colors['secondary'], linewidth=2, markersize=ms, label='Protrusion')
    #         if s_p is not None: 
    #             ax1.fill_between(t, m_p - s_p, m_p + s_p, color=colors['secondary'], alpha=0.25)
                
    #     m_tip = np.array(results.get(f'uptake_tip{suffix}', []))
    #     s_tip = get_sem(f'uptake_tip{suffix}_std', 'count_tip')
    #     if len(m_tip) > 0:
    #         ax1.plot(t, m_tip, marker_style, color=colors['quaternary'], linewidth=2, markersize=ms, label='Tip (Top 5%)')
    #         if s_tip is not None: 
    #             ax1.fill_between(t, m_tip - s_tip, m_tip + s_tip, color=colors['quaternary'], alpha=0.25)

    #     # --- Panel 2: Cell Body ---
    #     m_b = np.array(results.get(f'uptake_cell_body{suffix}', []))
    #     s_b = get_sem(f'uptake_cell_body{suffix}_std', 'count_cell_body')
    #     if len(m_b) > 0:
    #         ax2.plot(t, m_b, marker_style, color=colors['tertiary'], linewidth=2, markersize=ms, label='Cell Body')
    #         if s_b is not None: 
    #             ax2.fill_between(t, m_b - s_b, m_b + s_b, color=colors['tertiary'], alpha=0.25)

    #     # --- Panel 3: Total Cell ---
    #     m_t = np.array(results.get(f'uptake_total{suffix}', []))
    #     s_t = get_sem(f'uptake_total{suffix}_std', 'count_total')
    #     if len(m_t) > 0:
    #         ax3.plot(t, m_t, marker_style, color=colors['primary'], linewidth=2, markersize=ms, label='Total Cell')
    #         if s_t is not None: 
    #             ax3.fill_between(t, m_t - s_t, m_t + s_t, color=colors['primary'], alpha=0.25)

    #     # --- Formatting & Styling ---
    #     ax1.set_title("Protrusion")
    #     ax2.set_title("Cell Body")
    #     ax3.set_title("Total Cell")
    #     ax1.set_ylabel(ylabel)
        
    #     for ax in [ax1, ax2, ax3]:
    #         ax.set_xlabel("Time (s)")
    #         ax.grid(True, alpha=0.3)
    #         ax.legend(loc='upper left')

    #     fig.suptitle(f"Trap {trap_idx}: {title}", y=1.05, fontsize=16, weight='bold')
    #     plt.tight_layout()
    #     utils.save_plot_png(output_dir / fig_name)
    #     plt.close(fig)

    # # --- Generate the three 3-panel plots ---
    # plot_3panel_metric(f"Trap_{trap_idx:02d}_Uptake_Absolute.png", "Dye Uptake (Absolute)", "Mean Intensity (a.u.) ± SEM", "")
    # plot_3panel_metric(f"Trap_{trap_idx:02d}_Uptake_Normalized.png", "Normalized Uptake", "Normalized Fluorescence ($\Delta F/F_0$) ± SEM", "_norm")
    # if 'uptake_total_minmax' in results:
    #     plot_3panel_metric(f"Trap_{trap_idx:02d}_Uptake_MinMax.png", "Min-Max Normalized Uptake", "Normalized Intensity (0-1) ± SEM", "_minmax")

    # --- D. Heterogeneity Analysis (StdDev & CV) ---
    if 'uptake_total_std' in results:
        fig_het, (ax_std, ax_cv) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # 1. Standard Deviation Plot
        ax_std.axvline(pulse_time, color=colors['pulse'], linestyle='--', alpha=0.5)
        ax_std.plot(t, results['uptake_total_std'], marker_style, color=colors['primary'], label='Total')
        ax_std.plot(t, results['uptake_protrusion_std'], marker_style, color=colors['secondary'], label='Protrusion')
        ax_std.plot(t, results['uptake_cell_body_std'], marker_style, color=colors['tertiary'], label='Body')
        ax_std.set_ylabel("Standard Deviation (a.u.)")
        ax_std.set_title(f"Trap {trap_idx}: Spatial Heterogeneity (Physical Variation)")
        ax_std.legend(loc='upper left')
        
        # 2. CV Plot
        def safe_cv(mean_arr, std_arr):
            m = np.array(mean_arr); s = np.array(std_arr)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = s / m
                cv[m < 10] = 0
            return cv

        cv_tot = safe_cv(results['uptake_total'], results['uptake_total_std'])
        cv_prot = safe_cv(results['uptake_protrusion'], results['uptake_protrusion_std'])
        cv_body = safe_cv(results['uptake_cell_body'], results['uptake_cell_body_std'])

        ax_cv.axvline(pulse_time, color=colors['pulse'], linestyle='--', alpha=0.5)
        ax_cv.plot(t, cv_tot, marker_style, color=colors['primary'], label='Total')
        ax_cv.plot(t, cv_prot, marker_style, color=colors['secondary'], label='Protrusion')
        ax_cv.plot(t, cv_body, marker_style, color=colors['tertiary'], label='Body')
        
        ax_cv.set_ylabel("Coefficient of Variation ($\sigma/\mu$)")
        ax_cv.set_xlabel("Time (s)")
        ax_cv.set_title("Relative Uniformity (Lower = More Uniform)")
        
        utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Uptake_Heterogeneity.png")
        plt.close(fig_het)

    # --- E. Kymograph & Diffusion (Existing logic) ---
    profiles = results['spatial_profiles']
    if profiles:
        # Kymograph
        max_w = max(len(p) for p in profiles)
        kymo = np.zeros((len(profiles), max_w))
        for i, p in enumerate(profiles): kymo[i, :len(p)] = p
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scale = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        extent = [(pipette_x) * scale, (pipette_x - max_w) * scale, t[-1], t[0]]
        mfa_cmap = utils.get_mfa_continuous_cmap()
        
        im = ax.imshow(kymo, aspect='auto', extent=extent, cmap=mfa_cmap, interpolation='nearest')
        ax.axvline(0, color=colors['pulse'], linestyle='--', linewidth=1, label='Pipette Entrance')
        ax.axhline(pulse_time, color='white', linestyle='--', linewidth=1, label='Pulse')
        plt.colorbar(im, ax=ax, label="Intensity")
        ax.set_xlabel("Position relative to channel entrance (µm)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Trap {trap_idx}: Uptake Kymograph")
        utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Uptake_Kymograph.png")
        plt.close(fig)

        # Diffusion Profiles
        fig, ax = plt.subplots(figsize=(10, 6))
        n_curves = 7
        indices = np.linspace(0, len(t)-1, n_curves, dtype=int)
        time_colors = utils.get_time_colormap(len(indices))
        max_w = max(len(p) for p in profiles)
        x_axis = (pipette_x - np.arange(max_w)) * params.get('scale_factor', 0.629)
        
        for i, idx in enumerate(indices):
            if idx >= len(profiles): continue
            p = profiles[idx]
            y = np.zeros(max_w); y[:len(p)] = p
            ax.plot(x_axis, y, color=time_colors[i], linewidth=2.5, label=f"{t[idx]:.1f}s")

        ax.axvline(0, color=colors['pulse'], linestyle='--', alpha=0.6, label='Entrance')
        ax.invert_xaxis() 
        ax.set_xlabel("Position relative to channel entrance (µm)")
        ax.set_ylabel("Intensity")
        ax.legend(title="Time")
        utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Diffusion_Profiles.png")
        plt.close(fig)

# --- 2. KYMOGRAPH PLOT (Unchanged) ---
def plot_kymograph(kymograph_matrix: np.ndarray, trap_index: int, output_dir: Path, 
                   pipette_x: int, protrusions_px: List[float], time_data: List[float], params: Dict[str, Any]):
    if kymograph_matrix is None: return
    utils.set_paper_style()
    colors = utils.MFA_COLORS
    scale = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
    
    h, w = kymograph_matrix.shape
    total_time = time_data[-1] if time_data else h
    
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [(pipette_x) * scale, (pipette_x - w) * scale, total_time, 0]
    
    ax.imshow(kymograph_matrix, cmap='gray', aspect='auto', extent=extent)
    
    if protrusions_px and len(protrusions_px) == h:
        y_times = np.array(time_data) if len(time_data) == h else np.arange(h)
        tip_x_um = [l * scale for l in protrusions_px]
        ax.plot(tip_x_um, y_times, color=colors['secondary'], linewidth=2.0, label='Detected Tip', alpha=0.9)
    
    ax.axvline(0, color=colors['pulse'], linestyle='--', linewidth=1.0, label='Entrance')
    ax.set_title(f'Trap #{trap_index} Kymograph')
    ax.set_xlabel('Position relative to channel entrance (µm)')
    ax.set_ylabel('Time (s)')
    ax.legend(loc='upper right')
    
    utils.save_plot_png(Path(output_dir) / f"trap_{trap_index:02d}_kymograph.png")
    plt.close(fig)

# --- 3. PROTRUSION TRACE PLOT (Unchanged) ---
def plot_protrusion_trace(debug_images: List[np.ndarray], time_points: np.ndarray, 
                          protrusions: np.ndarray, trap_index: int, save_path: Path, 
                          params: Dict[str, Any], rupture_time: Optional[float], 
                          intensities: Optional[np.ndarray], 
                          pulse_time: Optional[float] = None):
    """Generates the suite of protrusion dynamics plots."""
    utils.set_paper_style()
    colors = utils.MFA_COLORS
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1.5, 1.5])
    fig.suptitle(f'Trap #{trap_index}: Protrusion Dynamics')

    valid = [img for img in debug_images if img is not None]
    if valid:
        idxs = np.linspace(0, len(valid)-1, min(4, len(valid)), dtype=int)
        for i, idx in enumerate(idxs):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(cv2.cvtColor(valid[idx], cv2.COLOR_BGR2RGB))
            ax.set_title(f"t={time_points[idx]:.1f}s")
            ax.axis('off')

    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(time_points, protrusions, 'o-', color=colors['primary'], markersize=4, label='Length')
    ax1.set_ylabel('Length (μm)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
    if intensities is not None:
        ax2.plot(time_points, intensities, '-', color=colors['quaternary'], linewidth=2, label='Haze Intensity')
    ax2.set_ylabel('Intensity (a.u.)'); ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)

    if rupture_time is not None:  # Explicit check allows 0.0s to be valid
        for ax in [ax1, ax2]:
            ax.axvline(rupture_time, color=colors['rupture'], linestyle='--', linewidth=2.5, label='Rupture Detected')
            ax.legend()
            
    if pulse_time is not None:
        for ax in [ax1, ax2]:
            ax.axvline(pulse_time, color=colors['pulse'], linestyle=':', linewidth=2, label='Pulse Applied')
            ax.legend()

    utils.save_plot_png(save_path)
    plt.close(fig)

# --- 4. FITTING ANALYSIS PLOT (Unchanged) ---
class MFAPlotter:
    def __init__(self, fitter, params=None):
        self.fitter = fitter
        self.colors = utils.MFA_COLORS
        self.params = params or {}

    def create_analysis_plot(self, trap_index: int, save_path: Path):
        summary = self.fitter.get_best_fit_summary()
        if not summary: return
        utils.set_paper_style()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        ax1.plot(self.fitter.t_raw, self.fitter.l_raw, 'o-', color=self.colors['primary'], alpha=0.6)
        if summary['rupture_detected']:
            ax1.axvline(summary['rupture_time_s'], color=self.colors['rupture'], linestyle='--')
        ax1.set_title('Raw Data')
        
        ax2.plot(self.fitter.t_fit, self.fitter.l_fit, 'o', color=self.colors['primary'], alpha=0.5, label='Data')
        pred = self.fitter.predict(summary['best_model_name'], self.fitter.t_fit)
        ax2.plot(self.fitter.t_fit, pred, '-', color=self.colors['secondary'], linewidth=3, label='Fit')
        ax2.set_title(f"Best Fit: {summary['best_model_name']}")
        ax2.legend()
        
        res = self.fitter.l_fit - pred
        ax3.plot(self.fitter.t_fit, res, 'o-', color=self.colors['primary'])
        ax3.axhline(0, color='black')
        ax3.set_title('Residuals')
        
        ax4.axis('off')
        ax4.set_title("Fitted Parameters")
        text_str = f"Model: {summary['best_model_name']}\n"
        text_str += f"R²: {summary['r_squared']:.4f}\n\n"
        for k, v in summary['parameters'].items():
            text_str += f"{k}: {v:.2e}\n"
        if summary['rupture_detected']:
            text_str += f"\nRupture: {summary['rupture_time_s']:.2f}s"

        ax4.text(0.1, 0.9, text_str, transform=ax4.transAxes, fontsize=12, va='top', fontfamily='monospace')
        
        utils.save_plot_png(save_path)
        plt.close(fig)

    def plot_all_models_comparison(self, trap_index: int, save_path: Path):
        if not self.fitter.fit_results: return
        utils.set_paper_style()
        fig, ax = plt.subplots(figsize=(18, 12))
        
        ax.plot(self.fitter.t_fit, self.fitter.l_fit, 'o', color=self.colors['primary'], label='Data', alpha=0.6)
        t_smooth = np.linspace(self.fitter.t_fit.min(), self.fitter.t_fit.max(), 300)
        
        for name, res in self.fitter.fit_results.items():
            if res['params'] is None: continue
            l_pred = self.fitter.predict(name, t_smooth)
            if l_pred is not None:
                style = '-' if name == self.fitter.best_fit_model_name else '--'
                color = self.colors['secondary'] if name == self.fitter.best_fit_model_name else None
                ax.plot(t_smooth, l_pred, style, linewidth=2, color=color, label=f"{name} (R2={res['r_squared']:.3f})")
        
        ax.legend()
        ax.set_title(f"Model Comparison - Trap {trap_index}")
        utils.save_plot_png(save_path)
        plt.close(fig)
        
# --- 4. ACTIN PLOT ---
def plot_actin_dashboard(results: Dict[str, Any], trap_idx: int, output_dir: Path, params: Dict[str, Any], pipette_x: Optional[float] = None):
    """Generates the suite of Actin distribution plots."""
    utils.set_paper_style()
    output_dir = Path(output_dir)
    t = np.array(results.get('time_s', []))
    if len(t) == 0: return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, results.get('actin_ratio_pb', []), 'o-', color=utils.MFA_COLORS['secondary'], linewidth=2)
    ax.set_ylabel("Actin Ratio (Protrusion / Body)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Trap {trap_idx}: Actin Distribution")
    
    utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Actin_Ratio.png")
    plt.close(fig)

def plot_actin_kymograph_and_profiles(results: Dict[str, Any], trap_idx: int, output_dir: Path, params: Dict[str, Any], pipette_x: float):
    """Generates the Space-Time Kymograph and Diffusion Profiles for Actin."""
    utils.set_paper_style()
    output_dir = Path(output_dir)
    t = np.array(results.get('time_s', []))
    profiles = results.get('spatial_profiles', [])
    if len(t) == 0 or not profiles: return

    colors = utils.MFA_COLORS
    scale = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
    max_w = max(len(p) for p in profiles)
    
    # --- 1. Actin Kymograph ---
    kymo = np.zeros((len(profiles), max_w))
    for i, p in enumerate(profiles): kymo[i, :len(p)] = p
    
    fig_kymo, ax_kymo = plt.subplots(figsize=(10, 6))
    extent = [(pipette_x) * scale, (pipette_x - max_w) * scale, t[-1], t[0]]
    
    im = ax_kymo.imshow(kymo, aspect='auto', extent=extent, cmap='RdBu_r', interpolation='nearest') # Using RdBu_r to match your attached image
    ax_kymo.axvline(0, color='white', linestyle='--', linewidth=1, label='Entrance', alpha=0.7)
    
    plt.colorbar(im, ax=ax_kymo, label="Actin Intensity")
    ax_kymo.set_xlabel("Position relative to channel entrance (µm)")
    ax_kymo.set_ylabel("Time (s)")
    ax_kymo.set_title(f"Trap {trap_idx}: Actin Uptake Kymograph")
    utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Actin_Kymograph.png")
    plt.close(fig_kymo)

    # --- 2. Actin Profiles over time ---
    fig_prof, ax_prof = plt.subplots(figsize=(10, 6))
    n_curves = 6
    indices = np.linspace(0, len(t)-1, n_curves, dtype=int)
    time_colors = utils.get_time_colormap(len(indices))
    
    x_axis = (pipette_x - np.arange(max_w)) * scale
    
    for i, idx in enumerate(indices):
        if idx >= len(profiles): continue
        p = profiles[idx]
        y = np.zeros(max_w); y[:len(p)] = p
        ax_prof.plot(x_axis, y, color=time_colors[i], linewidth=2.5, label=f"{t[idx]:.1f}s")

    ax_prof.axvline(0, color=colors['primary'], linestyle='--', alpha=0.6, label='Entrance')
    ax_prof.invert_xaxis() 
    ax_prof.set_xlabel("Position relative to channel entrance (µm)")
    ax_prof.set_ylabel("Mean Actin Intensity")
    ax_prof.legend(title="Time", bbox_to_anchor=(1.05, 1), loc='upper left')
    utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Actin_Profiles.png")
    plt.close(fig_prof)


def plot_aggregate_metrics(all_results: List[Dict[str, Any]], time_data: List[float], output_dir: Path, experiment_id: str):
    """Plots the change (Delta) in Area and Solidity for all traps as a bar chart."""
    utils.set_paper_style()
    output_dir = Path(output_dir)
    
    trap_ids = []
    delta_areas = []
    delta_solidities = []
    
    for res in sorted(all_results, key=lambda x: x['trap_index']):
        if res.get('status') in ('success', 'detection_only', 'fit_failed') and 'data' in res:
            data = res['data']
            if 'area' in data and 'solidity' in data and len(data['area']) > 0:
                area = np.array(data['area'])
                solidity = np.array(data['solidity'])
                
                r_idx = data.get('rupture_idx')
                # Determine the final valid frame (either rupture or end of experiment)
                end_idx = r_idx if (r_idx is not None and r_idx < len(area)) else len(area) - 1
                
                if end_idx < 3: 
                    continue # Not enough data to calculate a reliable change
                
                # Average the first 3 frames for a stable initial baseline
                init_area = np.nanmean(area[:3])
                init_sol = np.nanmean(solidity[:3])
                
                # Average the last 3 valid frames for a stable final reading
                final_area = np.nanmean(area[max(0, end_idx-3):end_idx])
                final_sol = np.nanmean(solidity[max(0, end_idx-3):end_idx])
                
                trap_ids.append(f"Trap {data['trap_index'] + 1}")
                delta_areas.append(final_area - init_area)
                delta_solidities.append(final_sol - init_sol)

    if not trap_ids:
        return

    # --- 1. Plot Delta Area ---
    fig_area, ax_area = plt.subplots(figsize=(12, 6))
    bars = ax_area.bar(trap_ids, delta_areas, color=utils.MFA_COLORS['secondary'], alpha=0.8, edgecolor='black')
    ax_area.axhline(0, color='black', linewidth=1.5)
    ax_area.set_ylabel("Change in Total Cell Area ($\Delta$ µm²)")
    ax_area.set_title(f"Cell Area Dynamics (Start vs. Rupture) - {experiment_id}")
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        offset = 5 if yval >= 0 else -15
        ax_area.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.1f}", ha='center', va='bottom' if yval >=0 else 'top', fontsize=9)
        
    plt.tight_layout()
    utils.save_plot_png(output_dir / f"{experiment_id}_Aggregate_Delta_Area.png")
    plt.close(fig_area)
    
    # --- 2. Plot Delta Solidity ---
    fig_sol, ax_sol = plt.subplots(figsize=(12, 6))
    bars_sol = ax_sol.bar(trap_ids, delta_solidities, color=utils.MFA_COLORS['tertiary'], alpha=0.8, edgecolor='black')
    ax_sol.axhline(0, color='black', linewidth=1.5)
    ax_sol.set_ylabel("Change in Cell Body Solidity ($\Delta$)")
    ax_sol.set_title(f"Cell Solidity Dynamics (Start vs. Rupture) - {experiment_id}")
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars_sol:
        yval = bar.get_height()
        offset = 0.01 if yval >= 0 else -0.02
        ax_sol.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.3f}", ha='center', va='bottom' if yval >=0 else 'top', fontsize=9)

    plt.tight_layout()
    utils.save_plot_png(output_dir / f"{experiment_id}_Aggregate_Delta_Solidity.png")
    plt.close(fig_sol)
    

def plot_aggregate_dye_metrics(all_results: List[Dict[str, Any]], output_dir: Path, experiment_id: str, params: Dict[str, Any]):
    """Plots Dye Uptake (Absolute, Normalized, MinMax) for all traps on single 3-panel figures."""
    utils.set_paper_style()
    output_dir = Path(output_dir)
    
    # Filter for traps that successfully ran dye analysis
    valid_results = [res for res in all_results if res.get('status') in ('success', 'detection_only', 'fit_failed') and res.get('dye_data') is not None]
    if not valid_results:
        return
        
    valid_results.sort(key=lambda x: x['trap_index'])
    colors = utils.get_time_colormap(max(len(valid_results), 1))
    
    # Extract pulse timing
    pulse_time = None
    dye_params = params.get('dye_uptake_parameters', {})
    if dye_params.get('enable', False):
        p_idx = dye_params.get('pulse_frame', 10) - 1
        t_first = valid_results[0]['data']['time']
        if 0 <= p_idx < len(t_first):
            pulse_time = t_first[p_idx]
    
    def _plot_aggregated_3panel(metric_suffix: str, title: str, ylabel: str, filename_suffix: str):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        
        if pulse_time is not None:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(pulse_time, color=utils.MFA_COLORS['pulse'], linestyle='--', linewidth=2.5, label='Pulse', alpha=0.7)
                
        plotted_any = False
        for i, res in enumerate(valid_results):
            dye_data = res['dye_data']
            t = np.array(dye_data.get('time_s', []))
            if len(t) == 0: continue
            
            trap_id = res['trap_index'] + 1
            
            m_p = np.array(dye_data.get(f'uptake_protrusion{metric_suffix}', []))
            m_b = np.array(dye_data.get(f'uptake_cell_body{metric_suffix}', []))
            m_t = np.array(dye_data.get(f'uptake_total{metric_suffix}', []))
            
            # Truncate plotting cleanly if the cell ruptured
            r_idx = res['data'].get('rupture_idx')
            if r_idx is not None and r_idx < len(t):
                t, m_p, m_b, m_t = t[:r_idx], m_p[:r_idx], m_b[:r_idx], m_t[:r_idx]
            
            if len(m_p) > 0:
                ax1.plot(t, m_p, color=colors[i], linewidth=2, alpha=0.7, label=f"Trap {trap_id}")
                ax2.plot(t, m_b, color=colors[i], linewidth=2, alpha=0.7, label=f"Trap {trap_id}")
                ax3.plot(t, m_t, color=colors[i], linewidth=2, alpha=0.7, label=f"Trap {trap_id}")
                plotted_any = True
                
        if plotted_any:
            ax1.set_title("Protrusion")
            ax2.set_title("Cell Body")
            ax3.set_title("Total Cell")
            ax1.set_ylabel(ylabel)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel("Time (s)")
                ax.grid(True, alpha=0.3)
            
            # Place legend cleanly outside the last plot
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=max(1, len(valid_results)//15 + 1))
            
            fig.suptitle(f"{experiment_id} - {title}", y=1.05, fontsize=16, weight='bold')
            plt.tight_layout()
            utils.save_plot_png(output_dir / f"{experiment_id}_Aggregate_Uptake{filename_suffix}.png")
        plt.close(fig)

    # Trigger the 3 versions
    _plot_aggregated_3panel("", "Dye Uptake (Absolute)", "Mean Intensity (a.u.)", "_Absolute")
    _plot_aggregated_3panel("_norm", "Normalized Uptake", "Normalized Fluorescence ($\Delta$F/F0)", "_Normalized")
    
    # Safe check for MinMax incase it gets disabled in config
    if f'uptake_total_minmax' in valid_results[0]['dye_data']:
        _plot_aggregated_3panel("_minmax", "Min-Max Normalized Uptake", "Normalized Intensity (0-1)", "_MinMax")
        
def plot_shear_analysis_card(metrics: Dict[str, float], save_path: Path):
    utils.set_paper_style(base_fontsize=12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    ax.text(0.5, 0.9, "Shear Analysis", ha='center', fontsize=16, weight='bold', color=utils.MFA_COLORS['primary'])
    
    y = 0.7
    for k, v in metrics.items():
        ax.text(0.3, y, k, ha='left', color=utils.MFA_COLORS['tertiary'])
        ax.text(0.7, y, f"{v:.4f}", ha='right')
        y -= 0.1
        
    utils.save_plot_png(save_path)
    plt.close(fig)