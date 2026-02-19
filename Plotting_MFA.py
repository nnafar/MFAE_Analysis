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

    # --- A. Timecourse (Absolute with Shaded SEM) ---
    fig_abs, ax_abs = plt.subplots(figsize=(10, 6))
    ax_abs.axvline(pulse_time, color=colors['pulse'], linestyle='--', linewidth=2.5, label='Pulse')
    
    series_config = [
        ('uptake_total', 'uptake_total_std', 'count_total', colors['primary'], 'Total'),
        ('uptake_protrusion', 'uptake_protrusion_std', 'count_protrusion', colors['secondary'], 'Protrusion'),
        ('uptake_cell_body', 'uptake_cell_body_std', 'count_cell_body', colors['tertiary'], 'Body')
    ]

    for key_mean, key_std, key_count, color, label in series_config:
        mean_data = np.array(results[key_mean])
        
        # 1. Plot Mean
        ax_abs.plot(t, mean_data, marker_style, color=color, linewidth=2, markersize=ms, label=label)
        
        # 2. Add Shaded SEM
        sem_data = get_sem(key_std, key_count)
        if sem_data is not None:
            ax_abs.fill_between(t, mean_data - sem_data, mean_data + sem_data, color=color, alpha=0.25, edgecolor=None)
    
    ax_abs.set_ylabel("Mean Intensity (a.u.) ± SEM")
    ax_abs.set_xlabel("Time (s)")
    ax_abs.set_title(f"Trap {trap_idx}: Dye Uptake (Absolute)")
    ax_abs.legend()
    utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Uptake_Absolute.png")
    plt.close(fig_abs)

    # --- B. Timecourse (Normalized with Shaded SEM) ---
    fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
    ax_norm.axvline(pulse_time, color=colors['pulse'], linestyle='--', linewidth=2.5, label='Pulse')
    
    series_norm = [
        ('uptake_total_norm', 'uptake_total_norm_std', 'count_total', colors['primary'], 'Total'),
        ('uptake_protrusion_norm', 'uptake_protrusion_norm_std', 'count_protrusion', colors['secondary'], 'Protrusion'),
        ('uptake_cell_body_norm', 'uptake_cell_body_norm_std', 'count_cell_body', colors['tertiary'], 'Body')
    ]

    for key_mean, key_std, key_count, color, label in series_norm:
        mean_data = np.array(results[key_mean])
        
        # 1. Plot Mean
        ax_norm.plot(t, mean_data, marker_style, color=color, linewidth=2, markersize=ms, label=label)

        # 2. Add Shaded SEM
        sem_data = get_sem(key_std, key_count)
        if sem_data is not None:
             ax_norm.fill_between(t, mean_data - sem_data, mean_data + sem_data, color=color, alpha=0.25, edgecolor=None)
    
    ax_norm.set_ylabel("Normalized Fluorescence ($\Delta F/F_0$) ± SEM")
    ax_norm.set_xlabel("Time (s)")
    ax_norm.set_title(f"Trap {trap_idx}: Normalized Uptake")
    ax_norm.legend()
    utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Uptake_Normalized.png")
    plt.close(fig_norm)

    # --- C. Min-Max (Now with Shaded SEM) ---
    if 'uptake_total_minmax' in results:
        fig_mm, ax_mm = plt.subplots(figsize=(10, 6))
        ax_mm.axvline(pulse_time, color=colors['pulse'], linestyle='--', linewidth=2.5, label='Pulse')
        
        # New Series config for Min-Max
        # Note: We use the pre-calculated scaled Stds (uptake_total_minmax_std)
        series_minmax = [
            ('uptake_total_minmax', 'uptake_total_minmax_std', 'count_total', colors['primary'], 'Total'),
            ('uptake_protrusion_minmax', 'uptake_protrusion_minmax_std', 'count_protrusion', colors['secondary'], 'Protrusion'),
            ('uptake_cell_body_minmax', 'uptake_cell_body_minmax_std', 'count_cell_body', colors['tertiary'], 'Body')
        ]

        for key_mean, key_std, key_count, color, label in series_minmax:
            if key_mean not in results: continue
            mean_data = np.array(results[key_mean])
            
            # 1. Plot Mean
            ax_mm.plot(t, mean_data, marker_style, color=color, linewidth=2, markersize=ms, label=label)
            
            # 2. Add Shaded SEM
            # Important: get_sem divides the scaled std by sqrt(count)
            sem_data = get_sem(key_std, key_count)
            if sem_data is not None:
                 ax_mm.fill_between(t, mean_data - sem_data, mean_data + sem_data, color=color, alpha=0.25, edgecolor=None)

        ax_mm.set_ylabel("Normalized Intensity (0-1) ± SEM")
        ax_mm.set_xlabel("Time (s)")
        ax_mm.set_title(f"Trap {trap_idx}: Min-Max Normalized Uptake")
        ax_mm.legend()
        utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Uptake_MinMax.png")
        plt.close(fig_mm)

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