# -*- coding: utf-8 -*-
"""
Centralized plotting module for MFA analysis.

ROLE IN PIPELINE:
This module generates all static figures (PNGs). It ensures consistent styling
(fonts, line widths, colors) across all outputs.

KEY FIGURES:
1.  Analysis Plot (4-Panel):
    - Top Left: Raw data vs. Cleaned data.
    - Top Right: The best fit curve overlaid on data.
    - Bottom Left: Residuals (Error) plot to check for systematic bias.
    - Bottom Right: Rupture analysis detail.
2.  Model Comparison: Overlays all attempted models (Jeffreys, Burgers, etc.)
    to visualize which one describes the data best.
"""
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import Utils_MFA as utils

if TYPE_CHECKING:
    from Fitting_MFA import FittingMFA

logger = logging.getLogger(__name__)

# Standard units for axis labels
UNITS = {'E': 'Pa', 'E1': 'Pa', 'E2': 'Pa', 'eta': 'Pa·s', 'eta1': 'Pa·s', 'eta2': 'Pa·s', 'm': 'μm/s', 'b': 'μm', 'a': 'μm/sᵇ', 'c': 'μm'}

class MFAPlotter:
    """Handles visualization for fitting results."""
    def __init__(self, fitter: 'FittingMFA', params: Optional[Dict[str, Any]] = None, color_scheme: str = 'blue_red') -> None:
        self.fitter = fitter
        self.params = params or {}
        self.colors = utils.get_color_scheme(color_scheme)
        self.dpi = self.params.get('default_dpi', 300)
        self.fig_large = tuple(self.params.get('figure_size_large', (18.0, 12.0)))

    def create_analysis_plot(self, trap_index: int, save_path: Optional[Union[str, Path]] = None, show: bool = False) -> None:
        """
        Generates the main 4-panel dashboard for a single trap.
        Useful for quickly verifying if a fit "looks right".
        """
        summary = self.fitter.get_best_fit_summary()
        if not summary: return
        
        utils.set_paper_style(dpi=self.dpi)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.fig_large, constrained_layout=True)
        fig.suptitle(f'MFA Analysis - Trap #{trap_index}', fontsize=16, fontweight='bold')
        
        # 1. Overview: Raw vs Fit data
        self._plot_panel_1_overview(ax1, self.fitter.t_raw, self.fitter.l_raw, self.fitter.t_fit, self.fitter.l_fit, summary)
        # 2. Model: Best fit curve
        self._plot_panel_2_model_fit(ax2, self.fitter.t_raw, self.fitter.l_raw, self.fitter.t_fit, self.fitter.l_fit, summary)
        # 3. Residuals: (Data - Model)
        self._plot_panel_3_residuals(ax3)
        # 4. Rupture: Visualization of the cut-off point
        self._plot_panel_4_rupture(ax4, self.fitter.t_raw, self.fitter.l_raw, summary)
        
        if save_path: utils.save_plot_png(save_path, dpi=self.dpi)
        plt.close(fig)

    def plot_all_models_comparison(self, trap_index: int, save_path: Optional[Union[str, Path]] = None) -> None:
        """Plots a comparison of ALL fitted models against the experimental data."""
        if not self.fitter.fit_results:
            return

        utils.set_paper_style(dpi=self.dpi)
        fig, ax = plt.subplots(figsize=self.fig_large)
        
        # Plot Experimental Data
        ax.plot(self.fitter.t_raw, self.fitter.l_raw, 'o', color='lightgray', alpha=0.5, markersize=4, label='Raw Data')
        if self.fitter.t_fit is not None and self.fitter.l_fit is not None:
             ax.plot(self.fitter.t_fit, self.fitter.l_fit, 'o', color=self.colors['fitting_data'], markersize=4, label='Fitted Data')

        # Plot All Models
        t_smooth = np.linspace(self.fitter.t_raw.min(), self.fitter.t_raw.max(), 500)
        # Sort models by R2 for legend order
        sorted_models = sorted(self.fitter.fit_results.items(), key=lambda x: x[1]['r_squared'], reverse=True)

        for name, result in sorted_models:
            if result['params'] is None: continue
            
            l_pred = self.fitter.predict(name, t_smooth)
            if l_pred is not None:
                r2 = result['r_squared']
                label = f"{name} ($R^2={r2:.3f}$)"
                linestyle = '-' if name == self.fitter.best_fit_model_name else '--'
                linewidth = 3 if name == self.fitter.best_fit_model_name else 1.5
                alpha = 1.0 if name == self.fitter.best_fit_model_name else 0.7
                
                ax.plot(t_smooth, l_pred, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)

        ax.set_title(f"Model Comparison - Trap #{trap_index}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Protrusion Length (μm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            utils.save_plot_png(save_path, dpi=self.dpi)
        plt.close(fig)

    def _plot_panel_1_overview(self, ax: plt.Axes, t_raw: np.ndarray, l_raw: np.ndarray, t_fit: np.ndarray, l_fit: np.ndarray, summary: Dict[str, Any]) -> None:
        ax.plot(t_raw, l_raw, 'o-', color=self.colors['data_points'], alpha=0.6, markersize=4, label='All Data')
        if t_fit is not None: ax.plot(t_fit, l_fit, 'o', color=self.colors['fitting_data'], markersize=6)
        if summary['rupture_detected']:
            # Fixed key access: rupture_time -> rupture_time_s
            ax.axvline(summary['rupture_time_s'], color=self.colors['rupture_point'], linestyle='--', linewidth=3, label="Rupture")
        ax.set_title('Data Overview'); ax.set_ylabel('Length (μm)'); ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_panel_2_model_fit(self, ax: plt.Axes, t_raw: np.ndarray, l_raw: np.ndarray, t_fit: np.ndarray, l_fit: np.ndarray, summary: Dict[str, Any]) -> None:
        ax.plot(t_raw, l_raw, 'o', color='lightgray', alpha=0.5, markersize=3)
        if t_fit is not None:
            ax.plot(t_fit, l_fit, 'o', color=self.colors['fitting_data'], markersize=5)
            t_smooth = np.linspace(t_fit.min(), t_fit.max(), 300)
            ax.plot(t_smooth, self.fitter.predict(summary["best_model_name"], t_smooth), '-', color=self.colors['model_fit'], linewidth=3)
        
        txt = f"R² = {summary['r_squared']:.4f}\n" + "\n".join([f"{k}={v:.2e}" for k, v in summary['parameters'].items()])
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'Best Fit: {summary["best_model_name"]}'); ax.set_xlabel('Time (s)'); ax.grid(True, alpha=0.3)

    def _plot_panel_3_residuals(self, ax: plt.Axes) -> None:
        if self.fitter.t_fit is not None:
            res = self.fitter.l_fit - self.fitter.predict(self.fitter.best_fit_model_name, self.fitter.t_fit)
            ax.plot(self.fitter.t_fit, res, 'o-', color=self.colors['data_points'])
        ax.axhline(0, color='black', alpha=0.3); ax.set_title('Residuals'); ax.set_ylabel('μm'); ax.grid(True, alpha=0.3)

    def _plot_panel_4_rupture(self, ax: plt.Axes, t_raw: np.ndarray, l_raw: np.ndarray, summary: Dict[str, Any]) -> None:
        ax.plot(t_raw, l_raw, 'o-', color=self.colors['fitting_data' if not summary['rupture_detected'] else 'data_points'])
        if summary['rupture_detected']: 
            # Fixed key access: rupture_time -> rupture_time_s
            ax.axvline(summary['rupture_time_s'], color=self.colors['rupture_point'], linestyle='--', linewidth=3)
        ax.set_title('Rupture Analysis'); ax.set_xlabel('Time (s)'); ax.grid(True, alpha=0.3)

def plot_protrusion_trace(debug_images: List[np.ndarray], time_points: np.ndarray, protrusions: np.ndarray, trap_index: int, save_path: Optional[Union[str, Path]] = None, params: Optional[Dict[str, Any]] = None, color_scheme: str = 'blue_red', rupture_time: Optional[float] = None, intensities: Optional[np.ndarray] = None) -> None:
    params = params or {}
    colors = utils.get_color_scheme(color_scheme)
    utils.set_paper_style(dpi=params.get('default_dpi', 300))
    
    fig = plt.figure(figsize=tuple(params.get('figure_size_large', (16.0, 10.0))))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1.5, 1.5])
    fig.suptitle(f'Trap #{trap_index}: Protrusion & Rupture Detection', fontsize=16, fontweight='bold')

    # Row 0: Images
    valid_imgs = [img for img in debug_images if img is not None]
    if valid_imgs:
        idxs = np.linspace(0, len(debug_images) - 1, min(4, len(valid_imgs)), dtype=int)
        for i, idx in enumerate(idxs):
            ax = fig.add_subplot(gs[0, i])
            if debug_images[idx] is not None:
                ax.imshow(debug_images[idx])
                if idx < len(time_points): ax.set_title(f"t={time_points[idx]:.1f}s")
            ax.axis('off')

    # Row 1: Protrusion
    ax_trace = fig.add_subplot(gs[1, :])
    ax_trace.plot(time_points, protrusions, 'o-', color=colors['data_points'], markersize=4, label='Length')
    ax_trace.set_ylabel('Length (μm)'); ax_trace.set_title("Protrusion Dynamics"); ax_trace.grid(True, alpha=0.3)

    # Row 2: Intensity
    ax_int = fig.add_subplot(gs[2, :], sharex=ax_trace)
    if intensities is not None and len(intensities) == len(time_points):
        ax_int.plot(time_points, intensities, '-', color='#E6550D', linewidth=2, label='Downstream Haze')
    ax_int.set_ylabel('Intensity (a.u.)'); ax_int.set_title("Rupture Monitor"); ax_int.set_xlabel('Time (s)'); ax_int.grid(True, alpha=0.3)

    # RED RUPTURE LINE
    if rupture_time is not None:
        ax_trace.axvline(x=rupture_time, color=colors['rupture_point'], linestyle='--', linewidth=2.5, label='Rupture')
        ax_int.axvline(x=rupture_time, color=colors['rupture_point'], linestyle='--', linewidth=2.5, label='Rupture')
        ax_trace.legend(); ax_int.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: utils.save_plot_png(save_path); plt.close(fig)
    else: plt.show()
    

def plot_shear_analysis_card(metrics: Dict[str, float], save_path: Path):
    """
    Creates a visual 'report card' for the shear metrics derived from Son (2007).
    """
    utils.set_paper_style(base_fontsize=12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.9, "Rectangular Channel Flow Analysis", 
            ha='center', va='center', fontsize=16, weight='bold', color=utils.DATA_POINTS_COLOR)
    
    ax.text(0.5, 0.82, "Method: Son, Y. Polymer 48 (2007)", 
            ha='center', va='center', fontsize=10, style='italic', color='gray')

    # Metrics
    lines = [
        ("Geometry", ""),
        (f"Aspect Ratio (H/W)", f"{metrics['Aspect_Ratio']:.3f}"),
        (f"Son Shape Factor (f*)", f"{metrics['Son_Factor_fstar']:.4f}"),
        (f"Channel Length", f"{metrics['Channel_Length_um']:.1f} µm"),
        ("", ""),
        ("Fluid Dynamics", ""),
        (f"Wall Shear Stress", f"{metrics['Wall_Shear_Stress_Pa']:.2f} Pa"),
        (f"Wall Shear Rate", f"{metrics['Wall_Shear_Rate_s1']:.1e} s⁻¹"),
        (f"Theoretical Flow Rate", f"{metrics['Flow_Rate_nL_min']:.2f} nL/min")
    ]
    
    y_start = 0.70
    for i, (label, value) in enumerate(lines):
        y = y_start - (i * 0.08)
        
        if value == "":
            # Header
            ax.text(0.3, y, label, ha='left', va='center', fontsize=12, weight='bold', color='#394B9A')
            if label == "": # Separator line
                 ax.axhline(y, xmin=0.2, xmax=0.8, color='gray', alpha=0.3, linewidth=1)
        else:
            # Data row
            ax.text(0.3, y, label, ha='left', va='center', fontsize=11)
            weight = 'bold' if "Pa" in value or "s⁻¹" in value else 'normal'
            ax.text(0.7, y, value, ha='right', va='center', fontsize=11, weight=weight, fontfamily='monospace')

    plt.tight_layout()
    utils.save_plot_png(save_path, dpi=300)
    plt.close(fig)