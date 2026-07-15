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

logger = logging.getLogger(__name__)

# Standard units for axis labels
UNITS = {'E': 'Pa', 'E1': 'Pa', 'E2': 'Pa', 'eta': 'Pa·s', 'eta1': 'Pa·s', 'eta2': 'Pa·s', 'm': 'μm/s', 'b': 'μm', 'a': 'μm/sᵇ', 'c': 'μm'}

# --- 1. DYE UPTAKE DASHBOARD ---
def plot_dye_uptake_dashboard(results: Dict[str, Any], trap_idx: int, output_dir: Path,
                              params: Dict[str, Any], pipette_x: Optional[float] = None,
                              pulse_time: Optional[float] = None,
                              rupture_time: Optional[float] = None):
    """
    Per-trap dye-uptake diagnostic plots.

    Retained:
        Trap_XX_Uptake_Kymograph.png   -- spatial dye intensity over time
        Trap_XX_Diffusion_Profiles.png -- selected time-slice profiles

    Removed:
        Trap_XX_Uptake_Heterogeneity.png (std / CV panel)
        Trap_XX_Uptake_Normalized.png    (dF/F0 3-panel)
        Trap_XX_Uptake_MinMax.png        (min-max-scaled 3-panel)

    Rupture handling: uptake analysis now runs past the detected rupture
    frame, so both retained plots draw a red horizontal marker at
    rupture_time to make the rupture visible without cutting the trace.
    """
    utils.set_paper_style()
    output_dir = Path(output_dir)
    t = np.array(results['time_s'])
    if len(t) == 0:
        return

    colors = utils.MFA_COLORS

    if pipette_x is None:
        pipette_x = results.get('pipette_x_px', 0)

    # --- E. Kymograph & Diffusion ---
    profiles = results['spatial_profiles']
    if profiles:
        # Kymograph
        max_w = max(len(p) for p in profiles)
        kymo = np.zeros((len(profiles), max_w))
        for i, p in enumerate(profiles):
            kymo[i, :len(p)] = p

        fig, ax = plt.subplots(figsize=(10, 6))
        scale = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        extent = [(pipette_x) * scale, (pipette_x - max_w) * scale, t[-1], t[0]]
        mfa_cmap = utils.get_mfa_continuous_cmap()

        im = ax.imshow(kymo, aspect='auto', extent=extent,
                       cmap=mfa_cmap, interpolation='nearest')
        ax.axvline(0, color=colors['pulse'], linestyle='--',
                   linewidth=1, label='Pipette Entrance')

        dye_params = params.get('dye_uptake_parameters', {})
        if dye_params.get('has_pulse', True) and pulse_time is not None:
            ax.axhline(pulse_time, color='white', linestyle='--',
                       linewidth=1, label='Pulse')

        # Rupture marker: red horizontal line spanning the plot.  Uptake
        # analysis is no longer truncated at rupture, so this is purely a
        # visual annotation.
        if rupture_time is not None:
            ax.axhline(rupture_time, color=colors['rupture'],
                       linestyle='--', linewidth=2, label='Rupture')

        plt.colorbar(im, ax=ax, label="Intensity")
        ax.set_xlabel("Position relative to channel entrance (\u00b5m)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Trap {trap_idx}: Uptake Kymograph")
        if pulse_time is not None or rupture_time is not None:
            ax.legend(loc='upper right')
        utils.save_plot_pdf(output_dir / f"Trap_{trap_idx:02d}_Uptake_Kymograph.png")
        plt.close(fig)

        # Diffusion Profiles
        fig, ax = plt.subplots(figsize=(10, 6))
        n_curves = 7
        indices = np.linspace(0, len(t) - 1, n_curves, dtype=int)
        time_colors = utils.get_time_colormap(len(indices))
        max_w = max(len(p) for p in profiles)
        x_axis = (pipette_x - np.arange(max_w)) * params.get(
            'experiment_parameters', {}
        ).get('scale_factor', 0.629)

        for i, idx in enumerate(indices):
            if idx >= len(profiles):
                continue
            p = profiles[idx]
            y = np.zeros(max_w)
            y[:len(p)] = p
            ax.plot(x_axis, y, color=time_colors[i], linewidth=2.5,
                    label=f"{t[idx]:.1f}s")

        ax.axvline(0, color=colors['pulse'], linestyle='--',
                   alpha=0.6, label='Entrance')

        # Rupture marker on the diffusion profile plot: the profile is
        # spatial (x = position) rather than temporal, so we cannot mark
        # rupture as a time line.  Instead, if any of the plotted slices
        # is after rupture, we tag them in the legend with an asterisk and
        # add a note.  This keeps the marker visible without cluttering
        # the axis.
        if rupture_time is not None:
            post_indices = [i for i, idx in enumerate(indices)
                            if idx < len(t) and t[idx] > rupture_time]
            if post_indices:
                handles, labels = ax.get_legend_handles_labels()
                # Prepend an asterisk to labels for post-rupture curves.
                for i in post_indices:
                    labels[i] = f"* {labels[i]}"
                # Append a legend entry explaining the asterisk.
                from matplotlib.lines import Line2D
                handles.append(Line2D([0], [0], color=colors['rupture'],
                                      linestyle='--', linewidth=2))
                labels.append(f"* = after rupture (t = {rupture_time:.1f}s)")
                ax.legend(handles, labels, title="Time")
            else:
                ax.legend(title="Time")
        else:
            ax.legend(title="Time")

        ax.invert_xaxis()
        ax.set_xlabel("Position relative to channel entrance (\u00b5m)")
        ax.set_ylabel("Intensity")
        utils.save_plot_pdf(output_dir / f"Trap_{trap_idx:02d}_Diffusion_Profiles.png")
        plt.close(fig)

# --- 2. KYMOGRAPH PLOT ---
def plot_kymograph(kymograph_matrix: np.ndarray, trap_index: int, output_dir: Path,
                   pipette_x: int, protrusions_px: List[float], time_data: List[float],
                   params: Dict[str, Any], channel_label: Optional[str] = None):
    """
    Save a kymograph image.  When channel_label is provided (e.g. "Membrane"
    or "Dye"), the filename and title include that label so multiple channels
    can coexist in the same output folder without overwriting one another.
    Legacy calls without a channel label continue to produce the original
    filename for backward compatibility.
    """
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

    title_bits = [f'Trap #{trap_index}']
    if channel_label:
        title_bits.append(channel_label)
    title_bits.append('Kymograph')
    ax.set_title(' '.join(title_bits))

    ax.set_xlabel('Position relative to channel entrance (µm)')
    ax.set_ylabel('Time (s)')
    ax.legend(loc='upper right')

    if channel_label:
        fname = f"Trap_{trap_index:02d}_{channel_label}_Kymograph.png"
    else:
        fname = f"trap_{trap_index:02d}_kymograph.png"
    utils.save_plot_pdf(Path(output_dir) / fname)
    plt.close(fig)

# --- 3. PROTRUSION TRACE PLOT ---
def plot_protrusion_trace(debug_images: List[np.ndarray], time_points: np.ndarray, 
                          protrusions: np.ndarray, trap_index: int, save_path: Path, 
                          params: Dict[str, Any], rupture_time: Optional[float], 
                          intensities: Optional[np.ndarray], 
                          pulse_time: Optional[float] = None):
    """Generates the suite of protrusion dynamics plots."""
    utils.set_paper_style()
    colors = utils.MFA_COLORS
    
    # --- Time Synchronization ---
    valid_indices = np.where(~np.isnan(protrusions))[0]
    entry_idx = valid_indices[0] if len(valid_indices) > 0 else 0
    t_shift = time_points[entry_idx]
    
    time_shifted = time_points - t_shift
    if rupture_time is not None: rupture_time -= t_shift
    if pulse_time is not None: pulse_time -= t_shift

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1.5, 1.5])
    fig.suptitle(f'Trap #{trap_index}: Protrusion Dynamics')

    valid = [img for img in debug_images if img is not None]
    if valid:
        idxs = np.linspace(0, len(valid)-1, min(4, len(valid)), dtype=int)
        for i, idx in enumerate(idxs):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(cv2.cvtColor(valid[idx], cv2.COLOR_BGR2RGB))
            ax.set_title(f"t={time_shifted[idx]:.1f}s")
            ax.axis('off')

    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(time_shifted, protrusions, 'o-', color=colors['primary'], markersize=4, label='Length')
    ax1.set_ylabel('Length (μm)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
    if intensities is not None:
        ax2.plot(time_shifted, intensities, '-', color=colors['quaternary'], linewidth=2, label='Haze Intensity')
    ax2.set_ylabel('Intensity (a.u.)'); ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)

    if rupture_time is not None:
        for ax in [ax1, ax2]:
            ax.axvline(rupture_time, color=colors['rupture'], linestyle='--', linewidth=2.5, label='Rupture Detected')
            ax.legend()
            
    if pulse_time is not None:
        for ax in [ax1, ax2]:
            ax.axvline(pulse_time, color=colors['pulse'], linestyle=':', linewidth=2, label='Pulse Applied')
            ax.legend()

    utils.save_plot_pdf(save_path)
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
        
        utils.save_plot_pdf(save_path)
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
        utils.save_plot_pdf(save_path)
        plt.close(fig)
        
# --- 4. ACTIN PLOTS ---
#
# Only two actin plots are retained in the per-trap output:
#   plot_actin_kymograph -- image of the spatial actin profile over time.
#   plot_actin_zones     -- normalised zone-mean traces and dI/dt.
#
# The dashboard (_Ratio), cortex-structure, and standalone profiles plots
# were removed because their content is either redundant with the kymograph
# and zone plots or was derived from CSV columns (relative signal, structure
# labels, cortex CV) that the pipeline no longer exports.
# ---------------------------------------------------------------------------

def plot_actin_kymograph(results: Dict[str, Any], trap_idx: int, output_dir: Path,
                         params: Dict[str, Any], pipette_x: float,
                         pulse_time: Optional[float] = None,
                         rupture_time: Optional[float] = None):
    """
    Save a kymograph of the actin channel: spatial intensity profile (rows =
    frames, columns = position along the aspiration axis) as an image.

    The underlying data comes from results['spatial_profiles'], which is a
    list of variable-length 1D arrays (one per frame) already computed by
    ActinQuantification.  Rows shorter than the maximum width are zero-padded
    on the right so imshow sees a rectangular matrix.
    """
    utils.set_paper_style()
    output_dir = Path(output_dir)

    t = np.array(results.get('time_s', []))
    profiles = results.get('spatial_profiles', [])
    if len(t) == 0 or not profiles:
        return

    colors = utils.MFA_COLORS
    scale = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
    max_w = max(len(p) for p in profiles)

    kymo = np.zeros((len(profiles), max_w))
    for i, p in enumerate(profiles):
        kymo[i, :len(p)] = p

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [(pipette_x) * scale, (pipette_x - max_w) * scale, t[-1], t[0]]

    im = ax.imshow(kymo, aspect='auto', extent=extent, cmap='RdBu_r', interpolation='nearest')
    ax.axvline(0, color='white', linestyle='--', linewidth=1, label='Entrance', alpha=0.7)

    if rupture_time is not None:
        ax.axhline(rupture_time, color=colors['rupture'], linestyle='--', linewidth=2, label='Rupture')
    if pulse_time is not None:
        ax.axhline(pulse_time, color=colors['pulse'], linestyle=':', linewidth=2, label='Pulse')

    plt.colorbar(im, ax=ax, label="Actin Intensity")
    ax.set_xlabel("Position relative to channel entrance (µm)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Trap {trap_idx}: Actin Kymograph")
    if rupture_time is not None or pulse_time is not None:
        ax.legend()

    utils.save_plot_pdf(output_dir / f"Trap_{trap_idx:02d}_Actin_Kymograph.png")
    plt.close(fig)


def plot_actin_zones(results: Dict[str, Any], trap_idx: int, output_dir: Path, params: Dict[str, Any], pulse_time: Optional[float] = None, rupture_time: Optional[float] = None):
    """
    Two-panel plot showing actin dynamics across the four spatial zones:

        Top    – Normalised mean intensity per zone over time (I / I_0).
                 Each zone is one line.  Values > 1 = actin accumulated relative
                 to the pre-pulse baseline.  Values < 1 = actin was lost.

        Bottom – Rate of change dI/dt [a.u./s] for each zone.
                 Positive = actin accumulating.  Negative = actin leaving.
                 A horizontal grey line at zero separates accumulation from loss.

    Zone colour scheme (matches debug video):
        Protrusion tip       – dark blue
        Protrusion base      – medium blue
        Perinuclear body     – pale red
        Distal body          – light red
    """
    utils.set_paper_style()
    output_dir = Path(output_dir)
    t = np.array(results.get('time_s', []))
    if len(t) == 0:
        return

    colors = utils.MFA_COLORS
    zone_styles = {
        'tip':         {'label': 'Protrusion tip',   'color': colors['dark_blue'],   'ls': '-'},
        'base':        {'label': 'Protrusion base',  'color': colors['medium_blue'], 'ls': '-'},
        'perinuclear': {'label': 'Perinuclear body', 'color': colors['pale_red'],    'ls': '-'},
        'distal':      {'label': 'Distal body',      'color': colors['medium_red'],  'ls': '-'},
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Trap {trap_idx}: Actin Zone Dynamics", fontweight='bold')

    ax_norm  = axes[0]
    ax_didt  = axes[1]

    for zone, style in zone_styles.items():
        norm_key = f'zone_{zone}_norm'
        didt_key = f'zone_{zone}_didt'

        norm_vals = np.array(results.get(norm_key, []))
        didt_vals = np.array(results.get(didt_key, []))

        if len(norm_vals) == len(t):
            ax_norm.plot(t, norm_vals,
                         linestyle=style['ls'], color=style['color'],
                         linewidth=2, label=style['label'])

        if len(didt_vals) == len(t):
            # Replace NaN in frame 0 with 0 for display only
            didt_display = np.where(np.isnan(didt_vals), 0, didt_vals)
            ax_didt.plot(t, didt_display,
                         linestyle=style['ls'], color=style['color'],
                         linewidth=2, label=style['label'])

    # Reference lines
    ax_norm.axhline(1.0, color=colors['primary'], linestyle=':', linewidth=1,
                    alpha=0.5, label='Baseline ($I_0$)')
    ax_didt.axhline(0.0, color=colors['primary'], linestyle='-', linewidth=1,
                    alpha=0.4)

    # Pulse / rupture markers on both panels
    for ax in axes:
        if pulse_time is not None:
            ax.axvline(pulse_time, color=colors['pulse'], linestyle=':',
                       linewidth=2, label='Pulse Applied')
        if rupture_time is not None:
            ax.axvline(rupture_time, color=colors['rupture'], linestyle='--',
                       linewidth=2, label='Rupture Detected')

    ax_norm.set_ylabel("Normalised Actin Intensity (I / $I_0$)")
    ax_norm.set_title("Zone Mean Intensity (Normalised to Pre-Pulse Baseline)")
    ax_norm.legend(fontsize=10)

    ax_didt.set_ylabel("dI/dt  (a.u. / s)")
    ax_didt.set_xlabel("Time (s)")
    ax_didt.set_title("Rate of Change of Actin Intensity per Zone")
    ax_didt.legend(fontsize=10)

    fig.tight_layout()
    utils.save_plot_pdf(output_dir / f"Trap_{trap_idx:02d}_Actin_Zones.png")
    plt.close(fig)



def plot_aggregate_metrics(all_results: List[Dict[str, Any]], time_data: List[float], output_dir: Path, experiment_id: str):
    """
    Produces two aggregate plots from all successfully processed traps:

    1. Cell Area Dynamics — grouped bar chart showing normalized area change (%) for the
       Protrusion, Cell Body, and Total regions.  Positive bars = expansion; negative = shrinkage.
       When a pulse was applied, each region gets TWO adjacent bars: pre-pulse change and
       post-pulse change, each normalized to the area at that phase's own start point.

    2. Cell Body Solidity — scatter plot (mean ± std across frames) per trap.
       When a pulse was applied, two points are plotted side-by-side per trap: one for the
       pre-pulse window and one for the post-pulse window.
    """
    utils.set_paper_style()
    output_dir = Path(output_dir)

    # -----------------------------------------------------------------------
    # Helper: compute normalized area change for one array window.
    #   norm_delta = (mean(arr[end-3:end]) - mean(arr[start:start+3]))
    #                / mean(arr[start:start+3])  × 100
    # Returns 0.0 when not enough data or baseline is zero.
    # -----------------------------------------------------------------------
    def _norm_delta(arr, start, end):
        """
        arr   — 1-D numpy array of area values (may contain NaN).
        start — first frame index of the window (inclusive).
        end   — last frame index of the window (exclusive).
        """
        if len(arr) == 0 or end <= start:
            return 0.0
        baseline = np.nanmean(arr[start:start + 3])
        final    = np.nanmean(arr[max(start, end - 3):end])
        if baseline > 0:
            return (final - baseline) / baseline * 100.0
        return 0.0

    # -----------------------------------------------------------------------
    # Collect per-trap metrics from the results list.
    # -----------------------------------------------------------------------

    # Each entry is a dict describing one trap.  Keys populated below:
    #   'label'              — e.g. "Trap 5"
    #   'has_pulse'          — bool
    #   'pre_*' / 'post_*'  — normalized area change per region (one or two phases)
    #   'sol_pre_mean/std'   — mean ± std of solidity in the pre-pulse (or full) window
    #   'sol_post_mean/std'  — mean ± std of solidity in the post-pulse window (NaN if no pulse)
    collected = []

    for res in sorted(all_results, key=lambda x: x['trap_index']):
        if res.get('status') not in ('success', 'detection_only', 'fit_failed'):
            continue
        data = res.get('data', {})

        sol   = np.array(data.get('solidity', []), dtype=float)
        a_p   = np.array(data.get('area_prot', []), dtype=float)
        a_b   = np.array(data.get('area_body', []), dtype=float)
        a_t   = np.array(data.get('area', []),      dtype=float)

        # Need at least some valid solidity frames to proceed.
        valid = np.where(~np.isnan(sol))[0]
        if len(valid) < 3:
            continue

        start_idx = int(valid[0])

        # End index: rupture frame (exclusive) or last valid frame.
        r_idx = data.get('rupture_idx')
        if r_idx is not None and 0 < r_idx < len(sol):
            end_idx = int(r_idx)
        else:
            end_idx = int(valid[-1]) + 1

        # Pulse index: where the electroporation pulse was applied (0-based).
        pulse_idx = data.get('pulse_idx')    # None when no pulse
        has_pulse = (
            pulse_idx is not None
            and start_idx < pulse_idx < end_idx   # must fall inside the valid aspiration window
        )

        entry = {
            'label':     f"Trap {data['trap_index'] + 1}",
            'has_pulse': has_pulse,
        }

        if has_pulse:
            # --- Pre-pulse window: start → pulse ---
            entry['pre_prot']  = _norm_delta(a_p, start_idx, pulse_idx)
            entry['pre_body']  = _norm_delta(a_b, start_idx, pulse_idx)
            entry['pre_tot']   = _norm_delta(a_t, start_idx, pulse_idx)

            # --- Post-pulse window: pulse → end, normalized to area AT the pulse ---
            # We shift the baseline to the pulse frame so the post-pulse change is
            # measured relative to the cell's state immediately after the pulse.
            entry['post_prot'] = _norm_delta(a_p, pulse_idx, end_idx)
            entry['post_body'] = _norm_delta(a_b, pulse_idx, end_idx)
            entry['post_tot']  = _norm_delta(a_t, pulse_idx, end_idx)

            # Solidity statistics per phase
            sol_pre  = sol[start_idx:pulse_idx]
            sol_post = sol[pulse_idx:end_idx]
            entry['sol_pre_mean']  = float(np.nanmean(sol_pre))  if len(sol_pre)  > 0 else np.nan
            entry['sol_pre_std']   = float(np.nanstd(sol_pre))   if len(sol_pre)  > 1 else 0.0
            entry['sol_post_mean'] = float(np.nanmean(sol_post)) if len(sol_post) > 0 else np.nan
            entry['sol_post_std']  = float(np.nanstd(sol_post))  if len(sol_post) > 1 else 0.0
        else:
            # --- Full window: start → end ---
            entry['pre_prot'] = _norm_delta(a_p, start_idx, end_idx)
            entry['pre_body'] = _norm_delta(a_b, start_idx, end_idx)
            entry['pre_tot']  = _norm_delta(a_t, start_idx, end_idx)

            sol_full = sol[start_idx:end_idx]
            entry['sol_pre_mean'] = float(np.nanmean(sol_full)) if len(sol_full) > 0 else np.nan
            entry['sol_pre_std']  = float(np.nanstd(sol_full))  if len(sol_full) > 1 else 0.0
            # No post-pulse data
            entry['sol_post_mean'] = np.nan
            entry['sol_post_std']  = np.nan

        collected.append(entry)

    if not collected:
        logger.warning("plot_aggregate_metrics: no valid traps found, skipping plots.")
        return

    # Detect whether ANY trap has a pulse (controls how bars/points are laid out).
    any_pulse = any(e['has_pulse'] for e in collected)

    # ======================================================================
    # PLOT 1: Cell Area Dynamics — three horizontal panels, one per region.
    # ======================================================================
    # Splitting into panels lets each region use its own y-axis scale, so the
    # small cell body / total changes are not crushed by the large protrusion signal.
    #
    # Color coding: Protrusion = medium red, Cell Body = light blue, Total = black.
    # Phase coding: solid fill = pre-pulse (or full trace); hatched = post-pulse.

    from matplotlib.patches import Patch  # proxy artists for the legend

    c_prot = utils.MFA_COLORS['secondary']   # medium red
    c_body = utils.MFA_COLORS['tertiary']    # light blue
    c_tot  = utils.MFA_COLORS['primary']     # black (use dark_blue so hatching is visible)

    n = len(collected)
    labels = [e['label'] for e in collected]
    x = np.arange(n)

    # Each panel: (region label, pre-key, post-key, bar colour)
    panels = [
        ('Protrusion',  'pre_prot', 'post_prot', c_prot),
        ('Cell Body',   'pre_body', 'post_body', c_body),
        ('Total',       'pre_tot',  'post_tot',  c_tot),
    ]

    fig_area, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(max(18, n * 2.0), 6),
        sharey=False          # independent y-axes so each panel uses its own scale
    )

    # Bar geometry — two bars per trap when pulse exists, one otherwise.
    if any_pulse:
        bar_w   = 0.35        # width of each individual bar
        off_pre  = -bar_w / 2
        off_post =  bar_w / 2
    else:
        bar_w   = 0.55
        off_pre  = 0.0        # single centred bar

    for ax, (region_name, pre_key, post_key, colour) in zip(axes, panels):

        pre_vals  = [e[pre_key]               for e in collected]
        post_vals = [e.get(post_key, 0.0)     for e in collected]

        kw_solid = dict(width=bar_w, edgecolor='black', linewidth=0.8)
        kw_hatch = dict(width=bar_w, edgecolor='black', linewidth=0.8,
                        hatch='//', alpha=0.80)

        ax.bar(x + off_pre, pre_vals, color=colour, **kw_solid)

        if any_pulse:
            ax.bar(x + off_post, post_vals, color=colour, **kw_hatch)

        ax.axhline(0, color='black', linewidth=1.2)
        ax.set_title(region_name, fontsize=13, fontweight='bold')
        ax.set_ylabel("Normalized Area Change (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Add a subtle grid on y only to ease reading
        ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

    # Shared legend on the top-right panel
    if any_pulse:
        legend_handles = [
            Patch(facecolor=utils.MFA_COLORS['light_grey'], edgecolor='black',
                  label='Pre-pulse  (norm. to frame 0)'),
            Patch(facecolor=utils.MFA_COLORS['light_grey'], edgecolor='black',
                  hatch='//', alpha=0.80,
                  label='Post-pulse  (norm. to pulse frame)'),
        ]
        axes[-1].legend(handles=legend_handles, fontsize=9, loc='upper right')
        suptitle = f"Cell Area Dynamics (Pre- vs. Post-Pulse) — {experiment_id}"
    else:
        suptitle = f"Cell Area Dynamics (Start vs. Rupture) — {experiment_id}"

    fig_area.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    utils.save_plot_pdf(output_dir / f"{experiment_id}_Aggregate_Delta_Area.png")
    plt.close(fig_area)

    # ======================================================================
    # PLOT 2: Cell Body Solidity — scatter (mean ± std per trap)
    # ======================================================================
    # Each trap gets one point (no pulse) or two adjacent points (pre/post pulse).
    # X positions are spaced so the two phases are clearly distinct but still
    # visually grouped per trap.

    fig_sol, ax_sol = plt.subplots(figsize=(max(10, n * 1.1), 5))

    COLOR_PRE  = utils.MFA_COLORS['dark_blue']   # pre-pulse (or full trace)
    COLOR_POST = utils.MFA_COLORS['medium_red']  # post-pulse
    JITTER_HALF = 0.15   # half-separation between pre/post points within a trap group

    x_ticks = []       # one tick per trap (centred between pre/post if pulse)
    x_labels = []

    for i, e in enumerate(collected):
        base_x = float(i)

        if any_pulse:
            x_pre  = base_x - JITTER_HALF
            x_post = base_x + JITTER_HALF
        else:
            x_pre = base_x

        # Pre-pulse (or full) point
        m_pre = e['sol_pre_mean']
        s_pre = e['sol_pre_std']
        if not np.isnan(m_pre):
            ax_sol.errorbar(x_pre, m_pre, yerr=s_pre,
                            fmt='o', color=COLOR_PRE,
                            markersize=7, capsize=4, linewidth=1.5,
                            zorder=3)

        # Post-pulse point (only when pulse exists for this trap)
        if any_pulse and e['has_pulse']:
            m_post = e['sol_post_mean']
            s_post = e['sol_post_std']
            if not np.isnan(m_post):
                ax_sol.errorbar(x_post, m_post, yerr=s_post,
                                fmt='s', color=COLOR_POST,
                                markersize=7, capsize=4, linewidth=1.5,
                                zorder=3)

        x_ticks.append(base_x)
        x_labels.append(e['label'])

    # Legend
    if any_pulse:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_PRE,
                   markersize=9, label='Pre-pulse'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_POST,
                   markersize=9, label='Post-pulse'),
        ]
        ax_sol.legend(handles=legend_handles)

    ax_sol.set_xticks(x_ticks)
    ax_sol.set_xticklabels(x_labels, rotation=45, ha='right')
    ax_sol.set_ylabel("Cell Body Solidity (mean ± std)")
    ax_sol.set_ylim(bottom=0)

    title_suffix = "Pre- vs. Post-Pulse" if any_pulse else "Full Aspiration Trace"
    ax_sol.set_title(f"Cell Body Solidity — {title_suffix} — {experiment_id}")

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / f"{experiment_id}_Aggregate_Delta_Solidity.png")
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
        pulse_idx = max(0, int(dye_params.get('pulse_frame', 10)) - 1)
        t_first = valid_results[0]['data']['time']
        if 0 <= pulse_idx < len(t_first):
            pulse_time = t_first[pulse_idx]
    
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
            utils.save_plot_pdf(output_dir / f"{experiment_id}_Aggregate_Uptake{filename_suffix}.png")
        plt.close(fig)

    # Only the absolute aggregate uptake plot is retained.  The _Normalized
    # (dF/F0) and _MinMax variants were removed because normalization is
    # now the responsibility of downstream / bulk code, and rescaling here
    # was creating three variants of essentially the same information.
    _plot_aggregated_3panel("", "Dye Uptake (Absolute)", "Mean Intensity (a.u.)", "_Absolute")
        
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
        
    utils.save_plot_pdf(save_path)
    plt.close(fig)