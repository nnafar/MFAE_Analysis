# -*- coding: utf-8 -*-
"""
Multipanel figures: one PDF per plot type, with every analysed trap as a panel.

ROLE IN PIPELINE:
Runs once in the main process after all trap workers have finished (Phase 3
of MFA_analysis). Each worker returns the small arrays its panels need in
result['plot_data']; this module lays them out on a grid. The panels are
drawn with the same helpers as the single-trap figures in Plotting_MFA, so
both always look the same.

Within a figure, panels share their axes and colour scale, so traps can be
compared directly.

Output (per experiment):
    Protrusion Length Detection/<id>_Detection_Traces.pdf      length + haze per trap
    Protrusion Length Detection/<id>_Membrane_Kymographs.pdf   membrane kymograph per trap
    Dye Uptake/<id>_Uptake_Kymographs.pdf            dye kymograph per trap
    Dye Uptake/<id>_Diffusion_Profiles.pdf           dye profiles per trap
    Actin Quantification/<id>_Actin_Kymographs.pdf   actin kymograph per trap
    Actin Quantification/<id>_Actin_Zones.pdf        zone dynamics per trap
"""

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import Plotting_MFA as pm
import Utils_MFA as utils

logger = logging.getLogger(__name__)

PANEL_W_IN = 3.4      # width of one trap panel (inches)
PANEL_H_IN = 2.8      # height of one trap panel (inches)

# Fixed, compact font sizes: these figures are on-screen overviews of many
# small panels, not single thesis figures, so the print-scaled sizes from
# Utils_MFA.PLOT_STYLE would be far too large here.
_GRID_RC = {
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize":  9,
    "axes.titlesize":  10,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
}


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def plot_trap_multipanels(all_results: List[Dict[str, Any]], subdirs: Dict[str, Path],
                          experiment_id: str, params: Dict[str, Any]) -> List[Path]:
    """
    Write every multipanel figure for which at least one trap has data.
    A failure in one figure is logged and does not stop the others.
    Returns the paths of the PDFs that were written.
    """
    traps = sorted((r for r in all_results if r.get('plot_data')),
                   key=lambda r: r['trap_index'])
    if not traps:
        logger.info("No trap data for multipanel figures.")
        return []

    ncols = params.get('plotting_parameters', {}).get('multipanel_columns', 4)
    scale = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
    dye_p = params.get('dye_uptake_parameters', {})
    show_dye_pulse = dye_p.get('has_pulse', True)

    # 'protrusion' is the Protrusion Length Detection folder ('tracking_visuals'
    # was its previous key; still accepted).
    prot_dir = subdirs.get('protrusion', subdirs.get('tracking_visuals'))
    jobs: List[Tuple[str, Callable[..., Optional[plt.Figure]], Path, dict]] = [
        ("Detection traces", _detection_traces,
         prot_dir / f"{experiment_id}_Detection_Traces.pdf", {}),
        ("Membrane kymographs", _membrane_kymographs,
         prot_dir / f"{experiment_id}_Membrane_Kymographs.pdf", {}),
        ("Uptake kymographs", _uptake_kymographs,
         subdirs['dye'] / f"{experiment_id}_Uptake_Kymographs.pdf",
         {'show_pulse': show_dye_pulse}),
        ("Diffusion profiles", _diffusion_profiles,
         subdirs['dye'] / f"{experiment_id}_Diffusion_Profiles.pdf", {}),
        ("Actin kymographs", _actin_kymographs,
         subdirs['actin'] / f"{experiment_id}_Actin_Kymographs.pdf", {}),
        ("Actin zones", _actin_zones,
         subdirs['actin'] / f"{experiment_id}_Actin_Zones.pdf", {}),
    ]

    utils.set_paper_style()
    written = []
    for label, build, path, extra in jobs:
        try:
            with mpl.rc_context(_GRID_RC):
                fig = build(traps, ncols=ncols, scale=scale, **extra)
                if fig is None:
                    continue
                fig.suptitle(f"{experiment_id}: {label}")
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, format='pdf', dpi=300)
                plt.close(fig)
            logger.info(f"Multipanel figure saved: {path.name}")
            written.append(path)
        except Exception:
            logger.warning(f"Multipanel figure '{label}' failed.", exc_info=True)
            plt.close('all')
    return written


# =============================================================================
# GRID LAYOUT HELPERS
# =============================================================================

def _grid_shape(n: int, ncols: int) -> Tuple[int, int]:
    ncols = max(1, min(ncols, n))
    return math.ceil(n / ncols), ncols


def _single_axes_grid(n: int, ncols: int):
    """One Axes per trap, all sharing x and y. Unused cells are hidden."""
    nrows, ncols = _grid_shape(n, ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(PANEL_W_IN * ncols, PANEL_H_IN * nrows),
                             layout='constrained')
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    # With hidden cells, the bottom visible panel in a column may not be on
    # the last row; make sure it still shows its x tick labels.
    for ax in flat[max(0, n - ncols):n]:
        ax.xaxis.set_tick_params(labelbottom=True)
    return fig, flat[:n]


def _stacked_axes_grid(n: int, ncols: int, height_ratios=(1.6, 1.0)):
    """
    Two vertically stacked Axes per trap (top, bottom). All tops share y with
    each other, all bottoms share y with each other, and every Axes shares x.
    """
    nrows, ncols = _grid_shape(n, ncols)
    fig = plt.figure(figsize=(PANEL_W_IN * ncols, PANEL_H_IN * 1.3 * nrows),
                     layout='constrained')
    outer = fig.add_gridspec(nrows, ncols)
    pairs = []
    first_top = first_bottom = None
    for i in range(n):
        inner = outer[i // ncols, i % ncols].subgridspec(2, 1, height_ratios=height_ratios, hspace=0.0)
        top = fig.add_subplot(inner[0], sharex=first_top, sharey=first_top)
        bottom = fig.add_subplot(inner[1], sharex=top, sharey=first_bottom)
        first_top = first_top or top
        first_bottom = first_bottom or bottom
        top.tick_params(labelbottom=False)
        pairs.append((top, bottom))
    return fig, pairs


def _figure_legend(fig, axes) -> None:
    """One legend for the whole figure, built from the labelled artists of all panels."""
    seen: Dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    # White lines (e.g. the pulse marker on dark kymographs) would be invisible
    # in a legend on a white page; show them in grey there instead.
    for label, handle in seen.items():
        if isinstance(handle, Line2D) and mpl.colors.to_hex(handle.get_color()) == '#ffffff':
            seen[label] = Line2D([], [], color='0.5', linestyle=handle.get_linestyle(),
                                 linewidth=handle.get_linewidth())
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc='outside right upper',
                   frameon=False)


def _unify_image_limits(axes) -> None:
    """Shared axes take the limits of the last image drawn; use the union of all instead."""
    extents = [im.get_extent() for ax in axes for im in ax.get_images()]
    if not extents:
        return
    x_hi = max(e[0] for e in extents)
    x_lo = min(e[1] for e in extents)
    t_max = max(e[2] for e in extents)
    t_min = min(e[3] for e in extents)
    axes[0].set_xlim(x_hi, x_lo)
    axes[0].set_ylim(t_max, t_min)


def _pooled_limits(profile_sets: List[List[np.ndarray]],
                   low_pct: float = 1.0, high_pct: float = 99.5) -> Tuple[float, float]:
    """Colour limits from the real (unpadded) values of every trap, robust to outliers."""
    values = np.concatenate([np.asarray(p, dtype=float).ravel()
                             for profiles in profile_sets for p in profiles if len(p)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [low_pct, high_pct])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _label_left_column(pairs, ncols: int, top_label: str, bottom_label: str) -> None:
    """y labels on the first panel of each row of a stacked grid."""
    for i, (top, bottom) in enumerate(pairs):
        if i % max(1, ncols) == 0:
            top.set_ylabel(top_label)
            bottom.set_ylabel(bottom_label)


def _title(ax, result) -> None:
    ax.set_title(f"Trap {result['trap_index'] + 1}")


# =============================================================================
# FIGURES
# =============================================================================

def _detection_traces(traps, ncols, scale, **_):
    usable = [r for r in traps if r['plot_data'].get('trace')]
    if not usable:
        return None
    fig, pairs = _stacked_axes_grid(len(usable), ncols)
    for r, (ax_len, ax_haze) in zip(usable, pairs):
        pd_ = r['plot_data']
        tr = pd_['trace']
        pm.draw_protrusion_trace(ax_len, ax_haze, tr['time'], tr['protrusions'],
                                 tr.get('intensities'),
                                 rupture_time=pd_.get('rupture_time'),
                                 pulse_time=pd_.get('pulse_time'), markersize=2)
        _title(ax_len, r)
    fig.supxlabel("Time since cell entry (s)")
    _label_left_column(pairs, _grid_shape(len(usable), ncols)[1], "Length (µm)", "Haze (a.u.)")
    _figure_legend(fig, [ax for pair in pairs for ax in pair])
    return fig


def _membrane_kymographs(traps, ncols, scale, **_):
    usable = [r for r in traps if r['plot_data'].get('membrane_kymo') is not None]
    if not usable:
        return None
    fig, axes = _single_axes_grid(len(usable), ncols)
    for r, ax in zip(usable, axes):
        pd_ = r['plot_data']
        k = pd_['membrane_kymo']
        pm.draw_membrane_kymograph(ax, np.asarray(k['matrix']), pd_['pipette_x'],
                                   k['protrusions_px'], pd_['trace']['time'], scale)
        _title(ax, r)
    _unify_image_limits(axes)
    fig.supxlabel(pm.POSITION_LABEL)
    fig.supylabel("Time (s)")
    _figure_legend(fig, axes)
    return fig


def _profile_kymographs(traps, ncols, scale, get_results, draw, cbar_label,
                        show_pulse=True):
    """Grid of profile-based kymographs (dye or actin) on one shared colour scale."""
    usable = []
    for r in traps:
        res = get_results(r)
        if res and res.get('spatial_profiles') and len(res.get('time_s', [])):
            usable.append((r, res))
    if not usable:
        return None
    vmin, vmax = _pooled_limits([res['spatial_profiles'] for _, res in usable])
    fig, axes = _single_axes_grid(len(usable), ncols)
    im = None
    for (r, res), ax in zip(usable, axes):
        pd_ = r['plot_data']
        im = draw(ax, res['spatial_profiles'], np.asarray(res['time_s'], dtype=float),
                  pd_['pipette_x'], scale,
                  pulse_time=pd_.get('pulse_time') if show_pulse else None,
                  rupture_time=pd_.get('rupture_time'), vmin=vmin, vmax=vmax)
        _title(ax, r)
    _unify_image_limits(axes)
    fig.colorbar(im, ax=axes, shrink=0.6, label=cbar_label)
    fig.supxlabel(pm.POSITION_LABEL)
    fig.supylabel("Time (s)")
    _figure_legend(fig, axes)
    return fig


def _uptake_kymographs(traps, ncols, scale, show_pulse=True, **_):
    return _profile_kymographs(traps, ncols, scale, lambda r: r.get('dye_data'),
                               pm.draw_uptake_kymograph, "Dye intensity (shared scale)",
                               show_pulse=show_pulse)


def _actin_kymographs(traps, ncols, scale, **_):
    return _profile_kymographs(traps, ncols, scale, lambda r: r['plot_data'].get('actin'),
                               pm.draw_actin_kymograph, "Actin intensity (shared scale)")


def _diffusion_profiles(traps, ncols, scale, **_):
    usable = [r for r in traps
              if r.get('dye_data') and r['dye_data'].get('spatial_profiles')
              and len(r['dye_data'].get('time_s', []))]
    if not usable:
        return None
    # One colour = one time point in every panel, explained by a single colourbar.
    t_max = max(float(np.max(r['dye_data']['time_s'])) for r in usable)
    t_min = min(float(np.min(r['dye_data']['time_s'])) for r in usable)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("time", utils.get_time_colormap(9))
    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max if t_max > t_min else t_min + 1)

    fig, axes = _single_axes_grid(len(usable), ncols)
    for r, ax in zip(usable, axes):
        res = r['dye_data']
        pm.draw_diffusion_profiles(ax, res['spatial_profiles'],
                                   np.asarray(res['time_s'], dtype=float),
                                   r['plot_data']['pipette_x'], scale,
                                   color_for_time=lambda t: cmap(norm(t)))
        _title(ax, r)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 shrink=0.6, label="Time (s)")
    fig.supxlabel(pm.POSITION_LABEL)
    fig.supylabel("Dye intensity (a.u.)")
    return fig


def _actin_zones(traps, ncols, scale, **_):
    usable = [r for r in traps
              if r['plot_data'].get('actin') and len(r['plot_data']['actin'].get('time_s', []))]
    if not usable:
        return None
    fig, pairs = _stacked_axes_grid(len(usable), ncols, height_ratios=(1.0, 1.0))
    for r, (ax_norm, ax_didt) in zip(usable, pairs):
        pd_ = r['plot_data']
        pm.draw_actin_zones(ax_norm, ax_didt, pd_['actin'],
                            pulse_time=pd_.get('pulse_time'),
                            rupture_time=pd_.get('rupture_time'))
        _title(ax_norm, r)
    fig.supxlabel("Time (s)")
    _label_left_column(pairs, _grid_shape(len(usable), ncols)[1], "I / $I_0$", "dI/dt (a.u./s)")
    _figure_legend(fig, [ax for pair in pairs for ax in pair])
    return fig
