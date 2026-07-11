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
from scipy.stats import mannwhitneyu

import bulk_file_handling as bfh
import Utils_MFA as utils
from Utils_MFA import MFA_COLORS

logger = logging.getLogger(__name__)

# =============================================================================
# 1. VISUAL CONFIGURATION
# =============================================================================

# NOTE: Style is controlled centrally by Utils_MFA.set_paper_style().
# Do NOT call plt.style.use() or sns.set_context() at import time here:
# any caller that runs set_paper_style() after importing this module would
# still see seaborn's context scaling from "talk", producing inconsistent
# fonts and gridlines across figures.

PALETTE_REGION = {
    "Body": "#1f77b4",       # Blue
    "Protrusion": "#ff7f0e", # Orange
    "Total": "#333333"       # Dark Grey/Black
}

MI_MODEL_PALETTE = {
    "Linear":    "#e377c2",  # Pink
    "Power-Law": "#17becf",  # Cyan
}

# Viscoelastic model palette. The three models are nested by complexity:
#   Kelvin-Voigt (2 params)  ⊂  Jeffreys (3 params)  ⊂  Burgers (4 params)
# This is an ordinal gradient, not an opposition, so we use a sequential
# single-hue ramp (the MFA_COLORS blues, light -> dark) rather than a
# divergent palette. Perceptually monotonic: darker = more complex.
VISCO_MODEL_PALETTE = {
    "Kelvin-Voigt": MFA_COLORS['light_blue'],   # 2 params (simplest)
    "Jeffreys":     MFA_COLORS['medium_blue'],  # 3 params
    "Burgers":      MFA_COLORS['dark_blue'],    # 4 params (most complex)
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

def align_time_to_pulse(time_array: np.ndarray, pulse_time_s: float, condition_type: str) -> np.ndarray:
    if len(time_array) == 0:
        return time_array
    if condition_type == "ASP":
        return time_array - time_array[0]
    return time_array - pulse_time_s

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

def _compute_common_uptake_xlim(
        traps: List[bfh.TrapData],
        rupture_fraction: float = 0.20,
        max_pre_s: float = 30.0,
) -> Tuple[float, float]:
    """
    Computes a consistent x-axis range (in pulse-aligned seconds) for uptake
    multipanel plots so that all panels in a group share the same window.

    Algorithm
    ---------
    Post-pulse end (x_max):
        Collect the maximum aligned time from each trap's uptake array.
        Discard traces whose post-pulse duration is below
        ``rupture_fraction * group_median`` (these are truncated recordings
        that would otherwise impose an unreasonably short window).
        x_max = minimum of the remaining durations.

    Pre-pulse start (x_min):
        For EP data, the aligned time starts at a large negative value
        (aspiration phase before the pulse).  Showing hundreds of seconds of
        baseline context is uninformative and makes the interesting post-pulse
        region tiny.  x_min is therefore capped at ``-max_pre_s`` (default
        -30 s), still showing enough baseline to confirm the pre-pulse signal
        is flat.
        For ASP data (time zeroed to cell entry), x_min = 0.

    Parameters
    ----------
    traps           : all TrapData objects for one condition group
    rupture_fraction: fraction of group median below which a trace is
                      considered too short to contribute to setting x_max
    max_pre_s       : maximum pre-pulse context to show for EP data [s]

    Returns
    -------
    (x_min, x_max) : floats.  Falls back to (0, 1) if no valid data found.
    """
    post_durations = []
    pre_starts     = []

    for trap in traps:
        ud = trap.uptake_data
        if 'Time_s' not in ud:
            continue
        # FETCH PHYSICAL PULSE TIME
        pf = trap.metadata.pulse_frame
        t_raw = trap.protrusion_data.get('Time_s', [])
        pulse_time_s = float(t_raw[pf]) if (0 < pf < len(t_raw)) else 0.0

        at = align_time_to_pulse(ud['Time_s'], pulse_time_s, trap.metadata.condition_type)
        if len(at) < 2:
            continue
        t_max = float(np.nanmax(at))
        t_min = float(np.nanmin(at))
        if t_max > 0:
            post_durations.append(t_max)
        pre_starts.append(t_min)

    # --- x_max: common post-pulse end ---
    if not post_durations:
        return 0.0, 1.0
    median_post = float(np.median(post_durations))
    threshold   = rupture_fraction * median_post
    normal      = [d for d in post_durations if d >= threshold]
    x_max       = float(min(normal)) if normal else float(min(post_durations))

    # --- x_min: pre-pulse context ---
    # For EP, show a fixed window of at most max_pre_s before the pulse.
    # Collect the actual median start time so we don't request more data
    # than any trace actually has (avoids empty whitespace on the left).
    if pre_starts:
        median_pre = float(np.median(pre_starts))
        x_min = max(median_pre, -max_pre_s)
    else:
        x_min = 0.0

    return x_min, x_max


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
# STATISTICAL ANNOTATION HELPERS
# -----------------------------------------------------------------------------

def _mw_stars(vals_a: np.ndarray, vals_b: np.ndarray) -> Tuple[str, float]:
    """
    Runs a two-sided Mann-Whitney U test and returns a star-rating plus the
    raw p-value.

    Returns
    -------
    stars : str
        "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), or "ns" (not significant
        or sample too small).
    p     : float
        The raw p-value (1.0 if the test could not be run).
    """
    # Guard against tiny groups — the U test is unreliable below n≈5 per side.
    if len(vals_a) < 5 or len(vals_b) < 5:
        return "ns", 1.0
    try:
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        stars = ("***" if p < 0.001 else
                 ("**" if p < 0.01  else
                  ("*"  if p < 0.05  else "ns")))
        return stars, p
    except Exception:
        return "ns", 1.0


def _cliffs_delta(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    """
    Cliff's Delta: how large is the difference between two groups, in a
    non-parametric sense?

    Imagine picking one observation from A and one from B at random. Cliff's
    Delta is the probability that A > B minus the probability that A < B.
    The result ranges from -1 to +1:
        +1  → A always larger than B
        -1  → B always larger than A
         0  → the two distributions overlap completely.

    Computed by broadcasting: every A × B pair is compared once, which is
    much faster than a nested Python loop for the trap counts we work with.
    """
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a == 0 or n_b == 0:
        return 0.0

    a_col = vals_a[:, np.newaxis]   # shape (n_a, 1)
    b_row = vals_b[np.newaxis, :]   # shape (1, n_b)
    dominance = np.sign(a_col - b_row)   # +1 / 0 / -1 per pair
    return float(dominance.sum() / (n_a * n_b))


def _delta_linewidth(delta: float) -> float:
    """
    Maps |Cliff's Delta| to a bracket line width, encoding effect size
    visually. Thresholds follow Romano et al. (2006):

        |δ| < 0.147  → 1.5 px  (negligible)
        |δ| < 0.330  → 2.5 px  (small)
        |δ| < 0.474  → 3.5 px  (medium)
        |δ| ≥ 0.474  → 4.8 px  (large)

    The floor (1.5 px) is deliberately above 1 px so that even negligible
    effects remain visible after the figure is shrunk to the printed A5 size.
    """
    abs_d = abs(delta)
    if abs_d < 0.147:
        return 1.5
    if abs_d < 0.330:
        return 2.5
    if abs_d < 0.474:
        return 3.5
    return 4.8


def _add_bracket(ax, x1: float, x2: float, y_top: float, label: str,
                 lw: float = 0.9, color: str = 'black',
                 fontsize: Optional[float] = None, inset: float = 0.13) -> None:
    """
    Draws a single significance bracket between two categories on `ax`.

    A flat horizontal bar is drawn just above `y_top`, and the stars
    annotation sits above the bar. The bar's line width encodes Cliff's
    Delta (passed in as `lw`), and the stars encode the p-value tier — so
    the reader gets both "is it real?" and "how big is it?" from one mark.

    Handles both linear and log y-axes: on log axes, the vertical offsets
    are computed in log space and mapped back to data coordinates, so the
    bracket floats visually the same distance above the data regardless of
    whether the axis spans 0–1 or 10⁰–10⁵.

    Parameters
    ----------
    x1, x2   : x-positions of the two boxes being compared (category indices).
    y_top    : data y-value at the top of the taller box/data max.
    label    : star string ("***", "**", "*"), or None to skip drawing.
    lw       : bracket line width, from _delta_linewidth().
    inset    : how far to pull each endpoint in from x1/x2, in category units,
               so the bracket connects the boxes rather than spanning the full
               category-to-category distance.
    """
    if label is None:
        return
    if fontsize is None:
        fontsize = 9  # matches Utils_MFA paper style for annotations

    # Pull the endpoints in so the bracket sits over the boxes, not the gaps.
    x1_drawn = x1 + inset if x2 > x1 else x1 - inset
    x2_drawn = x2 - inset if x2 > x1 else x2 + inset

    y_lo, y_hi = ax.get_ylim()

    if ax.get_yscale() == 'log':
        # Work in log space so vertical offsets look uniform on a log axis.
        # A small additive offset in log10 is a multiplicative offset in data:
        # 0.08 in log10 ≈ ×1.20, 0.13 in log10 ≈ ×1.35.
        log_lo, log_hi = np.log10(y_lo), np.log10(y_hi)
        log_range = log_hi - log_lo
        y_line = 10 ** (np.log10(y_top) + log_range * 0.08)
        y_text = 10 ** (np.log10(y_top) + log_range * 0.13)
        needed_top = 10 ** (np.log10(y_text) + log_range * 0.08)
    else:
        y_range = y_hi - y_lo
        y_line  = y_top + y_range * 0.08
        y_text  = y_top + y_range * 0.13
        needed_top = y_text + y_range * 0.08

    # Single flat horizontal bar — no vertical tick lines (matches GUV style).
    ax.plot([x1_drawn, x2_drawn], [y_line, y_line], lw=lw, color=color)
    ax.text((x1 + x2) / 2, y_text, label,
            ha='center', va='bottom', fontsize=fontsize, color=color)

    # Expand the axis if the stars would fall off the top.
    if needed_top > y_hi:
        ax.set_ylim(y_lo, needed_top)


def _build_stat_label(vals_a: np.ndarray, vals_b: np.ndarray
                      ) -> Tuple[str, Optional[str], float, float]:
    """
    Runs the Mann-Whitney test and Cliff's Delta together, and packages the
    inputs needed by `_add_bracket`.

    Returns
    -------
    stars : str      -- "***" / "**" / "*" / "ns"
    label : str|None -- the stars string (or None if not significant, so the
                        caller can skip drawing the bracket entirely)
    lw    : float    -- bracket line width encoding |δ|
    delta : float    -- the raw Cliff's Delta value (signed), for logging
    """
    stars, _ = _mw_stars(vals_a, vals_b)
    delta = _cliffs_delta(vals_a, vals_b)
    if stars == "ns":
        return stars, None, 0.8, delta
    lw = _delta_linewidth(delta)
    return stars, stars, lw, delta


# =============================================================================
# 3. STATISTICAL PLOTS
# =============================================================================

def plot_per_trap_protrusion_distribution(grouped_data, output_dir: Path):
    import matplotlib.lines as mlines
    logger.info("Generating Plot: Per-Trap Protrusion Stability (Paired Pre/Post)...")
    
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        meta = traps[0].metadata
        cond_label = get_cond_label(meta)
        is_ep = meta.condition_type == "EP"
        
        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            p_data = trap.protrusion_data
            if 'Time_s' in p_data and 'Protrusion_Length_um' in p_data:
                t_raw = p_data['Time_s']
                l_raw = p_data['Protrusion_Length_um']
                
                if len(t_raw) == 0 or len(l_raw) == 0:
                    continue
                
                # Unique identifier to pair pre and post measurements
                exp_id = trap.metadata.experiment_number
                
                if is_ep:
                    at = align_time_to_pulse(t_raw, trap.metadata.pulse_frame, trap.metadata.condition_type)
                    pre_mask = at < 0
                    post_mask = at >= 0
                    
                    val_pre = _safe_max(l_raw[pre_mask], default=np.nan) if np.any(pre_mask) else np.nan
                    val_post = _safe_max(l_raw[post_mask], default=np.nan) if np.any(post_mask) else np.nan
                    
                    records.append({
                        'Trap': f"T{trap.trap_id}",
                        'Trap_Int': trap.trap_id,
                        'Exp_ID': exp_id,
                        'Phase': 'Pre-pulse',
                        'Length_um': val_pre,
                        'Linked_Post': val_post
                    })
                    records.append({
                        'Trap': f"T{trap.trap_id}",
                        'Trap_Int': trap.trap_id,
                        'Exp_ID': exp_id,
                        'Phase': 'Post-pulse',
                        'Length_um': val_post,
                        'Linked_Pre': val_pre
                    })
                else:
                    val_all = _safe_max(l_raw, default=np.nan)
                    if pd.notna(val_all):
                        records.append({
                            'Trap': f"T{trap.trap_id}",
                            'Trap_Int': trap.trap_id,
                            'Exp_ID': exp_id,
                            'Phase': 'Full Trace',
                            'Length_um': val_all
                        })
        
        if not records: continue
        df = pd.DataFrame(records)
        df = df.dropna(subset=['Length_um'])
        
        # Aggregate a 'Global' group to explicitly show the overall size range
        df_global = df.copy()
        df_global['Trap'] = 'All Traps'
        df_global['Trap_Int'] = 999 
        df_combined = pd.concat([df, df_global], ignore_index=True)
        
        trap_order = [f"T{t}" for t in sorted(df['Trap_Int'].unique())] + ['All Traps']
        
        plt.figure(figsize=(16, 7))
        
        if is_ep:
            palette = {'Pre-pulse': '#ff7f0e', 'Post-pulse': '#9467bd'}
            
            # Background boxplots for the distribution ranges
            sns.boxplot(data=df_combined, x='Trap', y='Length_um', hue='Phase', palette=palette, 
                        showfliers=False, boxprops=dict(alpha=0.3), order=trap_order)
            
            plotted_pairs = set()
            growth_thresh = 0.5
            trap_to_x = {t: i for i, t in enumerate(trap_order)}
            
            # Manual scatter and paired line plotting
            for _, row in df_combined.iterrows():
                x_center = trap_to_x[row['Trap']]
                is_global = row['Trap'] == 'All Traps'
                
                if row['Phase'] == 'Pre-pulse':
                    x_offset = -0.2
                    y_val = row['Length_um']
                    y_linked = row['Linked_Post']
                    
                    plt.scatter(x_center + x_offset, y_val, color=palette['Pre-pulse'], 
                                s=30, edgecolor='k', lw=0.5, zorder=3, alpha=0.8)
                    
                    pair_id = f"{row['Trap']}_{row['Exp_ID']}"
                    if pd.notna(y_linked) and pair_id not in plotted_pairs:
                        delta = y_linked - y_val
                        if delta > growth_thresh: color = '#2ca02c' # Green - Grows
                        elif delta < -growth_thresh: color = '#d62728' # Red - Retracts
                        else: color = '#7f7f7f' # Gray - Stable
                        
                        # Isolate line drawing to individual traps, omit for the global summary
                        if not is_global:
                            plt.plot([x_center - 0.2, x_center + 0.2], [y_val, y_linked], 
                                     color=color, alpha=0.5, lw=1.5, zorder=2)
                        plotted_pairs.add(pair_id)
                        
                elif row['Phase'] == 'Post-pulse':
                    x_offset = 0.2
                    plt.scatter(x_center + x_offset, row['Length_um'], color=palette['Post-pulse'], 
                                s=30, edgecolor='k', lw=0.5, zorder=3, alpha=0.8)

            handles = [
                mlines.Line2D([], [], color='#ff7f0e', marker='o', lw=0, label='Pre-pulse Max'),
                mlines.Line2D([], [], color='#9467bd', marker='o', lw=0, label='Post-pulse Max'),
                mlines.Line2D([], [], color='#2ca02c', lw=2, label=f'Grows (> {growth_thresh} µm)'),
                mlines.Line2D([], [], color='#7f7f7f', lw=2, label='Stable'),
                mlines.Line2D([], [], color='#d62728', lw=2, label=f'Retracts (< -{growth_thresh} µm)')
            ]
            plt.legend(handles=handles, title='Phase / Behavior', loc='upper right')
            
        else:
            sns.boxplot(data=df_combined, x='Trap', y='Length_um', color='lightgray', 
                        showfliers=False, boxprops=dict(alpha=0.4), order=trap_order)
            sns.stripplot(data=df_combined, x='Trap', y='Length_um', color='black', size=6, 
                          jitter=True, edgecolor='black', linewidth=0.8, alpha=0.9, order=trap_order)

        # Draw visual separator for the global summary column
        plt.axvline(len(trap_order) - 1.5, color='black', ls='--', lw=1, alpha=0.5)

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

        # Compute shared x-axis window once — same logic as the other
        # uptake multipanels.  Replaces the old hardcoded set_xlim(-5, 120).
        x_min, x_max = _compute_common_uptake_xlim(traps)

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
                if 'Time_s' in ud and 'Body_VolNorm' in ud:
                    # FETCH PHYSICAL PULSE TIME
                    pf = trap.metadata.pulse_frame
                    t_raw = trap.protrusion_data.get('Time_s', [])
                    pulse_time_s = float(t_raw[pf]) if (0 < pf < len(t_raw)) else 0.0

                    at = align_time_to_pulse(ud['Time_s'], pulse_time_s, trap.metadata.condition_type)
                    raw['t'].append(at)
                    raw['b'].append(ud['Body_VolNorm'])
                    if 'Protrusion_VolNorm' in ud:
                        raw['p'].append(ud['Protrusion_VolNorm'])
                    else:
                        raw['p'].append(np.full_like(ud['Body_VolNorm'], np.nan))
                    if 'Total_VolNorm' in ud:
                        raw['tot'].append(ud['Total_VolNorm'])
                    else:
                        raw['tot'].append(np.full_like(ud['Body_VolNorm'], np.nan))

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

            # Use only signal within the common window for the y-scale
            win = (ct >= x_min) & (ct <= x_max)
            if np.any(win):
                arr1 = stats['body_mean'][win] + stats['body_std'][win]
                arr2 = stats['prot_mean'][win] + stats['prot_std'][win]
                arr3 = stats['total_mean'][win] + stats['total_std'][win]
                
                valid_vals = np.concatenate([
                    arr1[np.isfinite(arr1)], 
                    arr2[np.isfinite(arr2)], 
                    arr3[np.isfinite(arr3)]
                ])
                if len(valid_vals) > 0:
                    current_max = float(np.nanpercentile(valid_vals, 99))
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
            ax.set_xlim(x_min, x_max)
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
        fig.text(0.01, 0.5, 'Intensity / Volume (ADU/µm³)', va='center', rotation='vertical', fontsize=16)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        save_name = f"Uptake_MeanSD_{cond_label}.png"
        plt.savefig(output_dir / save_name, dpi=PANEL_DPI)
        plt.close()


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
                ('Body_VolNorm',       'Body'),
                ('Protrusion_VolNorm', 'Protrusion'),
                ('Total_VolNorm',      'Total'),
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
    plt.ylabel("τ (s)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "Uptake_Exponential_TimeConstant.png", dpi=SAVE_DPI)
    plt.close()

def plot_per_trap_uptake_distribution(grouped_data, output_dir: Path):
    """
    Per-trap boxplot of the fitted exponential amplitude A (ADU/µm³) for each
    region (Body, Protrusion, Total).

    Uses volume-normalised intensity (Body_VolNorm / Protrusion_VolNorm /
    Total_VolNorm) and the fitted amplitude A from fit_exponential_uptake(),
    matching what is stored in the mechanics CSV.  Volume normalisation removes
    the cell-size confound so amplitude is proportional to membrane permeability,
    not cell size.
    """
    import bulk_mechanics as bm

    logger.info("Generating Plot: Body vs Protrusion Uptake Amplitude (ADU/µm³) per Trap...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps:
            continue
        cond_label = get_cond_label(traps[0].metadata)

        records = []
        for trap in sorted(traps, key=lambda t: t.trap_id):
            ud = trap.uptake_data
            if 'Time_s' not in ud:
                continue

            # pulse_frame is only meaningful for EP cells; for ASP use 0
            pf = (trap.metadata.pulse_frame
                  if trap.metadata.condition_type == "EP"
                  else 0)

            # Map each region label to its volume-normalised column name
            region_cols = [
                ('Body',       'Body_VolNorm'),
                ('Protrusion', 'Protrusion_VolNorm'),
                ('Total',      'Total_VolNorm'),
            ]

            for region_name, col_name in region_cols:
                if col_name not in ud or len(ud[col_name]) == 0:
                    continue

                # Fit A*(1 - exp(-t/tau)) to the absolute signal.
                # fit_exponential_uptake subtracts the pre-pulse baseline
                # internally, so we pass the raw absolute array directly.
                fit = bm.fit_exponential_uptake(
                    ud['Time_s'], ud[col_name], pulse_frame=pf
                )

                # Only keep the amplitude if the fit converged
                if fit['A'] is not None:
                    records.append({
                        'Trap'   : f"T{trap.trap_id}",
                        'Amp_A'  : fit['A'],
                        'Region' : region_name,
                    })

        if not records:
            continue

        df = pd.DataFrame(records)
        df['Trap'] = df['Trap'].astype('category')

        # Floor at 1 ADU so that near-zero fits don't break the log scale.
        # Values below 1 ADU are indistinguishable from baseline noise anyway.
        df['Amp_A'] = df['Amp_A'].clip(lower=1e-3)

        # Sort trap labels strictly by integer value (T2, T3, … T18)
        trap_order = sorted(df['Trap'].unique(), key=lambda x: int(x[1:]))

        plt.figure(figsize=(16, 7))
        sns.boxplot(
            data=df, x='Trap', y='Amp_A', hue='Region',
            palette=PALETTE_REGION,
            showfliers=False, boxprops=dict(alpha=0.4),
            order=trap_order,
        )
        sns.stripplot(
            data=df, x='Trap', y='Amp_A', hue='Region',
            palette=PALETTE_REGION,
            size=6, jitter=True, dodge=True,
            edgecolor='black', linewidth=0.8, alpha=0.9,
            order=trap_order,
        )

        plt.yscale('log')
        plt.ylim(bottom=0.5)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:3], labels[:3], title='Region', loc='upper right')

        plt.ylabel("Fitted Amplitude A (ADU/µm³, volume-normalised)")
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

        # Compute shared x-axis window once per group so every panel uses
        # the same time range.  This mirrors the common-duration logic in
        # bulk_mechanics and prevents long-recording outliers from compressing
        # all other panels.
        x_min, x_max = _compute_common_uptake_xlim(traps)
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        if n == 0: continue
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        _add_global_legend(fig, mode='uptake')
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=14)
        fig.text(0.01, 0.5, 'Intensity / Volume (ADU/µm³)', va='center', rotation='vertical', fontsize=14)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            ax.set_xlim(x_min, x_max)
            
            for trap in trap_list:
                ud = trap.uptake_data
                if 'Time_s' not in ud: continue
                
                # FETCH PHYSICAL PULSE TIME
                pf = trap.metadata.pulse_frame
                t_raw = trap.protrusion_data.get('Time_s', [])
                pulse_time_s = float(t_raw[pf]) if (0 < pf < len(t_raw)) else 0.0
                
                at = align_time_to_pulse(ud['Time_s'], pulse_time_s, trap.metadata.condition_type)
                pf_for_fit = pulse_time_s if trap.metadata.condition_type == "EP" else 0.0
                
                def _plot_fit(col_name, region_name):
                    if col_name not in ud: return
                    y_data = ud[col_name]
                    color = PALETTE_REGION[region_name]
                    ax.plot(at, y_data, '.', ms=2, color=color, alpha=0.3)
                    
                    # PASS PULSE_TIME_S TO FIT
                    fit = bm.fit_exponential_uptake(ud['Time_s'], y_data, pulse_time_s=pf_for_fit)
                    if fit['tau'] is not None:
                        t_post = at[at > 0]
                        baseline = fit.get('baseline', 0.0) or 0.0
                        curve = baseline + bm._exp_uptake(t_post, fit['A'], fit['tau'])
                        ax.plot(t_post, curve, ls=STYLE_DEFAULT, color=color, lw=1.5)
                        y_pos = 0.9 if region_name == 'Body' else (0.8 if region_name == 'Protrusion' else 0.7)
                        ax.text(0.05, y_pos, f"{region_name[0]}: τ={fit['tau']:.1f}",
                                transform=ax.transAxes, fontsize=7, color=color)

                _plot_fit('Body_VolNorm',       'Body')
                _plot_fit('Protrusion_VolNorm',  'Protrusion')
                _plot_fit('Total_VolNorm',        'Total')

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

                # FETCH PHYSICAL PULSE TIME
                pf = trap.metadata.pulse_frame
                pulse_time_s = float(time_raw[pf]) if (0 < pf < len(time_raw)) else 0.0
                
                at = align_time_to_pulse(time_raw, pulse_time_s, trap.metadata.condition_type)
                
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

        # Shared x-axis window for all panels in this group
        x_min, x_max = _compute_common_uptake_xlim(traps)
        
        trap_groups = _group_by_trap_id(traps)
        sorted_ids = sorted(trap_groups.keys())
        n = len(sorted_ids); cols = 5; rows = math.ceil(n / cols)
        if n == 0: continue
        
        global_max_y = 0.0
        for trap in traps:
            ud = trap.uptake_data
            if 'Time_s' not in ud: continue

            pf = trap.metadata.pulse_frame
            t_raw = trap.protrusion_data.get('Time_s', [])
            pulse_time_s = float(t_raw[pf]) if (0 < pf < len(t_raw)) else 0.0

            at = align_time_to_pulse(ud['Time_s'], pulse_time_s, trap.metadata.condition_type)
            win_mask = (at >= x_min) & (at <= x_max)
            
            for col in ['Body_VolNorm', 'Protrusion_VolNorm', 'Total_VolNorm']:
                if col in ud and len(ud[col]) > 0:
                    vals = ud[col][win_mask]
                    valid_vals = vals[np.isfinite(vals)]
                    if len(valid_vals) > 0:
                        # Use 99th percentile to ignore massive single-frame volume spikes
                        current_max = float(np.nanpercentile(valid_vals, 99))
                        if current_max > global_max_y: global_max_y = current_max
        
        y_limit = global_max_y * 1.1 if global_max_y > 0 else 1.0

        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
        
        _add_global_legend(fig, mode='uptake')
        fig.text(0.5, 0.01, 'Time from Pulse (s)', ha='center', fontsize=16)
        fig.text(0.02, 0.5, 'Intensity / Volume (ADU/µm³)', va='center', rotation='vertical', fontsize=16)
        
        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            trap_list = trap_groups[tid]
            ax.set_title(f"Trap {tid}", fontsize=10, fontweight='bold')
            ax.set_ylim(-0.1, y_limit)
            ax.set_xlim(x_min, x_max)
            
            for trap in trap_list:
                ud = trap.uptake_data
                if 'Time_s' not in ud: continue

                # FETCH PHYSICAL PULSE TIME
                pf = trap.metadata.pulse_frame
                t_raw = trap.protrusion_data.get('Time_s', [])
                pulse_time_s = float(t_raw[pf]) if (0 < pf < len(t_raw)) else 0.0

                at = align_time_to_pulse(ud['Time_s'], pulse_time_s, trap.metadata.condition_type)
                
                if 'Body_VolNorm' in ud:
                    ax.plot(at, ud['Body_VolNorm'], ls=STYLE_DEFAULT, color=PALETTE_REGION['Body'], lw=1.5, alpha=0.7)
                if 'Protrusion_VolNorm' in ud:
                    ax.plot(at, ud['Protrusion_VolNorm'], ls=STYLE_DEFAULT, color=PALETTE_REGION['Protrusion'], lw=1.5, alpha=0.7)
                if 'Total_VolNorm' in ud:
                    ax.plot(at, ud['Total_VolNorm'], ls=STYLE_DEFAULT, color=PALETTE_REGION['Total'], lw=1.5, alpha=0.7)
                    
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

    # Pre-compute reduced labels across all ASP conditions so filenames and
    # titles drop fields that are constant across the plotted set (see
    # _reduce_labels). Only ASP groups are considered because the loop
    # below filters non-ASP conditions.
    _asp_full_labels = [
        get_cond_label(traps[0].metadata)
        for key, traps in grouped_data.items()
        if traps and traps[0].metadata.condition_type == "ASP"
    ]
    _asp_label_map = _reduce_labels(sorted(set(_asp_full_labels)))

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue

        meta = traps[0].metadata
        if meta.condition_type != "ASP": continue

        cond_label = get_cond_label(meta)
        cond_label_short = _asp_label_map.get(cond_label, cond_label)
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
            mlines.Line2D([], [], color=VISCO_MODEL_PALETTE['Kelvin-Voigt'], lw=2, label='Kelvin-Voigt'),
            mlines.Line2D([], [], color=VISCO_MODEL_PALETTE['Jeffreys'],     lw=2, label='Jeffreys'),
            mlines.Line2D([], [], color=VISCO_MODEL_PALETTE['Burgers'],      lw=2, label='Burgers')
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
                    ax.plot(t_smooth, l_pred, color=VISCO_MODEL_PALETTE['Kelvin-Voigt'], lw=1.5, alpha=0.8)
                if 'Jeffreys' in models_info and models_info['Jeffreys']['params']:
                    p = models_info['Jeffreys']['params']
                    l_pred = bm._jeffreys(t_smooth, r_eff, meta.pressure, C, p['E'], p['eta1'], p['eta2'])
                    ax.plot(t_smooth, l_pred, color=VISCO_MODEL_PALETTE['Jeffreys'], lw=1.5, alpha=0.8)
                if 'Burgers' in models_info and models_info['Burgers']['params']:
                    p = models_info['Burgers']['params']
                    l_pred = bm._burgers(t_smooth, r_eff, meta.pressure, C, p['E1'], p['eta1'], p['E2'], p['eta2'])
                    ax.plot(t_smooth, l_pred, color=VISCO_MODEL_PALETTE['Burgers'], lw=1.5, alpha=0.8)

                best = visco['best_model']
                ax.text(0.05, 0.9, f"Winner: {best}", transform=ax.transAxes, fontsize=8, fontweight='bold', color=VISCO_MODEL_PALETTE.get(best, 'black'))
                
        for j in range(len(sorted_ids), len(axes)): axes[j].axis('off')
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        utils.save_plot_pdf(output_dir / f"Viscoelastic_Fits_Panel_{cond_label_short}.pdf", dpi=PANEL_DPI)
        plt.close()


# =============================================================================
# 9. ASP-SPECIFIC PLOTS  (reduced grouping: CellType × Treatment × Pressure)
# =============================================================================

def _asp_category_label(meta: bfh.ExperimentMetadata) -> str:
    return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa"


def _asp_category_label_from_row(row: pd.Series) -> str:
    return f"{row['Cell_Type']}_{row['Treatment']}_{row['Pressure_Pa']}Pa"


def _reduce_labels(labels: List[str], sep: str = '_') -> Dict[str, str]:
    """
    Compress a set of underscore-separated labels to show only varying fields.

    Splits each label into components on `sep`, checks which positions vary
    across the list, and returns a mapping from each full label to a
    reduced label that keeps only the varying positions.

    Examples
    --------
    Only the treatment varies:
        _reduce_labels(['MDAMB231_CytD_1100Pa', 'MDAMB231_WT_1100Pa'])
        -> {'MDAMB231_CytD_1100Pa': 'CytD',
            'MDAMB231_WT_1100Pa':   'WT'}

    Two fields vary:
        _reduce_labels(['MDAMB231_CytD_1100Pa', 'HeLa_WT_1100Pa'])
        -> {'MDAMB231_CytD_1100Pa': 'MDAMB231_CytD',
            'HeLa_WT_1100Pa':       'HeLa_WT'}

    Safe fallbacks (returns labels unchanged) if:
      - the list is empty or has one element,
      - the labels do not all have the same number of components,
      - no position varies (all labels identical).
    """
    if len(labels) < 2:
        return {l: l for l in labels}
    parts = [l.split(sep) for l in labels]
    n = len(parts[0])
    if not all(len(p) == n for p in parts):
        return {l: l for l in labels}
    varying = [i for i in range(n) if len({p[i] for p in parts}) > 1]
    if not varying:
        return {l: l for l in labels}
    return {l: sep.join(parts[k][i] for i in varying) for k, l in enumerate(labels)}


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

    # Use the shared viscoelastic palette so colours are consistent with the
    # parameter boxplots and the model selection frequency chart.
    model_color = VISCO_MODEL_PALETTE

    # Pre-compute reduced labels across the full set of pools. Each figure's
    # SUPTITLE uses the reduced label (e.g. "CytD" instead of
    # "MDAMB231_CytD_1100Pa") so the varying field is what a reader sees.
    # Filenames keep the full label -- they need to stay unique and
    # descriptive when files are viewed outside the thesis.
    _pool_labels_full = sorted(asp_pools.keys())
    _pool_label_map = _reduce_labels(_pool_labels_full)

    for cat_label in _pool_labels_full:
        cat_label_short = _pool_label_map[cat_label]
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
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        utils.save_plot_pdf(output_dir / f"ASP_BestFit_Panel_{cat_label_short}.pdf", dpi=PANEL_DPI)
        plt.close()


def plot_asp_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Boxplots of viscoelastic parameters for ASP cells, grouped by the reduced
    key (CellType × Treatment × Pressure). Points coloured by winning model.
    Also produces a model selection frequency bar chart.
    """
    logger.info("Generating: ASP Parameter Boxplots (reduced grouping)...")

    # Filter cascade for the plotted set. All four conditions must hold:
    #   1. Condition_Type == 'ASP'
    #   2. Best_Model is set (a viscoelastic winner exists after BIC selection,
    #      which already excludes bound-hit fits and models skipped by the
    #      identifiability guard).
    #   3. E_Pa is finite (a real parameter, not NaN).
    #   4. Visco_R2_Flag is True (the winning model clears the R² floor set
    #      at pipeline level -- currently r2_floor=0.85).
    # The unfiltered CSV (mechanics_results_all_traps.csv) preserves the audit
    # trail; only the plot itself is filtered.
    df = mechanics_df[
        (mechanics_df['Condition_Type'] == 'ASP') &
        mechanics_df['Best_Model'].notna() &
        mechanics_df['E_Pa'].notna() &
        (mechanics_df['Visco_R2_Flag'] == True)
    ].copy()

    if df.empty:
        logger.info("  No fitted ASP data — skipping.")
        return

    df['Category'] = df.apply(_asp_category_label_from_row, axis=1)
    sorted_cats_full = sorted(df['Category'].unique())

    # Compress labels to only the fields that vary across the plotted set.
    # If only "treatment" differs, "MDAMB231_CytD_1100Pa" -> "CytD".
    # If more differs, more is kept. Falls through to the full label when
    # only one category is present.
    label_map = _reduce_labels(sorted_cats_full)
    df['Category'] = df['Category'].map(label_map)
    sorted_cats = [label_map[c] for c in sorted_cats_full]

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

        sns.boxplot(data=sub, x='Category', y=col, order=sorted_cats,
                    ax=ax, showfliers=False, color='lightgray')
        sns.stripplot(data=sub, x='Category', y=col, order=sorted_cats,
                      hue='Best_Model',
                      palette={'Kelvin-Voigt': '#1f77b4',
                               'Jeffreys':     '#ff7f0e',
                               'Burgers':      '#2ca02c'},
                      dodge=False, alpha=0.6, ax=ax, size=5)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

        # Log-scale y-axis: viscoelastic parameters span orders of magnitude,
        # and linear axes compress the low-end distribution into an
        # unreadable strip along the baseline.
        if (sub[col] > 0).all():
            ax.set_yscale('log')

        # -------------------------------------------------------------
        # Pairwise significance annotation between adjacent categories.
        # Each bracket carries two independent signals:
        #     stars  -> Mann-Whitney U p-value tier
        #     lw     -> Cliff's Delta magnitude (visual effect size)
        # Log-scale-aware offsets are handled inside _add_bracket.
        # -------------------------------------------------------------
        pairs = [(sorted_cats[j], sorted_cats[j + 1])
                 for j in range(len(sorted_cats) - 1)]

        for cat_a, cat_b in pairs:
            vals_a = sub.loc[sub['Category'] == cat_a, col].dropna().values
            vals_b = sub.loc[sub['Category'] == cat_b, col].dropna().values
            stars, label, lw, delta = _build_stat_label(vals_a, vals_b)

            # Log the numeric result so the values that go into the thesis
            # text ("p = 0.017, δ = 0.38") are always traceable to a run.
            n_a, n_b = len(vals_a), len(vals_b)
            logger.info(
                f"  [{title.splitlines()[0]:<30}] {cat_a} vs {cat_b}: "
                f"n=({n_a},{n_b})  stars={stars}  delta={delta:+.3f}"
            )

            if label is None:
                continue  # not significant, no bracket drawn

            x1 = sorted_cats.index(cat_a)
            x2 = sorted_cats.index(cat_b)
            # The visible top of the data is the taller group's max.
            # (Whiskers/points can extend above the box; anchoring to nanmax
            # keeps the bracket clear of the fliers as well.)
            y_top = float(np.nanmax(np.concatenate([vals_a, vals_b])))
            _add_bracket(ax, x1, x2, y_top, label, lw=lw)

        if i != 0:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "ASP_Parameter_Boxplots.pdf", dpi=SAVE_DPI)
    plt.close()

    # --- Model selection frequency bar chart ---
    model_counts = (df.groupby(['Category', 'Best_Model'])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(sorted_cats, fill_value=0))
    cols_present = [c for c in ["Kelvin-Voigt", "Jeffreys", "Burgers"] if c in model_counts.columns]

    model_counts[cols_present].plot(
        kind='bar', stacked=True,
        figsize=(max(6, len(sorted_cats) * 1.5), 5),
        color=[VISCO_MODEL_PALETTE[c] for c in cols_present])
    plt.ylabel("Number of Traps")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "ASP_Model_Selection_Frequency.pdf", dpi=SAVE_DPI)
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
    sorted_conds_full = sorted(df['Cond'].unique())

    # Compress labels to only the fields that vary (see _reduce_labels).
    label_map = _reduce_labels(sorted_conds_full)
    df['Cond'] = df['Cond'].map(label_map)
    sorted_conds = [label_map[c] for c in sorted_conds_full]

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
        axes[0].set_ylim(-2, 2)
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
        plt.ylim(-2, 2)
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
    tau_col_map = {'Uptake_Body_VolNorm_tau': 'Body', 'Uptake_Prot_VolNorm_tau': 'Protrusion', 'Uptake_Total_VolNorm_tau': 'Total'}
    df = mechanics_df.copy()
    df['Cond'] = df.apply(_cond_label_from_row, axis=1)

    df_melt = df.melt(id_vars=['Cond', 'Condition_Type'],
                      value_vars=list(tau_col_map.keys()),
                      var_name='Region_raw', value_name='Uptake_Tau_s').dropna(subset=['Uptake_Tau_s'])
    df_melt['Region'] = df_melt['Region_raw'].map(tau_col_map)
    med_tau = df_melt['Uptake_Tau_s'].median()
    df_melt = df_melt[df_melt['Uptake_Tau_s'] <= med_tau * 20]

    if df_melt.empty: return
    sorted_conds_full = sorted(df_melt['Cond'].unique())

    # Compress axis labels to only the fields that vary across the plotted set
    # (see _reduce_labels). Filenames and internal columns keep the full label.
    label_map = _reduce_labels(sorted_conds_full)
    df_melt['Cond'] = df_melt['Cond'].map(label_map)
    sorted_conds = [label_map[c] for c in sorted_conds_full]

    plt.figure(figsize=(max(10, len(sorted_conds) * 2.5), 6))
    sns.boxplot(data=df_melt, x='Cond', y='Uptake_Tau_s', hue='Region',
                palette=PALETTE_REGION, order=sorted_conds, showfliers=False)
    sns.stripplot(data=df_melt, x='Cond', y='Uptake_Tau_s', hue='Region',
                  dodge=True, palette='dark:black', alpha=0.4, legend=False,
                  order=sorted_conds, size=4)
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
    plt.ylabel("τ (s)")
    plt.xlabel("Region")
    plt.legend(title='Condition Type', loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "Uptake_Tau_ASP_vs_EP.png", dpi=SAVE_DPI)
    plt.close()

def plot_spearman_correlation(df_scalars: pd.DataFrame, output_dir: Path) -> None:
    # cols_to_drop = [
    #     'Prot_Frame', 'Prot_Protrusion_Length_px', 'Prot_Linear_Size_Prot_um',
    #     'Prot_Linear_Size_Body_um', 'Prot_Linear_Size_Total_um', 'Prot_Total_Area_um2',
    #     'Prot_Volume_Prot_um3', 'Prot_Volume_Body_um3', 'Prot_Volume_Total_um3',
    #     'Prot_Downstream_Intensity', 'Uptake_Protrusion_Area_um2',
    #     'Uptake_Body_Area_um2', 'Uptake_Total_Area_um2', 'Uptake_Linear_Size_Prot_um',
    #     'Uptake_Linear_Size_Body_um', 'Uptake_Linear_Size_Total_um',
    #     'Uptake_Volume_Prot_um3', 'Uptake_Volume_Body_um3', 'Uptake_Volume_Total_um3',
    #     'Uptake_Total_Intensity', 'Uptake_Protrusion_Intensity', 'Uptake_Tip_Intensity',
    #     'Uptake_Body_Intensity', 'Uptake_Total_MinMax', 'Uptake_Protrusion_MinMax',
    #     'Uptake_Tip_MinMax', 'Uptake_Body_MinMax', 'Uptake_Total_Normalized_dF_F0', 'Trap_ID']
    
    
    nice_names = {
        'Duration_ms': 'Pulse Duration [ms]',
        'Trap_ID': 'Trap No.',
        'Prot_Body_Solidity': 'Body Solidity',
        'Prot_Body_Area_um2': 'Body Area [µm^2]',
        'Prot_Protrusion_Area_um2': 'Protrusion Area [µm^2]',
        'Prot_Protrusion_Length_um': 'Protrusion Length [µm]',
        'Uptake_Body_VolNorm_A': 'Body Uptake A (ADU/µm³)',
        'Uptake_Prot_VolNorm_A': 'Protrusion Uptake A (ADU/µm³)',
        'Uptake_Total_VolNorm_A': 'Total Uptake A (ADU/µm³)',
    }
    
    # Explicitly select only the desired columns in the specified order
    cols_to_keep = [col for col in nice_names.keys() if col in df_scalars.columns]
    
    # We must also keep 'Condition' for the loop later, so we grab it separately
    # but don't rename it in the nice_names dict.
    df_clean = df_scalars[['Condition'] + cols_to_keep].copy() if 'Condition' in df_scalars.columns else df_scalars[cols_to_keep].copy()
    df_clean = df_clean.rename(columns=nice_names)
    
    def _plot_and_save(df_subset: pd.DataFrame, filename_prefix: str, title: str):
        df_numeric = df_subset.select_dtypes(include=[np.number])
        df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]
        
        if df_numeric.shape[1] < 2: 
            return
            
        corr_matrix = df_numeric.corr(method='spearman')
        
        # Generate mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1,
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Spearman ρ"}, ax=ax)
        
        if ax.figure.axes[-1]:
            ax.figure.axes[-1].yaxis.label.set_size(16)
            ax.figure.axes[-1].tick_params(labelsize=14)
        
        ax.set_title(title, pad=20, weight='bold', fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_prefix}.png", dpi=300)
        plt.close(fig)
        corr_matrix.to_csv(output_dir / f"{filename_prefix}.csv")

    logger.info("Generating Plot: Spearman Correlation (Combined)...")
    _plot_and_save(
        df_clean, 
        "spearman_correlation_heatmap_combined", 
        "Spearman Correlation of Morphological and Fluorescent Metrics (All Data)"
    )

    if 'Condition' in df_clean.columns:
        for cond in sorted(df_clean['Condition'].dropna().unique()):
            df_cond = df_clean[df_clean['Condition'] == cond]
            safe_cond = str(cond).replace(' ', '_')
            
            if len(df_cond) >= 3:
                logger.info(f"Generating Plot: Spearman Correlation for {cond}...")
                _plot_and_save(
                    df_cond, 
                    f"spearman_correlation_heatmap_{safe_cond}", 
                    f"Spearman Correlation - {cond}"
                )
            else:
                logger.debug(f"Skipping correlation for {cond}: insufficient data points (n={len(df_cond)}).")
    
def plot_parameter_pairplot(df_scalars: pd.DataFrame, output_dir: Path) -> None:
    nice_names = {
        'Trap_ID': 'Trap No.',
        'Prot_Body_Solidity': 'Body Solidity',
        'Prot_Body_Area_um2': 'Body Area [µm^2]',
        'Prot_Protrusion_Area_um2': 'Protrusion Area [µm^2]',
        'Prot_Protrusion_Length_um': 'Protrusion Length [µm]',
        'Uptake_Body_VolNorm_A': 'Body Uptake A (ADU/µm³)',
        'Uptake_Prot_VolNorm_A': 'Protrusion Uptake A (ADU/µm³)',
        'Uptake_Total_VolNorm_A': 'Total Uptake A (ADU/µm³)',
        'Duration_ms': 'Pulse Duration [ms]',
    }
    
    cols_to_keep = [col for col in nice_names.keys() if col in df_scalars.columns]
    df_clean = df_scalars[['Condition'] + cols_to_keep].copy() if 'Condition' in df_scalars.columns else df_scalars[cols_to_keep].copy()
    df_clean = df_clean.rename(columns=nice_names)
    
    def _plot_and_save_pairplot(df_subset: pd.DataFrame, filename: str, title: str):
        # 1. Force numeric conversion and float casting
        df_num = df_subset.copy()
        for col in df_num.columns:
            if col != 'Condition':
                df_num[col] = pd.to_numeric(df_num[col], errors='coerce').astype(float)
        
        df_numeric = df_num.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        # 2. Filter out constants (like Pulse Duration) using variance threshold
        # Handles floating point noise (e.g. 0.1 vs 0.1000000001)
        cols_with_variance = []
        for col in df_numeric.columns:
            if df_numeric[col].dropna().nunique() > 1:
                # Use epsilon threshold of 10^-9
                if (df_numeric[col].max() - df_numeric[col].min()) > 1e-9:
                    cols_with_variance.append(col)
        
        df_numeric = df_numeric[cols_with_variance]
        
        if df_numeric.shape[1] < 2: 
            return

        # 3. Generate the plot
        g = sns.pairplot(df_numeric, corner=True, diag_kind='kde',
                         plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5})
        

        # --- THE "FORCE-MANUAL" LABEL OVERRIDE ---
        # We iterate through the leftmost axes of every row
        for i, col_name in enumerate(df_numeric.columns):
            ax = g.axes[i, 0] # Leftmost axis of row i
            if ax is not None:
                # Set the label explicitly
                ax.set_ylabel(col_name, fontsize=12, fontweight='bold', rotation=90)
                
                # CRITICAL: Diagonal axes have labelleft=False by default in corner mode.
                # We turn on the axis ticks for the anchor, but we can keep the 
                # numeric labels (0.0, 0.5, etc.) hidden if you prefer.
                ax.tick_params(axis='y', left=True, labelleft=False)
                
                # Ensure the label is actually visible and not clipped
                ax.get_yaxis().set_visible(True)

        # 4. Clean up x-axis labels (Bottom row)
        for j, col_name in enumerate(df_numeric.columns):
            ax = g.axes[-1, j]
            if ax is not None:
                ax.set_xlabel(col_name, fontsize=12, fontweight='bold')

        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(g.fig)

    # Execution logic...
    if 'Condition' in df_clean.columns:
        for cond in sorted(df_clean['Condition'].dropna().unique()):
            df_cond = df_clean[df_clean['Condition'] == cond]
            if len(df_cond) >= 3:
                _plot_and_save_pairplot(df_cond, f"spearman_parameter_pairplot_{str(cond).replace(' ', '_')}.png", f"Pair Plot - {cond}")

# =============================================================================
# HELPER: Body size column detection
# =============================================================================

def _find_body_size_column(pd_data: Dict[str, np.ndarray]) -> Optional[str]:
    """
    Searches the protrusion data dict for a column representing cell body size.

    Priority order: direct linear dimension first (preferred), then area
    columns (caller must take sqrt to convert to a linear scale).

    Returns
    -------
    str : column name, or None if nothing suitable is found.
    """
    priority = [
        'Body_Length_um',
        'Body_Major_Axis_um',
        'Cell_Body_Length_um',
        'Body_Minor_Axis_um',
        'Body_Width_um',
        'Body_Area_um2',
        'Cell_Body_Area_um2',
    ]
    for col in priority:
        if col in pd_data and len(pd_data[col]) > 0:
            return col
    # Fallback: any column whose name contains both 'body' and a size keyword
    for col in pd_data:
        col_l = col.lower()
        if 'body' in col_l and any(k in col_l for k in ('length', 'area', 'axis', 'width')):
            return col
    return None


# =============================================================================
# EP POST-PULSE BEHAVIOR PLOT
# =============================================================================

def plot_ep_post_pulse_behavior(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Two-panel stacked bar chart showing the fraction and raw count of EP traps
    classified as 'Extends', 'Stable', or 'Retracts' immediately after the
    electroporation pulse, per condition.

    The classification threshold is controlled by
    bulk_mechanics.EP_BEHAVIOR_SLOPE_THRESHOLD.  Results are also written to a
    CSV for downstream statistical testing.
    """
    logger.info("Generating: EP Post-Pulse Behavior chart...")
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df[
        (mechanics_df['Condition_Type'] == 'EP') &
        mechanics_df['EP_Post_Pulse_Behavior'].notna()
    ].copy()

    if df.empty:
        logger.info("  No EP behavior data found — skipping.")
        return

    df['Cond'] = df.apply(_cond_label_from_row, axis=1)
    sorted_conds_full = sorted(df['Cond'].unique())

    # Compress axis labels to only the fields that vary (see _reduce_labels).
    label_map = _reduce_labels(sorted_conds_full)
    df['Cond'] = df['Cond'].map(label_map)
    sorted_conds = [label_map[c] for c in sorted_conds_full]

    # Build counts table; ensure all three categories are present
    counts = (df.groupby(['Cond', 'EP_Post_Pulse_Behavior'])
                .size()
                .unstack(fill_value=0)
                .reindex(sorted_conds, fill_value=0))
    for cat in ('Extends', 'Stable', 'Retracts'):
        if cat not in counts.columns:
            counts[cat] = 0
    col_order = ['Extends', 'Stable', 'Retracts']
    counts    = counts[col_order]
    totals    = counts.sum(axis=1)
    fractions = counts.div(totals.replace(0, np.nan), axis=0).fillna(0)

    palette = {'Extends': '#2ca02c', 'Stable': '#1f77b4', 'Retracts': '#d62728'}
    colors  = [palette[c] for c in col_order]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(sorted_conds) * 2.5), 5))

    # Panel 1 — raw counts
    counts.plot(kind='bar', stacked=True, ax=axes[0], color=colors, legend=False)
    axes[0].set_title("Post-Pulse Protrusion Behavior\n(raw counts)", fontweight='bold')
    axes[0].set_ylabel("Number of Traps")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)
    # Annotate total n per bar
    for j, cond in enumerate(sorted_conds):
        n_total = int(totals.loc[cond])
        axes[0].text(j, n_total + 0.2, f"n={n_total}", ha='center',
                     va='bottom', fontsize=8)

    # Panel 2 — fractions
    fractions.plot(kind='bar', stacked=True, ax=axes[1], color=colors)
    axes[1].set_title("Post-Pulse Protrusion Behavior\n(fraction)", fontweight='bold')
    axes[1].set_ylabel("Fraction of Traps")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Behavior', loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "EP_PostPulse_Behavior.png",
                dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

    # Save count table for downstream stats
    counts['Total'] = totals
    counts.to_csv(output_dir / "EP_PostPulse_Behavior_Counts.csv")
    logger.info("  EP post-pulse behavior chart saved.")


# =============================================================================
# UPTAKE vs LINEAR SIZE — per-trap 2×2 figures
# =============================================================================

def plot_slope_comparison(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """Three-panel slope comparison: full-trace slope, EP pre-vs-post, and delta."""
    logger.info("Generating: Slope Comparison (ASP vs EP)...")
    if mechanics_df is None or mechanics_df.empty: return

    df = mechanics_df.copy()
    df['Cond'] = df.apply(_cond_label_from_row, axis=1)

    # Compress axis labels to only the fields that vary (see _reduce_labels).
    _slope_sorted_full = sorted(df['Cond'].unique())
    _slope_label_map = _reduce_labels(_slope_sorted_full)
    df['Cond'] = df['Cond'].map(_slope_label_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    palette = {'ASP': '#1f77b4', 'EP': '#d62728'}

    ax = axes[0]
    valid = df['Linear_Slope'].notna()
    if valid.any():
        sns.boxplot(data=df[valid], x='Cond', y='Linear_Slope', hue='Condition_Type',
                    palette=palette, showfliers=False, ax=ax)
        sns.stripplot(data=df[valid], x='Cond', y='Linear_Slope', hue='Condition_Type',
                      palette=palette, dodge=True, alpha=0.6, jitter=True, ax=ax, legend=False)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylim(-2, 2)
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
    ax.set_ylim(-2, 2)
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
    ax.set_ylim(-2, 2)
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

    # Pre-compute reduced labels across all base conditions so suptitles and
    # filenames drop fields that are constant across the plotted set. Each
    # figure is one condition, so the reduction is computed globally over
    # the set of conditions before the per-figure loop.
    _base_full_labels = [f"{c}_{t}_{p}Pa" for (c, t, p) in base_conditions]
    _base_label_map = _reduce_labels(sorted(set(_base_full_labels)))

    for (cell_type, treatment, pressure_pa), groups in base_conditions.items():
        asp_traps = groups['ASP']; ep_traps  = groups['EP']
        if not asp_traps and not ep_traps: continue

        full_label  = f"{cell_type}_{treatment}_{pressure_pa}Pa"
        short_label = _base_label_map.get(full_label, full_label)

        # 3 panels: Body, Protrusion, Total
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        panel_defs = [
            (axes[0], 'Body_VolNorm',       'Body'),
            (axes[1], 'Protrusion_VolNorm',  'Protrusion'),
            (axes[2], 'Total_VolNorm',       'Total'),
        ]

        for ax, col, region_label in panel_defs:
            for ctype, trap_list, color in [('ASP', asp_traps, '#1f77b4'), ('EP',  ep_traps,  '#d62728')]:
                if not trap_list: continue

                times_list, data_list = [], []
                for trap in trap_list:
                    ud = trap.uptake_data
                    if 'Time_s' not in ud or col not in ud: continue
                    
                    # FETCH PHYSICAL PULSE TIME
                    pf = trap.metadata.pulse_frame
                    t_raw = trap.protrusion_data.get('Time_s', [])
                    pulse_time_s = float(t_raw[pf]) if (0 < pf < len(t_raw)) else 0.0
                    
                    t_up_raw = ud['Time_s']
                    t_aligned = align_time_to_pulse(t_up_raw, pulse_time_s, trap.metadata.condition_type)
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
            ax.set_ylabel(f"{region_label} Intensity / Volume (ADU/µm³)")
            ax.set_title(f"{region_label} Uptake")
            ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f"Uptake_ASP_vs_EP_{short_label}.png", dpi=SAVE_DPI)
        plt.close()