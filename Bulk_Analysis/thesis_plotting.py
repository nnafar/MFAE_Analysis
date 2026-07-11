# -*- coding: utf-8 -*-
"""
Thesis Plotting Module for ASP Mechanics and Model-Independent (MI) Analysis.

ASP viscoelastic visualisations:
1. Population L(t) dynamics (Mean ± SD) and Single-cell max protrusion box plots.
2. Multipanel viscoelastic best-fit overlays.
3. Viscoelastic parameter comparison box plots.

Model-independent visualisations (ASP full trace vs EP pre-pulse window):
4. Population L(t) dynamics collapsed to ASP vs EP-pre, and per-condition
   per-trap max-protrusion box plots (mirrors #1 style).
5. Multipanel MI best-fit overlays (linear + power-law).
6. Population MI parameter box plots (Linear_Slope, PL_a, PL_b), ASP vs EP-pre.
7. Per-trap (per-pocket) MI parameter box plots, one figure per condition.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from pathlib import Path
from scipy.interpolate import interp1d

import bulk_file_handling as bfh
import bulk_mechanics as bm
import bulk_plotting as bp
import Utils_MFA as utils

logger = logging.getLogger(__name__)
utils.set_paper_style()

def _get_asp_label(meta: bfh.ExperimentMetadata) -> str:
    return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_ASP"


def _extract_asp_phase_trace(trap: bfh.TrapData):
    """
    Returns (t_zeroed, l) for the aspiration phase of a trap.

    - ASP condition: full protrusion trace, time zeroed to the first sample.
    - EP condition:  pre-pulse portion only (t < pulse), zeroed to the first
      sample of that window.

    This mirrors the extraction convention used by fit_model_independent()
    in bulk_mechanics so the L(t) curves plotted here match the traces the
    MI fits actually saw.

    Returns (None, None) if the trace is unusable.
    """
    p_data = trap.protrusion_data
    if 'Time_s' not in p_data or 'Protrusion_Length_um' not in p_data:
        return None, None

    t_raw = np.array(p_data['Time_s'], dtype=float)
    l_raw = np.array(p_data['Protrusion_Length_um'], dtype=float)

    valid = np.isfinite(t_raw) & np.isfinite(l_raw)
    t_raw = t_raw[valid]
    l_raw = l_raw[valid]

    if len(t_raw) < 5:
        return None, None

    ctype = trap.metadata.condition_type
    if ctype == "ASP":
        return t_raw - t_raw[0], l_raw

    if ctype == "EP":
        pf = trap.metadata.pulse_frame
        if not (0 < pf < len(t_raw)):
            return None, None
        t_aligned = t_raw - t_raw[pf]
        pre_mask = (t_aligned < 0) & (l_raw > 0)
        if np.sum(pre_mask) < 5:
            return None, None
        t_pre = t_aligned[pre_mask]
        l_pre = l_raw[pre_mask]
        return t_pre - t_pre[0], l_pre

    return None, None


def _mi_winner_r2(row: pd.Series) -> float:
    """R² of the model MI_Best_Model selected. NaN if no MI winner."""
    m = row.get('MI_Best_Model')
    if m == 'Linear':
        return row.get('Linear_R2', np.nan)
    if m == 'Power-Law':
        return row.get('PL_R2', np.nan)
    return np.nan

def plot_asp_protrusion_dynamics(grouped_data: Dict, output_dir: Path) -> None:
    """
    Generates isolated single-cell box plots per condition and a single
    combined population L(t) mean ± SD overlay for all ASP conditions.
    Constructs the interpolation grid using the mean time interval of the experimental data.

    NOTE: this function assumes `grouped_data` has already been filtered upstream
    to the viscoelastic-passing cohort (see master_bulk_thesis.py PHASE 3). Any
    filter cascade should live there so the L(t) plot, per-trap boxplots, and
    the viscoelastic parameter boxplot all share exactly one cell population.
    """
    logger.info("Generating Thesis Plot: Combined ASP Protrusion Dynamics and Boxplots...")

    combined_lt_data = {}

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue

        meta = traps[0].metadata
        if meta.condition_type != "ASP": continue

        # NOTE: We deliberately do NOT apply compute_common_duration() to the
        # traces going into the mean L(t) plot. That truncation is designed
        # for viscoelastic *fitting* (equal time horizon for the optimizer)
        # and its 0.95 rupture_fraction default aggressively caps every
        # trace at the shortest non-ruptured duration.
        #
        # For the plot, we instead compute the mean over the natural range
        # of each trace and truncate at the time when the fraction of cells
        # still contributing drops below MIN_FRAC_CONTRIBUTING (see below).
        # That makes the duration reflect the majority of the cohort while
        # still trimming the tail dominated by a handful of long survivors.
        cond_label = _get_asp_label(meta)

        times_list = []
        lengths_list = []
        max_lengths_records = []

        # Extract data
        for trap in sorted(traps, key=lambda t: t.trap_id):
            p_data = trap.protrusion_data
            if 'Time_s' not in p_data or 'Protrusion_Length_um' not in p_data:
                continue

            t_raw = np.array(p_data['Time_s'], dtype=float)
            l_raw = np.array(p_data['Protrusion_Length_um'], dtype=float)

            valid_mask = np.isfinite(t_raw) & np.isfinite(l_raw)
            t_raw = t_raw[valid_mask]
            l_raw = l_raw[valid_mask]

            if len(t_raw) < 5: continue

            t_zeroed = t_raw - t_raw[0]

            if len(t_zeroed) < 5: continue
            
            times_list.append(t_zeroed)
            lengths_list.append(l_raw)
            
            max_val = np.max(l_raw)
            if np.isfinite(max_val):
                max_lengths_records.append({
                    'Trap': f"T{trap.trap_id}",
                    'Trap_Int': trap.trap_id,
                    'Max_Length_um': float(max_val)
                })

        if not times_list or not max_lengths_records:
            continue
            
        # --- Generate Separate Single-Cell Box Plot ---
        df_max = pd.DataFrame(max_lengths_records)
        if not df_max.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            trap_order = [f"T{t}" for t in sorted(df_max['Trap_Int'].unique())]
            
            sns.boxplot(data=df_max, x='Trap', y='Max_Length_um', color='lightgray', 
                        showfliers=False, boxprops=dict(alpha=0.4), order=trap_order, ax=ax)
            sns.stripplot(data=df_max, x='Trap', y='Max_Length_um', color='black', size=6, 
                          jitter=True, edgecolor='black', linewidth=0.8, alpha=0.7, order=trap_order, ax=ax)
            
            ax.set_xlabel('Trap ID')
            ax.set_ylabel('Max Protrusion Length (µm)')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            utils.save_plot_pdf(output_dir / f"Thesis_Boxplot_{cond_label}.pdf")
            plt.close()
            
        # --- Calculate L(t) Data (Mean Interval Grid) ---
        all_dts = []
        for t_arr in times_list:
            if len(t_arr) > 1:
                all_dts.extend(np.diff(t_arr))
                
        if not all_dts:
            continue
            
        mean_dt = float(np.nanmean(all_dts))

        # Grid runs from 0 to the longest observed trace endpoint. Traces
        # that end earlier contribute NaN beyond their last point, which
        # nanmean/nanstd handle naturally.
        max_end = max(float(t[-1]) for t in times_list)
        t_grid = np.arange(0, max_end + mean_dt, mean_dt)
        matrix = np.full((len(times_list), len(t_grid)), np.nan)

        for i, (t_arr, l_arr) in enumerate(zip(times_list, lengths_list)):
            if len(t_arr) > 1:
                f = interp1d(t_arr, l_arr, bounds_error=False, fill_value=np.nan)
                matrix[i, :] = f(t_grid)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_L = np.nanmean(matrix, axis=0)
            sd_L   = np.nanstd(matrix, axis=0)
            n_t    = np.sum(np.isfinite(matrix), axis=0)  # cells contributing at each t

        # Truncate the plot to the time by which at least MIN_FRAC_CONTRIBUTING
        # of the starting cohort is still recording. This makes the mean L(t)
        # reflect the majority of cells rather than a handful of long tails.
        # A 0.25 threshold means we plot as far as the 25%-longest cells reach.
        MIN_FRAC_CONTRIBUTING = 0.25
        n_start = len(times_list)
        min_n_required = max(1, int(np.ceil(MIN_FRAC_CONTRIBUTING * n_start)))
        plot_mask = (
            np.isfinite(mean_L)
            & np.isfinite(sd_L)
            & (n_t >= min_n_required)
        )

        if np.any(plot_mask):
            plot_dur = float(t_grid[plot_mask][-1])
            logger.info(
                f"  Plot window for {cond_label}: {plot_dur:.1f} s "
                f"(>={MIN_FRAC_CONTRIBUTING:.0%} of n={n_start} cells still contributing)"
            )
            combined_lt_data[cond_label] = {
                't'    : t_grid[plot_mask],
                'mean' : mean_L[plot_mask],
                'sd'   : sd_L[plot_mask],
                'n_t'  : n_t[plot_mask],
                'dur'  : plot_dur,
            }

    # --- Generate Combined Population L(t) Plot ---
    if combined_lt_data:
        fig, ax = plt.subplots(figsize=(9, 6))

        available_colors = []
        if hasattr(utils, 'MFA_COLORS') and isinstance(utils.MFA_COLORS, dict):
            preferred_keys = ['dark_blue', 'red', 'green', 'orange', 'purple', 'medium_blue']
            for pk in preferred_keys:
                if pk in utils.MFA_COLORS: 
                    available_colors.append(utils.MFA_COLORS[pk])
            for val in utils.MFA_COLORS.values():
                if val not in available_colors: 
                    available_colors.append(val)
        if not available_colors:
            available_colors = plt.cm.tab10.colors

        # Legend labels: strip the "_ASP" suffix that _get_asp_label appends,
        # then compress to only the fields that vary across the plotted set
        # via bulk_plotting._reduce_labels. For a run with only treatment
        # varying, "MDAMB231_CytD_1100Pa_ASP" becomes "CytD".
        _stripped = {k: k.replace("_ASP", "") for k in combined_lt_data.keys()}
        _legend_map = bp._reduce_labels(sorted(_stripped.values()))

        global_max_t = 0
        for idx, (cond_label, data) in enumerate(combined_lt_data.items()):
            c_t = data['t']
            c_mean = data['mean']
            c_sd = data['sd']

            color = available_colors[idx % len(available_colors)]
            legend_label = _legend_map.get(_stripped[cond_label], _stripped[cond_label])

            ax.plot(c_t, c_mean, color=color, lw=2, label=f'{legend_label} Mean')
            ax.fill_between(c_t, c_mean - c_sd, c_mean + c_sd, color=color, alpha=0.2, edgecolor=None)

            if data['dur'] is not None and data['dur'] > global_max_t:
                global_max_t = data['dur']

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Protrusion Length (µm)')
        ax.legend()
        
        if global_max_t > 0:
            ax.set_xlim(-10, global_max_t)
            
        plt.tight_layout()
        utils.save_plot_pdf(output_dir / "Thesis_Combined_Protrusion_Dynamics_ASP.pdf")
        plt.close()

def plot_thesis_asp_best_fit_multipanel(all_grouped_data: Dict, output_dir: Path, r_eff: float, C: float = 1.0) -> None:
    """
    Multipanel grid for ASP cells: cleaned data + BEST viscoelastic model.
    """
    logger.info("Generating Thesis Plot: ASP Best-Fit Multipanel...")
    bp.plot_asp_best_fit_multipanel(all_grouped_data, output_dir, r_eff, C)

def plot_thesis_asp_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Boxplots of viscoelastic parameters (E, eta) for ASP cells.
    """
    logger.info("Generating Thesis Plot: ASP Viscoelastic Parameters...")
    bp.plot_asp_parameter_boxplots(mechanics_df, output_dir)

def run_thesis_plots(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path, r_eff: float):
    """
    Executes the thesis plot suite.

    Each plot step is wrapped in its own try/except so that a failure in one
    (e.g. a PermissionError from a PDF viewer holding an old file open on
    Windows) does not cancel the remaining plots. Failures are logged with a
    traceback and the suite continues.
    """
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Thesis plot '{label}' failed: {e}", exc_info=False)

    _try("ASP Protrusion Dynamics",
         plot_asp_protrusion_dynamics, grouped_data, output_dir)
    _try("ASP Best-Fit Multipanel",
         plot_thesis_asp_best_fit_multipanel, grouped_data, output_dir, r_eff, C=1.0)
    _try("ASP Parameter Boxplots",
         plot_thesis_asp_parameter_boxplots, mechanics_df, output_dir)


# =============================================================================
# MODEL-INDEPENDENT (MI) THESIS PLOTS
#
# ASP full trace vs EP pre-pulse window. Callers pass grouped_data that has
# already been filtered to the MI-passing cohort (MI_Best_Model set AND
# winner R² >= r2_floor). See master_bulk_thesis.py for the filter.
# =============================================================================

# Two-way condition palette for MI plots. Blue anchors "aspiration only" as
# the mechanical reference; red anchors the "pre-pulse" comparators. EP is
# split by (voltage, duration_label) so each pulse protocol gets its own
# curve on the L(t) plot; the EP red ramp is walked in order.
MI_ASP_COLOR = utils.MFA_COLORS.get('dark_blue', '#1f4e79') if hasattr(utils, 'MFA_COLORS') else '#1f4e79'
MI_EP_COLOR_RAMP = (
    [utils.MFA_COLORS[k] for k in ('dark_red', 'medium_red', 'light_red', 'pale_red')
     if k in utils.MFA_COLORS]
    if hasattr(utils, 'MFA_COLORS') else
    ['#b31529', '#d75f4c', '#f6a482', '#fddbc7']
)


def _condition_bucket(meta: bfh.ExperimentMetadata) -> str:
    """
    Bucket label used on the combined L(t) plot.

    - ASP conditions collapse to a single 'ASP' bucket (treatment pooled).
    - EP conditions split by (voltage, duration_label) so each pulse protocol
      gets its own curve. Treatment is still pooled within a protocol, matching
      the parameter-boxplot convention.
    """
    if meta.condition_type == "ASP":
        return "ASP"
    return f"EP-pre {meta.voltage}V {meta.duration_label}"


def _row_bucket(row: pd.Series) -> str:
    """Row-level analogue of _condition_bucket for mechanics_df rows."""
    if row.get('Condition_Type') == 'ASP':
        return "ASP"
    return f"EP-pre {row.get('Voltage_V')}V {row.get('Duration_label')}"


def _safe_filename(label: str) -> str:
    """Sanitise a bucket label for use in a PDF filename (spaces -> underscores)."""
    return label.replace(' ', '_')


def plot_mi_protrusion_dynamics(grouped_data: Dict, output_dir: Path) -> None:
    """
    MI-analogue of plot_asp_protrusion_dynamics.

    Produces:
      - One per-trap max-protrusion box plot per condition (e.g. one PDF for
        each ASP condition, one PDF for each EP condition). Max is taken over
        whatever window the MI fit sees: full trace for ASP, pre-pulse window
        for EP.
      - One combined L(t) mean ± SD plot with two curves: ASP (pooled across
        treatments) vs EP-pre (pooled across treatments), per user request.

    Assumes `grouped_data` has already been filtered upstream to the MI cohort.
    """
    logger.info("Generating Thesis Plot: MI Protrusion Dynamics (ASP vs EP-pre)...")

    # Traces and max-length records pooled by bucket. Buckets are:
    #   - 'ASP'                        (all ASP conditions, treatment pooled)
    #   - 'EP-pre {V}V {duration_label}' (one per unique EP pulse protocol)
    # Both dicts share the same set of keys so all downstream plots (per-trap
    # boxplot, mean L(t) curve) reflect the same cohort per bucket.
    bucketed_traces: Dict[str, list] = {}
    bucketed_max_records: Dict[str, list] = {}

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps:
            continue

        meta = traps[0].metadata
        bucket = _condition_bucket(meta)

        for trap in sorted(traps, key=lambda t: t.trap_id):
            t_zeroed, l = _extract_asp_phase_trace(trap)
            if t_zeroed is None:
                continue

            bucketed_traces.setdefault(bucket, []).append((t_zeroed, l))

            max_val = np.max(l)
            if np.isfinite(max_val):
                bucketed_max_records.setdefault(bucket, []).append({
                    'Trap':          f"T{trap.trap_id}",
                    'Trap_Int':      trap.trap_id,
                    'Max_Length_um': float(max_val),
                })

    # --- Per-trap max protrusion boxplot, one per bucket ---
    for bucket, records in bucketed_max_records.items():
        df_max = pd.DataFrame(records)
        if df_max.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        trap_order = [f"T{t}" for t in sorted(df_max['Trap_Int'].unique())]

        sns.boxplot(data=df_max, x='Trap', y='Max_Length_um', color='lightgray',
                    showfliers=False, boxprops=dict(alpha=0.4),
                    order=trap_order, ax=ax)
        sns.stripplot(data=df_max, x='Trap', y='Max_Length_um', color='black',
                      size=6, jitter=True, edgecolor='black', linewidth=0.8,
                      alpha=0.7, order=trap_order, ax=ax)

        ax.set_title(bucket, fontweight='bold')
        ax.set_xlabel('Trap ID')
        ax.set_ylabel('Max Protrusion Length (µm)')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        utils.save_plot_pdf(
            output_dir / f"Thesis_MI_Boxplot_{_safe_filename(bucket)}.pdf"
        )
        plt.close()

    # --- Combined L(t) mean ± SD: ASP + one curve per EP protocol ---
    if not bucketed_traces:
        return

    # Deterministic curve order: ASP first, then EP protocols sorted
    # alphabetically (which happens to sort by voltage first, then duration
    # label, for the standard "EP-pre {V}V {ms|us}" pattern).
    ep_buckets = sorted(k for k in bucketed_traces.keys() if k != "ASP")
    ordered_buckets = (["ASP"] if "ASP" in bucketed_traces else []) + ep_buckets

    # Assign colors: ASP gets the dark blue anchor; EP protocols walk the red
    # ramp in order. If there are more EP protocols than ramp entries, the
    # extras cycle back to the start of the ramp.
    def _bucket_color(bucket_label: str, ep_index: int) -> str:
        if bucket_label == "ASP":
            return MI_ASP_COLOR
        if not MI_EP_COLOR_RAMP:
            return 'red'
        return MI_EP_COLOR_RAMP[ep_index % len(MI_EP_COLOR_RAMP)]

    fig, ax = plt.subplots(figsize=(9, 6))
    global_max_t = 0.0

    ep_counter = 0
    for bucket in ordered_buckets:
        traces = bucketed_traces[bucket]
        if not traces:
            continue

        # Interpolation grid uses the mean sample interval across all traces
        # in this bucket, then runs to the longest trace endpoint. Same
        # convention as plot_asp_protrusion_dynamics.
        all_dts = []
        for t_arr, _ in traces:
            if len(t_arr) > 1:
                all_dts.extend(np.diff(t_arr))
        if not all_dts:
            continue

        mean_dt = float(np.nanmean(all_dts))
        max_end = max(float(t[-1]) for t, _ in traces)
        t_grid = np.arange(0, max_end + mean_dt, mean_dt)

        matrix = np.full((len(traces), len(t_grid)), np.nan)
        for i, (t_arr, l_arr) in enumerate(traces):
            if len(t_arr) > 1:
                f = interp1d(t_arr, l_arr, bounds_error=False, fill_value=np.nan)
                matrix[i, :] = f(t_grid)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_L = np.nanmean(matrix, axis=0)
            sd_L   = np.nanstd(matrix, axis=0)
            n_t    = np.sum(np.isfinite(matrix), axis=0)

        # Truncate the curve at the time by which at least 25% of the starting
        # cohort is still recording. Prevents a handful of long tails from
        # dominating the mean at late times.
        MIN_FRAC_CONTRIBUTING = 0.25
        n_start = len(traces)
        min_n_required = max(1, int(np.ceil(MIN_FRAC_CONTRIBUTING * n_start)))
        plot_mask = np.isfinite(mean_L) & np.isfinite(sd_L) & (n_t >= min_n_required)

        if not np.any(plot_mask):
            continue

        c_t    = t_grid[plot_mask]
        c_mean = mean_L[plot_mask]
        c_sd   = sd_L[plot_mask]

        if bucket == "ASP":
            color = _bucket_color(bucket, 0)
        else:
            color = _bucket_color(bucket, ep_counter)
            ep_counter += 1
        ax.plot(c_t, c_mean, color=color, lw=2, label=f'{bucket} (n={n_start})')
        ax.fill_between(c_t, c_mean - c_sd, c_mean + c_sd,
                        color=color, alpha=0.2, edgecolor=None)

        logger.info(
            f"  MI L(t) window for {bucket}: 0 to {c_t[-1]:.1f} s "
            f"(>={MIN_FRAC_CONTRIBUTING:.0%} of n={n_start} traces still contributing)"
        )

        if c_t[-1] > global_max_t:
            global_max_t = float(c_t[-1])

    ax.set_xlabel('Time from aspiration start (s)')
    ax.set_ylabel('Protrusion Length (µm)')
    ax.legend()
    if global_max_t > 0:
        ax.set_xlim(-1, global_max_t)

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_Combined_Protrusion_Dynamics_MI.pdf")
    plt.close()


def plot_thesis_mi_best_fit_multipanel(grouped_data: Dict, output_dir: Path) -> None:
    """
    Multipanel grid per condition: cleaned data + Linear and Power-Law overlays,
    with the BIC winner labelled inside each panel.

    Reuses bp.plot_model_independent_fits_multipanel which already handles
    both ASP (full trace) and EP (pre-pulse window) branches.
    """
    logger.info("Generating Thesis Plot: MI Best-Fit Multipanel...")
    bp.plot_model_independent_fits_multipanel(grouped_data, output_dir)


def plot_thesis_mi_parameter_boxplots(mechanics_df: pd.DataFrame,
                                       output_dir: Path,
                                       r2_floor: float = 0.85) -> None:
    """
    Population-level MI parameter boxplots.

    X-axis: 'ASP' + one bucket per EP pulse protocol, using the same
    (voltage, duration_label) split as the combined L(t) plot. Treatments are
    pooled within a bucket.

    Cohort filter (must match the one used to build mi_filtered_grouped_data
    in master_bulk_thesis.py):
        - MI_Best_Model is set (either Linear or Power-Law fit succeeded and
          BIC picked a winner).
        - Winner's R² >= r2_floor.

    Panels: Linear_Slope, PL_a, PL_b. Points are coloured by MI_Best_Model.
    Mann-Whitney U + Cliff's Delta bracket between adjacent buckets, using the
    same convention as bp.plot_asp_parameter_boxplots.
    """
    logger.info("Generating Thesis Plot: MI Parameter Boxplots (ASP vs EP protocols)...")

    df = mechanics_df.copy()
    df['MI_Winner_R2'] = df.apply(_mi_winner_r2, axis=1)
    df = df[df['MI_Best_Model'].notna() & (df['MI_Winner_R2'] >= r2_floor)].copy()

    if df.empty:
        logger.info("  No traps pass MI cohort filter — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket, axis=1)

    # ASP first, then EP protocols sorted alphabetically (matches L(t) order).
    buckets_present = df['Bucket'].unique().tolist()
    ep_buckets = sorted(b for b in buckets_present if b != 'ASP')
    order = (['ASP'] if 'ASP' in buckets_present else []) + ep_buckets

    # Figure width scales with number of buckets so labels stay readable.
    fig_w = max(18, 3 * len(order) + 6)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 6))

    # (column, ylabel, title, log_y). PL_a is amplitude in µm; can span
    # orders of magnitude so a log axis reads more cleanly. Slope and
    # exponent stay on linear axes.
    panels = [
        ('Linear_Slope', 'Slope (µm/s)',      'Linear Slope',    False),
        ('PL_a',         'Amplitude a (µm)',  'Power-Law a',     True),
        ('PL_b',         'Exponent b',        'Power-Law b',     False),
    ]

    for i, (col, ylabel, title, log_y) in enumerate(panels):
        ax = axes[i]
        sub = df[df[col].notna()]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(data=sub, x='Bucket', y=col, order=order,
                    ax=ax, showfliers=False, color='lightgray')
        sns.stripplot(data=sub, x='Bucket', y=col, order=order,
                      hue='MI_Best_Model', palette=bp.MI_MODEL_PALETTE,
                      dodge=False, alpha=0.6, ax=ax, size=5)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

        if log_y and (sub[col] > 0).all():
            ax.set_yscale('log')

        # --- Pairwise Mann-Whitney U + Cliff's Delta between adjacent buckets ---
        # Mirrors the convention in bp.plot_asp_parameter_boxplots: each
        # bracket carries stars (p-tier) and line width (|Cliff's Delta|).
        pairs = [(order[j], order[j + 1]) for j in range(len(order) - 1)]

        for cat_a, cat_b in pairs:
            vals_a = sub.loc[sub['Bucket'] == cat_a, col].dropna().values
            vals_b = sub.loc[sub['Bucket'] == cat_b, col].dropna().values
            stars, label, lw, delta = bp._build_stat_label(vals_a, vals_b)

            logger.info(
                f"  [{title:<15}] {cat_a} vs {cat_b}: "
                f"n=({len(vals_a)},{len(vals_b)})  stars={stars}  delta={delta:+.3f}"
            )

            if label is None:
                continue

            x1 = order.index(cat_a)
            x2 = order.index(cat_b)
            y_top = float(np.nanmax(np.concatenate([vals_a, vals_b])))
            bp._add_bracket(ax, x1, x2, y_top, label, lw=lw)

        if i != 0:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_MI_Parameter_Boxplots.pdf")
    plt.close()


def plot_thesis_mi_per_trap_parameters(mechanics_df: pd.DataFrame,
                                        output_dir: Path,
                                        r2_floor: float = 0.85) -> None:
    """
    Per-pocket (per Trap ID) MI parameter boxplots, one figure per bucket.

    Buckets follow the same rule as the population boxplots and the L(t) plot:
    'ASP' (all ASP conditions pooled across treatments) and one bucket per
    EP pulse protocol 'EP-pre {V}V {duration_label}' (treatments pooled
    within a protocol).

    Each figure has three panels: Linear_Slope, PL_a, PL_b, with the boxplot
    pooling across experiments (and treatments) for each trap.

    Points are coloured by MI_Best_Model. Applies the same MI cohort filter
    as plot_thesis_mi_parameter_boxplots.
    """
    logger.info("Generating Thesis Plot: MI Per-Trap Parameters...")

    df = mechanics_df.copy()
    df['MI_Winner_R2'] = df.apply(_mi_winner_r2, axis=1)
    df = df[df['MI_Best_Model'].notna() & (df['MI_Winner_R2'] >= r2_floor)].copy()

    if df.empty:
        logger.info("  No traps pass MI cohort filter — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket, axis=1)

    panels = [
        ('Linear_Slope', 'Slope (µm/s)',       'Linear Slope',   False),
        ('PL_a',         'Amplitude a (µm)',   'Power-Law a',    True),
        ('PL_b',         'Exponent b',         'Power-Law b',    False),
    ]

    for bucket, sub_bucket in df.groupby('Bucket'):
        if sub_bucket.empty:
            continue

        # Consistent trap ordering across panels within a bucket figure.
        trap_ids_sorted = sorted(sub_bucket['Trap_ID'].dropna().astype(int).unique())
        trap_order = [f"T{t}" for t in trap_ids_sorted]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(bucket, fontweight='bold', y=1.02)

        for i, (col, ylabel, title, log_y) in enumerate(panels):
            ax = axes[i]
            sub = sub_bucket[sub_bucket[col].notna()].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            sub['Trap'] = 'T' + sub['Trap_ID'].astype(int).astype(str)

            sns.boxplot(data=sub, x='Trap', y=col, order=trap_order,
                        ax=ax, showfliers=False, color='lightgray',
                        boxprops=dict(alpha=0.4))
            sns.stripplot(data=sub, x='Trap', y=col, order=trap_order,
                          hue='MI_Best_Model', palette=bp.MI_MODEL_PALETTE,
                          dodge=False, alpha=0.7, ax=ax, size=5)

            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Trap ID')
            ax.tick_params(axis='x', rotation=45)

            if log_y and (sub[col] > 0).all():
                ax.set_yscale('log')

            if i != 0:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

        plt.tight_layout()
        utils.save_plot_pdf(
            output_dir / f"Thesis_MI_PerTrap_Params_{_safe_filename(bucket)}.pdf"
        )
        plt.close()


def run_thesis_mi_plots(mi_grouped_data: Dict,
                         mechanics_df: pd.DataFrame,
                         output_dir: Path,
                         r2_floor: float = 0.85) -> None:
    """
    Executes the MI thesis plot suite. Same try/except pattern as
    run_thesis_plots so a failure in one plot does not abort the others.

    Parameters
    ----------
    mi_grouped_data : dict pre-filtered to the MI cohort (see
        master_bulk_thesis.py). Includes both ASP and EP groups.
    mechanics_df    : unfiltered mechanics DataFrame; each MI plot function
        applies its own R² floor internally so filters stay explicit at the
        point of use.
    r2_floor        : MI winner R² threshold, shared with the master-level
        cohort filter to keep them consistent.
    """
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"MI thesis plot '{label}' failed: {e}", exc_info=False)

    _try("MI Protrusion Dynamics",
         plot_mi_protrusion_dynamics, mi_grouped_data, output_dir)
    _try("MI Best-Fit Multipanel",
         plot_thesis_mi_best_fit_multipanel, mi_grouped_data, output_dir)
    _try("MI Parameter Boxplots",
         plot_thesis_mi_parameter_boxplots, mechanics_df, output_dir, r2_floor)
    _try("MI Per-Trap Parameters",
         plot_thesis_mi_per_trap_parameters, mechanics_df, output_dir, r2_floor)