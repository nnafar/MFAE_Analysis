# -*- coding: utf-8 -*-
"""
Thesis Plotting Module for ASP Mechanics and Model-Independent (MI) Analysis.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
from pathlib import Path
from scipy.interpolate import interp1d

import bulk_file_handling as bfh
import bulk_mechanics as bm
import bulk_plotting as bp
import bulk_utils as utils

logger = logging.getLogger(__name__)
utils.set_paper_style()

def _get_asp_label(meta: bfh.ExperimentMetadata) -> str:
    return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa_ASP"

def _extract_asp_phase_trace(trap: bfh.TrapData):
    p_data = trap.protrusion_data
    if 'Time_s' not in p_data or 'Protrusion_Length_um' not in p_data:
        return None, None

    t_raw = np.array(p_data['Time_s'], dtype=float)
    l_raw = np.array(p_data['Protrusion_Length_um'], dtype=float)

    valid = np.isfinite(t_raw) & np.isfinite(l_raw)
    t_raw = t_raw[valid]
    l_raw = l_raw[valid]

    if len(t_raw) < 15:
        return None, None

    ctype = trap.metadata.condition_type
    is_post = bool(getattr(trap, 'post_pulse_entry', False))

    if ctype == "ASP" or is_post:
        return t_raw - t_raw[0], l_raw

    if ctype == "EP":
        pf = trap.metadata.pulse_frame
        if not (0 < pf < len(t_raw)):
            return None, None
        t_aligned = t_raw - t_raw[pf]
        pre_mask = (t_aligned < 0) & (l_raw > 0)
        if np.sum(pre_mask) < 15:
            return None, None
        t_pre = t_aligned[pre_mask]
        l_pre = l_raw[pre_mask]
        return t_pre - t_pre[0], l_pre

    return None, None

def _restrict_to_common_ep_treatments(df: pd.DataFrame) -> pd.DataFrame:
    if 'Condition_Type' not in df.columns or 'Treatment' not in df.columns:
        return df
    ep_treatments = set(
        df.loc[df['Condition_Type'] == 'EP', 'Treatment'].dropna().unique()
    )
    if not ep_treatments:
        return df
    keep = (
        (df['Condition_Type'] != 'ASP')
        | df['Treatment'].isin(ep_treatments)
    )
    return df[keep].copy()

def _mi_winner_r2(row: pd.Series) -> float:
    m = row.get('MI_Best_Model')
    if m == 'Linear':
        return row.get('Linear_R2', np.nan)
    if m == 'Power-Law':
        return row.get('PL_R2', np.nan)
    return np.nan

def plot_asp_protrusion_dynamics(grouped_data: Dict, output_dir: Path, global_asp_dur: Optional[float] = None) -> None:
    logger.info("Generating Thesis Plot: Combined ASP Protrusion Dynamics and Boxplots...")

    combined_lt_data = {}

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps:
            continue

        meta = traps[0].metadata

        if meta.condition_type != "ASP":
            continue

        asp_traps = [t for t in sorted(traps, key=lambda t: t.trap_id) if not getattr(t, 'post_pulse_entry', False)]
        if not asp_traps:
            continue
            
        cond_label = _get_asp_label(meta)
        
        times_list = []
        lengths_list = []
        max_lengths_records = []

        for trap in asp_traps:
            t_zeroed, l_raw = _extract_asp_phase_trace(trap)
            if t_zeroed is None or len(t_zeroed) < 15:
                continue

            if global_asp_dur is not None:
                win_mask = t_zeroed <= global_asp_dur
                t_zeroed = t_zeroed[win_mask]
                l_raw = l_raw[win_mask]

            if len(t_zeroed) < 15:
                continue

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

        df_max = pd.DataFrame(max_lengths_records)
        if not df_max.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            trap_order = [f"T{t}" for t in sorted(df_max['Trap_Int'].unique())]

            sns.boxplot(data=df_max, x='Trap', y='Max_Length_um', color='lightgray',
                        showfliers=False, boxprops=dict(alpha=0.4), order=trap_order, ax=ax)
            sns.stripplot(data=df_max, x='Trap', y='Max_Length_um', color='black', size=6,
                            jitter=True, edgecolor='black', linewidth=0.8, alpha=0.7, order=trap_order, ax=ax)

            ax.set_xlabel('Trap ID')
            ax.set_ylim(0, 40)
            ax.set_ylabel('Max Protrusion Length (µm)')
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            utils.save_plot_pdf(output_dir / f"Thesis_Boxplot_{cond_label}.pdf")
            plt.close()

        all_dts = []
        for t_arr in times_list:
            if len(t_arr) > 1:
                all_dts.extend(np.diff(t_arr))

        if not all_dts:
            continue

        mean_dt = float(np.nanmean(all_dts))

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
            n_t    = np.sum(np.isfinite(matrix), axis=0)  

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
        
        if global_asp_dur is not None:
            ax.set_xlim(-10, global_asp_dur)
            
        plt.tight_layout()
        utils.save_plot_pdf(output_dir / "Thesis_Combined_Protrusion_Dynamics_ASP.pdf")
        plt.close()

def plot_thesis_asp_best_fit_multipanel(all_grouped_data: Dict, output_dir: Path, r_eff: float, C: float = 1.0, global_asp_dur: Optional[float] = None) -> None:
    import matplotlib.lines as mlines
    import math

    logger.info("Generating Thesis Plot: ASP Best-Fit Multipanel (reduced grouping)...")

    asp_pools: Dict[str, List[bfh.TrapData]] = {}
    for key, traps in all_grouped_data.items():
        if not traps: continue
        meta = traps[0].metadata
        if meta.condition_type != "ASP": continue
        
        cat = bp._asp_category_label(meta)
        if cat not in asp_pools: asp_pools[cat] = []
        asp_pools[cat].extend(traps)

    if not asp_pools:
        logger.info("  No ASP data found — skipping.")
        return

    model_color = bp.VISCO_MODEL_PALETTE

    _pool_labels_full = sorted(asp_pools.keys())
    _pool_label_map = bp._reduce_labels(_pool_labels_full)

    for cat_label in _pool_labels_full:
        cat_label_short = _pool_label_map[cat_label]
        trap_list_all = asp_pools[cat_label]

        trap_groups = bp._group_by_trap_id(trap_list_all)
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
        dur_note = f"  [window: {global_asp_dur:.0f} s]" if global_asp_dur else ""
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

                if global_asp_dur is not None:
                    t_from_entry = t_raw - t_raw[0]
                    win_mask = t_from_entry <= global_asp_dur
                    t_raw = t_raw[win_mask]
                    l_raw = l_raw[win_mask]

                t_clean, l_clean = bm._clean_trace(t_raw, l_raw)
                if len(t_clean) < 15: continue
                t_zeroed = t_clean - t_clean[0]

                ax.plot(t_zeroed, l_clean, 'o', c='black', ms=2, alpha=0.35)

                pressure = trap.metadata.pressure
                visco = bm.fit_viscoelastic(
                    t_zeroed, l_clean, r_eff, pressure, C,
                    n_starts=3, min_points=15)

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
        utils.save_plot_pdf(output_dir / f"ASP_BestFit_Panel_{cat_label_short}.pdf")
        plt.close()

def plot_thesis_asp_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: ASP Viscoelastic Parameters...")
    bp.plot_asp_parameter_boxplots(mechanics_df, output_dir)

def run_thesis_plots(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path, r_eff: float, global_asp_dur: Optional[float] = None):
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Thesis plot '{label}' failed: {e}", exc_info=False)

    _try("ASP Protrusion Dynamics",
         plot_asp_protrusion_dynamics, grouped_data, output_dir, global_asp_dur)
    _try("ASP Best-Fit Multipanel",
         plot_thesis_asp_best_fit_multipanel, grouped_data, output_dir, r_eff, 1.0, global_asp_dur)
    _try("ASP Parameter Boxplots",
         plot_thesis_asp_parameter_boxplots, mechanics_df, output_dir)

MI_ASP_COLOR = utils.MFA_COLORS.get('dark_blue', '#1f4e79') if hasattr(utils, 'MFA_COLORS') else '#1f4e79'
MI_EP_COLOR_RAMP = (
    [utils.MFA_COLORS[k] for k in ('dark_red', 'medium_red', 'light_red', 'pale_red')
     if k in utils.MFA_COLORS]
    if hasattr(utils, 'MFA_COLORS') else
    ['#b31529', '#d75f4c', '#f6a482', '#fddbc7']
)

def _condition_bucket(meta: bfh.ExperimentMetadata) -> str:
    if meta.condition_type == "ASP":
        return "ASP"
    return f"EP-pre {meta.voltage}V {meta.duration_label}"

def _row_bucket(row: pd.Series) -> str:
    if row.get('Condition_Type') == 'ASP':
        return "ASP"
    return f"EP-pre {row.get('Voltage_V')}V {row.get('Duration_label')}"

def _safe_filename(label: str) -> str:
    return label.replace(' ', '_')

def plot_mi_protrusion_dynamics(grouped_data: Dict, output_dir: Path, global_pre_dur: Optional[float] = None) -> None:
    logger.info("Generating Thesis Plot: MI Protrusion Dynamics (ASP vs EP-pre)...")

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

            if global_pre_dur is not None:
                win_mask = t_zeroed <= global_pre_dur
                t_zeroed = t_zeroed[win_mask]
                l = l[win_mask]

            if len(t_zeroed) < 15:
                continue

            bucketed_traces.setdefault(bucket, []).append((t_zeroed, l))

            max_val = np.max(l)
            if np.isfinite(max_val):
                bucketed_max_records.setdefault(bucket, []).append({
                    'Trap':          f"T{trap.trap_id}",
                    'Trap_Int':      trap.trap_id,
                    'Max_Length_um': float(max_val),
                })

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

    if not bucketed_traces:
        return

    ep_buckets = sorted(k for k in bucketed_traces.keys() if k != "ASP")
    ordered_buckets = (["ASP"] if "ASP" in bucketed_traces else []) + ep_buckets

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

        if c_t[-1] > global_max_t:
            global_max_t = float(c_t[-1])

    ax.set_xlabel('Time from aspiration start (s)')
    ax.set_ylabel('Protrusion Length (µm)')
    ax.legend()
    if global_pre_dur is not None:
        ax.set_xlim(-1, global_pre_dur)
    elif global_max_t > 0:
        ax.set_xlim(-1, global_max_t)

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_Combined_Protrusion_Dynamics_MI.pdf")
    plt.close()

def plot_thesis_mi_best_fit_multipanel(grouped_data: Dict, output_dir: Path, global_pre_dur: Optional[float] = None) -> None:
    logger.info("Generating Thesis Plot: MI Best-Fit Multipanel...")
    bp.plot_model_independent_fits_multipanel(grouped_data, output_dir, global_pre_dur)

def plot_thesis_mi_parameter_boxplots(mechanics_df: pd.DataFrame,
                                       output_dir: Path,
                                       r2_floor: float = 0.85) -> None:
    logger.info("Generating Thesis Plot: MI Parameter Boxplots (ASP vs EP protocols)...")

    df = mechanics_df.copy()
    df['MI_Winner_R2'] = df.apply(_mi_winner_r2, axis=1)
    df = df[df['MI_Best_Model'].notna() & (df['MI_Winner_R2'] >= r2_floor)].copy()
    df = _restrict_to_common_ep_treatments(df)

    if 'Post_Pulse_Entry_Flag' in df.columns:
        df = df[~df['Post_Pulse_Entry_Flag'].astype(bool)].copy()

    if df.empty:
        logger.info("  No traps pass MI cohort filter — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket, axis=1)

    buckets_present = df['Bucket'].unique().tolist()
    ep_buckets = sorted(b for b in buckets_present if b != 'ASP')
    order = (['ASP'] if 'ASP' in buckets_present else []) + ep_buckets

    fig_w = max(18, 3 * len(order) + 6)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 6))

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

        pairs = [(order[j], order[j + 1]) for j in range(len(order) - 1)]

        for cat_a, cat_b in pairs:
            vals_a = sub.loc[sub['Bucket'] == cat_a, col].dropna().values
            vals_b = sub.loc[sub['Bucket'] == cat_b, col].dropna().values
            stars, label, lw, delta = bp._build_stat_label(vals_a, vals_b)

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
    logger.info("Generating Thesis Plot: MI Per-Trap Parameters...")

    df = mechanics_df.copy()
    df['MI_Winner_R2'] = df.apply(_mi_winner_r2, axis=1)
    df = df[df['MI_Best_Model'].notna() & (df['MI_Winner_R2'] >= r2_floor)].copy()
    df = _restrict_to_common_ep_treatments(df)

    if 'Post_Pulse_Entry_Flag' in df.columns:
        df = df[~df['Post_Pulse_Entry_Flag'].astype(bool)].copy()

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

def run_thesis_mi_prepulse_plots(mi_grouped_data: Dict,
                                 mechanics_df: pd.DataFrame,
                                 output_dir: Path,
                                 global_pre_dur: Optional[float] = None) -> None:
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"MI thesis plot '{label}' failed: {e}", exc_info=False)

    _try("MI Protrusion Dynamics",
         plot_mi_protrusion_dynamics, mi_grouped_data, output_dir, global_pre_dur)
    _try("MI Best-Fit Multipanel",
         plot_thesis_mi_best_fit_multipanel, mi_grouped_data, output_dir, global_pre_dur)
    _try("MI Parameter Boxplots",
         plot_thesis_mi_parameter_boxplots, mechanics_df, output_dir)
    _try("MI Per-Trap Parameters",
         plot_thesis_mi_per_trap_parameters, mechanics_df, output_dir)
    _try("MI Mechanics vs Uptake",
         plot_thesis_mechanics_vs_uptake_mi, mechanics_df, output_dir)
    _try("Spatial Mechanics vs Uptake",
         plot_thesis_spatial_mechanics_uptake, mechanics_df, output_dir)

def run_thesis_mi_wholetrace_plots(mi_whole_grouped_data: Dict,
                                   mechanics_df: pd.DataFrame,
                                   output_dir: Path,
                                   global_whole_dur: Optional[float] = None) -> None:
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"MI thesis plot '{label}' failed: {e}", exc_info=False)

    _try("MI Whole-Trace Parameter Boxplots", plot_thesis_mi_wholetrace_parameter_boxplots, mechanics_df, output_dir)
    _try("MI Whole-Trace Per-Trap Parameters", plot_thesis_mi_wholetrace_per_trap_parameters, mechanics_df, output_dir)
    
    if mi_whole_grouped_data:
        _try("MI Whole-Trace Protrusion Dynamics",
             plot_mi_wholetrace_protrusion_dynamics, mi_whole_grouped_data, output_dir, global_whole_dur)
        _try("MI Whole-Trace Best-Fit Multipanel",
             plot_thesis_mi_wholetrace_best_fit_multipanel, mi_whole_grouped_data, output_dir, global_whole_dur)

MI_EP_POST_COLOR_RAMP = ['#4a3269', '#6b4c9a', '#9077b5', '#b5a1cf']

def _extract_whole_trace(trap: bfh.TrapData):
    p_data = trap.protrusion_data
    if 'Time_s' not in p_data or 'Protrusion_Length_um' not in p_data:
        return None, None

    t_raw = np.array(p_data['Time_s'], dtype=float)
    l_raw = np.array(p_data['Protrusion_Length_um'], dtype=float)

    valid = np.isfinite(t_raw) & np.isfinite(l_raw)
    t_raw = t_raw[valid]
    l_raw = l_raw[valid]

    if len(t_raw) < 15:
        return None, None
    return t_raw - t_raw[0], l_raw

def _pulse_time_from_entry(trap: bfh.TrapData) -> float:
    if trap.metadata.condition_type != "EP":
        return float('nan')
    p_data = trap.protrusion_data
    if 'Time_s' not in p_data:
        return float('nan')
    t_raw = np.asarray(p_data['Time_s'], dtype=float)
    if len(t_raw) == 0:
        return float('nan')
    pf = trap.metadata.pulse_frame
    if pf < 0 or pf >= len(t_raw):
        return float('nan')
    return float(t_raw[pf] - t_raw[0])

def _trap_bucket_wholetrace(trap: bfh.TrapData) -> str:
    meta = trap.metadata
    if meta.condition_type == 'ASP':
        return 'ASP'
    is_post = bool(getattr(trap, 'post_pulse_entry', False))
    prefix = 'EP-post' if is_post else 'EP-whole'
    return f"{prefix} {meta.voltage}V {meta.duration_label}"

def _mi_whole_winner_r2(row: pd.Series) -> float:
    m = row.get('MI_Whole_Best_Model')
    if m == 'Linear':
        return row.get('MI_Whole_Linear_R2', np.nan)
    if m == 'Power-Law':
        return row.get('MI_Whole_PL_R2', np.nan)
    return np.nan

def _row_bucket_wholetrace(row: pd.Series) -> str:
    if row.get('Condition_Type') == 'ASP':
        return 'ASP'
    is_post = bool(row.get('Post_Pulse_Entry_Flag', False))
    prefix = 'EP-post' if is_post else 'EP-whole'
    return f"{prefix} {row.get('Voltage_V')}V {row.get('Duration_label')}"

def _wholetrace_bucket_color(bucket: str,
                              ep_whole_index: int,
                              ep_post_index: int) -> str:
    if bucket == 'ASP':
        return MI_ASP_COLOR
    if bucket.startswith('EP-whole'):
        return MI_EP_COLOR_RAMP[ep_whole_index % len(MI_EP_COLOR_RAMP)] if MI_EP_COLOR_RAMP else 'red'
    if bucket.startswith('EP-post'):
        return MI_EP_POST_COLOR_RAMP[ep_post_index % len(MI_EP_POST_COLOR_RAMP)]
    return 'gray'

def _wholetrace_bucket_order(buckets_present: list) -> list:
    ep_whole = sorted(b for b in buckets_present if b.startswith('EP-whole'))
    ep_post  = sorted(b for b in buckets_present if b.startswith('EP-post'))
    return (['ASP'] if 'ASP' in buckets_present else []) + ep_whole + ep_post

def plot_thesis_mi_wholetrace_parameter_boxplots(mechanics_df: pd.DataFrame,
                                                 output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: MI Whole-Trace Parameter Boxplots (ASP + EP-whole + EP-post)...")

    df = mechanics_df.copy()
    df = df[df['MI_Whole_Best_Model'].notna() & (df['MI_Whole_R2_Flag'] == True)].copy()
    df = _restrict_to_common_ep_treatments(df)

    if df.empty:
        logger.info("  No traps pass MI_Whole cohort filter — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket_wholetrace, axis=1)
    order = _wholetrace_bucket_order(df['Bucket'].unique().tolist())

    if not order:
        logger.info("  No buckets to plot — skipping.")
        return

    bucket_dur_labels = {}
    dur_col = 'MI_Whole_Window_Duration_s'
    for bucket in order:
        durs = df.loc[df['Bucket'] == bucket, dur_col].dropna().values
        if len(durs) == 0:
            bucket_dur_labels[bucket] = ''
            continue
        med = float(np.median(durs))
        q1  = float(np.percentile(durs, 25))
        q3  = float(np.percentile(durs, 75))
        lo  = float(np.min(durs))
        hi  = float(np.max(durs))
        bucket_dur_labels[bucket] = f"med {med:.0f}s\n[{lo:.0f}–{hi:.0f}]"

    fig_w = max(18, 3 * len(order) + 6)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 6))

    panels = [
        ('MI_Whole_Linear_Slope', 'Slope (µm/s)',       'Linear Slope',    False),
        ('MI_Whole_PL_a',         'Amplitude a (µm)',   'Power-Law a',     True),
        ('MI_Whole_PL_b',         'Exponent b',         'Power-Law b',     False),
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
                      hue='MI_Whole_Best_Model', palette=bp.MI_MODEL_PALETTE,
                      dodge=False, alpha=0.6, ax=ax, size=5)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

        if i == 1:
            xticks = ax.get_xticks()
            for xt, bucket in zip(xticks, order):
                caption = bucket_dur_labels.get(bucket, '')
                if not caption:
                    continue
                ax.annotate(
                    caption, xy=(xt, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -55), textcoords='offset points',
                    ha='center', va='top', fontsize=7, color='gray',
                )

        if log_y and (sub[col] > 0).all():
            ax.set_yscale('log')

        pairs = [(order[j], order[j + 1]) for j in range(len(order) - 1)]
        for cat_a, cat_b in pairs:
            vals_a = sub.loc[sub['Bucket'] == cat_a, col].dropna().values
            vals_b = sub.loc[sub['Bucket'] == cat_b, col].dropna().values
            stars, label, lw, delta = bp._build_stat_label(vals_a, vals_b)

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
    utils.save_plot_pdf(output_dir / "Thesis_MI_WholeTrace_Parameter_Boxplots.pdf")
    plt.close()

def plot_thesis_mi_wholetrace_per_trap_parameters(mechanics_df: pd.DataFrame,
                                                    output_dir: Path,
                                                    r2_floor: float = 0.85) -> None:
    logger.info("Generating Thesis Plot: MI Whole-Trace Per-Trap Parameters...")

    df = mechanics_df.copy()
    df['MI_Whole_Winner_R2'] = df.apply(_mi_whole_winner_r2, axis=1)
    df = df[df['MI_Whole_Best_Model'].notna() & (df['MI_Whole_Winner_R2'] >= r2_floor)].copy()
    df = _restrict_to_common_ep_treatments(df)

    if df.empty:
        logger.info("  No traps pass MI_Whole cohort filter — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket_wholetrace, axis=1)

    panels = [
        ('MI_Whole_Linear_Slope', 'Slope (µm/s)',      'Linear Slope',    False),
        ('MI_Whole_PL_a',         'Amplitude a (µm)',  'Power-Law a',     True),
        ('MI_Whole_PL_b',         'Exponent b',        'Power-Law b',     False),
    ]

    for bucket, sub_bucket in df.groupby('Bucket'):
        if sub_bucket.empty:
            continue

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
                          hue='MI_Whole_Best_Model', palette=bp.MI_MODEL_PALETTE,
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
            output_dir / f"Thesis_MI_WholeTrace_PerTrap_Params_{_safe_filename(bucket)}.pdf"
        )
        plt.close()

def plot_mi_wholetrace_protrusion_dynamics(mi_whole_grouped_data: Dict,
                                             output_dir: Path,
                                             global_whole_dur: Optional[float] = None) -> None:
    logger.info("Generating Thesis Plot: MI Whole-Trace Protrusion Dynamics...")

    bucketed_traces: Dict[str, list] = {}
    bucketed_pulses: Dict[str, list] = {}

    for key in sorted(mi_whole_grouped_data.keys()):
        traps = mi_whole_grouped_data[key]
        if not traps:
            continue

        for trap in sorted(traps, key=lambda t: t.trap_id):
            t_zeroed, l = _extract_whole_trace(trap)
            if t_zeroed is None or len(t_zeroed) < 15:
                continue

            if global_whole_dur is not None:
                win_mask = t_zeroed <= global_whole_dur
                t_zeroed = t_zeroed[win_mask]
                l = l[win_mask]
            
            if len(t_zeroed) < 15:
                continue

            bucket = _trap_bucket_wholetrace(trap)
            bucketed_traces.setdefault(bucket, []).append((t_zeroed, l))

            pt = _pulse_time_from_entry(trap)
            if np.isfinite(pt):
                if global_whole_dur is None or pt <= global_whole_dur:
                    bucketed_pulses.setdefault(bucket, []).append(pt)

    if not bucketed_traces:
        logger.info("  No traces to plot — skipping.")
        return

    order = _wholetrace_bucket_order(list(bucketed_traces.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    global_max_t = 0.0
    ep_whole_idx = 0
    ep_post_idx  = 0

    for bucket in order:
        traces = bucketed_traces.get(bucket, [])
        if not traces:
            continue

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

        MIN_FRAC_CONTRIBUTING = 0.25
        n_start = len(traces)
        min_n_required = max(1, int(np.ceil(MIN_FRAC_CONTRIBUTING * n_start)))
        plot_mask = np.isfinite(mean_L) & np.isfinite(sd_L) & (n_t >= min_n_required)

        if not np.any(plot_mask):
            continue

        c_t = t_grid[plot_mask]
        c_mean = mean_L[plot_mask]
        c_sd   = sd_L[plot_mask]

        if bucket == 'ASP':
            color = MI_ASP_COLOR
        elif bucket.startswith('EP-whole'):
            color = _wholetrace_bucket_color(bucket, ep_whole_idx, ep_post_idx)
            ep_whole_idx += 1
        else:
            color = _wholetrace_bucket_color(bucket, ep_whole_idx, ep_post_idx)
            ep_post_idx += 1

        ax.plot(c_t, c_mean, color=color, lw=2, label=f'{bucket} (n={n_start})')
        ax.fill_between(c_t, c_mean - c_sd, c_mean + c_sd,
                        color=color, alpha=0.2, edgecolor=None)

        if bucket.startswith('EP-whole'):
            pulses = bucketed_pulses.get(bucket, [])
            if pulses:
                med_pulse = float(np.median(pulses))
                q1_pulse  = float(np.percentile(pulses, 25))
                q3_pulse  = float(np.percentile(pulses, 75))
                if 0 < med_pulse < c_t[-1]:
                    ax.axvline(med_pulse, color=color, ls='--', lw=1.2, alpha=0.7)
                    ax.axvspan(q1_pulse, q3_pulse, color=color, alpha=0.08)

        if c_t[-1] > global_max_t:
            global_max_t = float(c_t[-1])

    ax.set_xlabel('Time from entry (s)')
    ax.set_ylabel('Protrusion Length (µm)')
    ax.legend(loc='best', fontsize=8)
    if global_whole_dur is not None:
        ax.set_xlim(-1, global_whole_dur)
    elif global_max_t > 0:
        ax.set_xlim(-1, global_max_t)

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_MI_WholeTrace_Protrusion_Dynamics.pdf")
    plt.close()

def plot_thesis_mi_wholetrace_best_fit_multipanel(mi_whole_grouped_data: Dict,
                                                    output_dir: Path,
                                                    global_whole_dur: Optional[float] = None) -> None:
    import matplotlib.lines as mlines
    import math

    logger.info("Generating Thesis Plot: MI Whole-Trace Best-Fit Multipanel...")

    PANEL_DPI = getattr(bp, 'PANEL_DPI', 200)

    for key in sorted(mi_whole_grouped_data.keys()):
        traps = mi_whole_grouped_data[key]
        if not traps:
            continue
        meta = traps[0].metadata
        cond_label = bp.get_cond_label(meta)

        normal_traps = [t for t in traps if not getattr(t, 'post_pulse_entry', False)]
        post_traps   = [t for t in traps if     getattr(t, 'post_pulse_entry', False)]

        subgroups = []
        if normal_traps:
            subgroups.append((normal_traps, cond_label, False))
        if post_traps:
            subgroups.append((post_traps, f"{cond_label}_POST", True))

        for subgroup_traps, file_label, is_post_subgroup in subgroups:
            sorted_traps = sorted(subgroup_traps, key=lambda t: t.trap_id)
            n = len(sorted_traps)
            if n == 0:
                continue

            cols = 5
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
            if n == 1:
                axes = [axes]
            else:
                axes = np.asarray(axes).flatten()

            handles = [
                mlines.Line2D([], [], color='black', marker='o', lw=0,
                              label='Data', alpha=0.4, ms=4),
                mlines.Line2D([], [], color=bp.MI_MODEL_PALETTE['Linear'],
                              lw=2, label='Linear fit'),
                mlines.Line2D([], [], color=bp.MI_MODEL_PALETTE['Power-Law'],
                              lw=2, label='Power-Law fit'),
                mlines.Line2D([], [], color='gray', ls='--', lw=1.2,
                              label='Pulse frame (EP)'),
            ]
            fig.legend(handles=handles, loc='upper center',
                       bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                       fontsize=11)
            dur_note = f"  [window: {global_whole_dur:.0f} s]" if global_whole_dur else ""
            fig.text(0.5, 0.01, f'Time from entry (s){dur_note}', ha='center', fontsize=13)
            fig.text(0.01, 0.5, 'Protrusion Length (µm)',
                     va='center', rotation='vertical', fontsize=13)

            if meta.condition_type == 'ASP':
                subtitle = f'ASP whole trace  ({file_label})'
            elif is_post_subgroup:
                subtitle = f'EP post-pulse-entry whole trace  ({file_label})'
            else:
                subtitle = f'EP whole trace including pulse  ({file_label})'
            fig.suptitle(subtitle, y=1.04, fontsize=12, fontweight='bold')

            for i, trap in enumerate(sorted_traps):
                ax = axes[i]
                ax.set_title(f'Trap {trap.trap_id}', fontsize=10, fontweight='bold')

                t_zeroed, l_raw = _extract_whole_trace(trap)
                if t_zeroed is None:
                    ax.axis('off')
                    continue

                if global_whole_dur is not None:
                    win_mask = t_zeroed <= global_whole_dur
                    t_zeroed = t_zeroed[win_mask]
                    l_raw = l_raw[win_mask]

                t_clean, l_clean = bm._clean_trace(t_zeroed, l_raw)
                if len(t_clean) < 15:
                    ax.axis('off')
                    continue

                ax.plot(t_clean, l_clean, 'o', c='black', ms=2, alpha=0.35)

                mi_fit = bm.fit_model_independent(t_clean, l_clean)
                t_smooth = np.linspace(t_clean[0], t_clean[-1], 200)

                if mi_fit['linear']:
                    p = mi_fit['linear']['params']
                    ax.plot(t_smooth,
                            bm._linear(t_smooth, p['slope'], p['intercept']),
                            color=bp.MI_MODEL_PALETTE['Linear'], lw=1.5, alpha=0.85)

                if mi_fit['power_law']:
                    p = mi_fit['power_law']['params']
                    ax.plot(t_smooth,
                            bm._power_law(t_smooth, p['a'], p['exponent_b'], p['c']),
                            color=bp.MI_MODEL_PALETTE['Power-Law'], lw=1.5, alpha=0.85)

                winner = mi_fit.get('best_model')
                if winner:
                    color = bp.MI_MODEL_PALETTE.get(winner, 'black')
                    ax.text(0.05, 0.92, f'AICc: {winner}',
                            transform=ax.transAxes, fontsize=8,
                            fontweight='bold', color=color)

                pt = _pulse_time_from_entry(trap)
                if np.isfinite(pt) and t_clean[0] <= pt <= t_clean[-1]:
                    ax.axvline(pt, color='gray', ls='--', lw=1.0, alpha=0.7)

            for j in range(n, len(axes)):
                axes[j].axis('off')

            plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
            out_path = output_dir / f"MI_WholeTrace_Fits_Panel_{file_label}.png"
            plt.savefig(out_path, dpi=PANEL_DPI, bbox_inches='tight')
            plt.close()
            
            
def plot_thesis_mechanics_vs_uptake_mi(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    import scipy.stats as stats
    logger.info("Generating Thesis Plot: Mechanics vs Uptake (MI Parameters & Trap ID)...")

    df = mechanics_df.copy()
    df = df[df['MI_Best_Model'].notna() & (df['MI_R2_Flag'] == True)].copy()
    
    if 'Post_Pulse_Entry_Flag' in df.columns:
        df = df[~df['Post_Pulse_Entry_Flag'].astype(bool)].copy()
        
    df = _restrict_to_common_ep_treatments(df)
    
    if df.empty:
        logger.info("  No data passes criteria for Mechanics vs Uptake — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket, axis=1)
    
    # Explicit 2x2 layout: pairing (x_col, x_label, y_col, y_label)
    panels = [
        ('Trap_ID', 'Trap ID (1=Far, 18=Near)', 'Uptake_Body_VolNorm_A', 'Body Uptake Plateau\n(ADU/µm³) [log]'),
        ('Linear_Slope', 'Linear Slope (µm/s)', 'Uptake_Prot_VolNorm_A', 'Protrusion Uptake Plateau\n(ADU/µm³) [log]'),
        ('PL_a', 'Power-Law a (µm)',            'Uptake_Prot_VolNorm_A', 'Protrusion Uptake Plateau\n(ADU/µm³) [log]'),
        ('PL_b', 'Power-Law b',                 'Uptake_Prot_VolNorm_A', 'Protrusion Uptake Plateau\n(ADU/µm³) [log]')
    ]
                   
    for bucket, sub_df in df.groupby('Bucket'):
        if len(sub_df) < 5:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        fig.suptitle(f"Mechanics & Spatial Position vs. Uptake: {bucket}", fontweight='bold', y=1.02)
        
        if bucket == 'ASP':
            color = MI_ASP_COLOR
        else:
            color = MI_EP_COLOR_RAMP[0] if MI_EP_COLOR_RAMP else 'red'
        
        axes_flat = axes.flatten()
        for k, (x_col, x_label, y_col, y_label) in enumerate(panels):
            ax = axes_flat[k]
            
            valid = sub_df[x_col].notna() & sub_df[y_col].notna()
            if valid.sum() < 5:
                ax.set_visible(False)
                continue
                
            x = sub_df.loc[valid, x_col].values
            y = sub_df.loc[valid, y_col].values
            
            # Floor values at 1e-3 to prevent log(0) errors
            y_safe = np.where(y > 1e-3, y, 1e-3)
            
            # For Trap ID, apply jitter to the x-axis so overlapping cells are visible
            x_plot = x
            if x_col == 'Trap_ID':
                rng = np.random.default_rng(seed=42)
                x_plot = x + rng.uniform(-0.25, 0.25, size=len(x))
            
            sns.regplot(
                x=x_plot, y=y_safe, ax=ax, 
                scatter_kws={'alpha': 0.6, 'color': color, 'edgecolor': 'white', 'linewidths': 0.5}, 
                line_kws={'color': 'black', 'lw': 1.5, 'ls': '--'}
            )
            
            rho, p = stats.spearmanr(x, y)
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.text(0.05, 0.95, f"$\\rho$={rho:.2f}\n{p_str}", transform=ax.transAxes, 
                    ha='left', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='gray'))
                    
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_yscale('log')
            if x_col == 'Trap_ID':
                ax.set_xticks(range(2, 19, 2))
                ax.set_xlim(0, 19)
        
        plt.tight_layout()
        utils.save_plot_pdf(output_dir / f"Thesis_Mechanics_vs_Uptake_{_safe_filename(bucket)}.pdf")
        plt.close()
        
        
def plot_thesis_spatial_mechanics_uptake(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    import scipy.stats as stats
    logger.info("Generating Thesis Plot: Spatial Mechanics vs Uptake (3D)...")

    df = mechanics_df.copy()
    df = df[df['MI_Best_Model'].notna() & (df['MI_R2_Flag'] == True)].copy()
    
    if 'Post_Pulse_Entry_Flag' in df.columns:
        df = df[~df['Post_Pulse_Entry_Flag'].astype(bool)].copy()
        
    df = _restrict_to_common_ep_treatments(df)
    
    if df.empty:
        logger.info("  No data passes criteria for Spatial Mechanics vs Uptake — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket, axis=1)
    
    # Evaluate all 3 Protrusion mechanics parameters here
    mech_cols = [
        ('Linear_Slope', 'Linear Slope (µm/s)'), 
        ('PL_a', 'Power-Law a (µm)'),
        ('PL_b', 'Power-Law b')
    ]
    uptake_col = 'Uptake_Prot_VolNorm_A'
    u_label = 'Log10 Protrusion Uptake\nPlateau (ADU/µm³)'
                   
    for bucket, sub_df in df.groupby('Bucket'):
        if len(sub_df) < 5:
            continue
            
        # Adjusted figsize for a 1x3 layout
        fig = plt.figure(figsize=(22, 7))
        fig.suptitle(f"Protrusion Spatial Trends: {bucket}", fontweight='bold', y=0.98, fontsize=14)
        
        if bucket == 'ASP':
            color = MI_ASP_COLOR
        else:
            color = MI_EP_COLOR_RAMP[0] if MI_EP_COLOR_RAMP else 'red'
            
        for plot_idx, (m_col, m_label) in enumerate(mech_cols, start=1):
            ax = fig.add_subplot(1, 3, plot_idx, projection='3d')
            
            valid = sub_df['Trap_ID'].notna() & sub_df[m_col].notna() & sub_df[uptake_col].notna()
            if valid.sum() < 3:
                ax.axis('off')
                continue
                
            trap_ids = sub_df.loc[valid, 'Trap_ID'].values.astype(float)
            mech_vals = sub_df.loc[valid, m_col].values
            uptake_vals = sub_df.loc[valid, uptake_col].values
            
            # Floor values to prevent log(0) errors, then take log10 manually for Z-axis
            uptake_vals_safe = np.where(uptake_vals > 1e-3, uptake_vals, 1e-3)
            z_vals = np.log10(uptake_vals_safe)
            
            rng = np.random.default_rng(seed=42)
            jitter_x = rng.uniform(-0.25, 0.25, size=len(trap_ids))
            
            ax.scatter(
                trap_ids + jitter_x, mech_vals, z_vals, 
                c=color, s=50, alpha=0.75, edgecolor='white', linewidths=0.5, depthshade=True
            )
            
            rho_m, _ = stats.spearmanr(trap_ids, mech_vals)
            rho_u, _ = stats.spearmanr(trap_ids, uptake_vals)
            ax.set_title(f"$\\rho_{{Trap,Mech}}$={rho_m:.2f}  |  $\\rho_{{Trap,Uptk}}$={rho_u:.2f}", fontsize=11, pad=10)
            
            ax.set_xlabel("Trap ID", labelpad=12, fontweight='bold')
            ax.set_ylabel(m_label, labelpad=12, fontweight='bold')
            ax.set_zlabel(u_label, labelpad=12, fontweight='bold')
            
            ax.set_xticks(range(2, 19, 2))
            ax.set_xlim(0, 19)
            
            ax.view_init(elev=20, azim=45)
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        utils.save_plot_pdf(output_dir / f"Thesis_Spatial_Mech_Uptake_3D_{_safe_filename(bucket)}.pdf")
        plt.close()
        


# --------------------------------------------------------------------------
# Fate cohort histogram
# --------------------------------------------------------------------------
def plot_thesis_fate_counts(mechanics_df: pd.DataFrame,
                             attrition_df: pd.DataFrame,
                             output_dir: Path) -> None:
    """
    Grouped bar chart: count of intact (no label) vs ruptured-post (_R) cells
    per pulse condition, with the pre-pulse-rupture (_R0) count shown as a
    third bar for context.

    Only two categories affect the analysis cohort:
        intact          <- mechanics_df where Fate_Status == 'intact'
        ruptured_post   <- mechanics_df where Fate_Status == 'ruptured_post'

    A third informational bar is drawn from the attrition tally:
        ruptured_pre_pulse (_R0)  <- pre-aspiration rupture, excluded from
                                     analysis but relevant to the
                                     "fraction ruptured per pulse regime"
                                     framing.

    All other attrition buckets (non_viable, post_pulse_arrival,
    detection_failure) are excluded per the current chapter scope.

    ASP cells are dropped because "pulse regime" is the grouping axis;
    ASP has no pulse.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Fate histogram skipped: mechanics_df is empty.")
        return
    if 'Fate_Status' not in mechanics_df.columns:
        logger.warning("Fate histogram skipped: no Fate_Status column.")
        return

    ep_mech = mechanics_df.loc[mechanics_df['Condition_Type'] == 'EP'].copy()
    if ep_mech.empty:
        logger.warning("Fate histogram skipped: no EP cells in mechanics_df.")
        return

    # Count intact / ruptured_post per Duration_label
    counts_analysis = (
        ep_mech.groupby(['Duration_label', 'Fate_Status'])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure both columns exist so downstream indexing is safe
    for col in ('intact', 'ruptured_post'):
        if col not in counts_analysis.columns:
            counts_analysis[col] = 0

    # Pull _R0 counts from attrition, restricted to EP experiments
    r0_counts = pd.Series(dtype=int)
    if attrition_df is not None and not attrition_df.empty:
        ep_att = attrition_df.loc[attrition_df['Condition_Type'] == 'EP']
        if not ep_att.empty and 'ruptured_pre_pulse' in ep_att.columns:
            r0_counts = ep_att.groupby('Duration_label')['ruptured_pre_pulse'].sum()

    # Combine into a single tidy DataFrame keyed by Duration_label
    durations = sorted(
        set(counts_analysis.index) | set(r0_counts.index),
        key=_duration_sort_key,
    )
    tidy = pd.DataFrame({
        'intact'            : [counts_analysis.get('intact',        pd.Series()).get(d, 0) for d in durations],
        'ruptured_post'     : [counts_analysis.get('ruptured_post', pd.Series()).get(d, 0) for d in durations],
        'ruptured_pre_pulse': [int(r0_counts.get(d, 0))                                     for d in durations],
    }, index=durations)

    # ---- Draw ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.02, 3.2))
    x       = np.arange(len(durations))
    width   = 0.27

    colours = {
        'intact'            : '#4C78A8',   # analysis-in blue
        'ruptured_post'     : '#E15759',   # ruptured red
        'ruptured_pre_pulse': '#B0B0B0',   # excluded grey
    }
    labels = {
        'intact'            : 'Intact',
        'ruptured_post'     : 'Ruptured (post-pulse)',
        'ruptured_pre_pulse': 'Ruptured (pre-pulse, excluded)',
    }

    for i, key in enumerate(('intact', 'ruptured_post', 'ruptured_pre_pulse')):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, tidy[key].to_numpy(),
                      width=width, color=colours[key], label=labels[key],
                      edgecolor='white', linewidth=0.6)
        # Value labels above each bar
        for rect, val in zip(bars, tidy[key].to_numpy()):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(),
                        f"{int(val)}",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(durations)
    ax.set_xlabel("Pulse duration")
    ax.set_ylabel("Cell count")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_Fate_Counts_by_Pulse.pdf")
    plt.close()

    # Also drop the counts alongside the figure so the numbers can be quoted
    tidy_out = tidy.copy()
    tidy_out.index.name = 'Duration_label'
    tidy_out['analysis_total']       = tidy_out['intact'] + tidy_out['ruptured_post']
    tidy_out['fraction_ruptured']    = np.where(
        tidy_out['analysis_total'] > 0,
        tidy_out['ruptured_post'] / tidy_out['analysis_total'],
        np.nan,
    )
    tidy_out.to_csv(output_dir / "Thesis_Fate_Counts_by_Pulse.csv")
    logger.info(f"Fate histogram written; fractions ruptured: "
                + ", ".join(f"{d}={f:.2f}" for d, f in tidy_out['fraction_ruptured'].items()
                            if np.isfinite(f)))


def _duration_sort_key(label: str) -> float:
    """Sort duration labels so 100us < 5ms < 10ms < 100ms etc."""
    label = str(label).strip().lower()
    for unit, factor in (('us', 1e-3), ('ms', 1.0), ('s', 1e3)):
        if label.endswith(unit):
            try:
                return float(label[:-len(unit)]) * factor
            except ValueError:
                return float('inf')
    return float('inf')


# ==========================================================================
# BATCH 2 — Claim 2 (sensitivity to biological perturbation)
# ==========================================================================
# Colours: WT gets the cool end of the palette, CytD the warm end.
_TREATMENT_COLOR = {
    'WT'  : utils.MFA_COLORS['dark_blue'],
    'CytD': utils.MFA_COLORS['medium_red'],
}
_TREATMENT_ORDER = ['WT', 'CytD']


def _actin_f0_trace(trap: bfh.TrapData, region: str):
    """
    Return (t_seconds, I_over_F0) for one region of one trap, or (None, None).

    region is 'Body' or 'Prot' (matches the CSV column suffix).
    Time is zero-referenced to the first frame so cross-cell interpolation
    can share a common axis.
    """
    ad = getattr(trap, 'actin_data', {}) or {}
    if 'Time_s' not in ad:
        return None, None
    mean_key = f'Actin_{region}_Mean'
    f0_key   = f'F0_{region}'
    if mean_key not in ad or f0_key not in ad:
        return None, None

    t   = np.asarray(ad['Time_s'], dtype=float)
    I   = np.asarray(ad[mean_key], dtype=float)
    F0  = np.asarray(ad[f0_key],   dtype=float)

    if len(t) < 5 or len(I) < 5 or len(F0) == 0:
        return None, None

    # F0 is stored broadcast; take the first finite positive value.
    f0_scalar = next((float(v) for v in F0 if np.isfinite(v) and v > 0), None)
    if f0_scalar is None:
        return None, None

    valid = np.isfinite(t) & np.isfinite(I)
    if valid.sum() < 5:
        return None, None
    t = t[valid]; I = I[valid]

    return t - t[0], I / f0_scalar


def plot_thesis_asp_actin_trace_by_condition(grouped_data: Dict,
                                              mechanics_df: pd.DataFrame,
                                              output_dir: Path,
                                              global_asp_dur: Optional[float] = None) -> None:
    """
    Mean ± SD of F0-normalised actin fluorescence during aspiration,
    WT vs CytD, for the body and protrusion regions.

    Body region is the primary panel (no geometric confound during
    aspiration).  Protrusion is plotted alongside with a caveat: as L(t)
    grows, protrusion actin gets stretched along a longer tube, so a
    drop in the trace mixes actin content changes with geometric dilution.
    The comparison remains valid across matched conditions.
    """
    # Restrict to ASP intact cells only — mechanics_df is the filter.
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Actin trace plot skipped: empty mechanics_df.")
        return

    asp_pass = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'ASP')
        & (mechanics_df['Fate_Status'] == 'intact')
    ]
    accepted = set(zip(asp_pass['Experiment_Folder'], asp_pass['Trap_ID']))
    if not accepted:
        logger.warning("Actin trace plot skipped: no ASP intact cells in mechanics_df.")
        return

    # Collect per-condition traces
    by_treatment: Dict[str, Dict[str, list]] = {
        t: {'Body': [], 'Prot': []} for t in _TREATMENT_ORDER
    }
    max_t_seen: float = 0.0

    for gk, traps in grouped_data.items():
        if not traps or traps[0].metadata.condition_type != 'ASP':
            continue
        for trap in traps:
            key = (trap.metadata.full_path.name, trap.trap_id)
            if key not in accepted:
                continue
            treatment = trap.metadata.treatment
            if treatment not in by_treatment:
                continue
            for region in ('Body', 'Prot'):
                t, y = _actin_f0_trace(trap, region)
                if t is None:
                    continue
                by_treatment[treatment][region].append((t, y))
                if len(t) and t[-1] > max_t_seen:
                    max_t_seen = float(t[-1])

    # Apply the same cross-condition duration cap the mechanics pipeline uses.
    t_cap = global_asp_dur if global_asp_dur is not None else max_t_seen
    if t_cap <= 0:
        logger.warning("Actin trace plot skipped: no valid duration.")
        return

    # Interpolate each trace onto a common grid, then aggregate.
    common_t = np.linspace(0.0, t_cap, 60)

    def _aggregate(traces):
        if not traces:
            return None, None
        stack = []
        for t, y in traces:
            if t[-1] < t_cap * 0.5:
                continue   # too short to contribute
            f = interp1d(t, y, bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            stack.append(f(common_t))
        if not stack:
            return None, None
        arr = np.vstack(stack)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    # ---- Draw --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(5.02, 2.8), sharey=True)
    region_titles = {'Body': 'Body', 'Prot': 'Protrusion'}

    for ax, region in zip(axes, ('Body', 'Prot')):
        for treatment in _TREATMENT_ORDER:
            traces = by_treatment[treatment][region]
            mu, sd = _aggregate(traces)
            n = len(traces)
            if mu is None:
                continue
            colour = _TREATMENT_COLOR[treatment]
            ax.plot(common_t, mu, color=colour, lw=1.5,
                    label=f"{treatment} (n={n})")
            ax.fill_between(common_t, mu - sd, mu + sd,
                            color=colour, alpha=0.20, linewidth=0)

        ax.axhline(1.0, color='0.6', lw=0.8, ls='--', zorder=0)
        ax.set_title(region_titles[region], fontsize=10)
        ax.set_xlabel("Time from aspiration onset (s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(r"Actin intensity $I(t)/F_0$")
    axes[0].legend(frameon=False, fontsize=8, loc='best')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_ASP_Actin_Trace_by_Condition.pdf")
    plt.close()
    logger.info("Actin trace plot written.")


def plot_thesis_asp_actin_vs_mechanics(mechanics_df: pd.DataFrame,
                                        output_dir: Path) -> None:
    """
    Scatter: pre-pulse actin (F0-normalised) vs elastic modulus E, per cell.
    Two panels: body and protrusion.  Spearman ρ annotated per treatment.

    Filters to ASP intact cells with a valid viscoelastic fit.
    """
    from scipy import stats
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'ASP')
        & (mechanics_df['Fate_Status'] == 'intact')
        & mechanics_df['E_Pa'].notna()
        & (mechanics_df['Visco_R2_Flag'] == True)
    ].copy()

    if df.empty:
        logger.warning("Actin-vs-E plot skipped: no ASP intact cells with valid E.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.02, 2.8), sharey=False)
    for ax, region in zip(axes, ('Body', 'Prot')):
        actin_col = f'Actin_{region}_PrePulse_F0Norm'
        if actin_col not in df.columns:
            continue

        for treatment in _TREATMENT_ORDER:
            sub = df.loc[df['Treatment'] == treatment,
                         [actin_col, 'E_Pa']].dropna()
            if sub.empty:
                continue
            colour = _TREATMENT_COLOR[treatment]
            ax.scatter(sub[actin_col], sub['E_Pa'], s=22,
                       color=colour, edgecolor='white', linewidths=0.4,
                       alpha=0.75, label=f"{treatment} (n={len(sub)})")

            if len(sub) >= 5:
                rho, p = stats.spearmanr(sub[actin_col], sub['E_Pa'])
                y_anchor = 0.95 if treatment == 'WT' else 0.88
                ax.text(0.03, y_anchor,
                        rf"$\rho_{{{treatment}}}={rho:.2f}$ (p={p:.2g})",
                        transform=ax.transAxes, fontsize=8, color=colour,
                        va='top')

        ax.set_yscale('log')
        ax.set_xlabel(rf"Actin $I/F_0$ ({region.lower()}, pre-pulse)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(r"$E$ (Pa)")
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_ASP_Actin_vs_E.pdf")
    plt.close()
    logger.info("Actin-vs-E scatter written.")


def plot_thesis_asp_trap_dependency(mechanics_df: pd.DataFrame,
                                     output_dir: Path) -> None:
    """
    Three-panel scatter of ASP quantities vs Trap_ID:
    (1) Max pre-pulse protrusion length
    (2) Elastic modulus E
    (3) Flow viscosity η₁

    Coloured by treatment, Spearman ρ annotated per treatment per panel.
    Trap number is a proxy for aspiration pressure (3 kPa at trap 1 →
    1 kPa at trap 18) and for electrode distance.  A flat regression here
    is what supports the "no meaningful pressure gradient effect" claim.
    """
    from scipy import stats
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'ASP')
        & (mechanics_df['Fate_Status'] == 'intact')
    ].copy()

    if df.empty:
        logger.warning("ASP trap dependency plot skipped: no ASP intact cells.")
        return

    panels = [
        ('Max_Prot_length_PrePulse_um', 'Protrusion length (µm)',           False),
        ('E_Pa',                        r"$E$ (Pa)",                        True ),
        ('eta1_Pa_s',                   r"$\eta_1$ (Pa$\cdot$s)",           True ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharex=True)
    for ax, (col, ylabel, use_log) in zip(axes, panels):
        if col not in df.columns:
            continue

        for treatment in _TREATMENT_ORDER:
            sub = df.loc[df['Treatment'] == treatment,
                         ['Trap_ID', col]].dropna()
            if sub.empty:
                continue
            colour = _TREATMENT_COLOR[treatment]
            ax.scatter(sub['Trap_ID'], sub[col], s=20,
                       color=colour, edgecolor='white', linewidths=0.4,
                       alpha=0.75, label=f"{treatment} (n={len(sub)})")

            if len(sub) >= 5:
                rho, p = stats.spearmanr(sub['Trap_ID'], sub[col])
                y_anchor = 0.95 if treatment == 'WT' else 0.88
                ax.text(0.03, y_anchor,
                        rf"$\rho_{{{treatment}}}={rho:.2f}$ (p={p:.2g})",
                        transform=ax.transAxes, fontsize=7.5, color=colour,
                        va='top')

        if use_log:
            ax.set_yscale('log')
        ax.set_xlabel("Trap ID")
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8, loc='best')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_ASP_Trap_Dependency.pdf")
    plt.close()
    logger.info("ASP trap-dependency scatter written.")


def plot_thesis_asp_body_volume_vs_E(mechanics_df: pd.DataFrame,
                                      output_dir: Path) -> None:
    """
    Sanity check: pre-pulse cell body volume vs elastic modulus E.

    If E correlates with body volume, the "stiffness" readout could
    partly be picking up cell-size availability of cortical material for
    the protrusion rather than intrinsic mechanics.  A flat regression
    supports the intrinsic-mechanics interpretation.
    """
    from scipy import stats
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'ASP')
        & (mechanics_df['Fate_Status'] == 'intact')
        & mechanics_df['E_Pa'].notna()
        & (mechanics_df['Visco_R2_Flag'] == True)
        & mechanics_df['Cell_Body_Volume_PrePulse_um3'].notna()
    ].copy()

    if df.empty:
        logger.warning("Body volume vs E plot skipped: no eligible cells.")
        return

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for treatment in _TREATMENT_ORDER:
        sub = df.loc[df['Treatment'] == treatment,
                     ['Cell_Body_Volume_PrePulse_um3', 'E_Pa']].dropna()
        if sub.empty:
            continue
        colour = _TREATMENT_COLOR[treatment]
        ax.scatter(sub['Cell_Body_Volume_PrePulse_um3'], sub['E_Pa'],
                   s=22, color=colour, edgecolor='white', linewidths=0.4,
                   alpha=0.75, label=f"{treatment} (n={len(sub)})")

        if len(sub) >= 5:
            rho, p = stats.spearmanr(
                sub['Cell_Body_Volume_PrePulse_um3'], sub['E_Pa'])
            y_anchor = 0.95 if treatment == 'WT' else 0.88
            ax.text(0.03, y_anchor,
                    rf"$\rho_{{{treatment}}}={rho:.2f}$ (p={p:.2g})",
                    transform=ax.transAxes, fontsize=8, color=colour,
                    va='top')

    ax.set_xlabel(r"Body volume (µm$^3$, pre-pulse)")
    ax.set_ylabel(r"$E$ (Pa)")
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='best')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_ASP_BodyVolume_vs_E.pdf")
    plt.close()
    logger.info("Body volume vs E scatter written.")


def run_thesis_claim2_plots(grouped_data: Dict,
                             mechanics_df: pd.DataFrame,
                             output_dir: Path,
                             global_asp_dur: Optional[float] = None) -> None:
    """Convenience runner: all Claim 2 (sensitivity) plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_thesis_asp_actin_trace_by_condition(
        grouped_data, mechanics_df, output_dir, global_asp_dur)
    plot_thesis_asp_actin_vs_mechanics(mechanics_df, output_dir)
    plot_thesis_asp_trap_dependency(mechanics_df, output_dir)
    plot_thesis_asp_body_volume_vs_E(mechanics_df, output_dir)