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
import Utils_MFA as utils

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
                            bm._power_law(t_smooth, p['a'], p['exponent_b']),
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
    logger.info("Generating Thesis Plot: Mechanics vs Uptake (MI Parameters)...")

    df = mechanics_df.copy()
    # Apply the same fitting criteria filter as the parameter boxplots
    df = df[df['MI_Best_Model'].notna() & (df['MI_R2_Flag'] == True)].copy()
    
    if 'Post_Pulse_Entry_Flag' in df.columns:
        df = df[~df['Post_Pulse_Entry_Flag'].astype(bool)].copy()
        
    df = _restrict_to_common_ep_treatments(df)
    
    if df.empty:
        logger.info("  No data passes criteria for Mechanics vs Uptake — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket, axis=1)
    
    mech_cols = [
        ('Linear_Slope', 'Linear Slope (µm/s)'), 
        ('PL_a', 'Power-Law a (µm)'),
        ('PL_b', 'Power-Law b')
    ]
    uptake_cols = [
        ('Uptake_Body_VolNorm_A', 'Body Uptake Plateau (ADU/µm³)'), 
        ('Uptake_Prot_VolNorm_A', 'Protrusion Uptake Plateau (ADU/µm³)')
    ]
                   
    for bucket, sub_df in df.groupby('Bucket'):
        # Require at least 5 data points to generate meaningful correlations
        if len(sub_df) < 5:
            continue
            
        fig, axes = plt.subplots(len(mech_cols), len(uptake_cols), figsize=(10, 12))
        fig.suptitle(f"Mechanics vs. Uptake: {bucket}", fontweight='bold', y=1.02)
        
        # Match colors to existing MI plots
        if bucket == 'ASP':
            color = MI_ASP_COLOR
        else:
            color = MI_EP_COLOR_RAMP[0] if MI_EP_COLOR_RAMP else 'red'
        
        for i, (m_col, m_label) in enumerate(mech_cols):
            for j, (u_col, u_label) in enumerate(uptake_cols):
                ax = axes[i, j]
                
                valid = sub_df[m_col].notna() & sub_df[u_col].notna()
                if valid.sum() < 5:
                    ax.set_visible(False)
                    continue
                    
                x = sub_df.loc[valid, m_col].values
                y = sub_df.loc[valid, u_col].values
                
                sns.regplot(
                    x=x, y=y, ax=ax, 
                    scatter_kws={'alpha': 0.6, 'color': color, 'edgecolor': 'white', 'linewidths': 0.5}, 
                    line_kws={'color': 'black', 'lw': 1.5, 'ls': '--'}
                )
                
                rho, p = stats.spearmanr(x, y)
                p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                ax.text(0.05, 0.95, f"$\\rho$={rho:.2f}\n{p_str}", transform=ax.transAxes, 
                        ha='left', va='top', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='gray'))
                        
                ax.set_xlabel(m_label)
                ax.set_ylabel(u_label)
        
        plt.tight_layout()
        utils.save_plot_pdf(output_dir / f"Thesis_Mechanics_vs_Uptake_{_safe_filename(bucket)}.pdf")
        plt.close()
        
        
def plot_thesis_spatial_mechanics_uptake(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    import scipy.stats as stats
    logger.info("Generating Thesis Plot: Spatial Mechanics vs Uptake (Dual Axis)...")

    df = mechanics_df.copy()
    # Apply fitting criteria filter
    df = df[df['MI_Best_Model'].notna() & (df['MI_R2_Flag'] == True)].copy()
    
    if 'Post_Pulse_Entry_Flag' in df.columns:
        df = df[~df['Post_Pulse_Entry_Flag'].astype(bool)].copy()
        
    df = _restrict_to_common_ep_treatments(df)
    
    if df.empty:
        logger.info("  No data passes criteria for Spatial Mechanics vs Uptake — skipping.")
        return

    df['Bucket'] = df.apply(_row_bucket, axis=1)
    
    mech_cols = [
        ('Linear_Slope', 'Linear Slope (µm/s)'), 
        ('PL_a', 'Power-Law a (µm)'),
        ('PL_b', 'Power-Law b')
    ]
    uptake_cols = [
        ('Uptake_Body_VolNorm_A', 'Body Uptake Plateau (ADU/µm³)'), 
        ('Uptake_Prot_VolNorm_A', 'Protrusion Uptake Plateau (ADU/µm³)')
    ]
                   
    for bucket, sub_df in df.groupby('Bucket'):
        # Require at least 5 data points
        if len(sub_df) < 5:
            continue
            
        # Increased figsize to prevent label collision
        fig, axes = plt.subplots(len(mech_cols), len(uptake_cols), figsize=(14, 12))
        fig.suptitle(f"Spatial Trends: Mechanics & Uptake\n{bucket}", fontweight='bold', y=1.02)
        
        for i, (m_col, m_label) in enumerate(mech_cols):
            for j, (u_col, u_label) in enumerate(uptake_cols):
                ax1 = axes[i, j]
                
                valid = sub_df['Trap_ID'].notna() & sub_df[m_col].notna() & sub_df[u_col].notna()
                if valid.sum() < 3:
                    ax1.set_visible(False)
                    continue
                    
                trap_ids = sub_df.loc[valid, 'Trap_ID'].values.astype(float)
                mech_vals = sub_df.loc[valid, m_col].values
                uptake_vals = sub_df.loc[valid, u_col].values
                
                # Apply deterministic jitter to discrete trap IDs to separate overlapping points
                rng = np.random.default_rng(seed=42)
                jitter_m = rng.uniform(-0.15, 0.15, size=len(trap_ids))
                jitter_u = rng.uniform(-0.15, 0.15, size=len(trap_ids))
                
                # 1. Plot Mechanics on the Left Y-Axis (Blue)
                color_m = '#1f77b4' 
                sns.regplot(
                    x=trap_ids + jitter_m, y=mech_vals, ax=ax1, 
                    color=color_m, scatter_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'white', 'linewidths': 0.5}, 
                    line_kws={'lw': 2}
                )
                ax1.set_ylabel(m_label, color=color_m, fontweight='bold')
                ax1.tick_params(axis='y', labelcolor=color_m)
                
                # 2. Plot Uptake on the Right Y-Axis (Red)
                ax2 = ax1.twinx()
                color_u = '#d62728' 
                sns.regplot(
                    x=trap_ids + jitter_u, y=uptake_vals, ax=ax2, 
                    color=color_u, scatter_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'white', 'linewidths': 0.5}, 
                    line_kws={'lw': 2, 'ls': '--'}
                )
                ax2.set_ylabel(u_label, color=color_u, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor=color_u)
                ax2.grid(False) # Turn off grid for second axis to prevent crossing gridlines
                
                # 3. Calculate spatial correlations to quantify the trend
                rho_m, p_m = stats.spearmanr(trap_ids, mech_vals)
                rho_u, p_u = stats.spearmanr(trap_ids, uptake_vals)
                
                annot_text = (
                    f"Mech vs Trap: $\\rho$={rho_m:.2f} (p={p_m:.2f})\n"
                    f"Uptk vs Trap: $\\rho$={rho_u:.2f} (p={p_u:.2f})"
                )
                
                ax1.text(0.05, 0.95, annot_text, transform=ax1.transAxes, 
                        ha='left', va='top', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='gray'), zorder=10)
                
                ax1.set_xlabel("Trap ID (1 = Far, 18 = Near Electrode)")
                
                # Fix the overlapping x-ticks by forcing intervals of 2
                ax1.set_xticks(range(2, 19, 2))
                ax1.set_xlim(0, 19)
                
        plt.tight_layout()
        utils.save_plot_pdf(output_dir / f"Thesis_Spatial_Mech_Uptake_{_safe_filename(bucket)}.pdf")
        plt.close()