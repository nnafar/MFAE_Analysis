# -*- coding: utf-8 -*-
"""
Thesis Plotting Module for ASP Mechanics and Model-Independent (MI) Analysis.
"""

import logging
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from scipy.interpolate import interp1d

import bulk_file_handling as bfh
import bulk_mechanics as bm
import bulk_plotting as bp
import bulk_utils as utils

logger = logging.getLogger(__name__)
utils.set_paper_style()

# ---------------------------------------------------------------------------
# Directory Routing Helpers
# ---------------------------------------------------------------------------
def _get_subfolder_from_bucket(bucket: str) -> str:
    """Extracts the condition string from a fate-labelled bucket name to route plots."""
    if 'ASP' in bucket:
        return 'ASP'
    # Matches patterns like "EP-pre 100V 100us" -> "100V_100us"
    m = re.search(r'(\d+V)[_\s]+([a-zA-Z0-9]+)', bucket)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return 'combined'

# ---------------------------------------------------------------------------
# General Data Extractors
# ---------------------------------------------------------------------------
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

def _mi_winner_slope(row: pd.Series) -> float:
    m = row.get('MI_Best_Model')
    if m == 'Linear':
        return row.get('Linear_Slope', np.nan)
    if m == 'Power-Law':
        return row.get('PL_b', np.nan)
    return np.nan

def _mi_whole_winner_slope(row: pd.Series) -> float:
    m = row.get('MI_Whole_Best_Model')
    if m == 'Linear':
        return row.get('MI_Whole_Linear_Slope', np.nan)
    if m == 'Power-Law':
        return row.get('MI_Whole_PL_b', np.nan)
    return np.nan

# ---------------------------------------------------------------------------
# Plotting Implementations
# ---------------------------------------------------------------------------
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
            ax.set_ylim(0, 40)

            plt.tight_layout()
            
            # Single condition -> subfolder
            target_dir = output_dir / "ASP"
            target_dir.mkdir(parents=True, exist_ok=True)
            utils.save_plot_pdf(target_dir / f"Thesis_Boxplot_{cond_label}.pdf")
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
        plot_mask = (np.isfinite(mean_L) & np.isfinite(sd_L) & (n_t >= min_n_required))

        if np.any(plot_mask):
            plot_dur = float(t_grid[plot_mask][-1])
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
        
        # Combined plot -> combined folder
        target_dir = output_dir / "combined"
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / "Thesis_Combined_Protrusion_Dynamics_ASP.pdf")
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
        
        target_dir = output_dir / "ASP"
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"ASP_BestFit_Panel_{cat_label_short}.pdf")
        plt.close()

def plot_thesis_asp_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: ASP Viscoelastic Parameters...")
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    bp.plot_asp_parameter_boxplots(mechanics_df, target_dir)

def plot_thesis_asp_actin_f0_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: ASP Actin F0 Boxplots...")
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    bp.plot_asp_actin_f0_boxplots(mechanics_df, target_dir)

def plot_thesis_prepulse_visco_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: Pre-Pulse Viscoelastic Parameters...")
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    bp.plot_prepulse_visco_parameter_boxplots(mechanics_df, target_dir)

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
    _try("ASP Actin F0 Boxplots",
         plot_thesis_asp_actin_f0_boxplots, mechanics_df, output_dir)
    _try("Pre-Pulse Viscoelastic Parameter Boxplots",
         plot_thesis_prepulse_visco_parameter_boxplots, mechanics_df, output_dir)

MI_ASP_COLOR = utils.MFA_COLORS.get('dark_blue', '#1f4e79') if hasattr(utils, 'MFA_COLORS') else '#1f4e79'
MI_EP_COLOR_RAMP = (
    [utils.MFA_COLORS[k] for k in ('dark_red', 'medium_red', 'light_red', 'pale_red')
     if k in utils.MFA_COLORS]
    if hasattr(utils, 'MFA_COLORS') else
    ['#b31529', '#d75f4c', '#f6a482', '#fddbc7']
)

_MI_MODEL_UTILS_COLOR = {
    'Linear'   : utils.MFA_COLORS['medium_red'],
    'Power-Law': utils.MFA_COLORS['medium_blue'],
}

def _condition_bucket(meta: bfh.ExperimentMetadata) -> str:
    if meta.condition_type == "ASP":
        return "ASP"
    return f"EP-pre {meta.voltage}V {meta.duration_label}"

def _condition_bucket_fate(meta: bfh.ExperimentMetadata, fate_status: str) -> str:
    if meta.condition_type == "ASP":
        return "ASP"
    base = f"EP-pre {meta.voltage}V {meta.duration_label}"
    if fate_status == "ruptured_post":
        return f"{base} (ruptured)"
    return f"{base} (intact)"

def _row_bucket(row: pd.Series) -> str:
    if row.get('Condition_Type') == 'ASP':
        return "ASP"
    return f"EP-pre {row.get('Voltage_V')}V {row.get('Duration_label')}"

def _row_bucket_fate(row: pd.Series) -> str:
    if row.get('Condition_Type') == 'ASP':
        return "ASP"
    base = f"EP-pre {row.get('Voltage_V')}V {row.get('Duration_label')}"
    if row.get('Fate_Status') == "ruptured_post":
        return f"{base} (ruptured)"
    return f"{base} (intact)"

def _safe_filename(label: str) -> str:
    return label.replace(' ', '_')

def plot_mi_protrusion_dynamics(grouped_data: Dict, output_dir: Path, global_pre_dur: Optional[float] = None) -> None:
    logger.info("Generating Thesis Plot: MI Protrusion Dynamics (ASP vs EP-pre)...")

    bucketed_traces: Dict[str, list] = {}
    bucketed_max_records: Dict[Tuple[str, str], list] = {}

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps:
            continue

        meta = traps[0].metadata

        for trap in sorted(traps, key=lambda t: t.trap_id):
            bucket = _condition_bucket_fate(meta, trap.fate_status)
            subfolder = _get_subfolder_from_bucket(bucket)
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
                bucketed_max_records.setdefault((bucket, subfolder), []).append({
                    'Trap':          f"T{trap.trap_id}",
                    'Trap_Int':      trap.trap_id,
                    'Max_Length_um': float(max_val),
                })

    for (bucket, subfolder), records in bucketed_max_records.items():
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
        
        target_dir = output_dir / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_MI_Boxplot_{_safe_filename(bucket)}.pdf")
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
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_Combined_Protrusion_Dynamics_MI.pdf")
    plt.close()

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

    df['Bucket']         = df.apply(_row_bucket_fate, axis=1)
    df['Winner_Slope']   = df.apply(_mi_winner_slope, axis=1)

    buckets_present = df['Bucket'].unique().tolist()
    ep_buckets = sorted(b for b in buckets_present if b != 'ASP')
    order = (['ASP'] if 'ASP' in buckets_present else []) + ep_buckets

    fig_w = max(12, 2.4 * len(order) + 4)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 6))

    ax = axes[0]
    sub = df[df['Winner_Slope'].notna()]
    if not sub.empty:
        sns.boxplot(data=sub, x='Bucket', y='Winner_Slope', order=order,
                    ax=ax, showfliers=False, color='lightgray')
        sns.stripplot(data=sub, x='Bucket', y='Winner_Slope', order=order,
                      hue='MI_Best_Model', palette=_MI_MODEL_UTILS_COLOR,
                      dodge=False, alpha=0.7, ax=ax, size=5)

        ax.set_title('Slope (winner-model)', fontweight='bold')
        ax.set_ylabel(r'Rate: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law')
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

        pairs = [(order[j], order[j + 1]) for j in range(len(order) - 1)]
        for cat_a, cat_b in pairs:
            vals_a = sub.loc[sub['Bucket'] == cat_a, 'Winner_Slope'].dropna().values
            vals_b = sub.loc[sub['Bucket'] == cat_b, 'Winner_Slope'].dropna().values
            stars, label, lw, delta = bp._build_stat_label(vals_a, vals_b)
            if label is None:
                continue
            x1 = order.index(cat_a); x2 = order.index(cat_b)
            y_top = float(np.nanmax(np.concatenate([vals_a, vals_b])))
            bp._add_bracket(ax, x1, x2, y_top, label, lw=lw)

    ax = axes[1]
    sub_pl = df[(df['MI_Best_Model'] == 'Power-Law') & df['PL_a'].notna()]
    if sub_pl.empty:
        ax.set_visible(False)
    else:
        pl_colour = _MI_MODEL_UTILS_COLOR['Power-Law']
        sns.boxplot(data=sub_pl, x='Bucket', y='PL_a', order=order,
                    ax=ax, showfliers=False, color='lightgray')
        sns.stripplot(data=sub_pl, x='Bucket', y='PL_a', order=order,
                      color=pl_colour, dodge=False, alpha=0.7, ax=ax, size=5)

        ax.set_title('Power-Law amplitude a (PL winners only)', fontweight='bold')
        ax.set_ylabel(r'Amplitude $a$ (µm)')
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        if (sub_pl['PL_a'] > 0).all():
            ax.set_yscale('log')

        pairs = [(order[j], order[j + 1]) for j in range(len(order) - 1)]
        for cat_a, cat_b in pairs:
            vals_a = sub_pl.loc[sub_pl['Bucket'] == cat_a, 'PL_a'].dropna().values
            vals_b = sub_pl.loc[sub_pl['Bucket'] == cat_b, 'PL_a'].dropna().values
            stars, label, lw, delta = bp._build_stat_label(vals_a, vals_b)
            if label is None:
                continue
            x1 = order.index(cat_a); x2 = order.index(cat_b)
            y_top = float(np.nanmax(np.concatenate([vals_a, vals_b])))
            bp._add_bracket(ax, x1, x2, y_top, label, lw=lw)

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_MI_Parameter_Boxplots.pdf")
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

    df['Bucket']       = df.apply(_row_bucket_fate, axis=1)
    df['Winner_Slope'] = df.apply(_mi_winner_slope, axis=1)

    for bucket, sub_bucket in df.groupby('Bucket'):
        if sub_bucket.empty:
            continue

        trap_ids_sorted = sorted(sub_bucket['Trap_ID'].dropna().astype(int).unique())
        trap_order = [f"T{t}" for t in trap_ids_sorted]
        sub_bucket = sub_bucket.copy()
        sub_bucket['Trap'] = 'T' + sub_bucket['Trap_ID'].astype(int).astype(str)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(bucket, fontweight='bold', y=1.02)

        ax = axes[0]
        sub = sub_bucket[sub_bucket['Winner_Slope'].notna()]
        if sub.empty:
            ax.set_visible(False)
        else:
            sns.boxplot(data=sub, x='Trap', y='Winner_Slope', order=trap_order,
                        ax=ax, showfliers=False, color='lightgray',
                        boxprops=dict(alpha=0.4))
            sns.stripplot(data=sub, x='Trap', y='Winner_Slope', order=trap_order,
                          hue='MI_Best_Model', palette=_MI_MODEL_UTILS_COLOR,
                          dodge=False, alpha=0.75, ax=ax, size=5)
            ax.set_title('Slope (winner-model)', fontweight='bold')
            ax.set_ylabel(r'Rate: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law')
            ax.set_xlabel('Trap ID')
            ax.tick_params(axis='x', rotation=45)

        ax = axes[1]
        sub_pl = sub_bucket[
            (sub_bucket['MI_Best_Model'] == 'Power-Law')
            & sub_bucket['PL_a'].notna()
        ]
        if sub_pl.empty:
            ax.set_visible(False)
        else:
            pl_colour = _MI_MODEL_UTILS_COLOR['Power-Law']
            sns.boxplot(data=sub_pl, x='Trap', y='PL_a', order=trap_order,
                        ax=ax, showfliers=False, color='lightgray',
                        boxprops=dict(alpha=0.4))
            sns.stripplot(data=sub_pl, x='Trap', y='PL_a', order=trap_order,
                          color=pl_colour, dodge=False, alpha=0.75, ax=ax, size=5)
            ax.set_title('Power-Law amplitude a (PL winners only)',
                         fontweight='bold')
            ax.set_ylabel(r'Amplitude $a$ (µm)')
            ax.set_xlabel('Trap ID')
            ax.tick_params(axis='x', rotation=45)
            if (sub_pl['PL_a'] > 0).all():
                ax.set_yscale('log')

        plt.tight_layout()
        
        target_dir = output_dir / _get_subfolder_from_bucket(bucket)
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_MI_PerTrap_Params_{_safe_filename(bucket)}.pdf")
        plt.close()

def run_thesis_mi_prepulse_plots(mi_grouped_data: Dict,
                                 mechanics_df: pd.DataFrame,
                                 output_dir: Path,
                                 global_pre_dur: Optional[float] = None) -> None:
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Pre-pulse thesis plot '{label}' failed: {e}", exc_info=False)

    _try("Pre-Pulse Protrusion Dynamics",
         plot_mi_protrusion_dynamics, mi_grouped_data, output_dir, global_pre_dur)
    _try("Pre-Pulse Viscoelastic Parameter Boxplots",
         plot_thesis_prepulse_visco_parameter_boxplots, mechanics_df, output_dir)
    _try("Fate Pre-Pulse Viscoelastic Boxplots",
         plot_thesis_fate_prepulse_visco_boxplots, mechanics_df, output_dir)

def run_thesis_mi_wholetrace_plots(mi_whole_grouped_data: Dict,
                                   mechanics_df: pd.DataFrame,
                                   output_dir: Path,
                                   global_whole_dur: Optional[float] = None) -> None:
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"MI thesis plot '{label}' failed: {e}", exc_info=False)

    _try("MI Whole-Trace Parameter Boxplots",
         plot_thesis_mi_wholetrace_parameter_boxplots, mechanics_df, output_dir)
    _try("Fate Whole-Trace MI Boxplots",
         plot_thesis_fate_wholetrace_mi_boxplots, mechanics_df, output_dir)

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

def _row_bucket_wholetrace(row: pd.Series) -> str:
    if row.get('Condition_Type') == 'ASP':
        return 'ASP'
    is_post = bool(row.get('Post_Pulse_Entry_Flag', False))
    prefix = 'EP-post' if is_post else 'EP-whole'
    return f"{prefix} {row.get('Voltage_V')}V {row.get('Duration_label')}"

def _row_bucket_wholetrace_fate(row: pd.Series) -> str:
    if row.get('Condition_Type') == 'ASP':
        return 'ASP'
    is_post = bool(row.get('Post_Pulse_Entry_Flag', False))
    prefix = 'EP-post' if is_post else 'EP-whole'
    base = f"{prefix} {row.get('Voltage_V')}V {row.get('Duration_label')}"
    if row.get('Fate_Status') == "ruptured_post":
        return f"{base} (ruptured)"
    return f"{base} (intact)"

def _wholetrace_bucket_color(bucket: str, ep_whole_index: int, ep_post_index: int) -> str:
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

    df['Bucket'] = df.apply(_row_bucket_wholetrace_fate, axis=1)
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

    df['Winner_Slope_Whole'] = df.apply(_mi_whole_winner_slope, axis=1)

    fig_w = max(12, 2.4 * len(order) + 4)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 6))

    panels = [
        ('Winner_Slope_Whole',
         r'Rate: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law',
         'Slope (winner-model)', False, False),
        ('MI_Whole_PL_a',
         r'Amplitude $a$ (µm)',
         'Power-Law amplitude a (PL winners only)', True, True),
    ]

    for i, (col, ylabel, title, log_y, pl_only) in enumerate(panels):
        ax = axes[i]
        sub = df[df[col].notna()].copy()
        if pl_only:
            sub = sub[sub['MI_Whole_Best_Model'] == 'Power-Law']
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(data=sub, x='Bucket', y=col, order=order,
                    ax=ax, showfliers=False, color='lightgray')
        if pl_only:
            sns.stripplot(data=sub, x='Bucket', y=col, order=order,
                          color=_MI_MODEL_UTILS_COLOR['Power-Law'],
                          dodge=False, alpha=0.7, ax=ax, size=5)
        else:
            sns.stripplot(data=sub, x='Bucket', y=col, order=order,
                          hue='MI_Whole_Best_Model',
                          palette=_MI_MODEL_UTILS_COLOR,
                          dodge=False, alpha=0.7, ax=ax, size=5)

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
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_MI_WholeTrace_Parameter_Boxplots.pdf")
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

    df['Bucket']              = df.apply(_row_bucket_wholetrace_fate, axis=1)
    df['Winner_Slope_Whole']  = df.apply(_mi_whole_winner_slope, axis=1)

    for bucket, sub_bucket in df.groupby('Bucket'):
        if sub_bucket.empty:
            continue

        trap_ids_sorted = sorted(sub_bucket['Trap_ID'].dropna().astype(int).unique())
        trap_order = [f"T{t}" for t in trap_ids_sorted]
        sub_bucket = sub_bucket.copy()
        sub_bucket['Trap'] = 'T' + sub_bucket['Trap_ID'].astype(int).astype(str)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(bucket, fontweight='bold', y=1.02)

        ax = axes[0]
        sub = sub_bucket[sub_bucket['Winner_Slope_Whole'].notna()]
        if sub.empty:
            ax.set_visible(False)
        else:
            sns.boxplot(data=sub, x='Trap', y='Winner_Slope_Whole', order=trap_order,
                        ax=ax, showfliers=False, color='lightgray',
                        boxprops=dict(alpha=0.4))
            sns.stripplot(data=sub, x='Trap', y='Winner_Slope_Whole', order=trap_order,
                          hue='MI_Whole_Best_Model', palette=_MI_MODEL_UTILS_COLOR,
                          dodge=False, alpha=0.75, ax=ax, size=5)
            ax.set_title('Slope (winner-model)', fontweight='bold')
            ax.set_ylabel(r'Rate: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law')
            ax.set_xlabel('Trap ID')
            ax.tick_params(axis='x', rotation=45)

        ax = axes[1]
        sub_pl = sub_bucket[
            (sub_bucket['MI_Whole_Best_Model'] == 'Power-Law')
            & sub_bucket['MI_Whole_PL_a'].notna()
        ]
        if sub_pl.empty:
            ax.set_visible(False)
        else:
            pl_colour = _MI_MODEL_UTILS_COLOR['Power-Law']
            sns.boxplot(data=sub_pl, x='Trap', y='MI_Whole_PL_a', order=trap_order,
                        ax=ax, showfliers=False, color='lightgray',
                        boxprops=dict(alpha=0.4))
            sns.stripplot(data=sub_pl, x='Trap', y='MI_Whole_PL_a', order=trap_order,
                          color=pl_colour, dodge=False, alpha=0.75, ax=ax, size=5)
            ax.set_title('Power-Law amplitude a (PL winners only)',
                         fontweight='bold')
            ax.set_ylabel(r'Amplitude $a$ (µm)')
            ax.set_xlabel('Trap ID')
            ax.tick_params(axis='x', rotation=45)
            if (sub_pl['MI_Whole_PL_a'] > 0).all():
                ax.set_yscale('log')

        plt.tight_layout()
        target_dir = output_dir / _get_subfolder_from_bucket(bucket)
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_MI_WholeTrace_PerTrap_Params_{_safe_filename(bucket)}.pdf")
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
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_MI_WholeTrace_Protrusion_Dynamics.pdf")
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
                subfolder = 'ASP'
            elif is_post_subgroup:
                subtitle = f'EP post-pulse-entry whole trace  ({file_label})'
                subfolder = f"{meta.voltage}V_{meta.duration_label}"
            else:
                subtitle = f'EP whole trace including pulse  ({file_label})'
                subfolder = f"{meta.voltage}V_{meta.duration_label}"
                
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
            
            target_dir = output_dir / subfolder
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / f"MI_WholeTrace_Fits_Panel_{file_label}.png"
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

    df['Bucket']       = df.apply(_row_bucket, axis=1)
    df['Winner_Slope'] = df.apply(_mi_winner_slope, axis=1)

    body_uptake = 'Uptake_Body_VolNorm_A'
    prot_uptake = 'Uptake_Prot_VolNorm_A'

    for bucket, sub_df in df.groupby('Bucket'):
        if len(sub_df) < 5:
            continue

        bucket_colour = MI_ASP_COLOR if bucket == 'ASP' else (
            MI_EP_COLOR_RAMP[0] if MI_EP_COLOR_RAMP else utils.MFA_COLORS['medium_red']
        )

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        fig.suptitle(f"Mechanics & Spatial Position vs. Uptake: {bucket}",
                     fontweight='bold', y=1.02)
        (ax_tl, ax_tr), (ax_bl, ax_br) = axes

        def _annotate_rho(ax, x, y, xy=(0.05, 0.95)):
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 3:
                return
            rho, p = stats.spearmanr(x[valid], y[valid])
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.text(*xy, f"$\\rho$={rho:.2f}\n{p_str}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', alpha=0.8, ec='gray'))

        def _log_safe(y):
            return np.where(y > 1e-3, y, 1e-3)

        rng = np.random.default_rng(seed=42)

        m = sub_df['Trap_ID'].notna() & sub_df[body_uptake].notna()
        if m.sum() >= 3:
            x = sub_df.loc[m, 'Trap_ID'].to_numpy(dtype=float)
            y = _log_safe(sub_df.loc[m, body_uptake].to_numpy())
            x_j = x + rng.uniform(-0.25, 0.25, size=len(x))
            sns.regplot(x=x_j, y=y, ax=ax_tl,
                        scatter_kws={'alpha': 0.65, 'color': bucket_colour,
                                     'edgecolor': 'white', 'linewidths': 0.5},
                        line_kws={'color': 'black', 'lw': 1.5, 'ls': '--'})
            _annotate_rho(ax_tl, x, y)
        ax_tl.set_xlabel("Trap ID (1=Far, 18=Near)")
        ax_tl.set_ylabel("Body Uptake Plateau\n(ADU/µm³) [log]")
        ax_tl.set_yscale('log')
        ax_tl.set_xticks(range(2, 19, 2)); ax_tl.set_xlim(0, 19)

        m = sub_df['Trap_ID'].notna() & sub_df[prot_uptake].notna()
        n_prot = int(m.sum())
        if n_prot >= 3:
            x = sub_df.loc[m, 'Trap_ID'].to_numpy(dtype=float)
            y = _log_safe(sub_df.loc[m, prot_uptake].to_numpy())
            x_j = x + rng.uniform(-0.25, 0.25, size=len(x))
            sns.regplot(x=x_j, y=y, ax=ax_tr,
                        scatter_kws={'alpha': 0.65, 'color': bucket_colour,
                                     'edgecolor': 'white', 'linewidths': 0.5},
                        line_kws={'color': 'black', 'lw': 1.5, 'ls': '--'})
            _annotate_rho(ax_tr, x, y)
        ax_tr.set_xlabel("Trap ID (1=Far, 18=Near)")
        ax_tr.set_ylabel("Protrusion Uptake Plateau\n(ADU/µm³) [log]")
        ax_tr.set_yscale('log')
        ax_tr.set_xticks(range(2, 19, 2)); ax_tr.set_xlim(0, 19)
        ax_tr.text(0.98, 0.02, f"n={n_prot}", transform=ax_tr.transAxes,
                   ha='right', va='bottom', fontsize=8, color='gray')

        m = sub_df['Winner_Slope'].notna() & sub_df[prot_uptake].notna()
        if m.sum() >= 3:
            sub_m = sub_df.loc[m, ['Winner_Slope', prot_uptake, 'MI_Best_Model']].copy()
            sub_m['_y'] = _log_safe(sub_m[prot_uptake].to_numpy())
            for model_name, colour in _MI_MODEL_UTILS_COLOR.items():
                pts = sub_m[sub_m['MI_Best_Model'] == model_name]
                if pts.empty:
                    continue
                ax_bl.scatter(pts['Winner_Slope'], pts['_y'],
                              s=30, alpha=0.75, color=colour,
                              edgecolor='white', linewidths=0.5,
                              label=f"{model_name} (n={len(pts)})")
            _annotate_rho(ax_bl, sub_m['Winner_Slope'].to_numpy(),
                          sub_m['_y'].to_numpy())
            ax_bl.legend(frameon=False, fontsize=8, loc='lower right')
        ax_bl.set_xlabel(r'Slope: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law')
        ax_bl.set_ylabel("Protrusion Uptake Plateau\n(ADU/µm³) [log]")
        ax_bl.set_yscale('log')

        m = (
            (sub_df['MI_Best_Model'] == 'Power-Law')
            & sub_df['PL_a'].notna()
            & sub_df[prot_uptake].notna()
        )
        n_pla = int(m.sum())
        if n_pla >= 3:
            x = sub_df.loc[m, 'PL_a'].to_numpy(dtype=float)
            y = _log_safe(sub_df.loc[m, prot_uptake].to_numpy())
            sns.regplot(x=x, y=y, ax=ax_br,
                        scatter_kws={'alpha': 0.75,
                                     'color': _MI_MODEL_UTILS_COLOR['Power-Law'],
                                     'edgecolor': 'white', 'linewidths': 0.5},
                        line_kws={'color': 'black', 'lw': 1.5, 'ls': '--'})
            _annotate_rho(ax_br, x, y)
        ax_br.set_xlabel(r"Power-Law amplitude $a$ (µm)")
        ax_br.set_ylabel("Protrusion Uptake Plateau\n(ADU/µm³) [log]")
        ax_br.set_yscale('log')
        ax_br.text(0.98, 0.02, f"n(PL winners)={n_pla}",
                   transform=ax_br.transAxes, ha='right', va='bottom',
                   fontsize=8, color='gray')

        for ax in (ax_tl, ax_tr, ax_bl, ax_br):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        target_dir = output_dir / _get_subfolder_from_bucket(bucket)
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_Mechanics_vs_Uptake_{_safe_filename(bucket)}.pdf")
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
    
    mech_cols = [
        ('Linear_Slope', 'Linear Slope (µm/s)'), 
        ('PL_a', 'Power-Law a (µm)'),
        ('PL_b', 'Power-Law b')
    ]
    uptake_col = 'Uptake_Prot_VolNorm_A'
    u_label = 'Log10 Protrusion Uptake\nPlateau (ADU/µm³)'
                   
    for bucket, sub_df in df.groupby('Bucket'):
        if bucket == 'ASP':
            continue
        if len(sub_df) < 5:
            continue

        fig = plt.figure(figsize=(22, 7))
        fig.suptitle(f"Protrusion Spatial Trends: {bucket}", fontweight='bold', y=0.98, fontsize=14)
        
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
        target_dir = output_dir / _get_subfolder_from_bucket(bucket)
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_Spatial_Mech_Uptake_3D_{_safe_filename(bucket)}.pdf")
        plt.close()

def plot_thesis_fate_counts(mechanics_df: pd.DataFrame,
                             attrition_df: pd.DataFrame,
                             output_dir: Path) -> None:
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

    ep_mech['Pulse_Condition'] = ep_mech['Voltage_V'].astype(str) + 'V_' + ep_mech['Duration_label']

    counts_analysis = (
        ep_mech.groupby(['Pulse_Condition', 'Fate_Status'])
        .size()
        .unstack(fill_value=0)
    )
    for col in ('intact', 'ruptured_post'):
        if col not in counts_analysis.columns:
            counts_analysis[col] = 0

    EXCLUDED_BUCKETS = [
        'ruptured_pre_pulse',
        'not_viable',
        'post_pulse_arrival',
        'detection_failure',
    ]
    excluded_counts = pd.Series(dtype=int)
    if attrition_df is not None and not attrition_df.empty:
        ep_att = attrition_df.loc[attrition_df['Condition_Type'] == 'EP'].copy()
        if not ep_att.empty:
            ep_att['Pulse_Condition'] = ep_att['Voltage_V'].astype(str) + 'V_' + ep_att['Duration_label']
            present_buckets = [b for b in EXCLUDED_BUCKETS if b in ep_att.columns]
            if present_buckets:
                excluded_counts = (
                    ep_att.assign(_excluded=ep_att[present_buckets].sum(axis=1))
                          .groupby('Pulse_Condition')['_excluded'].sum()
                )

    pulse_conds = sorted(set(counts_analysis.index) | set(excluded_counts.index))
    tidy = pd.DataFrame({
        'intact'        : [counts_analysis.get('intact',        pd.Series()).get(pc, 0) for pc in pulse_conds],
        'ruptured_post' : [counts_analysis.get('ruptured_post', pd.Series()).get(pc, 0) for pc in pulse_conds],
        'excluded'      : [int(excluded_counts.get(pc, 0))                              for pc in pulse_conds],
    }, index=pulse_conds)

    colours = {
        'intact'        : utils.MFA_COLORS['dark_blue'],
        'ruptured_post' : utils.MFA_COLORS['medium_red'],
        'excluded'      : utils.MFA_COLORS['light_grey'],
    }
    labels = {
        'intact'        : 'Intact',
        'ruptured_post' : 'Ruptured (post-pulse)',
        'excluded'      : 'Excluded (all reasons)',
    }

    for pc in pulse_conds:
        counts = [
            int(tidy.loc[pc, 'intact']),
            int(tidy.loc[pc, 'ruptured_post']),
            int(tidy.loc[pc, 'excluded']),
        ]
        keys_present = [k for k, c in zip(('intact', 'ruptured_post', 'excluded'), counts) if c > 0]
        vals_present = [c for c in counts if c > 0]
        if not vals_present:
            logger.warning(f"Fate pie ({pc}) skipped: all-zero counts.")
            continue

        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        total = sum(vals_present)
        pie_colours = [colours[k] for k in keys_present]
        pie_labels  = [f"{labels[k]}\n(n={c}, {100*c/total:.0f}%)"
                       for k, c in zip(keys_present, vals_present)]

        ax.pie(vals_present, labels=pie_labels, colors=pie_colours,
               startangle=90, counterclock=False,
               wedgeprops=dict(edgecolor='white', linewidth=1.2),
               textprops=dict(fontsize=8))
        ax.set_title(f"{pc} pulse  (N={total} candidates)",
                     fontsize=10, fontweight='bold')
        ax.axis('equal')

        plt.tight_layout()
        target_dir = output_dir / pc
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_Fate_Pie_{pc}.pdf")
        plt.close()

    tidy_out = tidy.copy()
    tidy_out.index.name = 'Pulse_Condition'
    tidy_out['analysis_total']    = tidy_out['intact'] + tidy_out['ruptured_post']
    tidy_out['candidate_total']   = tidy_out['analysis_total'] + tidy_out['excluded']
    tidy_out['fraction_ruptured'] = np.where(
        tidy_out['analysis_total'] > 0,
        tidy_out['ruptured_post'] / tidy_out['analysis_total'],
        np.nan,
    )
    tidy_out['fraction_excluded'] = np.where(
        tidy_out['candidate_total'] > 0,
        tidy_out['excluded'] / tidy_out['candidate_total'],
        np.nan,
    )
    
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    tidy_out.to_csv(combined_dir / "Thesis_Fate_Counts_by_Pulse.csv")
    logger.info(f"Fate histogram written; fractions ruptured: "
                + ", ".join(f"{pc}={f:.2f}" for pc, f in tidy_out['fraction_ruptured'].items() if np.isfinite(f)))


def _duration_sort_key(label: str) -> float:
    label = str(label).strip().lower()
    for unit, factor in (('us', 1e-3), ('ms', 1.0), ('s', 1e3)):
        if label.endswith(unit):
            try:
                return float(label[:-len(unit)]) * factor
            except ValueError:
                return float('inf')
    return float('inf')

def plot_thesis_fate_per_trap(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Fate per-trap skipped: mechanics_df is empty.")
        return
    if 'Fate_Status' not in mechanics_df.columns:
        logger.warning("Fate per-trap skipped: no Fate_Status column.")
        return

    ep_mech = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'EP')
        & mechanics_df['Fate_Status'].isin(['intact', 'ruptured_post'])
    ].copy()
    if ep_mech.empty:
        logger.warning("Fate per-trap skipped: no EP cells in mechanics_df.")
        return

    colours = {
        'intact'        : utils.MFA_COLORS['dark_blue'],
        'ruptured_post' : utils.MFA_COLORS['medium_red'],
    }
    labels = {
        'intact'        : 'Intact',
        'ruptured_post' : 'Ruptured (post-pulse)',
    }

    ep_mech['Pulse_Condition'] = ep_mech['Voltage_V'].astype(str) + 'V_' + ep_mech['Duration_label']
    pulse_conds = sorted(ep_mech['Pulse_Condition'].dropna().unique(), key=lambda x: _duration_sort_key(x.split('_')[1]))

    for pc in pulse_conds:
        sub = ep_mech.loc[ep_mech['Pulse_Condition'] == pc]
        if sub.empty:
            continue

        counts = (
            sub.groupby(['Trap_ID', 'Fate_Status'])
               .size()
               .unstack(fill_value=0)
        )
        for c in ('intact', 'ruptured_post'):
            if c not in counts.columns:
                counts[c] = 0
        counts = counts.sort_index()

        trap_ids   = counts.index.astype(int).to_numpy()
        intact_v   = counts['intact'].to_numpy()
        ruptured_v = counts['ruptured_post'].to_numpy()

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        x = np.arange(len(trap_ids))
        ax.bar(x, intact_v,
               color=colours['intact'], label=labels['intact'],
               edgecolor='white', linewidth=0.5)
        ax.bar(x, ruptured_v, bottom=intact_v,
               color=colours['ruptured_post'], label=labels['ruptured_post'],
               edgecolor='white', linewidth=0.5)

        for xi, i_v, r_v in zip(x, intact_v, ruptured_v):
            if i_v + r_v > 0:
                ax.text(xi, i_v + r_v + 0.15, str(int(i_v + r_v)),
                        ha='center', va='bottom', fontsize=7, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in trap_ids], rotation=45, ha='right')
        ax.set_xlabel("Trap ID")
        ax.set_ylabel("Cell count (analysable cohort)")
        ax.set_title(f"Fate per trap — {pc} pulse (N={int(intact_v.sum() + ruptured_v.sum())})",
                     fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper right')

        plt.tight_layout()
        
        target_dir = output_dir / pc
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_Fate_PerTrap_{pc}.pdf")
        plt.close()

    logger.info(f"Fate per-trap stacked bars written for conditions: " + ", ".join(pulse_conds))

_TREATMENT_COLOR = {
    'WT'  : utils.MFA_COLORS['dark_blue'],
    'CytD': utils.MFA_COLORS['medium_red'],
}
_TREATMENT_ORDER = ['WT', 'CytD']

def _actin_f0_trace(trap: bfh.TrapData, region: str):
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

    f0_scalar = next((float(v) for v in F0 if np.isfinite(v) and v > 0), None)
    if f0_scalar is None:
        return None, None

    t_aligned = t - t[0]

    valid = np.isfinite(t_aligned) & np.isfinite(I) & (I > 0)
    if valid.sum() < 5:
        return None, None

    return t_aligned[valid], I[valid] / f0_scalar

def plot_thesis_asp_actin_trace_by_condition(grouped_data: Dict,
                                              mechanics_df: pd.DataFrame,
                                              output_dir: Path,
                                              global_asp_dur: Optional[float] = None) -> None:
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

    t_cap = global_asp_dur if global_asp_dur is not None else max_t_seen
    if t_cap <= 0:
        logger.warning("Actin trace plot skipped: no valid duration.")
        return

    common_t = np.linspace(0.0, t_cap, 60)

    def _aggregate(traces):
        if not traces:
            return None, None
        stack = []
        for t, y in traces:
            if t[-1] < t_cap * 0.5:
                continue
            f = interp1d(t, y, bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            stack.append(f(common_t))
        if not stack:
            return None, None
        arr = np.vstack(stack)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
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
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    axes[0].set_ylabel(r"Actin intensity $I(t)/F_0$")
    axes[0].legend(frameon=False, fontsize=8, loc='best')

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_ASP_Actin_Trace_by_Condition.pdf")
    plt.close()
    logger.info("Actin trace plot written.")

def plot_thesis_asp_actin_vs_mechanics(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
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

    mech_panels = [
        ('E_Pa',      r"$E$ (Pa)",                        True),
        ('eta1_Pa_s', r"$\eta_1$ (Pa$\cdot$s)",           True),
        ('Tau_s',     r"$\tau$ (s, Kelvin-Voigt only)",   True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharey=False)
    for row_idx, region in enumerate(('Body', 'Prot')):
        actin_col = f'Actin_{region}_PrePulse_F0Norm'
        for col_idx, (mech_col, mech_label, use_log) in enumerate(mech_panels):
            ax = axes[row_idx, col_idx]
            if actin_col not in df.columns or mech_col not in df.columns:
                ax.set_visible(False)
                continue

            for treatment in _TREATMENT_ORDER:
                sub = df.loc[df['Treatment'] == treatment, [actin_col, mech_col]].dropna()
                if sub.empty:
                    continue
                colour = _TREATMENT_COLOR[treatment]
                ax.scatter(sub[actin_col], sub[mech_col], s=22,
                           color=colour, edgecolor='white', linewidths=0.4,
                           alpha=0.75, label=f"{treatment} (n={len(sub)})")

                if len(sub) >= 5:
                    rho, p = stats.spearmanr(sub[actin_col], sub[mech_col])
                    y_anchor = 0.95 if treatment == 'WT' else 0.88
                    ax.text(0.03, y_anchor,
                            rf"$\rho_{{{treatment}}}={rho:.2f}$ (p={p:.2g})",
                            transform=ax.transAxes, fontsize=7.5,
                            color=colour, va='top')

            if use_log:
                ax.set_yscale('log')
            if row_idx == 1:
                ax.set_xlabel(rf"Actin $I/F_0$ ({region.lower()}, pre-pulse)")
            if col_idx == 0:
                ax.set_ylabel(f"{region}\n{mech_label}")
            else:
                ax.set_ylabel(mech_label)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    axes[0, 0].legend(frameon=False, fontsize=7.5, loc='lower right')

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_ASP_Actin_vs_E.pdf")
    plt.close()
    logger.info("Actin-vs-E scatter written.")

def plot_thesis_asp_trap_dependency(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
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
            sub = df.loc[df['Treatment'] == treatment, ['Trap_ID', col]].dropna()
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
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_ASP_Trap_Dependency.pdf")
    plt.close()
    logger.info("ASP trap-dependency scatter written.")

def plot_thesis_asp_body_volume_vs_E(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
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
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_ASP_BodyVolume_vs_E.pdf")
    plt.close()
    logger.info("Body volume vs E scatter written.")

def plot_thesis_asp_prot_length_vs_mechanics(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    from scipy import stats
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'ASP')
        & (mechanics_df['Fate_Status'] == 'intact')
        & mechanics_df['E_Pa'].notna()
        & (mechanics_df['Visco_R2_Flag'] == True)
        & mechanics_df['Max_Prot_length_PrePulse_um'].notna()
    ].copy()

    if df.empty:
        logger.warning("Prot length vs mechanics plot skipped: no eligible cells.")
        return

    panels = [
        ('E_Pa',      r"$E$ (Pa)"),
        ('eta1_Pa_s', r"$\eta_1$ (Pa$\cdot$s)"),
        ('Tau_s',     r"$\tau$ (s, Kelvin-Voigt only)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (col, ylabel) in zip(axes, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        for treatment in _TREATMENT_ORDER:
            sub = df.loc[df['Treatment'] == treatment,
                         ['Max_Prot_length_PrePulse_um', col]].dropna()
            if sub.empty:
                continue
            colour = _TREATMENT_COLOR[treatment]
            ax.scatter(sub['Max_Prot_length_PrePulse_um'], sub[col],
                       s=22, color=colour, edgecolor='white', linewidths=0.4,
                       alpha=0.75, label=f"{treatment} (n={len(sub)})")

            if len(sub) >= 5:
                rho, p = stats.spearmanr(
                    sub['Max_Prot_length_PrePulse_um'], sub[col])
                y_anchor = 0.95 if treatment == 'WT' else 0.88
                ax.text(0.03, y_anchor,
                        rf"$\rho_{{{treatment}}}={rho:.2f}$ (p={p:.2g})",
                        transform=ax.transAxes, fontsize=7.5, color=colour,
                        va='top')

        ax.set_yscale('log')
        ax.set_xlabel(r"Max protrusion length (µm, pre-pulse)")
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    axes[0].legend(frameon=False, fontsize=8, loc='best')

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_ASP_ProtLength_vs_Mechanics.pdf")
    plt.close()
    logger.info("Prot-length vs mechanics scatter written.")

def plot_thesis_asp_actin_f0_boxplot(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'ASP')
        & (mechanics_df['Fate_Status'] == 'intact')
        & mechanics_df['E_Pa'].notna()
        & (mechanics_df['Visco_R2_Flag'] == True)
    ].copy()

    if df.empty:
        logger.warning("F0 boxplot skipped: no eligible ASP cells.")
        return

    if 'F0_Body' not in df.columns or 'F0_Prot' not in df.columns:
        logger.warning("F0 boxplot skipped: F0 columns not in mechanics_df.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.02, 3.2), sharey=False)
    panels = [
        ('F0_Body', r'$F_0$ body (ADU)',       'Body'),
        ('F0_Prot', r'$F_0$ protrusion (ADU)', 'Protrusion'),
    ]

    for ax, (col, ylabel, title) in zip(axes, panels):
        sub = df.loc[df[col].notna(), ['Treatment', col]].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(data=sub, x='Treatment', y=col, order=_TREATMENT_ORDER,
                    ax=ax, showfliers=False, color='lightgray')
        for i, treatment in enumerate(_TREATMENT_ORDER):
            pts = sub.loc[sub['Treatment'] == treatment, col].to_numpy()
            if len(pts) == 0:
                continue
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(-0.15, 0.15, size=len(pts))
            ax.scatter(np.full(len(pts), i) + jitter, pts,
                       s=22, color=_TREATMENT_COLOR[treatment],
                       edgecolor='white', linewidths=0.4, alpha=0.75, zorder=3)

        vals_wt   = sub.loc[sub['Treatment'] == 'WT',   col].dropna().to_numpy()
        vals_cytd = sub.loc[sub['Treatment'] == 'CytD', col].dropna().to_numpy()
        stars, label, lw, delta = bp._build_stat_label(vals_cytd, vals_wt)
        
        if label is not None:
            x1 = _TREATMENT_ORDER.index('CytD')
            x2 = _TREATMENT_ORDER.index('WT')
            y_top = float(np.nanmax(np.concatenate([vals_wt, vals_cytd])))
            bp._add_bracket(ax, x1, x2, y_top, label, lw=lw)

        if (sub[col] > 0).all():
            ax.set_yscale('log')

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_ASP_Actin_F0_Boxplot.pdf")
    plt.close()

def run_thesis_claim1_plots(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path, global_asp_dur: Optional[float] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_thesis_asp_actin_trace_by_condition(grouped_data, mechanics_df, output_dir, global_asp_dur)
    plot_thesis_asp_actin_vs_mechanics(mechanics_df, output_dir)
    plot_thesis_asp_trap_dependency(mechanics_df, output_dir)
    plot_thesis_asp_body_volume_vs_E(mechanics_df, output_dir)
    plot_thesis_asp_prot_length_vs_mechanics(mechanics_df, output_dir)

_FATE_COLOR = {
    'intact'        : utils.MFA_COLORS['dark_blue'],
    'ruptured_post' : utils.MFA_COLORS['medium_red'],
    'asp_baseline'  : '#808080',
}

def _ep_fate_bucket(duration_label: str, fate_status: str) -> str:
    dl = str(duration_label) if duration_label is not None else '?'
    fs = str(fate_status)   if fate_status   is not None else '?'
    fs_short = {'intact': 'intact', 'ruptured_post': 'ruptured'}.get(fs, fs)
    return f"{dl}_{fs_short}"

def plot_thesis_ep_fate_mi_boxplots(mechanics_df: pd.DataFrame, output_dir: Path, r2_floor: float = 0.80) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Fate-mechanics boxplot skipped: empty mechanics_df.")
        return

    df = mechanics_df.copy()
    df['MI_Winner_R2'] = df.apply(_mi_winner_r2, axis=1)

    ep_df = df.loc[
        (df['Condition_Type'] == 'EP')
        & df['MI_Best_Model'].notna()
        & (df['MI_Winner_R2'] >= r2_floor)
        & df['Fate_Status'].isin(['intact', 'ruptured_post'])
    ].copy()

    if ep_df.empty:
        logger.warning("Fate-mechanics boxplot skipped: no EP cells pass MI R² floor.")
        return

    ep_df['Bucket']       = ep_df.apply(
        lambda r: _ep_fate_bucket(r.get('Duration_label'), r.get('Fate_Status')),
        axis=1,
    )
    ep_df['Winner_Slope'] = ep_df.apply(_mi_winner_slope, axis=1)

    durations = sorted(
        ep_df['Duration_label'].dropna().unique(),
        key=_duration_sort_key,
    )
    order = []
    for d in durations:
        for fate_short in ('intact', 'ruptured'):
            b = f"{d}_{fate_short}"
            if b in ep_df['Bucket'].values:
                order.append(b)

    panels = [
        ('Winner_Slope',
         r'Rate: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law',
         'Slope (winner-model)', False, False),
        ('PL_a',
         r'Amplitude $a$ (µm)',
         'Power-Law amplitude a (PL winners only)', True, True),
    ]

    _fate_box_tint = {
        'intact'        : utils.MFA_COLORS['pale_blue'],
        'ruptured_post' : utils.MFA_COLORS['pale_red'],
    }

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(order) + 6), 3.8))

    for i, (col, ylabel, title, log_y, pl_only) in enumerate(panels):
        ax = axes[i]
        sub = ep_df.loc[ep_df[col].notna(),
                        ['Bucket', col, 'Fate_Status', 'MI_Best_Model']].copy()
        if pl_only:
            sub = sub[sub['MI_Best_Model'] == 'Power-Law']
        if sub.empty:
            ax.set_visible(False)
            continue

        box_palette = {}
        for b in order:
            rows = sub[sub['Bucket'] == b]
            if rows.empty:
                continue
            fate = rows['Fate_Status'].iloc[0]
            box_palette[b] = _fate_box_tint.get(fate, utils.MFA_COLORS['light_grey'])

        sns.boxplot(data=sub, x='Bucket', y=col, order=order,
                    ax=ax, showfliers=False,
                    palette=box_palette, hue='Bucket', legend=False)

        for j, bucket in enumerate(order):
            rows = sub[sub['Bucket'] == bucket]
            if rows.empty:
                continue
            rng = np.random.default_rng(42 + j)
            jitter = rng.uniform(-0.15, 0.15, size=len(rows))
            colours = rows['MI_Best_Model'].map(_MI_MODEL_UTILS_COLOR).fillna('gray')
            ax.scatter(np.full(len(rows), j) + jitter, rows[col],
                       s=22, c=colours,
                       edgecolor='white', linewidths=0.4, alpha=0.85, zorder=3)

        if log_y and (sub[col].dropna() > 0).all():
            ax.set_yscale('log')

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for d in durations:
            b_intact = f"{d}_intact"
            b_rupt   = f"{d}_ruptured"
            if b_intact not in order or b_rupt not in order:
                continue
            v_int = sub.loc[sub['Bucket'] == b_intact, col].dropna().to_numpy()
            v_rup = sub.loc[sub['Bucket'] == b_rupt,   col].dropna().to_numpy()
            if len(v_int) == 0 or len(v_rup) == 0:
                continue
            stars, label, lw, delta = bp._build_stat_label(v_int, v_rup)
            if label is None:
                continue
            x1 = order.index(b_intact)
            x2 = order.index(b_rupt)
            y_top = float(np.nanmax(np.concatenate([v_int, v_rup])))
            bp._add_bracket(ax, x1, x2, y_top, label, lw=lw)

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    handles = [
        mpatches.Patch(facecolor=utils.MFA_COLORS['pale_blue'],
                       edgecolor='gray', label='Intact (box tint)'),
        mpatches.Patch(facecolor=utils.MFA_COLORS['pale_red'],
                       edgecolor='gray', label='Ruptured (box tint)'),
        mlines.Line2D([], [], marker='o', linestyle='',
                      markerfacecolor=_MI_MODEL_UTILS_COLOR['Linear'],
                      markeredgecolor='white',
                      label='Linear winner (point)'),
        mlines.Line2D([], [], marker='o', linestyle='',
                      markerfacecolor=_MI_MODEL_UTILS_COLOR['Power-Law'],
                      markeredgecolor='white',
                      label='Power-Law winner (point)'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=4,
               frameon=False, fontsize=8)

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_EP_Fate_MI_Boxplots.pdf")
    plt.close()

def _fate_box_bucket(row: pd.Series) -> str:
    if row.get('Condition_Type') == 'ASP':
        if row.get('Treatment') == 'WT':
            return 'ASP_WT'
        return f"ASP_{row.get('Treatment', '?')}"
    fate = row.get('Fate_Status')
    fate_short = 'intact' if fate == 'intact' else ('ruptured' if fate == 'ruptured_post' else str(fate))
    return f"{row.get('Duration_label')}_{fate_short}"

def _fate_box_order(buckets_present: list, durations: list) -> list:
    asp_here = sorted(b for b in buckets_present if b.startswith('ASP_'))
    order = list(asp_here)
    for d in durations:
        for fate_short in ('intact', 'ruptured'):
            b = f"{d}_{fate_short}"
            if b in buckets_present:
                order.append(b)
    return order

def plot_thesis_fate_prepulse_visco_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Fate pre-pulse viscoelastic boxplot skipped: empty mechanics_df.")
        return

    df = mechanics_df.copy()
    df = df.loc[
        df['PrePulse_Best_Model'].notna()
        & df['PrePulse_E_Pa'].notna()
        & (df['PrePulse_Visco_R2_Flag'] == True)
    ].copy()

    keep_asp = (df['Condition_Type'] == 'ASP') & (df['Treatment'] == 'WT')
    keep_ep  = (df['Condition_Type'] == 'EP') & df['Fate_Status'].isin(['intact', 'ruptured_post'])
    df = df[keep_asp | keep_ep].copy()

    if df.empty:
        logger.warning("Fate pre-pulse viscoelastic boxplot skipped: no cells pass filter.")
        return

    df['Bucket'] = df.apply(_fate_box_bucket, axis=1)

    durations = sorted(
        df.loc[df['Condition_Type'] == 'EP', 'Duration_label'].dropna().unique(),
        key=_duration_sort_key,
    )
    order = _fate_box_order(df['Bucket'].unique().tolist(), durations)

    panels = [
        ('PrePulse_E_Pa',      r'$E$ (Pa)',
         'Parallel elastic modulus'),
        ('PrePulse_eta2_Pa_s', r'$\eta_{2}$ (Pa$\cdot$s)',
         'Flow viscosity (Jeffreys / Burgers)'),
    ]

    _fate_box_tint = {
        'ASP'          : utils.MFA_COLORS['light_grey'],
        'intact'       : utils.MFA_COLORS['pale_blue'],
        'ruptured_post': utils.MFA_COLORS['pale_red'],
    }

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(order) + 6), 3.8))

    for i, (col, ylabel, title) in enumerate(panels):
        ax = axes[i]
        sub = df.loc[df[col].notna(),
                     ['Bucket', col, 'Fate_Status', 'Condition_Type', 'PrePulse_Best_Model']].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        box_palette = {}
        for b in order:
            rows = sub[sub['Bucket'] == b]
            if rows.empty:
                continue
            if rows['Condition_Type'].iloc[0] == 'ASP':
                box_palette[b] = _fate_box_tint['ASP']
            else:
                fate = rows['Fate_Status'].iloc[0]
                box_palette[b] = _fate_box_tint.get(fate, utils.MFA_COLORS['light_grey'])

        sns.boxplot(data=sub, x='Bucket', y=col, order=order,
                    ax=ax, showfliers=False,
                    palette=box_palette, hue='Bucket', legend=False)

        for j, bucket in enumerate(order):
            rows = sub[sub['Bucket'] == bucket]
            if rows.empty:
                continue
            rng = np.random.default_rng(42 + j)
            jitter = rng.uniform(-0.15, 0.15, size=len(rows))
            colours = rows['PrePulse_Best_Model'].map(
                lambda m: bp.VISCO_MODEL_PALETTE.get(m, 'gray')
            )
            ax.scatter(np.full(len(rows), j) + jitter, rows[col],
                       s=22, c=colours,
                       edgecolor='white', linewidths=0.4, alpha=0.85, zorder=3)

        if (sub[col].dropna() > 0).all():
            ax.set_yscale('log')

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for d in durations:
            b_intact = f"{d}_intact"
            b_rupt   = f"{d}_ruptured"
            if b_intact not in order or b_rupt not in order:
                continue
            v_int = sub.loc[sub['Bucket'] == b_intact, col].dropna().to_numpy()
            v_rup = sub.loc[sub['Bucket'] == b_rupt,   col].dropna().to_numpy()
            if len(v_int) < 2 or len(v_rup) < 2:
                continue
            stars, label, lw, delta = bp._build_stat_label(v_int, v_rup)
            if label is None:
                continue
            x1 = order.index(b_intact)
            x2 = order.index(b_rupt)
            y_top = float(np.nanmax(np.concatenate([v_int, v_rup])))
            bp._add_bracket(ax, x1, x2, y_top, label, lw=lw)

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    handles = [
        mpatches.Patch(facecolor=utils.MFA_COLORS['light_grey'],
                       edgecolor='gray', label='ASP-WT reference'),
        mpatches.Patch(facecolor=utils.MFA_COLORS['pale_blue'],
                       edgecolor='gray', label='EP intact'),
        mpatches.Patch(facecolor=utils.MFA_COLORS['pale_red'],
                       edgecolor='gray', label='EP ruptured'),
    ]
    for model_name in ('Kelvin-Voigt', 'Jeffreys', 'Burgers'):
        c = bp.VISCO_MODEL_PALETTE.get(model_name)
        if c is None:
            continue
        handles.append(mlines.Line2D([], [], marker='o', linestyle='',
                                     markerfacecolor=c,
                                     markeredgecolor='white',
                                     label=f'{model_name} winner'))
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.08), ncol=3,
               frameon=False, fontsize=8)

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_Fate_PrePulse_Visco_Boxplots.pdf")
    plt.close()

def plot_thesis_fate_wholetrace_mi_boxplots(mechanics_df: pd.DataFrame, output_dir: Path, r2_floor: float = 0.80) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Fate whole-trace MI boxplot skipped: empty mechanics_df.")
        return

    df = mechanics_df.copy()

    def _wt_winner_r2(row):
        m = row.get('MI_Whole_Best_Model')
        if m == 'Linear':    return row.get('MI_Whole_Linear_R2', np.nan)
        if m == 'Power-Law': return row.get('MI_Whole_PL_R2', np.nan)
        return np.nan

    def _wt_winner_slope(row):
        m = row.get('MI_Whole_Best_Model')
        if m == 'Linear':    return row.get('MI_Whole_Linear_Slope', np.nan)
        if m == 'Power-Law': return row.get('MI_Whole_PL_b', np.nan)
        return np.nan

    df['MI_Whole_Winner_R2'] = df.apply(_wt_winner_r2, axis=1)
    df = df.loc[
        df['MI_Whole_Best_Model'].notna()
        & (df['MI_Whole_Winner_R2'] >= r2_floor)
    ].copy()

    keep_asp = (df['Condition_Type'] == 'ASP') & (df['Treatment'] == 'WT')
    keep_ep  = (df['Condition_Type'] == 'EP') & df['Fate_Status'].isin(['intact', 'ruptured_post'])
    df = df[keep_asp | keep_ep].copy()

    if df.empty:
        logger.warning("Fate whole-trace MI boxplot skipped: no cells pass filter.")
        return

    df['Bucket']       = df.apply(_fate_box_bucket, axis=1)
    df['Winner_Slope'] = df.apply(_wt_winner_slope, axis=1)

    durations = sorted(
        df.loc[df['Condition_Type'] == 'EP', 'Duration_label'].dropna().unique(),
        key=_duration_sort_key,
    )
    order = _fate_box_order(df['Bucket'].unique().tolist(), durations)

    panels = [
        ('Winner_Slope',
         r'Rate: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law',
         'Whole-trace slope (winner)', False, False),
        ('MI_Whole_PL_a',
         r'Amplitude $a$ (µm)',
         'Whole-trace power-law $a$ (PL winners only)', True, True),
    ]

    _fate_box_tint = {
        'ASP'          : utils.MFA_COLORS['light_grey'],
        'intact'       : utils.MFA_COLORS['pale_blue'],
        'ruptured_post': utils.MFA_COLORS['pale_red'],
    }

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(order) + 6), 3.8))

    for i, (col, ylabel, title, log_y, pl_only) in enumerate(panels):
        ax = axes[i]
        sub = df.loc[df[col].notna(),
                     ['Bucket', col, 'Fate_Status', 'Condition_Type', 'MI_Whole_Best_Model']].copy()
        if pl_only:
            sub = sub[sub['MI_Whole_Best_Model'] == 'Power-Law']
        if sub.empty:
            ax.set_visible(False)
            continue

        box_palette = {}
        for b in order:
            rows = sub[sub['Bucket'] == b]
            if rows.empty:
                continue
            if rows['Condition_Type'].iloc[0] == 'ASP':
                box_palette[b] = _fate_box_tint['ASP']
            else:
                fate = rows['Fate_Status'].iloc[0]
                box_palette[b] = _fate_box_tint.get(fate, utils.MFA_COLORS['light_grey'])

        sns.boxplot(data=sub, x='Bucket', y=col, order=order,
                    ax=ax, showfliers=False,
                    palette=box_palette, hue='Bucket', legend=False)

        for j, bucket in enumerate(order):
            rows = sub[sub['Bucket'] == bucket]
            if rows.empty:
                continue
            rng = np.random.default_rng(42 + j)
            jitter = rng.uniform(-0.15, 0.15, size=len(rows))
            colours = rows['MI_Whole_Best_Model'].map(_MI_MODEL_UTILS_COLOR).fillna('gray')
            ax.scatter(np.full(len(rows), j) + jitter, rows[col],
                       s=22, c=colours,
                       edgecolor='white', linewidths=0.4, alpha=0.85, zorder=3)

        if log_y and (sub[col].dropna() > 0).all():
            ax.set_yscale('log')

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for d in durations:
            b_intact = f"{d}_intact"
            b_rupt   = f"{d}_ruptured"
            if b_intact not in order or b_rupt not in order:
                continue
            v_int = sub.loc[sub['Bucket'] == b_intact, col].dropna().to_numpy()
            v_rup = sub.loc[sub['Bucket'] == b_rupt,   col].dropna().to_numpy()
            if len(v_int) < 2 or len(v_rup) < 2:
                continue
            stars, label, lw, delta = bp._build_stat_label(v_int, v_rup)
            if label is None:
                continue
            x1 = order.index(b_intact)
            x2 = order.index(b_rupt)
            y_top = float(np.nanmax(np.concatenate([v_int, v_rup])))
            bp._add_bracket(ax, x1, x2, y_top, label, lw=lw)

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    handles = [
        mpatches.Patch(facecolor=utils.MFA_COLORS['light_grey'],
                       edgecolor='gray', label='ASP-WT reference'),
        mpatches.Patch(facecolor=utils.MFA_COLORS['pale_blue'],
                       edgecolor='gray', label='EP intact'),
        mpatches.Patch(facecolor=utils.MFA_COLORS['pale_red'],
                       edgecolor='gray', label='EP ruptured'),
        mlines.Line2D([], [], marker='o', linestyle='',
                      markerfacecolor=_MI_MODEL_UTILS_COLOR.get('Linear', 'gray'),
                      markeredgecolor='white',
                      label='Linear winner'),
        mlines.Line2D([], [], marker='o', linestyle='',
                      markerfacecolor=_MI_MODEL_UTILS_COLOR.get('Power-Law', 'gray'),
                      markeredgecolor='white',
                      label='Power-Law winner'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.08), ncol=5,
               frameon=False, fontsize=8)

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_Fate_WholeTrace_MI_Boxplots.pdf")
    plt.close()

def plot_thesis_ep_wholetrace_by_fate(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path, global_whole_dur: Optional[float] = None) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Whole-trace plot skipped: empty mechanics_df.")
        return

    asp_pass = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'ASP')
        & (mechanics_df['Fate_Status'] == 'intact')
    ]
    ep_pass = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'EP')
        & mechanics_df['Fate_Status'].isin(['intact', 'ruptured_post'])
    ]
    asp_pairs = set(zip(asp_pass['Experiment_Folder'], asp_pass['Trap_ID']))
    ep_pairs  = set(zip(ep_pass['Experiment_Folder'],  ep_pass['Trap_ID']))

    traces: Dict[Tuple[str, str], list] = {}
    pulse_times: Dict[str, list] = {}
    ep_durations_present: set = set()

    for gk, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            t_zeroed, L = _extract_whole_trace(trap)
            if t_zeroed is None:
                continue
            if global_whole_dur is not None:
                m = t_zeroed <= global_whole_dur
                t_zeroed = t_zeroed[m]
                L = L[m]
            if len(t_zeroed) < 15:
                continue

            if meta.condition_type == 'ASP' and key in asp_pairs:
                for dl in ep_durations_present.copy():
                    traces.setdefault((dl, 'asp_baseline'), []).append((t_zeroed, L))
                traces.setdefault(('__ALL_ASP__', 'asp_baseline'), []).append((t_zeroed, L))
            elif meta.condition_type == 'EP' and key in ep_pairs:
                fate_row = ep_pass.loc[
                    (ep_pass['Experiment_Folder'] == meta.full_path.name)
                    & (ep_pass['Trap_ID'] == trap.trap_id)
                ]
                if fate_row.empty:
                    continue
                fate = fate_row['Fate_Status'].iloc[0]
                dl   = meta.duration_label
                ep_durations_present.add(dl)
                traces.setdefault((dl, fate), []).append((t_zeroed, L))

                pt = _pulse_time_from_entry(trap)
                if np.isfinite(pt):
                    pulse_times.setdefault(dl, []).append(pt)

    asp_all = traces.get(('__ALL_ASP__', 'asp_baseline'), [])
    for dl in ep_durations_present:
        if (dl, 'asp_baseline') not in traces:
            traces[(dl, 'asp_baseline')] = asp_all
    traces.pop(('__ALL_ASP__', 'asp_baseline'), None)

    durations = sorted(ep_durations_present, key=_duration_sort_key)
    if not durations:
        logger.warning("Whole-trace plot skipped: no EP durations present.")
        return

    n_panels = len(durations)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.4),
                              sharey=True, squeeze=False)
    axes = axes[0]

    def _aggregate(trace_list, t_grid):
        if not trace_list:
            return None, None, 0
        matrix = np.full((len(trace_list), len(t_grid)), np.nan)
        for i, (t_arr, L_arr) in enumerate(trace_list):
            if len(t_arr) < 2:
                continue
            f = interp1d(t_arr, L_arr, bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            matrix[i, :] = f(t_grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m  = np.nanmean(matrix, axis=0)
            sd = np.nanstd(matrix, axis=0)
        return m, sd, len(trace_list)

    for ax, dl in zip(axes, durations):
        max_t = 0.0
        for kind in ('asp_baseline', 'intact', 'ruptured_post'):
            for t, _L in traces.get((dl, kind), []):
                if len(t) and t[-1] > max_t:
                    max_t = float(t[-1])
        if max_t <= 0:
            continue
        common_t = np.linspace(0.0, max_t, 120)

        pts = pulse_times.get(dl, [])
        if pts:
            pts_arr = np.asarray(pts, dtype=float)
            med = float(np.nanmedian(pts_arr))
            q1  = float(np.nanpercentile(pts_arr, 25))
            q3  = float(np.nanpercentile(pts_arr, 75))
            ax.axvspan(q1, q3, color='0.85', alpha=0.4, zorder=0)
            ax.axvline(med, color='0.4', lw=0.9, ls='--', zorder=1)

        for kind, label in (
            ('asp_baseline', 'ASP baseline'),
            ('intact',       'EP intact'),
            ('ruptured_post','EP ruptured'),
        ):
            trace_list = traces.get((dl, kind), [])
            if not trace_list:
                continue
            m, sd, n = _aggregate(trace_list, common_t)
            if m is None:
                continue
            colour = _FATE_COLOR[kind]
            ax.plot(common_t, m, color=colour, lw=1.5,
                    label=f"{label} (n={n})")
            ax.fill_between(common_t, m - sd, m + sd,
                            color=colour, alpha=0.18, linewidth=0)

        ax.set_title(f"{dl} pulse", fontsize=10)
        ax.set_xlabel("Time from aspiration entry (s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    axes[0].set_ylabel(r"Protrusion length $L(t)$ (µm)")
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_EP_WholeTrace_by_Fate.pdf")
    plt.close()

def run_thesis_claim2_plots(grouped_data: Dict,
                             mechanics_df: pd.DataFrame,
                             output_dir: Path,
                             global_whole_dur: Optional[float] = None,
                             whole_trace_dir: Optional[Path] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wt_dir = whole_trace_dir if whole_trace_dir is not None else output_dir
    wt_dir.mkdir(parents=True, exist_ok=True)

    plot_thesis_ep_wholetrace_by_fate(grouped_data, mechanics_df, wt_dir, global_whole_dur)
    plot_thesis_fate_prepulse_visco_boxplots(mechanics_df, output_dir)
    plot_thesis_fate_wholetrace_mi_boxplots(mechanics_df, wt_dir)

def plot_thesis_actin_prepulse_vs_postpulse_paired(mechanics_df: pd.DataFrame, output_dir: Path, r2_floor: float = 0.85) -> None:
    from scipy import stats
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Paired actin plot skipped: empty mechanics_df.")
        return

    df = mechanics_df.copy()
    df = df.loc[
        (df['Condition_Type'] == 'EP')
        & df['Fate_Status'].isin(['intact', 'ruptured_post'])
    ]
    if df.empty:
        logger.warning("Paired actin plot skipped: no EP cells pass filter.")
        return

    durations_present = [d for d in _EP_PULSE_ORDER
                         if d in df['Duration_label'].unique()]
    if not durations_present:
        logger.warning("Paired actin plot skipped: no expected pulse durations "
                       "(100us / 5ms) in EP cohort.")
        return

    fig, axes = plt.subplots(2, len(durations_present),
                              figsize=(4.0 * len(durations_present), 6.4),
                              squeeze=False)

    for row_idx, region in enumerate(('Body', 'Prot')):
        pre_col  = f'Actin_{region}_PrePulse_F0Norm'
        post_col = f'Actin_{region}_PostPulse_F0Norm'
        if pre_col not in df.columns or post_col not in df.columns:
            for c in range(len(durations_present)):
                axes[row_idx, c].set_visible(False)
            continue

        for col_idx, dur in enumerate(durations_present):
            ax = axes[row_idx, col_idx]
            sub_d = df.loc[df['Duration_label'] == dur,
                            ['Treatment', pre_col, post_col]].dropna()
            if sub_d.empty:
                ax.set_visible(False)
                continue

            treatments_present = [t for t in _TREATMENT_ORDER if t in sub_d['Treatment'].values]
            if not treatments_present:
                ax.set_visible(False)
                continue

            positions = {}
            xticklabels = []
            x_idx = 0
            for t in treatments_present:
                positions[t] = (x_idx, x_idx + 1)
                xticklabels.extend([f"{t}\npre", f"{t}\npost"])
                x_idx += 2

            for treatment in treatments_present:
                pts = sub_d.loc[sub_d['Treatment'] == treatment]
                if pts.empty:
                    continue
                x_pre, x_post = positions[treatment]
                colour = _TREATMENT_COLOR[treatment]

                for _, r in pts.iterrows():
                    ax.plot([x_pre, x_post], [r[pre_col], r[post_col]],
                            color='0.75', lw=0.5, alpha=0.7, zorder=1)

                ax.boxplot([pts[pre_col].to_numpy(), pts[post_col].to_numpy()],
                           positions=[x_pre, x_post], widths=0.55,
                           patch_artist=True,
                           boxprops=dict(facecolor='lightgray', alpha=0.4,
                                         edgecolor=colour),
                           medianprops=dict(color=colour, lw=1.4),
                           whiskerprops=dict(color=colour),
                           capprops=dict(color=colour),
                           flierprops=dict(marker='.', markersize=3,
                                            markerfacecolor=colour,
                                            markeredgecolor='none'),
                           showfliers=False)

                rng = np.random.default_rng(42 + col_idx * 10 + row_idx)
                for x_, col_ in ((x_pre, pre_col), (x_post, post_col)):
                    vals = pts[col_].to_numpy()
                    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(np.full(len(vals), x_) + jitter, vals,
                               s=18, color=colour,
                               edgecolor='white', linewidths=0.4,
                               alpha=0.75, zorder=3)

                if len(pts) >= 5:
                    try:
                        w_stat, p_val = stats.wilcoxon(pts[pre_col], pts[post_col])
                        stars, wlabel, lw_, delta = bp._build_stat_label(
                            pts[pre_col].to_numpy(),
                            pts[post_col].to_numpy())
                        p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                        y_top = float(np.nanmax(np.concatenate(
                            [pts[pre_col].to_numpy(),
                             pts[post_col].to_numpy()])))
                        ax.text((x_pre + x_post) / 2, y_top * 1.02,
                                p_str, ha='center', va='bottom',
                                fontsize=7.5, color=colour)
                    except ValueError:
                        pass

            ax.set_xticks(range(len(xticklabels)))
            ax.set_xticklabels(xticklabels, fontsize=8)
            ax.set_xlim(-0.7, len(xticklabels) - 0.3)
            ax.axhline(1.0, color='0.6', lw=0.6, ls='--', zorder=0)
            if col_idx == 0:
                ax.set_ylabel(f"{region}\n" r"Actin $I/F_0$")
            ax.set_title(f"{dur} pulse — {region}", fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_EP_Actin_PrePost_Paired.pdf")
    plt.close()

def plot_thesis_actin_around_pulse_trace(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path, window_pre_s: float = 15.0, window_post_s: float = 15.0) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Actin-around-pulse plot skipped: empty mechanics_df.")
        return

    df = mechanics_df.copy()
    df = df.loc[
        (df['Condition_Type'] == 'EP')
        & df['Fate_Status'].isin(['intact', 'ruptured_post'])
    ]
    accepted = set(zip(df['Experiment_Folder'], df['Trap_ID']))
    if not accepted:
        logger.warning("Actin-around-pulse plot skipped: no EP cells accepted.")
        return

    traces: Dict[Tuple[str, str, str], list] = {}
    pulse_conds_seen: set = set()

    for gk, traps in grouped_data.items():
        if not traps or traps[0].metadata.condition_type != 'EP':
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            if key not in accepted:
                continue
            treatment = meta.treatment
            pulse_cond = f"{meta.voltage}V_{meta.duration_label}"
            if treatment not in _TREATMENT_ORDER:
                continue
            pulse_conds_seen.add(pulse_cond)

            ad = getattr(trap, 'actin_data', {}) or {}
            if 'Time_s' not in ad:
                continue
            t_raw = np.asarray(ad['Time_s'], dtype=float)
            if len(t_raw) == 0:
                continue
            pf = meta.pulse_frame
            if not (0 < pf < len(t_raw)):
                continue
            t_pulse_zero = t_raw - t_raw[pf]

            for region in ('Body', 'Prot'):
                mean_key = f'Actin_{region}_Mean'
                f0_key   = f'F0_{region}'
                if mean_key not in ad or f0_key not in ad:
                    continue
                I  = np.asarray(ad[mean_key], dtype=float)
                F0 = np.asarray(ad[f0_key],   dtype=float)
                f0_scalar = next((float(v) for v in F0
                                  if np.isfinite(v) and v > 0), None)
                if f0_scalar is None:
                    continue
                valid = (np.isfinite(t_pulse_zero) & np.isfinite(I) & (I > 0)
                         & (t_pulse_zero >= -window_pre_s)
                         & (t_pulse_zero <=  window_post_s))
                if valid.sum() < 5:
                    continue
                traces.setdefault((treatment, pulse_cond, region), []).append(
                    (t_pulse_zero[valid], I[valid] / f0_scalar)
                )

    p_conds = sorted(pulse_conds_seen, key=lambda x: _duration_sort_key(x.split('_')[1]))
    if not p_conds:
        logger.warning("Actin-around-pulse plot skipped: no valid EP durations.")
        return

    common_t = np.linspace(-window_pre_s, window_post_s, 90)

    def _agg(trs):
        if not trs:
            return None, None, 0
        stack = []
        for t, y in trs:
            f = interp1d(t, y, bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            stack.append(f(common_t))
        if not stack:
            return None, None, 0
        arr = np.vstack(stack)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), len(trs)

    for pc in p_conds:
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True, squeeze=False)

        any_data_this_dur = False
        for row_idx, region in enumerate(('Body', 'Prot')):
            ax = axes[row_idx, 0]
            ax.axvline(0.0, color=utils.MFA_COLORS.get('pulse', 'gray'),
                       lw=1.0, ls='--', zorder=1)
            ax.axhline(1.0, color='0.7', lw=0.6, ls=':', zorder=0)

            for treatment in _TREATMENT_ORDER:
                mu, sd, n = _agg(traces.get((treatment, pc, region), []))
                if mu is None:
                    continue
                any_data_this_dur = True
                colour = _TREATMENT_COLOR[treatment]
                ax.plot(common_t, mu, color=colour, lw=1.6,
                        label=f"{treatment} (n={n})")
                ax.fill_between(common_t, mu - sd, mu + sd,
                                color=colour, alpha=0.20, linewidth=0)

            ax.set_title(f"{pc} pulse — {region}", fontsize=9)
            if row_idx == 1:
                ax.set_xlabel("Time from pulse (s)")
            ax.set_ylabel(f"{region}\n" r"Actin $I(t)/F_0$")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if not any_data_this_dur:
            plt.close(fig)
            continue

        axes[0, 0].legend(frameon=False, fontsize=8, loc='best')
        plt.tight_layout()

        target_dir = output_dir / pc
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_EP_Actin_Around_Pulse_{pc}.pdf")
        plt.close()

def plot_thesis_protrusion_around_pulse(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path, window_pre_s: float = 15.0, window_post_s: float = 15.0) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Protrusion-around-pulse plot skipped: empty mechanics_df.")
        return

    df = mechanics_df.copy()
    df = df.loc[
        (df['Condition_Type'] == 'EP')
        & df['Fate_Status'].isin(['intact', 'ruptured_post'])
        & df['EP_Post_Pulse_Behavior'].notna()
    ]
    if df.empty:
        logger.warning("Protrusion-around-pulse plot skipped: no EP cells with post-pulse behaviour.")
        return

    behavior_lookup = df.set_index(['Experiment_Folder', 'Trap_ID'])['EP_Post_Pulse_Behavior'].to_dict()

    traces: Dict[Tuple[str, str], list] = {}
    pulse_conds_seen: set = set()

    for gk, traps in grouped_data.items():
        if not traps or traps[0].metadata.condition_type != 'EP':
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            behaviour = behavior_lookup.get(key)
            if behaviour is None:
                continue

            pulse_cond = f"{meta.voltage}V_{meta.duration_label}"
            pulse_conds_seen.add(pulse_cond)

            p_data = getattr(trap, 'protrusion_data', {}) or {}
            if 'Time_s' not in p_data or 'Protrusion_Length_um' not in p_data:
                continue
            t_raw = np.asarray(p_data['Time_s'], dtype=float)
            l_raw = np.asarray(p_data['Protrusion_Length_um'], dtype=float)
            if len(t_raw) == 0:
                continue

            pf = meta.pulse_frame
            if not (0 < pf < len(t_raw)):
                continue
            t_pulse_zero = t_raw - t_raw[pf]

            valid = (np.isfinite(t_pulse_zero) & np.isfinite(l_raw) & (l_raw >= 0)
                     & (t_pulse_zero >= -window_pre_s)
                     & (t_pulse_zero <=  window_post_s))
            if valid.sum() < 5:
                continue
            traces.setdefault((pulse_cond, behaviour), []).append(
                (t_pulse_zero[valid], l_raw[valid])
            )

    p_conds = sorted(pulse_conds_seen, key=lambda x: _duration_sort_key(x.split('_')[1]))
    if not p_conds:
        logger.warning("Protrusion-around-pulse plot skipped: no valid EP durations.")
        return

    common_t = np.linspace(-window_pre_s, window_post_s, 90)

    def _agg(trs):
        if not trs:
            return None, None, 0
        stack = []
        for t, y in trs:
            f = interp1d(t, y, bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            stack.append(f(common_t))
        if not stack:
            return None, None, 0
        arr = np.vstack(stack)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), len(trs)

    behaviour_order  = ['Retracts', 'Stable', 'Extends']
    behaviour_colour = {
        'Retracts': utils.MFA_COLORS.get('dark_blue',  '#2c6fa4'),
        'Stable'  : utils.MFA_COLORS.get('mid_grey',   '#7a7a7a'),
        'Extends' : utils.MFA_COLORS.get('dark_red',   '#a83232'),
    }

    for pc in p_conds:
        behaviours_here = [b for b in behaviour_order if traces.get((pc, b))]
        if not behaviours_here:
            continue

        fig, axes = plt.subplots(1, len(behaviours_here),
                                  figsize=(4.5 * len(behaviours_here), 4.0),
                                  sharey=True, squeeze=False)

        for j, behaviour in enumerate(behaviours_here):
            ax = axes[0, j]
            ax.axvline(0.0, color=utils.MFA_COLORS.get('pulse', 'gray'),
                       lw=1.0, ls='--', zorder=1)

            trs = traces.get((pc, behaviour), [])
            mu, sd, n = _agg(trs)
            if mu is None:
                ax.set_visible(False)
                continue

            colour = behaviour_colour.get(behaviour, 'black')
            alpha_line = 1.0 if n >= 3 else 0.6
            lw_line    = 1.6 if n >= 3 else 1.0

            ax.plot(common_t, mu, color=colour, lw=lw_line, alpha=alpha_line,
                    label=f"n = {n}")
            ax.fill_between(common_t, mu - sd, mu + sd,
                            color=colour, alpha=0.20, linewidth=0)

            ax.set_title(f"{pc} pulse — {behaviour} (n = {n})", fontsize=10)
            ax.set_xlabel("Time from pulse (s)")
            if j == 0:
                ax.set_ylabel(r"Protrusion length $L(t)$ (µm)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        
        target_dir = output_dir / pc
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_EP_Protrusion_Around_Pulse_{pc}.pdf")
        plt.close()

def plot_thesis_mean_uptake_trace_wt(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path, min_frac_contributing: float = 0.25) -> None:
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Mean uptake trace skipped: empty mechanics_df.")
        return

    ok = mechanics_df.loc[
        (mechanics_df['Treatment'] == 'WT')
        & (
            ((mechanics_df['Condition_Type'] == 'ASP')
             & (mechanics_df['Fate_Status'] == 'intact'))
            | ((mechanics_df['Condition_Type'] == 'EP')
               & mechanics_df['Fate_Status'].isin(['intact', 'ruptured_post']))
        )
    ]
    accepted = set(zip(ok['Experiment_Folder'], ok['Trap_ID']))
    if not accepted:
        logger.warning("Mean uptake trace skipped: no WT cells accepted.")
        return

    conditions = ('ASP', 'EP_100V_100us', 'EP_100V_5ms')
    per_cond_traces: Dict[str, Dict[str, list]] = {
        c: {'Body': [], 'Prot': [], 'Total': []} for c in conditions
    }
    per_cond_pulse: Dict[str, list] = {c: [] for c in conditions}
    max_t_seen = 0.0

    region_col = {
        'Body' : 'Body_VolNorm',
        'Prot' : 'Protrusion_VolNorm',
        'Total': 'Total_VolNorm',
    }

    for gk, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            meta = trap.metadata
            if meta.treatment != 'WT':
                continue
            key = (meta.full_path.name, trap.trap_id)
            if key not in accepted:
                continue

            if meta.condition_type == 'ASP':
                cond_key = 'ASP'
            elif meta.condition_type == 'EP':
                cond_key = f"EP_{meta.voltage}V_{meta.duration_label}"
                if cond_key not in conditions:
                    # Dynamically allow it or continue
                    continue
            else:
                continue

            ud = getattr(trap, 'uptake_data', {}) or {}
            if 'Time_s' not in ud:
                continue
            t_raw = np.asarray(ud['Time_s'], dtype=float)
            if len(t_raw) < 5:
                continue
            t_zero = t_raw - t_raw[0]

            for region, col in region_col.items():
                if col not in ud:
                    continue
                y = np.asarray(ud[col], dtype=float)
                if len(y) != len(t_zero):
                    continue
                valid = np.isfinite(t_zero) & np.isfinite(y)
                if valid.sum() < 5:
                    continue
                per_cond_traces[cond_key][region].append(
                    (t_zero[valid], y[valid])
                )
                if t_zero[valid][-1] > max_t_seen:
                    max_t_seen = float(t_zero[valid][-1])

            if cond_key.startswith('EP_'):
                pf = meta.pulse_frame
                if 0 < pf < len(t_raw):
                    per_cond_pulse[cond_key].append(float(t_raw[pf] - t_raw[0]))

    if max_t_seen <= 0:
        return

    common_t = np.linspace(0.0, max_t_seen, 120)

    def _aggregate(traces_list):
        if not traces_list:
            return None, None, 0
        matrix = np.full((len(traces_list), len(common_t)), np.nan)
        for i, (t, y) in enumerate(traces_list):
            if len(t) < 2:
                continue
            f = interp1d(t, y, bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            matrix[i, :] = f(common_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_y = np.nanmean(matrix, axis=0)
            sd_y   = np.nanstd(matrix, axis=0)
            n_t    = np.sum(np.isfinite(matrix), axis=0)
        n_start = len(traces_list)
        min_n = max(1, int(np.ceil(min_frac_contributing * n_start)))
        mask = np.isfinite(mean_y) & np.isfinite(sd_y) & (n_t >= min_n)
        if not np.any(mask):
            return None, None, n_start
        return mean_y[mask], sd_y[mask], n_start

    cond_colour = {
        'ASP'          : utils.MFA_COLORS['light_grey'],
        'EP_100V_100us': utils.MFA_COLORS['medium_blue'],
        'EP_100V_5ms'  : utils.MFA_COLORS['dark_blue'],
    }
    cond_edge = {
        'ASP'          : '0.4',
        'EP_100V_100us': utils.MFA_COLORS['medium_blue'],
        'EP_100V_5ms'  : utils.MFA_COLORS['dark_blue'],
    }
    cond_label = {
        'ASP'          : 'ASP (no pulse)',
        'EP_100V_100us': 'EP 100V 100 µs',
        'EP_100V_5ms'  : 'EP 100V 5 ms',
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=True)

    for ax, region in zip(axes, ('Body', 'Prot', 'Total')):
        for cond_key in conditions:
            trs = per_cond_traces[cond_key][region]
            if not trs:
                continue
            mean_y, sd_y, n_start = _aggregate(trs)
            if mean_y is None:
                continue
            matrix = np.full((n_start, len(common_t)), np.nan)
            for i, (t, y) in enumerate(trs):
                if len(t) < 2:
                    continue
                f = interp1d(t, y, bounds_error=False, fill_value=np.nan,
                             assume_sorted=True)
                matrix[i, :] = f(common_t)
            n_t = np.sum(np.isfinite(matrix), axis=0)
            min_n = max(1, int(np.ceil(min_frac_contributing * n_start)))
            mask = np.isfinite(np.nanmean(matrix, axis=0)) & (n_t >= min_n)
            t_plot = common_t[mask]

            ax.plot(t_plot, mean_y,
                    color=cond_edge.get(cond_key, 'k'), lw=1.6,
                    label=f"{cond_label.get(cond_key, cond_key)} (n={n_start})")
            ax.fill_between(t_plot, mean_y - sd_y, mean_y + sd_y,
                            color=cond_colour.get(cond_key, 'k'), alpha=0.25,
                            linewidth=0)

            if cond_key.startswith('EP_'):
                pts = per_cond_pulse[cond_key]
                if pts:
                    med = float(np.median(pts))
                    q1  = float(np.percentile(pts, 25))
                    q3  = float(np.percentile(pts, 75))
                    ax.axvspan(q1, q3, color=cond_edge.get(cond_key, 'k'),
                               alpha=0.08, zorder=0)
                    ax.axvline(med, color=cond_edge.get(cond_key, 'k'),
                               ls='--', lw=0.9, alpha=0.7, zorder=1)

        ax.set_title(region, fontsize=10)
        ax.set_xlabel("Time from aspiration onset (s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(r"$I(t)$  (ADU/µm³)")
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')

    plt.tight_layout()
    target_dir = output_dir / "combined"
    target_dir.mkdir(parents=True, exist_ok=True)
    utils.save_plot_pdf(target_dir / "Thesis_WT_MeanUptakeTrace.pdf")
    plt.close()

def run_thesis_claim2_uptake_actin_plots(grouped_data: Dict, mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Claim 2 uptake/actin plot '{label}' failed: {e}", exc_info=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    _try("Mean WT uptake trace",
         plot_thesis_mean_uptake_trace_wt, grouped_data, mechanics_df, output_dir)
    _try("Paired actin pre/post",
         plot_thesis_actin_prepulse_vs_postpulse_paired, mechanics_df, output_dir)
    _try("Actin around pulse trace",
         plot_thesis_actin_around_pulse_trace, grouped_data, mechanics_df, output_dir)
    _try("Protrusion around pulse trace",
         plot_thesis_protrusion_around_pulse, grouped_data, mechanics_df, output_dir)