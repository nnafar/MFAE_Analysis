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
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from scipy.interpolate import interp1d

import bulk_file_handling as bfh
import bulk_mechanics as bm
import bulk_plotting as bp
import bulk_utils as utils
from thesis_plotting_actin_by_fate import register_by_fate_actin_plots
from thesis_plotting_uptake import (
    apply_runaway_gate,
    REGION_PREFIX,
    DEFAULT_RUNAWAY_RULE,
    ANALYSIS_FATE_STATES,
)

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


def _mi_winner_slope(row: pd.Series) -> float:
    """
    'Slope-analogue' for the combined slope panel: each cell contributes
    exactly one value drawn from its winning MI model.

        MI_Best_Model == 'Linear'    -> Linear_Slope  (µm/s)
        MI_Best_Model == 'Power-Law' -> PL_b          (dimensionless exponent)

    Both are read as a 'rate of protrusion length change', but note that
    the units differ. The panel keeps them on the same axis, colour-coded
    by model, so a reader can tell which cells contributed which quantity.
    """
    m = row.get('MI_Best_Model')
    if m == 'Linear':
        return row.get('Linear_Slope', np.nan)
    if m == 'Power-Law':
        return row.get('PL_b', np.nan)
    return np.nan


def _mi_whole_winner_slope(row: pd.Series) -> float:
    """Same 'slope-analogue' idea but for the whole-trace MI fits."""
    m = row.get('MI_Whole_Best_Model')
    if m == 'Linear':
        return row.get('MI_Whole_Linear_Slope', np.nan)
    if m == 'Power-Law':
        return row.get('MI_Whole_PL_b', np.nan)
    return np.nan

def plot_asp_protrusion_dynamics(grouped_data: Dict,
                                 output_dir: Path,
                                 global_asp_dur: Optional[float] = None,
                                 output_dir_map: Optional[Dict[str, Path]] = None) -> None:
    """
    Per-condition trap boxplots go to ``output_dir_map['WT'|'CytD']`` when
    the map is supplied; the cross-treatment overlay always lands in
    ``output_dir`` (expected to be ``combined/``).
    """
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
            trap_target_dir = _resolve_treatment_dir(cond_label, output_dir, output_dir_map)
            utils.save_plot_pdf(trap_target_dir / f"Thesis_Boxplot_{cond_label}.pdf")
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
        # Reference styling: matches plot_thesis_combined_prepulse_trace_by_fate.
        bp.apply_thesis_rcparams()
        # Drawn at printed size for a 0.49\\textwidth subfigure. The 1.60 in
        # height sets the top of the left-hand column of Figure 1.6; panel C
        # is height-matched to it plus panel B.
        fig, ax = plt.subplots(
            figsize=utils.canvas_size_for(0.49, height_in=1.60))

        _stripped = {k: k.replace("_ASP", "") for k in combined_lt_data.keys()}
        _legend_map = bp._reduce_labels(sorted(_stripped.values()))

        for cond_label, data in combined_lt_data.items():
            c_t = data['t']
            c_mean = data['mean']
            c_sd = data['sd']

            # Determine treatment from the condition label. ASP-only figure
            # compares WT vs CytD, so we use the ASP-treatment palette
            # rather than the pulse-condition palette.
            cell_type = 'CytD' if 'CytD' in cond_label else 'WT'
            colour = bp.ASP_TREATMENT_COLOUR.get(
                cell_type, utils.get_style_color(cell_type, 'ASP'))

            legend_label = _legend_map.get(_stripped[cond_label],
                                            _stripped[cond_label])

            # Both cohorts are "baseline" ASP data (no pulse), so fate is
            # 'baseline' — solid line, filled marker keyed to treatment.
            lkw = bp.build_trace_line_kwargs(pulse_colour=colour,
                                              fate='baseline',
                                              treatment=cell_type,
                                              is_baseline=False)
            ax.plot(c_t, c_mean,
                    markevery=bp.markevery_from(c_t),
                    label=legend_label, **lkw)
            ax.fill_between(c_t, c_mean - c_sd, c_mean + c_sd,
                            color=colour, alpha=bp.TRACE_BAND_ALPHA,
                            linewidth=0)

        ax.set_xlabel("Time from aspiration onset (s)")
        ax.set_ylabel(r"Protrusion length $L(t)$ (µm)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='lower right')

        if global_asp_dur is not None:
            ax.set_xlim(0, global_asp_dur)

        plt.tight_layout()
        utils.save_plot_pdf(
            output_dir / "Thesis_Combined_Protrusion_Dynamics_ASP.pdf")
        plt.close()

def plot_thesis_asp_best_fit_multipanel(all_grouped_data: Dict,
                                         output_dir: Path,
                                         r_eff: float,
                                         C: float = 1.0,
                                         global_asp_dur: Optional[float] = None,
                                         output_dir_map: Optional[Dict[str, Path]] = None,
                                         # --- New Font Size Parameters (Defaulted to 14) ---
                                         legend_size: int      = 19,
                                         axis_label_size: int  = 19,
                                         panel_title_size: int = 19,
                                         annotation_size: int  = 19,
                                         tick_label_size: int  = 19,
                                         ) -> None:
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
        
        # Legend font size set to parameter (14)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=legend_size)
        dur_note = f"  [window: {global_asp_dur:.0f} s]" if global_asp_dur else ""
        
        # Figure-level axis labels set to parameter (14)
        fig.text(0.5, 0.01, f'Time from entry (s){dur_note}', ha='center', fontsize=axis_label_size)
        fig.text(0.01, 0.5, 'Protrusion Length (µm)', va='center', rotation='vertical', fontsize=axis_label_size)

        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            
            # Subpanel axis tick mark labels set to parameter (14)
            ax.tick_params(axis='both', which='both', labelsize=tick_label_size)
            
            # Subpanel title set to parameter (14)
            ax.set_title(f"Trap {tid}", fontweight='bold', fontsize=panel_title_size)

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

                ax.plot(t_zeroed, l_clean, 'o', c='black', ms=2, alpha=0.55)

                pressure = trap.metadata.pressure
                visco = bm.fit_viscoelastic(
                    t_zeroed, l_clean, r_eff, pressure, C,
                    n_starts=3, min_points=15)

                best = visco['best_model']
                if best is None:
                    # Failed fit text set to parameter (14)
                    ax.text(0.05, 0.9, "Fit failed", transform=ax.transAxes, fontsize=annotation_size, color='red')
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

                # Fit line is semi-transparent so the raw scatter beneath
                # remains visible through the trace (data is the ground
                # truth; the fit is one interpretation).
                ax.plot(t_smooth, l_pred, color=color, lw=2, alpha=0.55)
                r2_str = f"R²={visco['best_r2']:.2f}" if visco['best_r2'] is not None else ""
                
                # Model + R² annotation set to parameter (14)
                ax.text(0.05, 0.9, f"{best}  {r2_str}", transform=ax.transAxes, fontsize=annotation_size, fontweight='bold', color=color)

        for j in range(n, len(axes)): axes[j].axis('off')
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        target_dir = _resolve_treatment_dir(cat_label, output_dir, output_dir_map)
        utils.save_plot_pdf(target_dir / f"Thesis_ASP_BestFit_Panel_{cat_label_short}.pdf")
        plt.close()

def plot_thesis_asp_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: ASP Viscoelastic Parameters...")
    bp.plot_asp_parameter_boxplots(mechanics_df, output_dir, title_fontsize = 10,
    label_fontsize =   10,
    tick_fontsize  =   10,
    legend_fontsize =  10,
    bracket_fontsize = 10)

def plot_thesis_asp_actin_f0_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: ASP Actin F0 Boxplots...")
    bp.plot_asp_actin_f0_boxplots(mechanics_df, output_dir)

def plot_thesis_prepulse_visco_parameter_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating Thesis Plot: Pre-Pulse Viscoelastic Parameters...")
    bp.plot_prepulse_visco_parameter_boxplots(mechanics_df, output_dir,
                                              title_fontsize  = 25,
                                              label_fontsize  = 25,
                                              tick_fontsize   = 25,
                                              legend_fontsize = 25)

def run_thesis_plots(grouped_data: Dict,
                     mechanics_df: pd.DataFrame,
                     output_dir: Path,
                     r_eff: float,
                     global_asp_dur: Optional[float] = None,
                     output_dir_map: Optional[Dict[str, Path]] = None):
    """
    ASP-only plot suite (Claim 1).

    ``output_dir`` is expected to be the ``combined/`` subfolder — every
    cross-treatment figure (protrusion overlay, parameter boxplots, actin
    F0 boxplot) is written there. ``output_dir_map`` routes per-treatment
    files (per-condition trap boxplots, per-condition best-fit panels)
    into ``WT/`` and ``CytD/`` subfolders.

    NOTE: ``plot_thesis_prepulse_visco_parameter_boxplots`` used to be
    called here as well. It has been removed because it belongs to
    claim 2 (pre-pulse combined) and is emitted there by
    ``run_thesis_mi_prepulse_plots``.
    """
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Thesis plot '{label}' failed: {e}", exc_info=False)

    _try("ASP Protrusion Dynamics",
         plot_asp_protrusion_dynamics, grouped_data, output_dir,
         global_asp_dur, output_dir_map)
    _try("ASP Best-Fit Multipanel",
         plot_thesis_asp_best_fit_multipanel, grouped_data, output_dir,
         r_eff, 1.0, global_asp_dur, output_dir_map)
    _try("ASP Parameter Boxplots",
         plot_thesis_asp_parameter_boxplots, mechanics_df, output_dir)
    _try("ASP Actin F0 Boxplots",
         plot_thesis_asp_actin_f0_boxplots, mechanics_df, output_dir)

# =========================================================================
# GRAMMAR-BACKED COLOUR CONSTANTS
# =========================================================================
# The three legacy constants below (MI_ASP_COLOR, MI_EP_COLOR_RAMP,
# _TREATMENT_COLOR, _FATE_COLOR) still exist so downstream call sites that
# read them directly keep working. Their hex values now come from the
# canonical (cell_type, protocol) grammar defined in Utils_MFA so a single
# edit to the grammar dict propagates everywhere.
#
# Any NEW plotting code should call utils.get_style_color(cell, protocol),
# utils.get_scatter_kwargs(...), utils.get_line_kwargs(...), or
# utils.get_bar_kwargs(...) directly rather than reading these constants.
# =========================================================================

# MI_ASP_COLOR is used for pooled ASP overlays where the cohort has no
# single cell_type. It maps to WT-ASP because WT is the canonical control.
# For plots that DO know cell_type, use utils.get_style_color(cell, 'ASP').
MI_ASP_COLOR = utils.get_style_color('WT', 'ASP') if hasattr(utils, 'get_style_color') else '#1a243d'

# MI_EP_COLOR_RAMP is used where EP cohorts are pooled across cell types and
# only ranked by ordinal EP index. Ordered darkest -> lightest to preserve
# the historical semantics ("dark = strongest / first EP condition"). For
# plots that know the (cell_type, protocol) pair, use the grammar helper.
MI_EP_COLOR_RAMP = (
    [utils.MFA_COLORS[k] for k in ('dark_red', 'medium_red', 'light_red', 'pale_red')
     if k in utils.MFA_COLORS]
    if hasattr(utils, 'MFA_COLORS') else
    ['#b31529', '#d75f4c', '#f6a482', '#fddbc7']
)

# Model palette drawn from Utils_MFA (Linear vs Power-Law). This is
# ORTHOGONAL to the condition grammar (it colours which model fit won a
# comparison, not which condition the data came from), so it stays as-is.
_MI_MODEL_UTILS_COLOR = {
    'Linear'   : utils.MFA_COLORS['medium_red'],
    'Power-Law': utils.MFA_COLORS['medium_blue'],
}

def _condition_bucket(meta: bfh.ExperimentMetadata) -> str:
    if meta.condition_type == "ASP":
        return "ASP"
    return f"EP-pre {meta.voltage}V {meta.duration_label}"

def _condition_bucket_fate(meta: bfh.ExperimentMetadata, fate_status: str) -> str:
    """
    Fate-aware variant of `_condition_bucket`. ASP experiments have no
    pulse and are always returned as 'ASP'. EP experiments are split by
    fate into intact and ruptured-post cohorts so that pre-pulse
    mechanics can be compared between the two outcomes.
    """
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
    """
    Fate-aware variant of `_row_bucket`. Reads `Fate_Status` from the
    mechanics dataframe row and appends '(intact)' or '(ruptured)' to
    EP-pre labels. ASP rows are returned as 'ASP' regardless of fate,
    since there is no pulse under aspiration alone.
    """
    if row.get('Condition_Type') == 'ASP':
        return "ASP"
    base = f"EP-pre {row.get('Voltage_V')}V {row.get('Duration_label')}"
    if row.get('Fate_Status') == "ruptured_post":
        return f"{base} (ruptured)"
    return f"{base} (intact)"


def _parse_bucket_to_style(bucket_label: str) -> Tuple[str, str]:
    """
    Parse a bucket label produced by `_condition_bucket_fate` /
    `_row_bucket_fate` / `_condition_bucket` / `_row_bucket` into the
    (protocol, fate) pair used by the Utils_MFA style grammar.

    Recognised inputs and their parsings:

        'ASP'                                  -> ('ASP',   'intact')
        'EP-pre 100V 100us'                    -> ('100us', 'intact')
        'EP-pre 100V 100us (intact)'           -> ('100us', 'intact')
        'EP-pre 100V 100us (ruptured)'         -> ('100us', 'ruptured_post')
        'EP-pre 100V 5ms'                      -> ('5ms',   'intact')
        'EP-pre 100V 5ms (intact)'             -> ('5ms',   'intact')
        'EP-pre 100V 5ms (ruptured)'           -> ('5ms',   'ruptured_post')
        'EP-whole 100V 100us'                  -> ('100us', 'intact')
        'EP-whole 100V 100us (intact)'         -> ('100us', 'intact')
        'EP-whole 100V 100us (ruptured)'       -> ('100us', 'ruptured_post')
        'EP-whole 100V 5ms (intact)'           -> ('5ms',   'intact')

    Anything unrecognised returns ('ASP', 'intact') and logs a warning
    so a mistyped bucket surfaces as a plot in the WT-ASP colour rather
    than a crash.
    """
    if bucket_label == 'ASP':
        return ('ASP', 'intact')
    label = str(bucket_label).lower()
    # Fate
    fate = 'ruptured_post' if '(ruptured)' in label else 'intact'
    # Protocol
    if '100us' in label:
        protocol = '100us'
    elif '5ms' in label:
        protocol = '5ms'
    else:
        logger.warning("_parse_bucket_to_style: could not identify protocol "
                       "in %r; defaulting to 'ASP'.", bucket_label)
        return ('ASP', 'intact')
    return (protocol, fate)


def _cell_type_from_grouped_data(grouped_data: Dict) -> str:
    """
    Best-effort inference of cell type from a `grouped_data` payload.
    """
    for key in grouped_data.keys(): # <-- Added .keys() to prevent KeyError: 0
        traps = grouped_data[key]
        if not traps:
            continue
        meta = traps[0].metadata
        
        # Prioritize 'treatment' which holds 'WT' or 'CytD' in this dataset
        ct = getattr(meta, 'treatment', None)
        if ct in ('WT', 'CytD'):
            return ct
            
        # Fallback to cell_type just in case
        ct_alt = getattr(meta, 'cell_type', None)
        if ct_alt in ('WT', 'CytD'):
            return ct_alt
            
    logger.warning("_cell_type_from_grouped_data: could not infer treatment; "
                   "defaulting to 'WT'.")
    return 'WT'

def _safe_filename(label: str) -> str:
    return label.replace(' ', '_')


# ---------------------------------------------------------------------------
# Output-folder routing helpers
# ---------------------------------------------------------------------------
# Plot functions that emit one file per bucket (per-condition MI boxplots,
# per-duration fate pies, per-treatment ASP panels, ...) accept an optional
# ``output_dir_map`` dict keyed by a small folder-key token so the caller
# can send each file into the right subfolder. When the map is absent or
# the token is missing, the plot falls back to the default directory so
# nothing is silently lost.


def _bucket_to_folder_key(bucket_label: str) -> str:
    """
    Map a plot bucket label (e.g. 'ASP', 'EP-pre 100V 100us (intact)') onto a
    folder key used by ``output_dir_map``. Returns one of 'ASP', '100us', '5ms',
    '10ms', '100ms', '1ms', or '' (unknown -> caller falls back to combined).
    """
    if bucket_label == 'ASP':
        return 'ASP'
    for tok in ('100us', '5ms', '10ms', '100ms', '1ms'):
        if tok in bucket_label:
            return tok
    return ''


def _resolve_output_dir(bucket_label: str,
                        default_dir: Path,
                        output_dir_map: Optional[Dict[str, Path]] = None) -> Path:
    """Route a single-condition plot into the right subfolder."""
    if output_dir_map is None:
        return default_dir
    key = _bucket_to_folder_key(bucket_label)
    return output_dir_map.get(key, default_dir)


def _treatment_to_folder_key(cond_or_treatment: str) -> str:
    """
    Extract the treatment token ('WT' or 'CytD') from a condition label or a
    bare treatment string. Used to route claim-1 per-treatment plots into
    the WT/ or CytD/ subfolder.
    """
    s = str(cond_or_treatment)
    # CytD must be tested before WT because a "WT" substring could also
    # occur in labels that mention CytD (defensive; unlikely in practice).
    if 'CytD' in s:
        return 'CytD'
    if 'WT' in s:
        return 'WT'
    return ''


def _resolve_treatment_dir(cond_or_treatment: str,
                           default_dir: Path,
                           output_dir_map: Optional[Dict[str, Path]] = None) -> Path:
    if output_dir_map is None:
        return default_dir
    key = _treatment_to_folder_key(cond_or_treatment)
    return output_dir_map.get(key, default_dir)

def plot_mi_protrusion_dynamics(grouped_data: Dict,
                                output_dir: Path,
                                global_pre_dur: Optional[float] = None,
                                output_dir_map: Optional[Dict[str, Path]] = None,
                                mechanics_df: Optional[pd.DataFrame] = None) -> None:
    """
    Per-bucket ``Thesis_MI_Boxplot_<bucket>.pdf`` files are routed via
    ``output_dir_map`` when supplied ('ASP', '100us', '5ms'). The
    cross-bucket overlay ``Thesis_Combined_Protrusion_Dynamics_MI.pdf``
    always lands in ``output_dir`` (the ``combined/`` subfolder).

    When ``mechanics_df`` is supplied, only cells that pass the MI
    whole-trace R^2 gate (``MI_Whole_R2_Flag == True``) contribute to the
    aggregated overlay, so the figure shares its cohort with the MI
    parameter and MI trace figures. When it is not supplied (legacy
    callers), no such filter is applied.
    """
    logger.info("Generating Thesis Plot: MI Protrusion Dynamics (ASP vs EP-pre)...")

    # MI cohort membership set (folder, trap_id). Empty means no filter.
    mi_pairs: set = set()
    if mechanics_df is not None and not mechanics_df.empty and \
            'MI_Whole_R2_Flag' in mechanics_df.columns:
        mi_pass = mechanics_df.loc[mechanics_df['MI_Whole_R2_Flag'] == True]
        mi_pairs = set(zip(mi_pass['Experiment_Folder'], mi_pass['Trap_ID']))

    bucketed_traces: Dict[str, list] = {}
    bucketed_max_records: Dict[str, list] = {}

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps:
            continue

        meta = traps[0].metadata

        for trap in sorted(traps, key=lambda t: t.trap_id):
            # MI cohort gate — skip cells that didn't pass MI R^2 when a
            # mechanics_df was passed in.
            if mi_pairs:
                cell_key = (meta.full_path.name, trap.trap_id)
                if cell_key not in mi_pairs:
                    continue

            bucket = _condition_bucket_fate(meta, trap.fate_status)

            # Frame-index extraction (mirrors
            # plot_thesis_combined_prepulse_trace_by_fate). Using
            # _extract_asp_phase_trace here rejects 100 µs cells because
            # their pre-pulse window rarely reaches 15 frames when the
            # sum(pre_mask)<15 guard is applied *after* the >0 mask; the
            # frame-index approach keeps them.
            p_data = trap.protrusion_data
            if ('Time_s' not in p_data
                    or 'Protrusion_Length_um' not in p_data):
                continue
            times = np.asarray(p_data['Time_s'], dtype=float)
            lens  = np.asarray(p_data['Protrusion_Length_um'], dtype=float)
            valid = np.isfinite(times) & np.isfinite(lens)
            times = times[valid]
            lens  = lens[valid]
            if len(times) < 5:
                continue

            ctype = meta.condition_type
            is_post = bool(getattr(trap, 'post_pulse_entry', False))
            if ctype == 'ASP' or is_post:
                t_zeroed = times - times[0]
                l = lens
            elif ctype == 'EP':
                pf = meta.pulse_frame
                if not (0 < pf < len(times)):
                    continue
                t_all = times - times[0]
                pre_mask = (np.arange(len(times)) < pf) & (lens > 0)
                t_zeroed = t_all[pre_mask]
                l = lens[pre_mask]
                if len(t_zeroed) == 0:
                    continue
                # Re-zero to the first accepted pre-pulse frame so all
                # cohorts share the same time origin.
                t_zeroed = t_zeroed - t_zeroed[0]
            else:
                continue

            if global_pre_dur is not None:
                win_mask = t_zeroed <= global_pre_dur
                t_zeroed = t_zeroed[win_mask]
                l = l[win_mask]

            # Lowered from 15 to 5 to keep short 100 µs pre-pulse windows.
            # Cohort cell counts still surface via the legend n= labels.
            if len(t_zeroed) < 5:
                continue

            bucketed_traces.setdefault(bucket, []).append((t_zeroed, l))

            max_val = np.max(l) if len(l) else np.nan
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
        target_dir = _resolve_output_dir(bucket, output_dir, output_dir_map)
        utils.save_plot_pdf(
            target_dir / f"Thesis_MI_Boxplot_{_safe_filename(bucket)}.pdf"
        )
        plt.close()

    if not bucketed_traces:
        return

    ep_buckets = sorted(k for k in bucketed_traces.keys() if k != "ASP")
    ordered_buckets = (["ASP"] if "ASP" in bucketed_traces else []) + ep_buckets

    # Infer cell_type from the grouped_data cohort so the marker follows
    # the (Treatment) axis. Colour follows (protocol) via the bucket parser
    # routed through bp.PULSE_TRACE_PALETTE; fate drives linestyle.
    cell_type_for_plot = _cell_type_from_grouped_data(grouped_data)

    # Reference styling: matches plot_thesis_combined_prepulse_trace_by_fate.
    bp.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    global_max_t = 0.0

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
                f = interp1d(t_arr, l_arr, bounds_error=False,
                              fill_value=np.nan)
                matrix[i, :] = f(t_grid)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_L = np.nanmean(matrix, axis=0)
            sd_L   = np.nanstd(matrix, axis=0)
            n_t    = np.sum(np.isfinite(matrix), axis=0)

        MIN_FRAC_CONTRIBUTING = 0.25
        n_start = len(traces)
        min_n_required = max(1, int(np.ceil(MIN_FRAC_CONTRIBUTING * n_start)))
        plot_mask = (np.isfinite(mean_L) & np.isfinite(sd_L)
                      & (n_t >= min_n_required))

        if not np.any(plot_mask):
            continue

        c_t    = t_grid[plot_mask]
        c_mean = mean_L[plot_mask]
        c_sd   = sd_L[plot_mask]

        # Colour routing follows the reference: NO_PULSE for ASP,
        # PULSE_TRACE_PALETTE for EP, fate drives linestyle.
        protocol, fate = _parse_bucket_to_style(bucket)
        if bucket == 'ASP':
            colour = bp.NO_PULSE_COLOUR
            fate_key = 'baseline'
        else:
            colour = bp.PULSE_TRACE_PALETTE.get(('EP', protocol),
                                                 bp.PULSE_100US_COLOUR)
            fate_key = fate
        is_baseline = (bucket == 'ASP') and (len(ep_buckets) > 0)

        lkw = bp.build_trace_line_kwargs(pulse_colour=colour,
                                          fate=fate_key,
                                          treatment=cell_type_for_plot,
                                          is_baseline=is_baseline)
        ax.plot(c_t, c_mean, markevery=bp.markevery_from(c_t),
                label=f'{bucket} (n={n_start})', **lkw)
        ax.fill_between(c_t, c_mean - c_sd, c_mean + c_sd,
                        color=colour, alpha=bp.TRACE_BAND_ALPHA,
                        linewidth=0)

        if c_t[-1] > global_max_t:
            global_max_t = float(c_t[-1])

    ax.set_xlabel("Time from aspiration onset (s)")
    ax.set_ylabel(r"Protrusion length $L(t)$ (µm)")
    ax.set_title("MI protrusion mechanics (pre-pulse window)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower right')

    if global_pre_dur is not None:
        ax.set_xlim(0, global_pre_dur)
    elif global_max_t > 0:
        ax.set_xlim(0, global_max_t)

    plt.tight_layout()
    utils.save_plot_pdf(
        output_dir / "Thesis_Combined_Protrusion_Dynamics_MI.pdf")
    plt.close()

def plot_thesis_mi_best_fit_multipanel(grouped_data: Dict, output_dir: Path, global_pre_dur: Optional[float] = None) -> None:
    logger.info("Generating Thesis Plot: MI Best-Fit Multipanel...")
    bp.plot_model_independent_fits_multipanel(grouped_data, output_dir, global_pre_dur)

def plot_thesis_mi_parameter_boxplots(mechanics_df: pd.DataFrame,
                                       output_dir: Path,
                                       r2_floor: float = 0.85) -> None:
    """
    Two-panel MI parameter boxplot.

    Panel 1 ('Slope'):
        Every cell contributes exactly one value — Linear_Slope if the AICc
        winner was Linear, PL_b if the winner was Power-Law — coloured by
        the winning model so mixed-model buckets remain visually decodable.
        Units differ (µm/s vs dimensionless), so the y-axis is labelled as
        a generic 'characteristic rate' rather than a single unit.

    Panel 2 ('Power-Law a'):
        Amplitude a in µm, restricted to Power-Law winners only — Linear
        winners have no meaningful a and are dropped from this panel.

    Colours come from `_MI_MODEL_UTILS_COLOR`, which draws from
    `bulk_utils.MFA_COLORS`.
    """
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

    # ---- Panel 1: combined slope-analogue -----------------------------------
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

    # ---- Panel 2: PL_a alone (Power-Law winners only) -----------------------
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
    utils.save_plot_pdf(output_dir / "Thesis_MI_Parameter_Boxplots.pdf")
    plt.close()

def plot_thesis_mi_per_trap_parameters(mechanics_df: pd.DataFrame,
                                        output_dir: Path,
                                        r2_floor: float = 0.85) -> None:
    """
    Per-trap MI parameter breakdown, same 2-panel scheme as
    `plot_thesis_mi_parameter_boxplots`:
        Panel 1: winner-model slope (Linear_Slope or PL_b), coloured by model.
        Panel 2: PL_a for Power-Law winners only.

    One figure per bucket (ASP / EP-pre condition).
    """
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

        # ---- Panel 1: combined slope -------------------------------------
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

        # ---- Panel 2: PL_a alone -----------------------------------------
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
        utils.save_plot_pdf(
            output_dir / f"Thesis_MI_PerTrap_Params_{_safe_filename(bucket)}.pdf"
        )
        plt.close()

def run_thesis_mi_prepulse_plots(mi_grouped_data: Dict,
                                 mechanics_df: pd.DataFrame,
                                 output_dir: Path,
                                 global_pre_dur: Optional[float] = None,
                                 output_dir_map: Optional[Dict[str, Path]] = None) -> None:
    """
    Pre-pulse plot suite. Uses viscoelastic fits on the matched-window
    pre-pulse cohort (PrePulse_* columns).

    ``output_dir_map`` routes per-bucket ``Thesis_MI_Boxplot_*.pdf``
    files into per-condition subfolders (``ASP/``, ``100V_100us/``,
    ``100V_5ms/``). Cross-cohort figures (protrusion overlay, pre-pulse
    parameter boxplots, fate pre-pulse boxplots) stay in ``output_dir``.
    """
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Pre-pulse thesis plot '{label}' failed: {e}", exc_info=False)

    _try("Pre-Pulse Protrusion Dynamics",
         plot_mi_protrusion_dynamics, mi_grouped_data, output_dir,
         global_pre_dur, output_dir_map, mechanics_df)
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

def _row_bucket_wholetrace_fate(row: pd.Series) -> str:
    """
    Fate-aware variant of `_row_bucket_wholetrace`. Reads `Fate_Status`
    from the mechanics dataframe row and appends '(intact)' or
    '(ruptured)' to EP-whole and EP-post labels. ASP rows are returned
    as 'ASP' regardless of fate.
    """
    if row.get('Condition_Type') == 'ASP':
        return 'ASP'
    is_post = bool(row.get('Post_Pulse_Entry_Flag', False))
    prefix = 'EP-post' if is_post else 'EP-whole'
    base = f"{prefix} {row.get('Voltage_V')}V {row.get('Duration_label')}"
    if row.get('Fate_Status') == "ruptured_post":
        return f"{base} (ruptured)"
    return f"{base} (intact)"

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
    """
    Whole-trace MI parameter SuperPlot.

    Cohort: MI whole-trace fits with R^2 flag pass, restricted to EP
    treatments common with the ASP baseline (see
    `_restrict_to_common_ep_treatments`). Categories are the whole-trace
    buckets returned by `_row_bucket_wholetrace_fate` and ordered by
    `_wholetrace_bucket_order` (ASP first, then EP-whole, then EP-post).

    Panels:
      1. Winner-model slope (`_mi_whole_winner_slope`): µm/s for Linear
         winners, dimensionless exponent b for Power-Law winners. Plotted
         on the same axis; the reader distinguishes model via the strip
         (SuperPlot dots are coloured per replicate).
      2. Power-Law amplitude a (PL winners only), log-scaled.

    Statistics: within-duration Intact-vs-Ruptured brackets on EP-whole
    (matches the fate-contrast plotter). No brackets on ASP.
    """
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df.copy()
    df = df[df['MI_Whole_Best_Model'].notna()
            & (df['MI_Whole_R2_Flag'] == True)].copy()
    df = _restrict_to_common_ep_treatments(df)
    if df.empty:
        return

    df['Bucket'] = df.apply(_row_bucket_wholetrace_fate, axis=1)
    order = _wholetrace_bucket_order(df['Bucket'].unique().tolist())
    if not order:
        return

    df['Winner_Slope_Whole'] = df.apply(_mi_whole_winner_slope, axis=1)

    # SuperPlot expects a 'Category' column identifying x-axis position.
    df['Category'] = df['Bucket']

    # Intact-vs-Ruptured bracket pairs within each EP-whole duration.
    bracket_pairs: List[Tuple[int, int]] = []
    for i, cat in enumerate(order):
        if '(intact)' in cat.lower():
            twin = cat.replace('(intact)', '(ruptured)')
            if twin in order:
                bracket_pairs.append((i, order.index(twin)))

    panels = [
        ('Winner_Slope_Whole',
         r'Rate: $m$ (µm/s) if Linear; $b$ (–) if Power-Law',
         'Slope (winner-model)'),
        ('MI_Whole_PL_a',
         r'Amplitude $a$ (µm)',
         'Power-Law amplitude $a$\n(PL winners only)'),
    ]

    bp.render_visco_parameter_superplot(
        df=df,
        panels=panels,
        output_pdf=output_dir / "Thesis_MI_WholeTrace_Parameter_Boxplots.pdf",
        category_col='Category',
        categories=order,
        figsize=(max(12.0, 2.4 * len(order) + 4.0), 6.0),
        grid_shape=(1, 3),  # panel-1, panel-2, legend
        bracket_pairs=bracket_pairs,
    )
    logger.info("MI whole-trace parameter SuperPlot written.")

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

        # ---- Panel 1: combined slope ------------------------------------
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

        # ---- Panel 2: PL_a alone ----------------------------------------
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

    # Reference styling: matches plot_thesis_combined_prepulse_trace_by_fate.
    bp.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    global_max_t = 0.0

    # Infer cell_type; whole-trace overlay is typically WT-only cohort.
    cell_type_for_plot = _cell_type_from_grouped_data(mi_whole_grouped_data)

    # Baseline ASP is dimmed when EP-whole cohorts are also plotted, so
    # the reader can tell it apart from a comparison condition.
    has_ep = any(b.startswith('EP-whole') or b.startswith('EP-post')
                 for b in order)

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
                f = interp1d(t_arr, l_arr, bounds_error=False,
                              fill_value=np.nan)
                matrix[i, :] = f(t_grid)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_L = np.nanmean(matrix, axis=0)
            sd_L   = np.nanstd(matrix, axis=0)
            n_t    = np.sum(np.isfinite(matrix), axis=0)

        MIN_FRAC_CONTRIBUTING = 0.25
        n_start = len(traces)
        min_n_required = max(1, int(np.ceil(MIN_FRAC_CONTRIBUTING * n_start)))
        plot_mask = (np.isfinite(mean_L) & np.isfinite(sd_L)
                      & (n_t >= min_n_required))

        if not np.any(plot_mask):
            continue

        c_t = t_grid[plot_mask]
        c_mean = mean_L[plot_mask]
        c_sd   = sd_L[plot_mask]

        # Colour routing matches the reference: NO_PULSE for ASP,
        # PULSE_TRACE_PALETTE for EP-whole/EP-post, fate drives linestyle.
        protocol, fate = _parse_bucket_to_style(bucket)
        if bucket == 'ASP':
            colour = bp.NO_PULSE_COLOUR
            fate_key = 'baseline'
        else:
            colour = bp.PULSE_TRACE_PALETTE.get(('EP', protocol),
                                                 bp.PULSE_100US_COLOUR)
            fate_key = fate
        is_baseline = (bucket == 'ASP') and has_ep

        lkw = bp.build_trace_line_kwargs(pulse_colour=colour,
                                          fate=fate_key,
                                          treatment=cell_type_for_plot,
                                          is_baseline=is_baseline)
        ax.plot(c_t, c_mean, markevery=bp.markevery_from(c_t),
                label=f'{bucket} (n={n_start})', **lkw)
        ax.fill_between(c_t, c_mean - c_sd, c_mean + c_sd,
                        color=colour, alpha=bp.TRACE_BAND_ALPHA,
                        linewidth=0)

        if bucket.startswith('EP-whole'):
            pulses = bucketed_pulses.get(bucket, [])
            if pulses:
                med_pulse = float(np.median(pulses))
                q1_pulse  = float(np.percentile(pulses, 25))
                q3_pulse  = float(np.percentile(pulses, 75))
                if 0 < med_pulse < c_t[-1]:
                    ax.axvline(med_pulse, color=colour, ls='--', lw=1.2,
                                alpha=0.7)
                    ax.axvspan(q1_pulse, q3_pulse, color=colour, alpha=0.08)

        if c_t[-1] > global_max_t:
            global_max_t = float(c_t[-1])

    ax.set_xlabel("Time from aspiration onset (s)")
    ax.set_ylabel(r"Protrusion length $L(t)$ (µm)")
    ax.set_title("MI whole-trace protrusion mechanics")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower right')

    if global_whole_dur is not None:
        ax.set_xlim(0, global_whole_dur)
    elif global_max_t > 0:
        ax.set_xlim(0, global_max_t)

    plt.tight_layout()
    utils.save_plot_pdf(
        output_dir / "Thesis_MI_WholeTrace_Protrusion_Dynamics.pdf")
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
    """
    2x2 layout, per bucket (ASP / EP-pre):

        (0,0) Trap_ID -> Body uptake plateau
        (0,1) Trap_ID -> Protrusion uptake plateau
        (1,0) Winner-model slope -> Protrusion uptake plateau, coloured by
              MI_Best_Model (Linear m or PL b).  Single Spearman across all
              points (mixed units — annotated as such).
        (1,1) PL_a -> Protrusion uptake plateau, Power-Law winners only.
    """
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

    # Analysis cohort.  Uptake A is only meaningful for cells whose fit
    # converged (per-region R2 flag) and had an identifiable plateau
    # (runaway exclusion).  Runaway fits pass the R2 gate comfortably, so
    # the flag alone is not enough, and the log axes below hide them
    # visually while they still enter every Spearman rho.  Blanking the
    # amplitude rather than dropping the row keeps the mechanics panels
    # on their own cohorts.
    for _region, _a_col in (('Body', body_uptake),
                            ('Protrusion', prot_uptake)):
        _flag = f'{REGION_PREFIX[_region]}_R2_Flag'
        if _flag in df.columns:
            df.loc[~df[_flag].fillna(False).astype(bool), _a_col] = np.nan
        _kept = apply_runaway_gate(df, _region, rule=DEFAULT_RUNAWAY_RULE)
        df.loc[~df.index.isin(_kept.index), _a_col] = np.nan

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

        # ---- helpers --------------------------------------------------------
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

        # ---- (0,0) Trap × Body uptake --------------------------------------
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

        # ---- (0,1) Trap × Protrusion uptake --------------------------------
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

        # ---- (1,0) Winner-model slope × Prot uptake (coloured by model) ----
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
                              edgecolor=colour, linewidths=0.5,
                              label=f"{model_name} (n={len(pts)})")
            _annotate_rho(ax_bl, sub_m['Winner_Slope'].to_numpy(),
                          sub_m['_y'].to_numpy())
            ax_bl.legend(frameon=False, fontsize=8, loc='lower right')
        ax_bl.set_xlabel(r'Slope: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law')
        ax_bl.set_ylabel("Protrusion Uptake Plateau\n(ADU/µm³) [log]")
        ax_bl.set_yscale('log')

        # ---- (1,1) PL_a × Prot uptake (Power-Law winners only) -------------
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
        utils.save_plot_pdf(
            output_dir / f"Thesis_Mechanics_vs_Uptake_{_safe_filename(bucket)}.pdf"
        )
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

    # Analysis cohort, matching the uptake-kinetics table: per-region R2
    # flag plus runaway exclusion.  Without this the 3-D surfaces are
    # fitted through amplitudes of order 1e4 that carry no information.
    _flag = f'{REGION_PREFIX["Protrusion"]}_R2_Flag'
    if _flag in df.columns:
        df = df[df[_flag].fillna(False).astype(bool)]
    df = apply_runaway_gate(df, 'Protrusion', rule=DEFAULT_RUNAWAY_RULE)
    u_label = 'Log10 Protrusion Uptake\nPlateau (ADU/µm³)'
                   
    for bucket, sub_df in df.groupby('Bucket'):
        # ASP is dropped here on purpose — the point of this figure is
        # spatial trends in the electroporated cells; ASP is covered by the
        # mean I(t) uptake trace (which shows dye uptake in ASP is
        # essentially flat and only appears after a pulse).
        if bucket == 'ASP':
            continue
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
                c=color, s=50, alpha=0.75, edgecolor=color, linewidths=0.5, depthshade=True
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
                             output_dir: Path,
                             output_dir_map: Optional[Dict[str, Path]] = None) -> None:
    """
    Pie chart of cell fate composition, one file per (Treatment, Duration_label)
    combination so the reader can compare WT vs CytD side-by-side within a
    pulse condition.

    Wedges per pie:
        intact          <- mechanics_df where Fate_Status == 'intact'
        ruptured_post   <- mechanics_df where Fate_Status == 'ruptured_post' (_R)
        excluded        <- sum of every non-analysed attrition bucket:
                             ruptured_pre_pulse    (_R0)
                             not_viable            (_DOA, _PI, _R_DOA, _R0_DOA)
                             post_pulse_arrival    (_after, _POST)
                             detection_failure     (_MASK, _SHAPE, _IRREGULAR)

    STYLE GRAMMAR (see Utils_MFA.MFA_STYLE_*):
        intact         = solid fill in the (cell_type, protocol) colour
        ruptured_post  = same colour, hatched (MFA_STYLE_RUPTURED_HATCH)
        excluded       = neutral grey (MFA_STYLE_EXCLUDED_COLOR)

    ASP cells are dropped because "pulse regime" is the grouping axis;
    ASP has no pulse.

    File naming: `Thesis_Fate_Pie_{Treatment}_{Duration_label}.pdf`
                 e.g. `Thesis_Fate_Pie_WT_100us.pdf`.
    Counts CSV (`Thesis_Fate_Counts_by_Pulse.csv`) is keyed on
    (Treatment, Duration_label) with the same numeric columns as before.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Fate histogram skipped: mechanics_df is empty.")
        return
    if 'Fate_Status' not in mechanics_df.columns:
        logger.warning("Fate histogram skipped: no Fate_Status column.")
        return
    if 'Treatment' not in mechanics_df.columns:
        logger.warning("Fate histogram skipped: no Treatment column.")
        return

    ep_mech = mechanics_df.loc[mechanics_df['Condition_Type'] == 'EP'].copy()
    if ep_mech.empty:
        logger.warning("Fate histogram skipped: no EP cells in mechanics_df.")
        return

    # ---- Count intact / ruptured_post per (Treatment, Duration_label) -----
    counts_analysis = (
        ep_mech.groupby(['Treatment', 'Duration_label', 'Fate_Status'])
        .size()
        .unstack(fill_value=0)
    )
    for col in ('intact', 'ruptured_post'):
        if col not in counts_analysis.columns:
            counts_analysis[col] = 0

    # ---- Pull excluded counts per (Treatment, Duration_label) -------------
    EXCLUDED_BUCKETS = [
        'ruptured_pre_pulse',
        'not_viable',
        'post_pulse_arrival',
        'detection_failure',
    ]
    excluded_counts = pd.Series(dtype=int)
    if attrition_df is not None and not attrition_df.empty:
        ep_att = attrition_df.loc[attrition_df['Condition_Type'] == 'EP']
        if not ep_att.empty and 'Treatment' in ep_att.columns:
            present_buckets = [b for b in EXCLUDED_BUCKETS if b in ep_att.columns]
            if present_buckets:
                excluded_counts = (
                    ep_att.assign(_excluded=ep_att[present_buckets].sum(axis=1))
                          .groupby(['Treatment', 'Duration_label'])['_excluded'].sum()
                )

    # ---- Assemble tidy count table (rows = (Treatment, Duration_label)) ---
    all_keys = set(counts_analysis.index) | set(excluded_counts.index)
    # Sort by treatment first (WT before CytD), then by duration
    treatment_order = {'WT': 0, 'CytD': 1}
    sorted_keys = sorted(
        all_keys,
        key=lambda k: (treatment_order.get(k[0], 99), _duration_sort_key(k[1])),
    )
    tidy = pd.DataFrame({
        'intact'        : [int(counts_analysis.get('intact',        pd.Series()).get(k, 0)) for k in sorted_keys],
        'ruptured_post' : [int(counts_analysis.get('ruptured_post', pd.Series()).get(k, 0)) for k in sorted_keys],
        'excluded'      : [int(excluded_counts.get(k, 0))                                   for k in sorted_keys],
    }, index=pd.MultiIndex.from_tuples(sorted_keys, names=['Treatment', 'Duration_label']))

    # ---- Draw: one pie per (Treatment, Duration_label) --------------------
    labels_by_key = {
        'intact'        : 'Intact',
        'ruptured_post' : 'Ruptured (post-pulse)',
        'excluded'      : 'Excluded (all reasons)',
    }

    for (treatment, duration), row in tidy.iterrows():
        counts = [int(row['intact']), int(row['ruptured_post']), int(row['excluded'])]
        keys   = ('intact', 'ruptured_post', 'excluded')
        keys_present = [k for k, c in zip(keys, counts) if c > 0]
        vals_present = [c for c in counts if c > 0]
        if not vals_present:
            logger.warning(f"Fate pie ({treatment}, {duration}) skipped: all-zero counts.")
            continue

        # Grammar-based colours for intact/ruptured; grey for excluded.
        # Under the flipped hatch semantics: intact = solid + white ///
        # hatch (loud), ruptured = white bg + coloured \\\ hatch (recessive).
        # Excluded stays neutral grey.
        protocol = utils.get_protocol_key(
            type('M', (), {'condition_type': 'EP', 'duration_label': duration})()
        )
        grammar_color = utils.get_style_color(treatment, protocol)

        # Build per-wedge kwargs. Excluded goes through raw kwargs; intact
        # and ruptured go through get_bar_kwargs so their hatch pattern
        # matches every other bar/pie in Chapter 3.
        wedge_kwargs_list = []
        for k in keys_present:
            if k == 'excluded':
                wedge_kwargs_list.append(dict(
                    facecolor=utils.MFA_STYLE_EXCLUDED_COLOR,
                    edgecolor='white',
                    hatch='',
                    linewidth=1.0,
                ))
            else:
                wedge_kwargs_list.append(utils.get_bar_kwargs(
                    treatment, protocol, fate=k,
                ))

        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        total = sum(vals_present)
        pie_labels  = [f"{labels_by_key[k]}\n(n={c}, {100*c/total:.0f}%)"
                       for k, c in zip(keys_present, vals_present)]

        wedges, _texts = ax.pie(
            vals_present,
            labels=pie_labels,
            # Colours are set per-wedge below, so pass a neutral placeholder.
            colors=['white'] * len(vals_present),
            startangle=90,
            counterclock=False,
            textprops=dict(fontsize=8),
        )
        # Apply grammar-specified facecolor / edgecolor / hatch per wedge.
        for w, wkw in zip(wedges, wedge_kwargs_list):
            w.set_facecolor(wkw['facecolor'])
            w.set_edgecolor(wkw['edgecolor'])
            w.set_linewidth(wkw['linewidth'])
            if wkw.get('hatch'):
                w.set_hatch(wkw['hatch'])

        ax.set_title(f"{treatment} - {duration} pulse  (N={total} candidates)",
                     fontsize=10, fontweight='bold')
        ax.axis('equal')

        plt.tight_layout()
        # Directory routing uses duration key (existing convention); the
        # Treatment split is captured in the filename.
        target_dir = (output_dir_map.get(duration, output_dir)
                      if output_dir_map is not None else output_dir)
        utils.save_plot_pdf(target_dir / f"Thesis_Fate_Pie_{treatment}_{duration}.pdf")
        plt.close()

    # ---- Write counts CSV (with derived fractions) -----------------------
    tidy_out = tidy.copy()
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
    tidy_out.to_csv(output_dir / "Thesis_Fate_Counts_by_Pulse.csv")
    frac_lines = [f"({t},{d})={f:.2f}"
                  for (t, d), f in tidy_out['fraction_ruptured'].items()
                  if np.isfinite(f)]
    logger.info("Fate pies written; fractions ruptured: " + ", ".join(frac_lines))


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


def plot_thesis_fate_per_trap(mechanics_df: pd.DataFrame,
                               output_dir: Path,
                               output_dir_map: Optional[Dict[str, Path]] = None) -> None:
    """
    Stacked bar per (Treatment, Duration_label): Trap_ID on x-axis, intact vs
    ruptured_post stacked.  Excluded cells appear in the pie chart function
    (which reports the per-pulse candidate budget); this figure is
    analysis-cohort only so the stack height is the number of analysable
    cells per trap.

    STYLE GRAMMAR (see Utils_MFA.MFA_STYLE_*):
        intact         = solid fill in the (Treatment, Duration_label) colour
        ruptured_post  = same colour, hatched (MFA_STYLE_RUPTURED_HATCH)

    One PDF per (Treatment, Duration_label) combination:
        e.g. `Thesis_Fate_PerTrap_WT_100us.pdf`,
             `Thesis_Fate_PerTrap_CytD_5ms.pdf`.
    Colours read from Utils_MFA.MFA_STYLE_COLOR.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Fate per-trap skipped: mechanics_df is empty.")
        return
    if 'Fate_Status' not in mechanics_df.columns:
        logger.warning("Fate per-trap skipped: no Fate_Status column.")
        return
    if 'Treatment' not in mechanics_df.columns:
        logger.warning("Fate per-trap skipped: no Treatment column.")
        return

    ep_mech = mechanics_df.loc[
        (mechanics_df['Condition_Type'] == 'EP')
        & mechanics_df['Fate_Status'].isin(['intact', 'ruptured_post'])
    ].copy()
    if ep_mech.empty:
        logger.warning("Fate per-trap skipped: no EP cells in mechanics_df.")
        return

    labels = {
        'intact'        : 'Intact',
        'ruptured_post' : 'Ruptured (post-pulse)',
    }

    # Iterate over (Treatment, Duration_label) in canonical order
    treatments_present = [t for t in _TREATMENT_ORDER
                          if (ep_mech['Treatment'] == t).any()]
    durations_present  = sorted(
        ep_mech['Duration_label'].dropna().unique(),
        key=_duration_sort_key,
    )

    for treatment in treatments_present:
        for d in durations_present:
            sub = ep_mech.loc[
                (ep_mech['Treatment'] == treatment)
                & (ep_mech['Duration_label'] == d)
            ]
            if sub.empty:
                logger.debug(f"Fate per-trap ({treatment}, {d}): no cells, skipping.")
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

            # Grammar colour for this (treatment, protocol) combination
            protocol = utils.get_protocol_key(
                type('M', (), {'condition_type': 'EP', 'duration_label': d})()
            )
            intact_kwargs = utils.get_bar_kwargs(treatment, protocol, fate='intact')
            rupt_kwargs   = utils.get_bar_kwargs(treatment, protocol, fate='ruptured_post')

            fig, ax = plt.subplots(figsize=(7.5, 3.2))
            x = np.arange(len(trap_ids))
            ax.bar(x, intact_v, label=labels['intact'], **intact_kwargs)
            ax.bar(x, ruptured_v, bottom=intact_v, label=labels['ruptured_post'],
                   **rupt_kwargs)

            for xi, i_v, r_v in zip(x, intact_v, ruptured_v):
                if i_v + r_v > 0:
                    ax.text(xi, i_v + r_v + 0.15, str(int(i_v + r_v)),
                            ha='center', va='bottom', fontsize=7, color='gray')

            ax.set_xticks(x)
            ax.set_xticklabels([f"T{t}" for t in trap_ids], rotation=45, ha='right')
            ax.set_xlabel("Trap ID")
            ax.set_ylabel("Cell count (analysable cohort)")
            n_total = int(intact_v.sum() + ruptured_v.sum())
            ax.set_title(f"Fate per trap - {treatment} {d} pulse (N={n_total})",
                         fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=False, fontsize=8, loc='upper right')

            plt.tight_layout()
            target_dir = (output_dir_map.get(d, output_dir)
                          if output_dir_map is not None else output_dir)
            utils.save_plot_pdf(target_dir / f"Thesis_Fate_PerTrap_{treatment}_{d}.pdf")
            plt.close()

    logger.info("Fate per-trap stacked bars written for "
                + ", ".join(f"{t}x{d}" for t in treatments_present
                            for d in durations_present))


# ==========================================================================
# BATCH 2 — Claim 2 (sensitivity to biological perturbation)
# ==========================================================================
# Colours: ASP-only WT-vs-CytD treatment comparisons use the canonical
# ASP_TREATMENT_COLOUR palette defined in bulk_plotting (WT = #1065ab,
# CytD = #b31529). Distinct from the pulse-comparison palette (dark navy
# for No pulse etc.), which is only used when EP conditions appear
# alongside ASP on the same axes.
_TREATMENT_COLOR = dict(bp.ASP_TREATMENT_COLOUR)
_TREATMENT_ORDER = ['WT', 'CytD']


def _actin_f0_trace(trap: bfh.TrapData, region: str):
    """
    Return (t_seconds, I_over_F0) for one region of one trap, or (None, None).

    region is 'Body' or 'Prot' (matches the CSV column suffix).
    Time is zero-referenced to the first detection frame (aspiration onset)
    so cross-cell interpolation can share a common axis.

    Frames where the region mean is zero are treated as segmentation
    failures (no mask), not as real zero-intensity readings, and are
    dropped from the returned trace.  This is critical for the
    protrusion region, which is empty for the first N frames before the
    protrusion has formed; if these zeros are kept, the cross-cell mean
    is dragged toward zero at early time points.
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

    # Align to aspiration onset BEFORE filtering, so protrusion traces
    # that start later keep their absolute time offset.
    t_aligned = t - t[0]

    valid = np.isfinite(t_aligned) & np.isfinite(I) & (I > 0)
    if valid.sum() < 5:
        return None, None

    return t_aligned[valid], I[valid] / f0_scalar


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
            t_idx = _TREATMENT_ORDER.index(treatment)
            ls = utils.get_group_linestyle(t_idx)
            mk = utils.get_group_marker(t_idx)
            ax.plot(common_t, mu, color=colour, lw=1.8, ls=ls,
                    marker=mk, markevery=max(1, len(common_t)//10),
                    markersize=5, label=f"{treatment} (n={n})")
            ax.fill_between(common_t, mu - sd, mu + sd,
                            color=colour, alpha=0.18, linewidth=0)

        ax.axhline(1.0, color='0.6', lw=0.8, ls='--', zorder=0)
        ax.set_title(region_titles[region], fontsize=10)
        ax.set_xlabel("Time from aspiration onset (s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    axes[0].set_ylabel(r"Actin intensity $I(t)/F_0$")
    axes[0].legend(frameon=False, fontsize=8, loc='best')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_ASP_Actin_Trace_by_Condition.pdf")
    plt.close()
    logger.info("Actin trace plot written.")

def plot_thesis_asp_actin_vs_mechanics(mechanics_df: pd.DataFrame,
                                        output_dir: Path,
                                        title_size: int = 14,
                                        axis_label_size: int = 14,
                                        tick_label_size: int = 14,
                                        annotation_size: int = 14,
                                        legend_size: int = 14,
                                        ) -> None:
    """
    Scatter: pre-pulse actin (F0-normalised) vs elastic modulus E, per cell.
    Two panels: body and protrusion.  Spearman ρ annotated per treatment.

    Filters to ASP intact cells with a valid viscoelastic fit.
    """
    from scipy import stats
    import numpy as np

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
        ('E_Pa',       r"$E$ (Pa)",                  True),
        ('eta1_Pa_s', r"$\eta_1$ (Pa$\cdot$s)",           True),
        ('Tau_s',     r"$\tau$ (s, Kelvin-Voigt only)",   True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharey=False)
    
    for row_idx, region in enumerate(('Body', 'Prot')):
        actin_col = f'Actin_{region}_PrePulse_F0Norm'
        for col_idx, (mech_col, mech_label, use_log) in enumerate(mech_panels):
            ax = axes[row_idx, col_idx]

            ax.tick_params(axis='both', which='both', labelsize=tick_label_size)

            if actin_col not in df.columns or mech_col not in df.columns:
                ax.set_visible(False)
                continue

            for treatment in _TREATMENT_ORDER:
                sub = df.loc[df['Treatment'] == treatment, [actin_col, mech_col]].copy()
                
                # Convert columns explicitly to float (coercing strings/objects to NaN)
                x_series = pd.to_numeric(sub[actin_col], errors='coerce')
                y_series = pd.to_numeric(sub[mech_col], errors='coerce')
                
                # Safe finite mask on numeric Series
                mask = np.isfinite(x_series) & np.isfinite(y_series)
                sub = sub[mask].copy()
                
                # Update sub with cast values to avoid future object-type errors downstream
                sub[actin_col] = x_series[mask]
                sub[mech_col] = y_series[mask]

                if sub.empty:
                    continue
                
                colour = _TREATMENT_COLOR[treatment]
                ax.scatter(sub[actin_col], sub[mech_col], s=22,
                           color=colour, edgecolor=colour, linewidths=0.4,
                           alpha=0.75, label=f"{treatment} (n={len(sub)})")

                # SAFE CHECK: Needs at least 5 points AND non-zero variance in both variables
                x_vals = sub[actin_col].values
                y_vals = sub[mech_col].values
                
                if len(sub) >= 5 and np.std(x_vals) > 0 and np.std(y_vals) > 0:
                    try:
                        res = stats.spearmanr(x_vals, y_vals)
                        rho = getattr(res, 'statistic', res[0] if isinstance(res, (tuple, list, np.ndarray)) else res)
                        p = getattr(res, 'pvalue', res[1] if isinstance(res, (tuple, list, np.ndarray)) else np.nan)
                        
                        # Extract scalar float cleanly if wrapped in array/Series
                        rho_val = float(rho) if hasattr(rho, '__float__') else float(np.asarray(rho).item())
                        p_val = float(p) if hasattr(p, '__float__') else float(np.asarray(p).item())

                        if np.isfinite(rho_val):
                            y_anchor = 0.95 if treatment == 'WT' else 0.88
                            ax.text(0.03, y_anchor,
                                    rf"$\rho_{{{treatment}}}={rho_val:.2f}$ (p={p_val:.2g})",
                                    transform=ax.transAxes, fontsize=annotation_size,
                                    color=colour, va='top')
                    except Exception as err:
                        logger.debug(f"Spearman computation skipped for {treatment} ({region}/{mech_col}): {err}")

            if use_log:
                ax.set_yscale('log')
            if row_idx == 1:
                ax.set_xlabel(rf"Actin $I/F_0$ ({region.lower()}, pre-pulse)", fontsize=axis_label_size)
            if col_idx == 0:
                ax.set_ylabel(f"{region}\n{mech_label}", fontsize=axis_label_size)
            else:
                ax.set_ylabel(mech_label, fontsize=axis_label_size)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Add legend safely if artists exist on axes[0, 0]
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, fontsize=legend_size, loc='lower right')

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
    1 kPa at trap 18) and for electrode distance. A flat regression here
    is what supports the "no meaningful pressure gradient effect" claim.

    Filters to ASP intact cells with a valid viscoelastic fit
    (``Visco_R2_Flag == True``) so the scatter cohort matches the
    ASP-only parameter boxplot and mean-trace figures. Panels 2 and 3
    depend on the fit, and gating panel 1 on the same criterion keeps a
    single reported n per treatment across the three panels.
    """
    import numpy as np
    import pandas as pd
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
        logger.warning("ASP trap dependency plot skipped: no ASP intact cells "
                       "with valid viscoelastic fit.")
        return

    panels = [
        ('Max_Prot_length_PrePulse_um', 'Protrusion length (µm)',         False),
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
                       color=colour, edgecolor=colour, linewidths=0.4,
                       alpha=0.75, label=f"{treatment} (n={len(sub)})")

            if len(sub) >= 5:
                # Compute Spearman correlation via pandas to avoid np.cov shape crash
                rho = sub['Trap_ID'].corr(sub[col], method='spearman')

                if pd.notna(rho):
                    if abs(rho) < 1.0:
                        dof = len(sub) - 2
                        t_stat = rho * np.sqrt(dof / (1.0 - rho**2))
                        p = 2 * stats.t.sf(np.abs(t_stat), df=dof)
                    else:
                        p = 0.0

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
                   s=22, color=colour, edgecolor=colour, linewidths=0.4,
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


def plot_thesis_asp_prot_length_vs_mechanics(mechanics_df: pd.DataFrame,
                                               output_dir: Path) -> None:
    """
    Two-panel scatter: max pre-pulse protrusion length vs elastic modulus E
    and vs parallel viscosity η₁.  Coloured by treatment, Spearman ρ
    annotated per treatment per panel.

    Rationale: L_max is set by the balance between driving pressure and
    cortical resistance at steady state.  If E scales with L_max within
    a treatment, the two are reading the same underlying property
    (cortical compliance); if they don't, they're independent readouts.
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
                       s=22, color=colour, edgecolor=colour, linewidths=0.4,
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
    utils.save_plot_pdf(output_dir / "Thesis_ASP_ProtLength_vs_Mechanics.pdf")
    plt.close()
    logger.info("Prot-length vs mechanics scatter written.")


def plot_thesis_asp_actin_f0_boxplot(mechanics_df: pd.DataFrame,
                                       output_dir: Path) -> None:
    """
    Two-panel boxplot: raw F0_Body and F0_Prot by treatment.

    F0-normalisation of the actin trace erases the between-cell baseline,
    so a between-condition actin comparison has to look at F0 itself.
    This plot answers the question the F0-normalised trace can't: does
    CytD reduce baseline actin fluorescence?

    Filters to ASP intact cells with a valid viscoelastic fit (same
    cohort as the viscoelastic boxplots, so figures are comparable).
    """
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

        # Match the box+strip style used elsewhere
        sns.boxplot(data=sub, x='Treatment', y=col, order=_TREATMENT_ORDER,
                    ax=ax, showfliers=False, color='lightgray')
        # Colour stripplot points by treatment using our palette
        for i, treatment in enumerate(_TREATMENT_ORDER):
            pts = sub.loc[sub['Treatment'] == treatment, col].to_numpy()
            if len(pts) == 0:
                continue
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(-0.15, 0.15, size=len(pts))
            ax.scatter(np.full(len(pts), i) + jitter, pts,
                       s=22, color=_TREATMENT_COLOR[treatment],
                       edgecolor=_TREATMENT_COLOR[treatment], linewidths=0.4, alpha=0.75, zorder=3)

        # Significance bracket
        vals_wt   = sub.loc[sub['Treatment'] == 'WT',   col].dropna().to_numpy()
        vals_cytd = sub.loc[sub['Treatment'] == 'CytD', col].dropna().to_numpy()
        stars, label, lw, delta = bp._build_stat_label(vals_cytd, vals_wt)
        n_wt, n_cytd = len(vals_wt), len(vals_cytd)
        logger.info(
            f"  [F0 {title:<10}] CytD vs WT: n=({n_cytd},{n_wt})  "
            f"stars={stars}  delta={delta:+.3f}"
        )
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
    utils.save_plot_pdf(output_dir / "Thesis_ASP_Actin_F0_Boxplot.pdf")
    plt.close()
    logger.info("F0 boxplot written.")


def run_thesis_claim1_plots(grouped_data: Dict,
                             mechanics_df: pd.DataFrame,
                             output_dir: Path,
                             global_asp_dur: Optional[float] = None,
                             output_dir_map: Optional[Dict[str, Path]] = None) -> None:
    """
    Claim 1 (ASP heterogeneity) plot suite.

    Every figure produced by this runner is a cross-treatment comparison
    (WT vs CytD) and therefore lands in ``output_dir`` — expected to be
    the ``combined/`` subfolder. ``output_dir_map`` is accepted for
    symmetry with ``run_thesis_plots`` and reserved for future per-
    treatment claim-1 figures; it is not currently used here.

    Actin F0 boxplot removed at Nikki's request (2026-07): the raw F0
    comparison isn't the story the chapter tells; the F0-normalised
    trace and Actin-vs-E already carry the CytD contrast.
    """
    del output_dir_map  # reserved
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_thesis_asp_actin_trace_by_condition(
        grouped_data, mechanics_df, output_dir, global_asp_dur)
    plot_thesis_asp_actin_vs_mechanics(mechanics_df, output_dir)
    plot_thesis_asp_trap_dependency(mechanics_df, output_dir)
    plot_thesis_asp_body_volume_vs_E(mechanics_df, output_dir)
    plot_thesis_asp_prot_length_vs_mechanics(mechanics_df, output_dir)


# ==========================================================================
# BATCH 3a — Claim 2 core (fate-mechanics story)
# ==========================================================================
# Colours for fate-cohort plots that pool across cell types.
# When the plot function knows the cell_type, it should use
# utils.get_scatter_kwargs(cell, protocol, fate=...) / get_line_kwargs(...)
# instead of reading this dict, so the colour tracks the (cell_type,
# protocol) grammar rather than a single generic hue per fate.
#
# The asp_baseline slot is kept for backwards compatibility but under the
# new grammar the recommended approach for baseline overlays is to call
# the grammar helpers with is_baseline=True, which dims the on-grammar
# colour by utils.MFA_STYLE_BASELINE_ALPHA rather than switching to grey.
_FATE_COLOR = {
    'intact'        : utils.get_style_color('WT', 'ASP'),   # generic dark  fallback
    'ruptured_post' : utils.MFA_COLORS['medium_red'],       # generic red   fallback
    'asp_baseline'  : '#808080',                            # legacy grey   (deprecated)
}


def _ep_fate_bucket(duration_label: str, fate_status: str) -> str:
    """
    Compact bucket label for fate-cohort figures.
    e.g. '100us_intact', '5ms_ruptured'.
    Falls through to raw values if either field is missing.
    """
    dl = str(duration_label) if duration_label is not None else '?'
    fs = str(fate_status)   if fate_status   is not None else '?'
    fs_short = {'intact': 'intact', 'ruptured_post': 'ruptured'}.get(fs, fs)
    return f"{dl}_{fs_short}"


def plot_thesis_ep_fate_mi_boxplots(mechanics_df: pd.DataFrame,
                                      output_dir: Path,
                                      r2_floor: float = 0.80) -> None:
    if mechanics_df is None or mechanics_df.empty:
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
        return

    ep_df['Bucket']       = ep_df.apply(
        lambda r: _ep_fate_bucket(r.get('Duration_label'), r.get('Fate_Status')), axis=1
    )
    ep_df['Winner_Slope'] = ep_df.apply(_mi_winner_slope, axis=1)

    durations = sorted(ep_df['Duration_label'].dropna().unique(), key=_duration_sort_key)
    order = []
    for d in durations:
        for fate_short in ('intact', 'ruptured'):
            b = f"{d}_{fate_short}"
            if b in ep_df['Bucket'].values:
                order.append(b)

    panels = [
        ('Winner_Slope', r'Rate: $m$ (µm/s) if Linear;  $b$ (–) if Power-Law', 'Slope (winner-model)', False, False),
        ('PL_a', r'Amplitude $a$ (µm)', 'Power-Law amplitude a (PL winners only)', True, True),
    ]

    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch, Rectangle
    def _tint(hex_color: str, alpha: float = 0.30) -> tuple:
        r, g, b, _ = to_rgba(hex_color)
        return (r, g, b, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(order) + 6), 3.8))

    for i, (col, ylabel, title, log_y, pl_only) in enumerate(panels):
        ax = axes[i]
        sub = ep_df.loc[ep_df[col].notna(), ['Bucket', col, 'Fate_Status', 'MI_Best_Model', 'Duration_label']].copy()
        if pl_only:
            sub = sub[sub['MI_Best_Model'] == 'Power-Law']
        if sub.empty:
            ax.set_visible(False)
            continue

        box_face, box_edge, box_hatch = {}, {}, {}
        for b in order:
            rows = sub[sub['Bucket'] == b]
            if rows.empty: continue
            dur_label = rows['Duration_label'].iloc[0]
            protocol  = dur_label if dur_label in ('100us', '5ms') else 'ASP'
            fate      = rows['Fate_Status'].iloc[0]
            edge      = utils.get_style_color('WT', protocol)
            
            box_edge[b] = edge
            if fate == 'ruptured_post':
                box_face[b]  = 'white'
                box_hatch[b] = utils.MFA_STYLE_RUPTURED_HATCH
            else:
                box_face[b]  = _tint(edge, 0.30)
                box_hatch[b] = ''

        sns.boxplot(data=sub, x='Bucket', y=col, order=order, hue='Bucket',
                    ax=ax, showfliers=False,
                    palette=box_face, legend=False, dodge=False)

        box_patches = [p for p in ax.patches if isinstance(p, (PathPatch, Rectangle))]
        for patch in box_patches:
            if isinstance(patch, Rectangle):
                box_x = patch.get_x() + patch.get_width() / 2.0
            else:
                box_x = patch.get_path().vertices[:, 0].mean()
                
            tick_idx = int(round(box_x))
            if 0 <= tick_idx < len(order):
                bucket = order[tick_idx]
                h = box_hatch.get(bucket, '')
                e = box_edge.get(bucket)
                f = box_face.get(bucket)
                
                if e is not None: patch.set_edgecolor(e)
                if f is not None:
                    patch.set_facecolor(f)
                    patch.set_alpha(1.0)
                if h:
                    patch.set_hatch(h)
                else:
                    patch.set_hatch(None)

        for j, bucket in enumerate(order):
            rows = sub[sub['Bucket'] == bucket]
            if rows.empty: continue
            rng = np.random.default_rng(42 + j)
            jitter = rng.uniform(-0.15, 0.15, size=len(rows))
            colours = rows['MI_Best_Model'].map(_MI_MODEL_UTILS_COLOR).fillna('gray')
            ax.scatter(np.full(len(rows), j) + jitter, rows[col],
                       s=22, c=colours, edgecolor=colours, linewidths=0.4, alpha=0.85, zorder=3)

        if log_y and (sub[col].dropna() > 0).all(): ax.set_yscale('log')

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    handles = []
    for d in durations:
        col_wt_dur = utils.get_style_color('WT', d) if d in ('100us', '5ms') else utils.get_style_color('WT', 'ASP')
        handles.append(mpatches.Patch(facecolor=_tint(col_wt_dur, 0.30), edgecolor=col_wt_dur, label=f'Intact ({d})'))
        handles.append(mpatches.Patch(facecolor='white', edgecolor=col_wt_dur, hatch=utils.MFA_STYLE_RUPTURED_HATCH, label=f'Ruptured ({d})'))
    handles += [
        mlines.Line2D([], [], marker='o', linestyle='', markerfacecolor=_MI_MODEL_UTILS_COLOR['Linear'], markeredgecolor=_MI_MODEL_UTILS_COLOR['Linear'], label='Linear winner'),
        mlines.Line2D([], [], marker='o', linestyle='', markerfacecolor=_MI_MODEL_UTILS_COLOR['Power-Law'], markeredgecolor=_MI_MODEL_UTILS_COLOR['Power-Law'], label='Power-Law winner'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_EP_Fate_MI_Boxplots.pdf")
    plt.close()
    logger.info("Fate-mechanics MI boxplot written.")


def _fate_box_bucket(row: pd.Series) -> str:
    """
    Bucket label for the fate boxplots: ASP-WT is a single reference
    bucket, EP cells are labelled by duration × fate.
    """
    if row.get('Condition_Type') == 'ASP':
        if row.get('Treatment') == 'WT':
            return 'ASP_WT'
        return f"ASP_{row.get('Treatment', '?')}"
    fate = row.get('Fate_Status')
    fate_short = 'intact' if fate == 'intact' else ('ruptured' if fate == 'ruptured_post' else str(fate))
    return f"{row.get('Duration_label')}_{fate_short}"


def _fate_box_order(buckets_present: list, durations: list) -> list:
    """ASP reference first (WT then any others), then per-duration intact/ruptured."""
    asp_here = sorted(b for b in buckets_present if b.startswith('ASP_'))
    order = list(asp_here)
    for d in durations:
        for fate_short in ('intact', 'ruptured'):
            b = f"{d}_{fate_short}"
            if b in buckets_present:
                order.append(b)
    return order


def plot_thesis_fate_prepulse_visco_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Fate-contrast SuperPlot on the pre-pulse viscoelastic parameters.

    Shares the same cohort and category structure as
    `bp.plot_prepulse_visco_parameter_boxplots` (No pulse + EP by
    fate), but restricts the statistical brackets to the within-duration
    Intact-vs-Ruptured contrasts that Claim 2 depends on.
    """
    if mechanics_df is None or mechanics_df.empty:
        return

    df = mechanics_df.copy()
    df = df.loc[df['PrePulse_Best_Model'].notna()
                & df['PrePulse_E_Pa'].notna()
                & (df['PrePulse_Visco_R2_Flag'] == True)].copy()

    keep_asp = (df['Condition_Type'] == 'ASP') & (df['Treatment'] == 'WT')
    keep_ep  = (df['Condition_Type'] == 'EP') & df['Fate_Status'].isin(['intact', 'ruptured_post'])
    df = df[keep_asp | keep_ep].copy()

    if df.empty:
        return

    df['Category'] = df.apply(bp._prepulse_new_category_label, axis=1)
    categories = bp._prepulse_category_order(df)

    # Fate-contrast brackets: pair each 'Intact' category with the matching
    # 'Ruptured' category at the same duration. The 'No pulse' baseline is
    # not bracketed here (it appears as an unbracketed reference).
    bracket_pairs: List[Tuple[int, int]] = []
    for i, cat in enumerate(categories):
        if '(Intact)' in cat:
            ruptured_twin = cat.replace('(Intact)', '(Ruptured)')
            if ruptured_twin in categories:
                bracket_pairs.append((i, categories.index(ruptured_twin)))

    panels = [
        ('PrePulse_E_Pa',      r'$E$ (Pa)',                'Parallel Spring Modulus'),
        ('PrePulse_E1_Pa',     r'$E_{1}$ (Pa)',            'Burgers Maxwell Spring\n(Burgers only)'),
        ('PrePulse_eta1_Pa_s', r'$\eta_{1}$ (Pa$\cdot$s)', 'Parallel Dashpot'),
        ('PrePulse_eta2_Pa_s', r'$\eta_{2}$ (Pa$\cdot$s)', 'Flow Viscosity\n(Jeffreys / Burgers)'),
        ('PrePulse_Tau_s',     r'$\tau$ (s)',              'Characteristic Time'),
    ]

    bp.render_visco_parameter_superplot(
        df=df,
        panels=panels,
        output_pdf=output_dir / "Thesis_Fate_PrePulse_Visco_Boxplots.pdf",
        category_col='Category',
        categories=categories,
        bracket_pairs=bracket_pairs,
    )
    logger.info("Fate pre-pulse viscoelastic SuperPlot written.")


def plot_thesis_fate_wholetrace_mi_boxplots(mechanics_df: pd.DataFrame,
                                             output_dir: Path,
                                             r2_floor: float = 0.80) -> None:
    """
    Fate-contrast SuperPlot on the whole-trace MI parameters.

    Uses the same category scheme as `plot_thesis_fate_prepulse_visco_boxplots`
    (No pulse baseline + EP by fate) so the two figures read as a matched
    pair. Statistical brackets are restricted to within-duration
    Intact-vs-Ruptured contrasts.

    Cohort gate: `MI_Whole_Best_Model` present *and* winner-model R^2
    at least `r2_floor` (0.80 by default).
    """
    if mechanics_df is None or mechanics_df.empty:
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
    df = df.loc[df['MI_Whole_Best_Model'].notna()
                & (df['MI_Whole_Winner_R2'] >= r2_floor)].copy()

    keep_asp = (df['Condition_Type'] == 'ASP') & (df['Treatment'] == 'WT')
    keep_ep  = ((df['Condition_Type'] == 'EP')
                & df['Fate_Status'].isin(['intact', 'ruptured_post']))
    df = df[keep_asp | keep_ep].copy()
    if df.empty:
        return

    df['Winner_Slope'] = df.apply(_wt_winner_slope, axis=1)

    # Category labels follow the same convention as the pre-pulse
    # SuperPlot so the two figures read as a matched pair.
    df['Category'] = df.apply(bp._prepulse_new_category_label, axis=1)
    categories = bp._prepulse_category_order(df)

    # Within-duration Intact-vs-Ruptured brackets.
    bracket_pairs: List[Tuple[int, int]] = []
    for i, cat in enumerate(categories):
        if '(Intact)' in cat:
            twin = cat.replace('(Intact)', '(Ruptured)')
            if twin in categories:
                bracket_pairs.append((i, categories.index(twin)))

    panels = [
        ('Winner_Slope',
         r'Rate: $m$ (µm/s) if Linear; $b$ (–) if Power-Law',
         'Whole-trace slope\n(winner-model)'),
        ('MI_Whole_PL_a',
         r'Amplitude $a$ (µm)',
         'Whole-trace power-law $a$\n(PL winners only)'),
    ]

    bp.render_visco_parameter_superplot(
        df=df,
        panels=panels,
        output_pdf=output_dir / "Thesis_Fate_WholeTrace_MI_Boxplots.pdf",
        category_col='Category',
        categories=categories,
        figsize=(max(12.0, 2.4 * len(categories) + 4.0), 6.0),
        grid_shape=(1, 3),  # panel-1, panel-2, legend
        bracket_pairs=bracket_pairs,
    )
    logger.info("Fate whole-trace MI SuperPlot written.")


def plot_thesis_ep_wholetrace_by_fate(grouped_data: Dict,
                                       mechanics_df: pd.DataFrame,
                                       output_dir: Path,
                                       global_whole_dur: Optional[float] = None) -> None:
    """
    Whole L(t) trace comparison: ASP baseline vs EP intact vs EP ruptured,
    two panels (one per pulse condition), all aligned to aspiration entry.

    Pulse timing varies across EP cells (frame index 10-120 across
    experiments) so it can't sit at a single x-value.  Instead, the
    median pulse time for each condition is annotated as a vertical
    dashed line, with the IQR as a shaded band.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Whole-trace plot skipped: empty mechanics_df.")
        return

    # Cohort: cells that passed the MI whole-trace R^2 gate, so this figure
    # shares its cells with the MI parameter/protrusion figures. Fate axis
    # then splits the EP cells (intact vs ruptured_post); ASP contributes
    # its baseline curve.
    mi_pass = mechanics_df.loc[mechanics_df['MI_Whole_R2_Flag'] == True]

    asp_pass = mi_pass.loc[
        (mi_pass['Condition_Type'] == 'ASP')
        & (mi_pass['Fate_Status'] == 'intact')
    ]
    ep_pass = mi_pass.loc[
        (mi_pass['Condition_Type'] == 'EP')
        & mi_pass['Fate_Status'].isin(['intact', 'ruptured_post'])
    ]
    asp_pairs = set(zip(asp_pass['Experiment_Folder'], asp_pass['Trap_ID']))
    ep_pairs  = set(zip(ep_pass['Experiment_Folder'],  ep_pass['Trap_ID']))

    # Group traces by (duration_label, kind), where kind is
    # 'asp_baseline' | 'intact' | 'ruptured_post'
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
                # ASP is used as baseline for every EP pulse condition
                for dl in ep_durations_present.copy():
                    traces.setdefault((dl, 'asp_baseline'), []).append((t_zeroed, L))
                # Also keep for later assignment once we know all durations
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

    # Fill in ASP baseline for durations that appeared after ASP was collected
    asp_all = traces.get(('__ALL_ASP__', 'asp_baseline'), [])
    for dl in ep_durations_present:
        if (dl, 'asp_baseline') not in traces:
            traces[(dl, 'asp_baseline')] = asp_all
    traces.pop(('__ALL_ASP__', 'asp_baseline'), None)

    durations = sorted(ep_durations_present, key=_duration_sort_key)
    if not durations:
        logger.warning("Whole-trace plot skipped: no EP durations present.")
        return

    # ---- Draw --------------------------------------------------------------
    n_panels = len(durations)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.4),
                              sharey=True, squeeze=False)
    axes = axes[0]

    def _aggregate(trace_list, t_grid):
        """Interpolate each (t, L) trace to t_grid, return mean and SD."""
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

    # Infer cell_type from the cohort so marker follows Treatment axis.
    cell_type_for_plot = _cell_type_from_grouped_data(grouped_data)

    bp.apply_thesis_rcparams()

    for ax, dl in zip(axes, durations):
        # Common time grid per panel
        max_t = 0.0
        for kind in ('asp_baseline', 'intact', 'ruptured_post'):
            for t, _L in traces.get((dl, kind), []):
                if len(t) and t[-1] > max_t:
                    max_t = float(t[-1])
        if max_t <= 0:
            continue
        common_t = np.linspace(0.0, max_t, 120)

        # Draw pulse timing band first so it sits behind lines
        pts = pulse_times.get(dl, [])
        if pts:
            pts_arr = np.asarray(pts, dtype=float)
            med = float(np.nanmedian(pts_arr))
            q1  = float(np.nanpercentile(pts_arr, 25))
            q3  = float(np.nanpercentile(pts_arr, 75))
            ax.axvspan(q1, q3, color='0.85', alpha=0.4, zorder=0)
            ax.axvline(med, color='0.4', lw=0.9, ls='--', zorder=1)

        # Chapter-3 pulse-condition palette: dark navy for No pulse
        # (dimmed when EP is co-plotted), pulse-specific blue for the EP
        # cohorts; solid line + filled marker for intact/baseline,
        # dashed + open marker for ruptured. New-convention labels.
        ep_colour = bp.PULSE_TRACE_PALETTE.get(('EP', str(dl)), bp.PULSE_100US_COLOUR)

        nopulse_label   = bp.format_condition_label(condition_type='ASP')
        ep_intact_label = bp.format_condition_label(
            condition_type='EP', voltage_v=100,
            duration_label=str(dl), fate_status='intact', with_fate=True)
        ep_rupt_label   = bp.format_condition_label(
            condition_type='EP', voltage_v=100,
            duration_label=str(dl), fate_status='ruptured_post', with_fate=True)

        plot_order = [
            ('asp_baseline',  bp.NO_PULSE_COLOUR, 'baseline',      True,  nopulse_label),
            ('intact',        ep_colour,          'intact',        False, ep_intact_label),
            ('ruptured_post', ep_colour,          'ruptured_post', False, ep_rupt_label),
        ]
        for kind, colour, fate, is_baseline, base_label in plot_order:
            trace_list = traces.get((dl, kind), [])
            if not trace_list:
                continue
            m, sd, n = _aggregate(trace_list, common_t)
            if m is None:
                continue
            lkw = bp.build_trace_line_kwargs(pulse_colour=colour,
                                              fate=fate,
                                              treatment=cell_type_for_plot,
                                              is_baseline=is_baseline)
            ax.plot(common_t, m, markevery=bp.markevery_from(common_t),
                    label=f"{base_label} (n={n})", **lkw)
            ax.fill_between(common_t, m - sd, m + sd,
                            color=colour, alpha=bp.TRACE_BAND_ALPHA,
                            linewidth=0)

        ax.set_title(f"{dl} pulse")
        ax.set_xlabel("Time from aspiration entry (s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    axes[0].set_ylabel(r"Protrusion length $L(t)$ (µm)")
    axes[0].legend(frameon=False, loc='lower right')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_EP_WholeTrace_by_Fate.pdf")
    plt.close()
    logger.info("Whole-trace by fate written.")

def _align_to_entry(t: np.ndarray, L: np.ndarray):
    """
    Return (t, L) re-zeroed so t = 0 corresponds to cell entry, defined
    as the first frame with L > 0. Leading zero-length frames from the
    recording's dead time (before the cell enters the trap) are
    discarded. Returns (None, None) if no L > 0 frame exists.
    """
    if t is None or L is None or len(t) == 0:
        return None, None
    positive = L > 0
    if not np.any(positive):
        return None, None
    entry_idx = int(np.argmax(positive))
    return t[entry_idx:] - t[entry_idx], L[entry_idx:]

def plot_thesis_combined_prepulse_trace_by_fate(grouped_data: Dict,
                                                mechanics_df: pd.DataFrame,
                                                output_dir: Path,
                                                global_pre_dur: Optional[float] = None,
                                                title_fontsize: Optional[float] = None,
                                                label_fontsize: Optional[float] = None,
                                                tick_fontsize: Optional[float] = None,
                                                legend_fontsize: Optional[float] = None) -> None:
    """
    Mean L(t) trace for the pre-pulse window, split by pulse condition
    and fate.
 
    Cell selection is the pre-pulse viscoelastic-fit cohort
    (``PrePulse_Visco_R2_Flag == True``) so this figure and the
    pre-pulse viscoelastic parameter boxplots (§3.7) share exactly the
    same cells.
 
    Every trace is re-zeroed to cell entry (the first frame with
    L > 0) so that t = 0 has the same physical meaning for every
    cell: the moment the protrusion first appears in the trap. The
    ``global_pre_dur`` window is therefore measured from cell entry.
 
    A cohort with fewer than three cells is rendered as individual
    thin traces rather than as a mean with SD band, since a two-cell
    mean is not a meaningful average.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Combined pre-pulse trace skipped: empty mechanics_df.")
        return
 
    # -----------------------------------------------------------------
    # Cohort selection: pre-pulse viscoelastic gate.
    #
    # Matches the parameter boxplots drawn in the same figure, so the
    # mean trace and the per-cell fits are drawn from an identical set
    # of cells. Cells that cleared this gate necessarily had a defined
    # pre-pulse trajectory of at least 15 L > 0 points, so no further
    # length or point-count filter is applied downstream.
    # -----------------------------------------------------------------
    visco_pass = mechanics_df.loc[mechanics_df['PrePulse_Visco_R2_Flag'] == True]
 
    treatments = mechanics_df['Treatment'].dropna().unique().tolist()
    base_treatment = 'WT' if 'WT' in treatments else (treatments[0] if treatments else 'WT')
 
    asp_mask = ((visco_pass['Condition_Type'] == 'ASP')
                & (visco_pass['Treatment'] == base_treatment))
    asp_pairs = set(zip(visco_pass.loc[asp_mask, 'Experiment_Folder'],
                        visco_pass.loc[asp_mask, 'Trap_ID']))
 
    durations = sorted(
        mechanics_df.loc[mechanics_df['Condition_Type'] == 'EP', 'Duration_label']
                    .dropna().unique(),
        key=_duration_sort_key,
    )
    if not durations:
        logger.warning("Combined pre-pulse trace skipped: no EP durations.")
        return
    target_dur = durations[0]
 
    intact_mask = ((visco_pass['Condition_Type'] == 'EP')
                   & (visco_pass['Duration_label'] == target_dur)
                   & (visco_pass['Fate_Status'] == 'intact'))
    ep_intact_pairs = set(zip(visco_pass.loc[intact_mask, 'Experiment_Folder'],
                              visco_pass.loc[intact_mask, 'Trap_ID']))
 
    rupt_mask = ((visco_pass['Condition_Type'] == 'EP')
                 & (visco_pass['Duration_label'] == target_dur)
                 & (visco_pass['Fate_Status'] == 'ruptured_post'))
    ep_rupt_pairs = set(zip(visco_pass.loc[rupt_mask, 'Experiment_Folder'],
                            visco_pass.loc[rupt_mask, 'Trap_ID']))
 
    # -----------------------------------------------------------------
    # Per-cell trace extraction.
    #
    # Extraction proceeds in three steps, uniformly for ASP and EP:
    #   1. Grab the raw pre-pulse trajectory (ASP: full aspiration
    #      trace via the helper; EP: raw frames up to the pulse
    #      frame).
    #   2. Re-zero to cell entry via _align_to_entry, so leading dead
    #      time frames with L = 0 are dropped and t = 0 corresponds
    #      to the first L > 0 frame.
    #   3. Truncate to the shared global_pre_dur window, which is now
    #      measured from cell entry.
    # -----------------------------------------------------------------
    traces: Dict[str, list] = {'asp': [], 'intact': [], 'ruptured': []}
    for gk, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
 
            if meta.condition_type == 'ASP' and key in asp_pairs:
                t, L = _extract_asp_phase_trace(trap)
                if t is None:
                    continue
                t, L = _align_to_entry(t, L)
                if t is None:
                    continue
                if global_pre_dur is not None:
                    m = t <= global_pre_dur
                    t, L = t[m], L[m]
                if len(t) > 0:
                    traces['asp'].append((t, L))
 
            elif (meta.condition_type == 'EP'
                  and meta.duration_label == target_dur
                  and (key in ep_intact_pairs or key in ep_rupt_pairs)):
                pf = meta.pulse_frame
                times = np.asarray(trap.protrusion_data.get('Time_s', []),
                                    dtype=float)
                lens  = np.asarray(trap.protrusion_data.get('Protrusion_Length_um', []),
                                    dtype=float)
                if not (0 < pf < len(times)):
                    continue
 
                t_all = times - times[0]
                pre_mask = np.arange(len(times)) < pf
                t = t_all[pre_mask]
                L = lens[pre_mask]
                if len(t) == 0:
                    continue
 
                t, L = _align_to_entry(t, L)
                if t is None:
                    continue
                if global_pre_dur is not None:
                    m = t <= global_pre_dur
                    t, L = t[m], L[m]
                if len(t) == 0:
                    continue
 
                if key in ep_intact_pairs:
                    traces['intact'].append((t, L))
                elif key in ep_rupt_pairs:
                    traces['ruptured'].append((t, L))
 
    # -----------------------------------------------------------------
    # Aggregation grid.
    # -----------------------------------------------------------------
    max_t = global_pre_dur if global_pre_dur is not None else max(
        (float(t[-1]) for group in traces.values() for t, _L in group),
        default=0.0,
    )
    if max_t <= 0:
        logger.warning("Combined pre-pulse trace skipped: no traces matched criteria.")
        return
    common_t = np.linspace(0.0, float(max_t), 120)
 
    def _aggregate(trace_list):
        if not trace_list:
            return None, None, 0
        M = np.full((len(trace_list), len(common_t)), np.nan)
        for i, (t_arr, L_arr) in enumerate(trace_list):
            if len(t_arr) < 2:
                continue
            f = interp1d(t_arr, L_arr,
                         bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            M[i, :] = f(common_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return (np.nanmean(M, axis=0),
                    np.nanstd(M, axis=0),
                    len(trace_list))
 
    # -----------------------------------------------------------------
    # Rendering.
    # -----------------------------------------------------------------
    bp.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    if global_pre_dur is not None:
        ax.axvspan(0, float(global_pre_dur), color='0.92', alpha=0.5, zorder=0)

    dur_key   = str(target_dur)
    ep_colour = bp.PULSE_TRACE_PALETTE.get(('EP', dur_key), bp.PULSE_100US_COLOUR)

    nopulse_label   = bp.format_condition_label(condition_type='ASP')
    ep_intact_label = bp.format_condition_label(
        condition_type='EP', voltage_v=100, duration_label=dur_key,
        fate_status='intact', with_fate=True)
    ep_rupt_label   = bp.format_condition_label(
        condition_type='EP', voltage_v=100, duration_label=dur_key,
        fate_status='ruptured_post', with_fate=True)

    plot_order = [
        ('asp',      bp.NO_PULSE_COLOUR, 'baseline',       True,  nopulse_label),
        ('intact',   ep_colour,          'intact',         False, ep_intact_label),
        ('ruptured', ep_colour,          'ruptured_post',  False, ep_rupt_label),
    ]

    SMALL_COHORT_N = 3

    for key, colour, fate, is_baseline, base_label in plot_order:
        trace_list = traces[key]
        n = len(trace_list)
        if n == 0:
            continue

        lkw = bp.build_trace_line_kwargs(pulse_colour=colour,
                                         fate=fate,
                                         treatment=base_treatment,
                                         is_baseline=is_baseline)

        if n < SMALL_COHORT_N:
            per_cell_kwargs = {
                'color':    colour,
                'linewidth': 0.9,
                'alpha':    0.75,
                'linestyle': lkw.get('linestyle', '-'),
                'marker':    '',
            }
            for i, (t_arr, L_arr) in enumerate(trace_list):
                ax.plot(t_arr, L_arr,
                        label=(f"{base_label} (n={n}, per-cell)"
                               if i == 0 else None),
                        **per_cell_kwargs)
        else:
            m, sd, _ = _aggregate(trace_list)
            if m is None:
                continue
            ax.plot(common_t, m,
                    markevery=bp.markevery_from(common_t),
                    label=f"{base_label} (n={n})", **lkw)
            ax.fill_between(common_t, m - sd, m + sd,
                            color=colour, alpha=bp.TRACE_BAND_ALPHA, linewidth=0)

    # Applied font sizes here:
    ax.set_xlabel("Time from cell entry (s)", fontsize=label_fontsize)
    ax.set_ylabel(r"Protrusion length $L(t)$ (µm)", fontsize=label_fontsize)
    ax.set_title(r"Pre-pulse protrusion mechanics ($\Delta t_{\mathrm{pre}}$-matched)", fontsize=title_fontsize)
    
    # Tick labels and legend font sizes:
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower right', fontsize=legend_fontsize)

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_Combined_PrePulse_Trace_by_Fate.pdf")
    plt.close()
    logger.info("Combined pre-pulse mean L(t) written (visco cohort, entry-aligned).")

def plot_thesis_combined_wholetrace_trace_by_fate(grouped_data: Dict,
                                                    mechanics_df: pd.DataFrame,
                                                    output_dir: Path,
                                                    global_whole_dur: Optional[float] = None) -> None:
    """
    Single-panel mean L(t) over the whole-trace window global_whole_dur,
    overlaying five cohorts using the Chapter 3 pulse-condition palette
    and label convention:
        - No pulse                    -> dark navy,  solid
        - 100 V, 100 µs (Intact)      -> mid  blue,  solid
        - 100 V, 100 µs (Ruptured)    -> mid  blue,  dashed
        - 100 V, 5 ms   (Intact)      -> deep blue,  solid
        - 100 V, 5 ms   (Ruptured)    -> deep blue,  dashed

    Cell selection matches the MI cohort (`MI_Whole_R2_Flag == True`) so
    this figure and the MI parameter boxplots share exactly the same cells.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Combined whole-trace plot skipped: empty mechanics_df.")
        return

    # Cell selection: MI cohort (matches MI parameter/protrusion figures).
    mi_pass = mechanics_df.loc[mechanics_df['MI_Whole_R2_Flag'] == True]

    asp_mask = ((mi_pass['Condition_Type'] == 'ASP')
                & (mi_pass['Treatment'] == 'WT')
                & (mi_pass['Fate_Status'] == 'intact'))
    asp_pairs = set(zip(mi_pass.loc[asp_mask, 'Experiment_Folder'],
                        mi_pass.loc[asp_mask, 'Trap_ID']))

    def _ep_pairs(dur: str, fate: str) -> set:
        m = mi_pass.loc[
            (mi_pass['Condition_Type'] == 'EP')
            & (mi_pass['Duration_label'] == dur)
            & (mi_pass['Fate_Status'] == fate)
        ]
        return set(zip(m['Experiment_Folder'], m['Trap_ID']))

    # cohort tuples: (key, pairs, pulse_colour, fate, label)
    nopulse_label = bp.format_condition_label(condition_type='ASP')
    cohort_defs = [
        ('asp',
         asp_pairs,
         bp.NO_PULSE_COLOUR,
         'baseline',
         nopulse_label),
        ('100us_int',
         _ep_pairs('100us', 'intact'),
         bp.PULSE_TRACE_PALETTE[('EP', '100us')],
         'intact',
         bp.format_condition_label(condition_type='EP', voltage_v=100,
                                    duration_label='100us',
                                    fate_status='intact', with_fate=True)),
        ('100us_rup',
         _ep_pairs('100us', 'ruptured_post'),
         bp.PULSE_TRACE_PALETTE[('EP', '100us')],
         'ruptured_post',
         bp.format_condition_label(condition_type='EP', voltage_v=100,
                                    duration_label='100us',
                                    fate_status='ruptured_post', with_fate=True)),
        ('5ms_int',
         _ep_pairs('5ms', 'intact'),
         bp.PULSE_TRACE_PALETTE[('EP', '5ms')],
         'intact',
         bp.format_condition_label(condition_type='EP', voltage_v=100,
                                    duration_label='5ms',
                                    fate_status='intact', with_fate=True)),
        ('5ms_rup',
         _ep_pairs('5ms', 'ruptured_post'),
         bp.PULSE_TRACE_PALETTE[('EP', '5ms')],
         'ruptured_post',
         bp.format_condition_label(condition_type='EP', voltage_v=100,
                                    duration_label='5ms',
                                    fate_status='ruptured_post', with_fate=True)),
    ]

    pair_to_cohort: Dict[Tuple[str, int], str] = {}
    traces: Dict[str, list] = {c[0]: [] for c in cohort_defs}
    for cohort_key, pairs, _colour, _fate, _label in cohort_defs:
        for p in pairs:
            pair_to_cohort[p] = cohort_key

    for gk, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            cohort_key = pair_to_cohort.get(key)
            if cohort_key is None:
                continue
            t, L = _extract_whole_trace(trap)
            if t is None:
                continue
            if global_whole_dur is not None:
                m = t <= global_whole_dur
                t, L = t[m], L[m]
            if len(t) < 15:
                continue
            traces[cohort_key].append((t, L))

    max_t = global_whole_dur if global_whole_dur is not None else max(
        (float(t[-1]) for group in traces.values() for t, _L in group),
        default=0.0,
    )
    if max_t <= 0:
        logger.warning("Combined whole-trace plot skipped: no traces.")
        return
    common_t = np.linspace(0.0, float(max_t), 160)

    # Min-fraction contributing guard: matches
    # plot_asp_protrusion_dynamics and the pre-pulse variant above. Keeps
    # the tail of the mean trace from being a mean over a shrinking
    # sub-cohort.
    MIN_FRAC_CONTRIBUTING = 0.25

    def _aggregate(trace_list):
        if not trace_list:
            return None, None, None, 0
        M = np.full((len(trace_list), len(common_t)), np.nan)
        for i, (t_arr, L_arr) in enumerate(trace_list):
            if len(t_arr) < 2:
                continue
            f = interp1d(t_arr, L_arr, bounds_error=False, fill_value=np.nan,
                         assume_sorted=True)
            M[i, :] = f(common_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m_arr  = np.nanmean(M, axis=0)
            sd_arr = np.nanstd(M, axis=0)
            n_arr  = np.sum(np.isfinite(M), axis=0)
        n_start = len(trace_list)
        min_n_required = max(1, int(np.ceil(MIN_FRAC_CONTRIBUTING * n_start)))
        mask = np.isfinite(m_arr) & np.isfinite(sd_arr) & (n_arr >= min_n_required)
        return m_arr, sd_arr, mask, n_start

    bp.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for cohort_key, _pairs, colour, fate, base_label in cohort_defs:
        m, sd, mask, n = _aggregate(traces[cohort_key])
        if m is None or n == 0 or not np.any(mask):
            continue
        t_masked  = common_t[mask]
        m_masked  = m[mask]
        sd_masked = sd[mask]
        # No-pulse cohort reads as baseline (dimmed) so the EP cohorts
        # visually dominate the comparison.
        is_baseline = (cohort_key == 'asp')
        lkw = bp.build_trace_line_kwargs(pulse_colour=colour,
                                          fate=fate, treatment='WT',
                                          is_baseline=is_baseline)
        ax.plot(t_masked, m_masked,
                markevery=bp.markevery_from(t_masked),
                label=f"{base_label} (n={n})", **lkw)
        ax.fill_between(t_masked, m_masked - sd_masked, m_masked + sd_masked,
                        color=colour, alpha=bp.TRACE_BAND_ALPHA, linewidth=0)
        logger.info(
            f"  [WholeTrace] {cohort_key}: n_start={n} "
            f"plotted_to={float(t_masked[-1]):.1f}s "
            f"(min_n={max(1, int(np.ceil(MIN_FRAC_CONTRIBUTING * n)))})"
        )

    ax.set_xlabel("Time from aspiration onset (s)")
    ax.set_ylabel(r"Protrusion length $L(t)$ (µm)")
    ax.set_title("Whole-trace protrusion mechanics by fate")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower right')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_Combined_WholeTrace_Trace_by_Fate.pdf")
    plt.close()
    logger.info("Combined whole-trace mean L(t) written.")


def run_thesis_claim2_plots(grouped_data: Dict,
                             mechanics_df: pd.DataFrame,
                             output_dir: Path,
                             global_whole_dur: Optional[float] = None,
                             whole_trace_dir: Optional[Path] = None,
                             global_pre_dur: Optional[float] = None) -> None:
    """
    Convenience runner: Batch 3a Claim 2 core plots.
    """
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Claim 2 plot '{label}' failed: {e}", exc_info=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    wt_dir = whole_trace_dir if whole_trace_dir is not None else output_dir
    wt_dir.mkdir(parents=True, exist_ok=True)

    _try("EP wholetrace by fate", plot_thesis_ep_wholetrace_by_fate,
         grouped_data, mechanics_df, wt_dir, global_whole_dur)
    _try("Fate prepulse visco boxplots", plot_thesis_fate_prepulse_visco_boxplots,
         mechanics_df, output_dir)
    _try("Fate wholetrace mi boxplots", plot_thesis_fate_wholetrace_mi_boxplots,
         mechanics_df, wt_dir)

    _try("Combined prepulse trace by fate", 
         plot_thesis_combined_prepulse_trace_by_fate(grouped_data, mechanics_df, 
                                                     output_dir, global_pre_dur,
                                                     title_fontsize  =  15,
                                                     label_fontsize  =  15,
                                                     tick_fontsize   =  15,
                                                     legend_fontsize =  15))
    _try("Combined wholetrace trace by fate", plot_thesis_combined_wholetrace_trace_by_fate,
         grouped_data, mechanics_df, wt_dir, global_whole_dur)

# ==========================================================================
# BATCH 3b — Actin around the pulse, and the WT uptake trace
# ==========================================================================
# These three plots answer the "electroporation actually did something"
# question at three timescales: (1) paired pre/post scalar boxplots per
# condition, (2) time-resolved mean actin trace aligned to the pulse frame,
# (3) mean volume-normalised uptake trace for WT across ASP / 100us / 5ms.
#
# The paired scalar comparison depends on `Actin_{Body,Prot}_{PrePulse,
# PostPulse}_F0Norm` columns from mechanics_df.  Those columns are subject
# to the same pulse-frame indexing bug noted in the chapter (0-based vs
# 1-based offset in the pipeline); the trend direction is correct but the
# exact pre-vs-post effect size may shift by a frame once the bug is
# fixed.  A red-text placeholder in the LaTeX draft flags this.
# ==========================================================================
_EP_PULSE_ORDER = ('100us', '5ms')


def plot_thesis_actin_prepulse_vs_postpulse_paired(mechanics_df: pd.DataFrame,
                                                     output_dir: Path,
                                                     r2_floor: float = 0.85
                                                     ) -> None:
    """
    Paired pre/post actin boxplot with within-cell lines.

    Layout: 2 rows (Body, Prot) × 2 cols (100us, 5ms) = 4 subplots.
    Each subplot contains a paired boxplot for WT and CytD side-by-side.
    Each individual cell contributes two dots (pre, post) connected by a
    thin grey line, so the reader can see the within-cell direction.

    Wilcoxon signed-rank on each within-cell (pre, post) pair; stars +
    Cliff's delta annotated per (treatment, region, duration) group via
    the same helper as the other thesis plots.
    """
    from scipy import stats
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Paired actin plot skipped: empty mechanics_df.")
        return

    # EP cohort by fate only; pre-pulse mechanics fit quality does not
    # gate the actin readout since the actin values are per-frame
    # fluorescence integrals rather than fit-derived quantities. Keep
    # both intact and ruptured_post because the pre-pulse actin readout
    # is still meaningful for cells that later rupture.
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

    # Only render treatments actually present in the cohort. When CytD
    # wasn't run for the EP arm we drop those columns entirely so the WT
    # boxes don't get squashed into half the axis.
    treatments_present = [t for t in _TREATMENT_ORDER
                          if (df['Treatment'] == t).any()]
    if not treatments_present:
        logger.warning("Paired actin plot skipped: no known treatments in EP cohort.")
        return

    per_col_width = 2.2 * len(treatments_present)
    fig, axes = plt.subplots(2, len(durations_present),
                              figsize=(per_col_width * len(durations_present), 6.4),
                              squeeze=False)

    # x-tick geometry: each present treatment gets a (pre, post) pair with
    # a 1-unit gap between treatments. Positions and labels are recomputed
    # per subplot so they stay in sync with `treatments_present`.
    positions = {t: (2 * i, 2 * i + 1) for i, t in enumerate(treatments_present)}
    xticks = []
    xticklabels = []
    for t in treatments_present:
        p_pre, p_post = positions[t]
        xticks.extend([p_pre, p_post])
        xticklabels.extend([f"{t}\npre", f"{t}\npost"])

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

            for treatment in treatments_present:
                pts = sub_d.loc[sub_d['Treatment'] == treatment]
                if pts.empty:
                    continue
                x_pre, x_post = positions[treatment]
                # (treatment, dur) grammar colour so long vs short pulse
                # can be told apart at a glance; marker follows treatment.
                colour = utils.get_style_color(treatment, dur)
                marker = utils.get_style_marker(treatment)

                # Within-cell lines
                for _, r in pts.iterrows():
                    ax.plot([x_pre, x_post], [r[pre_col], r[post_col]],
                            color='0.75', lw=0.5, alpha=0.7, zorder=1)

                # Boxplots
                ax.boxplot([pts[pre_col].to_numpy(), pts[post_col].to_numpy()],
                           positions=[x_pre, x_post], widths=0.55,
                           patch_artist=True,
                           boxprops=dict(facecolor='lightgray', alpha=0.4,
                                         edgecolor=colour),
                           medianprops=dict(color=colour, lw=1.4),
                           whiskerprops=dict(color=colour),
                           capprops=dict(color=colour),
                           flierprops=dict(marker=marker, markersize=3,
                                            markerfacecolor=colour,
                                            markeredgecolor='none'),
                           showfliers=False)

                # Individual points
                rng = np.random.default_rng(42 + col_idx * 10 + row_idx)
                for x_, col_ in ((x_pre, pre_col), (x_post, post_col)):
                    vals = pts[col_].to_numpy()
                    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                    ax.scatter(np.full(len(vals), x_) + jitter, vals,
                               s=18, color=colour, marker=marker,
                               edgecolor=colour, linewidths=0.4,
                               alpha=0.75, zorder=3)

                # Wilcoxon within-cell
                if len(pts) >= 5:
                    try:
                        w_stat, p_val = stats.wilcoxon(pts[pre_col],
                                                        pts[post_col])
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

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=9)
            ax.set_xlim(min(xticks) - 0.7, max(xticks) + 0.7)
            ax.axhline(1.0, color='0.6', lw=0.6, ls='--', zorder=0)
            if col_idx == 0:
                ax.set_ylabel(f"{region}\n"
                              r"Actin $I/F_0$")
            ax.set_title(f"{dur} pulse — {region}", fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_EP_Actin_PrePost_Paired.pdf")
    plt.close()
    logger.info("Paired pre/post actin boxplot written (pulse-frame bug caveat applies).")


def plot_thesis_actin_around_pulse_trace(grouped_data: Dict,
                                          mechanics_df: pd.DataFrame,
                                          output_dir: Path,
                                          output_dir_map: Optional[Dict[str, Path]] = None,
                                          window_pre_s: float = 15.0,
                                          window_post_s: float = 15.0
                                          ) -> None:
    """
    Time-resolved mean actin fluorescence aligned to the pulse frame,
    one figure per pulse duration.

    Each figure has 2 rows (Body, Prot) × N cols (one per treatment).
    Within a subplot, mean +/- SD across cells is plotted, with the pulse
    itself as a vertical dashed line at t=0. Traces are F0-normalised
    per cell.

    Parameters
    ----------
    output_dir : Path
        Default directory in which per-duration files are saved when
        ``output_dir_map`` is not supplied. Files are named
        ``Thesis_EP_Actin_Around_Pulse_{duration}.pdf``.
    output_dir_map : dict[str, Path], optional
        Mapping from pulse duration label (e.g. ``'100us'``) to the
        directory in which that duration's figure should be saved. If
        provided, each duration's file goes to its mapped directory
        (falling back to ``output_dir`` for any duration not in the
        map). This lets the caller route ``100us`` and ``5ms`` figures
        into their respective ``at_pulse/100V_100us/`` and
        ``at_pulse/100V_5ms/`` subfolders.

    Only EP cells with a fate label of ``intact`` or ``ruptured_post``
    are included; per-region traces with fewer than 5 valid time points
    around the pulse are dropped.
    """
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

    # (treatment, duration, region) -> list of (t_aligned, I/F0) traces
    traces: Dict[Tuple[str, str, str], list] = {}
    durations_seen: set = set()

    for gk, traps in grouped_data.items():
        if not traps or traps[0].metadata.condition_type != 'EP':
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            if key not in accepted:
                continue
            treatment = meta.treatment
            duration  = meta.duration_label
            if treatment not in _TREATMENT_ORDER:
                continue
            durations_seen.add(duration)

            ad = getattr(trap, 'actin_data', {}) or {}
            if 'Time_s' not in ad:
                continue
            t_raw = np.asarray(ad['Time_s'], dtype=float)
            if len(t_raw) == 0:
                continue
            pf = meta.pulse_frame
            if not (0 < pf < len(t_raw)):
                continue
            t_pulse_zero = t_raw - t_raw[pf]  # NB: pulse-frame indexing caveat

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
                traces.setdefault((treatment, duration, region), []).append(
                    (t_pulse_zero[valid], I[valid] / f0_scalar)
                )

    durations = sorted(durations_seen, key=_duration_sort_key)
    if not durations:
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

    # One figure per duration. 2 rows (Body / Prot) x 1 col each; treatments
    # overlaid within a subplot as separate coloured lines.
    for dur in durations:
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.6),
                                  sharex=True, squeeze=False)

        any_data_this_dur = False
        for row_idx, region in enumerate(('Body', 'Prot')):
            ax = axes[row_idx, 0]
            ax.axvline(0.0, color=utils.MFA_COLORS.get('pulse', 'gray'),
                       lw=1.0, ls='--', zorder=1)
            ax.axhline(1.0, color='0.7', lw=0.6, ls=':', zorder=0)

            for treatment in _TREATMENT_ORDER:
                mu, sd, n = _agg(traces.get((treatment, dur, region), []))
                if mu is None:
                    continue
                any_data_this_dur = True
                # Grammar routing: this is EP data at duration `dur`, so
                # the colour comes from (treatment, dur). Fate isn't split
                # here (treatments are the axis), so use fate='intact'.
                lkw = utils.get_line_kwargs(treatment, dur,
                                            fate='intact', size=5)
                colour = lkw['color']
                ax.plot(common_t, mu, markevery=max(1, len(common_t)//10),
                        label=f"{treatment} (n={n})", **lkw)
                ax.fill_between(common_t, mu - sd, mu + sd,
                                color=colour, alpha=0.18, linewidth=0)

            ax.set_title(f"{dur} pulse — {region}", fontsize=9)
            if row_idx == 1:
                ax.set_xlabel("Time from pulse (s)")
            ax.set_ylabel(f"{region}\n" r"Actin $I(t)/F_0$")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if not any_data_this_dur:
            plt.close(fig)
            logger.info(f"  Skipping {dur}: no valid actin traces.")
            continue

        axes[0, 0].legend(frameon=False, fontsize=8, loc='best')
        plt.tight_layout()

        target_dir = output_dir
        if output_dir_map is not None and dur in output_dir_map:
            target_dir = output_dir_map[dur]
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_EP_Actin_Around_Pulse_{dur}.pdf")
        plt.close()
        logger.info(f"Actin-around-pulse trace ({dur}) written "
                    f"(pulse-frame indexing caveat applies).")


def plot_thesis_protrusion_around_pulse(grouped_data: Dict,
                                          mechanics_df: pd.DataFrame,
                                          output_dir: Path,
                                          output_dir_map: Optional[Dict[str, Path]] = None,
                                          window_pre_s: float = 15.0,
                                          window_post_s: float = 15.0
                                          ) -> None:
    """
    Time-resolved mean protrusion length aligned to the pulse frame,
    one figure per pulse duration and split by post-pulse behaviour.

    Each figure has one panel per behaviour bin (Extends, Stable,
    Retracts) as classified by ``EP_Post_Pulse_Behavior`` on the
    mechanics dataframe. Within each panel the mean protrusion length
    across cells is drawn as a solid line with a shaded standard-
    deviation band, and the pulse itself is a vertical dashed line at
    t = 0.

    Parameters
    ----------
    output_dir : Path
        Default directory in which per-duration files are saved.
    output_dir_map : dict[str, Path], optional
        Mapping from pulse duration label (e.g. ``'100us'``) to the
        directory in which that duration's figure should be saved.

    Only EP cells with a fate label of ``intact`` or ``ruptured_post``
    and a non-null ``EP_Post_Pulse_Behavior`` classification enter the
    plot; per-behaviour bins with fewer than 3 cells are drawn as thin
    faint lines with the smaller-n caveat visible from the panel
    header.
    """
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

    # (duration, behavior) -> list of (t_aligned, length) traces
    traces: Dict[Tuple[str, str], list] = {}
    durations_seen: set = set()

    for gk, traps in grouped_data.items():
        if not traps or traps[0].metadata.condition_type != 'EP':
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            behaviour = behavior_lookup.get(key)
            if behaviour is None:
                continue

            duration = meta.duration_label
            durations_seen.add(duration)

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
            traces.setdefault((duration, behaviour), []).append(
                (t_pulse_zero[valid], l_raw[valid])
            )

    durations = sorted(durations_seen, key=_duration_sort_key)
    if not durations:
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

    # Colours by behaviour: extends = growth-red, retracts = shrink-blue,
    # stable = mid-grey. Behaviour order is Retracts, Stable, Extends so
    # the reader scans from "shrinking" through "unchanged" to "growing".
    behaviour_order  = ['Retracts', 'Stable', 'Extends']
    behaviour_colour = {
        'Retracts': utils.MFA_COLORS.get('dark_blue',  '#2c6fa4'),
        'Stable'  : utils.MFA_COLORS.get('mid_grey',   '#7a7a7a'),
        'Extends' : utils.MFA_COLORS.get('dark_red',   '#a83232'),
    }

    for dur in durations:
        # Only show panels for behaviours that actually have cells at
        # this duration; if all three are populated we get a 1x3 layout.
        behaviours_here = [b for b in behaviour_order
                           if traces.get((dur, b))]
        if not behaviours_here:
            logger.info(f"  Skipping {dur}: no cells with classified post-pulse behaviour.")
            continue

        fig, axes = plt.subplots(1, len(behaviours_here),
                                  figsize=(4.5 * len(behaviours_here), 4.0),
                                  sharey=True, squeeze=False)

        for j, behaviour in enumerate(behaviours_here):
            ax = axes[0, j]
            ax.axvline(0.0, color=utils.MFA_COLORS.get('pulse', 'gray'),
                       lw=1.0, ls='--', zorder=1)

            trs = traces.get((dur, behaviour), [])
            mu, sd, n = _agg(trs)
            if mu is None:
                ax.set_visible(False)
                continue

            colour = behaviour_colour.get(behaviour, 'black')
            alpha_line = 1.0 if n >= 3 else 0.6
            lw_line    = 1.8 if n >= 3 else 1.0
            b_idx = behaviour_order.index(behaviour)
            ls = utils.get_group_linestyle(b_idx)
            mk = utils.get_group_marker(b_idx)

            ax.plot(common_t, mu, color=colour, lw=lw_line, ls=ls,
                    marker=mk, markevery=max(1, len(common_t)//10),
                    markersize=5, alpha=alpha_line, label=f"n = {n}")
            ax.fill_between(common_t, mu - sd, mu + sd,
                            color=colour, alpha=0.18, linewidth=0)

            ax.set_title(f"{dur} pulse — {behaviour} (n = {n})",
                         fontsize=10)
            ax.set_xlabel("Time from pulse (s)")
            if j == 0:
                ax.set_ylabel(r"Protrusion length $L(t)$ (µm)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()

        target_dir = output_dir
        if output_dir_map is not None and dur in output_dir_map:
            target_dir = output_dir_map[dur]
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(target_dir / f"Thesis_EP_Protrusion_Around_Pulse_{dur}.pdf")
        plt.close()
        logger.info(f"Protrusion-around-pulse trace ({dur}) written.")


def plot_thesis_mean_uptake_trace_wt(grouped_data: Dict,
                                       mechanics_df: pd.DataFrame,
                                       output_dir: Path,
                                       min_frac_contributing: float = 0.25
                                       ) -> None:
    """
    Mean volume-normalised uptake trace I(t) for WT cells, aligned to
    aspiration onset (t=0 at first frame of each trap).

    Three panels: Body, Protrusion, Total (`_VolNorm` columns from the
    per-trap Uptake CSV).  Each panel overlays three conditions on the
    same axes:

        - ASP           (grey, no pulse)
        - EP 100us      (blue ramp, pulse-frame band annotated)
        - EP 5ms        (blue ramp, pulse-frame band annotated)

    The point of the figure is the contrast: ASP is essentially flat
    (any 'uptake' is background drift with no pore opening), while both
    EP curves rise after the pulse.

    Uses the intact ASP cohort + intact/ruptured_post EP cohort so ruptured
    cells still contribute to the mean until they exit.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Mean uptake trace skipped: empty mechanics_df.")
        return

    # Restrict to WT (per Nikki's request); include ASP intact and EP
    # analysis-cohort (intact / ruptured_post) cells.
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

    # (condition_key) -> list of (t_from_entry, I(t)) per region
    conditions = ('ASP', 'EP_100us', 'EP_5ms')
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

            # Bucket
            if meta.condition_type == 'ASP':
                cond_key = 'ASP'
            elif meta.condition_type == 'EP':
                if meta.duration_label == '100us':
                    cond_key = 'EP_100us'
                elif meta.duration_label == '5ms':
                    cond_key = 'EP_5ms'
                else:
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

                # `*_VolNorm` arrives baseline-corrected from
                # `bulk_file_handling` as (I(t) - F0) / V(t), so no
                # subtraction is needed here.  Note this figure aligns on
                # aspiration onset while F0 is a pre-pulse window for EP
                # cells and a post-entry window for ASP cells, matching
                # the two branches in UptakeQuantification.

                valid = np.isfinite(t_zero) & np.isfinite(y)
                if valid.sum() < 5:
                    continue
                per_cond_traces[cond_key][region].append(
                    (t_zero[valid], y[valid])
                )
                if t_zero[valid][-1] > max_t_seen:
                    max_t_seen = float(t_zero[valid][-1])

            # Pulse time for EP
            if cond_key.startswith('EP_'):
                pf = meta.pulse_frame
                if 0 < pf < len(t_raw):
                    per_cond_pulse[cond_key].append(float(t_raw[pf] - t_raw[0]))

    if max_t_seen <= 0:
        logger.warning("Mean uptake trace skipped: no valid uptake data.")
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
        'ASP'     : utils.MFA_COLORS['light_grey'],
        'EP_100us': utils.MFA_COLORS['medium_blue'],
        'EP_5ms'  : utils.MFA_COLORS['dark_blue'],
    }
    cond_edge = {
        'ASP'     : '0.4',
        'EP_100us': utils.MFA_COLORS['medium_blue'],
        'EP_5ms'  : utils.MFA_COLORS['dark_blue'],
    }
    cond_label = {
        'ASP'     : 'ASP (no pulse)',
        'EP_100us': 'EP 100 µs',
        'EP_5ms'  : 'EP 5 ms',
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
            # Build a common_t sub-array matching the mask inside _aggregate
            # by re-running the mask logic (small duplication; simpler
            # than plumbing the mask out).
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

            c_idx = list(conditions).index(cond_key)
            ls = utils.get_group_linestyle(c_idx)
            mk = utils.get_group_marker(c_idx)
            ax.plot(t_plot, mean_y,
                    color=cond_edge[cond_key], lw=1.8, ls=ls,
                    marker=mk, markevery=max(1, len(t_plot)//10),
                    markersize=5,
                    label=f"{cond_label[cond_key]} (n={n_start})")
            ax.fill_between(t_plot, mean_y - sd_y, mean_y + sd_y,
                            color=cond_colour[cond_key], alpha=0.20,
                            linewidth=0)

            # Pulse annotation for EP conditions: median pulse time as
            # vertical dashed line, IQR as pale band.
            if cond_key.startswith('EP_'):
                pts = per_cond_pulse[cond_key]
                if pts:
                    med = float(np.median(pts))
                    q1  = float(np.percentile(pts, 25))
                    q3  = float(np.percentile(pts, 75))
                    ax.axvspan(q1, q3, color=cond_edge[cond_key],
                               alpha=0.08, zorder=0)
                    ax.axvline(med, color=cond_edge[cond_key],
                               ls='--', lw=0.9, alpha=0.7, zorder=1)

        ax.set_title(region, fontsize=10)
        ax.set_xlabel("Time from aspiration onset (s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(r"Volume-normalised uptake" "\n" r"$(I(t) - I_{0})\,/\,V(t)$  (ADU/µm³)")
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')

    plt.tight_layout()
    utils.save_plot_pdf(output_dir / "Thesis_WT_MeanUptakeTrace.pdf")
    plt.close()
    logger.info("Mean WT uptake trace written (ASP / 100us / 5ms).")


def run_thesis_claim2_uptake_actin_plots(grouped_data: Dict,
                                          mechanics_df: pd.DataFrame,
                                          output_dir: Path,
                                          output_dir_map: Optional[Dict[str, Path]] = None) -> None:
    """
    Runner for the EP-uptake / actin figures.

    Parameters
    ----------
    output_dir : Path
        Default directory used for combined figures (uptake trace,
        paired actin pre/post boxplot) and as the fallback target for
        the per-duration around-pulse figures.
    output_dir_map : dict[str, Path], optional
        Mapping from pulse duration label to the target directory for
        the per-duration around-pulse figures. When provided,
        ``Thesis_EP_Actin_Around_Pulse_{dur}.pdf`` and
        ``Thesis_EP_Protrusion_Around_Pulse_{dur}.pdf`` are written to
        ``output_dir_map[dur]`` for each ``dur`` in the map.
    """
    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Claim 2 uptake/actin plot '{label}' failed: {e}",
                         exc_info=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    _try("Mean WT uptake trace",
         plot_thesis_mean_uptake_trace_wt,
         grouped_data, mechanics_df, output_dir)
    _try("Paired actin pre/post",
         plot_thesis_actin_prepulse_vs_postpulse_paired,
         mechanics_df, output_dir)
    _try("Actin around pulse trace",
         plot_thesis_actin_around_pulse_trace,
         grouped_data, mechanics_df, output_dir, output_dir_map)
    _try("Protrusion around pulse trace",
         plot_thesis_protrusion_around_pulse,
         grouped_data, mechanics_df, output_dir, output_dir_map)
    _try("Actin by fate (paired + around-pulse)",
         register_by_fate_actin_plots,
         grouped_data, mechanics_df, output_dir, output_dir_map)