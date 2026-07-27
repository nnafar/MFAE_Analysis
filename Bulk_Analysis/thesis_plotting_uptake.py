# -*- coding: utf-8 -*-
"""
Thesis plotting for volume-normalised dye uptake.

Three top-level figures called from `master_bulk_thesis.py`:

    plot_thesis_mean_uptake_body_prot
        Two-panel mean +/- IQR band across cells, aligned on cell entry.
        Panels: body (left) and protrusion (right).  Conditions
        (ASP / 100us / 5ms) overlaid on each panel using the WT ramp.

    plot_thesis_uptake_amplitude_per_trap
        Two-panel scatter of the fitted plateau A vs Trap_ID (1-18).
        Panels: body and protrusion.  Filtered to
        Uptake_*_VolNorm_R2_Flag == True.  Coloured by condition,
        marker shape by fate.

    plot_thesis_prepulse_correlations
        Correlation matrix.  Rows: PrePulse_E_Pa, PrePulse_eta1_Pa_s,
        PrePulse_Tau_s.  Columns: Uptake_Body_VolNorm_A,
        Uptake_Prot_VolNorm_A, Max_Prot_length_PrePulse_um.
        Spearman rho and p annotated per panel.  Filtered to
        PrePulse_Visco_R2_Flag == True, and additionally to
        Uptake_*_VolNorm_R2_Flag == True for the uptake columns.

A single `register_uptake_plots` runner is exposed so `master_bulk_thesis`
can call one function to produce all three figures with matched
error-swallowing behaviour, mirroring `register_by_fate_actin_plots`.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

import bulk_file_handling as bfh
import bulk_utils as utils

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Visual grammar for uptake figures.
#
# Palette follows the Chapter 3 WT ramp defined in `bulk_utils`:
#     ASP    -> #1a243d  (deepest)
#     5ms    -> #1065AB  (mid)
#     100us  -> #5DA5D5  (lightest)
#
# Marker shape convention (from user memory):
#     intact         -> filled circle
#     ruptured_post  -> open circle (facecolors='none', coloured edge)
#
# Condition ordering used consistently across all three plots below.
# --------------------------------------------------------------------- #
CONDITION_ORDER = ('ASP', '5ms', '100us')
CONDITION_LABEL = {
    'ASP'  : 'ASP',
    '5ms'  : 'EP 5 ms',
    '100us': 'EP 100 µs',
}


def _condition_key(row: pd.Series) -> Optional[str]:
    """
    Bucket a mechanics_df row into one of ASP / 5ms / 100us for the
    uptake plots.  Rows that don't fit any bucket return None and are
    skipped upstream.  EP fate (intact vs ruptured_post) is preserved
    alongside via the 'Fate_Status' column and handled at the marker
    level, not by splitting into more buckets.
    """
    ct = row.get('Condition_Type')
    if ct == 'ASP':
        return 'ASP'
    if ct == 'EP':
        dl = row.get('Duration_label')
        if dl in ('5ms', '100us'):
            return dl
    return None


def _condition_color(key: str, treatment: str = 'WT') -> str:
    """Delegate to the shared style API; fall back on grey."""
    try:
        return utils.get_style_color(treatment, key)
    except Exception:
        # Hard-coded fallback matching the WT ramp.
        return {'ASP': '#1a243d', '5ms': '#1065AB', '100us': '#5DA5D5'
                }.get(key, '#808080')


# ===================================================================== #
# 1. Mean uptake trace (body + protrusion, two panels)                  #
# ===================================================================== #
def plot_thesis_mean_uptake_body_prot(grouped_data: Dict,
                                       mechanics_df: pd.DataFrame,
                                       output_dir: Path,
                                       treatment: str = 'WT',
                                       n_time_bins: int = 200,
                                       min_frac_contributing: float = 0.25,
                                       require_mi_flag: bool = False,
                                       require_prepulse_flag: bool = False,
                                       filename_suffix: str = ''
                                       ) -> None:
    """
    Mean volume-normalised uptake vs time-since-pulse, body and
    protrusion, one condition ramp per panel.

    Alignment: t = 0 at the pulse frame (`meta.pulse_frame`).  Traces
    are shifted so that the pulse sits at t = 0, and only the
    post-pulse portion (t >= 0) is plotted — the pre-pulse baseline
    is already subtracted upstream by the fitter, so the pre-pulse
    curve carries no biological information.

    Cohort: EP cells only (ASP has no pulse — omitted here; the
    ASP "no-pulse" control belongs in a separate figure).  Cells kept
    are `treatment` cells (default 'WT') with EP intact or
    ruptured_post fate; ruptured cells contribute to the mean up
    until the frame they exit.

    Aggregation: each trap's post-pulse trace is interpolated onto a
    shared time grid running from 0 to the 95th percentile of
    per-cell post-pulse ends, split into `n_time_bins` bins.
    Per-timepoint statistics are the median and interquartile range
    across cells; time bins with fewer than
    `min_frac_contributing * n_cells` contributing cells at that bin
    are dropped from the plotted x-range to prevent tail artefacts
    driven by a couple of long traces.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Mean uptake trace skipped: empty mechanics_df.")
        return

    # ---- Cohort membership (row-level) ------------------------------
    # EP cells only: aligning on pulse_frame requires a pulse to exist.
    # ASP cells (no pulse) belong in a separate no-pulse control figure.
    mask = (
        (mechanics_df['Treatment'] == treatment)
        & (mechanics_df['Condition_Type'] == 'EP')
        & mechanics_df['Fate_Status'].isin(['intact', 'ruptured_post'])
    )
    filter_labels = ['EP', 'intact|ruptured_post']
    if require_mi_flag:
        mask = mask & mechanics_df['MI_Whole_R2_Flag'].fillna(False).astype(bool)
        filter_labels.append('MI_Whole_R2_Flag')
    if require_prepulse_flag:
        mask = mask & mechanics_df['PrePulse_Visco_R2_Flag'].fillna(False).astype(bool)
        filter_labels.append('PrePulse_Visco_R2_Flag')
    ok = mechanics_df.loc[mask]
    filter_desc = ' + '.join(filter_labels)
    accepted = set(zip(ok['Experiment_Folder'], ok['Trap_ID']))
    if not accepted:
        logger.warning("Mean uptake trace skipped: no accepted EP cells for "
                       f"treatment={treatment} (filters=[{filter_desc}]).")
        return
    logger.info(f"  Mean uptake trace cohort ({treatment}): "
                f"n={len(accepted)}  filters=[{filter_desc}]")

    # ---- Collect per-cell traces per condition ----------------------
    # Only EP buckets are relevant here — pulse alignment requires a pulse.
    ep_conditions = tuple(c for c in CONDITION_ORDER if c != 'ASP')

    region_col = {'Body': 'Body_VolNorm', 'Prot': 'Protrusion_VolNorm'}
    traces: Dict[str, Dict[str, list]] = {
        c: {'Body': [], 'Prot': []} for c in ep_conditions
    }
    all_post_ends: list = []   # per-cell most-positive time (post-pulse extent)

    row_lookup = {
        (r['Experiment_Folder'], r['Trap_ID']): r
        for _, r in ok.iterrows()
    }

    for gk, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            meta = trap.metadata
            if meta.treatment != treatment:
                continue
            key = (meta.full_path.name, trap.trap_id)
            if key not in accepted:
                continue

            row = row_lookup.get(key)
            if row is None:
                continue
            cond_key = _condition_key(row)
            if cond_key is None or cond_key not in ep_conditions:
                continue

            ud = getattr(trap, 'uptake_data', {}) or {}
            if 'Time_s' not in ud:
                continue
            t_raw = np.asarray(ud['Time_s'], dtype=float)
            if len(t_raw) < 5:
                continue

            # Pulse alignment: t = 0 at the pulse frame.  Skip cells
            # where pulse_frame is missing or out of range.
            pf = getattr(meta, 'pulse_frame', None)
            if pf is None or not (0 <= int(pf) < len(t_raw)):
                continue
            t_pulse_aligned = t_raw - t_raw[int(pf)]

            for region, col in region_col.items():
                if col not in ud:
                    continue
                y = np.asarray(ud[col], dtype=float)
                if len(y) != len(t_pulse_aligned):
                    continue
                # Restrict to post-pulse (t >= 0) — the pre-pulse
                # baseline carries no biological signal here.
                m = (np.isfinite(t_pulse_aligned)
                     & np.isfinite(y)
                     & (t_pulse_aligned >= 0))
                if m.sum() < 5:
                    continue
                t_ok = t_pulse_aligned[m]
                y_ok = y[m]
                traces[cond_key][region].append((t_ok, y_ok))
                all_post_ends.append(float(t_ok[-1]))  # most positive

    # Bail if nothing to plot.
    total_traces = sum(len(v[r]) for v in traces.values() for r in v)
    if total_traces == 0:
        logger.warning("Mean uptake trace skipped: no traces collected.")
        return

    # ---- Common time grid -------------------------------------------
    # Post-pulse only: from 0 to the 95th-percentile of per-cell
    # post-pulse ends, so a couple of very long-tracked cells don't
    # stretch the axis.  The `min_frac_contributing` trim below drops
    # bins that fall past the point where too few cells remain.
    t_max = float(np.percentile(all_post_ends, 95)) if all_post_ends else 0.0
    if t_max <= 0:
        logger.warning("Mean uptake trace skipped: time grid collapsed.")
        return
    t_grid = np.linspace(0.0, t_max, n_time_bins)

    # ---- Plot -------------------------------------------------------
    utils.set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    region_titles = {'Body': 'Cell body', 'Prot': 'Protrusion'}

    for ax, region in zip(axes, ('Body', 'Prot')):
        for cond_key in ep_conditions:
            trace_list = traces[cond_key][region]
            if not trace_list:
                continue
            color = _condition_color(cond_key, treatment)

            # Interpolate each trace onto the common grid, NaN outside
            # the trace's own time range so short traces don't get
            # extrapolated.
            Y = np.full((len(trace_list), n_time_bins), np.nan)
            for i, (t, y) in enumerate(trace_list):
                # Sort by time defensively — interp1d rejects unsorted x
                # unless assume_sorted is set correctly, and per-cell
                # trace ordering is already ascending here.
                order = np.argsort(t)
                f = interp1d(t[order], y[order], bounds_error=False,
                             fill_value=np.nan, assume_sorted=True)
                Y[i, :] = f(t_grid)

            n_contributing = np.sum(~np.isnan(Y), axis=0)
            keep = n_contributing >= max(1, int(min_frac_contributing * len(trace_list)))
            if not keep.any():
                continue

            # Suppress the "All-NaN slice" warning for the bins outside
            # the keep mask — we drop them from the plot anyway.
            with np.errstate(invalid='ignore'), \
                 __import__('warnings').catch_warnings():
                __import__('warnings').filterwarnings(
                    'ignore', r'All-NaN slice encountered')
                med = np.nanmedian(Y, axis=0)
                q25 = np.nanpercentile(Y, 25, axis=0)
                q75 = np.nanpercentile(Y, 75, axis=0)

            t_plot = t_grid[keep]
            ax.fill_between(t_plot, q25[keep], q75[keep],
                            color=color, alpha=0.20, linewidth=0)
            ax.plot(t_plot, med[keep], color=color, linewidth=1.8,
                    label=f"{CONDITION_LABEL[cond_key]} (n={len(trace_list)})")

        ax.set_xlabel("Time since pulse (s)")
        ax.set_title(region_titles[region])
        ax.set_xlim(left=0.0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, loc='best', fontsize=9)

    axes[0].set_ylabel(r"Uptake (a.u. / $\mathrm{\mu m}^{3}$)")
    fig.suptitle(f"{treatment} — mean volume-normalised uptake, "
                 f"pulse-aligned (median $\\pm$ IQR)  [{filter_desc}]",
                 y=1.02, fontweight='bold')
    fig.tight_layout()
    out = (Path(output_dir)
           / f"Thesis_Uptake_Mean_Trace_BodyProt_{treatment}{filename_suffix}.pdf")
    utils.save_plot_pdf(out)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ===================================================================== #
# 2. Fitted amplitude vs Trap_ID (scatter)                              #
# ===================================================================== #
def plot_thesis_uptake_amplitude_per_trap(mechanics_df: pd.DataFrame,
                                            output_dir: Path,
                                            treatment: str = 'WT',
                                            require_mi_flag: bool = False,
                                            require_prepulse_flag: bool = False,
                                            filename_suffix: str = ''
                                            ) -> None:
    """
    Two-panel scatter of the fitted uptake plateau A vs Trap_ID.

    Panels: body (left) and protrusion (right).  Cells filtered per
    panel by `Uptake_{region}_VolNorm_R2_Flag == True`, so a cell that
    passed the body fit but failed the protrusion fit still contributes
    to the body panel.  Additional cohort gates via `require_mi_flag`
    and `require_prepulse_flag` are applied before the per-region
    uptake-R² gate.

    Colour: condition (ASP / 5ms / 100us) from the WT ramp.
    Marker: fate (intact = filled circle, ruptured_post = open circle).

    A Spearman rho + p across all cells in each panel is annotated
    top-left; per-condition rho values are omitted to keep the panel
    readable (they belong in a supplementary table if needed).
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Uptake-A per-trap skipped: empty mechanics_df.")
        return

    df = mechanics_df[mechanics_df['Treatment'] == treatment].copy()
    if df.empty:
        logger.warning(f"Uptake-A per-trap skipped: no {treatment} rows.")
        return

    # ---- Additional cohort gates -------------------------------------
    filter_labels = [f'{treatment}']
    if require_mi_flag:
        df = df[df['MI_Whole_R2_Flag'].fillna(False).astype(bool)]
        filter_labels.append('MI_Whole_R2_Flag')
    if require_prepulse_flag:
        df = df[df['PrePulse_Visco_R2_Flag'].fillna(False).astype(bool)]
        filter_labels.append('PrePulse_Visco_R2_Flag')
    filter_desc = ' + '.join(filter_labels)
    if df.empty:
        logger.warning(f"Uptake-A per-trap skipped: no rows survive filters "
                       f"[{filter_desc}].")
        return

    df['Cond_Key'] = df.apply(_condition_key, axis=1)
    df = df[df['Cond_Key'].notna()].copy()

    if df.empty:
        logger.warning("Uptake-A per-trap skipped: no cells match "
                       "ASP/5ms/100us buckets.")
        return

    utils.set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    region_spec = {
        'Body': ('Uptake_Body_VolNorm_A', 'Uptake_Body_VolNorm_R2_Flag',
                 'Body uptake plateau'),
        'Prot': ('Uptake_Prot_VolNorm_A', 'Uptake_Prot_VolNorm_R2_Flag',
                 'Protrusion uptake plateau'),
    }
    rng = np.random.default_rng(seed=42)

    for ax, region in zip(axes, ('Body', 'Prot')):
        a_col, flag_col, ylab = region_spec[region]
        keep = df[flag_col].fillna(False).astype(bool) & df[a_col].notna()
        sub = df[keep].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No cells pass R² ≥ 0.85",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(f"{region}  (n = 0)")
            continue

        for cond_key in CONDITION_ORDER:
            for fate, marker_kwargs in (
                ('intact',
                 dict(marker='o', facecolor=_condition_color(cond_key, treatment),
                      edgecolor='white', linewidths=0.5)),
                ('ruptured_post',
                 dict(marker='o', facecolor='none',
                      edgecolor=_condition_color(cond_key, treatment),
                      linewidths=1.4)),
            ):
                pts = sub[(sub['Cond_Key'] == cond_key)
                          & (sub['Fate_Status'] == fate)]
                if pts.empty:
                    continue
                x = pts['Trap_ID'].to_numpy(dtype=float)
                x = x + rng.uniform(-0.25, 0.25, size=len(x))
                y = pts[a_col].to_numpy(dtype=float)
                ax.scatter(x, y, s=40, alpha=0.75,
                           label=f"{CONDITION_LABEL[cond_key]} ({fate})",
                           **marker_kwargs)

        # Spearman across all points in this panel.
        x_all = sub['Trap_ID'].to_numpy(dtype=float)
        y_all = sub[a_col].to_numpy(dtype=float)
        m = np.isfinite(x_all) & np.isfinite(y_all)
        if m.sum() >= 3:
            rho, p = stats.spearmanr(x_all[m], y_all[m])
            p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
            ax.text(0.03, 0.97,
                    rf"$\rho$ = {rho:.2f}" + "\n" + p_str,
                    transform=ax.transAxes, ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', alpha=0.85, ec='gray'))

        ax.set_xlabel("Trap ID (1 = far, 18 = near)")
        ax.set_ylabel(ylab + r"  (a.u. / $\mathrm{\mu m}^{3}$)")
        ax.set_yscale('log')
        ax.set_xticks(range(2, 19, 2))
        ax.set_xlim(0, 19)
        ax.set_title(f"{region}  (n = {len(sub)})")
        ax.spines[['top', 'right']].set_visible(False)

    # One shared legend below the two panels — the per-axis legends
    # would double the marker count.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)

    fig.suptitle(f"{treatment} — uptake plateau A vs trap position  "
                 f"[{filter_desc}]",
                 y=1.02, fontweight='bold')
    fig.tight_layout()
    out = (Path(output_dir)
           / f"Thesis_Uptake_Amplitude_Per_Trap_{treatment}{filename_suffix}.pdf")
    utils.save_plot_pdf(out)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ===================================================================== #
# 3. Pre-pulse mechanics vs uptake / protrusion length                  #
# ===================================================================== #
def plot_thesis_prepulse_correlations(mechanics_df: pd.DataFrame,
                                        output_dir: Path,
                                        treatment: str = 'WT'
                                        ) -> None:
    """
    3x2 matrix of scatter plots — pre-pulse viscoelastic parameters
    vs uptake plateaus.  Uptake ↔ protrusion length is covered by a
    separate dedicated figure (`plot_thesis_uptake_vs_prot_length`),
    where the mechanics filter isn't required.

    Rows (pre-pulse mechanics):
        PrePulse_E_Pa           elastic modulus
        PrePulse_eta1_Pa_s      short-time viscosity
        PrePulse_Tau_s          relaxation time

    Columns:
        Uptake_Body_VolNorm_A   body plateau (EP cells only)
        Uptake_Prot_VolNorm_A   protrusion plateau (EP cells only)

    Cohort:
        - Mechanics y-axis always requires PrePulse_Visco_R2_Flag == True.
        - Uptake x-axis additionally requires the matching
          Uptake_{region}_VolNorm_R2_Flag == True and Condition_Type == 'EP'.
          (ASP cells have no pulse to drive PI uptake, so any signal
          there would be rupture — a categorical, not a continuous,
          readout.)

    X-axis is log for both uptake amplitude columns (multi-decade
    spread).  Spearman rho + p annotated per panel.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Pre-pulse correlations skipped: empty mechanics_df.")
        return

    df = mechanics_df[mechanics_df['Treatment'] == treatment].copy()
    if df.empty:
        logger.warning(f"Pre-pulse correlations skipped: no {treatment} rows.")
        return

    df = df[df['PrePulse_Visco_R2_Flag'].fillna(False).astype(bool)].copy()
    if df.empty:
        logger.warning("Pre-pulse correlations skipped: no cells pass "
                       "PrePulse_Visco_R2_Flag.")
        return

    mech_rows = [
        ('PrePulse_E_Pa',      r"$E$ (Pa)",              True),
        ('PrePulse_eta1_Pa_s', r"$\eta_{1}$ (Pa$\cdot$s)", True),
        ('PrePulse_Tau_s',     r"$\tau$ (s)",            True),
    ]
    # (col, label, R2_flag_col, cohort, x_log)
    # x_log: True for the fitted uptake plateaus (multi-decade range,
    # dominated by a handful of large-A cells on linear scale).  Kept
    # linear for max protrusion length (5–35 um, no decade spread).
    outcome_cols = [
        ('Uptake_Body_VolNorm_A',
         r"Body uptake $A$ (a.u./$\mathrm{\mu m}^{3}$)",
         'Uptake_Body_VolNorm_R2_Flag', 'EP_only', True),
        ('Uptake_Prot_VolNorm_A',
         r"Protrusion uptake $A$ (a.u./$\mathrm{\mu m}^{3}$)",
         'Uptake_Prot_VolNorm_R2_Flag', 'EP_only', True),
    ]

    utils.set_paper_style()
    fig, axes = plt.subplots(len(mech_rows), len(outcome_cols),
                             figsize=(8, 10), sharey='row')

    for i, (mcol, mlab, mlog) in enumerate(mech_rows):
        for j, (ycol, ylab, yflag, cohort, xlog) in enumerate(outcome_cols):
            ax = axes[i, j]

            sub = df.copy()
            if cohort == 'EP_only':
                sub = sub[sub['Condition_Type'] == 'EP']
            if yflag is not None:
                sub = sub[sub[yflag].fillna(False).astype(bool)]
            sub = sub[[mcol, ycol, 'Cell_Type', 'Fate_Status',
                       'Condition_Type', 'Duration_label']].dropna()

            if len(sub) < 3:
                ax.text(0.5, 0.5, f"n = {len(sub)}",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                if i == len(mech_rows) - 1:
                    ax.set_xlabel(ylab)
                if j == 0:
                    ax.set_ylabel(mlab)
                continue

            # Points coloured by condition (ASP / 5ms / 100us).
            for cond_key in CONDITION_ORDER:
                pts = sub[sub.apply(_condition_key, axis=1) == cond_key]
                if pts.empty:
                    continue
                color = _condition_color(cond_key, treatment)
                intact = pts[pts['Fate_Status'] == 'intact']
                rupt   = pts[pts['Fate_Status'] == 'ruptured_post']
                if not intact.empty:
                    ax.scatter(intact[ycol], intact[mcol], s=28,
                               facecolor=color, edgecolor='white',
                               linewidths=0.4, alpha=0.75)
                if not rupt.empty:
                    ax.scatter(rupt[ycol], rupt[mcol], s=28,
                               facecolor='none', edgecolor=color,
                               linewidths=1.3, alpha=0.85)

            # Spearman across all points in the panel.
            x_all = sub[ycol].to_numpy(dtype=float)
            y_all = sub[mcol].to_numpy(dtype=float)
            m = np.isfinite(x_all) & np.isfinite(y_all)
            if m.sum() >= 3:
                rho, p = stats.spearmanr(x_all[m], y_all[m])
                p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
                ax.text(0.03, 0.97,
                        rf"$\rho$ = {rho:.2f}" + "\n" + p_str +
                        f"\nn = {int(m.sum())}",
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.25',
                                  fc='white', alpha=0.85, ec='gray'))

            if mlog:
                ax.set_yscale('log')
            if xlog:
                # Log x-axis for the fitted uptake plateaus.  Any
                # non-positive A values would be discarded by matplotlib's
                # log axis with a UserWarning; the fit's bounds are
                # [0, inf) so A can be exactly 0 for a failed rise, but
                # those cases are already excluded by the R2_Flag
                # filter above.
                ax.set_xscale('log')
            ax.spines[['top', 'right']].set_visible(False)

            if i == len(mech_rows) - 1:
                ax.set_xlabel(ylab)
            if j == 0:
                ax.set_ylabel(mlab)

    fig.suptitle(f"{treatment} — pre-pulse mechanics vs uptake and protrusion length",
                 y=1.00, fontweight='bold')
    fig.tight_layout()
    out = Path(output_dir) / f"Thesis_PrePulse_Mechanics_Correlations_{treatment}.pdf"
    utils.save_plot_pdf(out)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ===================================================================== #
# 3b. MI whole-trace mechanics vs uptake / protrusion length            #
# ===================================================================== #
def plot_thesis_mi_whole_correlations(mechanics_df: pd.DataFrame,
                                        output_dir: Path,
                                        treatment: str = 'WT'
                                        ) -> None:
    """
    2x2 matrix of scatter plots — whole-trace MI descriptors vs
    uptake plateaus.  Analogue of `plot_thesis_prepulse_correlations`
    but with model-independent whole-trace parameters on the y-axis.
    Uptake ↔ protrusion length is covered by a separate figure
    (`plot_thesis_uptake_vs_prot_length`).

    Rows (MI whole-trace descriptors):
        MI_Whole_Linear_Slope   creep rate from the whole-trace
                                linear fit (applied to every MI-passing
                                cell regardless of AICc winner; see note)
        MI_Whole_PL_a           power-law amplitude, restricted to
                                cells where the power-law was the AICc
                                winner (`MI_Whole_Best_Model == 'Power-Law'`)

    Columns:
        Uptake_Body_VolNorm_A   body plateau (EP cells only)
        Uptake_Prot_VolNorm_A   protrusion plateau (EP cells only)

    Cohort:
        - All rows require `MI_Whole_R2_Flag == True`.
        - Uptake columns additionally require the matching
          `Uptake_{region}_VolNorm_R2_Flag == True` and Condition_Type == 'EP'.
        - PL_a row further requires `MI_Whole_Best_Model == 'Power-Law'`.

    Note on the slope row:
        Linear_Slope is always computed for every cell in the pipeline
        (it just isn't necessarily the AICc-selected model).  Using it
        for all MI-passing cells matches how the characteristic creep
        rate is discussed in §3.8 of the thesis and gives a single
        comparable descriptor across the cohort.

    X-axis: log for both uptake amplitude columns.

    Spearman rho + p annotated per panel.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("MI-whole correlations skipped: empty mechanics_df.")
        return

    df = mechanics_df[mechanics_df['Treatment'] == treatment].copy()
    if df.empty:
        logger.warning(f"MI-whole correlations skipped: no {treatment} rows.")
        return

    df = df[df['MI_Whole_R2_Flag'].fillna(False).astype(bool)].copy()
    if df.empty:
        logger.warning("MI-whole correlations skipped: no cells pass "
                       "MI_Whole_R2_Flag.")
        return

    mech_rows = [
        ('MI_Whole_Linear_Slope', r"Whole-trace slope ($\mathrm{\mu m}$/s)",
         False,  # y-log
         None),  # extra per-row filter
        ('MI_Whole_PL_a',         r"Power-law amplitude $a$",
         True,   # y-log — a spans decades
         ('MI_Whole_Best_Model', 'Power-Law')),
    ]
    outcome_cols = [
        ('Uptake_Body_VolNorm_A',
         r"Body uptake $A$ (a.u./$\mathrm{\mu m}^{3}$)",
         'Uptake_Body_VolNorm_R2_Flag', 'EP_only', True),
        ('Uptake_Prot_VolNorm_A',
         r"Protrusion uptake $A$ (a.u./$\mathrm{\mu m}^{3}$)",
         'Uptake_Prot_VolNorm_R2_Flag', 'EP_only', True),
    ]

    utils.set_paper_style()
    fig, axes = plt.subplots(len(mech_rows), len(outcome_cols),
                             figsize=(8, 7), sharey='row')

    for i, (mcol, mlab, mlog, row_extra) in enumerate(mech_rows):
        for j, (ycol, ylab, yflag, cohort, xlog) in enumerate(outcome_cols):
            ax = axes[i, j]

            sub = df.copy()
            if cohort == 'EP_only':
                sub = sub[sub['Condition_Type'] == 'EP']
            if yflag is not None:
                sub = sub[sub[yflag].fillna(False).astype(bool)]
            if row_extra is not None:
                col_name, val = row_extra
                sub = sub[sub[col_name] == val]
            sub = sub[[mcol, ycol, 'Cell_Type', 'Fate_Status',
                       'Condition_Type', 'Duration_label']].dropna()

            if len(sub) < 3:
                ax.text(0.5, 0.5, f"n = {len(sub)}",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                if i == len(mech_rows) - 1:
                    ax.set_xlabel(ylab)
                if j == 0:
                    ax.set_ylabel(mlab)
                continue

            # Points coloured by condition (5ms / 100us); ASP shouldn't
            # appear for EP-only columns but the prot-length column
            # includes ASP cells too if they pass MI_Whole_R2_Flag.
            for cond_key in CONDITION_ORDER:
                pts = sub[sub.apply(_condition_key, axis=1) == cond_key]
                if pts.empty:
                    continue
                color = _condition_color(cond_key, treatment)
                intact = pts[pts['Fate_Status'] == 'intact']
                rupt   = pts[pts['Fate_Status'] == 'ruptured_post']
                if not intact.empty:
                    ax.scatter(intact[ycol], intact[mcol], s=28,
                               facecolor=color, edgecolor='white',
                               linewidths=0.4, alpha=0.75)
                if not rupt.empty:
                    ax.scatter(rupt[ycol], rupt[mcol], s=28,
                               facecolor='none', edgecolor=color,
                               linewidths=1.3, alpha=0.85)

            # Spearman across all points in the panel.
            x_all = sub[ycol].to_numpy(dtype=float)
            y_all = sub[mcol].to_numpy(dtype=float)
            m = np.isfinite(x_all) & np.isfinite(y_all)
            if m.sum() >= 3:
                rho, p = stats.spearmanr(x_all[m], y_all[m])
                p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
                ax.text(0.03, 0.97,
                        rf"$\rho$ = {rho:.2f}" + "\n" + p_str +
                        f"\nn = {int(m.sum())}",
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.25',
                                  fc='white', alpha=0.85, ec='gray'))

            if mlog:
                ax.set_yscale('log')
            if xlog:
                ax.set_xscale('log')
            ax.spines[['top', 'right']].set_visible(False)

            if i == len(mech_rows) - 1:
                ax.set_xlabel(ylab)
            if j == 0:
                ax.set_ylabel(mlab)

    fig.suptitle(f"{treatment} — whole-trace MI descriptors vs uptake "
                 "and protrusion length",
                 y=1.00, fontweight='bold')
    fig.tight_layout()
    out = Path(output_dir) / f"Thesis_MI_Whole_Mechanics_Correlations_{treatment}.pdf"
    utils.save_plot_pdf(out)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ===================================================================== #
# 3c. Uptake amplitude vs pre-pulse protrusion length                   #
# ===================================================================== #
def plot_thesis_uptake_vs_prot_length(mechanics_df: pd.DataFrame,
                                        output_dir: Path,
                                        treatment: str = 'WT'
                                        ) -> None:
    """
    Two-panel scatter of uptake plateau A vs a pre-pulse geometric
    descriptor. Body panel (left): body-region uptake plateau A vs
    pre-pulse cell body volume. Protrusion panel (right):
    protrusion-region uptake plateau A vs max pre-pulse protrusion
    length.

    The body panel asks whether cell size at loading predicts the
    post-pulse body uptake plateau (Section 1.2.11). The protrusion
    panel asks whether a longer pre-pulse protrusion, which sits
    further inside the concentrated intra-channel field, predicts a
    larger post-pulse protrusion uptake plateau.

    Cohort, pooled across pulse duration (5 ms and 100 us are not
    split into separate panels or separate statistics):
        - EP cells of `treatment` with intact fate only.
          Ruptured_post cells are excluded: their protrusion and
          uptake trajectories are truncated by rupture, so the
          fitted geometry and uptake plateau do not describe the
          same intact membrane-cortex composite as the rest of the
          cohort.
        - Per-panel: `Uptake_{region}_VolNorm_R2_Flag == True` for
          the region shown in that panel. A cell that passed the
          body fit but failed the protrusion fit contributes to the
          body panel only.
        - Per-panel: `Uptake_{region}_VolNorm_A < 100` AND
          `Uptake_{region}_VolNorm_tau <= 20000` (s) jointly exclude the
          runaway mono-exponential fits described in Section 1.2.10.
          These are an identifiability failure of the fit: a near-linear
          rise over the recording window is equally well described by
          a small amplitude/moderate tau or a large amplitude/tau in
          the hundreds-of-thousands-to-millions-of-seconds range, so
          either branch can surface. The amplitude ceiling alone does
          not catch every case: a small number of fits land with a
          "sensible"-looking A (< 100) but a tau of order 1e5-1e6 s,
          i.e. days, which is just as unresolved given a recording
          window on the order of ~350 s post-pulse. The 20000 s
          ceiling sits in a clean gap in the pooled EP-intact data
          (resolved fits top out under ~13000 s; the leaking fits sit
          at 2.6-3.4e5 s), so no genuinely resolved fit is cut. No
          per-cell value from either failure branch belongs in a
          correlation.

    X-axis: log for uptake amplitude (multi-decade spread).
    Y-axis: linear (body volume in µm³, protrusion length in µm).
    Colour: condition (5ms / 100us) from the WT ramp.

    Spearman rho + p annotated per panel across all pooled points.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Uptake-vs-protlength skipped: empty mechanics_df.")
        return

    RUNAWAY_A_THRESHOLD = 100.0
    RUNAWAY_TAU_THRESHOLD_S = 20000.0

    df = mechanics_df[
        (mechanics_df['Treatment'] == treatment)
        & (mechanics_df['Condition_Type'] == 'EP')
        & (mechanics_df['Fate_Status'] == 'intact')
    ].copy()
    if df.empty:
        logger.warning("Uptake-vs-protlength skipped: no intact EP cells "
                       f"for treatment={treatment}.")
        return

    region_spec = {
        'Body': ('Uptake_Body_VolNorm_A', 'Uptake_Body_VolNorm_tau',
                 'Uptake_Body_VolNorm_R2_Flag',
                 'Body uptake plateau', 'Cell_Body_Volume_PrePulse_um3',
                 r"Body volume ($\mathrm{\mu m}^{3}$, pre-pulse)"),
        'Prot': ('Uptake_Prot_VolNorm_A', 'Uptake_Prot_VolNorm_tau',
                 'Uptake_Prot_VolNorm_R2_Flag',
                 'Protrusion uptake plateau', 'Max_Prot_length_PrePulse_um',
                 r"Max protrusion length ($\mathrm{\mu m}$)"),
    }

    utils.set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, region in zip(axes, ('Body', 'Prot')):
        a_col, tau_col, flag_col, xlab, y_col, ylab = region_spec[region]
        keep = (df[flag_col].fillna(False).astype(bool)
                & df[a_col].notna()
                & (df[a_col] < RUNAWAY_A_THRESHOLD)
                & df[tau_col].notna()
                & (df[tau_col] <= RUNAWAY_TAU_THRESHOLD_S)
                & df[y_col].notna())
        sub = df[keep].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No cells pass R² ≥ 0.85",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(f"{region}  (n = 0)")
            continue

        sub['Cond_Key'] = sub.apply(_condition_key, axis=1)

        for cond_key in CONDITION_ORDER:
            pts = sub[sub['Cond_Key'] == cond_key]
            if pts.empty:
                continue
            x = pts[a_col].to_numpy(dtype=float)
            y = pts[y_col].to_numpy(dtype=float)
            ax.scatter(x, y, s=40, alpha=0.75, marker='o',
                       facecolor=_condition_color(cond_key, treatment),
                       edgecolor='white', linewidths=0.5,
                       label=f"{CONDITION_LABEL[cond_key]} (n={len(pts)})")

        # Spearman across all pooled points (both durations together).
        x_all = sub[a_col].to_numpy(dtype=float)
        y_all = sub[y_col].to_numpy(dtype=float)
        m = np.isfinite(x_all) & np.isfinite(y_all)
        if m.sum() >= 3:
            rho, p = stats.spearmanr(x_all[m], y_all[m])
            p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
            ax.text(0.03, 0.97,
                    rf"$\rho$ = {rho:.2f}" + "\n" + p_str +
                    f"\nn = {int(m.sum())}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', alpha=0.85, ec='gray'))

        ax.set_xlabel(xlab + r"  (a.u. / $\mathrm{\mu m}^{3}$)")
        ax.set_xscale('log')
        ax.set_ylabel(ylab)
        ax.set_title(f"{region}  (n = {len(sub)})")
        ax.spines[['top', 'right']].set_visible(False)

    # Shared legend below.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)

    fig.suptitle(f"{treatment} — uptake plateau A vs pre-pulse geometry "
                 "(intact cells only)",
                 y=1.02, fontweight='bold')
    fig.tight_layout()
    out = Path(output_dir) / f"Thesis_Uptake_vs_ProtLength_{treatment}.pdf"
    utils.save_plot_pdf(out)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ===================================================================== #
# 4. Per-cell multipanel: raw uptake + best-fit overlay                 #
# ===================================================================== #
def plot_thesis_uptake_bestfit_multipanel(grouped_data: Dict,
                                            mechanics_df: pd.DataFrame,
                                            output_dir: Path,
                                            treatment: str = 'WT',
                                            cols: int = 5,
                                            require_mi_flag: bool = True,
                                            require_prepulse_flag: bool = False,
                                            filename_suffix: str = '') -> None:
    """
    Per-cell multipanel of post-pulse uptake with fit overlays.

    Mirrors `plot_thesis_asp_best_fit_multipanel` from `thesis_plotting`:
    one figure per protocol (100us / 5ms), one subplot per cell, with
    the mono- or bi-exponential fit overlaid on the raw scatter and
    the selected model + R² annotated in the corner.

    Both regions share a subplot:
        Body       -- blue scatter + blue fit line
        Protrusion -- orange scatter + orange fit line

    Fit line style: solid = passes Uptake_*_VolNorm_R2_Flag (R² ≥ 0.85),
    dashed = fails.  The cohort filters below control WHICH cells
    appear; the uptake flag controls which fit lines are marked
    reliable within that cohort.

    Baseline handling: the pipeline fitter subtracts a pre-pulse median
    from each region before fitting.  Here we recompute the same
    baseline and add it back to the predicted post-pulse rise so the
    overlay aligns with the raw scatter (which is not baseline-
    subtracted).

    Cohort filters
    --------------
    Base cohort is EP cells of `treatment` with intact / ruptured_post
    fate.  Additional optional gates:

    require_mi_flag : bool, default True
        Restrict to cells with `MI_Whole_R2_Flag == True` — the same
        cohort used by the whole-trace MI boxplots and by
        `plot_thesis_mechanics_vs_uptake_mi`.  Default TRUE so the
        multipanel matches the analysis cohort by default.
    require_prepulse_flag : bool, default False
        Restrict to cells with `PrePulse_Visco_R2_Flag == True`.  Use
        together with `require_mi_flag` for the strictest cut — the
        cells whose pre-pulse viscoelastic parameters AND whole-trace
        MI fits are both trustworthy.

    filename_suffix : str, default ''
        Appended to the output filename before the extension so
        stricter variants land next to the default without clobbering.

    Files: Thesis_Uptake_BestFit_Panel_{treatment}_{protocol}{suffix}.pdf
    """
    import math
    import matplotlib.lines as mlines

    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Uptake best-fit multipanel skipped: empty mechanics_df.")
        return

    # ---- Cohort membership -------------------------------------------
    # Base cohort: EP cells of the requested treatment, intact or
    # ruptured_post.  Optional additional gates on the MI whole-trace
    # and pre-pulse viscoelastic R² flags are applied per the caller's
    # `require_*` arguments.
    mask = (
        (mechanics_df['Treatment'] == treatment)
        & (mechanics_df['Condition_Type'] == 'EP')
        & mechanics_df['Fate_Status'].isin(['intact', 'ruptured_post'])
    )
    filter_labels = ['EP', 'intact|ruptured_post']
    if require_mi_flag:
        mask = mask & mechanics_df['MI_Whole_R2_Flag'].fillna(False).astype(bool)
        filter_labels.append('MI_Whole_R2_Flag')
    if require_prepulse_flag:
        mask = mask & mechanics_df['PrePulse_Visco_R2_Flag'].fillna(False).astype(bool)
        filter_labels.append('PrePulse_Visco_R2_Flag')

    ok = mechanics_df.loc[mask]
    filter_desc = ' + '.join(filter_labels)
    if ok.empty:
        logger.warning("Uptake best-fit multipanel skipped: "
                       f"no cells pass filters ({filter_desc}) for "
                       f"treatment={treatment}.")
        return
    logger.info(f"  Uptake best-fit multipanel cohort ({treatment}): "
                f"n={len(ok)}  filters=[{filter_desc}]")

    row_lookup = {
        (r['Experiment_Folder'], r['Trap_ID']): r
        for _, r in ok.iterrows()
    }

    # Region visual grammar (body / protrusion).
    region_style = {
        'Body': dict(color='#1f77b4',
                     data_col='Body_VolNorm',
                     A_col='Uptake_Body_VolNorm_A',
                     tau_col='Uptake_Body_VolNorm_tau',
                     A1_col='Uptake_Body_VolNorm_A1',
                     tau1_col='Uptake_Body_VolNorm_tau1',
                     A2_col='Uptake_Body_VolNorm_A2',
                     tau2_col='Uptake_Body_VolNorm_tau2',
                     model_col='Uptake_Body_VolNorm_Best_Model',
                     r2_col='Uptake_Body_VolNorm_R2',
                     flag_col='Uptake_Body_VolNorm_R2_Flag'),
        'Prot': dict(color='#ff7f0e',
                     data_col='Protrusion_VolNorm',
                     A_col='Uptake_Prot_VolNorm_A',
                     tau_col='Uptake_Prot_VolNorm_tau',
                     A1_col='Uptake_Prot_VolNorm_A1',
                     tau1_col='Uptake_Prot_VolNorm_tau1',
                     A2_col='Uptake_Prot_VolNorm_A2',
                     tau2_col='Uptake_Prot_VolNorm_tau2',
                     model_col='Uptake_Prot_VolNorm_Best_Model',
                     r2_col='Uptake_Prot_VolNorm_R2',
                     flag_col='Uptake_Prot_VolNorm_R2_Flag'),
    }

    def _predict(t, model, A, tau, A1, tau1, A2, tau2):
        """Reconstruct fitted rise (t >= 0) from stored parameters."""
        if model == 'Mono' and A is not None and tau is not None and tau > 0:
            return A * (1.0 - np.exp(-t / tau))
        if (model == 'Bi'
                and all(v is not None and np.isfinite(v)
                        for v in (A1, tau1, A2, tau2))
                and tau1 > 0 and tau2 > 0):
            return (A1 * (1.0 - np.exp(-t / tau1))
                    + A2 * (1.0 - np.exp(-t / tau2)))
        return None

    # ---- One figure per protocol ------------------------------------
    for protocol in ('100us', '5ms'):
        # Collect (trap_key, trap, row) tuples for this protocol.
        cells = []
        for gk, traps in grouped_data.items():
            if not traps:
                continue
            for trap in traps:
                meta = trap.metadata
                if meta.treatment != treatment:
                    continue
                if meta.condition_type != 'EP':
                    continue
                if getattr(meta, 'duration_label', None) != protocol:
                    continue
                key = (meta.full_path.name, trap.trap_id)
                row = row_lookup.get(key)
                if row is None:
                    continue
                cells.append((key, trap, row))

        if not cells:
            logger.info(f"  No {treatment} {protocol} cells; "
                        "skipping best-fit multipanel.")
            continue

        # Sort by trap_id so panels flow 1,2,3,... left-to-right.
        cells.sort(key=lambda c: (c[1].trap_id, c[0][0]))

        n = len(cells)
        n_rows = math.ceil(n / cols)
        fig, axes = plt.subplots(n_rows, cols,
                                 figsize=(4.0 * cols, 3.0 * n_rows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        # Shared legend at the top.
        handles = [
            mlines.Line2D([], [], color=region_style['Body']['color'],
                          marker='o', lw=0, ms=4, alpha=0.6, label='Body data'),
            mlines.Line2D([], [], color=region_style['Body']['color'],
                          lw=2, alpha=0.9, label='Body fit'),
            mlines.Line2D([], [], color=region_style['Prot']['color'],
                          marker='o', lw=0, ms=4, alpha=0.6, label='Protrusion data'),
            mlines.Line2D([], [], color=region_style['Prot']['color'],
                          lw=2, alpha=0.9, label='Protrusion fit'),
        ]
        fig.legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(0.5, 1.00), ncol=4,
                   frameon=False, fontsize=11)

        for i, (key, trap, row) in enumerate(cells):
            ax = axes_flat[i]
            meta = trap.metadata
            exp_short = meta.full_path.name.split('_')[0]  # date prefix
            fate_tag = 'R' if row['Fate_Status'] == 'ruptured_post' else 'I'
            ax.set_title(f"Trap {trap.trap_id}  ({exp_short}, {fate_tag})",
                         fontsize=9, fontweight='bold')

            ud = getattr(trap, 'uptake_data', {}) or {}
            if 'Time_s' not in ud:
                ax.text(0.5, 0.5, "No uptake data",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='gray')
                continue

            t_raw = np.asarray(ud['Time_s'], dtype=float)
            pf = getattr(meta, 'pulse_frame', None)
            if pf is None or not (0 <= int(pf) < len(t_raw)):
                ax.text(0.5, 0.5, "No pulse frame",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='gray')
                continue
            t_aligned = t_raw - t_raw[int(pf)]
            post_mask = t_aligned >= 0

            # Annotation text lines built up across regions.
            ann_lines = []

            for region_name, style in region_style.items():
                col = style['data_col']
                if col not in ud:
                    continue
                y_full = np.asarray(ud[col], dtype=float)
                if len(y_full) != len(t_aligned):
                    continue

                # Baseline from pre-pulse (same convention as the fitter).
                pre = (t_aligned < 0) & np.isfinite(y_full)
                baseline = (float(np.nanmedian(y_full[pre]))
                            if pre.sum() >= 2 else 0.0)

                # Raw post-pulse scatter.
                m = post_mask & np.isfinite(y_full)
                if m.sum() >= 3:
                    ax.plot(t_aligned[m], y_full[m], 'o',
                            color=style['color'], ms=2.5, alpha=0.5)

                # Fit overlay (uses stored parameters — no re-fitting).
                model = row.get(style['model_col'])
                A     = row.get(style['A_col'])
                tau   = row.get(style['tau_col'])
                A1    = row.get(style['A1_col'])
                tau1  = row.get(style['tau1_col'])
                A2    = row.get(style['A2_col'])
                tau2  = row.get(style['tau2_col'])
                r2    = row.get(style['r2_col'])
                flag  = bool(row.get(style['flag_col']))

                t_max = float(np.nanmax(t_aligned[m])) if m.any() else 0.0
                if t_max > 0:
                    t_smooth = np.linspace(0.0, t_max, 200)
                    rise = _predict(t_smooth, model, A, tau, A1, tau1, A2, tau2)
                    if rise is not None:
                        # Fit line uses solid for pass, dashed for fail
                        # so the eye picks out the poor fits at a glance.
                        ls = '-' if flag else '--'
                        ax.plot(t_smooth, baseline + rise,
                                color=style['color'], lw=1.7, alpha=0.9,
                                linestyle=ls)

                # Per-region annotation line.
                mark = '✓' if flag else '✗'
                if model is None:
                    ann_lines.append(f"{region_name}: fit failed")
                else:
                    r2_str = f"R²={r2:.2f}" if (r2 is not None
                                                  and np.isfinite(r2)) else "R²=NA"
                    ann_lines.append(f"{region_name}: {model} {r2_str} {mark}")

            if ann_lines:
                ax.text(0.03, 0.97, "\n".join(ann_lines),
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.25',
                                  fc='white', alpha=0.85, ec='gray'))

            ax.set_xlim(left=0.0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7)

        # Hide unused axes.
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis('off')

        fig.text(0.5, 0.005, "Time since pulse (s)",
                 ha='center', fontsize=12)
        fig.text(0.005, 0.5, r"Uptake (a.u. / $\mathrm{\mu m}^{3}$)",
                 va='center', rotation='vertical', fontsize=12)
        fig.suptitle(f"{treatment} — {protocol} — per-cell uptake with best fit "
                     f"(n = {n})  [{filter_desc}]",
                     y=1.02, fontweight='bold')
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.98])

        out = (Path(output_dir)
               / f"Thesis_Uptake_BestFit_Panel_{treatment}_{protocol}"
                 f"{filename_suffix}.pdf")
        utils.save_plot_pdf(out)
        plt.close(fig)
        logger.info(f"  Saved: {out.name}")


# ===================================================================== #
# Runner                                                                 #
# ===================================================================== #
def register_uptake_plots(grouped_data: Dict,
                            mechanics_df: pd.DataFrame,
                            output_dir: Path,
                            treatments: Tuple[str, ...] = ('WT',)) -> None:
    """
    Runner exposing all three uptake plots with matched error handling.

    Called from `master_bulk_thesis.py` after the mechanics dataframe
    is written.  Iterates over the supplied treatments so that CytD
    figures can be produced by passing `treatments=('WT', 'CytD')`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _try(label, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Uptake plot '{label}' failed: {e}",
                         exc_info=False)

    for treatment in treatments:
        # --- Mean trace: two variants ---------------------------------
        # Default (loose): all EP + fate cells, so the population shape
        # isn't skewed by dropping cells whose mechanics fit happens to
        # be poor.  Strict: MI + PrePulse cohort, matching the
        # correlation figures.
        _try(f"Mean uptake trace — all EP ({treatment})",
             plot_thesis_mean_uptake_body_prot,
             grouped_data, mechanics_df, output_dir, treatment=treatment,
             require_mi_flag=False, require_prepulse_flag=False,
             filename_suffix='')
        _try(f"Mean uptake trace — MI + PrePulse cohort ({treatment})",
             plot_thesis_mean_uptake_body_prot,
             grouped_data, mechanics_df, output_dir, treatment=treatment,
             require_mi_flag=True, require_prepulse_flag=True,
             filename_suffix='_MI_PrePulse_Pass')

        # --- Amplitude vs Trap ID: two variants -----------------------
        _try(f"Uptake amplitude per trap — all EP ({treatment})",
             plot_thesis_uptake_amplitude_per_trap,
             mechanics_df, output_dir, treatment=treatment,
             require_mi_flag=False, require_prepulse_flag=False,
             filename_suffix='')
        _try(f"Uptake amplitude per trap — MI + PrePulse cohort ({treatment})",
             plot_thesis_uptake_amplitude_per_trap,
             mechanics_df, output_dir, treatment=treatment,
             require_mi_flag=True, require_prepulse_flag=True,
             filename_suffix='_MI_PrePulse_Pass')

        # --- Correlations: three figures ------------------------------
        # PrePulse-visco parameters → PrePulse R² gate, uptake amplitude only.
        # MI whole-trace descriptors → MI R² gate, uptake amplitude only.
        # Uptake amplitude vs pre-pulse protrusion length: no mechanics
        # gate (only uptake R² per region), on the natural EP cohort.
        _try(f"Pre-pulse correlations ({treatment})",
             plot_thesis_prepulse_correlations,
             mechanics_df, output_dir, treatment=treatment)
        _try(f"MI whole-trace correlations ({treatment})",
             plot_thesis_mi_whole_correlations,
             mechanics_df, output_dir, treatment=treatment)
        _try(f"Uptake vs protrusion length ({treatment})",
             plot_thesis_uptake_vs_prot_length,
             mechanics_df, output_dir, treatment=treatment)

        # --- Best-fit multipanel: two variants ------------------------
        _try(f"Uptake best-fit multipanel — MI cohort ({treatment})",
             plot_thesis_uptake_bestfit_multipanel,
             grouped_data, mechanics_df, output_dir, treatment=treatment,
             require_mi_flag=True, require_prepulse_flag=False,
             filename_suffix='')
        _try(f"Uptake best-fit multipanel — MI + PrePulse cohort ({treatment})",
             plot_thesis_uptake_bestfit_multipanel,
             grouped_data, mechanics_df, output_dir, treatment=treatment,
             require_mi_flag=True, require_prepulse_flag=True,
             filename_suffix='_MI_PrePulse_Pass')