"""
thesis_plotting_actin_by_fate.py
=================================

Companion module to `thesis_plotting.py`. Adds fate-split (`intact`
vs `ruptured_post`) variants of the two pooled actin figures produced by
the main plotting module:

* `Thesis_EP_Actin_PrePost_Paired.pdf`
    -> `Thesis_EP_Actin_PrePost_Paired_ByFate.pdf`
* `Thesis_EP_Actin_Around_Pulse_{dur}.pdf`
    -> `Thesis_EP_Actin_Around_Pulse_{dur}_ByFate.pdf`

The original functions in `thesis_plotting.py` are left untouched, so
the pooled figures are still generated for the supplement and any
downstream references to those filenames continue to resolve. The
by-fate figures are intended for the main-text `sec:ep_actin_around_pulse`
subsection of Chapter 3.

Both plotters share the pulse-frame indexing caveat noted in
`thesis_plotting.py`: the exact pre-vs-post pairing may shift by one
frame once the 0-based / 1-based offset in the pipeline is fixed. The
direction of every paired shift and every between-fate contrast is
stable to that correction. The corresponding LaTeX draft carries a
red-text flag at the point of use rather than in the plotters.

Register the two functions by calling `register_by_fate_actin_plots(...)`
next to the existing `_try` block in
`thesis_plotting.plot_thesis_uptake_actin_figures` (or wherever the
paired- and around-pulse plotters are dispatched from).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

# Grammar helpers, palette, save-plot wrapper, and the shared p-value
# helper come from the same modules the main plotter imports from, so
# the two files stay in lockstep on style.
import bulk_utils as utils
import bulk_plotting as bp

logger = logging.getLogger(__name__)


# =============================================================================
# Constants defined locally rather than imported from `thesis_plotting`.
#
# The main plotting module imports `register_by_fate_actin_plots` from
# this file so it can wire the by-fate plotters into its dispatcher; if
# this file imported names back the other way we would get a circular
# import at module load. Values below are duplicated verbatim from
# `thesis_plotting.py` (see the definitions of `_TREATMENT_ORDER`,
# `_EP_PULSE_ORDER`, and `_duration_sort_key` in that module) and must
# be kept in sync with the main module by hand.
# =============================================================================
_TREATMENT_ORDER: Tuple[str, ...] = ('WT', 'CytD')

_EP_PULSE_ORDER: Tuple[str, ...] = ('100us', '5ms')


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


# =============================================================================
# Fate ordering. Intact first so the leftmost columns in each subplot show
# the reference cohort and ruptured columns sit to their right.
# =============================================================================
_FATE_ORDER: Tuple[str, str] = ('intact', 'ruptured_post')
_FATE_TICK_LABEL: Dict[str, str] = {
    'intact':        'intact',
    'ruptured_post': 'ruptured',
}


# =============================================================================
# Paired pre/post actin, split by fate
# =============================================================================
def plot_thesis_actin_prepulse_vs_postpulse_paired_by_fate(
    mechanics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Paired pre/post actin boxplot with within-cell lines, split by fate.

    Layout: 2 panels (rows) — top = Body region, bottom = Protrusion —
    each holding every (treatment, fate, duration) combination side by
    side. For a WT-only EP cohort with both fates present, each panel
    reads left-to-right as:

        [WT intact  100 µs pre / post]
        [WT intact   5 ms   pre / post]
        [WT ruptured 100 µs pre / post]
        [WT ruptured  5 ms  pre / post]

    with narrow gaps between (duration) blocks and a wider gap between
    fate blocks. If both WT and CytD are present, the whole sequence is
    repeated for CytD after a bigger gap. Each cell contributes two dots
    (pre, post) joined by a thin grey line so the within-cell direction
    is visible.

    Statistics annotated:
      * Wilcoxon signed-rank p-value on the within-cell (pre, post) pair,
        printed above each duration block.
      * Mann-Whitney U p-value on the post-pulse values between fate
        cohorts of the same (treatment, duration), printed as a bracket
        between the two post columns of a matched duration.

    Only cells with `Condition_Type == 'EP'` and `Fate_Status` in
    (`intact`, `ruptured_post`) are used, matching the filter of the
    pooled variant in `thesis_plotting.py`.

    Output filename: `Thesis_EP_Actin_PrePost_Paired_ByFate.pdf`.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning(
            "Paired actin (by fate) plot skipped: empty mechanics_df.")
        return

    df = mechanics_df.copy()
    df = df.loc[
        (df['Condition_Type'] == 'EP')
        & df['Fate_Status'].isin(list(_FATE_ORDER))
    ]
    if df.empty:
        logger.warning(
            "Paired actin (by fate) plot skipped: no EP cells pass filter.")
        return

    durations_present = [
        d for d in _EP_PULSE_ORDER
        if d in df['Duration_label'].unique()
    ]
    if not durations_present:
        logger.warning(
            "Paired actin (by fate) plot skipped: no expected pulse "
            "durations (100us / 5ms) in EP cohort.")
        return

    treatments_present = [
        t for t in _TREATMENT_ORDER if (df['Treatment'] == t).any()
    ]
    if not treatments_present:
        logger.warning(
            "Paired actin (by fate) plot skipped: no known treatments "
            "in EP cohort.")
        return

    # -------------------------------------------------------------------
    # Column geometry: single row per region. Loop order is
    #   treatment -> fate -> duration
    # so all durations of the same (treatment, fate) sit next to each
    # other. Gaps: 0.6 between durations within a fate block, 1.6 between
    # fates within a treatment, 2.4 between treatments.
    # -------------------------------------------------------------------
    positions: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    xticks: list = []
    xticklabels: list = []
    cursor = 0.0
    for treatment in treatments_present:
        for fate in _FATE_ORDER:
            for dur in durations_present:
                p_pre, p_post = cursor, cursor + 1.0
                positions[(treatment, fate, dur)] = (p_pre, p_post)
                xticks.extend([p_pre, p_post])
                # Compact two-line label so ticks don't collide.
                block = f"{_FATE_TICK_LABEL[fate]}\n{dur}"
                xticklabels.extend([f"{block}\npre", f"{block}\npost"])
                cursor += 2.6  # duration-block width
            cursor += 1.6  # extra gap between fates within a treatment
        cursor += 2.4  # extra gap between treatments

    # Figure size scales with the number of columns.
    n_cols = len(xticks)
    fig_w = max(9.0, 0.55 * n_cols + 2.0)
    fig, axes = plt.subplots(
        2, 1, figsize=(fig_w, 7.6), squeeze=False, sharex=True,
    )

    for row_idx, region in enumerate(('Body', 'Prot')):
        ax = axes[row_idx, 0]

        pre_col  = f'Actin_{region}_PrePulse_F0Norm'
        post_col = f'Actin_{region}_PostPulse_F0Norm'
        if pre_col not in df.columns or post_col not in df.columns:
            ax.set_visible(False)
            continue

        # Cache post-pulse values per (treatment, dur, fate) for the
        # between-fate Mann-Whitney annotation after all boxes are drawn.
        post_by_group: Dict[Tuple[str, str, str], np.ndarray] = {}

        for treatment in treatments_present:
            for fate in _FATE_ORDER:
                for dur in durations_present:
                    pts = df.loc[
                        (df['Duration_label'] == dur)
                        & (df['Treatment'] == treatment)
                        & (df['Fate_Status'] == fate),
                        [pre_col, post_col],
                    ].dropna(subset=[pre_col, post_col])
                    if pts.empty:
                        continue
                    x_pre, x_post = positions[(treatment, fate, dur)]

                    # Grammar routing: colour follows (treatment, dur);
                    # fate encoded via linestyle / marker fill.
                    lkw = utils.get_line_kwargs(
                        treatment, dur, fate=fate, size=5,
                    )
                    colour = lkw['color']
                    linestyle = lkw['linestyle']
                    marker = lkw['marker']

                    pre_vals  = pts[pre_col].to_numpy()
                    post_vals = pts[post_col].to_numpy()
                    post_by_group[(treatment, dur, fate)] = post_vals

                    # Within-cell connectors.
                    for pv, qv in zip(pre_vals, post_vals):
                        ax.plot(
                            [x_pre, x_post], [pv, qv],
                            color='0.75', lw=0.5, alpha=0.7, zorder=1,
                        )

                    # Fate-aware boxplot styling.
                    ax.boxplot(
                        [pre_vals, post_vals],
                        positions=[x_pre, x_post], widths=0.55,
                        patch_artist=True,
                        boxprops=dict(
                            facecolor='lightgray', alpha=0.4,
                            edgecolor=colour, linestyle=linestyle,
                        ),
                        medianprops=dict(
                            color=colour, lw=1.4, linestyle=linestyle,
                        ),
                        whiskerprops=dict(color=colour),
                        capprops=dict(color=colour),
                        showfliers=False,
                    )

                    # Individual points, ruptured -> open markers.
                    face = 'none' if fate == 'ruptured_post' else colour
                    edge = colour
                    lw_pt = 0.6 if fate == 'ruptured_post' else 0.4
                    rng = np.random.default_rng(
                        42 + row_idx * 100
                        + hash((treatment, fate, dur)) % 1000
                    )
                    for x_, vals in ((x_pre, pre_vals),
                                      (x_post, post_vals)):
                        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                        ax.scatter(
                            np.full(len(vals), x_) + jitter, vals,
                            s=18, marker=marker,
                            facecolors=face, edgecolors=edge,
                            linewidths=lw_pt, alpha=0.85, zorder=3,
                        )

                    # Within-cell Wilcoxon.
                    if len(pre_vals) >= 5:
                        try:
                            _, p_val = stats.wilcoxon(pre_vals, post_vals)
                            med_delta = float(
                                np.median(post_vals - pre_vals))
                            logger.info(
                                f"  [PairedActin Wilcoxon] region={region} "
                                f"treatment={treatment} fate={fate} "
                                f"dur={dur}: n={len(pre_vals)} "
                                f"med_delta={med_delta:+.4g} p={p_val:.4g}"
                            )
                            p_str = (
                                f"W p={p_val:.3f}" if p_val >= 0.001
                                else "W p<0.001"
                            )
                            y_top = float(np.nanmax(np.concatenate(
                                [pre_vals, post_vals])))
                            ax.text(
                                (x_pre + x_post) / 2, y_top * 1.02,
                                p_str, ha='center', va='bottom',
                                fontsize=6.5, color=colour,
                            )
                        except ValueError:
                            pass

        # Between-fate Mann-Whitney on post-pulse values, one per
        # (treatment, duration). Bracket spans the two post columns of
        # the matched duration.
        for treatment in treatments_present:
            for dur in durations_present:
                intact_post = post_by_group.get(
                    (treatment, dur, 'intact'), np.array([]))
                rup_post = post_by_group.get(
                    (treatment, dur, 'ruptured_post'), np.array([]))
                if len(intact_post) < 3 or len(rup_post) < 3:
                    continue
                try:
                    _, p_mw = stats.mannwhitneyu(
                        intact_post, rup_post, alternative='two-sided')
                    A = intact_post[:, None]
                    B = rup_post[None, :]
                    delta = (
                        (A > B).sum() - (A < B).sum()
                    ) / (len(intact_post) * len(rup_post))
                    logger.info(
                        f"  [PairedActin MW post] region={region} "
                        f"treatment={treatment} dur={dur}: "
                        f"n=({len(intact_post)},{len(rup_post)}) "
                        f"delta={delta:+.3f} p={p_mw:.4g}"
                    )
                    p_str = (
                        f"MW p={p_mw:.3g}" if p_mw >= 0.001
                        else "MW p<0.001"
                    )
                    _, x_intact_post = positions[
                        (treatment, 'intact', dur)]
                    _, x_rup_post = positions[
                        (treatment, 'ruptured_post', dur)]
                    y_all = np.concatenate([intact_post, rup_post])
                    y_top = float(np.nanmax(y_all)) * 1.14
                    ax.plot(
                        [x_intact_post, x_rup_post],
                        [y_top, y_top],
                        color='0.35', lw=0.7,
                    )
                    ax.text(
                        (x_intact_post + x_rup_post) / 2,
                        y_top * 1.01,
                        p_str, ha='center', va='bottom',
                        fontsize=7.0, color='0.15',
                    )
                except ValueError:
                    pass

        ax.axhline(1.0, color='0.6', lw=0.6, ls='--', zorder=0)
        ax.set_ylabel(f"{region}\n" r"Actin $I/F_0$")
        ax.set_title(f"{region} region", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared x-axis: apply ticks / labels once to the bottom panel.
    bottom_ax = axes[1, 0]
    bottom_ax.set_xticks(xticks)
    bottom_ax.set_xticklabels(xticklabels, fontsize=7.0)
    if xticks:
        bottom_ax.set_xlim(min(xticks) - 0.7, max(xticks) + 0.7)

    plt.tight_layout()
    utils.save_plot_pdf(
        output_dir / "Thesis_EP_Actin_PrePost_Paired_ByFate.pdf"
    )
    plt.close()
    logger.info(
        "Paired pre/post actin boxplot (by fate) written "
        "(pulse-frame bug caveat applies)."
    )


# =============================================================================
# Time-resolved actin around the pulse, split by fate
# =============================================================================
def plot_thesis_actin_around_pulse_trace_by_fate(
    grouped_data: Dict,
    mechanics_df: pd.DataFrame,
    output_dir: Path,
    output_dir_map: Optional[Dict[str, Path]] = None,
    window_pre_s: float = 15.0,
    window_post_s: float = 15.0,
    title_fontsize: float =  13,
    label_fontsize: float =  13,
    legend_fontsize: float = 13,
) -> None:
    """
    Time-resolved mean F0-normalised actin fluorescence aligned to the
    pulse frame, one figure per pulse duration, with intact and
    ruptured_post overlaid as separate traces per treatment.

    Layout: 2 rows (Body, Prot) x 1 col per figure. Within a subplot,
    each (treatment, fate) group is overlaid as a mean +/- SD line.
    Intact traces are solid; ruptured traces are dashed. The pulse
    itself is drawn as a vertical dashed line at t=0.

    Traces are aggregated by interpolating each per-cell (t, I/F0) onto
    a common grid over [-window_pre_s, +window_post_s], stacking, and
    taking nanmean +/- nanstd across cells.

    Only EP cells with a fate label of `intact` or `ruptured_post` are
    included, matching the pooled variant. Per-cell, per-region traces
    with fewer than 5 valid time points inside the window are dropped
    silently as in the pooled variant.

    Files are named `Thesis_EP_Actin_Around_Pulse_{dur}_ByFate.pdf` so
    they do not overwrite the pooled variants; if `output_dir_map` is
    provided the file for each duration goes to
    `output_dir_map[dur]`, falling back to `output_dir` for durations
    not in the map.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning(
            "Actin-around-pulse (by fate) plot skipped: empty mechanics_df."
        )
        return

    df = mechanics_df.copy()
    df_ep = df.loc[
        (df['Condition_Type'] == 'EP')
        & df['Fate_Status'].isin(list(_FATE_ORDER))
    ]
    # Map (folder, trap_id) -> fate so we can route each accepted EP trap
    # into its fate cohort at aggregation time.
    fate_map: Dict[Tuple[str, int], str] = {
        (row.Experiment_Folder, int(row.Trap_ID)): row.Fate_Status
        for row in df_ep.itertuples(index=False)
    }
    if not fate_map:
        logger.warning(
            "Actin-around-pulse (by fate) plot skipped: no EP cells accepted."
        )
        return

    # (treatment, fate, duration, region) -> list of (t_aligned, I/F0)
    traces: Dict[Tuple[str, str, str, str], list] = {}
    durations_seen: set = set()

    for gk, traps in grouped_data.items():
        if not traps:
            continue
        cond_type = traps[0].metadata.condition_type

        # -----------------------------------------------------------------
        # EP trace aggregation.
        # -----------------------------------------------------------------
        if cond_type != 'EP':
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            fate = fate_map.get(key)
            if fate is None:
                continue
            treatment = meta.treatment
            duration = meta.duration_label
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
            t_pulse_zero = t_raw - t_raw[pf]  # pulse-frame indexing caveat

            for region in ('Body', 'Prot'):
                mean_key = f'Actin_{region}_Mean'
                f0_key = f'F0_{region}'
                if mean_key not in ad or f0_key not in ad:
                    continue
                intensity = np.asarray(ad[mean_key], dtype=float)
                f0_arr = np.asarray(ad[f0_key], dtype=float)
                f0_scalar = next(
                    (float(v) for v in f0_arr
                     if np.isfinite(v) and v > 0),
                    None,
                )
                if f0_scalar is None:
                    continue
                valid = (
                    np.isfinite(t_pulse_zero)
                    & np.isfinite(intensity)
                    & (intensity > 0)
                    & (t_pulse_zero >= -window_pre_s)
                    & (t_pulse_zero <= window_post_s)
                )
                if valid.sum() < 5:
                    continue
                traces.setdefault(
                    (treatment, fate, duration, region), []
                ).append(
                    (t_pulse_zero[valid], intensity[valid] / f0_scalar)
                )

    durations = sorted(durations_seen, key=_duration_sort_key)
    if not durations:
        logger.warning(
            "Actin-around-pulse (by fate) plot skipped: no valid EP durations."
        )
        return

    common_t = np.linspace(-window_pre_s, window_post_s, 90)

    def _agg(trs: list) -> Tuple[Optional[np.ndarray],
                                 Optional[np.ndarray],
                                 int]:
        if not trs:
            return None, None, 0
        stack = []
        for t, y in trs:
            f = interp1d(
                t, y, bounds_error=False, fill_value=np.nan,
                assume_sorted=True,
            )
            stack.append(f(common_t))
        if not stack:
            return None, None, 0
        arr = np.vstack(stack)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return (
                np.nanmean(arr, axis=0),
                np.nanstd(arr, axis=0),
                len(trs),
            )

    # One figure per duration. 2 rows (Body / Prot); (treatment, fate)
    # overlaid within a subplot.
    for dur in durations:
        fig, axes = plt.subplots(
            2, 1, figsize=(6.0, 5.6), sharex=True, squeeze=False,
        )
        any_data_this_dur = False

        for row_idx, region in enumerate(('Body', 'Prot')):
            ax = axes[row_idx, 0]
            ax.axvline(
                0.0, color=utils.MFA_COLORS.get('pulse', 'gray'),
                lw=1.0, ls='--', zorder=1,
            )
            ax.axhline(1.0, color='0.7', lw=0.6, ls=':', zorder=0)

            # -------------------------------------------------------------
            # EP mean traces by (treatment, fate).
            # -------------------------------------------------------------
            for treatment in _TREATMENT_ORDER:
                for fate in _FATE_ORDER:
                    mu, sd, n = _agg(
                        traces.get((treatment, fate, dur, region), [])
                    )
                    if mu is None:
                        continue
                    any_data_this_dur = True
                    lkw = utils.get_line_kwargs(
                        treatment, dur, fate=fate, size=5,
                    )
                    colour = lkw['color']
                    label = (
                        f"{treatment} "
                        f"{_FATE_TICK_LABEL[fate]} (n={n})"
                    )
                    ax.plot(
                        common_t, mu,
                        markevery=max(1, len(common_t) // 10),
                        label=label, **lkw,
                    )
                    ax.fill_between(
                        common_t, mu - sd, mu + sd,
                        color=colour, alpha=0.12, linewidth=0,
                    )

            if row_idx == 0:
                ax.set_ylabel(r"Body $I(t)/F_0$", fontsize=label_fontsize)
            else:
                ax.set_xlabel("Time from pulse (s)", fontsize=label_fontsize)
                ax.set_ylabel(r"Protrusion $I(t)/F_0$", fontsize=label_fontsize)

            ax.tick_params(axis='both', labelsize=label_fontsize * 0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if not any_data_this_dur:
            plt.close(fig)
            logger.info(f"  Skipping {dur} (by fate): no valid actin traces.")
            continue

        fig.suptitle(f"{dur}", fontsize=title_fontsize, y=0.98)
        axes[0, 0].legend(frameon=False, fontsize=legend_fontsize, loc='best')
        plt.tight_layout()

        target_dir = output_dir
        if output_dir_map is not None and dur in output_dir_map:
            target_dir = output_dir_map[dur]
        target_dir.mkdir(parents=True, exist_ok=True)
        utils.save_plot_pdf(
            target_dir / f"Thesis_EP_Actin_Around_Pulse_{dur}_ByFate.pdf"
        )
        plt.close()
        logger.info(
            f"Actin-around-pulse trace ({dur}, by fate) written "
            f"(pulse-frame indexing caveat applies)."
        )

# =============================================================================
# Dispatcher registration helper
# =============================================================================
def register_by_fate_actin_plots(
    grouped_data: Dict,
    mechanics_df: pd.DataFrame,
    output_dir: Path,
    output_dir_map: Optional[Dict[str, Path]] = None,
) -> None:
    """
    Run both by-fate plotters with the same error-swallowing behaviour
    the main dispatcher uses. Intended to be called next to the existing
    `_try(...)` calls in `thesis_plotting.plot_thesis_uptake_actin_figures`
    (or the equivalent dispatcher in the caller's pipeline).

    A minimal wiring in the caller looks like:

        from thesis_plotting_actin_by_fate import register_by_fate_actin_plots
        ...
        register_by_fate_actin_plots(
            grouped_data, mechanics_df, output_dir, output_dir_map,
        )

    Exceptions from either plotter are logged and swallowed so the rest
    of the dispatcher continues.
    """
    def _try(label: str, func, *args, **kwargs) -> None:
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Actin-by-fate plot '{label}' failed: {e}",
                exc_info=False,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _try(
        "Paired actin pre/post (by fate)",
        plot_thesis_actin_prepulse_vs_postpulse_paired_by_fate,
        mechanics_df, output_dir,
    )
    _try(
        "Actin around pulse trace (by fate)",
        plot_thesis_actin_around_pulse_trace_by_fate,
        grouped_data, mechanics_df, output_dir, output_dir_map,
    )