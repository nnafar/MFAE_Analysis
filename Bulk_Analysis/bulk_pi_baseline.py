# -*- coding: utf-8 -*-
"""
PI Baseline / Dead-On-Arrival (DOA) Filter (F0-based).

Purpose
-------
Cells that carry PI at trap entry ("dead-on-arrival", DOA) have a raw
fluorescence pre-pulse baseline (F0) that sits well above the dye-free
background. Healthy cells have F0 near zero. This module flags DOA cells
by thresholding on F0 reconstructed from the intensity / normalized dF/F0
ratio (same reconstruction the F0-absolute Pre_Loaded check in
bulk_mechanics.py uses).

The dye can accumulate in the cell body, the protrusion, or both, so the
metric is max(F0_body, F0_prot) — a cell is flagged if either channel is
elevated. This mirrors the OR clause in the existing Pre_Loaded check but
threshold selection is data-driven from '_doa'-labelled examples rather
than hardcoded.

Design principles:
    - Applies only to EP conditions. ASP cells are exempt from the automatic
      threshold check per the current pipeline convention; ASP cells manually
      marked '_doa' are still excluded because the manual label is trusted
      ground truth.
    - Calibrates the threshold from '_doa'-marked traps' F0 values, falling
      back to the hard-coded default when too few labelled examples are
      present or when the calibrated threshold falls below the fallback
      (indicating the labelled distribution does not separate cleanly from
      healthy cells).

Public API
----------
compute_pi_baseline_df(grouped_data, n_baseline)
    Per-trap F0 metric plus DOA_Labeled flag. Column `PI_Baseline` holds
    max(F0_body, F0_prot); `PI_F0_Body_ADU` and `PI_F0_Prot_ADU` are kept
    separately for audit.
calibrate_threshold(pi_df, fallback, safety_factor, min_samples)
    Derives a threshold from the labelled F0 distribution or falls back.
flag_doa(pi_df, threshold)
    Adds boolean PI_DOA_Flag column (respects DOA_Labeled and ASP exemption).
merge_into_mechanics(mechanics_df, pi_df)
    Left-joins PI_Baseline + PI_DOA_Flag + F0 audit columns into mechanics_df.
filter_grouped_data(grouped_data, pi_df)
    Returns a copy of grouped_data with DOA-flagged traps removed.
plot_pi_baseline_distribution(pi_df, threshold, output_dir, ...)
    Diagnostic histogram with threshold line, ASP/EP overlaid, labelled DOA
    example locations marked. Log-scaled x-axis (F0 spans several decades).
run_pi_doa_filter(...)
    End-to-end orchestrator.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import bulk_file_handling as bfh
import Utils_MFA as utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# N_baseline = 5 matches UptakeQuantification.py 'baseline_frames' default,
# the first-5-frames window used inside bulk_mechanics._estimate_f0, and the
# window this module reconstructs F0 from below. Keeping the same window
# length makes the reconstructed F0 directly comparable to the F0-based
# Pre_Loaded check in bulk_mechanics.
DEFAULT_N_BASELINE = 5

# Fallback threshold used when fewer than DEFAULT_MIN_CAL_SAMPLES labelled
# DOA traps are available for calibration, OR when the calibrated threshold
# falls below this floor (indicating the labelled distribution does not
# separate from healthy cells and cannot support a defensible calibration).
# Matches the hardcoded PRE_LOADED_F0_THRESHOLD in bulk_mechanics.py so this
# module and the pipeline-level Pre_Loaded check use the same floor in the
# fallback case.
DEFAULT_PI_BASELINE_THRESHOLD = 1500.0

# Threshold = min(labelled F0) * SAFETY_FACTOR. Values slightly below 1.0
# add a small margin below the least-suspicious labelled cell so borderline
# unlabelled cells that sit just under the labelled minimum still get caught.
# A factor of 0.9 leaves a 10% safety margin — tight enough to preserve the
# labelled distribution's ground truth, loose enough to catch borderline
# cases the manual review may have missed.
DEFAULT_SAFETY_FACTOR = 0.90

# Minimum number of labelled DOA samples required to trust the calibrated
# threshold. Below this, calibration falls back to the hard-coded default.
DEFAULT_MIN_CAL_SAMPLES = 3


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def _estimate_f0(df_array: np.ndarray,
                  int_array: np.ndarray,
                  n_baseline: int) -> float:
    """
    Reconstruct F0 (absolute pre-pulse fluorescence, ADU) from the intensity
    and normalized-dF/F0 columns. Same math as bulk_mechanics._estimate_f0,
    duplicated here so bulk_pi_baseline stays a self-contained module.

    Body_Intensity[i]           = raw[i] - F0
    Body_Normalized_dF_F0[i]    = (raw[i] - F0) / F0
        →  raw[i] = Body_Intensity[i] + F0
        →  Body_Intensity[i] / Body_Normalized_dF_F0[i] = F0

    So F0 is a constant per trap that can be recovered from any frame where
    dF/F0 is not numerically zero. Uses the first `n_baseline` frames for a
    stable median estimate.
    """
    n_frames = min(n_baseline, len(df_array))
    if n_frames == 0:
        return 0.0

    df_sub  = df_array[:n_frames]
    int_sub = int_array[:n_frames]

    # dF/F0 near zero would blow up the reconstruction; skip those frames.
    valid = np.abs(df_sub) > 1e-4
    if not np.any(valid):
        return 0.0
    return float(np.nanmedian(int_sub[valid] / df_sub[valid]))


def _trap_f0(trap: bfh.TrapData, n_baseline: int):
    """
    Returns (f0_body, f0_prot) for a single trap. Either can be NaN if the
    corresponding intensity or dF/F0 column is missing / all-zero.

    Uses the first `n_baseline` frames of the uptake trace, matching the
    baseline window bulk_mechanics uses when reconstructing F0 for the
    Pre_Loaded check.
    """
    ud = trap.uptake_data
    if not ud:
        return float('nan'), float('nan')

    body_int = ud.get('Body_Intensity')
    body_df  = ud.get('Body_Normalized_dF_F0')
    prot_int = ud.get('Protrusion_Intensity')
    prot_df  = ud.get('Protrusion_Normalized_dF_F0')

    def _one(int_arr, df_arr):
        if int_arr is None or df_arr is None:
            return float('nan')
        int_arr = np.asarray(int_arr, dtype=float)
        df_arr  = np.asarray(df_arr,  dtype=float)
        if int_arr.size == 0 or df_arr.size == 0:
            return float('nan')
        if len(int_arr) != len(df_arr):
            return float('nan')
        return _estimate_f0(df_arr, int_arr, n_baseline)

    return _one(body_int, body_df), _one(prot_int, prot_df)


def compute_pi_baseline_df(grouped_data: Dict,
                            n_baseline: int = DEFAULT_N_BASELINE
                            ) -> pd.DataFrame:
    """
    Iterate over grouped_data and compute F0-based DOA metric per trap.

    Returns
    -------
    pd.DataFrame with columns:
        Experiment_Folder, Trap_ID, Condition_Type, N_Baseline,
        PI_F0_Body_ADU, PI_F0_Prot_ADU,
        PI_Baseline  (= max of body and prot F0, the metric flag_doa uses),
        DOA_Labeled

    Both channels are kept for audit — you can inspect which channel is
    driving the flag on any given trap. PI_Baseline itself is the working
    metric so downstream code that keyed off it before still works.
    """
    records: List[dict] = []
    for traps in grouped_data.values():
        for trap in traps:
            f0_body, f0_prot = _trap_f0(trap, n_baseline)

            # Max across channels — cell is DOA if either signal is elevated.
            # np.nanmax handles the case where one channel is NaN (no data
            # for that region) by taking the other. If both are NaN the
            # result is NaN, which is then not flagged.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if np.isnan(f0_body) and np.isnan(f0_prot):
                    pi_baseline = float('nan')
                else:
                    pi_baseline = float(np.nanmax([f0_body, f0_prot]))

            records.append({
                'Experiment_Folder': trap.metadata.full_path.name,
                'Trap_ID':           trap.trap_id,
                'Condition_Type':    trap.metadata.condition_type,
                'N_Baseline':        n_baseline,
                'PI_F0_Body_ADU':    f0_body,
                'PI_F0_Prot_ADU':    f0_prot,
                'PI_Baseline':       pi_baseline,
                'DOA_Labeled':       bool(getattr(trap, 'is_doa_labeled', False)),
            })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        logger.warning("compute_pi_baseline_df: no traps found in grouped_data.")
    return df


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------
def calibrate_threshold(pi_df: pd.DataFrame,
                         fallback: float = DEFAULT_PI_BASELINE_THRESHOLD,
                         safety_factor: float = DEFAULT_SAFETY_FACTOR,
                         min_samples: int = DEFAULT_MIN_CAL_SAMPLES
                         ) -> Tuple[float, str]:
    """
    Derive a PI_Baseline threshold from the labelled DOA distribution.

    Rule: threshold = min(labelled PI_Baseline) * safety_factor.
    Rationale: every labelled DOA cell should end up flagged, so the
    threshold sits at (or just below) the least-suspicious labelled example.
    A safety_factor of 0.9 leaves a small margin so unlabelled cells that
    approach but don't quite match the labelled minimum still get caught.

    Falls back to `fallback` when fewer than `min_samples` labelled traps
    have a finite PI_Baseline.

    Returns
    -------
    (threshold, source_label)
        threshold : float — the value to use for flagging
        source_label : str — 'calibrated' or 'fallback', for logging
    """
    labelled = pi_df.loc[
        pi_df['DOA_Labeled'] & pi_df['PI_Baseline'].notna(),
        'PI_Baseline'
    ].values

    n = len(labelled)
    if n < min_samples:
        logger.info(
            f"PI_DOA calibration: only {n} labelled DOA trap(s) with finite "
            f"PI_Baseline (need >={min_samples}) — using fallback "
            f"threshold {fallback:.4f}."
        )
        return float(fallback), 'fallback'

    lo = float(np.min(labelled))
    med = float(np.median(labelled))
    hi = float(np.max(labelled))
    calibrated_raw = lo * safety_factor

    # Guard against nonsense calibrations. The `min * safety_factor` rule
    # only produces a defensible threshold when the labelled DOA cells sit
    # clearly above the healthy distribution — i.e. when the labelled min is
    # a well-separated positive value. If the labelled cells cluster near
    # zero (which happens on EP experiments where the F0 pre-pulse window
    # sits inside the already-loaded dye signal, so raw - F0 ≈ 0 for the
    # cell body even in DOA cells), `lo` is near zero or negative and the
    # derived threshold flags virtually everything. In that case fall back
    # to the hard-coded default and leave the manual `_doa` labels to do
    # the flagging work on their own.
    if calibrated_raw < fallback:
        logger.warning(
            f"PI_DOA calibration: {n} labelled DOA trap(s) present but their "
            f"PI_Baseline distribution (min={lo:.1f}, median={med:.1f}, "
            f"max={hi:.1f}) does not separate from healthy cells on this "
            f"metric. Calibrated threshold would be {calibrated_raw:.1f} ADU "
            f"(below fallback {fallback:.1f} ADU) — falling back to default. "
            f"Manual _doa labels will still be flagged."
        )
        return float(fallback), 'fallback_after_bad_calibration'

    threshold = calibrated_raw
    logger.info(
        f"PI_DOA calibration: {n} labelled DOA trap(s), "
        f"PI_Baseline min={lo:.1f}, median={med:.1f}, max={hi:.1f}. "
        f"Threshold set to min * {safety_factor:.2f} = {threshold:.1f} ADU "
        f"(fallback default would be {fallback:.4f})."
    )
    return threshold, 'calibrated'


# ---------------------------------------------------------------------------
# Flagging + merging
# ---------------------------------------------------------------------------
def flag_doa(pi_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Add PI_DOA_Flag column. Rules:
      - Labelled DOA cells (DOA_Labeled=True) → always flagged.
      - ASP cells (Condition_Type=='ASP') → threshold check skipped; only
        the manual label can flag them.
      - EP cells (Condition_Type=='EP') → flagged iff PI_Baseline > threshold.
      - NaN PI_Baseline → not flagged (unless labelled).
    """
    out = pi_df.copy()

    labelled = out['DOA_Labeled'].astype(bool)
    is_ep = out['Condition_Type'] == 'EP'
    over_threshold = out['PI_Baseline'].fillna(-np.inf) > threshold

    out['PI_DOA_Flag'] = labelled | (is_ep & over_threshold)
    return out


def merge_into_mechanics(mechanics_df: pd.DataFrame,
                          pi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join PI_Baseline, PI_F0_Body_ADU, PI_F0_Prot_ADU, DOA_Labeled, and
    PI_DOA_Flag into mechanics_df on (Experiment_Folder, Trap_ID). Rows in
    mechanics_df without a match get NaN metric and False flags.

    Non-destructive: returns a new DataFrame; mechanics_df is not modified.
    """
    merge_cols = [
        'Experiment_Folder', 'Trap_ID',
        'PI_Baseline', 'PI_F0_Body_ADU', 'PI_F0_Prot_ADU',
        'DOA_Labeled', 'PI_DOA_Flag',
    ]
    if not all(c in pi_df.columns for c in merge_cols):
        missing = [c for c in merge_cols if c not in pi_df.columns]
        raise ValueError(
            f"pi_df is missing required columns: {missing}. "
            "Did you forget to call flag_doa() first?"
        )

    merged = mechanics_df.merge(
        pi_df[merge_cols],
        on=['Experiment_Folder', 'Trap_ID'],
        how='left',
    )
    merged['PI_DOA_Flag'] = merged['PI_DOA_Flag'].fillna(False).astype(bool)
    merged['DOA_Labeled'] = merged['DOA_Labeled'].fillna(False).astype(bool)
    return merged


# ---------------------------------------------------------------------------
# grouped_data filtering
# ---------------------------------------------------------------------------
def filter_grouped_data(grouped_data: Dict, pi_df: pd.DataFrame) -> Dict:
    """
    Remove DOA-flagged traps from grouped_data. Returns a new dict; the
    original is not mutated.
    """
    if 'PI_DOA_Flag' not in pi_df.columns:
        raise ValueError("pi_df is missing PI_DOA_Flag. Call flag_doa() first.")

    doa_pairs = set(zip(
        pi_df.loc[pi_df['PI_DOA_Flag'], 'Experiment_Folder'],
        pi_df.loc[pi_df['PI_DOA_Flag'], 'Trap_ID'],
    ))

    if not doa_pairs:
        return dict(grouped_data)

    filtered = {}
    for gk, traps in grouped_data.items():
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) not in doa_pairs
        ]
        if kept:
            filtered[gk] = kept
    return filtered


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------
def plot_pi_baseline_distribution(pi_df: pd.DataFrame,
                                   threshold: float,
                                   output_dir: Path,
                                   n_baseline: int = DEFAULT_N_BASELINE,
                                   threshold_source: str = 'calibrated',
                                   fallback_threshold: float = DEFAULT_PI_BASELINE_THRESHOLD) -> None:
    """
    Histogram of PI_Baseline, ASP and EP overlaid, with the active threshold
    marked. Labelled DOA traps are highlighted as inverted-triangle markers
    along the top of the plot so their calibration positions are visible.

    threshold_source ∈ {'calibrated', 'fallback'} — controls the label on
    the vertical line. When calibrated, the fallback default is also drawn
    as a lighter dashed line for reference.
    """
    df = pi_df[pi_df['PI_Baseline'].notna() & (pi_df['PI_Baseline'] > 0)].copy()
    if df.empty:
        logger.warning("plot_pi_baseline_distribution: no positive finite PI_Baseline values to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    asp_color = utils.MFA_COLORS.get('dark_blue', '#1f4e79') if hasattr(utils, 'MFA_COLORS') else '#1f4e79'
    ep_color  = utils.MFA_COLORS.get('dark_red',  '#b31529') if hasattr(utils, 'MFA_COLORS') else '#b31529'

    palette = {'ASP': asp_color, 'EP': ep_color}
    present = [c for c in ('ASP', 'EP') if c in df['Condition_Type'].unique()]

    # F0 spans several decades from healthy (~10-100 ADU) to DOA
    # (>10³ ADU), so a log-scale x-axis reads more cleanly than linear.
    # Robust upper bound so a single very-bright outlier doesn't collapse
    # the visible range onto a small corner.
    lo_data = float(np.nanpercentile(df['PI_Baseline'], 1))
    hi_data = float(np.nanpercentile(df['PI_Baseline'], 99))
    lo = max(min(lo_data, 1.0), 1.0)  # log scale can't show <= 0
    hi = max(hi_data, threshold * 2, fallback_threshold * 2)

    # Log-spaced bins so the histogram shape reads correctly on a log axis.
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    for cond in present:
        sub = df[df['Condition_Type'] == cond]
        if sub.empty:
            continue
        ax.hist(
            sub['PI_Baseline'].clip(lo, hi),
            bins=bins,
            color=palette[cond], alpha=0.55,
            label=f'{cond}  (n={len(sub)})',
            edgecolor='white', linewidth=0.5,
        )

    ax.set_xscale('log')

    # Active threshold. Label reflects source clearly so the reader can tell
    # a good calibration from a fallback-after-bad-calibration event.
    src_map = {
        'calibrated':                       'calibrated from labelled DOA',
        'fallback':                         'fallback default (no calibration)',
        'fallback_after_bad_calibration':   'fallback (calibration produced nonsense)',
    }
    src_label = src_map.get(threshold_source, threshold_source)
    ax.axvline(threshold, color='k', ls='--', lw=1.5,
               label=f'Threshold = {threshold:.1f} ADU  ({src_label})')

    # If calibrated cleanly, draw the fallback for comparison. Skip the
    # comparison line when we already fell back — it would just plot on top
    # of the active threshold and clutter the legend.
    if threshold_source == 'calibrated' and not np.isclose(threshold, fallback_threshold):
        ax.axvline(fallback_threshold, color='gray', ls=':', lw=1.0,
                   label=f'Fallback default = {fallback_threshold:.1f} ADU')

    # Labelled DOA markers along the top edge — show where the calibration
    # examples sit on the same axis. Helps read the threshold's provenance.
    labelled = df[df['DOA_Labeled']]
    if not labelled.empty:
        y_top = ax.get_ylim()[1]
        ax.scatter(
            labelled['PI_Baseline'].clip(lo, hi),
            [y_top * 0.97] * len(labelled),
            marker='v', s=50,
            color='goldenrod', edgecolor='black', linewidth=0.6, zorder=5,
            label=f'Labelled DOA  (n={len(labelled)})',
        )

    ax.set_xlabel(
        f'F0 pre-pulse fluorescence, max(body, protrusion) (ADU)  '
        f'— reconstructed from first {n_baseline} frames'
    )
    ax.set_ylabel('Trap count')
    ax.legend(fontsize=9)

    # Per-condition flag counts in a corner box.
    lines = []
    for cond in present:
        sub_pi = pi_df[pi_df['Condition_Type'] == cond]
        n_total = len(sub_pi)
        if 'PI_DOA_Flag' in sub_pi.columns:
            n_flagged = int(sub_pi['PI_DOA_Flag'].sum())
        else:
            n_flagged = int((sub_pi['PI_Baseline'] > threshold).sum())
        pct = 100.0 * n_flagged / n_total if n_total else 0.0
        lines.append(f'{cond}: {n_flagged}/{n_total} flagged ({pct:.1f}%)')
    if lines:
        ax.text(
            0.98, 0.98, '\n'.join(lines),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'),
        )

    plt.tight_layout()
    utils.save_plot_pdf(Path(output_dir) / "PI_Baseline_Distribution.pdf")
    plt.close()


# ---------------------------------------------------------------------------
# One-shot orchestration
# ---------------------------------------------------------------------------
def run_pi_doa_filter(all_grouped_data: Dict,
                       mechanics_df: pd.DataFrame,
                       output_dir: Path,
                       metric_output_dir: Path = None,
                       fallback_threshold: float = DEFAULT_PI_BASELINE_THRESHOLD,
                       safety_factor: float = DEFAULT_SAFETY_FACTOR,
                       min_cal_samples: int = DEFAULT_MIN_CAL_SAMPLES,
                       n_baseline: int = DEFAULT_N_BASELINE):
    """
    End-to-end pipeline step.

    1. Compute PI_Baseline per trap (including labelled DOA).
    2. Calibrate threshold from labelled DOA distribution (or fall back).
    3. Flag traps: labelled DOA always flagged; EP cells flagged if above
       threshold; ASP cells flagged only if labelled.
    4. Merge flag into mechanics_df.
    5. Filter grouped_data.
    6. Emit the diagnostic distribution PDF into `output_dir` (qc folder).
    7. Save the PI_Baseline table into `metric_output_dir` (mechanics folder,
       defaults to `output_dir` when not provided).
    8. Log per-condition counts.

    Parameters
    ----------
    output_dir : Path
        Where the diagnostic PDF is written (qc/ subfolder in the new
        layout).
    metric_output_dir : Path or None
        Where the per-trap metric CSV is written. Defaults to `output_dir`.

    Returns
    -------
    (filtered_grouped_data, mechanics_df_with_flag, pi_df)
    """
    output_dir = Path(output_dir)
    metric_output_dir = Path(metric_output_dir) if metric_output_dir else output_dir

    # 1. Compute PI_Baseline for all traps.
    pi_df = compute_pi_baseline_df(all_grouped_data, n_baseline=n_baseline)

    # 2. Calibrate threshold from labelled DOA distribution.
    threshold, source = calibrate_threshold(
        pi_df,
        fallback=fallback_threshold,
        safety_factor=safety_factor,
        min_samples=min_cal_samples,
    )

    # 3. Flag traps.
    pi_df = flag_doa(pi_df, threshold=threshold)

    # Per-condition audit split by labelled / auto-flagged.
    for cond, sub in pi_df.groupby('Condition_Type'):
        n_total   = len(sub)
        n_flagged = int(sub['PI_DOA_Flag'].sum())
        n_labelled = int(sub['DOA_Labeled'].sum())
        n_auto = n_flagged - n_labelled
        n_nan  = int(sub['PI_Baseline'].isna().sum())
        pct = 100.0 * n_flagged / n_total if n_total else 0.0
        logger.info(
            f"PI_DOA filter [{cond}]: {n_flagged}/{n_total} flagged "
            f"({pct:.1f}%)  [labelled={n_labelled}, auto={n_auto}]  "
            f"| {n_nan} trap(s) with no baseline (kept if unlabelled)  "
            f"| threshold={threshold:.1f} ADU, source={source}"
        )

    # 4. Persist the metric table.
    metric_output_dir.mkdir(parents=True, exist_ok=True)
    pi_path = metric_output_dir / "pi_baseline_per_trap.csv"
    pi_df.to_csv(pi_path, index=False)
    logger.info(f"PI_DOA filter: saved metric table to {pi_path.name}")

    # 5. Diagnostic plot.
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_pi_baseline_distribution(
            pi_df, threshold, output_dir,
            n_baseline=n_baseline,
            threshold_source=source,
            fallback_threshold=fallback_threshold,
        )
    except Exception as e:
        logger.error(f"PI_Baseline distribution plot failed: {e}", exc_info=False)

    # 6. Apply filters.
    filtered_grouped_data = filter_grouped_data(all_grouped_data, pi_df)
    mechanics_df_with_flag = merge_into_mechanics(mechanics_df, pi_df)

    return filtered_grouped_data, mechanics_df_with_flag, pi_df