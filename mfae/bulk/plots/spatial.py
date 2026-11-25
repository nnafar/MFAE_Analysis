# -*- coding: utf-8 -*-
"""
Spatial Analysis of EP Efficiency Across Trap Positions.

Addresses three scientific questions:

  Q1. Is EP localised, semi-localised, or bulk?
      Answer lives in the *variance* of uptake per trap position:
      high variance  → heterogeneous, cell-to-cell or position-to-position
      low variance   → consistent bulk field affecting all traps similarly

  Q2. Is the electric field asymmetric (gradient from trap 1 to trap 18)?
      Answer lives in the *trend* of uptake across trap positions:
      monotonic slope → stronger field at one end (near electrode)
      flat slope      → uniform field across the channel

  Q3. Is cell size a confounding factor?
      Larger cells present more membrane area but also dilute pore-entry dye
      into a bigger volume.  Two checks:
        (a) Do cell sizes vary by trap position?  If yes, size co-varies
            with field position and could explain an apparent gradient.
        (b) Within a condition, does body area predict uptake amplitude?
            A significant correlation means size is an independent predictor.

Electrode convention
--------------------
ELECTRODE_NEAR_TRAP = 18 means Trap 18 is physically closest to the
electrodes and Trap 1 is farthest.  Flip to 1 if your device is reversed.

Functions
---------
plot_trap_gradient(mechanics_df, results_dir)
    Per-trap scatter + summary line of uptake plateau (A) vs Trap_ID.
    One subplot column per EP condition.  Spearman ρ annotated.

plot_ep_spatial_heatmap(mechanics_df, results_dir)
    Heatmap: rows = experiment replicates, columns = Trap_ID,
    colour = uptake plateau A.  Reveals whether the spatial pattern is
    consistent across replicates or experiment-specific.

plot_cell_size_spatial(mechanics_df, results_dir)
    Left:  Cell body area vs Trap_ID — is size spatially uniform?
    Right: Uptake plateau A vs cell body area, coloured by Trap_ID — is
           size an independent predictor of uptake efficiency?
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats
from typing import Optional, List

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Which trap number sits closest to the electrodes?
# Default: Trap 18.  Change to 1 if your device is the other way around.
ELECTRODE_NEAR_TRAP: int = 18

SAVE_DPI  = 300
PANEL_DPI = 150

plt.style.use('seaborn-v0_8-whitegrid')

# Colour used to highlight the "near electrode" end of the channel
COLOR_NEAR = '#d62728'   # Red
COLOR_FAR  = '#1f77b4'   # Blue

# =============================================================================
# Internal helpers
# =============================================================================

def _ep_df(mechanics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return EP-only rows with Trap_ID populated.

    Why filter here rather than in each plot function?
    Centralising the filter means every spatial plot operates on
    exactly the same subset, so results are directly comparable.
    """
    ep = mechanics_df[mechanics_df['Condition_Type'] == 'EP'].copy()
    ep = ep.dropna(subset=['Trap_ID'])
    ep['Trap_ID'] = ep['Trap_ID'].astype(int)
    return ep


def _uptake_col_for_region(df: pd.DataFrame, region: str) -> Optional[str]:
    """
    Return the uptake amplitude column name for the requested region.

    Parameters
    ----------
    df     : the mechanics DataFrame (used only to verify the column exists)
    region : one of 'body', 'protrusion', or 'total'

    Column mapping
    --------------
    All three regions use the absolute ΔF amplitude (ADU, background-subtracted)
    rather than dF/F₀.  The dF/F₀ columns are kept in the CSV for backward
    compatibility but are unsuitable for spatial comparisons because each region
    divides by its own near-zero pre-pulse F₀, producing incomparable ratios.
    Volume-normalised intensity (ADU/μm³) removes the additional cell-size
    confound so the plateau amplitude reflects membrane permeability rather
    than cell volume.

    'body'       -> Uptake_Body_VolNorm_A
    'protrusion' -> Uptake_Prot_VolNorm_A
    'total'      -> Uptake_Total_VolNorm_A

    Returns None if the column is missing or all-NaN.
    """
    col_map = {
        'body'       : 'Uptake_Body_VolNorm_A',
        'protrusion' : 'Uptake_Prot_VolNorm_A',
        'total'      : 'Uptake_Total_VolNorm_A',
    }

    if region not in col_map:
        raise ValueError(f"Unknown region '{region}'. Choose from: {list(col_map)}")

    col = col_map[region]
    if col in df.columns and df[col].notna().any():
        return col

    return None


def _spearman_label(x: np.ndarray, y: np.ndarray) -> str:
    """
    Return a one-line annotation string: 'ρ = 0.42, p = 0.003'.
    Returns an empty string if there are fewer than 3 paired finite values.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return ""
    rho, p_val = stats.spearmanr(x[mask], y[mask])
    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
    return f"ρ = {rho:+.2f}, p = {p_str}"


def _near_far_label(ax, all_trap_ids: np.ndarray, y_pos: float):
    """
    Stamp 'far ◄' and '► near' text near the x-axis limits to indicate
    which end of the channel is closest to the electrode.

    We place the labels just inside the plot so they don't clip.
    """
    lo, hi = int(all_trap_ids.min()), int(all_trap_ids.max())
    if ELECTRODE_NEAR_TRAP >= hi:
        near_x, far_x = hi, lo
    else:
        near_x, far_x = lo, hi

    ax.text(far_x,  y_pos, '◄ far',  ha='center', va='top',
            fontsize=7, color='grey', style='italic')
    ax.text(near_x, y_pos, 'near ►', ha='center', va='top',
            fontsize=7, color='grey', style='italic')


def _unique_conditions(ep: pd.DataFrame) -> List[str]:
    """Return sorted list of unique EP condition strings."""
    if 'Condition' in ep.columns:
        return sorted(ep['Condition'].dropna().unique().tolist())
    return ['All EP']


# =============================================================================
# Plot 1: Trap gradient — uptake metric vs Trap_ID
# =============================================================================

def plot_trap_gradient(
    mechanics_df: pd.DataFrame,
    results_dir: Path,
    region: str = 'total',
) -> None:
    """
    Shows how uptake amplitude (dF/F0 plateau) varies across trap positions.

    Parameters
    ----------
    region : which ROI to plot — 'body', 'protrusion', or 'total' (default).
             Call this function once per region to produce three separate files.

    Layout:  one subplot per EP condition group.
    Per subplot:
      - Individual data points (semi-transparent, jittered on x for readability)
      - Per-trap median line
      - Linear trend line (OLS) with 95% confidence band
      - Spearman ρ annotation (robust to non-linear monotone trends)
      - Near / far electrode labels on x-axis

    Reading the plot
    ----------------
    Flat median line  → uniform field; EP is effectively bulk.
    Rising median     → field gradient; traps nearest the electrode
                        (Trap 18 by default) are more efficiently porated.
    High scatter      → cell-to-cell variation dominates over position;
                        localization is limited.
    """
    ep = _ep_df(mechanics_df)
    if ep.empty:
        logger.warning("plot_trap_gradient: no EP data found — skipping.")
        return

    uptake_col = _uptake_col_for_region(ep, region)
    if uptake_col is None:
        logger.warning(
            f"plot_trap_gradient: no uptake column for region='{region}' — skipping."
        )
        return

    # Choose a readable y-axis label
    y_label_map = {
        'Uptake_Total_VolNorm_A'   : 'Total Uptake Plateau (ADU/µm³)',
        'Uptake_Body_VolNorm_A'    : 'Body Uptake Plateau (ADU/µm³)',
        'Uptake_Prot_VolNorm_A'    : 'Protrusion Uptake Plateau (ADU/µm³)',
    }
    y_label = y_label_map.get(uptake_col, uptake_col)

    conditions = _unique_conditions(ep)
    n_conds = len(conditions)

    fig, axes = plt.subplots(
        1, n_conds,
        figsize=(max(5, 4 * n_conds), 5),
        sharey=False,
        squeeze=False
    )
    axes = axes[0]  # shape is (1, n_conds); take the single row

    for ax, cond in zip(axes, conditions):
        # Subset this condition
        sub = ep[ep['Condition'] == cond] if 'Condition' in ep.columns else ep
        sub = sub.dropna(subset=['Trap_ID', uptake_col])

        if sub.empty:
            ax.set_title(cond, fontsize=8)
            ax.set_visible(False)
            continue

        trap_ids = sub['Trap_ID'].values.astype(float)
        y_vals   = sub[uptake_col].values.astype(float)

        # --- Individual points with jitter so overlapping points are visible ---
        # np.random.seed makes the jitter reproducible between pipeline runs
        rng = np.random.default_rng(seed=42)
        jitter = rng.uniform(-0.25, 0.25, size=len(trap_ids))

        ax.scatter(
            trap_ids + jitter, y_vals,
            alpha=0.45, s=20, color='steelblue', zorder=2,
            label='Individual traps'
        )

        # --- Per-trap median line ---
        per_trap = sub.groupby('Trap_ID')[uptake_col].median().reset_index()
        per_trap.columns = ['Trap_ID', 'median']
        ax.plot(
            per_trap['Trap_ID'], per_trap['median'],
            'o-', color='navy', linewidth=1.5, markersize=5,
            zorder=3, label='Median per trap'
        )

        # --- Linear trend + 95% CI band ---
        # OLS trend gives an intuitive slope even if the true relationship
        # is non-linear; the Spearman ρ below catches any monotone trend.
        valid = np.isfinite(trap_ids) & np.isfinite(y_vals)
        if valid.sum() >= 3:
            slope, intercept, r, p, se = stats.linregress(
                trap_ids[valid], y_vals[valid]
            )
            x_fit  = np.linspace(trap_ids[valid].min(), trap_ids[valid].max(), 100)
            y_fit  = slope * x_fit + intercept

            # Approximate 95% CI: ±1.96 * SE_y
            # SE_y is the standard error of the fitted values:
            #   SE_y = se_slope * sqrt(sum((x - x_mean)^2)) for each x,
            # but a simpler uniform band using the residual std is good enough here.
            n_valid = valid.sum()
            residual_std = float(np.std(y_vals[valid] - (slope * trap_ids[valid] + intercept)))
            ci_half = 1.96 * residual_std / np.sqrt(n_valid)

            ax.plot(x_fit, y_fit, '--', color='tomato', linewidth=1.2,
                    zorder=4, label='Linear trend')
            ax.fill_between(x_fit, y_fit - ci_half, y_fit + ci_half,
                            alpha=0.12, color='tomato')

            spearman_str = _spearman_label(trap_ids[valid], y_vals[valid])
            ax.text(0.97, 0.97, spearman_str,
                    transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        # --- Near / far electrode labels ---
        # Place them below the x-axis tick labels using a small y-offset in
        # axis coordinates so they do not overlap with the data.
        ax.set_xlabel('Trap ID', fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)

        # Shorten long condition strings so they fit in the title
        short_cond = cond.replace('_', ' ')
        ax.set_title(short_cond, fontsize=8, pad=4)

        # Mark near/far electrode ends with vertical spans
        all_trap_ids_int = np.arange(1, 19)
        near_id = ELECTRODE_NEAR_TRAP
        far_id  = 1 if near_id == 18 else 18

        ax.axvspan(near_id - 0.5, near_id + 0.5,
                   alpha=0.08, color=COLOR_NEAR, label='Near electrode')
        ax.axvspan(far_id - 0.5,  far_id + 0.5,
                   alpha=0.08, color=COLOR_FAR,  label='Far from electrode')

        ax.set_xticks(sorted(sub['Trap_ID'].unique().astype(int)))
        ax.tick_params(axis='x', labelsize=7, rotation=45)
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(
        f'EP Uptake vs Trap Position\n'
        f'(Trap {ELECTRODE_NEAR_TRAP} = near electrode)',
        fontsize=10, y=1.02
    )
    fig.tight_layout()

    out_path = results_dir / f"spatial_trap_gradient_{region}.png"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


# =============================================================================
# Plot 2: Spatial heatmap — experiment × Trap_ID
# =============================================================================

def plot_ep_spatial_heatmap(
    mechanics_df: pd.DataFrame,
    results_dir: Path,
    region: str = 'total',
) -> None:
    """
    Heatmap where each row is one experiment replicate and each column is a
    Trap_ID.  The colour encodes uptake amplitude (absolute ΔF plateau, ADU).

    Each coloured cell is additionally annotated with two morphological numbers
    printed in small text:
        A: <initial cell body area in µm²>
        L: <maximum pre-pulse protrusion length in µm>

    This lets you visually check, for each individual trap, whether high uptake
    co-occurs with a large cell or a long protrusion — without needing a
    separate plot.

    Parameters
    ----------
    region : which ROI to plot — 'body', 'protrusion', or 'total' (default).

    Why this matters:
    -----------------
    The gradient plot (above) averages across replicates.  If the gradient
    only appears in some experiments (e.g. because electrode contact varied
    between chips), the heatmap will reveal replicate-to-replicate
    inconsistency that the average would hide.

    A consistent horizontal gradient across all rows → robust field asymmetry.
    A patchwork of colours → experiment-specific variation dominates.
    White cells → the trap was excluded (ruptured, QC-failed, or missing).

    One heatmap is saved per EP condition group so multi-condition datasets
    remain readable.
    """
    ep = _ep_df(mechanics_df)
    if ep.empty:
        logger.warning("plot_ep_spatial_heatmap: no EP data found — skipping.")
        return

    uptake_col = _uptake_col_for_region(ep, region)
    if uptake_col is None:
        logger.warning(
            f"plot_ep_spatial_heatmap: no uptake column for region='{region}' — skipping."
        )
        return

    y_label_map = {
        'Uptake_Total_VolNorm_A'   : 'Total Uptake Plateau (ADU/µm³)',
        'Uptake_Body_VolNorm_A'    : 'Body Uptake Plateau (ADU/µm³)',
        'Uptake_Prot_VolNorm_A'    : 'Protrusion Uptake Plateau (ADU/µm³)',
    }
    cbar_label = y_label_map.get(uptake_col, uptake_col)

    # Annotation columns: (column_name, short_prefix, format_string)
    # These will be overlaid as small text inside each coloured cell.
    # Both are always attempted; missing columns are silently skipped.
    ANN_SPECS = [
        ('Cell_Body_Area_PrePulse_um2', 'A', '{:.0f}µm²'),
        ('Max_Prot_Length_PrePulse_um', 'L', '{:.1f}µm'),
    ]

    conditions = _unique_conditions(ep)

    for cond in conditions:
        sub = ep[ep['Condition'] == cond] if 'Condition' in ep.columns else ep
        sub = sub.dropna(subset=['Trap_ID', uptake_col])
        if sub.empty:
            continue

        # --- Build pivot table: rows = experiment, columns = Trap_ID ---
        # 'Experiment_Folder' is the unique identifier for each replicate.
        # If multiple traps within the same experiment share the same Trap_ID
        # (should not happen, but just in case), take the mean.
        pivot = sub.pivot_table(
            index='Experiment_Folder',
            columns='Trap_ID',
            values=uptake_col,
            aggfunc='mean'
        )

        # Sort experiments chronologically (folder names start with YYMMDD)
        pivot = pivot.sort_index()

        # Ensure all trap columns 1-18 are present (fill missing with NaN)
        all_traps = list(range(1, 19))
        for t in all_traps:
            if t not in pivot.columns:
                pivot[t] = np.nan
        pivot = pivot[all_traps]  # enforce column order 1 → 18

        # --- Build annotation pivots (same grid, different values) ---
        # For each annotation column, build a pivot table that maps the same
        # (Experiment_Folder × Trap_ID) grid to the morphological value.
        # reindex() fills gaps with NaN so the indices align perfectly with
        # the main pivot even when some traps have no area/length data.
        ann_pivots = {}
        for col, prefix, fmt in ANN_SPECS:
            if col in sub.columns and sub[col].notna().any():
                ann_piv = sub.pivot_table(
                    index='Experiment_Folder',
                    columns='Trap_ID',
                    values=col,
                    aggfunc='mean',
                )
                ann_piv = ann_piv.reindex(index=pivot.index, columns=all_traps)
                ann_pivots[col] = (ann_piv, prefix, fmt)

        # --- Plot ---
        fig_h = max(3, 0.5 * len(pivot))   # scale height with number of experiments
        fig, ax = plt.subplots(figsize=(12, fig_h))

        # The global seaborn-whitegrid style switches gridlines on for every
        # axes object.  For an imshow heatmap the grid lines are drawn on top
        # of the pixel data and just chop up the cells visually, so turn them
        # off here for this axes only.
        ax.grid(False)

        # Use a sequential colourmap; white for NaN (missing traps)
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color='white')

        masked_data = np.ma.masked_invalid(pivot.values)
        im = ax.imshow(
            masked_data,
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
        )

        # --- Cell annotations ---
        # Overlay area and pre-pulse protrusion length as small text in each
        # coloured cell.  Text colour adapts to the cell brightness so it
        # stays legible on both pale-yellow (low uptake) and dark-red (high
        # uptake) cells.
        if ann_pivots:
            # Determine the data range to normalise cell brightness.
            # np.ma.getmaskarray always returns a full boolean array — avoids
            # the scalar-False quirk of masked_data.mask when nothing is masked.
            mask_arr = np.ma.getmaskarray(masked_data)
            valid_vals = masked_data.data[~mask_arr]
            v_min = float(np.nanmin(valid_vals)) if len(valid_vals) else 0.0
            v_max = float(np.nanmax(valid_vals)) if len(valid_vals) else 1.0
            v_range = max(v_max - v_min, 1e-9)

            for row_idx in range(len(pivot.index)):
                for col_idx in range(len(all_traps)):
                    if mask_arr[row_idx, col_idx]:
                        continue  # white (no-data) cell — nothing to annotate

                    # Choose black or white text based on how dark the cell is.
                    # YlOrRd transitions from pale yellow to dark red, so cells
                    # above ~55 % of the range are dark enough to need white text.
                    norm_val = (masked_data.data[row_idx, col_idx] - v_min) / v_range
                    txt_color = 'white' if norm_val > 0.55 else 'black'

                    # Collect annotation lines that have valid data for this cell
                    lines = []
                    for col, (ann_piv, prefix, fmt) in ann_pivots.items():
                        val = ann_piv.iloc[row_idx, col_idx]
                        if np.isfinite(val):
                            lines.append(f"{prefix}:{fmt.format(val)}")

                    if lines:
                        ax.text(
                            col_idx, row_idx,
                            '\n'.join(lines),
                            ha='center', va='center',
                            fontsize=4.5, color=txt_color,
                            linespacing=1.2,
                        )

        # Axis labels
        ax.set_xticks(range(len(all_traps)))
        ax.set_xticklabels(all_traps, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        # Shorten long folder names to fit on the y-axis
        short_labels = [str(idx)[:30] for idx in pivot.index]
        ax.set_yticklabels(short_labels, fontsize=6)
        ax.set_xlabel('Trap ID  (left = far from electrode, right = near)', fontsize=9)
        ax.set_ylabel('Experiment replicate', fontsize=9)

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(cbar_label, fontsize=8)

        # Annotate near/far electrode columns
        near_col_idx = all_traps.index(ELECTRODE_NEAR_TRAP)
        ax.axvline(near_col_idx, color=COLOR_NEAR, linewidth=1.5,
                   linestyle='--', alpha=0.7, label=f'Trap {ELECTRODE_NEAR_TRAP} (near electrode)')

        ax.set_title(
            f'Spatial Heatmap: {cond.replace("_", " ")}\n'
            f'(white = no data; Trap {ELECTRODE_NEAR_TRAP} = near electrode; '
            f'A=initial body area, L=max pre-pulse protrusion)',
            fontsize=9
        )

        fig.tight_layout()
        safe_cond = cond.replace(' ', '_').replace('/', '-')
        out_path = results_dir / f"spatial_heatmap_{safe_cond}_{region}.png"
        fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {out_path.name}")


# =============================================================================
# Plot 3: Cell size analysis — is body area a confound?
# =============================================================================

def plot_cell_size_spatial(
    mechanics_df: pd.DataFrame,
    results_dir: Path,
    region: str = 'total',
) -> None:
    """
    Two-panel figure per condition addressing whether cell size confounds the
    spatial gradient.

    Parameters
    ----------
    region : which ROI to plot — 'body', 'protrusion', or 'total' (default).
             The x-axis area column matches the region:
               body / total  → Cell_Body_Area_um2
               protrusion    → Cell_Prot_Area_um2

    Left panel — Region area vs Trap_ID:
        Tests whether cells are spatially sorted by size.

    Right panel — Uptake plateau vs region area, coloured by Trap_ID:
        Tests whether region size is an independent predictor of uptake.

    Interpretation guide
    --------------------
    Left ρ ≈ 0, right ρ ≈ 0  → size is not a confound; any gradient is real.
    Left ρ ≠ 0, right ρ ≠ 0  → size co-varies with position AND uptake;
                                 partial correlation or mixed model needed.
    Left ρ ≈ 0, right ρ ≠ 0  → size predicts uptake independently of position.
    """
    ep = _ep_df(mechanics_df)
    if ep.empty:
        logger.warning("plot_cell_size_spatial: no EP data — skipping.")
        return

    uptake_col = _uptake_col_for_region(ep, region)
    if uptake_col is None:
        logger.warning(
            f"plot_cell_size_spatial: no uptake column for region='{region}' — skipping."
        )
        return

    # Choose area column and axis label to match the region being plotted
    if region == 'protrusion':
        area_col   = 'Cell_Prot_Area_um2'
        area_label = 'Protrusion Area (µm²)'
    else:
        area_col   = 'Cell_Body_Area_um2'
        area_label = 'Cell Body Area (µm²)'

    if area_col not in ep.columns or ep[area_col].isna().all():
        logger.warning(
            f"plot_cell_size_spatial: '{area_col}' is missing or all NaN — skipping."
        )
        return

    y_label_map = {
        'Uptake_Total_VolNorm_A'   : 'Total Uptake Plateau (ADU/µm³)',
        'Uptake_Body_VolNorm_A'    : 'Body Uptake Plateau (ADU/µm³)',
        'Uptake_Prot_VolNorm_A'    : 'Protrusion Uptake Plateau (ADU/µm³)',
    }
    uptake_label = y_label_map.get(uptake_col, uptake_col)

    norm     = mcolors.Normalize(vmin=1, vmax=18)
    # turbo runs dark-blue → cyan → green → yellow → orange → red.
    # Unlike diverging maps (RdBu_r, coolwarm), it never passes through
    # white or light grey, so every trap ID stays clearly visible on a
    # white background.  Far traps (low ID) are blue; near traps (high ID)
    # are red — consistent with the colorbar label.
    cmap_pos = cm.turbo

    conditions = _unique_conditions(ep)

    for cond in conditions:
        sub = ep[ep['Condition'] == cond] if 'Condition' in ep.columns else ep
        sub = sub.dropna(subset=['Trap_ID', uptake_col, area_col])

        if len(sub) < 3:
            logger.warning(
                f"plot_cell_size_spatial: fewer than 3 rows for condition "
                f"'{cond}' — skipping."
            )
            continue

        trap_ids  = sub['Trap_ID'].values.astype(float)
        area_vals = sub[area_col].values.astype(float)
        y_vals    = sub[uptake_col].values.astype(float)

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

        # ---- LEFT: area vs trap position ----
        sc_left = ax_left.scatter(
            trap_ids, area_vals,
            c=trap_ids, cmap=cmap_pos, norm=norm,
            alpha=0.55, s=25, edgecolors='none'
        )
        per_trap_area = sub.groupby('Trap_ID')[area_col].median().reset_index()
        ax_left.plot(
            per_trap_area['Trap_ID'], per_trap_area[area_col],
            'k-o', linewidth=1.2, markersize=4, zorder=3, label='Median per trap'
        )
        spear_size = _spearman_label(trap_ids, area_vals)
        ax_left.text(0.97, 0.97, spear_size,
                     transform=ax_left.transAxes,
                     ha='right', va='top', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax_left.set_xlabel('Trap ID', fontsize=10)
        ax_left.set_ylabel(area_label, fontsize=10)
        ax_left.set_title('Cell Size vs Trap Position\n(Is size spatially uniform?)', fontsize=9)
        ax_left.legend(fontsize=8)
        fig.colorbar(sc_left, ax=ax_left, shrink=0.7).set_label('Trap ID', fontsize=8)

        # ---- RIGHT: uptake vs area, coloured by trap position ----
        # Floor values at 1 ADU before log-scaling so that near-zero or
        # negative baseline-subtracted fits don't produce log(0) errors.
        y_vals_plot = np.where(y_vals > 1e-3, y_vals, 1e-3)

        sc_right = ax_right.scatter(
            area_vals, y_vals_plot,
            c=trap_ids, cmap=cmap_pos, norm=norm,
            alpha=0.55, s=25, edgecolors='none'
        )
        spear_uptake = _spearman_label(area_vals, y_vals)
        ax_right.text(0.97, 0.97, spear_uptake,
                      transform=ax_right.transAxes,
                      ha='right', va='top', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        # Log scale reveals the spread that a linear axis hides when a few
        # high-amplitude outliers compress the majority of points near zero.
        # OLS is omitted: a linear fit drawn on a log axis is misleading;
        # the Spearman ρ annotation already quantifies the trend.
        ax_right.set_yscale('log')

        ax_right.set_xlabel(area_label, fontsize=10)
        ax_right.set_ylabel(uptake_label + '  [log scale]', fontsize=10)
        ax_right.set_title(
            f'Uptake vs Cell Size (coloured by trap position)\n'
            f'(Is cell size an independent predictor?)',
            fontsize=9
        )
        fig.colorbar(sc_right, ax=ax_right, shrink=0.7).set_label(
            'Trap ID (blue=far, red=near)', fontsize=8
        )

        short_cond = cond.replace('_', ' ')
        fig.suptitle(
            f'Cell Size as a Potential Confound — {short_cond}\n'
            f'(Trap {ELECTRODE_NEAR_TRAP} = near electrode)',
            fontsize=11, y=1.02
        )
        fig.tight_layout()

        safe_cond = cond.replace(' ', '_').replace('/', '-')
        out_path = results_dir / f"spatial_cell_size_{safe_cond}_{region}.png"
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {out_path.name}")