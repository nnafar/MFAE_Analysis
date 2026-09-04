# -*- coding: utf-8 -*-
"""
Plotting Module for Bulk MFAE Analysis.
"""

import logging
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu

import bulk_file_handling as bfh
import bulk_mechanics as bm
import bulk_utils as utils
from bulk_utils import MFA_COLORS

logger = logging.getLogger(__name__)

PALETTE_REGION = {
    "Body": "#1f77b4",       
    "Protrusion": "#ff7f0e", 
    "Total": "#333333"       
}

MI_MODEL_PALETTE = {
    "Linear":    "#e377c2",  
    "Power-Law": "#17becf",  
}

VISCO_MODEL_PALETTE = {
    "Kelvin-Voigt": MFA_COLORS['light_blue'],   
    "Jeffreys":     MFA_COLORS['medium_blue'],  
    "Burgers":      MFA_COLORS['dark_blue'],    
}

STYLE_DEFAULT = '-'      

COLOR_FIT_PRE = 'green'
COLOR_FIT_POST = 'magenta'

SAVE_DPI = 300
PANEL_DPI = 150

# ===========================================================================
# SuperPlot & pulse-trace infrastructure (Chapter 3)
# ===========================================================================
# Shared machinery for the Lord-et-al.-style SuperPlot figures and for the
# WT-condition trace figures (No pulse / 100 V, 100 µs / 100 V, 5 ms).
#
# SuperPlot conventions:
#   colour = Experiment #N (per-treatment ramp, deepest = earliest date)
#   shape  = Chip_ID (Chip 1 = circle, Chip 2 = triangle)
#   large marker per (Date, Chip) replicate at replicate mean
#   central bar = mean of replicate means; error bar = SEM of replicates
#   cell-level Mann-Whitney U on the bracket (Chapter 3 convention);
#   experiment-level MWU on replicate means is logged for the supplement.
#
# Trace conventions:
#   No pulse       -> #0a1e3d  (dark navy)
#   100 V, 5 ms    -> #1065ab  (deep blue)
#   100 V, 100 µs  -> #3a93c3  (mid blue)
# ---------------------------------------------------------------------------

# --- Full MFAE diverging palette (blue -> neutral -> red) ---
MFAE_PALETTE_FULL: List[str] = [
    "#1065ab", "#3a93c3", "#8ec4de", "#d1e5f0", "#e3dddd",
    "#fedbc7", "#f6a482", "#d75f4c", "#b31529",
]

# --- Per-treatment date ramps for SuperPlot experiment encoding ---
# Deepest shade = earliest experiment within a treatment. Six stops each;
# extend if a treatment exceeds six independent experiment-days.
WT_DATE_RAMP:   List[str] = ["#1065ab", "#3a93c3", "#5da5d5", "#8ec4de",
                             "#4a7bad", "#0a2e5a"]
CYTD_DATE_RAMP: List[str] = ["#b31529", "#d75f4c", "#f6a482", "#fedbc7",
                             "#8a1020", "#6b0c19"]

TREATMENT_RAMPS: Dict[str, List[str]] = {
    "WT":   WT_DATE_RAMP,
    "CytD": CYTD_DATE_RAMP,
}

# --- Chip_ID -> marker mapping (global, stable) ---
CHIP_MARKER_MAP: Dict[str, str] = {
    "Chip1": "o",
    "Chip2": "^",
    "Chip3": "s",
    "Chip4": "D",
    "Chip5": "v",
    "Chip6": "P",
}

# --- Fixed treatment order (WT before CytD, per Chapter 3 convention) ---
CATEGORY_ORDER: List[str] = ["WT", "CytD"]

# --- Pulse-condition trace palette (WT-condition comparisons) ---
NO_PULSE_COLOUR    = "#0a1e3d"   # dark navy
PULSE_5MS_COLOUR   = "#1065ab"   # deep blue
PULSE_100US_COLOUR = "#3a93c3"   # mid blue

# Indexed by (condition_type, duration_label). ASP is universal 'No pulse'.
PULSE_TRACE_PALETTE: Dict[Tuple[str, Optional[str]], str] = {
    ("ASP", None):    NO_PULSE_COLOUR,
    ("EP",  "5ms"):   PULSE_5MS_COLOUR,
    ("EP",  "100us"): PULSE_100US_COLOUR,
    ("EP",  "100µs"): PULSE_100US_COLOUR,
}

# --- Canonical WT-vs-CytD colours for ASP-only comparison figures ---
# Distinct from PULSE_TRACE_PALETTE: those darker/lighter shades are for
# comparing conditions within WT (No pulse vs 5 ms vs 100 µs). The pair
# below is for comparing treatments within the No-pulse (ASP) cohort.
ASP_TREATMENT_COLOUR: Dict[str, str] = {
    "WT":   "#1065ab",
    "CytD": "#b31529",
}

# --- Shared trace style (L(t) and I/F0 mean-trace figures) ---
# Reference style: plot_thesis_ep_wholetrace_by_fate (and via it the
# bulk_utils.get_line_kwargs helper, MFA_STYLE_LINE_LW = 1.8). Every
# trace figure in Chapter 3 pulls from these constants so line thickness,
# marker frequency, and fill translucency are visually identical
# figure-to-figure.
TRACE_LINEWIDTH        = utils.PLOT_STYLE["trace_linewidth"]
TRACE_MARKERSIZE       = utils.PLOT_STYLE["marker_size"]
TRACE_MARKEREDGE_WIDTH = utils.PLOT_STYLE["marker_edge_width"]
TRACE_MARKEVERY_FRAC   = 0.10   # one marker every 10% of the common grid
TRACE_BAND_ALPHA       = 0.18
TRACE_BASELINE_ALPHA   = 0.5    # No-pulse dimming when EP is co-plotted
TRACE_MARKER_BY_TREATMENT: Dict[str, str] = {"WT": "o", "CytD": "s"}
TRACE_LINESTYLE_BY_FATE:    Dict[str, str] = {
    "intact":         "-",
    "ruptured_post":  "--",
    "baseline":       "-",   # No-pulse baseline in pulse-comparison figures
}


def build_trace_line_kwargs(pulse_colour: str,
                            fate: str = "intact",
                            treatment: str = "WT",
                            is_baseline: bool = False) -> Dict:
    """
    Kwargs bundle for `ax.plot(...)` calls that draw a mean L(t) or mean
    I(t)/F0 trace. Every trace figure in Chapter 3 uses this so line
    thickness, marker size, and fate-encoded style are identical.

    Parameters
    ----------
    pulse_colour : str
        Hex colour from PULSE_TRACE_PALETTE (or ASP_TREATMENT_COLOUR for
        WT-vs-CytD comparisons).
    fate : {'intact', 'ruptured_post', 'baseline'}
        Drives linestyle and marker face:
            'intact' / 'baseline'  -> solid line, filled marker
            'ruptured_post'        -> dashed line, open marker with coloured edge
    treatment : {'WT', 'CytD'}
        Drives marker shape: circle for WT, square for CytD.
    is_baseline : bool
        If True, dims the whole line to TRACE_BASELINE_ALPHA. Used when
        the No-pulse trace appears alongside EP traces so it reads as
        background reference rather than a comparison condition.
    """
    marker    = TRACE_MARKER_BY_TREATMENT.get(treatment, "o")
    linestyle = TRACE_LINESTYLE_BY_FATE.get(fate, "-")
    alpha     = TRACE_BASELINE_ALPHA if is_baseline else 1.0
    if str(fate).lower() == "ruptured_post":
        return dict(color=pulse_colour, linestyle=linestyle,
                    linewidth=TRACE_LINEWIDTH,
                    marker=marker, markerfacecolor='none',
                    markeredgecolor=pulse_colour,
                    markeredgewidth=TRACE_MARKEREDGE_WIDTH,
                    markersize=TRACE_MARKERSIZE, alpha=alpha)
    return dict(color=pulse_colour, linestyle=linestyle,
                linewidth=TRACE_LINEWIDTH,
                marker=marker, markerfacecolor=pulse_colour,
                markeredgecolor='none', markeredgewidth=0,
                markersize=TRACE_MARKERSIZE, alpha=alpha)


def markevery_from(common_t) -> int:
    """One marker every ~10% of the common time grid (min 1)."""
    n = len(common_t)
    return max(1, int(n * TRACE_MARKEVERY_FRAC))

# --- Font sizes for thesis-ready figures -----------------------------------
# All 7 pt, which is \scriptsize in a 10 pt document. Figures are drawn at
# printed size, so these are the sizes that land on the page. Sourced from
# bulk_utils.PLOT_STYLE so there is one place to change them, and so this
# module cannot drift away from utils.set_paper_style().
FONT_BASE          = utils.PLOT_STYLE["fontsize_annot_pt"]
FONT_AXIS_TITLE    = utils.PLOT_STYLE["fontsize_title_pt"]
FONT_AXIS_LABEL    = utils.PLOT_STYLE["fontsize_label_pt"]
FONT_TICK          = utils.PLOT_STYLE["fontsize_tick_pt"]
FONT_LEGEND        = utils.PLOT_STYLE["fontsize_legend_pt"]
FONT_LEGEND_HEADER = utils.PLOT_STYLE["fontsize_legend_pt"]
FONT_BRACKET       = utils.PLOT_STYLE["fontsize_annot_pt"]


def apply_thesis_rcparams() -> None:
    """
    Apply the Chapter 3 defaults. Call once at the start of any plotting
    function that produces a thesis-ready figure.

    This used to set its own font sizes, which competed with
    utils.set_paper_style(): whichever ran last won, so the printed size of
    a figure depended on the call order inside each plotting function. It
    now delegates, so there is exactly one style definition in the pipeline.
    """
    utils.set_paper_style()


def parse_chip_id(experiment_folder: str) -> str:
    """
    Extract 'Chip1', 'Chip2', ... from an Experiment_Folder string of the
    form 'YYMMDD_CellType_Treatment_ChipN_ExperimentM-...'.  Returns
    'Unknown' if no such token is present.
    """
    for tok in str(experiment_folder).split("_"):
        if tok.startswith("Chip"):
            return tok
    return "Unknown"


def format_condition_label(condition_type: str,
                           voltage_v: Optional[float] = None,
                           duration_label: Optional[str] = None,
                           fate_status: Optional[str] = None,
                           with_fate: bool = False,
                           include_voltage: bool = True,
                           pre_post: bool = False) -> str:
    """
    Chapter 3 label convention:

        No pulse                          (ASP-only, any treatment)
        100 V, 100 µs                     (EP, no fate)
        100 V, 5 ms
        100 V, 100 µs (Intact)            (EP, with fate)
        100 V, 100 µs (Ruptured)
        Intact pre/post                   (paired comparison, pre_post=True)
        Ruptured pre/post

    Duration mapping: '100us' -> '100 µs', '5ms' -> '5 ms'.
    Fate mapping:     'intact' -> 'Intact', 'ruptured_post' -> 'Ruptured'.
    """
    fate_map = {"intact": "Intact", "ruptured_post": "Ruptured"}
    fate_disp = fate_map.get(str(fate_status).lower(), None) if fate_status else None

    if pre_post:
        return f"{fate_disp} pre/post" if fate_disp else "pre/post"

    if str(condition_type).upper() == "ASP":
        base = "No pulse"
    else:
        dur_disp = {"100us": "100 µs", "100µs": "100 µs",
                    "5ms": "5 ms"}.get(str(duration_label),
                                       str(duration_label) if duration_label else "")
        
        if include_voltage and voltage_v is not None:
            base = f"{int(voltage_v)} V, {dur_disp}"
        else:
            base = f"{dur_disp}"

    if with_fate and fate_disp:
        return f"{base} ({fate_disp})"
    return base


def _sp_build_style_maps(df: pd.DataFrame
                         ) -> Tuple[Dict[int, str], Dict[int, str], Dict[str, str]]:
    """
    For a filtered dataframe, assign:
      date_colours[date]: hex colour from the ramp of that Date's Treatment,
                          deepest shade for earliest date within treatment
      date_labels[date]:  'Experiment 1', 'Experiment 2', ... within treatment
      chip_markers[chip]: 'o', '^', 's', ...

    Requires columns: 'Date' (int), 'Treatment' (str), 'Chip_ID' (str).
    Raises ValueError if a Treatment has more dates than its ramp supports.
    """
    date_treatment = df.groupby("Date")["Treatment"].first().to_dict()

    dates_by_treatment: Dict[str, List[int]] = {}
    for d, t in date_treatment.items():
        dates_by_treatment.setdefault(str(t), []).append(int(d))
    for t in dates_by_treatment:
        dates_by_treatment[t] = sorted(dates_by_treatment[t])

    date_colours: Dict[int, str] = {}
    date_labels:  Dict[int, str] = {}
    for treatment, dates in dates_by_treatment.items():
        ramp = TREATMENT_RAMPS.get(treatment)
        if ramp is None:
            raise ValueError(
                f"No colour ramp defined for treatment '{treatment}'. "
                f"Extend TREATMENT_RAMPS in bulk_plotting.py."
            )
        if len(dates) > len(ramp):
            raise ValueError(
                f"Treatment '{treatment}' has {len(dates)} experiment-dates "
                f"but the ramp only provides {len(ramp)} colours. Extend "
                f"the ramp in bulk_plotting.py ({'WT_DATE_RAMP' if treatment == 'WT' else 'CYTD_DATE_RAMP'})."
            )
        for i, d in enumerate(dates):
            date_colours[d] = ramp[i]
            date_labels[d]  = f"Experiment {i + 1}"

    chip_ids = sorted(df["Chip_ID"].unique())
    fallback = ["o", "^", "s", "D", "v", "P", "X", "*"]
    chip_markers: Dict[str, str] = {}
    for i, c in enumerate(chip_ids):
        chip_markers[c] = CHIP_MARKER_MAP.get(c, fallback[i % len(fallback)])

    return date_colours, date_labels, chip_markers


def _sp_experiment_level_mw(rep_means_a: np.ndarray,
                            rep_means_b: np.ndarray) -> Tuple[float, int, int]:
    """Mann-Whitney U on replicate means. Returns (p, n_a, n_b)."""
    n_a, n_b = len(rep_means_a), len(rep_means_b)
    if n_a < 2 or n_b < 2:
        return float('nan'), n_a, n_b
    try:
        _, p = mannwhitneyu(rep_means_a, rep_means_b, alternative="two-sided")
    except ValueError:
        return float('nan'), n_a, n_b
    return p, n_a, n_b


# --- Geometry constants for panel rendering ---
_SP_JITTER_WIDTH               = 0.18
_SP_CELL_MARKER_SIZE           = 40
_SP_CELL_ALPHA                 = 0.50
_SP_REPLICATE_MARKER_SIZE      = 220
_SP_REPLICATE_OFFSET_HALFWIDTH = 0.18
_SP_MEAN_BAR_HALFWIDTH         = 0.32


def _sp_render_panel(ax,
                     panel_df: pd.DataFrame,
                     col: str,
                     categories: List[str],
                     date_colours: Dict[int, str],
                     chip_markers: Dict[str, str],
                     log_axis: bool,
                     rng: np.random.Generator,
                     mean_markersize: float = _SP_REPLICATE_MARKER_SIZE,) -> None:
    """
    Render a single SuperPlot panel on `ax`:
      - one small semi-transparent marker per cell (colour=Date, shape=Chip)
      - one large marker per (Date, Chip) replicate at replicate mean
      - central bar at mean-of-replicate-means + SEM error bar

    `panel_df` must contain the columns: 'Category', 'Date', 'Chip_ID',
    and the value column `col`.
    """
    all_values: List[float] = []

    for x_idx, cat in enumerate(categories):
        cat_df = panel_df[panel_df["Category"] == cat]
        if cat_df.empty:
            continue

        # Small cell-level dots
        for _, row in cat_df.iterrows():
            colour = date_colours.get(int(row["Date"]), "#7f7f7f")
            marker = chip_markers.get(row["Chip_ID"], "o")
            x = x_idx + rng.uniform(-_SP_JITTER_WIDTH, _SP_JITTER_WIDTH)
            ax.scatter(x, row[col],
                       s=mean_markersize, marker=marker,
                       facecolor=colour, edgecolor="none",
                       alpha=_SP_CELL_ALPHA, zorder=2)
            all_values.append(row[col])

        # Per-replicate mean markers with ordered horizontal offset
        replicate_keys = sorted(cat_df.groupby(["Date", "Chip_ID"]).groups.keys())
        n_reps = len(replicate_keys)
        if n_reps == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-_SP_REPLICATE_OFFSET_HALFWIDTH,
                                  _SP_REPLICATE_OFFSET_HALFWIDTH, n_reps)

        replicate_means: List[float] = []
        for (date, chip), x_off in zip(replicate_keys, offsets):
            rep_df = cat_df[(cat_df["Date"] == date) & (cat_df["Chip_ID"] == chip)]
            rep_mean = float(rep_df[col].mean())
            replicate_means.append(rep_mean)
            colour = date_colours.get(int(date), "#7f7f7f")
            marker = chip_markers.get(chip, "o")
            ax.scatter(x_idx + x_off, rep_mean,
                       s=mean_markersize, marker=marker,
                       facecolor=colour, edgecolor="black",
                       linewidth=1.6, alpha=1.0, zorder=4)

        # Central tendency + SEM
        # Explicit marker='' and linestyle='-' defends against
        # bulk_utils.set_paper_style() prop_cycles that would otherwise
        # inject default markers or dashed lines onto ax.plot() calls.
        if replicate_means:
            grand_mean = float(np.mean(replicate_means))
            sem = (float(np.std(replicate_means, ddof=1)
                         / np.sqrt(len(replicate_means)))
                   if len(replicate_means) > 1 else 0.0)
            ax.plot([x_idx - _SP_MEAN_BAR_HALFWIDTH,
                     x_idx + _SP_MEAN_BAR_HALFWIDTH],
                    [grand_mean, grand_mean],
                    color="black", lw=2.2, linestyle='-', marker='',
                    zorder=5)
            if sem > 0:
                ax.plot([x_idx, x_idx],
                        [grand_mean - sem, grand_mean + sem],
                        color="black", lw=1.6, linestyle='-', marker='',
                        zorder=5)
                cap = _SP_MEAN_BAR_HALFWIDTH * 0.35
                for y in (grand_mean - sem, grand_mean + sem):
                    ax.plot([x_idx - cap, x_idx + cap], [y, y],
                            color="black", lw=1.6, linestyle='-', marker='',
                            zorder=5)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.tick_params(axis='x', which='major', pad=6)
    if log_axis and all_values and min(all_values) > 0:
        ax.set_yscale("log")


def _sp_build_legend_handles(df: pd.DataFrame,
                             chips_present: List[str],
                             date_colours: Dict[int, str],
                             date_labels:  Dict[int, str],
                             chip_markers: Dict[str, str]) -> List[mlines.Line2D]:
    """
    Treatment-grouped legend:
        WT
          Experiment 1, 2, 3, ...
        CytD
          Experiment 1, 2, ...
        Chip
          Chip 1, Chip 2, ...
    """
    handles: List[mlines.Line2D] = []
    date_treatment = df.groupby("Date")["Treatment"].first().to_dict()

    for treatment in CATEGORY_ORDER:
        dates = sorted([int(d) for d, t in date_treatment.items()
                        if str(t) == treatment])
        if not dates:
            continue
        handles.append(mlines.Line2D([0], [0], marker="", linestyle="",
                                     label=r"$\bf{" + treatment + "}$"))
        for d in dates:
            handles.append(mlines.Line2D([0], [0], marker="o", linestyle="",
                                         markerfacecolor=date_colours[d],
                                         markeredgecolor="black",
                                         markersize=11,
                                         label=date_labels[d]))

    handles.append(mlines.Line2D([0], [0], marker="", linestyle="",
                                 label=r"$\bf{Chip}$"))
    for c in chips_present:
        marker = chip_markers[c]
        display_label = str(c).replace("Chip", "Chip ")
        handles.append(mlines.Line2D([0], [0], marker=marker, linestyle="",
                                     markerfacecolor="lightgray",
                                     markeredgecolor="black",
                                     markersize=11,
                                     label=display_label))
    return handles

def render_visco_parameter_superplot(
    df: pd.DataFrame,
    panels: List[Tuple[str, str, str]],
    output_pdf: Path,
    category_col: str = "Treatment",
    categories: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 10),
    grid_shape: Tuple[int, int] = (2, 3),
    rotation: float = 15,
    ncol=2,
    bracket_pairs: Optional[List[Tuple[int, int]]] = None,
    title_fontsize: Optional[float] = FONT_AXIS_TITLE,
    label_fontsize: Optional[float] = FONT_AXIS_LABEL,
    tick_fontsize: Optional[float] = FONT_TICK,
    legend_fontsize: Optional[float] = FONT_LEGEND,
    bracket_fontsize: Optional[float] = FONT_BRACKET,
    mean_markersize: float = _SP_REPLICATE_MARKER_SIZE,
) -> None:
    """
    Draw a SuperPlot grid (default 2x3 with the sixth slot for the legend).

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'Date', 'Treatment', 'Experiment_Folder' plus the
        value columns referenced by `panels`. A 'Chip_ID' column will be
        derived from 'Experiment_Folder' if missing.
    panels : list of (column, ylabel, title)
        One entry per parameter panel. Up to len(panels) panels are drawn;
        remaining grid slots hold the legend or are hidden.
    output_pdf : Path
        Destination PDF path.
    category_col : str
        DataFrame column carrying the x-axis category (default 'Treatment').
    categories : list of str, optional
        Ordered list of category values to draw. Defaults to CATEGORY_ORDER
        filtered by what's present.
    bracket_pairs : list of (i, j) index tuples, optional
        Which category pairs to draw statistical brackets between (indices
        into `categories`). Defaults:
          - 2 categories:  single bracket (0, 1)
          - >2 categories: adjacent pairs (0,1), (1,2), ...
        Pass an empty list to suppress all brackets.
    """
    apply_thesis_rcparams()

    df = df.copy()
    if "Chip_ID" not in df.columns:
        df["Chip_ID"] = df["Experiment_Folder"].apply(parse_chip_id)
    df["Category"] = df[category_col].astype(str)

    if categories is None:
        categories = [c for c in CATEGORY_ORDER if c in df["Category"].unique()]

    if bracket_pairs is None:
        if len(categories) == 2:
            bracket_pairs = [(0, 1)]
        elif len(categories) > 2:
            bracket_pairs = [(i, i + 1) for i in range(len(categories) - 1)]
        else:
            bracket_pairs = []

    date_colours, date_labels, chip_markers = _sp_build_style_maps(df)

    utils.check_panel_area(figsize, grid_shape[0], grid_shape[1])
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    axes = np.asarray(axes).flatten()
    rng = np.random.default_rng(seed=42)

    n_panels = len(panels)
    for i, ax in enumerate(axes):
        if i >= n_panels:
            ax.set_visible(False)
            continue

        col, ylabel, title = panels[i]
        panel_df = df[df[col].notna()].copy()
        if panel_df.empty:
            ax.set_visible(False)
            continue

        log_axis = (panel_df[col] > 0).all()
        _sp_render_panel(
            ax,
            panel_df,
            col,
            categories,
            date_colours,
            chip_markers,
            log_axis,
            rng,
            mean_markersize=mean_markersize,
        )

        # Apply configurable title, label, and tick font sizes
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=tick_fontsize)


        # Optional: Turn off minor tick labels completely if they clutter the axis
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        # --------------------

        # X-tick rotation only when labels are long enough to collide
        max_label_len = max((len(str(c)) for c in categories), default=0)
        if max_label_len > 8:
            ax.tick_params(axis="x", rotation=rotation, labelsize=tick_fontsize)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("center")

        # Brackets for the specified pairs
        for idx_a, idx_b in bracket_pairs:
            if idx_a >= len(categories) or idx_b >= len(categories):
                continue
            cat_a = categories[idx_a]
            cat_b = categories[idx_b]
            a = (
                panel_df.loc[panel_df["Category"] == cat_a, col]
                .dropna()
                .values
            )
            b = (
                panel_df.loc[panel_df["Category"] == cat_b, col]
                .dropna()
                .values
            )
            if len(a) < 2 or len(b) < 2:
                continue
            stars, label, lw, delta = _build_stat_label(a, b)

            rep_a = (
                panel_df[panel_df["Category"] == cat_a]
                .groupby(["Date", "Chip_ID"])[col]
                .mean()
                .values
            )
            rep_b = (
                panel_df[panel_df["Category"] == cat_b]
                .groupby(["Date", "Chip_ID"])[col]
                .mean()
                .values
            )
            p_exp, n_exp_a, n_exp_b = _sp_experiment_level_mw(rep_a, rep_b)

            logger.info(
                f"  [{title.splitlines()[0]:<30}] {cat_a} vs {cat_b}: "
                f"cells n=({len(a)},{len(b)})  stars={stars}  delta={delta:+.3f}  "
                f"exp n=({n_exp_a},{n_exp_b})  p_exp={p_exp:.4g}"
            )

            if label is not None:
                y_top = float(np.nanmax(np.concatenate([a, b])))
                _add_bracket(
                    ax, idx_a, idx_b, y_top, label, lw=lw, fontsize=bracket_fontsize
                )

    # Legend in the last unused slot
    if n_panels < len(axes):
        legend_ax = axes[-1]
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        chips_present = sorted(df["Chip_ID"].unique())
        handles = _sp_build_legend_handles(
            df, chips_present, date_colours, date_labels, chip_markers
        )
        legend_ax.legend(
            handles=handles,
            loc="center",
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
            handletextpad=0.8,
            labelspacing=0.7,
            title_fontsize=legend_fontsize,
        )

    plt.tight_layout()
    utils.save_plot_pdf(output_pdf, dpi=SAVE_DPI)
    plt.close(fig)

# ===========================================================================
# End SuperPlot & pulse-trace infrastructure
# ===========================================================================


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
    post_durations = []
    pre_starts     = []

    for trap in traps:
        ud = trap.uptake_data
        if 'Time_s' not in ud:
            continue
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

    if not post_durations:
        return 0.0, 1.0
    median_post = float(np.median(post_durations))
    threshold   = rupture_fraction * median_post
    normal      = [d for d in post_durations if d >= threshold]
    x_max       = float(min(normal)) if normal else float(min(post_durations))

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
    if arr is None or (isinstance(arr, np.ndarray) and len(arr) == 0):
        return default
    if isinstance(arr, (list, tuple)) and len(arr) == 0:
        return default
    try:
        val = float(np.nanmax(arr))
        return val if np.isfinite(val) else default
    except (ValueError, TypeError):
        return default

def _mw_stars(vals_a: np.ndarray, vals_b: np.ndarray) -> Tuple[str, float]:
    """Calculate two-sided Mann-Whitney U test p-value and significance stars."""
    if len(vals_a) < 2 or len(vals_b) < 2:
        return "ns", 1.0
    try:
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        stars = ("***" if p < 0.001 else
                 ("**" if p < 0.01  else
                  ("*"  if p < 0.05  else "ns")))
        return stars, float(p)
    except Exception:
        return "ns", 1.0

def _cliffs_delta(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    """Calculate non-parametric Cliff's Delta effect size."""
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a == 0 or n_b == 0:
        return 0.0

    a_col = np.asarray(vals_a)[:, np.newaxis]   
    b_row = np.asarray(vals_b)[np.newaxis, :]   
    dominance = np.sign(a_col - b_row)   
    return float(dominance.sum() / (n_a * n_b))

def _delta_linewidth(delta: float) -> float:
    """Scale bracket line thickness by Cliff's Delta thresholds (Romano et al.)."""
    abs_d = abs(delta)
    if abs_d < 0.147:   # Negligible
        return 1.2
    if abs_d < 0.330:   # Small
        return 2.2
    if abs_d < 0.474:   # Medium
        return 3.5
    return 4.8          # Large

def _add_bracket(ax, x1: float, x2: float, y_top: float, label: Optional[str],
                 lw: float = 0.9, color: str = 'black',
                 fontsize: Optional[float] = None, inset: float = 0.13) -> None:
    """Draw a complete statistical significance bracket with vertical ticks and label."""
    if label is None:
        return
    if fontsize is None:
        fontsize = globals().get('FONT_BRACKET', 10)

    x1_drawn = x1 + inset if x2 > x1 else x1 - inset
    x2_drawn = x2 - inset if x2 > x1 else x2 + inset

    y_lo, y_hi = ax.get_ylim()

    if ax.get_yscale() == 'log':
        y_top_safe = max(y_top, 1e-9)
        log_lo, log_hi = np.log10(max(y_lo, 1e-9)), np.log10(max(y_hi, 1e-9))
        log_range = log_hi - log_lo
        y_line = 10 ** (np.log10(y_top_safe) + log_range * 0.08)
        y_text = 10 ** (np.log10(y_top_safe) + log_range * 0.13)
        needed_top = 10 ** (np.log10(y_text) + log_range * 0.08)
    else:
        y_range = y_hi - y_lo
        y_line  = y_top + y_range * 0.08
        y_text  = y_top + y_range * 0.13
        needed_top = y_text + y_range * 0.08

    # Draws tick-down legs to anchor y_top alongside the horizontal bar
    ax.plot([x1_drawn, x1_drawn, x2_drawn, x2_drawn],
            [y_top, y_line, y_line, y_top],
            lw=lw, color=color)
    
    ax.text((x1 + x2) / 2, y_text, label,
            ha='center', va='bottom', fontsize=fontsize, color=color)

    if needed_top > y_hi:
        ax.set_ylim(y_lo, needed_top)

def _build_stat_label(vals_a: np.ndarray, vals_b: np.ndarray, show_p_val: bool = False
                      ) -> Tuple[str, Optional[str], float, float]:
    """Build statistical annotation string, bracket line weight, and effect size."""
    stars, p_val = _mw_stars(vals_a, vals_b)
    delta = _cliffs_delta(vals_a, vals_b)
    
    label = None if stars == "ns" else (f"p={p_val:.3g}" if show_p_val else stars)
    lw = _delta_linewidth(delta)
    return stars, label, lw, delta


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
        
        df_global = df.copy()
        df_global['Trap'] = 'All Traps'
        df_global['Trap_Int'] = 999 
        df_combined = pd.concat([df, df_global], ignore_index=True)
        
        trap_order = [f"T{t}" for t in sorted(df['Trap_Int'].unique())] + ['All Traps']
        
        plt.figure(figsize=(16, 7))
        
        if is_ep:
            palette = {'Pre-pulse': '#ff7f0e', 'Post-pulse': '#9467bd'}
            
            sns.boxplot(data=df_combined, x='Trap', y='Length_um', hue='Phase', palette=palette, 
                        showfliers=False, boxprops=dict(alpha=0.3), order=trap_order)
            
            plotted_pairs = set()
            growth_thresh = 0.5
            trap_to_x = {t: i for i, t in enumerate(trap_order)}
            
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
                        if delta > growth_thresh: color = '#2ca02c' 
                        elif delta < -growth_thresh: color = '#d62728' 
                        else: color = '#7f7f7f' 
                        
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
                        ud['Time_s'], ud[col_name], pulse_time_s=pf
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

            pf = (trap.metadata.pulse_frame
                  if trap.metadata.condition_type == "EP"
                  else 0)

            region_cols = [
                ('Body',       'Body_VolNorm'),
                ('Protrusion', 'Protrusion_VolNorm'),
                ('Total',      'Total_VolNorm'),
            ]

            for region_name, col_name in region_cols:
                if col_name not in ud or len(ud[col_name]) == 0:
                    continue

                fit = bm.fit_exponential_uptake(
                    ud['Time_s'], ud[col_name], pulse_time_s=pf
                )

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

        df['Amp_A'] = df['Amp_A'].clip(lower=1e-3)

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


def plot_uptake_fits_multipanel(grouped_data, output_dir: Path):
    import bulk_mechanics as bm

    logger.info("Generating Plot: Multipanel Uptake Fits...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)

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
    logger.info("Generating Plot: Multipanel Recoil Fits...")
    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
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


def _cond_label_from_row(row: pd.Series) -> str:
    if 'Condition' in row.index and pd.notna(row.get('Condition')):
        return row['Condition']
    if row.get('Condition_Type') == 'ASP':
        return f"{row['Cell_Type']}_{row['Treatment']}_{row['Pressure_Pa']}Pa_ASP"
    return (f"{row['Cell_Type']}_{row['Treatment']}_"
            f"{row['Pressure_Pa']}Pa_{row['Voltage_V']}V_{row['Duration_label']}")


def _asp_category_label(meta: bfh.ExperimentMetadata) -> str:
    return f"{meta.cell_type}_{meta.treatment}_{meta.pressure}Pa"


def _asp_category_label_from_row(row: pd.Series) -> str:
    return f"{row['Cell_Type']}_{row['Treatment']}_{row['Pressure_Pa']}Pa"


def _reduce_labels(labels: List[str], sep: str = '_') -> Dict[str, str]:
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
        r_eff: float, C: float = 1.0,
        # --- New Font Size Parameters ---
        legend_size: int      = 14,
        axis_label_size: int  = 14,
        panel_title_size: int = 14,
        annotation_size: int  = 14,
        tick_label_size: int  = 14,
        ) -> None:
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

    model_color = VISCO_MODEL_PALETTE

    _pool_labels_full = sorted(asp_pools.keys())
    _pool_label_map = _reduce_labels(_pool_labels_full)

    for cat_label in _pool_labels_full:
        cat_label_short = _pool_label_map[cat_label]
        trap_list_all = asp_pools[cat_label]

        common_dur = bm.TARGET_WINDOW_S

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
        
        # Updated legend font size
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=legend_size)
        
        dur_note = f"  [window: {common_dur:.0f} s]" if common_dur else ""
        
        # Updated figure-level axis label font sizes
        fig.text(0.5, 0.01, f'Time from entry (s){dur_note}', ha='center', fontsize=axis_label_size)
        fig.text(0.01, 0.5, 'Protrusion Length (µm)', va='center', rotation='vertical', fontsize=axis_label_size)

        for i, tid in enumerate(sorted_ids):
            ax = axes[i]
            
            # Set size for subpanel numeric tick labels
            ax.tick_params(axis='both', which='both', labelsize=tick_label_size)
            
            # Updated subpanel title font size
            ax.set_title(f"Trap {tid}", fontsize=panel_title_size, fontweight='bold')

            for trap in trap_groups[tid]:
                pd_data = trap.protrusion_data
                if ('Time_s' not in pd_data or 'Protrusion_Length_um' not in pd_data):
                    continue

                t_raw = pd_data['Time_s']
                l_raw = pd_data['Protrusion_Length_um']

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
                    # Updated failed-fit annotation font size
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

                ax.plot(t_smooth, l_pred, color=color, lw=2, alpha=0.85)
                r2_str = f"R²={visco['best_r2']:.2f}" if visco['best_r2'] is not None else ""
                
                # Updated fit result annotation font size
                ax.text(0.05, 0.9, f"{best}  {r2_str}", transform=ax.transAxes, fontsize=annotation_size, fontweight='bold', color=color)

        for j in range(n, len(axes)): axes[j].axis('off')
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        utils.save_plot_pdf(output_dir / f"ASP_BestFit_Panel_{cat_label_short}.pdf", dpi=PANEL_DPI)
        plt.close()

def plot_asp_parameter_boxplots(
    mechanics_df: pd.DataFrame,
    output_dir: Path,
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
    bracket_fontsize: Optional[float] = None,
) -> None:
    """
    ASP-only viscoelastic parameter SuperPlot (WT vs CytD) plus the
    stacked-bar model-selection frequency chart.

    The SuperPlot replaces the previous box+strip figure but writes to the
    same filename (Thesis_ASP_Visco_Parameter_Boxplots.pdf) so LaTeX
    references remain valid. Each cell is a small semi-transparent marker;
    colour encodes the experiment-day within treatment and shape encodes
    the chip. Large filled markers mark per-replicate means; the black bar
    is the mean of replicate means with an SEM error bar.

    Statistics on the bracket are cell-level Mann-Whitney U (the Chapter 3
    main-text convention); experiment-level MWU on replicate means is
    logged for the supplement.
    """
    logger.info("Generating: ASP Parameter SuperPlot...")

    df = mechanics_df[
        (mechanics_df["Condition_Type"] == "ASP")
        & mechanics_df["Best_Model"].notna()
        & mechanics_df["E_Pa"].notna()
        & (mechanics_df["Visco_R2_Flag"] == True)
    ].copy()

    if df.empty:
        logger.info("  No fitted ASP data — skipping.")
        return

    # ---- SuperPlot ---------------------------------------------------------
    # Panel spec matches the pre-pulse visco figure so the two are directly
    # side-by-side readable in the thesis.
    # Titles are single-line: at 7 pt on a 1.1 in panel a second title
    # line costs about a fifth of the panel height. The model
    # qualifications that used to sit on line two ("Burgers only",
    # "Jeffreys / Burgers") belong in the figure caption instead.
    asp_visco_panels: List[Tuple[str, str, str]] = [
        ('E_Pa',      r'$E$ (Pa)',                'Parallel spring'),
        ('E1_Pa',     r'$E_{1}$ (Pa)',            'Maxwell spring'),
        ('eta1_Pa_s', r'$\eta_{1}$ (Pa$\cdot$s)', 'Parallel dashpot'),
        ('eta2_Pa_s', r'$\eta_{2}$ (Pa$\cdot$s)', 'Flow viscosity'),
        ('Tau_s',     r'$\tau$ (s)',              'Relaxation time'),
    ]
    # 3 rows x 2 columns (5 parameters + legend) so the panel is taller
    # than it is wide and matches the stacked height of Figure 1.6A+B.
    # The 3.35 in height is the one number to adjust if the two columns of
    # the LaTeX figure do not come out level.
    # Increased height_in from 3.35 -> 6.5 to give 3 vertical rows adequate height
    render_visco_parameter_superplot(
        df=df,
        panels=asp_visco_panels,
        output_pdf=output_dir / "Thesis_ASP_Visco_Parameter_Boxplots.pdf",
        category_col='Treatment',
        figsize=utils.canvas_size_for(0.85, height_in=6.50),
        grid_shape=(3, 2),
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        bracket_fontsize=bracket_fontsize,
        mean_markersize=60.0,
    )

    # ---- Model-selection frequency -----------------------------------------
    df['Category'] = df.apply(_asp_category_label_from_row, axis=1)
    sorted_cats_full = sorted(df['Category'].unique())
    label_map = _reduce_labels(sorted_cats_full)
    df['Category'] = df['Category'].map(label_map)
    sorted_cats = [label_map[c] for c in sorted_cats_full]

    model_counts = (
        df.groupby(["Category", "Best_Model"])
        .size()
        .unstack(fill_value=0)
        .reindex(sorted_cats, fill_value=0)
    )
    cols_present = [
        c
        for c in ["Kelvin-Voigt", "Jeffreys", "Burgers"]
        if c in model_counts.columns
    ]

    fig, ax = plt.subplots(
        # figsize=utils.canvas_size_for(1.0, height_in=2.50)
    )
    model_counts[cols_present].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[VISCO_MODEL_PALETTE[c] for c in cols_present],
    )

    # 1. Y-Axis Label
    lbl_font = label_fontsize if label_fontsize is not None else getattr(utils, "FONT_LABEL", None)
    ax.set_ylabel("Number of Traps", fontsize=lbl_font*1.6)
    ax.set_xlabel("", fontsize=lbl_font*1.6)

    # 2. Tick Labels
    ax.tick_params(axis="both", labelsize=tick_fontsize*1.6)
    # Align labels horizontally on the same baseline directly beneath ticks
    plt.setp(
        ax.get_xticklabels(),
        rotation=0,
        ha="center",
        va="top",
        fontsize=tick_fontsize * 1.6,
    )
    # Pass fontsize directly to plt.xticks to prevent rotation from overriding tick size
    plt.xticks(rotation=0, ha="center", fontsize=tick_fontsize*1.6)

    # 3. Legend (Horizontal arrangement below the plot)
    leg_kw = {"fontsize": legend_fontsize * 1.6} if legend_fontsize is not None else {}
    ax.legend(
        ncol=len(cols_present),        # Places entries side-by-side in one row
        loc="upper center",            # Anchors top of legend box
        bbox_to_anchor=(0.5, -0.05),   # Shifts legend below the x-axis labels
        frameon=False,                 # Removes outer box border for a clean look
        **leg_kw
    )

    # 4. Title (if present)
    if title_fontsize is not None:
        ax.set_title(ax.get_title(), fontsize=title_fontsize)

    plt.tight_layout()
    utils.save_plot_pdf(
        output_dir / "Thesis_ASP_Visco_Model_Selection_Frequency.pdf",
        dpi=SAVE_DPI,
    )
    plt.close()

def plot_asp_actin_f0_boxplots(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    ASP-only actin F0 SuperPlot (WT vs CytD): body region (F0_Body) and
    protrusion region (F0_Prot). Uses the same cohort filter as
    `plot_asp_parameter_boxplots` so cell counts match the visco figure.
    """
    logger.info("Generating: ASP Actin F0 SuperPlot...")

    df = mechanics_df[
        (mechanics_df['Condition_Type'] == 'ASP') &
        mechanics_df['Best_Model'].notna() &
        mechanics_df['E_Pa'].notna() &
        (mechanics_df['Visco_R2_Flag'] == True)
    ].copy()

    if df.empty:
        logger.info("  No fitted ASP data — skipping.")
        return

    panels: List[Tuple[str, str, str]] = [
        ('F0_Body', r'$F_{0,\mathrm{body}}$ (ADU)', 'Cell-Body Actin Baseline'),
        ('F0_Prot', r'$F_{0,\mathrm{prot}}$ (ADU)', 'Protrusion Actin Baseline'),
    ]

    # Two panels + one legend slot -> 1x3 grid.
    render_visco_parameter_superplot(
        df=df,
        panels=panels,
        output_pdf=output_dir / "Thesis_ASP_Actin_F0_Boxplots.pdf",
        category_col='Treatment',
        figsize=(15, 5.5),
        grid_shape=(1, 3),
    )


def _prepulse_visco_category_label(row: pd.Series) -> str:
    """
    Legacy fate-aware category label for the pre-pulse viscoelastic boxplot
    (kept for backward compatibility). Returns 'ASP WT', 'ASP CytD',
    'EP-pre 100V 100us (intact)', etc.
    """
    ct = row.get('Condition_Type')
    if ct == 'ASP':
        return f"ASP {row.get('Treatment', '?')}"
    base = f"EP-pre {row.get('Voltage_V')}V {row.get('Duration_label')}"
    if row.get('Fate_Status') == "ruptured_post":
        return f"{base} (ruptured)"
    return f"{base} (intact)"


def _prepulse_new_category_label(row: pd.Series) -> str:
    """
    Chapter 3 new-convention category label for the pre-pulse cohort.
    ASP rows -> 'No pulse' (CytD baseline is annotated ' (CytD)' when
    include_cytd=True keeps it in the cohort). EP rows -> the full
    label from format_condition_label with fate suffix.
    """
    ct = str(row.get('Condition_Type', '')).upper()
    treatment = str(row.get('Treatment', ''))
    if ct == 'ASP':
        label = format_condition_label(condition_type='ASP')
        return label if treatment == 'WT' else f"{label} ({treatment})"
    return format_condition_label(
        condition_type='EP',
        voltage_v=row.get('Voltage_V'),
        duration_label=row.get('Duration_label'),
        fate_status=row.get('Fate_Status'),
        with_fate=True,
        include_voltage=False
    )


def _prepulse_category_order(df: pd.DataFrame) -> List[str]:
    """
    Ordered category list for the pre-pulse SuperPlot:
        No pulse
        No pulse (CytD)          [only if present]
        100 V, 100 µs (Intact)
        100 V, 100 µs (Ruptured)
        100 V, 5 ms (Intact)
        100 V, 5 ms (Ruptured)
    Uses the actual pulse durations present in the dataframe to build the
    EP entries so the function generalises beyond the current two durations.
    """
    present = list(df['Category'].unique())
    ordered: List[str] = []

    nopulse_wt = format_condition_label(condition_type='ASP')
    if nopulse_wt in present:
        ordered.append(nopulse_wt)
    for cyt_candidate in (f"{nopulse_wt} (CytD)",):
        if cyt_candidate in present:
            ordered.append(cyt_candidate)

    # Sort EP durations: shorter pulses first (100us -> 5ms).
    dur_priority = {'100us': 0, '100µs': 0, '5ms': 1}
    dur_labels = sorted(
        {str(d) for d in df.loc[df['Condition_Type'] == 'EP',
                                'Duration_label'].dropna().unique()},
        key=lambda d: dur_priority.get(d, 99),
    )
    for dur in dur_labels:
        for fate in ('intact', 'ruptured_post'):
            lab = format_condition_label(
                condition_type='EP', voltage_v=100,
                duration_label=dur, fate_status=fate, with_fate=True)
            if lab in present:
                ordered.append(lab)

    # Append anything else that happens to be present (safety net).
    for c in present:
        if c not in ordered:
            ordered.append(c)
    return ordered

def plot_prepulse_visco_parameter_boxplots(
    mechanics_df: pd.DataFrame,
    output_dir: Path,
    include_cytd: bool = False,
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
) -> None:
    r"""
    Pre-pulse viscoelastic parameter SuperPlot (Chapter 3 Claim 2).

    Compares 'No pulse' (WT ASP baseline; and optionally CytD baseline)
    against the EP-pre cohorts split by fate:
        No pulse
        100 V, 100 µs (Intact)
        100 V, 100 µs (Ruptured)
        100 V, 5 ms (Intact)
        100 V, 5 ms (Ruptured)

    Uses the matched-horizon PrePulse_* fits so cohorts enter the
    comparison with matched fit windows.
    """
    logger.info("Generating: Pre-Pulse Viscoelastic Parameter SuperPlot...")

    # Filter for valid fit models, non-null modulus values, and passed R2 flags
    df = mechanics_df[
        mechanics_df['PrePulse_Best_Model'].notna() &
        mechanics_df['PrePulse_E_Pa'].notna() &
        (mechanics_df['PrePulse_Visco_R2_Flag'] == True)
    ].copy()

    # Optionally exclude non-WT ASP baselines (e.g., CytD)
    if not include_cytd:
        df = df[~((df['Condition_Type'] == 'ASP') &
                  (df['Treatment'] != 'WT'))].copy()

    if df.empty:
        logger.info("  No cells pass pre-pulse viscoelastic filter — skipping.")
        return

    # Assign custom labels and retrieve category ordering
    df['Category'] = df.apply(_prepulse_new_category_label, axis=1)
    categories = _prepulse_category_order(df)

    # Define panel configurations: (column_name, y_axis_label, panel_title)
    panels = [
        ('PrePulse_E_Pa',        r'$E$ (Pa)', 'Parallel Spring Modulus'),
        ('PrePulse_E1_Pa',       r'$E_{1}$ (Pa)', 'Burgers Maxwell Spring\n(Burgers only)'),
        ('PrePulse_eta1_Pa_s',   r'$\eta_{1}$ (Pa$\cdot$s)', 'Parallel Dashpot'),
        ('PrePulse_eta2_Pa_s',   r'$\eta_{2}$ (Pa$\cdot$s)', 'Flow Viscosity\n(Jeffreys / Burgers)'),
        ('PrePulse_Tau_s',       r'$\tau$ (s)', 'Characteristic Time'),
    ]

    # --- ADD THIS FIX ---
    # Force all panel columns to be numeric floats. 
    # Invalid strings become np.nan, preventing the ufunc 'isnan' crash.
    for col_name, _, _ in panels:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    # --------------------

    # Forward the font arguments down to the rendering function:
    render_visco_parameter_superplot(
        df=df,
        panels=panels,
        output_pdf=output_dir / "Thesis_PrePulse_Visco_Parameter_Boxplots.pdf",
        category_col='Category',
        categories=categories,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        bracket_fontsize = tick_fontsize,
        grid_shape=(2,3),
        rotation=20
    )

def plot_model_independent_fits_multipanel(
        grouped_data: Dict, output_dir: Path, global_pre_dur: Optional[float] = None) -> None:
    import bulk_mechanics as bm

    logger.info("Generating Plot: Model-Independent Fits Multipanel...")

    for key in sorted(grouped_data.keys()):
        traps = grouped_data[key]
        if not traps: continue
        cond_label = get_cond_label(traps[0].metadata)
        meta = traps[0].metadata

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

                if meta.condition_type == "ASP":
                    if global_pre_dur is not None:
                        t_from_entry = t_raw - t_raw[0]
                        win_mask = t_from_entry <= global_pre_dur
                        t_raw = t_raw[win_mask]
                        l_raw = l_raw[win_mask]
                    t_fit_raw = t_raw - t_raw[0]
                    l_fit_raw = l_raw
                else:
                    pf = trap.metadata.pulse_frame
                    if not (0 < pf < len(t_raw)):
                        continue
                    t_aligned = t_raw - t_raw[pf]
                    pre_mask = (t_aligned < 0) & (l_raw > 0)
                    if global_pre_dur is not None:
                        pre_mask = pre_mask & (t_aligned >= -global_pre_dur)
                    if np.sum(pre_mask) < 15:
                        continue
                    t_fit_raw = t_aligned[pre_mask]
                    t_fit_raw = t_fit_raw - t_fit_raw[0]
                    l_fit_raw = l_raw[pre_mask]

                t_clean, l_clean = bm._clean_trace(t_fit_raw, l_fit_raw)
                if len(t_clean) < 15:
                    continue

                ax.plot(t_clean, l_clean, 'o', c='black', ms=2, alpha=0.35)

                mi_fit = bm.fit_model_independent(t_clean, l_clean)
                t_smooth = np.linspace(t_clean[0], t_clean[-1], 200)

                if mi_fit['linear']:
                    p = mi_fit['linear']['params']
                    ax.plot(t_smooth, bm._linear(t_smooth, p['slope'], p['intercept']),
                            color=MI_MODEL_PALETTE['Linear'], lw=1.5, alpha=0.8)

                if mi_fit['power_law']:
                    p = mi_fit['power_law']['params']
                    ax.plot(t_smooth, bm._power_law(t_smooth, p['a'], p['exponent_b'], p['c']),
                            color=MI_MODEL_PALETTE['Power-Law'], lw=1.5, alpha=0.8)

                winner = mi_fit.get('best_model')
                if winner:
                    color = MI_MODEL_PALETTE.get(winner, 'black')
                    ax.text(0.05, 0.9, f"AICc: {winner}",
                            transform=ax.transAxes, fontsize=7,
                            fontweight='bold', color=color)

        for j in range(n, len(axes)): axes[j].axis('off')
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"MI_Fits_Panel_{cond_label}.png", dpi=PANEL_DPI, bbox_inches='tight')
        plt.close()


def plot_model_independent_comparison(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating: Model-Independent Fit Comparison...")
    if mechanics_df is None or mechanics_df.empty: return

    df = mechanics_df.copy()
    df['Cond'] = df.apply(_cond_label_from_row, axis=1)
    sorted_conds_full = sorted(df['Cond'].unique())

    label_map = _reduce_labels(sorted_conds_full)
    df['Cond'] = df['Cond'].map(label_map)
    sorted_conds = [label_map[c] for c in sorted_conds_full]

    df_lin = df[df['Linear_Slope'].notna()]
    df_pl  = df[df['PL_b'].notna()]

    if df_lin.empty and df_pl.empty: return

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
    
    cols_to_keep = [col for col in nice_names.keys() if col in df_scalars.columns]
    
    df_clean = df_scalars[['Condition'] + cols_to_keep].copy() if 'Condition' in df_scalars.columns else df_scalars[cols_to_keep].copy()
    df_clean = df_clean.rename(columns=nice_names)
    
    def _plot_and_save(df_subset: pd.DataFrame, filename_prefix: str, title: str):
        df_numeric = df_subset.select_dtypes(include=[np.number])
        df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]
        
        if df_numeric.shape[1] < 2: 
            return
            
        corr_matrix = df_numeric.corr(method='spearman')
        
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
        df_num = df_subset.copy()
        for col in df_num.columns:
            if col != 'Condition':
                df_num[col] = pd.to_numeric(df_num[col], errors='coerce').astype(float)
        
        df_numeric = df_num.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        cols_with_variance = []
        for col in df_numeric.columns:
            if df_numeric[col].dropna().nunique() > 1:
                if (df_numeric[col].max() - df_numeric[col].min()) > 1e-9:
                    cols_with_variance.append(col)
        
        df_numeric = df_numeric[cols_with_variance]
        
        if df_numeric.shape[1] < 2: 
            return

        g = sns.pairplot(df_numeric, corner=True, diag_kind='kde',
                         plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5})
        
        for i, col_name in enumerate(df_numeric.columns):
            ax = g.axes[i, 0] 
            if ax is not None:
                ax.set_ylabel(col_name, fontsize=12, fontweight='bold', rotation=90)
                ax.tick_params(axis='y', left=True, labelleft=False)
                ax.get_yaxis().set_visible(True)

        for j, col_name in enumerate(df_numeric.columns):
            ax = g.axes[-1, j]
            if ax is not None:
                ax.set_xlabel(col_name, fontsize=12, fontweight='bold')

        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(g.fig)

    if 'Condition' in df_clean.columns:
        for cond in sorted(df_clean['Condition'].dropna().unique()):
            df_cond = df_clean[df_clean['Condition'] == cond]
            if len(df_cond) >= 3:
                _plot_and_save_pairplot(df_cond, f"spearman_parameter_pairplot_{str(cond).replace(' ', '_')}.png", f"Pair Plot - {cond}")


def _find_body_size_column(pd_data: Dict[str, np.ndarray]) -> Optional[str]:
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
    for col in pd_data:
        col_l = col.lower()
        if 'body' in col_l and any(k in col_l for k in ('length', 'area', 'axis', 'width')):
            return col
    return None


def plot_ep_post_pulse_behavior(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
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

    label_map = _reduce_labels(sorted_conds_full)
    df['Cond'] = df['Cond'].map(label_map)
    sorted_conds = [label_map[c] for c in sorted_conds_full]

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

    counts.plot(kind='bar', stacked=True, ax=axes[0], color=colors, legend=False)
    axes[0].set_title("Post-Pulse Protrusion Behavior\n(raw counts)", fontweight='bold')
    axes[0].set_ylabel("Number of Traps")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)
    for j, cond in enumerate(sorted_conds):
        n_total = int(totals.loc[cond])
        axes[0].text(j, n_total + 0.2, f"n={n_total}", ha='center',
                     va='bottom', fontsize=8)

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

    counts['Total'] = totals
    counts.to_csv(output_dir / "EP_PostPulse_Behavior_Counts.csv")
    logger.info("  EP post-pulse behavior chart saved.")


def plot_slope_comparison(mechanics_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Generating: Slope Comparison (ASP vs EP)...")
    if mechanics_df is None or mechanics_df.empty: return

    df = mechanics_df.copy()
    df['Cond'] = df.apply(_cond_label_from_row, axis=1)

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


def plot_uptake_asp_vs_ep(grouped_data: Dict, output_dir: Path) -> None:
    logger.info("Generating: Uptake Comparison ASP vs EP...")
    base_conditions: Dict[Tuple, Dict[str, List[bfh.TrapData]]] = {}

    for (cell_type, treatment, pressure_pa, voltage, duration_ms), traps in grouped_data.items():
        base_key = (cell_type, treatment, pressure_pa)
        if base_key not in base_conditions: base_conditions[base_key] = {'ASP': [], 'EP': []}
        ctype = traps[0].metadata.condition_type if traps else None
        if ctype in ('ASP', 'EP'): base_conditions[base_key][ctype].extend(traps)

    _base_full_labels = [f"{c}_{t}_{p}Pa" for (c, t, p) in base_conditions]
    _base_label_map = _reduce_labels(sorted(set(_base_full_labels)))

    for (cell_type, treatment, pressure_pa), groups in base_conditions.items():
        asp_traps = groups['ASP']; ep_traps  = groups['EP']
        if not asp_traps and not ep_traps: continue

        full_label  = f"{cell_type}_{treatment}_{pressure_pa}Pa"
        short_label = _base_label_map.get(full_label, full_label)

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