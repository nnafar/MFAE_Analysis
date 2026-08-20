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

Beyond the figures, this module also owns the uptake fit-quality
criterion and the summary table built from it:

    flag_runaway_fits / apply_runaway_gate / select_uptake_cohort
        Identify and remove mono-exponential fits with no identifiable
        plateau.  These pass the R2 gate comfortably, so the R2 flag
        alone is not sufficient.  The rule lives here because it is
        needed by the table below and by four scatter figures above.

    build_uptake_kinetics_table / write_uptake_kinetics_table
        The Chapter 3 amplitude and timescale table, with both pulse
        arms passed through identical gates and the gate counts carried
        into the footnotes.
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


# --------------------------------------------------------------------- #
# Runaway uptake fits: definition, thresholds, cohort selection.
#
# A mono-exponential A * (1 - exp(-t / tau)) fitted to a trace that is
# still rising linearly at the end of the acquisition has no
# identifiable plateau.  The optimiser walks A and tau off together
# along a degenerate valley and stops wherever the tolerance is met, so
# the returned A and tau are arbitrary: they describe the initial slope
# A / tau and nothing else.  Those fits pass the R2 gate comfortably
# (R2 = 0.90 to 0.98 in this dataset) because a straight line through
# rising data fits well.  The R2 gate cannot catch them.
#
# In this dataset they are unmistakable: non-runaway fits top out at
# tau = 5.7e3 s, runaways start at tau = 2.2e5 s, with nothing in
# between.  Any threshold in that gap gives identical results.
#
# These live here, next to the figures that consume them, because the
# same criterion is needed by the uptake-kinetics table and by four
# separate scatter figures in this module.
# --------------------------------------------------------------------- #
RUNAWAY_TAU_S: float = 20_000.0
RUNAWAY_A: float = 100.0

# --------------------------------------------------------------------- #
# Reference-time readout.
#
# The fitted amplitude A is an extrapolated plateau, and for most cells
# the fitted tau is comparable to the acquisition length, so A is a
# projection past the last frame rather than a measurement.  The
# reference-time readout sidesteps that: it reports the volume-normalised
# uptake actually recorded at a fixed time after the pulse.
#
# 100 s is chosen because it is the shortest post-pulse record in the
# dataset, so every retained cell reaches it without extrapolation.
# Cells whose record stops short are dropped, never extrapolated; the
# count of such cells is carried into the table footnote.
# --------------------------------------------------------------------- #
REFERENCE_TIME_S: float = 100.0

# The rule used by every uptake figure and by the kinetics table.
# This is the single point of control: change this line and the whole
# chapter moves together.  'tau_only' is the default because the joint
# rule requires both conditions and therefore retains a 5 ms body fit
# with A = 71.6 and tau = 3.4e5 s, a timescale two thousand times the
# acquisition length.  See `flag_runaway_fits` for the alternatives.
DEFAULT_RUNAWAY_RULE: str = 'tau_only'

# The fate cohort used by every uptake figure and by the kinetics table.
# Figures whose subject *is* fate (intact versus ruptured) override this
# explicitly; nothing else should.
ANALYSIS_FATE_STATES: Tuple[str, ...] = ('intact',)

# Column-name stem per region, matching mechanics_results_all_traps.csv.
REGION_PREFIX: Dict[str, str] = {
    'Body':       'Uptake_Body_VolNorm',
    'Protrusion': 'Uptake_Prot_VolNorm',
}

# Aliases so callers can use the short region keys already used by the
# plotting code ('Prot') or the printed names ('Protrusion').
REGION_ALIAS: Dict[str, str] = {
    'Body': 'Body', 'body': 'Body',
    'Prot': 'Protrusion', 'prot': 'Protrusion',
    'Protrusion': 'Protrusion', 'protrusion': 'Protrusion',
}

VALID_RUNAWAY_RULES: Tuple[str, ...] = ('joint', 'tau_only', 'none')

# Reverse lookup so a function holding only an R2 flag column name can
# still find the region it belongs to and apply the runaway gate.
_FLAG_TO_REGION: Dict[str, str] = {
    f'{prefix}_R2_Flag': canonical
    for canonical, prefix in REGION_PREFIX.items()
}

# Pulse arms, in the order they appear in the thesis table.
PULSE_ORDER: Tuple[str, ...] = ('100us', '5ms')
PULSE_LATEX: Dict[str, str] = {
    '100us': r'\SI{100}{\micro\second}',
    '5ms':   r'\SI{5}{\milli\second}',
}

# Descriptors summarised per region: (column suffix, printed name, dp).
UPTAKE_DESCRIPTORS: Tuple[Tuple[str, str, int], ...] = (
    ('A',   'A',       2),
    ('tau', 'tau (s)', 1),
)


def _resolve_region(region: str) -> str:
    """Map any accepted spelling of a region onto its canonical key."""
    try:
        return REGION_ALIAS[region]
    except KeyError:
        raise ValueError(
            f"Unknown region {region!r}; expected one of "
            f"{tuple(sorted(set(REGION_ALIAS)))}.") from None


def flag_runaway_fits(df: pd.DataFrame,
                      region: str,
                      rule: str = DEFAULT_RUNAWAY_RULE) -> pd.Series:
    """
    Return a boolean Series, True where the uptake fit for `region` is
    a runaway under `rule`.

    Parameters
    ----------
    df : DataFrame
        Any subset of mechanics_results_all_traps.  Only the two
        columns for `region` are read.
    region : str
        'Body' or 'Prot'/'Protrusion' (see REGION_ALIAS).
    rule : {'joint', 'tau_only', 'none'}
        'joint'     A >= RUNAWAY_A AND tau > RUNAWAY_TAU_S.  This is the
                    Chapter 3 convention.  Because it requires both
                    conditions it retains one 5 ms body fit with
                    A = 71.6, tau = 3.4e5 s.
        'tau_only'  tau > RUNAWAY_TAU_S.  Stricter, and the defensible
                    choice: a timescale of 3.4e5 s measured across a
                    ~110 s acquisition is not a measurement.
        'none'      No exclusion.  Diagnostic use only.

    The returned Series is indexed exactly like `df`, so it can be used
    directly as a mask: ``df[~flag_runaway_fits(df, 'Body')]``.
    """
    if rule not in VALID_RUNAWAY_RULES:
        raise ValueError(f"Unknown runaway rule {rule!r}; expected one of "
                         f"{VALID_RUNAWAY_RULES}.")
    prefix = REGION_PREFIX[_resolve_region(region)]

    # 'none' short-circuits to an all-False Series of the right length
    # and index, so downstream code needs no special case.
    if rule == 'none':
        return pd.Series(False, index=df.index)

    # Missing fits are NaN.  Comparing NaN with a number is always
    # False, which is the behaviour we want: a cell with no fit is not a
    # runaway, it is simply absent, and the R2 gate has already removed
    # it.  fillna(False) then makes that explicit rather than leaving a
    # nullable boolean that breaks `~mask` indexing.
    amplitude = pd.to_numeric(df[f'{prefix}_A'], errors='coerce')
    timescale = pd.to_numeric(df[f'{prefix}_tau'], errors='coerce')

    slow = (timescale > RUNAWAY_TAU_S).fillna(False)
    if rule == 'tau_only':
        return slow

    large = (amplitude >= RUNAWAY_A).fillna(False)
    return large & slow


def apply_runaway_gate(df: pd.DataFrame,
                       region: str,
                       rule: str = DEFAULT_RUNAWAY_RULE) -> pd.DataFrame:
    """
    Convenience wrapper: return `df` with the runaway rows for `region`
    removed.  Provided so the scatter figures in this module can drop
    runaways with a single call rather than each re-deriving the rule.
    """
    return df.loc[~flag_runaway_fits(df, region, rule=rule)]


def select_uptake_cohort(df: pd.DataFrame,
                         region: str,
                         duration_label: str,
                         treatment: str = 'WT',
                         fate_states: Tuple[str, ...] = (
                             ANALYSIS_FATE_STATES),
                         rule: str = DEFAULT_RUNAWAY_RULE
                         ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Build the analysis cohort for one region and one pulse arm, and
    report the bookkeeping alongside it.

    The three gates are applied in a fixed order so the counts nest:

        1. Membership   treatment, EP condition, pulse arm, fate states
        2. R2 gate      Uptake_<region>_VolNorm_R2_Flag is True
        3. Runaway      per `rule`

    Returns
    -------
    (cohort, counts)
        `cohort` is the retained subset of `df`.
        `counts` has keys 'candidates', 'gate_pass', 'runaway',
        'retained', which is everything a table footnote needs.
    """
    prefix = REGION_PREFIX[_resolve_region(region)]

    # --- Gate 1: membership -------------------------------------------
    membership = (
        (df['Treatment'] == treatment)
        & (df['Condition_Type'] == 'EP')
        & (df['Duration_label'] == duration_label)
        & df['Fate_Status'].isin(list(fate_states))
    )
    candidates = df.loc[membership]

    # --- Gate 2: per-region R2 flag -----------------------------------
    # fillna(False) makes a missing flag equivalent to a failed one, so
    # a cell with no fit can never slip through.
    gate = candidates[f'{prefix}_R2_Flag'].fillna(False).astype(bool)
    gate_pass = candidates.loc[gate]

    # --- Gate 3: runaway exclusion ------------------------------------
    runaway = flag_runaway_fits(gate_pass, region, rule=rule)
    retained = gate_pass.loc[~runaway]

    counts = {
        'candidates': int(len(candidates)),
        'gate_pass':  int(len(gate_pass)),
        'runaway':    int(runaway.sum()),
        'retained':   int(len(retained)),
    }
    return retained, counts


def measure_uptake_at_reference_time(grouped_data: Dict,
                                     mechanics_df: pd.DataFrame,
                                     region: str,
                                     duration_label: str,
                                     treatment: str = 'WT',
                                     fate_states: Tuple[str, ...] = (
                                         ANALYSIS_FATE_STATES),
                                     rule: str = DEFAULT_RUNAWAY_RULE,
                                     reference_time_s: float = (
                                         REFERENCE_TIME_S)
                                     ) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Return the volume-normalised uptake measured at a fixed time after
    the pulse, for one region and one pulse arm.

    This is the model-independent counterpart to the fitted amplitude
    A.  Where A is an asymptote the fit projects, this is the height the
    trace had actually reached at `reference_time_s`, read off the data.

    Cohort
    ------
    Identical to `select_uptake_cohort`, so the reference-time rows and
    the fitted rows in the kinetics table describe the same cells apart
    from the short-record drop described below.  Reusing that function
    rather than re-deriving the gates here is deliberate: re-derivation
    is how the two pulse arms drifted apart in the published table.

    Short records
    -------------
    A cell whose post-pulse record ends before `reference_time_s` is
    dropped, not extrapolated.  `np.interp` would otherwise return the
    last recorded value for any time past the end of the record, which
    looks like a measurement and is not one.

    Parameters
    ----------
    grouped_data : dict or DiskBackedDict
        The pipeline trace store: {group_key: [TrapData, ...]}.  Needed
        because the traces themselves are not in mechanics_df.  Only
        `.items()` is used, so both a plain dict and the pipeline's
        DiskBackedDict work.
    mechanics_df : DataFrame
        The full mechanics_results_all_traps table.
    region : str
        'Body' or 'Prot'/'Protrusion' (see REGION_ALIAS).
    duration_label : str
        '100us' or '5ms'.
    reference_time_s : float
        Time after the pulse at which to read the trace.

    Returns
    -------
    (values, counts)
        `values` is a 1-D float array, one entry per retained cell.
        `counts` carries the `select_uptake_cohort` bookkeeping plus
        two extra keys: 'no_trace' (cell in the cohort but no usable
        post-pulse trace in the pickle) and 'short_record' (record ends
        before `reference_time_s`).
    """
    canonical = _resolve_region(region)

    # Column holding the volume-normalised trace for this region.
    # These arrive from `bulk_file_handling` already baseline-corrected
    # as (I(t) - F0) / V(t), so no further subtraction is applied here.
    data_col = {'Body': 'Body_VolNorm',
                'Protrusion': 'Protrusion_VolNorm'}[canonical]

    cohort, counts = select_uptake_cohort(
        mechanics_df, canonical, duration_label,
        treatment=treatment, fate_states=fate_states, rule=rule)

    counts = dict(counts)          # copy, so we do not mutate the caller's
    counts['no_trace'] = 0
    counts['short_record'] = 0

    if cohort.empty or grouped_data is None:
        return np.array([], dtype=float), counts

    # The cohort is identified by (Experiment_Folder, Trap_ID), which is
    # the same pair as (meta.full_path.name, trap.trap_id) in the pickle.
    wanted = set(zip(cohort['Experiment_Folder'], cohort['Trap_ID']))
    found: Dict[Tuple[str, int], float] = {}

    # `.items()` rather than `.values()`: in the pipeline grouped_data is a
    # DiskBackedDict, which loads each group's pickle on demand and exposes
    # items(), keys() and __getitem__ but not values().
    for _group_key, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            meta = trap.metadata
            key = (meta.full_path.name, trap.trap_id)
            if key not in wanted or key in found:
                continue

            ud = getattr(trap, 'uptake_data', {}) or {}
            if 'Time_s' not in ud or data_col not in ud:
                continue
            t_raw = np.asarray(ud['Time_s'], dtype=float)
            y_raw = np.asarray(ud[data_col], dtype=float)
            if len(t_raw) < 5 or len(y_raw) != len(t_raw):
                continue

            # Pulse alignment: t = 0 at the pulse frame, matching every
            # other post-pulse readout in this module.
            pf = getattr(meta, 'pulse_frame', None)
            if pf is None or not (0 <= int(pf) < len(t_raw)):
                continue
            t_aligned = t_raw - t_raw[int(pf)]

            m = (np.isfinite(t_aligned) & np.isfinite(y_raw)
                 & (t_aligned >= 0))
            if m.sum() < 5:
                continue

            t_post = t_aligned[m]
            y_post = y_raw[m]
            order = np.argsort(t_post)     # np.interp needs sorted x
            t_post, y_post = t_post[order], y_post[order]

            # Refuse to read past the end of the record.
            if t_post[-1] < reference_time_s:
                counts['short_record'] += 1
                continue

            found[key] = float(np.interp(reference_time_s, t_post, y_post))

    counts['no_trace'] = int(len(wanted) - len(found)
                             - counts['short_record'])
    return np.asarray(list(found.values()), dtype=float), counts


def cliffs_delta(a, b) -> float:
    """
    Cliff's delta, the non-parametric effect size that pairs with
    Mann-Whitney U.

    Every value in `a` is compared with every value in `b`.  Delta is
    the fraction of pairs where a > b minus the fraction where a < b,
    so it runs from -1 (every a below every b) through 0 (complete
    overlap) to +1 (every a above every b).  Ties contribute nothing.

    Returns NaN if either group is empty, so a missing cohort produces
    a blank cell rather than a crash.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float('nan')

    # Broadcasting: a[:, None] is a column and b[None, :] is a row, so
    # the comparison builds an (len(a) x len(b)) boolean matrix of every
    # pairing at once.  Clearer and faster than a double loop.
    greater = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return float((greater - less) / (a.size * b.size))


def median_iqr(values) -> Tuple[float, float, float, int]:
    """
    Return (median, 25th percentile, 75th percentile, n), dropping
    non-finite entries first.  Chapter 3 reports an IQR with every
    median, so these four numbers always travel together.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float('nan'), float('nan'), float('nan'), 0)
    return (float(np.median(v)),
            float(np.percentile(v, 25)),
            float(np.percentile(v, 75)),
            int(v.size))


# ===================================================================== #
# 1. Mean uptake trace (body + protrusion, two panels)                  #
# ===================================================================== #
FONT_BASE          = utils.PLOT_STYLE["fontsize_annot_pt"]
FONT_AXIS_TITLE    = utils.PLOT_STYLE["fontsize_title_pt"]
FONT_AXIS_LABEL    = utils.PLOT_STYLE["fontsize_label_pt"]
FONT_TICK          = utils.PLOT_STYLE["fontsize_tick_pt"]
FONT_LEGEND        = utils.PLOT_STYLE["fontsize_legend_pt"]
FONT_LEGEND_HEADER = utils.PLOT_STYLE["fontsize_legend_pt"]
FONT_BRACKET       = utils.PLOT_STYLE["fontsize_annot_pt"]
def plot_thesis_mean_uptake_body_prot(grouped_data: Dict,
                                       mechanics_df: pd.DataFrame,
                                       output_dir: Path,
                                       treatment: str = 'WT',
                                       n_time_bins: int = 200,
                                       min_frac_contributing: float = 0.25,
                                       require_mi_flag: bool = False,
                                       require_prepulse_flag: bool = False,
                                       fate_states: Tuple[str, ...] = (
                                           ANALYSIS_FATE_STATES),
                                       subtract_baseline: bool = False,
                                       require_uptake_flag: bool = True,
                                       runaway_rule: str = (
                                           DEFAULT_RUNAWAY_RULE),
                                       filename_suffix: str = '',
                                       # NEW: Added font size parameters
                                       title_fontsize: Optional[float] = None,
                                       label_fontsize: Optional[float] = None,
                                       tick_fontsize: Optional[float] = None,
                                       legend_fontsize: Optional[float] = None
                                       ) -> None:
    """
    Mean volume-normalised uptake vs time-since-pulse, body and
    protrusion, one condition ramp per panel.

    Alignment: t = 0 at the pulse frame (`meta.pulse_frame`).  Traces
    are shifted so that the pulse sits at t = 0, and only the
    post-pulse portion (t >= 0) is plotted.

    Baseline: `bulk_file_handling` now builds `*_VolNorm` as
    (I(t) - F0) / V(t), so the column arrives already baseline-
    corrected and `subtract_baseline` defaults to False.  Setting it
    True subtracts a second baseline, which is close to zero and
    therefore near-harmless, but it is redundant and off by default.

    Cohort: EP cells only (ASP has no pulse; the ASP "no-pulse" control
    belongs in a separate figure).  Three gates, in order:

        1. Row level     `treatment`, EP, `Fate_Status` in `fate_states`,
                         plus the optional MI / pre-pulse mechanics flags.
        2. Per region    `Uptake_<region>_VolNorm_R2_Flag`, applied only
                         when `require_uptake_flag` is True.
        3. Per region    runaway exclusion per `runaway_rule`.

    Gates 2 and 3 are per region, matching `select_uptake_cohort`, so a
    cell that passes the body fit and fails the protrusion fit appears
    in the left panel and not the right.  The two panels therefore carry
    different n, which is reported in each legend.

    To reproduce the cohort behind the uptake-kinetics table, call with
    `fate_states=('intact',)`, `require_uptake_flag=True` and the same
    `runaway_rule` used to build the table.

    Residual mismatch with the table: even on identical cells, the
    plotted curve is a pointwise median across traces and its plateau is
    not the median of the per-cell fitted amplitudes.  Cells with
    different tau are averaged at each timepoint, which smears the shape
    into something that is not itself a mono-exponential.  Late-time
    values also come only from cells tracked that long, after the
    `min_frac_contributing` trim.  The two will agree in magnitude and
    ordering, not to the decimal place, and captions should say so.

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
        & mechanics_df['Fate_Status'].isin(list(fate_states))
    )
    filter_labels = ['EP', '|'.join(fate_states)]
    if require_mi_flag:
        mask = mask & mechanics_df['MI_Whole_R2_Flag'].fillna(False).astype(bool)
        filter_labels.append('MI_Whole_R2_Flag')
    if require_prepulse_flag:
        mask = mask & mechanics_df['PrePulse_Visco_R2_Flag'].fillna(False).astype(bool)
        filter_labels.append('PrePulse_Visco_R2_Flag')
    ok = mechanics_df.loc[mask]
    filter_desc = ' + '.join(filter_labels)
    if ok.empty:
        logger.warning("Mean uptake trace skipped: no accepted EP cells for "
                       f"treatment={treatment} (filters=[{filter_desc}]).")
        return

    # ---- Per-region gates (uptake R2, runaway) -----------------------
    # These are per region, not per row: a cell may pass the body fit
    # and fail the protrusion fit.  Building one accepted set per panel
    # mirrors `select_uptake_cohort`, which is what the kinetics table
    # uses, so the figure and the table can be made to agree on cells.
    region_gate_desc = []
    accepted_by_region: Dict[str, set] = {}
    for region_key, canonical in (('Body', 'Body'), ('Prot', 'Protrusion')):
        sub = ok
        if require_uptake_flag:
            flag_col = f'{REGION_PREFIX[canonical]}_R2_Flag'
            sub = sub[sub[flag_col].fillna(False).astype(bool)]
        if runaway_rule != 'none':
            sub = apply_runaway_gate(sub, canonical, rule=runaway_rule)
        accepted_by_region[region_key] = set(
            zip(sub['Experiment_Folder'], sub['Trap_ID']))
        region_gate_desc.append(f"{region_key} n={len(sub)}")

    if require_uptake_flag:
        filter_labels.append('Uptake_R2_Flag')
    if runaway_rule != 'none':
        filter_labels.append(f'runaway:{runaway_rule}')
    filter_desc = ' + '.join(filter_labels)

    # Union across regions: a trap is worth opening if either panel
    # wants it.  Per-panel membership is rechecked inside the loop.
    accepted = accepted_by_region['Body'] | accepted_by_region['Prot']
    if not accepted:
        logger.warning("Mean uptake trace skipped: no cells survive the "
                       f"per-region gates (filters=[{filter_desc}]).")
        return
    logger.info(f"  Mean uptake trace cohort ({treatment}): "
                f"{', '.join(region_gate_desc)}  filters=[{filter_desc}]")

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
                # Per-region gate: a trap can be accepted for the body
                # panel and rejected for the protrusion panel.
                if key not in accepted_by_region[region]:
                    continue
                y = np.asarray(ud[col], dtype=float)
                if len(y) != len(t_pulse_aligned):
                    continue

                # `*_VolNorm` now arrives from `bulk_file_handling` as
                # (I(t) - F0) / V(t), already baseline-corrected, so this
                # block is off by default.  When enabled it subtracts a
                # second, near-zero residual baseline using the same
                # definition as `bulk_mechanics.fit_uptake_kinetics`.
                baseline = 0.0
                if subtract_baseline:
                    pre_m = (np.isfinite(t_pulse_aligned)
                             & np.isfinite(y)
                             & (t_pulse_aligned <= 0))
                    if pre_m.sum() >= 2:
                        baseline = float(np.nanmedian(y[pre_m]))
                    else:
                        # Too few pre-pulse frames for a stable median.
                        # Leave the trace uncorrected rather than
                        # subtracting a single noisy frame, and say so.
                        logger.debug(
                            f"    {key} {region}: fewer than 2 pre-pulse "
                            "frames; trace left uncorrected.")

                # Restrict to post-pulse (t >= 0).  The pre-pulse
                # segment defines the baseline above and is not plotted.
                m = (np.isfinite(t_pulse_aligned)
                     & np.isfinite(y)
                     & (t_pulse_aligned >= 0))
                if m.sum() < 5:
                    continue
                t_ok = t_pulse_aligned[m]
                y_ok = y[m] - baseline
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
    
    # NEW: Resolve font sizes from arguments or fallback to globals
    t_font = title_fontsize if title_fontsize is not None else FONT_AXIS_TITLE
    l_font = label_fontsize if label_fontsize is not None else FONT_AXIS_LABEL
    tk_font = tick_fontsize if tick_fontsize is not None else FONT_TICK
    leg_font = legend_fontsize if legend_fontsize is not None else FONT_LEGEND

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, sharex=True)
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

        # NEW: Inject resolved fonts into standard formatting functions
        ax.set_xlabel("Time since pulse (s)", fontsize=l_font)
        ax.set_title(region_titles[region], fontsize=t_font)
        ax.set_xlim(left=0.0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=tk_font)
        ax.legend(frameon=False, loc='best', fontsize=leg_font)

    if subtract_baseline:
        axes[0].set_ylabel(
            r"Uptake above baseline (a.u. / $\mathrm{\mu m}^{3}$)", fontsize=l_font)
        baseline_note = "baseline-subtracted"
    else:
        axes[0].set_ylabel(r"Uptake (a.u. / $\mathrm{\mu m}^{3}$)", fontsize=l_font)
        baseline_note = "absolute, baseline retained"
        
    # fig.suptitle(f"{treatment} — mean volume-normalised uptake, "
    #              f"pulse-aligned ({baseline_note}; median $\\pm$ IQR)  "
    #              f"[{filter_desc}]",
    #              y=1.02, fontweight='bold', fontsize=t_font)
    fig.tight_layout()
    out = (Path(output_dir)
           / f"Thesis_Uptake_Mean_Trace_BodyProt_{treatment}{filename_suffix}.pdf")
    utils.save_plot_pdf(out)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ===================================================================== #
# 1b. Supplementary: mean uptake trace split by fit-quality subset      #
# ===================================================================== #
def plot_thesis_mean_uptake_by_fit_subset(grouped_data: Dict,
                                          mechanics_df: pd.DataFrame,
                                          output_dir: Path,
                                          treatment: str = 'WT',
                                          n_time_bins: int = 200,
                                          min_frac_contributing: float = 0.25,
                                          fate_states: Tuple[str, ...] = (
                                              ANALYSIS_FATE_STATES),
                                          runaway_rule: str = (
                                              DEFAULT_RUNAWAY_RULE),
                                          subtract_baseline: bool = False,
                                          duration_labels: Tuple[str, ...] = (
                                              '100us', '5ms'),
                                          filename_suffix: str = ''
                                          ) -> None:
    """
    Supplementary companion to `plot_thesis_mean_uptake_body_prot`.

    The main-text mean trace is gated to the cohort behind the
    uptake-kinetics table, which means cells are dropped from it.  This
    figure shows every cell in the row-level cohort, split into the
    three fit-quality subsets so nothing is silently removed and a
    reader can see exactly what the gates took out and why:

        analysed    passes the per-region uptake R2 gate and is not a
                    runaway.  This is the main-text and table cohort.
        runaway     passes the R2 gate but has no identifiable plateau
                    under `runaway_rule`.  These are the cells whose
                    traces are still rising linearly at the end of
                    acquisition, which is why their fitted A and tau
                    are meaningless and why they are excluded from any
                    summary of A or tau.
        R2 failed   does not pass the per-region uptake R2 gate.

    Layout: one row per pulse arm in `duration_labels`, two columns
    (body, protrusion).  Within a panel the three subsets share the
    pulse arm's colour and are distinguished by line style, with n in
    the legend.  Median and IQR band throughout, matching the main
    figure so the two are read the same way.

    The R2-failed subset is drawn without an IQR band: those traces
    have no common shape, so a band across them would suggest a
    coherent population that is not there.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Uptake fit-subset trace skipped: empty mechanics_df.")
        return
    if runaway_rule not in VALID_RUNAWAY_RULES:
        raise ValueError(f"Unknown runaway rule {runaway_rule!r}; expected "
                         f"one of {VALID_RUNAWAY_RULES}.")

    # ---- Row-level cohort --------------------------------------------
    mask = (
        (mechanics_df['Treatment'] == treatment)
        & (mechanics_df['Condition_Type'] == 'EP')
        & mechanics_df['Fate_Status'].isin(list(fate_states))
    )
    ok = mechanics_df.loc[mask]
    if ok.empty:
        logger.warning("Uptake fit-subset trace skipped: no accepted EP "
                       f"cells for treatment={treatment}.")
        return

    # ---- Assign every cell to exactly one subset, per region ---------
    # Keys are (duration_label, region, subset) -> set of trap keys.
    # Building the assignment up front keeps the trace loop simple and
    # guarantees the three subsets partition the cohort with no overlap.
    SUBSETS = ('analysed', 'runaway', 'r2_failed')
    SUBSET_LABEL = {
        'analysed':  'Analysed',
        'runaway':   'Runaway',
        'r2_failed': r'R$^{2}$ failed',
    }
    SUBSET_STYLE = {
        'analysed':  dict(linestyle='-',  linewidth=1.8, band=True),
        'runaway':   dict(linestyle='--', linewidth=1.5, band=True),
        'r2_failed': dict(linestyle=':',  linewidth=1.2, band=False),
    }

    membership: Dict[Tuple[str, str, str], set] = {}
    for duration in duration_labels:
        arm = ok[ok['Duration_label'] == duration]
        for region_key, canonical in (('Body', 'Body'),
                                      ('Prot', 'Protrusion')):
            flag_col = f'{REGION_PREFIX[canonical]}_R2_Flag'
            passed = arm[flag_col].fillna(False).astype(bool)

            gate_pass = arm[passed]
            r2_failed = arm[~passed]
            runaway_mask = flag_runaway_fits(gate_pass, canonical,
                                             rule=runaway_rule)

            for subset, frame in (
                ('analysed',  gate_pass[~runaway_mask]),
                ('runaway',   gate_pass[runaway_mask]),
                ('r2_failed', r2_failed),
            ):
                membership[(duration, region_key, subset)] = set(
                    zip(frame['Experiment_Folder'], frame['Trap_ID']))

    # Union of everything we might need, so traps are opened once.
    all_keys = set().union(*membership.values()) if membership else set()
    if not all_keys:
        logger.warning("Uptake fit-subset trace skipped: no cells assigned.")
        return

    # ---- Collect traces ----------------------------------------------
    region_col = {'Body': 'Body_VolNorm', 'Prot': 'Protrusion_VolNorm'}
    traces: Dict[Tuple[str, str, str], list] = {k: [] for k in membership}
    all_post_ends: list = []

    row_lookup = {
        (r['Experiment_Folder'], r['Trap_ID']): r
        for _, r in ok.iterrows()
    }

    for _gk, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            meta = trap.metadata
            if meta.treatment != treatment:
                continue
            key = (meta.full_path.name, trap.trap_id)
            if key not in all_keys:
                continue

            row = row_lookup.get(key)
            if row is None:
                continue
            duration = row.get('Duration_label')
            if duration not in duration_labels:
                continue

            ud = getattr(trap, 'uptake_data', {}) or {}
            if 'Time_s' not in ud:
                continue
            t_raw = np.asarray(ud['Time_s'], dtype=float)
            if len(t_raw) < 5:
                continue

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

                # Which subset does this cell belong to for this region?
                subset = next(
                    (s for s in SUBSETS
                     if key in membership[(duration, region, s)]), None)
                if subset is None:
                    continue

                # Same baseline definition as the main figure and the
                # fitter: median of frames at or before the pulse.
                baseline = 0.0
                if subtract_baseline:
                    pre_m = (np.isfinite(t_pulse_aligned)
                             & np.isfinite(y)
                             & (t_pulse_aligned <= 0))
                    if pre_m.sum() >= 2:
                        baseline = float(np.nanmedian(y[pre_m]))

                m = (np.isfinite(t_pulse_aligned)
                     & np.isfinite(y)
                     & (t_pulse_aligned >= 0))
                if m.sum() < 5:
                    continue
                t_ok = t_pulse_aligned[m]
                traces[(duration, region, subset)].append(
                    (t_ok, y[m] - baseline))
                all_post_ends.append(float(t_ok[-1]))

    if not all_post_ends:
        logger.warning("Uptake fit-subset trace skipped: no traces collected.")
        return

    # ---- Shared time grid --------------------------------------------
    t_max = float(np.percentile(all_post_ends, 95))
    if t_max <= 0:
        logger.warning("Uptake fit-subset trace skipped: time grid collapsed.")
        return
    t_grid = np.linspace(0.0, t_max, n_time_bins)

    # ---- Plot ---------------------------------------------------------
    utils.set_paper_style()
    n_rows = len(duration_labels)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(10, 4.2 * n_rows),
                             sharex=True, squeeze=False)
    region_titles = {'Body': 'Cell body', 'Prot': 'Protrusion'}

    for r_idx, duration in enumerate(duration_labels):
        color = _condition_color(duration, treatment)
        for c_idx, region in enumerate(('Body', 'Prot')):
            ax = axes[r_idx][c_idx]

            for subset in SUBSETS:
                trace_list = traces[(duration, region, subset)]
                if not trace_list:
                    continue
                style = SUBSET_STYLE[subset]

                Y = np.full((len(trace_list), n_time_bins), np.nan)
                for i, (t, y) in enumerate(trace_list):
                    order = np.argsort(t)
                    f = interp1d(t[order], y[order], bounds_error=False,
                                 fill_value=np.nan, assume_sorted=True)
                    Y[i, :] = f(t_grid)

                n_contributing = np.sum(~np.isnan(Y), axis=0)
                keep = n_contributing >= max(
                    1, int(min_frac_contributing * len(trace_list)))
                if not keep.any():
                    continue

                with np.errstate(invalid='ignore'), \
                     __import__('warnings').catch_warnings():
                    __import__('warnings').filterwarnings(
                        'ignore', r'All-NaN slice encountered')
                    med = np.nanmedian(Y, axis=0)
                    q25 = np.nanpercentile(Y, 25, axis=0)
                    q75 = np.nanpercentile(Y, 75, axis=0)

                t_plot = t_grid[keep]
                if style['band']:
                    ax.fill_between(t_plot, q25[keep], q75[keep],
                                    color=color, alpha=0.15, linewidth=0)
                ax.plot(t_plot, med[keep], color=color,
                        linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        label=f"{SUBSET_LABEL[subset]} (n={len(trace_list)})")

            ax.set_title(f"{CONDITION_LABEL.get(duration, duration)} — "
                         f"{region_titles[region]}")
            ax.set_xlim(left=0.0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False, loc='best', fontsize=8)
            if r_idx == n_rows - 1:
                ax.set_xlabel("Time since pulse (s)")
        axes[r_idx][0].set_ylabel(
            r"Uptake above baseline (a.u. / $\mathrm{\mu m}^{3}$)"
            if subtract_baseline
            else r"Uptake (a.u. / $\mathrm{\mu m}^{3}$)")

    fate_text = '|'.join(fate_states)
    fig.suptitle(f"{treatment} — uptake traces by fit-quality subset "
                 f"(median $\\pm$ IQR; runaway rule: {runaway_rule})  "
                 f"[EP + {fate_text}]",
                 y=1.00, fontweight='bold')
    fig.tight_layout()
    out = (Path(output_dir)
           / f"Thesis_Uptake_Mean_Trace_ByFitSubset_{treatment}"
             f"{filename_suffix}.pdf")
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
                                            runaway_rule: str = (
                                                DEFAULT_RUNAWAY_RULE),
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
                # Analysis cohort: the per-region R2 flag alone is not
                # enough, because runaway fits pass it comfortably.  Apply
                # the same runaway rule the kinetics table uses so this
                # panel describes the same cells.
        if runaway_rule != 'none':
            sub = apply_runaway_gate(sub, region, rule=runaway_rule)
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
                                        treatment: str = 'WT',
                                        runaway_rule: str = (
                                            DEFAULT_RUNAWAY_RULE)
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
                # Same runaway rule as the kinetics table: an
                # unidentifiable fit contributes a meaningless A to rho.
                if runaway_rule != 'none' and yflag in _FLAG_TO_REGION:
                    sub = apply_runaway_gate(
                        sub, _FLAG_TO_REGION[yflag], rule=runaway_rule)
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
                                        treatment: str = 'WT',
                                        runaway_rule: str = (
                                            DEFAULT_RUNAWAY_RULE)
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
                # Same runaway rule as the kinetics table: an
                # unidentifiable fit contributes a meaningless A to rho.
                if runaway_rule != 'none' and yflag in _FLAG_TO_REGION:
                    sub = apply_runaway_gate(
                        sub, _FLAG_TO_REGION[yflag], rule=runaway_rule)
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
                                        treatment: str = 'WT',
                                        runaway_rule: str = (
                                            DEFAULT_RUNAWAY_RULE)
                                        ) -> None:
    """
    Two-panel scatter of uptake plateau A vs pre-pulse maximum
    protrusion length.  Panels: body (left), protrusion (right).

    The question is whether a bigger pre-pulse protrusion (which
    reflects both cell deformability and how far the tip reached
    into the microfluidic channel before the pulse) predicts a
    larger post-pulse uptake plateau.  Because this correlation
    involves only quantities from the uptake fit and the
    protrusion-length descriptor (no viscoelastic or MI fit needed),
    no mechanics R² filter is applied — the cohort is defined only
    by the uptake R² flag per region.

    Cohort:
        - EP cells of `treatment` with intact / ruptured_post fate.
        - Additionally per-panel: `Uptake_{region}_VolNorm_R2_Flag == True`
          for the region shown in that panel.  A cell that passed the
          body fit but failed the protrusion fit contributes to the
          body panel only.

    X-axis: log for uptake amplitude (multi-decade spread).
    Y-axis: linear for protrusion length (5–35 µm, no decade spread).
    Colour: condition (5ms / 100us) from the WT ramp.
    Marker: fate (intact = filled, ruptured_post = open).

    Spearman rho + p annotated per panel across all points.
    """
    if mechanics_df is None or mechanics_df.empty:
        logger.warning("Uptake-vs-protlength skipped: empty mechanics_df.")
        return

    df = mechanics_df[
        (mechanics_df['Treatment'] == treatment)
        & (mechanics_df['Condition_Type'] == 'EP')
        & mechanics_df['Fate_Status'].isin(['intact', 'ruptured_post'])
    ].copy()
    if df.empty:
        logger.warning("Uptake-vs-protlength skipped: no EP cells for "
                       f"treatment={treatment}.")
        return

    region_spec = {
        'Body': ('Uptake_Body_VolNorm_A', 'Uptake_Body_VolNorm_R2_Flag',
                 'Body uptake plateau'),
        'Prot': ('Uptake_Prot_VolNorm_A', 'Uptake_Prot_VolNorm_R2_Flag',
                 'Protrusion uptake plateau'),
    }
    prot_len_col = 'Max_Prot_length_PrePulse_um'

    utils.set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, region in zip(axes, ('Body', 'Prot')):
        a_col, flag_col, xlab = region_spec[region]
        keep = (df[flag_col].fillna(False).astype(bool)
                & df[a_col].notna()
                & df[prot_len_col].notna())
        sub = df[keep].copy()
                # Analysis cohort: the per-region R2 flag alone is not
                # enough, because runaway fits pass it comfortably.  Apply
                # the same runaway rule the kinetics table uses so this
                # panel describes the same cells.
        if runaway_rule != 'none':
            sub = apply_runaway_gate(sub, region, rule=runaway_rule)
        if sub.empty:
            ax.text(0.5, 0.5, "No cells pass R² ≥ 0.85",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(f"{region}  (n = 0)")
            continue

        sub['Cond_Key'] = sub.apply(_condition_key, axis=1)

        for cond_key in CONDITION_ORDER:
            for fate, marker_kwargs in (
                ('intact',
                 dict(facecolor=_condition_color(cond_key, treatment),
                      edgecolor='white', linewidths=0.5)),
                ('ruptured_post',
                 dict(facecolor='none',
                      edgecolor=_condition_color(cond_key, treatment),
                      linewidths=1.4)),
            ):
                pts = sub[(sub['Cond_Key'] == cond_key)
                          & (sub['Fate_Status'] == fate)]
                if pts.empty:
                    continue
                x = pts[a_col].to_numpy(dtype=float)
                y = pts[prot_len_col].to_numpy(dtype=float)
                ax.scatter(x, y, s=40, alpha=0.75, marker='o',
                           label=f"{CONDITION_LABEL[cond_key]} ({fate})",
                           **marker_kwargs)

        # Spearman across all points.
        x_all = sub[a_col].to_numpy(dtype=float)
        y_all = sub[prot_len_col].to_numpy(dtype=float)
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
        ax.set_title(f"{region}  (n = {len(sub)})")
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel(r"Max protrusion length ($\mathrm{\mu m}$)")

    # Shared legend below.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)

    fig.suptitle(f"{treatment} — uptake plateau A vs max pre-pulse "
                 "protrusion length",
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
# ===================================================================== #
# 6. Uptake-kinetics summary table (not a figure)                       #
# ===================================================================== #
def build_uptake_kinetics_table(mechanics_df: pd.DataFrame,
                                treatment: str = 'WT',
                                fate_states: Tuple[str, ...] = (
                                    ANALYSIS_FATE_STATES),
                                rule: str = DEFAULT_RUNAWAY_RULE,
                                regions: Tuple[str, ...] = ('Body',
                                                            'Protrusion'),
                                grouped_data: Optional[Dict] = None,
                                reference_time_s: float = REFERENCE_TIME_S
                                ) -> pd.DataFrame:
    """
    Build the Chapter 3 uptake-kinetics table as a tidy DataFrame, one
    row per (region, descriptor).

    This function exists because the published version of that table
    was assembled by hand, which allowed the two pulse arms to drift
    apart: the 100 us arm had its runaway fits removed while the 5 ms
    arm kept its own, and the two were then compared with a
    Mann-Whitney U test.  Here both arms pass through the identical
    membership, R2 and runaway gates, and the gate counts are carried
    in the output so the footnotes cannot disagree with the body.

    Parameters
    ----------
    mechanics_df : DataFrame
        The full mechanics_results_all_traps table.
    treatment : str
        'WT' or 'CytD'.
    fate_states : tuple of str
        Which Fate_Status values to admit.  The published table used
        ('intact',).  Pass ('intact', 'ruptured_post') to match the
        default mean-trace figure instead.
    rule : {'joint', 'tau_only', 'none'}
        Runaway exclusion, applied identically to both arms.
    regions : tuple of str
        Subset of ('Body', 'Protrusion').
    grouped_data : dict, optional
        The pipeline pickle.  When supplied, one extra row per region
        reports the uptake measured directly at `reference_time_s`,
        which needs the traces and cannot be derived from mechanics_df.
        When None, only the fitted descriptors are reported.
    reference_time_s : float
        Time after the pulse for that direct readout.

    Returns
    -------
    DataFrame with, per row: Region, Descriptor, Rule, Fate_States;
    median / q25 / q75 / n per pulse arm; p_MannWhitney; Cliffs_delta;
    and candidates / gate_pass / runaway per pulse arm.
    """
    if rule not in VALID_RUNAWAY_RULES:
        raise ValueError(f"Unknown runaway rule {rule!r}; expected one of "
                         f"{VALID_RUNAWAY_RULES}.")

    fate_text = '|'.join(fate_states)
    rows = []

    for region in regions:
        canonical = _resolve_region(region)
        prefix = REGION_PREFIX[canonical]

        # Build both arms once, then reuse them for every descriptor.
        # Rebuilding per descriptor is exactly how the arms drifted
        # apart in the published table.
        cohorts: Dict[str, pd.DataFrame] = {}
        bookkeeping: Dict[str, Dict[str, int]] = {}
        for arm in PULSE_ORDER:
            cohort, counts = select_uptake_cohort(
                mechanics_df, canonical, arm,
                treatment=treatment, fate_states=fate_states, rule=rule)
            cohorts[arm] = cohort
            bookkeeping[arm] = counts
            logger.info(
                f"  {canonical:11s} {arm:6s}: candidates "
                f"{counts['candidates']}, R2 pass {counts['gate_pass']}, "
                f"runaway {counts['runaway']}, retained {counts['retained']}")

        for suffix, printed_name, _dp in UPTAKE_DESCRIPTORS:
            row: Dict[str, object] = {
                'Region':      canonical,
                'Descriptor':  printed_name,
                'Rule':        rule,
                'Fate_States': fate_text,
            }

            values: Dict[str, np.ndarray] = {}
            for arm in PULSE_ORDER:
                v = pd.to_numeric(cohorts[arm][f'{prefix}_{suffix}'],
                                  errors='coerce')
                v = v[np.isfinite(v)].to_numpy(dtype=float)
                values[arm] = v

                med, q25, q75, n = median_iqr(v)
                row[f'{arm}_median'] = med
                row[f'{arm}_q25'] = q25
                row[f'{arm}_q75'] = q75
                row[f'{arm}_n'] = n
                row[f'{arm}_candidates'] = bookkeeping[arm]['candidates']
                row[f'{arm}_gate_pass'] = bookkeeping[arm]['gate_pass']
                row[f'{arm}_runaway'] = bookkeeping[arm]['runaway']

            a, b = values[PULSE_ORDER[0]], values[PULSE_ORDER[1]]

            # Mann-Whitney needs at least one observation per group.
            # Below that, report NaN rather than inventing a p-value.
            if a.size > 0 and b.size > 0:
                p_value = float(
                    stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
            else:
                p_value = float('nan')

            row['p_MannWhitney'] = p_value
            row['Cliffs_delta'] = cliffs_delta(a, b)
            rows.append(row)

        # --- Reference-time row ---------------------------------------
        # Skipped when the pickle was not passed, so callers that only
        # have the CSV still get a valid (shorter) table rather than an
        # error.
        if grouped_data is not None:
            ref_row: Dict[str, object] = {
                'Region':          canonical,
                'Descriptor':      'I_ref',
                'Rule':            rule,
                'Fate_States':     fate_text,
                'Reference_Time_s': float(reference_time_s),
            }
            ref_values: Dict[str, np.ndarray] = {}
            for arm in PULSE_ORDER:
                v, ref_counts = measure_uptake_at_reference_time(
                    grouped_data, mechanics_df, canonical, arm,
                    treatment=treatment, fate_states=fate_states,
                    rule=rule, reference_time_s=reference_time_s)
                ref_values[arm] = v

                med, q25, q75, n = median_iqr(v)
                ref_row[f'{arm}_median'] = med
                ref_row[f'{arm}_q25'] = q25
                ref_row[f'{arm}_q75'] = q75
                ref_row[f'{arm}_n'] = n
                ref_row[f'{arm}_candidates'] = ref_counts['candidates']
                ref_row[f'{arm}_gate_pass'] = ref_counts['gate_pass']
                ref_row[f'{arm}_runaway'] = ref_counts['runaway']
                ref_row[f'{arm}_short_record'] = ref_counts['short_record']
                ref_row[f'{arm}_no_trace'] = ref_counts['no_trace']
                logger.info(
                    f"  {canonical:11s} {arm:6s}: reference-time n {n}, "
                    f"short record {ref_counts['short_record']}, "
                    f"no trace {ref_counts['no_trace']}")

            a, b = ref_values[PULSE_ORDER[0]], ref_values[PULSE_ORDER[1]]
            if a.size > 0 and b.size > 0:
                ref_row['p_MannWhitney'] = float(
                    stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
            else:
                ref_row['p_MannWhitney'] = float('nan')
            ref_row['Cliffs_delta'] = cliffs_delta(a, b)
            rows.append(ref_row)

    return pd.DataFrame(rows)


def _format_table_cell(median: float, q25: float, q75: float,
                       n: int, decimals: int) -> str:
    """Render one arm's entry as ``$median$ [$q25$--$q75$], $n$``."""
    if n == 0 or not np.isfinite(median):
        return r'---'
    return (f"${median:.{decimals}f}$ "
            f"[${q25:.{decimals}f}$--${q75:.{decimals}f}$], ${n}$")


def format_uptake_kinetics_latex(table: pd.DataFrame,
                                 label: str = 'tab: ch3 uptake kinetics'
                                 ) -> str:
    """
    Render the tidy table as a LaTeX tabular in the Chapter 3 style.

    Footnotes are generated from the bookkeeping columns rather than
    typed by hand, so the retention counts can never disagree with the
    numbers above them.
    """
    if table.empty:
        return '% No uptake-kinetics rows to render.\n'

    rule = str(table['Rule'].iloc[0])
    fate_text = str(table['Fate_States'].iloc[0])
    decimals_for = {name: dp for _s, name, dp in UPTAKE_DESCRIPTORS}

    lines = [
        r'\begin{table}[tb]',
        r'    \centering',
        r'    \caption{Post-pulse dye-uptake amplitude $A$ and timescale '
        r'$\tau$ by pulse protocol and region. Values are medians with '
        r'IQR; $p$ from Mann--Whitney $U$ test, \SI{100}{\micro\second} '
        r"versus \SI{5}{\milli\second}; $\delta$ is Cliff's delta. The "
        r'same $R^{2}$ gate and runaway-exclusion rule are applied to '
        r'both pulse arms. The reference-time row is read directly off '
        r'each trace and involves no extrapolation.}',
        f'    \\label{{{label}}}',
        r'    \begin{tabular}{lcccc}',
        r'        \hline',
        r'        Descriptor & ' + PULSE_LATEX['100us'] + r' [IQR], $n$ & '
        + PULSE_LATEX['5ms'] + r' [IQR], $n$ & $p$ & $\delta$ \\',
        r'        \hline',
    ]

    for _, row in table.iterrows():
        descriptor = str(row['Descriptor'])
        decimals = decimals_for.get(descriptor, 2)
        if descriptor.startswith('tau'):
            printed = r'$\tau$ (\si{\second})'
        elif descriptor == 'I_ref':
            # Direct readout, not a fitted parameter.  Named separately
            # so the reader is not tempted to read it as an asymptote.
            t_ref = float(row.get('Reference_Time_s', REFERENCE_TIME_S))
            printed = (r'uptake at \SI{' + f'{t_ref:g}'
                       + r'}{\second}')
        else:
            printed = f'${descriptor}$'
        printed = f"{row['Region']} {printed}"

        cell_a = _format_table_cell(row['100us_median'], row['100us_q25'],
                                    row['100us_q75'], int(row['100us_n']),
                                    decimals)
        cell_b = _format_table_cell(row['5ms_median'], row['5ms_q25'],
                                    row['5ms_q75'], int(row['5ms_n']),
                                    decimals)

        p_value = row['p_MannWhitney']
        p_text = '---' if not np.isfinite(p_value) else f'${p_value:.2f}$'
        delta = row['Cliffs_delta']
        d_text = '---' if not np.isfinite(delta) else f'${delta:+.2f}$'

        lines.append(f'        {printed} & {cell_a} & {cell_b} & '
                     f'{p_text} & {d_text} \\\\')

    lines.append(r'        \hline')

    # Footnote 1: R2 retention, read straight off the bookkeeping.
    retention_bits = []
    for region in table['Region'].unique():
        sub = table[table['Region'] == region].iloc[0]
        for arm in PULSE_ORDER:
            retention_bits.append(
                f"{PULSE_LATEX[arm]} {int(sub[f'{arm}_gate_pass'])}/"
                f"{int(sub[f'{arm}_candidates'])} {region.lower()}")
    lines.append(
        r'        \multicolumn{5}{l}{\footnotesize $R^{2}$ gate retention: '
        + '; '.join(retention_bits) + r'.} \\')

    # Footnote 2: runaway exclusions per arm and region.
    runaway_bits = []
    for region in table['Region'].unique():
        sub = table[table['Region'] == region].iloc[0]
        for arm in PULSE_ORDER:
            runaway_bits.append(
                f"{PULSE_LATEX[arm]} {region.lower()} "
                f"{int(sub[f'{arm}_runaway'])}")
    lines.append(
        r'        \multicolumn{5}{l}{\footnotesize Runaway fits excluded '
        r'(rule: \texttt{' + rule.replace('_', r'\_') + r'}): '
        + '; '.join(runaway_bits) + r'.} \\')

    # Footnote 2b: short records dropped from the reference-time rows.
    # Only emitted when those rows are present, so a table built without
    # the pickle carries no dangling footnote.
    ref_rows = table[table['Descriptor'] == 'I_ref']
    if not ref_rows.empty:
        short_bits = []
        for _, sub in ref_rows.iterrows():
            for arm in PULSE_ORDER:
                short_bits.append(
                    f"{PULSE_LATEX[arm]} {str(sub['Region']).lower()} "
                    f"{int(sub[f'{arm}_short_record'])}")
        t_ref = float(ref_rows['Reference_Time_s'].iloc[0])
        lines.append(
            r'        \multicolumn{5}{l}{\footnotesize Cells whose '
            r'post-pulse record ends before \SI{' + f'{t_ref:g}'
            + r'}{\second} are excluded from the reference-time rows '
            r'rather than extrapolated: ' + '; '.join(short_bits)
            + r'.} \\')

    # Footnote 3: cohort definition, so the figure pairing is explicit.
    lines.append(
        r'        \multicolumn{5}{l}{\footnotesize Cohort: EP cells with '
        r'\texttt{Fate\_Status} $\in$ \{' + fate_text.replace('_', r'\_')
        + r'\}.} \\')

    lines += [r'    \end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def write_uptake_kinetics_table(mechanics_df: pd.DataFrame,
                                output_dir: Path,
                                treatment: str = 'WT',
                                fate_states: Tuple[str, ...] = (
                                    ANALYSIS_FATE_STATES),
                                rule: str = DEFAULT_RUNAWAY_RULE,
                                filename_stem: Optional[str] = None,
                                grouped_data: Optional[Dict] = None,
                                reference_time_s: float = REFERENCE_TIME_S
                                ) -> pd.DataFrame:
    """
    Build the table, write both the tidy CSV and the LaTeX source, and
    return the tidy DataFrame.

    Two files are written so nothing has to be retyped into the thesis:

        <stem>.csv   every number, including the bookkeeping columns
        <stem>.tex   the formatted tabular, ready to \\input

    The default stem records the rule and the fate cohort, so running
    under a different rule produces a differently named pair of files
    rather than silently overwriting the last one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename_stem is None:
        fate_tag = '_'.join(fate_states)
        filename_stem = (f'Thesis_Uptake_Kinetics_{treatment}'
                         f'_{rule}_{fate_tag}')

    logger.info(f"Uptake kinetics table ({treatment}, rule={rule}, "
                f"fate={'|'.join(fate_states)}):")

    table = build_uptake_kinetics_table(
        mechanics_df, treatment=treatment,
        fate_states=fate_states, rule=rule,
        grouped_data=grouped_data,
        reference_time_s=reference_time_s)

    csv_path = output_dir / f'{filename_stem}.csv'
    tex_path = output_dir / f'{filename_stem}.tex'
    table.to_csv(csv_path, index=False)
    tex_path.write_text(format_uptake_kinetics_latex(table), encoding='utf-8')

    logger.info(f"  Saved: {csv_path.name}")
    logger.info(f"  Saved: {tex_path.name}")
    return table


# ===================================================================== #
# Runner                                                                #
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
        # --- Mean trace ------------------------------------------------
        # Analysis cohort throughout: intact, per-region uptake R2 gate,
        # runaway exclusion under DEFAULT_RUNAWAY_RULE.  Same cells as
        # the kinetics table below, so the two always agree on cohort.
        _try(f"Mean uptake trace ({treatment})",
            plot_thesis_mean_uptake_body_prot,  # Uncalled function reference
            grouped_data, mechanics_df, output_dir, 
            treatment=treatment,
            filename_suffix='',
            title_fontsize=12,                  # Font arguments passed as keyword args
            label_fontsize=12,
            tick_fontsize=12,
            legend_fontsize=12)

        # Supplementary: every cell in the intact cohort, split into
        # analysed / runaway / R2-failed, so the gates above are visible
        # rather than silent.
        _try(f"Uptake trace by fit subset ({treatment})",
             plot_thesis_mean_uptake_by_fit_subset,
             grouped_data, mechanics_df, output_dir, treatment=treatment,
             filename_suffix='')

        # --- Amplitude vs Trap ID -------------------------------------
        _try(f"Uptake amplitude per trap ({treatment})",
             plot_thesis_uptake_amplitude_per_trap,
             mechanics_df, output_dir, treatment=treatment,
             filename_suffix='')

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

        # --- Best-fit multipanel --------------------------------------
        _try(f"Uptake best-fit multipanel ({treatment})",
             plot_thesis_uptake_bestfit_multipanel,
             grouped_data, mechanics_df, output_dir, treatment=treatment,
             require_mi_flag=True, require_prepulse_flag=False,
             filename_suffix='')

        # --- Uptake-kinetics table ------------------------------------
        # Same cohort and same runaway rule as every figure above.
        _try(f"Uptake kinetics table ({treatment})",
             write_uptake_kinetics_table,
             mechanics_df, output_dir, treatment=treatment,
             grouped_data=grouped_data)