# =============================================================================
# thesis_figure_data.py
#
# Exports every value that sits behind a Chapter 3 figure to CSV, so any
# figure can be redrawn without re-running the pipeline.
#
# WHY THIS EXISTS
# ---------------
# `mechanics_results_all_traps.csv` already holds one row per cell with every
# fitted scalar, so the boxplot and correlation figures can be redrawn from it
# directly. What it does not hold is anything with a time axis: protrusion
# length L(t), actin I(t)/F0, and the uptake traces all live inside the
# per-trap objects the loader builds from the raw acquisitions. Those seven
# figures therefore needed a full pipeline run for even a cosmetic change.
#
# This module writes those time series out in long format, one row per
# (cell, frame), with enough identity columns to rebuild any cohort filter
# used in the chapter.
#
# WHAT IT DELIBERATELY DOES NOT DO
# --------------------------------
# It does not export the aggregated curves (the median and IQR bands the
# figures actually draw). Exporting the per-cell traces instead means the
# aggregation stays in one place, inside the plotting functions, rather than
# being duplicated here where it could silently drift. Anything the figures
# compute can be recomputed from these files; the reverse is not true.
#
# USAGE
# -----
# From the pipeline, one line after mechanics_df is built:
#
#     import thesis_figure_data as tfd
#     _try("Figure Data Export", tfd.export_all,
#          all_grouped_data, mechanics_df, attrition_df, results_dir)
#
# Standalone afterwards:
#
#     import thesis_figure_data as tfd
#     data = tfd.load_all("Bulk_Analysis_results/<date>/Figure_Data")
#     data["protrusion"]   # long-format DataFrame
#     data["mechanics"]    # per-cell scalars
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sub-directory created under the results directory.
DATA_DIRNAME = "Figure_Data"

# One output file per channel. The keys double as the keys returned by
# load_all(), so add a channel here and both directions stay in step.
CHANNEL_FILES: Dict[str, str] = {
    "protrusion": "Ch3_Data_Traces_Protrusion.csv",
    "actin":      "Ch3_Data_Traces_Actin.csv",
    "uptake":     "Ch3_Data_Traces_Uptake.csv",
}
MECHANICS_FILE = "Ch3_Data_Mechanics.csv"
ATTRITION_FILE = "Ch3_Data_Attrition.csv"
MANIFEST_FILE  = "Ch3_Data_Manifest.csv"

# Identity columns written ahead of the data columns in every trace file.
# Together these reproduce any cohort filter used in Chapter 3: treatment,
# pulse protocol, fate, and trap position.
_IDENTITY_COLUMNS: Tuple[str, ...] = (
    "Experiment_Folder", "Trap_ID", "Date", "Cell_Type", "Treatment",
    "Chip", "Experiment_Number", "Condition_Type", "Pressure_Pa",
    "Voltage_V", "Duration_ms", "Duration_label", "Fate_Status",
    "Pulse_Frame",
)

# Candidate names for the per-frame time column inside each data dict. The
# dicts are read straight from the per-cell CSV headers, so the exporter
# looks for the first name that is present rather than assuming one.
_TIME_KEY_CANDIDATES: Tuple[str, ...] = ("Time_s", "Time", "time_s")


# =============================================================================
# 1. HELPERS
# =============================================================================

def _identity_row(trap) -> Dict[str, object]:
    """
    Pull the identity fields off a TrapData object.

    `Experiment_Folder` is `metadata.full_path.name`, which is the same key
    the mechanics table uses, so the trace files join to
    mechanics_results_all_traps.csv on (Experiment_Folder, Trap_ID).
    """
    meta = trap.metadata
    return {
        "Experiment_Folder": meta.full_path.name,
        "Trap_ID":           trap.trap_id,
        "Date":              meta.date,
        "Cell_Type":         meta.cell_type,
        "Treatment":         meta.treatment,
        "Chip":              meta.chip,
        "Experiment_Number": meta.experiment_number,
        "Condition_Type":    meta.condition_type,
        "Pressure_Pa":       meta.pressure,
        "Voltage_V":         meta.voltage,
        "Duration_ms":       meta.duration,
        "Duration_label":    meta.duration_label,
        "Fate_Status":       trap.fate_status,
        "Pulse_Frame":       meta.pulse_frame,
    }


def _find_time_key(data: Dict[str, np.ndarray]) -> Optional[str]:
    """Return the name of the per-frame time column, or None if absent."""
    for key in _TIME_KEY_CANDIDATES:
        if key in data:
            return key
    return None


def _frame_length(data: Dict[str, np.ndarray], time_key: str) -> int:
    """Number of frames, taken from the time column."""
    return int(np.asarray(data[time_key]).size)


def _dict_to_frame(data: Dict[str, np.ndarray],
                   n_frames: int,
                   channel: str,
                   trap_label: str,
                   skipped: List[Dict[str, object]]) -> pd.DataFrame:
    """
    Turn one trap's data dict into a DataFrame of n_frames rows.

    Every key is exported rather than a hand-picked list, so a column added
    upstream appears here without this module needing an edit. Three cases
    are handled:

      - length == n_frames : a per-frame column, taken as is.
      - length == 1        : a per-cell scalar, broadcast down the column.
      - anything else      : recorded in `skipped` and left out, because
                             guessing an alignment would silently corrupt
                             the time axis.

    Constant-valued per-frame columns (the F0_* baselines are written this
    way upstream) fall into the first case and stay intact.
    """
    columns: Dict[str, np.ndarray] = {}

    for key, value in data.items():
        arr = np.asarray(value)

        if arr.ndim == 0:
            columns[key] = np.repeat(arr, n_frames)
            continue

        if arr.ndim > 1:
            skipped.append({"channel": channel, "trap": trap_label,
                            "column": key, "reason": f"ndim={arr.ndim}"})
            continue

        if arr.size == n_frames:
            columns[key] = arr
        elif arr.size == 1:
            columns[key] = np.repeat(arr[0], n_frames)
        else:
            skipped.append({"channel": channel, "trap": trap_label,
                            "column": key,
                            "reason": f"length {arr.size} != {n_frames}"})

    return pd.DataFrame(columns)


def _add_time_axes(df: pd.DataFrame,
                   time_key: str,
                   pulse_frame: Optional[int],
                   condition_type: object,
                   length_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add the two derived time axes the Chapter 3 figures plot against.

    `Time_since_pulse_s` is zero at the pulse frame, which is the axis used
    by the uptake and actin-around-pulse figures. It is NaN for aspiration-
    only cells, which never received a pulse.

    The aspiration-only test is on `condition_type`, not on `pulse_frame`.
    Those cells carry `pulse_frame = 0` rather than a missing value, and 0
    is a valid index, so a bounds check alone silently produces a
    pulse-relative axis measured from the first frame of an experiment
    where no pulse occurred.

    `Time_from_entry_s` is zero at the first frame where the protrusion has
    a non-zero length, which is the re-zeroing the pre-pulse trace figure
    applies so cells that enter the trap late are not offset from the rest.
    Frames before entry keep negative values so they can be trimmed or
    retained downstream. Only meaningful for the protrusion channel.
    """
    t = df[time_key].to_numpy(dtype=float)

    is_pulsed = str(condition_type).upper() != "ASP"
    if (is_pulsed and pulse_frame is not None
            and 0 <= int(pulse_frame) < len(t)):
        df["Time_since_pulse_s"] = t - t[int(pulse_frame)]
    else:
        df["Time_since_pulse_s"] = np.nan

    if length_col is not None and length_col in df.columns:
        lengths = df[length_col].to_numpy(dtype=float)
        entered = np.flatnonzero(np.isfinite(lengths) & (lengths > 0))
        if entered.size:
            df["Time_from_entry_s"] = t - t[entered[0]]
        else:
            df["Time_from_entry_s"] = np.nan

    return df


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Identity columns first, then time axes, then the data columns."""
    identity = [c for c in _IDENTITY_COLUMNS if c in df.columns]
    time_cols = [c for c in ("Time_s", "Time", "time_s",
                             "Time_since_pulse_s", "Time_from_entry_s")
                 if c in df.columns]
    rest = [c for c in df.columns if c not in identity and c not in time_cols]
    return df[identity + time_cols + rest]


def _iter_traps(grouped_data):
    """
    Yield every TrapData in a grouped_data mapping.

    Works with a plain dict and with the pipeline's DiskBackedDict, since
    both support .items() returning lists of traps.
    """
    for group_key, traps in grouped_data.items():
        if not traps:
            continue
        for trap in traps:
            yield trap


# =============================================================================
# 2. EXPORT
# =============================================================================

def export_traces(grouped_data,
                  output_dir: Path,
                  channels: Optional[Tuple[str, ...]] = None
                  ) -> Dict[str, pd.DataFrame]:
    """
    Write one long-format CSV per data channel and return them as DataFrames.

    Parameters
    ----------
    grouped_data
        The pipeline's mapping of group key -> list of TrapData.
    output_dir
        Directory the CSVs are written into. Created if absent.
    channels
        Which channels to export. Defaults to all three.

    Returns
    -------
    dict of channel name -> DataFrame (empty DataFrame where a channel had
    no usable data).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if channels is None:
        channels = tuple(CHANNEL_FILES.keys())

    attr_for_channel = {
        "protrusion": "protrusion_data",
        "actin":      "actin_data",
        "uptake":     "uptake_data",
    }

    collected: Dict[str, List[pd.DataFrame]] = {c: [] for c in channels}
    skipped: List[Dict[str, object]] = []
    n_traps = 0
    no_time: Dict[str, int] = {c: 0 for c in channels}
    empty: Dict[str, int] = {c: 0 for c in channels}

    for trap in _iter_traps(grouped_data):
        n_traps += 1
        identity = _identity_row(trap)
        trap_label = f"{identity['Experiment_Folder']}/T{trap.trap_id}"

        for channel in channels:
            data = getattr(trap, attr_for_channel[channel], None) or {}
            if not data:
                empty[channel] += 1
                continue

            time_key = _find_time_key(data)
            if time_key is None:
                # Without a time column the rows cannot be ordered or
                # aligned to the pulse, so the trap is left out rather than
                # exported with an implicit frame-index axis.
                no_time[channel] += 1
                skipped.append({"channel": channel, "trap": trap_label,
                                "column": "(whole trap)",
                                "reason": "no time column found"})
                continue

            n_frames = _frame_length(data, time_key)
            if n_frames == 0:
                empty[channel] += 1
                continue

            df = _dict_to_frame(data, n_frames, channel, trap_label, skipped)

            length_col = None
            if channel == "protrusion":
                for candidate in ("Protrusion_Length_um", "Protrusion_Length"):
                    if candidate in df.columns:
                        length_col = candidate
                        break

            df = _add_time_axes(df, time_key, identity["Pulse_Frame"],
                                identity["Condition_Type"], length_col)

            for col, value in identity.items():
                df[col] = value

            collected[channel].append(_order_columns(df))

    results: Dict[str, pd.DataFrame] = {}
    for channel in channels:
        frames = collected[channel]
        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame()
            logger.warning(f"  No {channel} data collected; writing empty file.")

        path = output_dir / CHANNEL_FILES[channel]
        out.to_csv(path, index=False)
        results[channel] = out

        if len(out):
            n_cells = out.groupby(["Experiment_Folder", "Trap_ID"]).ngroups
        else:
            n_cells = 0
        logger.info(f"  {path.name}: {len(out)} rows, {n_cells} cells, "
                    f"{len(out.columns)} columns")
        if empty[channel] or no_time[channel]:
            logger.info(
                f"    ({empty[channel]} traps had no {channel} data, "
                f"{no_time[channel]} had no time column)"
            )

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / MANIFEST_FILE, index=False)
        logger.info(
            f"  {MANIFEST_FILE}: {len(skipped)} column/trap combinations "
            "were not exported; see the file for reasons."
        )

    logger.info(f"  Scanned {n_traps} traps.")
    return results


def export_all(grouped_data,
               mechanics_df: pd.DataFrame,
               attrition_df: Optional[pd.DataFrame],
               results_dir: Path) -> Path:
    """
    Write the complete Chapter 3 figure-data bundle and return its directory.

    The bundle is self-contained: the per-cell scalars, the attrition tally,
    and the three trace files. Every figure in the chapter can be redrawn
    from these alone.
    """
    data_dir = Path(results_dir) / DATA_DIRNAME
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting Chapter 3 figure data...")

    if mechanics_df is not None and not mechanics_df.empty:
        mechanics_df.to_csv(data_dir / MECHANICS_FILE, index=False)
        logger.info(f"  {MECHANICS_FILE}: {len(mechanics_df)} cells, "
                    f"{len(mechanics_df.columns)} columns")

    if attrition_df is not None and not attrition_df.empty:
        attrition_df.to_csv(data_dir / ATTRITION_FILE, index=False)
        logger.info(f"  {ATTRITION_FILE}: {len(attrition_df)} experiments")

    export_traces(grouped_data, data_dir)

    logger.info(f"Figure data written to: {data_dir}")
    return data_dir


# =============================================================================
# 3. LOAD
# =============================================================================

def load_all(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Read a bundle written by export_all().

    Returns a dict with keys 'mechanics', 'attrition', 'protrusion',
    'actin' and 'uptake'. Missing files are returned as empty DataFrames
    rather than raising, so a partial bundle is still usable.
    """
    data_dir = Path(data_dir)
    out: Dict[str, pd.DataFrame] = {}

    named = dict(CHANNEL_FILES)
    named["mechanics"] = MECHANICS_FILE
    named["attrition"] = ATTRITION_FILE

    for key, filename in named.items():
        path = data_dir / filename
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            logger.warning(f"{filename} not found in {data_dir}")
            out[key] = pd.DataFrame()

    return out


def cell_traces(traces_df: pd.DataFrame, **filters) -> pd.DataFrame:
    """
    Filter a trace table by any identity column, as a convenience.

        cell_traces(data["protrusion"], Treatment="WT",
                    Condition_Type="EP", Duration_label="100us")

    Each keyword is matched against the column of the same name. A list or
    tuple value matches any of its members.
    """
    df = traces_df
    for column, wanted in filters.items():
        if column not in df.columns:
            raise KeyError(
                f"{column!r} is not a column in this table. Available: "
                f"{list(df.columns)[:20]}"
            )
        if isinstance(wanted, (list, tuple, set)):
            df = df[df[column].isin(list(wanted))]
        else:
            df = df[df[column] == wanted]
    return df


# =============================================================================
# 4. REBUILD grouped_data FROM A BUNDLE
# =============================================================================
# The trace figures take `grouped_data`, the loader's mapping of group key ->
# list of TrapData. Rebuilding that structure from the bundle means every
# existing plotting function works unchanged, reading from CSV instead of from
# a pipeline run. The alternative, rewriting seven figure functions to consume
# DataFrames, would fork the aggregation logic and let the two versions drift.

def _metadata_from_row(row: pd.Series):
    """
    Rebuild an ExperimentMetadata from the identity columns of a trace row.

    `full_path` is set to Path(Experiment_Folder) so that
    `metadata.full_path.name` returns the folder name, which is the key the
    plotting functions and the mechanics table join on.
    """
    import bulk_file_handling as bfh

    return bfh.ExperimentMetadata(
        date              = row["Date"],
        cell_type         = row["Cell_Type"],
        treatment         = row["Treatment"],
        chip              = row["Chip"],
        experiment_number = row["Experiment_Number"],
        pressure          = int(row["Pressure_Pa"]),
        voltage           = int(row["Voltage_V"]),
        duration          = float(row["Duration_ms"]),
        duration_label    = row["Duration_label"],
        pulse_frame       = int(row["Pulse_Frame"]),
        condition_type    = row["Condition_Type"],
        full_path         = Path(str(row["Experiment_Folder"])),
    )


def _channel_arrays(df: Optional[pd.DataFrame]) -> Dict[str, np.ndarray]:
    """
    Turn one cell's slice of a trace table back into a dict of arrays,
    dropping the identity and derived-time columns that the exporter added.

    The derived columns are dropped rather than kept because the plotting
    functions recompute them from `Time_s` and `pulse_frame`. Leaving them
    in would mean two sources of truth for the same axis.
    """
    if df is None or df.empty:
        return {}

    drop = set(_IDENTITY_COLUMNS) | {"Time_since_pulse_s", "Time_from_entry_s"}
    return {col: df[col].to_numpy()
            for col in df.columns if col not in drop}


def grouped_data_from_bundle(data_dir: Path,
                             data: Optional[Dict[str, pd.DataFrame]] = None
                             ) -> Dict[Tuple, List]:
    """
    Rebuild a `grouped_data` mapping from an exported bundle.

    Group keys reproduce the loader's own scheme,
    (cell_type, treatment, pressure, voltage, duration), so functions that
    inspect the key behave as they do in a live run.

    Usage
    -----
        import thesis_figure_data as tfd
        import thesis_plotting as tp

        data    = tfd.load_all(data_dir)
        grouped = tfd.grouped_data_from_bundle(data_dir, data)

        tp.plot_thesis_combined_protrusion_dynamics(
            grouped, Path("chap-3/Ch3-Figures"))

    Notes
    -----
    A rebuilt trap carries only what the bundle holds. Fields the loader
    computes at read time and the exporter does not capture, such as
    `metadata.fstar`, come back as their defaults; `Son_Factor_fstar` is in
    the mechanics table if a figure needs it.
    """
    import bulk_file_handling as bfh

    if data is None:
        data = load_all(data_dir)

    prot = data.get("protrusion", pd.DataFrame())
    if prot.empty:
        raise ValueError(
            f"No protrusion traces found in {data_dir}. The bundle is the "
            "source of cell identity here, so it cannot be rebuilt without "
            "them."
        )

    key_cols = ["Experiment_Folder", "Trap_ID"]

    # Index the other two channels by cell so each trap can be assembled in
    # one pass rather than re-filtering the whole table per cell.
    def _by_cell(name: str):
        df = data.get(name, pd.DataFrame())
        if df.empty:
            return {}
        return {k: g for k, g in df.groupby(key_cols, sort=False)}

    actin_by_cell  = _by_cell("actin")
    uptake_by_cell = _by_cell("uptake")

    # Fate lives in the mechanics table as well as the traces; the traces are
    # used so the rebuild does not depend on the mechanics file being present.
    grouped: Dict[Tuple, List] = {}
    n_traps = 0

    for cell_key, cell_df in prot.groupby(key_cols, sort=False):
        first = cell_df.iloc[0]
        meta = _metadata_from_row(first)

        trap = bfh.TrapData(
            trap_id         = int(first["Trap_ID"]),
            metadata        = meta,
            protrusion_data = _channel_arrays(cell_df),
            uptake_data     = _channel_arrays(uptake_by_cell.get(cell_key)),
            actin_data      = _channel_arrays(actin_by_cell.get(cell_key)),
            fate_status     = str(first["Fate_Status"]),
        )

        group_key = (meta.cell_type, meta.treatment, meta.pressure,
                     meta.voltage, meta.duration)
        grouped.setdefault(group_key, []).append(trap)
        n_traps += 1

    logger.info(f"Rebuilt {n_traps} traps in {len(grouped)} groups "
                f"from {Path(data_dir).name}")
    return grouped


def window_durations_from_bundle(grouped_data) -> Dict[str, Optional[float]]:
    """
    Recompute the three global window durations the trace figures use.

    Several figure functions take `global_asp_dur`, `global_pre_dur` or
    `global_whole_dur` and set their x-limit from it. The pipeline computes
    these once in Phase 1.5 and threads them through; a standalone replot
    has to recompute them, or the axis will run to the longest recording
    instead of the common window and the figure will not match the one in
    the chapter.

    The rule reproduced here is the pipeline's: within each group, take the
    median trace duration over traps with at least 15 points, then take the
    minimum of those medians across groups, so every group enters a
    comparison with an equal-length trajectory.

    Returns a dict with keys 'asp', 'pre' and 'whole', each in seconds or
    None when no group qualified.
    """
    asp_meds: List[float] = []
    pre_meds: List[float] = []
    whole_meds: List[float] = []

    for group_key, traps in grouped_data.items():
        if not traps:
            continue
        ctype = traps[0].metadata.condition_type

        valid = [t for t in traps
                 if len(t.protrusion_data.get("Time_s", [])) >= 15]
        if not valid:
            continue

        w_durs = [float(t.protrusion_data["Time_s"][-1]
                        - t.protrusion_data["Time_s"][0]) for t in valid]
        if w_durs:
            whole_meds.append(float(np.median(w_durs)))

        if ctype == "ASP":
            if w_durs:
                asp_meds.append(float(np.median(w_durs)))
        elif ctype == "EP":
            p_durs = []
            for t in valid:
                pf = t.metadata.pulse_frame
                times = t.protrusion_data["Time_s"]
                if 15 <= pf < len(times):
                    p_durs.append(float(times[pf] - times[0]))
            if p_durs:
                pre_meds.append(float(np.median(p_durs)))

    return {
        "asp":   min(asp_meds) if asp_meds else None,
        "pre":   min(pre_meds) if pre_meds else None,
        "whole": min(whole_meds) if whole_meds else None,
    }