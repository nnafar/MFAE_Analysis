# -*- coding: utf-8 -*-
"""
File Handling for Bulk MFAE Analysis.

Expected folder name format:
    YYMMDD_CellType_Treatment_ChipID_ExpNumber-xxxxxPa-xxxxV-xxxxms-framexxxx

Volume estimation (recomputed at load time, overwriting on-disk values):

    - Body  : sphere-equivalent from segmented cross-section area
              V_body = (4/(3*sqrt(pi))) * A_body^(3/2)
    - Prot  : cylinder + hemispherical cap using the effective aspiration
              radius r_eff (Son 2007), consistent with the viscoelastic fits
              V_prot = pi * r_eff^2 * L(t) + (2/3) * pi * r_eff^3
    - Total : V_total = V_body + V_prot   (additive, not applied to union area)

    Volume-normalized intensities I / V are recomputed from these volumes.
"""

import math
import re
import logging
import pandas as pd
import numpy as np
from copy import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local helper: effective aspiration radius (Son 2007)
# ---------------------------------------------------------------------------
# Kept in sync with `bulk_mechanics.compute_reff`; duplicated here to avoid a
# circular import (bulk_mechanics already imports bulk_file_handling).
def _fstar(width: float, height: float) -> float:
    dim_min = min(width, height)
    dim_max = max(width, height)
    x = dim_min / dim_max
    sum_term = sum(
        math.tanh(math.pi * n * x / 2) / n**5
        for n in range(1, 22, 2)
    )
    return 1.0 / ((1 + 1.0 / x)**2 * (1 - (192 / (math.pi**5 * x)) * sum_term))


def _compute_reff(width_um: float, height_um: float) -> float:
    """Effective cylindrical radius with matched hydraulic resistance."""
    fs  = _fstar(width_um, height_um)
    h_s = min(width_um, height_um)
    w_l = max(width_um, height_um)
    numerator   = (2.0 / (3.0 * math.pi)) * w_l * (h_s ** 3)
    denominator = (1 + h_s / w_l) ** 2 * fs
    return (numerator / denominator) ** 0.25


# ---------------------------------------------------------------------------
# Filename-tag policy
# ---------------------------------------------------------------------------
# Full-protrusion detection CSVs are named
#     trap_XX_detection_full[_<TAG>].csv
# where <TAG> is a manual annotation. The loader routes each candidate CSV
# into one of three outcomes:
#
#   * loaded with fate_status = 'intact'          -> mechanics fit as normal
#   * loaded with fate_status = 'ruptured_post'   -> mechanics fit on the
#                                                    pre-pulse window; fate
#                                                    is the outcome variable
#                                                    for Claim 4
#   * excluded entirely, counted only in attrition
#
# ANALYSIS_TAG_MAP: tag suffix -> fate_status for loaded cells.
# EXCLUSION_TAG_MAP: tag suffix -> attrition bucket for cells not loaded.
# Tag matching is case-insensitive and anchored to the end of the file stem
# to avoid substring collisions (e.g. a hypothetical "_POSTERIOR" matching
# "_POST").
ANALYSIS_TAG_MAP: Dict[str, str] = {
    ''       : 'intact',
    '_R'     : 'ruptured_post',
}

EXCLUSION_TAG_MAP: Dict[str, str] = {
    '_R0'         : 'ruptured_pre_pulse',
    '_DOA'        : 'not_viable',
    '_PI'         : 'not_viable',
    '_R_DOA'      : 'not_viable',
    '_R0_DOA'     : 'not_viable',
    '_after'      : 'post_pulse_arrival',
    '_POST'       : 'post_pulse_arrival',
    '_MASK'       : 'detection_failure',
    '_SHAPE'      : 'detection_failure',
    '_IRREGULAR'  : 'detection_failure',
}

# Ordered attrition bucket list, used when writing the summary CSV so
# columns come out in a stable order regardless of insertion order above.
ATTRITION_BUCKET_ORDER: List[str] = [
    'analysed_intact',
    'analysed_ruptured',
    'ruptured_pre_pulse',
    'not_viable',
    'post_pulse_arrival',
    'detection_failure',
]


def _parse_fate_tag(stem: str) -> Tuple[str, Optional[str]]:
    """
    Classify a detection-CSV filename stem by its trailing manual-annotation
    tag.

    Returns
    -------
    (kind, key)
        kind is one of 'analysis' or 'exclusion'.
        key is the fate_status (for 'analysis') or the attrition bucket
        (for 'exclusion').
        If the tag is unrecognised, returns ('exclusion', 'detection_failure')
        and logs a debug warning; unknown tags are treated conservatively as
        detection failures so they still show up in the attrition tally.
    """
    # Order tags longest-first so '_R_DOA' matches before '_R', and
    # '_R0_DOA' before '_R0'. Case-insensitive.
    all_tags = sorted(
        list(ANALYSIS_TAG_MAP.keys()) + list(EXCLUSION_TAG_MAP.keys()),
        key=len,
        reverse=True,
    )
    stem_lc = stem.lower()
    for tag in all_tags:
        if not tag:
            continue
        if stem_lc.endswith(tag.lower()):
            if tag in ANALYSIS_TAG_MAP:
                return 'analysis', ANALYSIS_TAG_MAP[tag]
            return 'exclusion', EXCLUSION_TAG_MAP[tag]

    # No tag -> intact (untagged is the analysis default)
    return 'analysis', ANALYSIS_TAG_MAP['']


@dataclass
class ExperimentMetadata:
    date: str
    cell_type: str
    treatment: str
    chip: str
    experiment_number: str
    pressure: int
    voltage: int
    duration: float
    duration_label: str
    pulse_frame: int
    condition_type: str
    full_path: Path
    fstar: Optional[float] = None


@dataclass
class TrapData:
    trap_id: int
    metadata: ExperimentMetadata
    protrusion_data: Dict[str, np.ndarray]
    uptake_data: Dict[str, np.ndarray]
    actin_data: Dict[str, np.ndarray] = field(default_factory=dict)
    fate_status: str = 'intact'   # 'intact' | 'ruptured_post'

    def __repr__(self):
        fate_tag = f"|{self.fate_status.upper()}" if self.fate_status != 'intact' else ''
        return (f"TrapData(ID={self.trap_id}, "
                f"{self.metadata.cell_type}|{self.metadata.treatment}|"
                f"{self.metadata.voltage}V_{self.metadata.duration_label}"
                f"{fate_tag})")


@dataclass
class ExperimentAttrition:
    """Per-experiment tally of candidate detection CSVs by outcome bucket."""
    experiment_folder: str
    cell_type: str
    treatment: str
    pressure: int
    voltage: int
    duration_ms: float
    duration_label: str
    condition_type: str
    bucket_counts: Dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return sum(self.bucket_counts.values())

    def as_row(self) -> Dict[str, object]:
        row = {
            'Experiment_Folder': self.experiment_folder,
            'Cell_Type': self.cell_type,
            'Treatment': self.treatment,
            'Pressure_Pa': self.pressure,
            'Voltage_V': self.voltage,
            'Duration_ms': self.duration_ms,
            'Duration_label': self.duration_label,
            'Condition_Type': self.condition_type,
            'Total_Candidates': self.total(),
        }
        for bucket in ATTRITION_BUCKET_ORDER:
            row[bucket] = self.bucket_counts.get(bucket, 0)
        return row


class BulkDataLoader:
    FOLDER_PATTERN = re.compile(
        r"(\d{6})_([^_\-]+)_([^_\-]+)_([^_\-]+)_([^_\-]+)"
        r"[-_](\d+)Pa"
        r"[-_](\d+)V"
        r"[-_](\d+)(ms|us|µs|s)"
        r"[-_]frame(\d+)",
        re.IGNORECASE
    )

    def __init__(
        self,
        root_output_dir: str,
        channel_width_um: float = 6.7,
        channel_height_um: float = 5.0,
        r_eff_um: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        root_output_dir
            Root directory containing per-experiment output folders.
        channel_width_um, channel_height_um
            Physical channel cross-section, used to derive r_eff via the Son
            (2007) formula if r_eff_um is not supplied.
        r_eff_um
            Effective aspiration radius (um). If None, computed from
            channel_width_um and channel_height_um. Pass an explicit value to
            keep the loader in lockstep with bulk_mechanics.DEFAULT_R_EFF.
        """
        self.root_dir = Path(root_output_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        self.grouped_data: Dict[Tuple[str, str, int, int, float], List[TrapData]] = {}
        # Per-experiment attrition tallies, populated during iter_groups().
        # Serialised via get_attrition_df() after loading is complete.
        self.attrition_records: List[ExperimentAttrition] = []

        self.W_um: float = float(channel_width_um)
        self.H_um: float = float(channel_height_um)
        self.r_eff_um: float = (
            float(r_eff_um) if r_eff_um is not None
            else _compute_reff(self.W_um, self.H_um)
        )
        # Hemispherical cap volume for the protrusion: (2/3) * pi * r_eff^3
        self.V_prot_cap_um3: float = (2.0 / 3.0) * math.pi * (self.r_eff_um ** 3)

        logger.info(
            f"BulkDataLoader geometry: W={self.W_um:.2f} um, H={self.H_um:.2f} um, "
            f"r_eff={self.r_eff_um:.3f} um, V_cap={self.V_prot_cap_um3:.2f} um^3"
        )

    # -------------------------------------------------------------------
    # Iteration / metadata parsing
    # -------------------------------------------------------------------
    def iter_groups(self):
        logger.info(f"Scanning root directory: {self.root_dir}")
        grouped_metadata = {}

        for entry in self.root_dir.iterdir():
            if entry.is_dir():
                metadata = self._parse_folder_name(entry.name, entry)
                if metadata:
                    group_key = (
                        metadata.cell_type,
                        metadata.treatment,
                        metadata.pressure,
                        metadata.voltage,
                        metadata.duration,
                    )
                    if group_key not in grouped_metadata:
                        grouped_metadata[group_key] = []
                    grouped_metadata[group_key].append(metadata)

        for group_key, meta_list in grouped_metadata.items():
            traps = []
            for meta in meta_list:
                logger.info(f"Loading Experiment: {meta.full_path.name}")
                traps.extend(self._load_experiment_traps(meta))

            if traps:
                logger.info(f"   -> Yielding group {group_key} ({len(traps)} traps).")
                yield group_key, traps

    def _parse_folder_name(self, folder_name: str, full_path: Path) -> Optional[ExperimentMetadata]:
        match = self.FOLDER_PATTERN.match(folder_name)
        if match:
            date_str        = match.group(1)
            cell_type       = match.group(2)
            treatment       = match.group(3)
            chip            = match.group(4)
            experiment_num  = match.group(5)
            pressure        = int(match.group(6))
            volts           = int(match.group(7))
            raw_val         = float(match.group(8))
            unit            = match.group(9).lower()
            frame           = int(match.group(10))

            if unit in ['µs', '\u03bcs']:
                unit = 'us'

            val_fmt = int(raw_val) if raw_val.is_integer() else raw_val
            label = f"{val_fmt}{unit}"

            duration_ms = raw_val
            if unit == 's':
                duration_ms = raw_val * 1000.0
            elif unit == 'us':
                duration_ms = raw_val / 1000.0

            condition_type = "ASP" if volts == 0 else "EP"

            return ExperimentMetadata(
                date=date_str,
                cell_type=cell_type,
                treatment=treatment,
                chip=chip,
                experiment_number=experiment_num,
                pressure=pressure,
                voltage=volts,
                duration=duration_ms,
                duration_label=label,
                pulse_frame=frame,
                condition_type=condition_type,
                full_path=full_path
            )

        if not folder_name.startswith(("Bulk_", "__", ".")):
            logger.debug(f"Skipping folder (no regex match): {folder_name}")
        return None

    # -------------------------------------------------------------------
    # Per-experiment loading
    # -------------------------------------------------------------------
    def _load_experiment_traps(self, meta: ExperimentMetadata) -> List[TrapData]:
        loaded_traps: List[TrapData] = []

        # Attrition tally for this experiment. Every candidate CSV lands in
        # exactly one bucket, so the counts sum to Total_Candidates.
        bucket_counts: Dict[str, int] = {b: 0 for b in ATTRITION_BUCKET_ORDER}

        # ---- f* from shear CSV (unchanged) ----------------------------
        dir_fitting = meta.full_path / "Fitting results"
        if dir_fitting.exists():
            shear_files = list(dir_fitting.glob("*_shear_analysis.csv"))
            if shear_files:
                try:
                    shear_df = pd.read_csv(shear_files[0])
                    if 'Son_Factor_fstar' in shear_df.columns and len(shear_df) > 0:
                        meta.fstar = float(shear_df['Son_Factor_fstar'].iloc[0])
                        logger.debug(f"   f* = {meta.fstar:.4f} (from {shear_files[0].name})")
                except Exception as e:
                    logger.warning(f"   Could not read shear CSV: {e}")

        dir_uptake = meta.full_path / "Dye Uptake"
        dir_full   = meta.full_path / "Full protrusion detection"
        dir_actin  = meta.full_path / "Actin Quantification"

        all_full_candidates = (
            sorted(dir_full.glob("trap_*_detection_full*.csv"))
            if dir_full.exists() else []
        )

        for f_file in all_full_candidates:
            stem = f_file.stem
            # Strip the fixed prefix so _parse_fate_tag only sees the trailing
            # annotation. Expected stem shape: 'trap_XX_detection_full[_TAG]'.
            prefix_match = re.match(
                r"^trap_\d+_detection_full", stem, re.IGNORECASE
            )
            if not prefix_match:
                logger.debug(f"  Unrecognised filename shape, skipping: {f_file.name}")
                bucket_counts['detection_failure'] += 1
                continue

            trailing = stem[prefix_match.end():]  # '' or '_TAG'
            kind, key = _parse_fate_tag(trailing)

            if kind == 'exclusion':
                bucket_counts[key] += 1
                logger.debug(f"  Excluded ({key}): {f_file.name}")
                continue

            # ---- Passed the exclusion gate ------------------------------
            id_match = re.search(r"trap_(\d+)_", f_file.name, re.IGNORECASE)
            if not id_match:
                bucket_counts['detection_failure'] += 1
                logger.debug(f"  No trap ID in filename, skipping: {f_file.name}")
                continue
            trap_id = int(id_match.group(1))

            # Load protrusion
            prot_dict = self._csv_to_dict(f_file)

            # Load uptake (support zero-padded and unpadded trap IDs)
            uptake_dict: Dict[str, np.ndarray] = {}
            u_candidates = [
                dir_uptake / f"Trap_{trap_id:02d}_Uptake_Data.csv",
                dir_uptake / f"Trap_{trap_id}_Uptake_Data.csv",
            ]
            for u_file in u_candidates:
                if u_file.exists():
                    uptake_dict = self._csv_to_dict(u_file, comment='#')
                    break

            if uptake_dict:
                self._recompute_volumes_and_norms(uptake_dict, prot_dict)

            # Load actin (same padding fallback pattern)
            actin_dict: Dict[str, np.ndarray] = {}
            a_candidates = [
                dir_actin / f"Trap_{trap_id:02d}_Actin_Data.csv",
                dir_actin / f"Trap_{trap_id}_Actin_Data.csv",
            ]
            for a_file in a_candidates:
                if a_file.exists():
                    actin_dict = self._csv_to_dict(a_file, comment='#')
                    break
            if not actin_dict:
                logger.debug(
                    f"  No actin CSV found for trap {trap_id} in {dir_actin.name}; "
                    "actin columns will be NaN for this trap."
                )

            trap_data = TrapData(
                trap_id=trap_id,
                metadata=copy(meta),
                protrusion_data=prot_dict,
                uptake_data=uptake_dict,
                actin_data=actin_dict,
                fate_status=key,   # 'intact' or 'ruptured_post'
            )
            loaded_traps.append(trap_data)

            if key == 'intact':
                bucket_counts['analysed_intact'] += 1
            elif key == 'ruptured_post':
                bucket_counts['analysed_ruptured'] += 1

        # Persist the tally for this experiment
        self.attrition_records.append(ExperimentAttrition(
            experiment_folder=meta.full_path.name,
            cell_type=meta.cell_type,
            treatment=meta.treatment,
            pressure=meta.pressure,
            voltage=meta.voltage,
            duration_ms=meta.duration,
            duration_label=meta.duration_label,
            condition_type=meta.condition_type,
            bucket_counts=bucket_counts,
        ))

        # Summary log at INFO; per-file rejections already logged at DEBUG above.
        total = sum(bucket_counts.values())
        n_loaded = bucket_counts['analysed_intact'] + bucket_counts['analysed_ruptured']
        logger.info(
            f"  Attrition: {n_loaded}/{total} loaded  "
            f"(intact={bucket_counts['analysed_intact']}, "
            f"ruptured={bucket_counts['analysed_ruptured']}, "
            f"R0={bucket_counts['ruptured_pre_pulse']}, "
            f"non-viable={bucket_counts['not_viable']}, "
            f"post-pulse arrival={bucket_counts['post_pulse_arrival']}, "
            f"detection failures={bucket_counts['detection_failure']})"
        )

        return loaded_traps

    def _csv_to_dict(self, file_path: Path, comment: str = None) -> Dict[str, np.ndarray]:
        try:
            df = pd.read_csv(file_path, comment=comment)
            return {
                col: pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
                for col in df.columns
            }
        except Exception as e:
            logger.error(f"Error loading CSV {file_path.name}: {e}")
            return {}

    # -------------------------------------------------------------------
    # Attrition serialisation
    # -------------------------------------------------------------------
    def get_attrition_df(self) -> pd.DataFrame:
        """
        Return per-experiment attrition tally as a DataFrame.

        One row per experiment folder, with columns for total candidate
        detection CSVs and counts in each attrition bucket. Empty if no
        experiments have been iterated yet.
        """
        if not self.attrition_records:
            return pd.DataFrame()
        return pd.DataFrame([rec.as_row() for rec in self.attrition_records])

    # -------------------------------------------------------------------
    # Volume recomputation with region-appropriate geometry
    # -------------------------------------------------------------------
    def _recompute_volumes_and_norms(
        self,
        uptake_dict: Dict[str, np.ndarray],
        prot_dict: Dict[str, np.ndarray],
    ) -> None:
        """
        Recompute per-frame areas, volumes, and volume-normalized dye
        intensities from the trimmed uptake CSV schema.

        Input CSV columns (from UptakeQuantification.export_csv):
            Body_Intensity, Prot_Intensity, Tip_Intensity
            Body_Mask_Count, Prot_Mask_Count, Tip_Mask_Count
            F0_Body, F0_Prot, F0_Tip
            Time_s

        Bulk pipeline is now the single source of truth for size and
        volume math.  Derived quantities are installed on uptake_dict in
        place, matching the historical column names so downstream code
        (bulk_mechanics, thesis_plotting) needs no further changes:

            Body_Area_um2, Prot_Area_um2, Total_Area_um2
            Volume_Body_um3, Volume_Prot_um3, Volume_Total_um3
            Body_VolNorm, Protrusion_VolNorm, Total_VolNorm
            Total_Intensity (= Body_Intensity + Prot_Intensity weighted by area)

        Geometry
        --------
        Body : sphere-equivalent     V_body = (4/(3*sqrt(pi))) * A_body^(3/2)
        Prot : cylinder + hemi cap   V_prot = pi * r_eff^2 * L + (2/3)*pi*r_eff^3
        Total : additive             V_total = V_body + V_prot

        L(t) is taken from the detection CSV column 'Protrusion_Length_um'
        (sub-pixel protrusion length from the kymograph pipeline).
        """
        # --- Detect schema and derive areas ---------------------------------
        # New schema uses mask counts + a broadcast scale factor column;
        # historical schema stored the areas directly.  Support both so
        # older CSVs still load.  Scale factor is per-acquisition and
        # varies between experiments, so we prefer the CSV's own value
        # over any global default.
        sf_arr = uptake_dict.get('Scale_Factor_um_per_px')
        if sf_arr is not None and len(sf_arr) > 0 and np.isfinite(sf_arr[0]):
            sf_um_per_px = float(sf_arr[0])
        else:
            # Fall back to the historical hard-coded value.  Logged once at
            # debug level to avoid spam when the whole cohort predates the
            # column.
            sf_um_per_px = 0.629
            logger.debug(
                "Scale_Factor_um_per_px missing from uptake CSV; "
                "assuming 0.629 um/px."
            )
        sf2 = sf_um_per_px ** 2

        def _area_from_counts(count_key: str, legacy_area_key: str):
            """Return an area array in um^2 from either new or legacy CSVs."""
            if count_key in uptake_dict:
                counts = np.asarray(uptake_dict[count_key], dtype=float)
                return counts * sf2
            if legacy_area_key in uptake_dict:
                return np.asarray(uptake_dict[legacy_area_key], dtype=float)
            return None

        A_body = _area_from_counts('Body_Mask_Count', 'Body_Area_um2')
        A_prot = _area_from_counts('Prot_Mask_Count', 'Protrusion_Area_um2')

        if A_body is None:
            logger.debug(
                "Uptake CSV lacks Body_Mask_Count and Body_Area_um2; "
                "skipping volume recomputation."
            )
            return

        # Volume: sphere-equivalent from segmented area.
        # V_body = (4 / (3*sqrt(pi))) * A_body^(3/2)   [prefactor from
        # V = (4/3) pi r^3 with r = sqrt(A/pi)]
        sphere_prefactor = 4.0 / (3.0 * math.sqrt(math.pi))
        V_body = sphere_prefactor * np.power(np.clip(A_body, 0.0, None), 1.5)

        # --- Protrusion volume: cylinder + hemispherical cap -----------------
        L_um = prot_dict.get('Protrusion_Length_um')
        if L_um is None:
            logger.debug(
                "Protrusion_Length_um missing from detection CSV; "
                "cannot recompute confined protrusion volume."
            )
            return

        # Align by row index; both CSVs are one row per acquisition frame.
        n = min(len(L_um), len(V_body))
        if len(L_um) != len(V_body):
            logger.warning(
                f"Frame-count mismatch between detection ({len(L_um)}) and "
                f"uptake ({len(V_body)}) CSVs; truncating to {n}."
            )
        L_aligned = np.asarray(L_um[:n], dtype=float)
        V_body    = V_body[:n]
        if A_body is not None:
            A_body = np.asarray(A_body[:n], dtype=float)
        if A_prot is not None:
            A_prot = np.asarray(A_prot[:n], dtype=float)

        # Guard against NaN L (rare, from failed detection frames): treat as 0
        L_safe = np.where(np.isfinite(L_aligned), L_aligned, 0.0)

        V_prot  = math.pi * (self.r_eff_um ** 2) * L_safe + self.V_prot_cap_um3
        V_total = V_body + V_prot

        # Install area columns so downstream code that expects the old names
        # keeps working.  Prot / total areas are also derived here so mask
        # counts do not need to leak into other modules.
        uptake_dict['Body_Area_um2']  = A_body
        if A_prot is not None:
            uptake_dict['Protrusion_Area_um2'] = A_prot
            uptake_dict['Total_Area_um2']      = A_body + A_prot

        uptake_dict['Volume_Body_um3']  = V_body
        uptake_dict['Volume_Prot_um3']  = V_prot
        uptake_dict['Volume_Total_um3'] = V_total

        # --- Total intensity (area-weighted) ---------------------------------
        # The trimmed CSV does not carry Total_Intensity because it isn't a
        # measured quantity; it's a derived summary.  Compute it here from
        # the mean intensities and mask counts using an area-weighted mean.
        if ('Body_Intensity' in uptake_dict and 'Prot_Intensity' in uptake_dict
                and A_prot is not None):
            body_int = np.asarray(uptake_dict['Body_Intensity'][:n], dtype=float)
            prot_int = np.asarray(uptake_dict['Prot_Intensity'][:n], dtype=float)
            denom    = A_body + A_prot
            with np.errstate(divide='ignore', invalid='ignore'):
                total_int = np.where(
                    denom > 0,
                    (body_int * A_body + prot_int * A_prot) / denom,
                    0.0,
                )
            uptake_dict['Total_Intensity'] = total_int
            # Also trim Body_Intensity / Prot_Intensity / Tip_Intensity to n
            # so all per-frame columns have consistent lengths.
            uptake_dict['Body_Intensity'] = body_int
            uptake_dict['Prot_Intensity'] = prot_int
            if 'Tip_Intensity' in uptake_dict:
                uptake_dict['Tip_Intensity'] = np.asarray(
                    uptake_dict['Tip_Intensity'][:n], dtype=float)

        # --- Volume-normalized intensities -----------------------------------
        pairs = [
            ('Body_VolNorm',       'Body_Intensity',       'Volume_Body_um3'),
            ('Protrusion_VolNorm', 'Prot_Intensity',       'Volume_Prot_um3'),
            ('Total_VolNorm',      'Total_Intensity',      'Volume_Total_um3'),
        ]
        for out_col, int_col, vol_col in pairs:
            if int_col not in uptake_dict:
                continue
            intensity = np.asarray(uptake_dict[int_col][:n], dtype=float)
            volume    = np.asarray(uptake_dict[vol_col],   dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                uptake_dict[out_col] = np.where(volume > 0, intensity / volume, 0.0)

        # Trim any other per-frame columns to the same length so downstream
        # code sees a consistent frame count.
        for k, v in list(uptake_dict.items()):
            if isinstance(v, np.ndarray) and len(v) > n:
                uptake_dict[k] = v[:n]


def extract_all_scalars(grouped_data: Dict[Tuple[str, str, int, int, float], List[TrapData]]) -> pd.DataFrame:
    rows = []
    for (cell_type, treatment, press, volt, dur), traps in grouped_data.items():
        for trap in traps:
            row_data = {
                'Condition': f"{cell_type}_{treatment}_{press}Pa_{volt}V_{dur}ms",
                'Condition_Type': trap.metadata.condition_type,
                'Date': trap.metadata.date,
                'Cell_Type': cell_type,
                'Treatment': treatment,
                'Experiment_Number': trap.metadata.experiment_number,
                'Pressure_Pa': press,
                'Voltage_V': volt,
                'Duration_ms': dur,
                'Trap_ID': trap.trap_id
            }

            if hasattr(trap, 'protrusion_data') and trap.protrusion_data:
                for k, v in trap.protrusion_data.items():
                    if 'time' not in k.lower() and isinstance(v, np.ndarray) and len(v) > 0:
                        row_data[f"Prot_{k}"] = np.max(v)

            if hasattr(trap, 'uptake_data') and trap.uptake_data:
                for k, v in trap.uptake_data.items():
                    if 'time' not in k.lower() and isinstance(v, np.ndarray) and len(v) > 0:
                        row_data[f"Uptake_{k}"] = np.max(v)

            rows.append(row_data)

    df = pd.DataFrame(rows)
    return df


def get_group_stats(grouped_data: Dict[Tuple[str, str, int, int, float], List[TrapData]]):
    summary = []
    sorted_keys = sorted(grouped_data.keys())
    for (cell_type, treatment, press, volt, dur) in sorted_keys:
        traps = grouped_data[(cell_type, treatment, press, volt, dur)]
        if not traps: continue
        label = traps[0].metadata.duration_label
        ctype = traps[0].metadata.condition_type
        summary.append(
            f"[{ctype}] {cell_type} | {treatment} | "
            f"{press}Pa | {volt}V {label}: {len(traps)} traps found."
        )
    return "\n".join(summary)