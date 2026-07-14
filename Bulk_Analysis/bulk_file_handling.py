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
    post_pulse_entry: bool = False

    def __repr__(self):
        tags = []
        if self.post_pulse_entry: tags.append('POST')
        tag_str = ('|' + '|'.join(tags)) if tags else ''
        return (f"TrapData(ID={self.trap_id}, "
                f"{self.metadata.cell_type}|{self.metadata.treatment}|"
                f"{self.metadata.voltage}V_{self.metadata.duration_label}"
                f"{tag_str})")


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
        loaded_traps = []

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

        _EXCLUDE_TAGS = ('_ruptured', '_irregular', '_doa')
        all_full_candidates = sorted(dir_full.glob("trap_*_detection_full*.csv")) if dir_full.exists() else []

        n_excluded = n_post_pulse = 0
        for f_file in all_full_candidates:
            stem_lower = f_file.stem.lower()
            if any(tag in stem_lower for tag in _EXCLUDE_TAGS):
                n_excluded += 1
                logger.debug(f"  Skipping flagged full file: {f_file.name}")
                continue

            is_post_pulse_entry = '_after' in stem_lower
            if is_post_pulse_entry:
                n_post_pulse += 1

            id_match = re.search(r"trap_(\d+)_", f_file.name, re.IGNORECASE)
            if not id_match: continue

            trap_id = int(id_match.group(1))

            prot_dict = self._csv_to_dict(f_file)

            u_file_name = f"Trap_{trap_id:02d}_Uptake_Data.csv"
            u_file = dir_uptake / u_file_name
            uptake_dict = {}

            if u_file.exists():
                uptake_dict = self._csv_to_dict(u_file, comment='#')
            else:
                u_file_alt = dir_uptake / f"Trap_{trap_id}_Uptake_Data.csv"
                if u_file_alt.exists():
                    uptake_dict = self._csv_to_dict(u_file_alt, comment='#')

            if uptake_dict:
                self._recompute_volumes_and_norms(uptake_dict, prot_dict)

            trap_data = TrapData(
                trap_id=trap_id,
                metadata=copy(meta),
                protrusion_data=prot_dict,
                uptake_data=uptake_dict,
                post_pulse_entry=is_post_pulse_entry
            )
            loaded_traps.append(trap_data)

        if n_excluded:
            logger.info(f"  Excluded {n_excluded} flagged trap(s).")
        if n_post_pulse:
            logger.info(f"  Loaded {n_post_pulse} post-pulse entry trap(s).")

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
    # Volume recomputation with region-appropriate geometry
    # -------------------------------------------------------------------
    def _recompute_volumes_and_norms(
        self,
        uptake_dict: Dict[str, np.ndarray],
        prot_dict: Dict[str, np.ndarray],
    ) -> None:
        """
        Recompute per-frame volumes and volume-normalized intensities using
        region-appropriate geometry, overwriting whatever was loaded from disk.

        Body  : sphere-equivalent           V_body = (4/(3*sqrt(pi))) * A_body^(3/2)
        Prot  : cylinder + hemisphere cap   V_prot = pi * r_eff^2 * L + (2/3)*pi*r_eff^3
        Total : additive                    V_total = V_body + V_prot

        L(t) is taken from the detection CSV column 'Protrusion_Length_um'
        (sub-pixel protrusion length from the kymograph pipeline).
        Body and prot CSVs come from the same acquisition stack, so their row
        counts should match; if they don't we truncate to the shorter and warn.
        """
        # --- Body volume: sphere-equivalent from segmented area --------------
        A_body = uptake_dict.get('Body_Area_um2')
        if A_body is None:
            logger.debug("Body_Area_um2 missing; skipping volume recomputation.")
            return

        # V_body = (4 / (3*sqrt(pi))) * A_body^(3/2)
        # Prefactor from V = (4/3) pi r^3 with r = sqrt(A/pi).
        sphere_prefactor = 4.0 / (3.0 * math.sqrt(math.pi))
        V_body = sphere_prefactor * np.power(A_body, 1.5)

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
        L_aligned = L_um[:n]
        V_body    = V_body[:n]

        # Guard against NaN L (rare, from failed detection frames): treat as 0
        L_safe = np.where(np.isfinite(L_aligned), L_aligned, 0.0)

        V_prot  = math.pi * (self.r_eff_um ** 2) * L_safe + self.V_prot_cap_um3
        V_total = V_body + V_prot

        # Overwrite / install volume columns
        uptake_dict['Volume_Body_um3']  = V_body
        uptake_dict['Volume_Prot_um3']  = V_prot
        uptake_dict['Volume_Total_um3'] = V_total

        # --- Volume-normalized intensities -----------------------------------
        pairs = [
            ('Body_VolNorm',       'Body_Intensity',       'Volume_Body_um3'),
            ('Protrusion_VolNorm', 'Protrusion_Intensity', 'Volume_Prot_um3'),
            ('Total_VolNorm',      'Total_Intensity',      'Volume_Total_um3'),
        ]
        for out_col, int_col, vol_col in pairs:
            if int_col not in uptake_dict:
                continue
            intensity = uptake_dict[int_col][:n]
            volume    = uptake_dict[vol_col]
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