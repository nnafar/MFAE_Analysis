# -*- coding: utf-8 -*-
"""
File Handling for Bulk MFAE Analysis.

Expected folder name format:
    YYMMDD_CellType_Treatment_ExpNumber-xxxxxPa-xxxxV-xxxxms-framexxxx

Where:
    CellType     = e.g. MDAMB231, MCF10A
    Treatment    = e.g. Control, CytoD, LatA
    ExpNumber    = e.g. Experiment275
    Pa / V / ms  = integers (use 0V-0ms-frame0 for aspiration-only runs)

Grouping key: (CellType, Treatment, Pressure_Pa, Voltage_V, Duration_ms)
This means replicates that share the same cell type, treatment, and
electroporation parameters are pooled together for statistics and plots.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from copy import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Configure Logging
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentMetadata:
    """Holds metadata extracted from the folder name."""
    date: str
    cell_type: str       # ExperimentID1 — e.g. "MDAMB231"
    treatment: str       # ExperimentID2 — e.g. "Control", "CytoD"
    device: str          # Experiment ID3 - e.g. "Device1"
    experiment_number: str # ExperimentID4 — e.g. "Experiment275"
    pressure: int        # e.g. 1100  (from 1100Pa)
    voltage: int         # e.g. 100
    duration: float      # NORMALIZED TO MS (for sorting / physics)
    duration_label: str  # ORIGINAL STRING (e.g. "5us", "10s") for plotting
    pulse_frame: int     # e.g. 10
    condition_type: str  # "EP" (electroporated) or "ASP" (aspiration only, 0V)
    full_path: Path
    fstar: Optional[float] = None  # Son (2007) shape factor, loaded from shear CSV

@dataclass
class TrapData:
    """Holds all data arrays for a single trap."""
    trap_id: int
    metadata: ExperimentMetadata
    
    # Data containers
    protrusion_data: Dict[str, np.ndarray] # From "Filtered" CSV
    uptake_data: Dict[str, np.ndarray]     # From "Uptake" CSV

    def __repr__(self):
        return (f"TrapData(ID={self.trap_id}, "
                f"{self.metadata.cell_type}|{self.metadata.treatment}|"
                f"{self.metadata.voltage}V_{self.metadata.duration_label})")

# =============================================================================
# FILE LOADER CLASS
# =============================================================================

class BulkDataLoader:
    """
    Scans, parses, and loads MFAE experiment data.
    """
    
    # EXPECTED PATTERN:
    # Format: YYMMDD_CellType_Treatment_DeviceNumber_ExpNumber-xxxxxPa-xxxxV-xxxxms-framexxxx
    FOLDER_PATTERN = re.compile(
        r"(\d{6})_([^_\-]+)_([^_\-]+)_([^_\-]+)_([^_\-]+)"
        r"[-_](\d+)Pa"
        r"[-_](\d+)V"
        r"[-_](\d+)(ms|us|µs|s)"
        r"[-_]frame(\d+)",
        re.IGNORECASE
    )

    def __init__(self, root_output_dir: str):
        self.root_dir = Path(root_output_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # Key: (CellType, Treatment, Pressure_Pa, Voltage_V, Duration_ms)
        # Experiments that share all five values are pooled as replicates.
        self.grouped_data: Dict[Tuple[str, str, int, int, float], List[TrapData]] = {}

    def iter_groups(self):
        """
        Main execution method to browse and yield loaded data iteratively.
        Groups experiments by (CellType, Treatment, Pressure, Voltage, Duration).
        Yields one group of TrapData objects at a time for processing.
        """
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
            cell_type       = match.group(2)   # e.g. "MDAMB231"
            treatment       = match.group(3)   # e.g. "Control", "CytoD"
            device         = match.group(4)    # e.g. "Device1"
            experiment_num  = match.group(5)   # e.g. "Experiment275"
            pressure        = int(match.group(6))
            volts           = int(match.group(7))
            raw_val         = float(match.group(8))
            unit            = match.group(9).lower()
            frame           = int(match.group(10))

            # Normalise any mu variant to "us" for consistent handling.
            if unit in ['µs', '\u03bcs']:
                unit = 'us'

            # 1. Create the display label (e.g. "5us", "10ms")
            val_fmt = int(raw_val) if raw_val.is_integer() else raw_val
            label = f"{val_fmt}{unit}"

            # 2. Normalize duration to milliseconds for sorting
            duration_ms = raw_val
            if unit == 's':
                duration_ms = raw_val * 1000.0
            elif unit == 'us':
                duration_ms = raw_val / 1000.0

            # 3. Classify condition type.
            #    ASP = aspiration-only control (0 V, 0 ms, frame 0).
            #    EP  = electroporated (non-zero voltage).
            condition_type = "ASP" if volts == 0 else "EP"

            return ExperimentMetadata(
                date=date_str,
                cell_type=cell_type,
                treatment=treatment,
                experiment_number=experiment_num,
                pressure=pressure,
                voltage=volts,
                duration=duration_ms,
                duration_label=label,
                pulse_frame=frame,
                condition_type=condition_type,
                full_path=full_path
            )

        # FIX #14: Log folders that were scanned but did not match the regex.
        # Helps diagnose typos in folder names (e.g. missing "Pa" suffix).
        # Ignore known non-experiment folders (Bulk_Analysis_results, etc.)
        if not folder_name.startswith(("Bulk_", "__", ".")):
            logger.debug(f"Skipping folder (no regex match): {folder_name}")
        return None

    def _load_experiment_traps(self, meta: ExperimentMetadata) -> List[TrapData]:
        loaded_traps = []

        # --- Load Son Factor f* from shear analysis CSV (per-experiment) ---
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

        dir_filtered = meta.full_path / "Filtered protrusion detection"
        dir_uptake = meta.full_path / "Dye Uptake"

        # Find filtered files
        filtered_files = sorted(list(dir_filtered.glob("trap_*_detection_filtered.csv")))

        for f_file in filtered_files:
            id_match = re.search(r"trap_(\d+)_", f_file.name, re.IGNORECASE)
            if not id_match: continue
            
            trap_id = int(id_match.group(1))
            
            # 1. Load Filtered Data
            prot_dict = self._csv_to_dict(f_file)

            # 2. Load Uptake Data
            # Try 02d format first (Trap_01), then single digit (Trap_1)
            u_file_name = f"Trap_{trap_id:02d}_Uptake_Data.csv"
            u_file = dir_uptake / u_file_name
            uptake_dict = {}
            
            if u_file.exists():
                uptake_dict = self._csv_to_dict(u_file, comment='#')
            else:
                u_file_alt = dir_uptake / f"Trap_{trap_id}_Uptake_Data.csv"
                if u_file_alt.exists():
                     uptake_dict = self._csv_to_dict(u_file_alt, comment='#')

            # FIX #2: Each trap gets its OWN copy of ExperimentMetadata.
            # Without this, all traps in the same experiment share a single
            # metadata object by reference.  If you later add per-trap
            # metadata (e.g. a per-trap f* or quality flag), modifying one
            # trap's metadata would accidentally change it for all traps.
            trap_data = TrapData(
                trap_id=trap_id,
                metadata=copy(meta),
                protrusion_data=prot_dict,
                uptake_data=uptake_dict,
            )
            loaded_traps.append(trap_data)

        return loaded_traps

    def _csv_to_dict(self, file_path: Path, comment: str = None) -> Dict[str, np.ndarray]:
        """
        Reads a CSV into a dict of {column_name: numpy_array}.

        FIX #1: Uses pd.to_numeric(..., errors='coerce') instead of a bare
        dtype=np.float64 cast. If a column contains non-numeric data (string
        labels, "NA" text, etc.), those values become NaN instead of crashing
        the whole pipeline with a ValueError.
        """
        try:
            df = pd.read_csv(file_path, comment=comment)
            return {
                col: pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
                for col in df.columns
            }
        except Exception as e:
            logger.error(f"Error loading CSV {file_path.name}: {e}")
            return {}

def extract_all_scalars(grouped_data: Dict[Tuple[str, str, int, int, float], List[TrapData]]) -> pd.DataFrame:
    """
    Extracts maximum values from time-series arrays to compute scalar metrics
    for Spearman correlation.

    NOTE: np.max is applied to every non-time numeric column. This is
    meaningful for monotonic signals (protrusion length, cumulative uptake)
    but may not suit columns where the peak isn't the right summary (e.g.
    velocity, normalized ratios). Review the resulting column list before
    interpreting the correlation matrix.
    """
    rows = []
    for (cell_type, treatment, press, volt, dur), traps in grouped_data.items():
        for trap in traps:
            row_data = {
                'Condition': f"{cell_type}_{treatment}_{press}Pa_{volt}V_{dur}ms",
                'Condition_Type': trap.metadata.condition_type,  # "EP" or "ASP"
                'Date': trap.metadata.date,
                'Cell_Type': cell_type,
                'Treatment': treatment,
                'Experiment_Number': trap.metadata.experiment_number,
                'Pressure_Pa': press,
                'Voltage_V': volt,
                'Duration_ms': dur,
                'Trap_ID': trap.trap_id
            }
            
            # Extract max values for protrusion data
            if hasattr(trap, 'protrusion_data') and trap.protrusion_data:
                for k, v in trap.protrusion_data.items():
                    if 'time' not in k.lower() and isinstance(v, np.ndarray) and len(v) > 0:
                        row_data[f"Prot_{k}"] = np.max(v)
                        
            # Extract max values for uptake data
            if hasattr(trap, 'uptake_data') and trap.uptake_data:
                for k, v in trap.uptake_data.items():
                    if 'time' not in k.lower() and isinstance(v, np.ndarray) and len(v) > 0:
                        row_data[f"Uptake_{k}"] = np.max(v)
                        
            rows.append(row_data)

    df = pd.DataFrame(rows)
    n_total = len(df)
    n_numeric = df.select_dtypes(include=[np.number]).dropna(how='all').shape[0]
    logger.info(
        f"extract_all_scalars: {n_total} traps total, "
        f"{n_numeric} with at least one numeric column populated."
    )
    return df

def get_group_stats(grouped_data: Dict[Tuple[str, str, int, int, float], List[TrapData]]):
    summary = []
    # Sort keys: cell type → treatment → pressure → voltage → duration
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