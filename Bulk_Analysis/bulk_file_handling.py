# -*- coding: utf-8 -*-
"""
File Handling for Bulk MFAE Analysis.
UPDATED: 
- Supports extracting Pressure (Pa) directly from the folder name.
- Groups data by (Pressure, Voltage, Duration).
- Extracts max scalar values for Spearman correlation mapping.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
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
    experiment_id: str
    pressure: int       # e.g., 1100 (from 1100Pa)
    voltage: int        # e.g., 100
    duration: float     # NORMALIZED TO MS (for sorting/physics)
    duration_label: str # ORIGINAL STRING (e.g., "5us", "10s") for plotting
    pulse_frame: int    # e.g., 10
    full_path: Path

@dataclass
class TrapData:
    """Holds all data arrays for a single trap."""
    trap_id: int
    metadata: ExperimentMetadata
    
    # Data containers
    protrusion_data: Dict[str, np.ndarray] # From "Filtered" CSV
    uptake_data: Dict[str, np.ndarray]     # From "Uptake" CSV
    
    # The total duration of the experiment (from "Full" CSV)
    full_experiment_duration: float = 0.0

    def __repr__(self):
        return (f"TrapData(ID={self.trap_id}, "
                f"Cond={self.metadata.voltage}V_{self.metadata.duration_label})")

# =============================================================================
# FILE LOADER CLASS
# =============================================================================

class BulkDataLoader:
    """
    Scans, parses, and loads MFAE experiment data.
    """
    
    # EXPECTED PATTERN:
    # 1. Separators: [-_] matches hyphen OR underscore.
    # 2. Units: (ms|us|µs|s) matches s, ms, or us.
    # 3. Expects a pressure value with "Pa"
    # Example: "YYMMDD_ExpID-1100Pa-100V-5s-frame10"
    FOLDER_PATTERN = re.compile(r"(\d{6})_(.+)[-_](\d+)Pa[-_](\d+)V[-_](\d+)(ms|us|µs|s)[-_]frame(\d+)", re.IGNORECASE)

    def __init__(self, root_output_dir: str):
        self.root_dir = Path(root_output_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        
        # Key: (Pressure_Pa, Voltage_V, Duration_Normalized_ms) -> Value: List[TrapData]
        self.grouped_data: Dict[Tuple[int, int, float], List[TrapData]] = {}

    def scan_and_load(self):
        """Main execution method to browse and load all data."""
        logger.info(f"Scanning root directory: {self.root_dir}")
        
        for entry in self.root_dir.iterdir():
            if entry.is_dir():
                metadata = self._parse_folder_name(entry.name, entry)
                if metadata:
                    logger.info(f"Found Experiment: {entry.name}")
                    logger.info(f"   -> Label: {metadata.pressure}Pa | {metadata.voltage}V {metadata.duration_label} (Norm: {metadata.duration}ms)")
                    
                    traps = self._load_experiment_traps(metadata)
                    
                    # Group by Pressure, Voltage, and Normalized Duration
                    group_key = (metadata.pressure, metadata.voltage, metadata.duration)
                    if group_key not in self.grouped_data:
                        self.grouped_data[group_key] = []
                    
                    self.grouped_data[group_key].extend(traps)
                    logger.info(f"   -> Loaded {len(traps)} traps.")

    def _parse_folder_name(self, folder_name: str, full_path: Path) -> Optional[ExperimentMetadata]:
        match = self.FOLDER_PATTERN.match(folder_name)
        if match:
            date_str = match.group(1)
            exp_id = match.group(2)
            pressure = int(match.group(3))
            volts = int(match.group(4))
            raw_val = float(match.group(5))
            unit = match.group(6).lower()
            frame = int(match.group(7))
            
            # 1. Create the Display Label (e.g. "5us")
            val_fmt = int(raw_val) if raw_val.is_integer() else raw_val
            label = f"{val_fmt}{unit}"

            # 2. Normalize to Milliseconds (for sorting)
            duration_ms = raw_val
            if unit == 's':
                duration_ms = raw_val * 1000.0
            elif unit in ['us', 'µs']:
                duration_ms = raw_val / 1000.0
            
            return ExperimentMetadata(
                date=date_str,
                experiment_id=exp_id,
                pressure=pressure,
                voltage=volts,
                duration=duration_ms,
                duration_label=label,
                pulse_frame=frame,
                full_path=full_path
            )
        return None

    def _load_experiment_traps(self, meta: ExperimentMetadata) -> List[TrapData]:
        loaded_traps = []

        dir_filtered = meta.full_path / "Filtered protrusion detection"
        dir_full = meta.full_path / "Full protrusion detection"
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

            # 3. Load FULL Data (for duration check)
            full_file_name = f"trap_{trap_id:02d}_detection_full.csv"
            full_file = dir_full / full_file_name
            full_duration = 0.0
            if full_file.exists():
                full_dict = self._csv_to_dict(full_file)
                if 'Time_s' in full_dict and len(full_dict['Time_s']) > 0:
                    full_duration = np.max(full_dict['Time_s'])
            else:
                full_file_alt = dir_full / f"trap_{trap_id}_detection_full.csv"
                if full_file_alt.exists():
                    full_dict = self._csv_to_dict(full_file_alt)
                    if 'Time_s' in full_dict and len(full_dict['Time_s']) > 0:
                        full_duration = np.max(full_dict['Time_s'])

            trap_data = TrapData(
                trap_id=trap_id,
                metadata=meta,
                protrusion_data=prot_dict,
                uptake_data=uptake_dict,
                full_experiment_duration=full_duration
            )
            loaded_traps.append(trap_data)

        return loaded_traps

    def _csv_to_dict(self, file_path: Path, comment: str = None) -> Dict[str, np.ndarray]:
        try:
            df = pd.read_csv(file_path, comment=comment)
            return {col: df[col].to_numpy() for col in df.columns}
        except Exception as e:
            logger.error(f"Error loading CSV {file_path.name}: {e}")
            return {}

def extract_all_scalars(grouped_data: Dict[Tuple[int, int, float], List[TrapData]]) -> pd.DataFrame:
    """Extracts maximum values from time-series arrays to compute scalar metrics for correlation."""
    rows = []
    for (press, volt, dur), traps in grouped_data.items():
        for trap in traps:
            row_data = {
                'Condition': f"{press}Pa_{volt}V_{dur}",
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
            
    return pd.DataFrame(rows)

def get_group_stats(grouped_data: Dict[Tuple[int, int, float], List[TrapData]]):
    summary = []
    # Sort keys by Pressure, then Voltage, then Duration
    sorted_keys = sorted(grouped_data.keys())
    for (press, volt, dur) in sorted_keys:
        traps = grouped_data[(press, volt, dur)]
        if not traps: continue
        # Use the label from the first trap in the group
        label = traps[0].metadata.duration_label
        summary.append(f"Condition [{press}Pa | {volt}V {label}]: {len(traps)} traps found.")
    return "\n".join(summary)