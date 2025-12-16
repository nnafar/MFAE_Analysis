# -*- coding: utf-8 -*-
"""
Handles file I/O and timestamp extraction for MFA analysis.

ROLE IN PIPELINE:
This module manages reading raw images and saving results. It isolates
platform-specific file issues (like path separators) and metadata formats.

KEY FEATURES:
1.  Robust Timestamping: Attempts 4 different methods to find the *real* time of each frame.
2.  Natural Sorting: Ensures frame '10' comes after '9', not '1'.
3.  Structured Output: Creates timestamped folders for every run.
"""
import os
import re
import logging
import pickle
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import tifffile

logger = logging.getLogger(__name__)


class FileRead():
    """
    Reads image files and extracts timestamps from metadata.
    """

    _FILENAME_PATTERNS = {
        "YYYY-MM-DD_HH-MM-SS": r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',
        "YYYYMMDD_HHMMSS": r'(\d{8}_\d{6})',
        "milliseconds_t": r'_t(\d+)ms',
        "seconds_s": r'_(\d+)s_',
    }

    def __init__(self, data_folder: Union[str, Path], params: Dict[str, Any]) -> None:
        """Initializes the file reader for a specific data directory."""
        self.data_folder = Path(data_folder)
        self.params = params
        self.results: Dict[str, Any] = {}
        self.metadata_method_used: Optional[str] = None
        
        # Primary Channel (Membrane) used for detection
        self.tif_files: List[Path] = []
        # Secondary Channel (Dye) used for quantification
        self.dye_files: List[Path] = []
        
        self.time_data: Optional[List[float]] = None
        self.pressure_data: List[float] = []
        self.num_files: int = 0

    def run(self) -> None:
        """Executes the file loading and timestamp extraction process."""
        logger.info("Starting file reading with metadata extraction...")

        all_files = self.find_tif_filenames(self.data_folder)
        if not all_files:
            raise FileNotFoundError(f"No .tif files found in {self.data_folder}")
        
        # --- CHANNEL IDENTIFIERS ---
        # Defaults to "C1" and "C2" if not specified in config
        mem_pattern = self.params.get('dye_uptake_parameters', {}).get('membrane_channel_pattern', 'C1')
        dye_pattern = self.params.get('dye_uptake_parameters', {}).get('dye_channel_pattern', 'C2')
        enable_dye = self.params.get('dye_uptake_parameters', {}).get('enable', False)
        
        # --- 1. FILTER MEMBRANE FILES (STRICT) ---
        # We ALWAYS attempt to filter by the membrane pattern first.
        # This prevents mixing C1 and C2 files even if dye uptake is disabled.
        membrane_candidates = [f for f in all_files if mem_pattern in f.name]
        
        if membrane_candidates:
            self.tif_files = self.sort_by_time_index(membrane_candidates)
            logger.info(f"Membrane Channel: Found {len(self.tif_files)} files matching '{mem_pattern}'.")
        else:
            # Fallback: If no files match "C1", assume it's a single-channel experiment 
            # with simple filenames (e.g. "img_001.tif")
            logger.info(f"No files matched pattern '{mem_pattern}'. Loading all found .tif files as Membrane channel.")
            self.tif_files = self.sort_by_time_index(all_files)

        # --- 2. FILTER DYE FILES (OPTIONAL) ---
        if enable_dye:
            self.dye_files = [f for f in all_files if dye_pattern in f.name]
            self.dye_files = self.sort_by_time_index(self.dye_files)
            
            if not self.dye_files:
                logger.warning(f"Dye analysis enabled but no files found matching '{dye_pattern}'. Check config.")
            else:
                logger.info(f"Dye Channel: Found {len(self.dye_files)} files matching '{dye_pattern}'.")
                
            # Ensure synchronization
            if len(self.tif_files) != len(self.dye_files):
                logger.warning(f"Channel Count Mismatch! Membrane: {len(self.tif_files)}, Dye: {len(self.dye_files)}")
                min_len = min(len(self.tif_files), len(self.dye_files))
                if min_len > 0:
                    logger.warning(f"Truncating both channels to {min_len} frames to maintain sync.")
                    self.tif_files = self.tif_files[:min_len]
                    self.dye_files = self.dye_files[:min_len]
            
        # Extract timestamps (using the primary membrane files)
        self.time_data = self.extract_timestamps_from_metadata()
        
        if self.time_data is None:
            logger.warning("Metadata extraction failed. Using manual frame interval.")
            self.time_data = self.create_manual_timestamps()
            
        self.num_files = len(self.tif_files)
        
        # Assuming constant pressure
        constant_pressure = self.params.get('constant_pressure', 1100)
        self.pressure_data = [constant_pressure] * self.num_files
        self.frames = np.arange(0, self.num_files)
        
        self.results['frames'] = self.frames
        self.results['time_data'] = self.time_data
        
        logger.info(f"Successfully loaded {self.num_files} frames for analysis.")
    
    def extract_timestamps_from_metadata(self) -> Optional[List[float]]:
        """
        Cascading Strategy:
        1. Check ImageJ Tiff tags (Most reliable for scientific cams).
        2. Check Standard EXIF tags (Consumer cameras).
        3. Check Generic TIFF tags.
        4. Parse Filename (e.g. "img_0.02s.tif").
        """
        methods_to_try = [
            self._extract_from_imagej_metadata,
            self._extract_from_exif_data,
            self._extract_from_tiff_tags,
            self._extract_from_filename_timestamps
        ]

        for method in methods_to_try:
            try:
                timestamps = method()
                if timestamps is not None and len(timestamps) == len(self.tif_files):
                    return timestamps
            except Exception as e:
                logger.debug(f"   Method {method.__name__} failed: {e}")
                continue
        return None

    def _extract_from_imagej_metadata(self) -> Optional[List[float]]:
        """Extracts timestamps from ImageJ metadata tags."""
        logger.debug("Trying ImageJ metadata extraction...")
        timestamps = []
        frame_interval = None

        for i, filepath in enumerate(self.tif_files):
            try:
                with tifffile.TiffFile(filepath) as tif:
                    if tif.is_imagej and tif.imagej_metadata:
                        metadata = tif.imagej_metadata
                        if frame_interval is None:
                            for key in ['finterval', 'frame_interval', 'spacing']:
                                if key in metadata:
                                    frame_interval = float(metadata[key])
                                    logger.debug(f"      Found frame interval: {frame_interval} seconds")
                                    break
                        if frame_interval is not None:
                            timestamps.append(i * frame_interval)
                        else:
                            return None
                    else:
                        return None
            except Exception:
                return None

        if timestamps:
            self.metadata_method_used = "ImageJ metadata"
            return timestamps
        return None

    def _extract_from_exif_data(self) -> Optional[List[float]]:
        """Extracts timestamps from standard EXIF data tags."""
        logger.debug("Trying EXIF timestamp extraction...")
        timestamps = []
        first_timestamp = None

        for filepath in self.tif_files:
            try:
                with Image.open(filepath) as img:
                    exif_data = img.getexif()
                    if exif_data:
                        datetime_str = None
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                                datetime_str = str(value)
                                break

                        if datetime_str:
                            dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
                            if first_timestamp is None:
                                first_timestamp = dt
                                timestamps.append(0.0)
                            else:
                                time_diff = (dt - first_timestamp).total_seconds()
                                timestamps.append(time_diff)
                        else:
                            return None
                    else:
                        return None
            except Exception:
                return None

        if timestamps:
            self.metadata_method_used = "EXIF timestamps"
            return timestamps
        return None

    def _extract_from_tiff_tags(self) -> Optional[List[float]]:
        """Extracts timing information from generic TIFF tags."""
        logger.debug("Trying TIFF tag extraction...")
        timestamps = []
        frame_interval = None

        for i, filepath in enumerate(self.tif_files):
            try:
                with tifffile.TiffFile(filepath) as tif:
                    page = tif.pages[0]
                    for tag in page.tags:
                        tag_name = tag.name.lower()
                        if 'interval' in tag_name or 'time' in tag_name:
                            try:
                                interval = float(tag.value)
                                if interval > 0:
                                    frame_interval = interval
                                    logger.debug(f"      Found timing in tag {tag.name}: {frame_interval}")
                                    break
                            except (ValueError, TypeError):
                                continue
                    if frame_interval is not None:
                        timestamps.append(i * frame_interval)
                    else:
                        return None
            except Exception:
                return None

        if timestamps:
            self.metadata_method_used = "TIFF tags"
            return timestamps
        return None

    def _extract_from_filename_timestamps(self) -> Optional[List[float]]:
        """Extracts timestamps by parsing filename patterns."""
        logger.debug("Trying filename timestamp patterns...")

        for name, pattern in self._FILENAME_PATTERNS.items():
            try:
                time_values = []
                first_dt = None
                for filepath in self.tif_files:
                    filename = filepath.name
                    match = re.search(pattern, filename)
                    if match:
                        time_str = match.group(1)
                        if 'ms' in pattern:
                            time_values.append(float(time_str) / 1000.0)
                        elif 's' in pattern:
                            time_values.append(float(time_str))
                        else:
                            fmt = "%Y-%m-%d_%H-%M-%S" if '-' in time_str else "%Y%m%d_%H%M%S"
                            dt = datetime.strptime(time_str, fmt)
                            if first_dt is None:
                                first_dt = dt
                            time_values.append((dt - first_dt).total_seconds())
                    else:
                        time_values = [] 
                        break
                
                if len(time_values) == len(self.tif_files):
                    self.metadata_method_used = f"Filename pattern: '{name}'"
                    return time_values
            except Exception:
                continue
        return None

    def create_manual_timestamps(self) -> List[float]:
        """Creates timestamps using a fixed, manually specified interval."""
        frame_interval = self.params.get('frame_interval', 0.2)
        logger.info(f"Using manual frame interval: {frame_interval} seconds")
        self.metadata_method_used = f"Manual calculation ({frame_interval}s interval)"
        return [i * frame_interval for i in range(len(self.tif_files))]

    def find_tif_filenames(self, path_to_dir: Path) -> List[Path]:
        """Finds all TIFF files in a directory, avoiding duplicates on case-insensitive OS."""
        files = set()
        files.update(path_to_dir.glob('*.tif'))
        files.update(path_to_dir.glob('*.tiff'))
        files.update(path_to_dir.glob('*.TIF'))
        files.update(path_to_dir.glob('*.TIFF'))
        return sorted(list(files))

    def sort_by_time_index(self, file_list: List[Path]) -> List[Path]:
        """
        Sorts filenames by the numeric index after '_t'.
        Example: 'C1_Exp1_t2.tif' comes before 'C1_Exp1_t10.tif'.
        """
        def extract_time_index(filepath: Path) -> int:
            basename = filepath.name
            # Regex looks for '_t' followed by digits
            match = re.search(r'_t(\d+)', basename)
            if match:
                return int(match.group(1))
            
            # Fallback: look for just digits at the end of the filename
            match_generic = re.search(r'(\d+)\.(tif|tiff)$', basename, re.IGNORECASE)
            if match_generic:
                return int(match_generic.group(1))
                
            return 0 

        return sorted(file_list, key=extract_time_index)

    def read_img(self, filename: Path) -> np.ndarray:
        """Reads an image file using OpenCV."""
        return cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)


class FileSave():
    """Handles saving of analysis results."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Initializes the file saver with an output directory."""
        if output_dir is None:
            self.output_dir = self._create_default_output_dir()
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_default_output_dir(self) -> Path:
        """Creates a default output directory path with a timestamp."""
        today = self.get_date()
        default_name = f"MFA_Results_{today}"
        return Path.cwd() / default_name

    def name(self, experiment_id: str = "MFA", exp_type: str = "analysis") -> str:
        """Generates a standardized filename with a timestamp."""
        timestamp = self.get_date()
        return f"{experiment_id}_{exp_type}_{timestamp}"

    def save_obj(self, obj: Any, filename: str) -> Path:
        """Saves a Python object as a pickle file."""
        filepath = (self.output_dir / filename).with_suffix('.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved: {filepath.name}")
        return filepath

    def save_csv(self, data: Any, filename: str) -> Path:
        """Saves dictionary or DataFrame as a CSV file."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Data must be a dictionary or pandas DataFrame.")

        filepath = (self.output_dir / filename).with_suffix('.csv')
        df.to_csv(filepath, index=False)
        logger.info(f"CSV saved: {filepath.name}")
        return filepath

    @staticmethod
    def get_date() -> str:
        """Gets the current date as a string in YYMMDD format."""
        return date.today().strftime("%y%m%d")

    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """Constructs a full path within the output directory."""
        if filename is None:
            return self.output_dir
        return self.output_dir / filename