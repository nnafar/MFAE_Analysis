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
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import tifffile

from mfae.imaging.trap_rois import CropImage

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
        membrane_candidates = [f for f in all_files if mem_pattern in f.name]
        
        if membrane_candidates:
            self.tif_files = self.sort_by_time_index(membrane_candidates)
            logger.info(f"Membrane Channel: Found {len(self.tif_files)} files matching '{mem_pattern}'.")
        else:
            logger.info(f"No files matched pattern '{mem_pattern}'. Loading all found .tif files as Membrane channel.")
            self.tif_files = self.sort_by_time_index(all_files)

        # Reject black frames (e.g., shutter artifacts at the end of the acquisition sequence)
        self.tif_files = self._reject_black_frames(self.tif_files)

        # --- 2. FILTER DYE FILES (OPTIONAL) ---
        if enable_dye:
            self.dye_files = [f for f in all_files if dye_pattern in f.name]
            self.dye_files = self.sort_by_time_index(self.dye_files)
            self.dye_files = self._reject_black_frames(self.dye_files)
            
            if not self.dye_files:
                logger.warning(f"Dye analysis enabled but no files found matching '{dye_pattern}'. Check config.")
            else:
                logger.info(f"Dye Channel: Found {len(self.dye_files)} files matching '{dye_pattern}'.")
                
        # --- 3. FILTER ACTIN FILES ---
        actin_pattern = self.params.get('actin_parameters', {}).get('channel_pattern', 'C3')
        enable_actin = self.params.get('actin_parameters', {}).get('enable', False)
        self.actin_files = []

        if enable_actin and actin_pattern == mem_pattern and membrane_candidates:
            # Actin is imaged in the membrane channel: same files, no second scan.
            self.actin_files = list(self.tif_files)
        elif enable_actin:
            self.actin_files = [f for f in all_files if actin_pattern in f.name]
            self.actin_files = self.sort_by_time_index(self.actin_files)
            self.actin_files = self._reject_black_frames(self.actin_files)
            
        # --- SYNCHRONIZATION ---
        # Find minimum length across all active channels
        lengths = [len(self.tif_files)]
        if self.dye_files: lengths.append(len(self.dye_files))
        if self.actin_files: lengths.append(len(self.actin_files))
        
        min_len = min(lengths)
        
        if len(self.tif_files) > min_len:
            self.tif_files = self.tif_files[:min_len]
        if self.dye_files and len(self.dye_files) > min_len:
            self.dye_files = self.dye_files[:min_len]
        if self.actin_files and len(self.actin_files) > min_len:
            self.actin_files = self.actin_files[:min_len] # Sync Actin
        
        # Extract timestamps (using the primary membrane files)
        self.time_data = self.extract_timestamps_from_metadata()
        
        if self.time_data is None:
            logger.warning("Metadata extraction failed. Using manual frame interval from config.")
            self.time_data = self.create_manual_timestamps()
        else:
            avg_dt = 0
            if len(self.time_data) > 1:
                avg_dt = (self.time_data[-1] - self.time_data[0]) / (len(self.time_data) - 1)
            
            logger.info(f"Time extraction successful | Method: {self.metadata_method_used}")
            logger.info(f"Detected Frame Interval: {avg_dt:.4f} s")
            
        self.num_files = len(self.tif_files)
        
        constant_pressure = self.params.get('constant_pressure', 1100)
        self.pressure_data = [constant_pressure] * self.num_files
        self.frames = np.arange(0, self.num_files)
        
        self.results['frames'] = self.frames
        self.results['time_data'] = self.time_data
        
        logger.info(f"Successfully loaded {self.num_files} frames for analysis.")

    def _reject_black_frames(self, file_list: List[Path], threshold: float = 1.0) -> List[Path]:
        """
        Checks the sequence (specifically the last frame) for 'black' (empty) images 
        and removes them to avoid analysis errors. This handles hardware timing discrepancies 
        resulting in blank acquisitions.
        """
        if not file_list: return file_list
        
        # Check the last frame specifically (common artifact)
        last_file = file_list[-1]
        try:
            img = self.read_img(last_file)
            if img is not None:
                mean_val = np.mean(img)
                if mean_val < threshold:
                    logger.warning(f"Rejected black frame (End of Stack): {last_file.name} (Mean: {mean_val:.2f})")
                    return file_list[:-1] # Drop last
        except Exception as e:
            logger.warning(f"Could not check last frame: {e}")
            
        return file_list

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
        """
        Extracts timestamps from ImageJ metadata tags.
        
        The frame interval is a single value stored in the first file's header —
        there is no reason to open all 76 files. We read only file[0] to find
        frame_interval, then compute timestamps arithmetically.
        """
        logger.debug("Trying ImageJ metadata extraction...")
        if not self.tif_files:
            return None

        frame_interval = None
        try:
            with tifffile.TiffFile(self.tif_files[0]) as tif:
                if tif.is_imagej and tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                    for key in ['finterval', 'frame_interval', 'spacing']:
                        if key in metadata:
                            frame_interval = float(metadata[key])
                            logger.debug(f"      Found frame interval: {frame_interval} seconds")
                            break
        except Exception:
            return None

        if frame_interval is None:
            return None

        self.metadata_method_used = "ImageJ metadata"
        return [i * frame_interval for i in range(len(self.tif_files))]

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
        """
        Extracts timing information from generic TIFF tags.
        
        Same principle as _extract_from_imagej_metadata: the interval is
        a single constant, so only the first file needs to be opened.
        """
        logger.debug("Trying TIFF tag extraction...")
        if not self.tif_files:
            return None

        frame_interval = None
        try:
            with tifffile.TiffFile(self.tif_files[0]) as tif:
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
        except Exception:
            return None

        if frame_interval is None:
            return None

        self.metadata_method_used = "TIFF tags"
        return [i * frame_interval for i in range(len(self.tif_files))]

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
        frame_interval = self.params.get('experiment_parameters', {}).get('frame_interval', 0.2)
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

# =============================================================================
# ZEISS CZI FILES
# =============================================================================

@dataclass(frozen=True)
class CziFrame:
    """
    One image plane of a CZI file. Stands in for a TIFF file path in the
    frame lists (tif_files / dye_files / actin_files), so the rest of the
    pipeline doesn't need to know which format it is reading.
    """
    path: Path
    channel: int        # 0-based channel index in the file
    t: int              # 0-based time index

    @property
    def frame_number(self) -> int:
        """1-based frame number, as Zen and Fiji show it (and as `frame<N>` counts)."""
        return self.t + 1

    @property
    def name(self) -> str:
        return f"{self.path.name}[channel {self.channel + 1}, frame {self.frame_number}]"

    def __str__(self) -> str:
        return f"{self.path}[C{self.channel + 1},T{self.frame_number}]"


def resolve_czi_file(data_folder: Union[str, Path], experiment_id: Optional[str] = None) -> Path:
    """
    The .czi file to analyse:
      - data_folder itself, if it is a .czi file;
      - otherwise the .czi file in that folder named after experiment_id
        (so one folder can hold the files of many experiments);
      - otherwise the only .czi file in the folder.
    """
    data_folder = Path(data_folder)
    if data_folder.is_file():
        return data_folder
    found = sorted(p for p in data_folder.iterdir() if p.suffix.lower() == '.czi')
    if not found:
        raise FileNotFoundError(f"No .czi file found in {data_folder}")
    if experiment_id:
        named = [p for p in found if p.stem.lower() == str(experiment_id).lower()]
        if named:
            return named[0]
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"{len(found)} .czi files in {data_folder} and none is named after the experiment "
        f"('{experiment_id}.czi'). Set experiment_id to one of these file names (without .czi), "
        f"or data_folder to the file itself: " + ", ".join(p.name for p in found))


def valid_timestamps(values, n_frames: int) -> Optional[List[float]]:
    """
    Frame times relative to the first frame, or None if they can't be trusted
    (wrong count, not strictly increasing, or not finite).
    """
    try:
        times = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(times) < n_frames or n_frames == 0:
        return None
    times = times[:n_frames]
    if not all(np.isfinite(times)) or any(b <= a for a, b in zip(times, times[1:])):
        return None
    return [t - times[0] for t in times]


class CziRead(FileRead):
    """
    Reads a Zeiss .czi time-lapse with the same interface as FileRead.

    Channels are chosen by number (counted from 1, as Fiji shows them):
    dye_uptake_parameters.membrane_channel_number / dye_channel_number and
    actin_parameters.channel_number. Frame times come from the file's
    TimeStamps attachment when present and plausible; otherwise the config's
    frame_interval is used and the log says so.
    """

    def __init__(self, data_folder: Union[str, Path], params: Dict[str, Any]) -> None:
        super().__init__(data_folder, params)
        self.actin_files: List[CziFrame] = []
        self.czi_path: Optional[Path] = None
        self.channel_names: List[str] = []
        self.pixel_size_um: Optional[float] = None
        self._reader = None

    # -- file access ---------------------------------------------------------
    def _open(self):
        if self._reader is None:
            from pylibCZIrw import czi as pyczi   # only needed for CZI input
            # A persistent reader (open_czi is a context manager for one-off use).
            self._reader = pyczi.CziReader(str(self.czi_path))
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_reader'] = None     # file handles don't cross process boundaries
        return state

    def read_img(self, frame: CziFrame) -> np.ndarray:
        img = self._open().read(plane={"T": frame.t, "C": frame.channel, "Z": 0})
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        return img

    # -- loading -------------------------------------------------------------
    def run(self) -> None:
        logger.info("Reading CZI file...")
        self.czi_path = resolve_czi_file(
            self.data_folder, self.params.get('paths', {}).get('experiment_id'))
        reader = self._open()
        box = reader.total_bounding_box
        n_t = box.get('T', (0, 1))[1]
        n_c = box.get('C', (0, 1))[1]
        n_z = box.get('Z', (0, 1))[1]
        if n_z > 1:
            raise ValueError(f"{self.czi_path.name} is a z-stack ({n_z} planes); "
                             "only single-plane time-lapses are supported.")
        try:
            scenes = reader.scenes_bounding_rectangle
        except Exception:
            scenes = {}
        if len(scenes) > 1:
            raise ValueError(f"{self.czi_path.name} has {len(scenes)} scenes/positions; "
                             "only single-position recordings are supported.")
        self._read_metadata(reader)
        logger.info(f"CZI file: {self.czi_path.name} | {n_t} frames | {n_c} channels: "
                    + ", ".join(f"{i}={n!r}" for i, n in enumerate(self.channel_names, 1)))

        dye_p = self.params.get('dye_uptake_parameters', {})
        actin_p = self.params.get('actin_parameters', {})

        def frames(number: Optional[int], label: str) -> List[CziFrame]:
            if number is None:
                return []
            if not 1 <= number <= n_c:
                raise ValueError(f"{label} channel number {number} doesn't exist: "
                                 f"{self.czi_path.name} has channels 1-{n_c}.")
            return [CziFrame(self.czi_path, number - 1, t) for t in range(n_t)]

        self.tif_files = self._reject_black_frames(
            frames(dye_p.get('membrane_channel_number'), 'Membrane'))
        if dye_p.get('enable', False):
            self.dye_files = self._reject_black_frames(
                frames(dye_p.get('dye_channel_number'), 'Dye'))
        if actin_p.get('enable', False):
            if actin_p.get('channel_number') == dye_p.get('membrane_channel_number'):
                self.actin_files = list(self.tif_files)
            else:
                self.actin_files = self._reject_black_frames(
                    frames(actin_p.get('channel_number'), 'Actin'))

        # Same synchronisation as for TIFFs: trim every channel to the shortest.
        lengths = [len(l) for l in (self.tif_files, self.dye_files, self.actin_files) if l]
        n = min(lengths) if lengths else 0
        self.tif_files, self.dye_files, self.actin_files = (
            l[:n] for l in (self.tif_files, self.dye_files, self.actin_files))

        times = valid_timestamps(self._timestamps() or [], n)
        if times is not None:
            self.time_data = times
            self.metadata_method_used = "CZI timestamps"
        else:
            logger.warning("No usable frame timestamps in the CZI file. "
                           "Using frame_interval from the config.")
            self.time_data = self.create_manual_timestamps()

        self.num_files = n
        constant_pressure = self.params.get('constant_pressure', 1100)
        self.pressure_data = [constant_pressure] * n
        self.frames = np.arange(0, n)
        self.results['frames'] = self.frames
        self.results['time_data'] = self.time_data
        logger.info(f"Successfully loaded {n} frames for analysis ({self.metadata_method_used}).")

    def _read_metadata(self, reader) -> None:
        try:
            info = reader.metadata["ImageDocument"]["Metadata"]
        except Exception:
            return
        channels = (info.get("Information", {}).get("Image", {}).get("Dimensions", {})
                    .get("Channels", {}).get("Channel", []))
        if isinstance(channels, dict):
            channels = [channels]
        self.channel_names = [str(c.get("@Name") or c.get("Fluor") or "") for c in channels]
        distances = info.get("Scaling", {}).get("Items", {}).get("Distance", [])
        if isinstance(distances, dict):
            distances = [distances]
        for d in distances:
            if d.get("@Id") == "X" and d.get("Value"):
                self.pixel_size_um = float(d["Value"]) * 1e6   # stored in metres

    def _timestamps(self) -> Optional[List[float]]:
        """Raw frame times (seconds) from the TimeStamps attachment, if readable."""
        try:
            import czifile
        except ImportError:
            logger.warning("czifile is not installed, so CZI frame timestamps can't be read "
                           "(pip install czifile).")
            return None
        try:
            with czifile.CziFile(str(self.czi_path)) as czi:
                for entry in czi.attachments():
                    if entry.attachment_entry.name == "TimeStamps":
                        return list(np.asarray(entry.data(), dtype=float).ravel())
        except Exception as e:
            logger.warning(f"Could not read CZI timestamps: {e}")
        return None


def make_file_reader(data_folder: Union[str, Path], params: Dict[str, Any]) -> FileRead:
    """The reader for paths.image_format ('tiff' -> FileRead, 'czi' -> CziRead)."""
    image_format = params.get('paths', {}).get('image_format', 'tiff')
    return CziRead(data_folder, params) if image_format == 'czi' else FileRead(data_folder, params)


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
    


class FileHandlingMFA:
    """
    Orchestrates file reading and the interactive setup wizard.
    Acts as a facade for FileRead and CropImage.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.read_params = {
            **config.get('experiment_parameters', {}),
            'dye_uptake_parameters': config.get('dye_uptake_parameters', {})
        }
        
        self.reader = FileRead(config['paths']['data_folder'], self.read_params)
        self.saver = FileSave(config['paths']['data_folder']) 

    def run_setup_wizard(self) -> Tuple[bool, Dict[str, Any]]:
        """
        1. Loads files.
        2. Launches GUI for trap selection.
        Returns: (success_flag, dictionary_of_results)
        """
        # 1. Load Files
        try:
            self.reader.run()
        except Exception as e:
            logger.error(f"Failed to load files: {e}")
            return False, {}

        # 2. Run Interactive Setup (CropImage)
        if self.config['workflow_settings'].get('verify_traps_interactively', True):
            gui_params = {
                **self.config.get('experiment_parameters', {}),
                **self.config.get('workflow_settings', {}),
                **self.config.get('image_processing', {})
            }
            
            processor = CropImage(gui_params)
            status = processor.run(self.reader)
            
            if status != 'confirm':
                logger.warning("Interactive setup cancelled.")
                return False, {}
                
            return True, {
                'rotation_angle': processor.rotation_angle,
                'roi_coords': processor.roi_coords,
                'trap_rois': processor.all_trap_rois,
                'pipette_coords': processor.pipette_coords,
                'selected_traps': processor.selected_traps, # <--- ADDED: Pass selection back
                'image_processor': processor
            }
        else:
            logger.error("Non-interactive mode is not currently supported in this version.")
            return False, {}