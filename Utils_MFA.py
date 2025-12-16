# -*- coding: utf-8 -*-
"""
A collection of shared utility functions for the MFA analysis pipeline.

ROLE IN PIPELINE:
This is the "toolbox" module. It contains generic functions used by multiple
parts of the pipeline to avoid code duplication.

KEY COMPONENTS:
1.  Logging: Sets up the system to record errors and progress to a file.
2.  Frame Caching: A "Least Recently Used" (LRU) cache that keeps only the 
    most recent ~50 images in RAM, allowing smooth scrolling without filling memory.
3.  Plot Styling: Global settings for Matplotlib to ensure publication-ready plots.
"""
import gc
import os
import time
import logging
import sys
import tkinter as tk
from typing import List, Dict, Any, Tuple, Optional, Union, TYPE_CHECKING
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from FileHandling_MFA import FileRead

logger = logging.getLogger(__name__)

# =============================================================================
# 1. LOGGING CONFIGURATION
# =============================================================================

def setup_logging(debug_mode: bool = False, log_file: Path = Path('mfa_analysis.log')) -> None:
    """
    Configures the Python logging system.
    - Console: Shows INFO (progress) or DEBUG (detailed) messages.
    - File: Always records DEBUG messages for post-mortem analysis.
    """
    level = logging.DEBUG if debug_mode else logging.INFO
    
    # Base configuration
    root_logger = logging.getLogger()
    
    # Check if handlers already exist to prevent duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(logging.DEBUG) # Set root to DEBUG to capture all levels
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level) # User-facing level
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG) # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        root_logger.warning(f"Could not open log file {log_file}: {e}. Logging to console only.")

    # Suppress overly verbose logs from third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

# =============================================================================
# 2. PLOTTING AND STYLING UTILITIES
# =============================================================================

# --- A base color constant used by plotting functions ---
DATA_POINTS_COLOR = '#2c2c2c'

# Centralized color constant for on-screen text
TEXT_COLOR = (255, 255, 255) # BGR for White

BLUE_RED_COLORS = {
    'data_points': DATA_POINTS_COLOR, 'model_fit': '#394B9A', 'fitting_data': '#394B9A',
    'model_fit_alt': '#DD3D2D', 'excluded_data': '#DD3D2D', 'rupture_point': '#A50026',
    'pre_rupture_region': '#98CAE1', 'post_rupture_region': '#FEDA8B'
}
PURPLE_GREEN_COLORS = {
    'data_points': DATA_POINTS_COLOR, 'model_fit': '#1B7837', 'fitting_data': '#1B7837',
    'model_fit_alt': '#762A83', 'excluded_data': '#762A83', 'rupture_point': '#A50026',
    'pre_rupture_region': '#ACD39E', 'post_rupture_region': '#C2A5CF'
}

def get_color_scheme(scheme_name: str = 'blue_red') -> Dict[str, str]:
    """Retrieves a dictionary containing a colorblind-friendly color scheme."""
    if scheme_name.lower() in ['blue_red', 'br']:
        return BLUE_RED_COLORS
    elif scheme_name.lower() in ['purple_green', 'pg']:
        return PURPLE_GREEN_COLORS
    logger.warning(f"Unknown color scheme '{scheme_name}', defaulting to 'blue_red'.")
    return BLUE_RED_COLORS

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to a BGR tuple."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0]) # Convert RGB to BGR

def set_paper_style(base_fontsize: int = 10, dpi: int = 300) -> None:
    """Applies global Matplotlib styles for publication-quality plots."""
    mpl.rcParams.update({
        'font.size': base_fontsize, 'font.family': 'sans-serif',
        'axes.labelsize': base_fontsize + 2, 'axes.titlesize': base_fontsize + 4,
        'legend.fontsize': base_fontsize, 'xtick.labelsize': base_fontsize,
        'ytick.labelsize': base_fontsize, 'lines.linewidth': 2.0,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': dpi
    })

def save_plot_png(save_path: Union[str, Path], dpi: int = 300) -> None:
    """Saves the current Matplotlib figure to a high-resolution PNG file."""
    save_path_png = Path(save_path).with_suffix(".png")
    save_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path_png), dpi=dpi, bbox_inches='tight', format='png')
    logger.info(f"Plot saved to: {save_path_png.name}")

# =============================================================================
# 3. IMAGE PROCESSING AND DISPLAY UTILITIES
# =============================================================================

def normalize_to_8bit(image: np.ndarray) -> np.ndarray:
    """Normalizes an image of any bit depth to the standard 8-bit (0-255) range."""
    if image is None:
        logger.warning("normalize_to_8bit received a None image. Returning black square.")
        return np.zeros((100, 100), dtype=np.uint8)
    if image.dtype == np.uint8:
        return image.copy()
    try:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    except cv2.error as e:
        logger.warning(f"Failed to normalize image. Returning as is. Error: {e}")
        if image.ndim > 2: # Handle potential color images
             return image[..., 0].astype(np.uint8)
        return image.astype(np.uint8)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image by a specified angle around its center."""
    if image is None:
        logger.error("Attempted to rotate a None image.")
        return np.zeros((100, 100), dtype=np.uint8) # Return a dummy image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def add_text_overlay(img: np.ndarray, text_lines: List[str], bottom_margin: int = 30) -> None:
    """Adds multiple lines of text to an image's bottom-left corner."""
    h = img.shape[0]
    # Iterate through lines in reverse to draw from the bottom up
    for i, text in enumerate(reversed(text_lines)):
        y_pos = h - bottom_margin - (i * 30)
        cv2.putText(img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

def prepare_display_image(image: np.ndarray, window_width: int, window_height: int) -> np.ndarray:
    """Prepares an image for display by resizing and centering it on a black canvas."""
    image_8bit = normalize_to_8bit(image)
    if len(image_8bit.shape) == 2:
        image_8bit = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)

    h, w = image_8bit.shape[:2]
    scale = min((window_width * 0.95) / w, (window_height * 0.95) / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_8bit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    y_offset = (window_height - new_h) // 2
    x_offset = (window_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

# =============================================================================
# 4. INTERACTIVE WINDOW UTILITIES
# =============================================================================

def get_screen_dimensions() -> Tuple[int, int]:
    """Retrieves the primary screen's width and height in pixels.
    
    Uses Tkinter to get screen size. Falls back to a default if Tkinter fails.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return screen_w, screen_h
    except (ImportError, tk.TclError):
        logger.warning("Tkinter not available. Using default screen size of 1920x1080.")
        return 1920, 1080

def create_centered_window(window_name: str, width: int, height: int) -> None:
    """Creates and centers a resizable OpenCV window on the screen."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    screen_w, screen_h = get_screen_dimensions()
    pos_x = max(0, (screen_w - width) // 2)
    pos_y = max(0, (screen_h - height) // 2)
    cv2.moveWindow(window_name, pos_x, pos_y)

# =============================================================================
# 5. DATA EXPORT UTILITIES
# =============================================================================

def save_protrusion_and_fit_csv(trap_index: int, time_points: np.ndarray, protrusions: np.ndarray, E: float, eta1: float, eta2: float, r_squared: float, output_dir: Union[str, Path]) -> None:
    """Saves time-series protrusion data and Jeffreys model parameters to a CSV."""
    df = pd.DataFrame({
        'Time_s': time_points,
        f'Trap_{trap_index+1}_Protrusion_um': protrusions
    })
    # Add fit parameters as single-value columns that repeat
    df['E_Pa'] = E
    df['eta1_Pa_s'] = eta1
    df['eta2_Pa_s'] = eta2
    df['R_squared'] = r_squared
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_filename = f"trap_{trap_index+1:02d}_timeseries_and_fit.csv"
    csv_path = output_path / csv_filename
    df.to_csv(csv_path, index=False)
    logger.info(f"   Detailed CSV saved: {csv_path.name}")

# =============================================================================
# 6. WORKFLOW AND DATA LOADING UTILITIES
# =============================================================================

class ProgressTracker:
    """A simple class to track and display progress through trap processing."""
    def __init__(self, total_traps: int) -> None:
        self.total_traps = total_traps
        self.start_time = time.time()
        self.trap_times: List[float] = []
        self.processed_count = 0
        self.trap_start_time = time.time() # Initialize

    def start_trap(self, trap_index: int) -> None:
        """Marks the start of processing for a new trap."""
        self.trap_start_time = time.time()
        elapsed = self.trap_start_time - self.start_time
        # Use trap_index for total, but processed_count for ETA
        logger.info(f"\n--- Processing Trap {trap_index+1}/{self.total_traps} (Elapsed: {elapsed:.1f}s) ---")

    def finish_trap(self, trap_index: int, skipped: bool = False) -> None:
        """Marks the end of processing for a trap and updates ETA."""
        if not skipped:
            trap_time = time.time() - self.trap_start_time
            self.trap_times.append(trap_time)
            self.processed_count += 1
            avg_time = np.mean(self.trap_times)
            remaining_traps = self.total_traps - (trap_index + 1)
            eta = remaining_traps * avg_time
            logger.info(f"Trap #{trap_index+1} done in {trap_time:.1f}s (ETA: {eta:.1f}s)")
        else:
            logger.warning(f"Trap #{trap_index+1} skipped.")

    def print_final_summary(self) -> None:
        """Prints a final summary of the total processing time."""
        total_time = time.time() - self.start_time
        logger.info("\n=== ANALYSIS COMPLETE ===")
        logger.info(f"Total time: {total_time:.1f}s | Traps processed: {self.processed_count}/{self.total_traps}")

class MemoryEfficientFrameLoader:
    """
    Caches a limited number of image frames in memory to reduce I/O.
    Crucial for the interactive setup phase where the user scrolls through frames.
    """
    def __init__(self, file_reader: 'FileRead', max_cache_size: int = 50) -> None:
        self.file_reader = file_reader
        self.max_cache_size = max_cache_size
        self.cache: Dict[int, np.ndarray] = {}
        self.order: List[int] = []

    def get_frame(self, idx: int) -> np.ndarray:
        """Retrieves a frame by index, checking the cache first."""
        # 1. Hit: Return cached image
        if idx in self.cache:
            return self.cache[idx]
        
        # 2. Miss: Load from disk
        img = self.file_reader.read_img(self.file_reader.tif_files[idx])
        if img is None:
            logger.error(f"Failed to read image at index {idx}: {self.file_reader.tif_files[idx]}")
            # Return a blank image to prevent crashes
            return np.zeros((100, 100), dtype=np.uint8) 
        
        # 3. Update Cache (Evict oldest if full)
        self.cache[idx] = img
        self.order.append(idx)
        
        if len(self.order) > self.max_cache_size:
            oldest_idx = self.order.pop(0)
            del self.cache[oldest_idx]
            
        return img

    def clear_cache(self) -> None:
        """Manually dumps the cache to free RAM.
        
        Note: gc.collect() is included here because this cache holds large
        image arrays (often 100+ MB per frame). Explicit collection ensures
        memory is released immediately rather than waiting for Python's
        automatic GC, which is important during the interactive setup phase
        where users may scroll through hundreds of frames rapidly.
        """
        self.cache.clear()
        self.order.clear()
        gc.collect()