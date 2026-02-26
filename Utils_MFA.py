# -*- coding: utf-8 -*-
"""
A collection of shared utility functions for the MFA analysis pipeline.

ROLE IN PIPELINE:
This is the "toolbox" module. It contains generic functions used by multiple
parts of the pipeline to avoid code duplication.

KEY COMPONENTS:
1.  Logging: Sets up the system to record errors and progress to a file.
2.  Unified Styling: Centralized color palette and Matplotlib settings.
3.  Image Processing: Helper functions for rotation, normalization, and display.
4.  Interactive Windows: Tools for managing OpenCV GUI windows.
5.  Frame Caching: LRU cache for smooth scrolling in the setup phase.
6.  Worker Safety: Standalone functions for parallel processing.
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
    import Utils_MFA as utils

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

# --- Centralized Color Palette (Blue to Red Scheme) ---
# Extracted from user reference image 'Blue to Red'
MFA_COLORS = {
    # Reference Palette (Hex)
    'dark_blue':   '#1065AB', # R 016, G 101, B 171
    'medium_blue': '#3A93C3', # R 058, G 147, B 195
    'light_blue':  '#8EC4DE', # R 142, G 196, B 222
    'pale_blue':   '#D1E5F0', # R 209, G 229, B 240
    'light_grey':  '#E7E6E3', # R 231, G 230, B 227
    'white':       '#F9F9F9', # R 249, G 249, B 249
    'pale_red':    '#FDDBC7', # R 254, G 219, B 199
    'light_red':   '#F6A482', # R 246, G 164, B 130
    'medium_red':  '#D75F4C', # R 215, G 095, B 076
    'dark_red':    '#B31529', # R 179, G 021, B 041
    
    # UI Semantic Mapping
    'ui_guide':    '#3A93C3', # Medium Blue (ROI Box, Rotation Line)
    'ui_pipette':  '#1065AB', # Dark Blue   (Pipette Entrance Line)
    'ui_mask':     '#B31529', # Dark Red    (Threshold Mask)
    'ui_text':     '#F9F9F9', # White       (Overlay Text)

    # Plotting Defaults
    'primary':     '#000000', # Black (Raw Data Points / Main Trace)
    'secondary':   '#D75F4C', # Medium Red (Protrusion / Fits)
    'tertiary':    '#8EC4DE', # Light Blue (Cell Body / Mask Overlay)
    'quaternary':  '#B31529', # Dark Red (Rupture Haze / Intensity Trace)
    
    'pulse':       '#3A93C3', # Medium Blue (Vertical line for pulse)
    'rupture':     '#B31529', # Dark Red (Vertical line for rupture)
    
    'grid':        '#D1E5F0', # Pale Blue
    'background':  '#FFFFFF',
    
    'time_gradient_start': '#1065AB', 
    'time_gradient_end':   '#B31529'
}

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Converts hex string to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

# Pre-calculate BGR colors for OpenCV UI
UI_COLORS = {
    'guide':   hex_to_bgr(MFA_COLORS['ui_guide']),
    'pipette': hex_to_bgr(MFA_COLORS['ui_pipette']),
    'mask':    hex_to_bgr(MFA_COLORS['ui_mask']),
    'text':    hex_to_bgr(MFA_COLORS['ui_text']),
}

def get_ui_color(element_type: str) -> Tuple[int, int, int]:
    """Returns the standardized BGR color for a UI element."""
    return UI_COLORS.get(element_type, (255, 255, 255))

def get_bgr_color(color_name: str) -> Tuple[int, int, int]:
    """Retrieves a BGR tuple from the MFA_COLORS palette by name."""
    hex_val = MFA_COLORS.get(color_name, MFA_COLORS['white'])
    return hex_to_bgr(hex_val)

def set_paper_style(base_fontsize: int = 14, dpi: int = 300) -> None:
    """Applies publication-quality style to Matplotlib plots."""
    plt.rcdefaults() 
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': base_fontsize,
        'axes.labelsize': base_fontsize,
        'axes.titlesize': base_fontsize + 2,
        'axes.titleweight': 'bold',
        'legend.fontsize': base_fontsize - 2,
        'legend.frameon': True,
        'figure.dpi': dpi,
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.4,
        'grid.color': MFA_COLORS['grid'],
        'lines.linewidth': 2.5,
        'axes.prop_cycle': mpl.cycler(color=[
            MFA_COLORS['primary'],    # Black
            MFA_COLORS['secondary'],  # Medium Red
            MFA_COLORS['tertiary']    # Dark Blue
        ])
    })

def get_time_colormap(n_steps: int) -> List[Tuple[float, float, float, float]]:
    """Returns a list of colors forming the full Blue -> Red gradient."""
    # Define the full 9-color gradient from the user's palette
    colors = [
        MFA_COLORS['dark_blue'], MFA_COLORS['medium_blue'], MFA_COLORS['light_blue'],
        MFA_COLORS['pale_blue'], MFA_COLORS['light_grey'], MFA_COLORS['pale_red'],
        MFA_COLORS['light_red'], MFA_COLORS['medium_red'], MFA_COLORS['dark_red']
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mfa_full_gradient", colors)
    return [cmap(i / (n_steps - 1)) for i in range(n_steps)]

def get_mfa_continuous_cmap() -> mpl.colors.LinearSegmentedColormap:
    """Returns a continuous matplotlib colormap for Kymographs (Blue->White->Red)."""
    # Optimized for heatmaps: Dark Blue (Low) -> White (Mid) -> Dark Red (High)
    nodes = [0.0, 0.2, 0.5, 0.8, 1.0]
    colors = [
        '#000000', # Black (Background)
        MFA_COLORS['dark_blue'],
        MFA_COLORS['light_blue'],
        MFA_COLORS['medium_red'],
        MFA_COLORS['dark_red']
    ]
    return mpl.colors.LinearSegmentedColormap.from_list("mfa_kymo", list(zip(nodes, colors)))

def save_plot_png(save_path: Union[str, Path], dpi: int = 300) -> None:
    save_path_png = Path(save_path).with_suffix(".png")
    save_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path_png), dpi=dpi, bbox_inches='tight', format='png')
    logger.info(f"Plot saved: {save_path_png.name}")

# =============================================================================
# 3. IMAGE PROCESSING AND DISPLAY UTILITIES
# =============================================================================

def normalize_to_8bit(image: np.ndarray) -> np.ndarray:
    """Normalizes an image using 1st and 99.9th percentiles to ignore hot pixels."""
    if image is None:
        logger.warning("normalize_to_8bit received a None image. Returning black square.")
        return np.zeros((100, 100), dtype=np.uint8)
    if image.dtype == np.uint8:
        return image
    
    try:
        # Percentile-based contrast stretching
        p_low, p_high = np.percentile(image, (1.0, 99.9))
        if p_high > p_low:
            normalized = np.clip((image - p_low) / (p_high - p_low) * 255.0, 0, 255)
        else:
            normalized = np.zeros_like(image, dtype=float)
        return normalized.astype(np.uint8)
    except Exception as e:
        logger.warning(f"Failed to normalize image. Returning as is. Error: {e}")
        if image.ndim > 2:
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

def generate_dual_masks(image: np.ndarray, pipette_x: int, threshold_prot: int, 
                        threshold_body: int, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centrally managed mask generation for Protrusion and Cell Body.
    Prevents circular imports between quantification modules.
    """
    img_8u = normalize_to_8bit(image)
    gray = img_8u if len(img_8u.shape) == 2 else cv2.cvtColor(img_8u, cv2.COLOR_BGR2GRAY)
    
    img_params = params.get('image_processing', {})
    guv_settings = params.get('guv_settings', {})
    margin_fraction = img_params.get('wall_clip_margin', 0.25)
    # fill_holes is driven entirely by guv_settings so it stays in one place.
    fill_holes = (
        guv_settings.get('enable', False) and
        guv_settings.get('fill_membrane_holes', True)
    )
    
    clahe = cv2.createCLAHE(clipLimit=img_params.get('clahe_clip_limit', 2.0), tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, tuple(img_params.get('gaussian_kernel_size', (3,3))), 0)
    
    h, w = gray.shape
    margin = int(h * margin_fraction)
    pip_x = max(0, min(w, int(pipette_x)))
    
    # 1. Protrusion Mask (Left of pipette, walls clipped)
    _, bin_prot = cv2.threshold(blurred, threshold_prot, 255, cv2.THRESH_BINARY)
    
    # Pass pip_x to allow optional split-convex hull filling before clipping
    mask_prot = _clean_mask_internal(bin_prot, fill_holes, pip_x)
    
    if margin > 0:
        mask_prot[:margin, :] = 0
        mask_prot[h-margin:, :] = 0
    mask_prot[:, pip_x:] = 0
    
    # 2. Body Mask (Right of pipette)
    # In GUV mode the membrane ring brightness fluctuates frame-to-frame, so a
    # fixed threshold_body systematically clips the dimmer ring arcs and degrades
    # coverage.  We compute an Otsu threshold on just the body ROI for each frame
    # and scale it down slightly so the outer ring arcs (which fall below the
    # Otsu boundary) are still captured.  The blob-selection step in
    # _clean_guv_mask then discards anything that isn't actually the GUV.
    # For cells this branch is never entered and behaviour is unchanged.
    is_guv = guv_settings.get('enable', False)
    if is_guv:
        body_roi = blurred[margin:h - margin, pip_x:]
        if body_roi.size > 0 and cv2.countNonZero(body_roi) > 50:
            otsu_val, _ = cv2.threshold(body_roi, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Scale factor < 1.0 lowers the threshold so dimmer ring arcs are
            # included.  Exposed in config as guv_settings → body_threshold_fraction
            # (default 0.75).  The blob-scoring step handles false inclusions.
            fraction = guv_settings.get('body_threshold_fraction', 0.75)
            adaptive_thr = max(int(otsu_val * fraction), 20)
        else:
            adaptive_thr = threshold_body   # fallback: ROI is empty or all-black
        _, bin_body = cv2.threshold(blurred, adaptive_thr, 255, cv2.THRESH_BINARY)
    else:
        _, bin_body = cv2.threshold(blurred, threshold_body, 255, cv2.THRESH_BINARY)
    mask_body = _clean_mask_internal(bin_body, fill_holes, pip_x)
    mask_body[:, :pip_x] = 0
    
    # Keep only the blob whose left edge is nearest to pipette_x.
    # Passing cells in the far pocket will have a left edge much further
    # to the right, so they get discarded automatically.
    mask_body = _keep_blob_nearest_to_x(mask_body, pip_x)

    return mask_prot, mask_body

def _clean_mask_internal(binary: np.ndarray, fill_holes: bool = False, pipette_x: Optional[int] = None) -> np.ndarray:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Advanced Split-Convex Hull filling for patchy GUV membranes
    if fill_holes and pipette_x is not None:
        left_half = closed.copy()
        left_half[:, pipette_x:] = 0
        
        right_half = closed.copy()
        right_half[:, :pipette_x] = 0
        
        filled = np.zeros_like(closed)
        
        pts_left = cv2.findNonZero(left_half)
        if pts_left is not None:
            hull_left = cv2.convexHull(pts_left)
            cv2.drawContours(filled, [hull_left], -1, 255, -1)
            
        pts_right = cv2.findNonZero(right_half)
        if pts_right is not None:
            hull_right = cv2.convexHull(pts_right)
            cv2.drawContours(filled, [hull_right], -1, 255, -1)
            
        closed = filled
        
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

def _keep_blob_nearest_to_x(binary_mask: np.ndarray, reference_x: int) -> np.ndarray:
    """
    Keeps only the blob whose left edge is closest to reference_x.
    
    Why this works for our experiment:
        The main cell body always sits directly to the right of the pipette
        entrance, so its left edge is near reference_x. A passing cell that
        drifts through the pocket sits further right, giving it a larger
        left-edge distance. We discard everything except the nearest blob.
    
    Parameters
    ----------
    binary_mask  : uint8 binary image (255 = foreground)
    reference_x  : the pipette entrance x-coordinate
    
    Returns
    -------
    A new mask containing only the single nearest blob, or an empty mask
    if no blobs are found.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    
    if num_labels <= 1:
        # Only background label — nothing to keep.
        return np.zeros_like(binary_mask)
    
    best_label    = -1
    best_distance = float('inf')
    
    for i in range(1, num_labels):   # skip label 0 = background
        # CC_STAT_LEFT is the x-coordinate of the blob's leftmost pixel.
        left_edge = stats[i, cv2.CC_STAT_LEFT]
        
        # Distance between this blob's left edge and the pipette entrance.
        # The main cell will have the smallest value here.
        distance = abs(left_edge - reference_x)
        
        if distance < best_distance:
            best_distance = distance
            best_label    = i
    
    if best_label == -1:
        return np.zeros_like(binary_mask)
    
    output = np.zeros_like(binary_mask)
    output[labels == best_label] = 255
    return output

def generate_cortex_masks(mask_total: np.ndarray, cortex_thickness_px: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a binary cell mask into a cortex shell and an interior (lumen).

    Steps:
        1. Erode the full cell mask inward by cortex_thickness_px pixels → lumen.
        2. Subtract lumen from original → cortex shell (the ring that eroded away).

    Parameters
    ----------
    mask_total : np.ndarray
        Binary mask of the whole cell region (255 = cell, 0 = background).
    cortex_thickness_px : int
        Shell thickness in pixels. At 0.629 µm/px, 3 px ≈ 1.9 µm.
        Configurable in config.yaml under actin_parameters → cortex_thickness_px.

    Returns
    -------
    mask_cortex : np.ndarray
        Binary mask of just the outer shell.
    mask_lumen : np.ndarray
        Binary mask of just the interior.
    """
    # Elliptical kernel erodes evenly in all directions (avoids over-eroding corners).
    kernel_size = 2 * cortex_thickness_px + 1   # must be odd, e.g. 3 px → 7×7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask_lumen  = cv2.erode(mask_total, kernel, iterations=1)
    mask_cortex = cv2.subtract(mask_total, mask_lumen)   # clamps at 0, no negatives

    return mask_cortex, mask_lumen


def calculate_spatial_profile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculates mean intensity per column within the mask."""
    col_sums = np.sum(image * (mask > 0), axis=0)
    col_counts = np.sum((mask > 0), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = col_sums / col_counts
        profile[col_counts == 0] = 0
    return profile

def _keep_connected_to_pipette_internal(mask: np.ndarray, pip_x: int, tolerance: int = 5) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1: return mask
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        x, y, w, h_rect, area = stats[i]
        if (x + w) >= (pip_x - tolerance):
            new_mask[labels == i] = 255
    return new_mask

# =============================================================================
# 4. INTERACTIVE WINDOW UTILITIES
# =============================================================================

def draw_ui_text(img: np.ndarray, text: str, pos: Tuple[int, int], 
                 scale: float = 0.8, color: Optional[Tuple[int,int,int]] = None, thickness: int = 2) -> None:
    """
    Standardized text drawing function with high-contrast drop shadow.
    Increased size and thickness for better visibility.
    """
    if color is None: color = UI_COLORS['text']
    
    # Stronger Drop Shadow (Thick Black Outline) for contrast
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3)
    # Main Text
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    
def add_text_overlay(img: np.ndarray, text_lines: List[str], bottom_margin: int = 30) -> None:
    """
    Adds multiple lines of text to an image's bottom-left corner.
    Includes a semi-transparent background box to ensure readability.
    """
    if not text_lines: return
    
    h, w = img.shape[:2]
    
    # Settings for larger, readable text
    font_scale = 0.8
    font_thickness = 2
    line_height = 35
    padding = 15
    
    # Calculate Box Dimensions
    num_lines = len(text_lines)
    text_block_height = num_lines * line_height
    
    # Estimate width (approximate, usually sufficient)
    max_char_count = max(len(line) for line in text_lines)
    box_width = int(max_char_count * 15 * font_scale) + (padding * 2)
    box_width = min(box_width, w) # Clamp to image width
    
    # Box Coordinates
    y_start = h - bottom_margin - text_block_height - padding
    y_end = h - bottom_margin + padding
    
    # Draw Semi-Transparent Background Box
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y_start), (box_width, y_end), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Draw Lines
    # We iterate forward to draw Top-to-Bottom within the calculated box
    current_y = y_start + line_height + 5
    for text in text_lines:
        draw_ui_text(img, text, (padding, current_y), scale=font_scale, thickness=font_thickness, color=UI_COLORS['text'])
        current_y += line_height

def create_centered_window(window_name: str, width: int, height: int) -> None:
    """Creates and centers a resizable OpenCV window on the screen."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    screen_w, screen_h = get_screen_dimensions()
    pos_x = max(0, (screen_w - width) // 2)
    pos_y = max(0, (screen_h - height) // 2)
    cv2.moveWindow(window_name, pos_x, pos_y)

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

# =============================================================================
# 7. IMAGE MANIPULATION UTILITIES (WORKER SAFE)
# =============================================================================

def crop_single_trap(frame: np.ndarray, roi: List[int], rotation_angle: float = 0.0) -> Optional[np.ndarray]:
    """
    Worker-safe cropping. If rotation_angle is 0, it uses high-speed slicing.
    This prevents redundant rotation of pre-processed frames in shared memory.
    """
    if frame is None or roi is None:
        return None
        
    # Only rotate if the angle is not zero (handles raw frame processing)
    if rotation_angle != 0:
        frame = rotate_image(frame, rotation_angle)
    
    y_min, y_max, x_min, x_max = roi
    h, w = frame.shape[:2]
    
    # Safety clipping
    y_min, x_min = max(0, y_min), max(0, x_min)
    y_max, x_max = min(h, y_max), min(w, x_max)
    
    return frame[y_min:y_max, x_min:x_max].copy()