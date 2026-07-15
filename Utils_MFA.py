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
    # fontTools: matplotlib delegates PDF font embedding (pdf.fonttype=42) to
    # fontTools, which by default logs an INFO line per glyph-table subset and
    # a DEBUG line per table read. Silencing the parent logger covers every
    # child (fontTools.subset, fontTools.ttLib.ttFont, fontTools.subset.timer,
    # fontTools.ttLib.tables.otBase, ...).
    logging.getLogger('fontTools').setLevel(logging.WARNING)

# =============================================================================
# 2. PLOTTING AND STYLING UTILITIES
# =============================================================================
# This block was ported from the GUV chapter's plotting.py so that every
# figure in every chapter of the thesis prints at the same physical size,
# with the same fonts, at the same target point size. The core idea: define
# ONE canonical printed width (page_textwidth_in, derived from an A5 page at
# hscale=0.75) and one target point size for each text role (title, label,
# tick, legend, annot). Figures drawn at a non-standard width use
# _fs_for_width() so their fonts are scaled to come out at the right point
# size after LaTeX resizes them on the final page.
#
# The categorical data palette has been swapped to the same blue ramp used
# in the GUV chapter so both chapters read as one visual family. The blue
# -> red time gradient (kymographs / time-color sequences) is kept — it
# encodes time DIRECTION, which a monochrome ramp cannot express.
#
# Pulse and rupture event markers use warm accent hues so they stay
# distinguishable against the now-all-blue data traces.
# =============================================================================

import seaborn as sns  # required by set_paper_style() below

# --- Centralized Color Palette (Blue-family with warm event accents) -------
MFA_COLORS = {
    # ── Blue ramp (canonical categorical palette) ──────────────────────────
    # Same hex codes as the GUV chapter's CONDITION_PALETTE. Legacy keys
    # 'dark_blue' / 'medium_blue' / 'light_blue' / 'pale_blue' are kept so
    # any old references still resolve — they just now point at blue-purple
    # tones instead of the previous saturated-blue tones.
    'dark_blue':   '#1a243d',
    'medium_blue': '#34487a',
    'light_blue':  '#5d6ea6',
    'pale_blue':   '#9896bb',

    # ── Neutrals for grid lines, empty states, etc. ────────────────────────
    'light_grey':  '#E7E6E3',
    'white':       '#F9F9F9',

    # ── Warm accents (RESERVED for event markers / annotations) ────────────
    # Data traces should not draw from these — they are reserved so pulse
    # and rupture vertical lines stay visible against blue data.
    'pale_red':    '#FDDBC7',
    'light_red':   '#F6A482',
    'medium_red':  '#D75F4C',
    'dark_red':    '#B31529',

    # ── UI semantic mapping (OpenCV overlays — kept unchanged) ─────────────
    # OpenCV UI elements sit on live microscopy frames, not data plots, so
    # they keep their high-contrast blue/red hues for clarity against
    # grayscale image data.
    'ui_guide':    '#3A93C3',
    'ui_pipette':  '#1065AB',
    'ui_mask':     '#B31529',
    'ui_text':     '#F9F9F9',

    # ── Plotting defaults (semantic slots — DO NOT rename) ─────────────────
    # Downstream plotting code references these by name. Only the hex
    # values have been rewired to the blue palette; keys are unchanged.
    'primary':     '#1a243d',  # darkest blue — main trace / raw points
    'secondary':   '#5d6ea6',  # medium blue  — protrusion / fits
    'tertiary':    '#9896bb',  # pale blue    — cell body
    'quaternary':  '#34487a',  # dark blue    — total / intensity trace

    # ── Event markers (warm, for contrast against blue data) ───────────────
    'pulse':       '#D75F4C',  # medium red — pulse vertical dashed line
    'rupture':     '#B31529',  # dark red   — rupture vertical dashed line

    'grid':        '#D9E4E9',
    'background':  '#FFFFFF',

    # ── Blue -> red time gradient (kymograph / time-colored traces) ────────
    # Diverging on purpose. Blue = early, red = late is the strongest
    # time-direction cue available and no monochrome ramp matches it.
    'time_gradient_start': '#1a243d',
    'time_gradient_end':   '#B31529',
}

# --- Region palette (parallel to bulk_plotting's PALETTE_REGION) -----------
# Mirrors the GUV cortex-density gradient logic: the region of primary
# interest sits at medium tone, contextual regions at lighter tones, and
# aggregate summaries at the darkest tone.
MFA_REGION_PALETTE = {
    'Protrusion': '#5d6ea6',  # medium — the focal region of interest
    'Body':       '#9896bb',  # light  — contextual backdrop
    'Total':      '#1a243d',  # darkest — aggregate summary
    'Tip':        '#34487a',  # dark   — top-N% subregion within protrusion
}

# =============================================================================
# 2a. PAGE-AWARE FIGURE SIZING (ported from GUV chapter plotting.py)
# =============================================================================

# The single shared "settings sheet" for every figure. Downstream plotting
# code should read canvas widths and target font sizes from here so a
# single edit changes the whole thesis's visual proportions.
PLOT_STYLE = {
    "font_family":          "Arial",
    "sns_style":            "ticks",
    "sns_context":          "paper",

    # A5 page 17 cm wide x hscale 0.75 = 12.75 cm = 5.02 in.
    # This is the width _fs() targets when computing font sizes, so text
    # is drawn at the correct point size after LaTeX resizes the figure
    # to fit this width on the final page.
    "page_textwidth_in":    5.02,

    # Target PRINTED font sizes (points) on the final page.
    "fontsize_title_pt":    12.0,
    "fontsize_label_pt":    12.0,
    "fontsize_tick_pt":     12.0,
    "fontsize_legend_pt":   12.0,
    "fontsize_annot_pt":    12.0,

    # Line widths (unitless — no pt-conversion needed).
    "axes_linewidth":       1.5,
    "tick_linewidth":       1.5,

    # Canvas widths (inches). "Single column" = one Axes.
    # Grids multiply per_panel_* by nrows / ncols.
    "single_col_width_in":      7.0,
    "single_col_height_in":     5.5,
    "per_panel_width_in":       3.0,
    "per_panel_height_in":      3.6,

    "dpi_raster":               600,   # for raster elements embedded in PDF
}


def _fs(role: str) -> float:
    """
    Returns the matplotlib fontsize needed for text to PRINT at the target
    point size in PLOT_STYLE, assuming the figure is drawn at the standard
    single-column width. Roles: title, label, tick, legend, annot.

    For grids or other non-standard widths, use _fs_for_width() instead —
    pass in that figure's actual width so the conversion accounts for how
    much the figure will be scaled on the printed page.
    """
    return _fs_for_width(role, PLOT_STYLE["single_col_width_in"])


def _fs_for_width(role: str, drawn_width_in: float) -> float:
    """
    Same as _fs() but for figures whose canvas width isn't the standard
    single-column width. Pass in the width (inches) actually used in
    plt.subplots(figsize=(drawn_width_in, ...)).
    """
    target_pt = PLOT_STYLE[f"fontsize_{role}_pt"]
    scale_factor = PLOT_STYLE["page_textwidth_in"] / drawn_width_in
    return target_pt / scale_factor


def _get_figsize_single_col() -> Tuple[float, float]:
    """Standard (width, height) in inches for a one-panel figure."""
    return (PLOT_STYLE["single_col_width_in"], PLOT_STYLE["single_col_height_in"])


def _get_figsize_grid(nrows: int, ncols: int) -> Tuple[float, float]:
    """
    (width, height) in inches for an nrows x ncols grid. Because width
    scales with ncols, downstream code MUST use _fs_for_width(role, width)
    when setting fonts on grid figures.
    """
    width  = PLOT_STYLE["per_panel_width_in"]  * ncols
    height = PLOT_STYLE["per_panel_height_in"] * nrows
    return (width, height)

# =============================================================================
# 2b. OPENCV UI COLORS (unchanged)
# =============================================================================

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

# =============================================================================
# 2c. GLOBAL STYLE + SAVE HELPERS
# =============================================================================

def set_paper_style(*_legacy_args, **_legacy_kwargs) -> None:
    """
    Applies the shared seaborn theme and matplotlib rcParams used by every
    figure in the pipeline. Ports the GUV chapter's set_paper_style():
        - seaborn "ticks" style, "paper" context
        - Arial (falling back to DejaVu Sans)
        - top/right spines off
        - pdf.fonttype = 42 (Type-3-free PDFs)
        - Tick/label/title/legend sizes come from PLOT_STYLE via _fs()

    Grid figures with a non-standard width should still call
    _fs_for_width(role, width) locally when setting sizes; the rcParams
    here only cover the standard-width case.

    *args/**kwargs are absorbed so legacy call sites that used to pass
    base_fontsize / dpi still work — those parameters are now controlled
    centrally through PLOT_STYLE.
    """
    sns.set_theme(
        style   = PLOT_STYLE["sns_style"],
        context = PLOT_STYLE["sns_context"],
    )
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    [PLOT_STYLE["font_family"], "DejaVu Sans"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     PLOT_STYLE["axes_linewidth"],
        "xtick.major.width":  PLOT_STYLE["tick_linewidth"],
        "ytick.major.width":  PLOT_STYLE["tick_linewidth"],
        "xtick.labelsize":    _fs("tick"),
        "ytick.labelsize":    _fs("tick"),
        "axes.labelsize":     _fs("label"),
        "axes.titlesize":     _fs("title"),
        "legend.fontsize":    _fs("legend"),
        "legend.frameon":     False,
        "figure.facecolor":   MFA_COLORS['background'],
        "grid.color":         MFA_COLORS['grid'],
        "grid.alpha":         0.4,
        "pdf.fonttype":       42,
        "svg.fonttype":       "none",
    })

def get_time_colormap(n_steps: int) -> List[Tuple[float, float, float, float]]:
    """
    Discrete sample of the blue -> red time gradient. Kept diverging on
    purpose: blue -> red encodes time direction in kymographs and time-
    colored trace overlays. No monochrome ramp is equivalent.
    """
    colors = [
        MFA_COLORS['dark_blue'],   MFA_COLORS['medium_blue'], MFA_COLORS['light_blue'],
        MFA_COLORS['pale_blue'],   MFA_COLORS['light_grey'],  MFA_COLORS['pale_red'],
        MFA_COLORS['light_red'],   MFA_COLORS['medium_red'],  MFA_COLORS['dark_red'],
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mfa_full_gradient", colors)
    if n_steps == 1:
        return [cmap(0.5)]
    return [cmap(i / (n_steps - 1)) for i in range(n_steps)]

def get_mfa_continuous_cmap() -> mpl.colors.LinearSegmentedColormap:
    """
    Continuous colormap for kymographs (Black -> Blue -> White -> Red).
    Same rationale as get_time_colormap: kymographs need direction, not
    category, so the diverging axis is preserved.
    """
    nodes = [0.0, 0.2, 0.5, 0.8, 1.0]
    colors = [
        '#000000',
        MFA_COLORS['dark_blue'],
        MFA_COLORS['light_blue'],
        MFA_COLORS['medium_red'],
        MFA_COLORS['dark_red'],
    ]
    return mpl.colors.LinearSegmentedColormap.from_list("mfa_kymo", list(zip(nodes, colors)))

def save_plot_pdf(save_path: Union[str, Path], dpi: Optional[int] = None) -> None:
    """
    Saves the current matplotlib figure as a PDF at the standardized DPI.

    Why PDF instead of PNG:
        PDFs store drawings as vectors, so they stay perfectly sharp at any
        print/zoom level — important for thesis figures embedded in LaTeX.
        dpi still matters for RASTER elements baked into the figure (e.g.
        scatter markers, imshow images); PLOT_STYLE["dpi_raster"] controls
        that so it's consistent across every figure.

    Whatever extension is in save_path is replaced with '.pdf'. This keeps
    existing call sites — which pass '.png' paths — working during the
    migration.
    """
    if dpi is None:
        dpi = PLOT_STYLE["dpi_raster"]
    save_path_pdf = Path(save_path).with_suffix(".pdf")
    save_path_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path_pdf), dpi=dpi, bbox_inches='tight', format='pdf')
    logger.info(f"Plot saved: {save_path_pdf.name}")

# Backward-compat alias: any legacy code still calling save_plot_png() now
# transparently writes a PDF. Safe to remove once every call site has been
# migrated to save_plot_pdf().
def save_plot_png(save_path: Union[str, Path], dpi: Optional[int] = None) -> None:
    """DEPRECATED shim during the PNG -> PDF migration.
    Writes a PDF regardless of the extension in save_path."""
    save_plot_pdf(save_path, dpi=dpi)

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
    h, w = gray.shape
    
    img_params = params.get('image_processing', {})
    guv_settings = params.get('guv_settings', {})
    is_guv = guv_settings.get('enable', False)
    
    # FIX: Explicitly abort if called in GUV mode to prevent thresholding the lumen
    if is_guv:
        logger.error("generate_dual_masks called in GUV mode! Thresholding is invalid for GUVs. Returning empty masks.")
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    margin_fraction = img_params.get('wall_clip_margin', 0.25)

    # fill_membrane_holes is a GUV-specific technique: it fills the hollow
    # interior of a vesicle membrane ring using convex hulls.  For regular cells,
    # applying convex hulls replaces the actual (possibly irregular/concave) cell
    # boundary with a smoothed bounding polygon — exactly what causes the mask to
    # not follow the cell contour.  Default is False so this never fires unless
    # the config explicitly sets it, which only makes sense in GUV mode.
    fill_holes = guv_settings.get('fill_membrane_holes', False)
    
    clahe = cv2.createCLAHE(clipLimit=img_params.get('clahe_clip_limit', 2.0), tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, tuple(img_params.get('gaussian_kernel_size', (3,3))), 0)
    
    margin = int(h * margin_fraction)
    pip_x = max(0, min(w, int(pipette_x)))
    
    # 1. Protrusion Mask 
    _, bin_prot = cv2.threshold(blurred, threshold_prot, 255, cv2.THRESH_BINARY)
    mask_prot = _clean_mask_internal(bin_prot, fill_holes, pip_x)
    if margin > 0:
        mask_prot[:margin, :] = 0
        mask_prot[h-margin:, :] = 0
    mask_prot[:, pip_x:] = 0
    
    # 2. Body Mask 
    _, bin_body = cv2.threshold(blurred, threshold_body, 255, cv2.THRESH_BINARY)
    mask_body = _clean_mask_internal(bin_body, fill_holes, pip_x)
    mask_body[:, :pip_x] = 0
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
# 3b. SHARED SPATIAL-ZONE UTILITIES
#
# These are consumed by both ActinQuantification and UptakeQuantification so
# the two modules use identical geometric definitions for the "tip" region and
# for the four-zone split.  Keeping the logic here is the single source of
# truth; any change to how a zone is defined propagates to every downstream
# quantification module.
# =============================================================================

def build_zone_masks(mask_prot: np.ndarray,
                     mask_body: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split the protrusion and body into four spatial zones based on their
    x-column extents.

    Protrusion (leftward of the pipette entrance):
        tip  -> left half (deepest in channel, farthest from entrance)
        base -> right half (nearest to entrance)

    Body (rightward of the pipette entrance):
        perinuclear -> near 1/3 of the body (adjacent to entrance)
        distal      -> far  2/3 of the body (away from entrance)

    Empty inputs return zero-filled masks so downstream mean() calls stay
    graceful.
    """
    # --- Protrusion split at midpoint --------------------------------------
    prot_cols = np.where(mask_prot.any(axis=0))[0]
    if prot_cols.size >= 2:
        tip_x_start = int(prot_cols[0])     # deepest column
        prot_x_end  = int(prot_cols[-1])    # column nearest entrance
        midpoint    = (tip_x_start + prot_x_end) // 2

        mask_tip  = mask_prot.copy()
        mask_tip[:, midpoint + 1:] = 0

        mask_base = mask_prot.copy()
        mask_base[:, :midpoint + 1] = 0
    else:
        mask_tip  = np.zeros_like(mask_prot)
        mask_base = np.zeros_like(mask_prot)

    # --- Body split at 1/3 -------------------------------------------------
    body_cols = np.where(mask_body.any(axis=0))[0]
    if body_cols.size >= 2:
        body_x_start = int(body_cols[0])
        body_x_end   = int(body_cols[-1])
        body_extent  = body_x_end - body_x_start
        split_x      = body_x_start + max(1, body_extent // 3)

        mask_perinuclear = mask_body.copy()
        mask_perinuclear[:, split_x + 1:] = 0

        mask_distal = mask_body.copy()
        mask_distal[:, :split_x + 1] = 0
    else:
        mask_perinuclear = np.zeros_like(mask_body)
        mask_distal      = np.zeros_like(mask_body)

    return {
        'tip':         mask_tip,
        'base':        mask_base,
        'perinuclear': mask_perinuclear,
        'distal':      mask_distal,
    }


def build_tip_mask(mask_prot: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper that returns only the protrusion-tip mask (far half
    of the protrusion, deepest in channel).  Uses the same midpoint split as
    build_zone_masks so both are consistent by construction.
    """
    prot_cols = np.where(mask_prot.any(axis=0))[0]
    if prot_cols.size < 2:
        return np.zeros_like(mask_prot)

    tip_x_start = int(prot_cols[0])
    prot_x_end  = int(prot_cols[-1])
    midpoint    = (tip_x_start + prot_x_end) // 2

    mask_tip = mask_prot.copy()
    mask_tip[:, midpoint + 1:] = 0
    return mask_tip


# =============================================================================
# 3c. PULSE-FRAME INDEX HELPERS
#
# The imaging software counts frames starting at either 0 (files t0000, t0001,
# ...) or 1 (files t0001, t0002, ...).  The config field `pulse_frame` refers
# to the filename suffix of the pulsed frame -- e.g. pulse_frame: 10 means
# file t0010 -- but the analysis pipeline needs a 0-based position within the
# sorted file list.  The offset between "filename suffix" and "list position"
# equals the first file's numeric suffix, so we detect it once and reuse.
# =============================================================================

import re as _re  # local alias to avoid re-importing at module top


def extract_frame_number_from_filename(filename) -> Optional[int]:
    """
    Parse the numeric suffix from a frame filename.

    Recognizes '_t123' (preferred) and, as a fallback, trailing digits before
    the '.tif' / '.tiff' extension.  Returns None if no number can be parsed.
    """
    name = str(filename)
    # Preferred pattern: '_t' followed by digits
    m = _re.search(r'_t(\d+)', name)
    if m:
        return int(m.group(1))
    # Fallback: trailing digits before .tif / .tiff
    m = _re.search(r'(\d+)\.(tif|tiff)$', name, _re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def get_first_frame_number(file_list) -> int:
    """
    Return the numeric suffix of the first file in a sorted file list.
    Defaults to 1 (the historical assumption in this codebase) if the number
    can't be parsed, since t0001-start is the more common Zeiss convention.
    """
    if not file_list:
        return 1
    n = extract_frame_number_from_filename(file_list[0])
    return n if n is not None else 1


def pulse_frame_to_index(config_pulse_frame, first_frame_number: int = 1) -> int:
    """
    Convert the config's `pulse_frame` (which matches the filename suffix
    of the pulsed frame) into a 0-based array position within the sorted
    file list.

    Example:
        Files start at t0000: first_frame_number = 0
            pulse_frame = 10 -> file t0010 -> position 10
        Files start at t0001: first_frame_number = 1
            pulse_frame = 10 -> file t0010 -> position 9

    General rule: position = pulse_frame - first_frame_number.
    Clamped to >= 0 for safety.
    """
    if config_pulse_frame is None:
        return 0
    return max(0, int(config_pulse_frame) - int(first_frame_number))

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