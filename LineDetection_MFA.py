# -*- coding: utf-8 -*-
"""
Automated Detection of Membrane Protrusion in Micropipette Aspiration (MFA)

ROLE IN PIPELINE:
This module is the "eyes" of the analysis. It takes raw image frames and 
outputs the physical length of the cell protrusion over time.

KEY ALGORITHMS:
1.  Adaptive Preprocessing: Uses CLAHE to normalize lighting differences.
2.  Sub-pixel Edge Detection: Uses gradient analysis to measure lengths with 
    accuracy greater than the pixel grid (e.g., 50.4 px instead of 50 px).
3.  CUSUM Rupture Detection: A statistical quality control algorithm used to 
    detect the exact moment a cell bursts by monitoring "haze" inside the pipette.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import Utils_MFA as utils

logger = logging.getLogger(__name__)

# =============================================================================
# DETECTION ALGORITHM
# =============================================================================

class LineDetectionMFA:
    """
    Main class for the protrusion detection pipeline.
    Maintains state across frames to track temporal events like rupture.
    """

    def __init__(self, roi_images: List[np.ndarray], pipette_coords: List[int], params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the detection object.

        Args:
            roi_images: List of cropped numpy arrays (Regions of Interest).
            pipette_coords: [x, y] coordinates of the pipette tip (user-selected).
            params: Dictionary of configuration parameters.
        """
        self.roi_images = roi_images
        self.pipette_coords = pipette_coords
        self.params = params or {}


        # CUMSUM parameters
        self.entry_velocity_threshold = self.params.get('entry_velocity_threshold_px', 2.0)
        self.min_cusum_baseline = self.params.get('min_cusum_baseline_frames', 5)
        self.min_sigma = self.params.get('min_intensity_noise_floor', 0.2)
        self.min_drift = self.params.get('min_drift_tolerance', 0.2)
        self.min_cusum_thresh = self.params.get('min_cusum_threshold', 2.0)
        
        
        # Geometry & Processing Parameters
        self.min_area_threshold: int = self.params.get('min_area_threshold', 100)         
        self.small_object_threshold: int = self.params.get('small_object_threshold', 50)  
        self.window_width: int = self.params.get('window_width', 1200)
        self.window_height: int = self.params.get('window_height', 800)
        self.wall_clip_margin: float = self.params.get('wall_clip_margin', 0.40)         

        # --- Results Container ---
        self.results: Dict[str, Any] = {
            # Pre-processing
            'signal': 'stop',               # Flow control signal (confirm/restart/stop)
            'protrusion_lengths_px': [],    # Raw length data in pixels
            'protrusion_lengths_um': [],    # Converted length in microns 
            # Rupture Detection
            'downstream_intensities': [],   # Brightness values inside the pipette (for rupture)'
            'rupture_detected': False,      # Boolean flag for rupture event
            'rupture_frame_index': None,    # Frame index where rupture occurred
            'debug_images': [],             # Visualizations with overlays
            # Protrusion Detection
            'pipette_start_x_used': None,   # The X-coordinate used as "Zero"
            'threshold_prot': None,         # Primary threshold (clipped)
            'threshold_body': None,         # Secondary threshold (full height)
            # Data
            'processing_metadata': {},      # Extra info
            'analysis_timestamp': datetime.now().isoformat(),
            'detection_confidence': []      # Metric 0.0 or 1.0 indicating tracking success
        }

    def run_detection(self) -> Dict[str, Any]:
        """
        Executes the full pipeline with USER INTERACTION.
        Used during the setup phase to let the user tune parameters.
        """
        if not self.roi_images or all(frame is None for frame in self.roi_images):
            logger.warning("No valid frames found for analysis.")
            self.results['signal'] = 'stop'
            return self.results

        logger.info("=" * 60)
        logger.info("MEMBRANE DETECTION PIPELINE")
        logger.info("=" * 60)
        
        # Pick a middle frame for setup (usually has the cell visible)
        mid_idx = len(self.roi_images) // 2
        test_image = self.roi_images[mid_idx] if self.roi_images[mid_idx] is not None else self.roi_images[0]
        
        if test_image is None:
            logger.error("CRITICAL: No valid images found in ROI sequence. All frames are None. Check image loading and ROI extraction.")
            self.results['signal'] = 'stop'
            return self.results
        
        # --- Step 1: User sets Pipette Entrance ---
        # This defines the "Zero" point for length calculations
        signal, pipette_start_x = self._interactive_pipette_positioning(test_image)
        if signal in ('restart', 'stop'):
            self.results['signal'] = signal
            return self.results
        self.results['pipette_start_x_used'] = pipette_start_x

        # --- Step 2: User sets Protrusion Threshold (Standard Mode) ---
        # Focus on the protrusion area (clipped walls)
        logger.info("Step 2: Protrusion Threshold (Wall Clipping Active)")
        signal, thr_prot = self._interactive_threshold_selection(test_image, pipette_start_x, clip_walls=True)
        if signal in ('restart', 'stop'):
            self.results['signal'] = signal
            return self.results
        self.results['threshold_prot'] = thr_prot
        
        # --- Step 3: Cell Body Threshold (Dye Mode Only) ---
        # If dye is enabled, we need a separate threshold for the body (Full Height)
        dye_enabled = self.params.get('dye_uptake_parameters', {}).get('enable', False)
        
        if dye_enabled:
            logger.info("Step 3: Cell Body Threshold (Full Height for Dye)")
            signal, thr_body = self._interactive_threshold_selection(test_image, pipette_start_x, clip_walls=False)
            if signal in ('restart', 'stop'):
                self.results['signal'] = signal
                return self.results
            self.results['threshold_body'] = thr_body
        else:
            # Fallback: Use the same threshold if dye not enabled
            self.results['threshold_body'] = thr_prot            
            
        # --- Step 4: Automated Processing (Protrusion Detection) ---
        # We only process length using the protrusion threshold here
        self._process_all_frames(pipette_start_x, thr_prot)
        self._generate_comprehensive_results()

        self.results['signal'] = 'confirm'
        return self.results

    def run_detection_with_parameters(self, pipette_start_x: int, threshold_prot: int) -> Dict[str, Any]:
        """
        Batch Mode. Runs processing without UI using provided params.
        NOTE: This only runs the length detection using threshold_prot.
        """
        self.results['pipette_start_x_used'] = pipette_start_x
        self.results['threshold_prot'] = threshold_prot
        # threshold_body is not used for length detection, so we ignore it here.
        
        self._process_all_frames(pipette_start_x, threshold_prot)
        self._generate_comprehensive_results()
        self.results['signal'] = 'confirm'
        return self.results
    
    # =========================================================================
    #                       UI / INTERACTION METHODS
    # =========================================================================

    def _interactive_pipette_positioning(self, test_image: np.ndarray) -> Tuple[str, Optional[int]]:
        """
        Opens a GUI window. User moves a vertical line (Cyan) to mark the pipette entrance.
        """
        current_pipette_start_x = self.pipette_coords[0]
        image_width = test_image.shape[1]
        if not (0 <= current_pipette_start_x < image_width): current_pipette_start_x = image_width // 2

        window_name = 'Pipette Entrance Positioning'
        utils.create_centered_window(window_name, self.window_width, self.window_height)

        while True:
            display_image = utils.prepare_display_image(test_image, self.window_width, self.window_height)
            h, w = display_image.shape[:2]
            original_h, original_w = test_image.shape[:2]
            
            # Scale logic to ensure overlay matches mouse/drawing coordinates
            display_scale = min(self.window_width / original_w, self.window_height / original_h) * 0.9
            x_offset = (self.window_width - int(original_w * display_scale)) // 2
            display_pipette_x = x_offset + int(current_pipette_start_x * display_scale)
            
            cv2.line(display_image, (display_pipette_x, 0), (display_pipette_x, h), (0, 255, 255), 3)
            utils.add_text_overlay(display_image, [
                "Pipette Entrance: {} px".format(current_pipette_start_x), 
                "A/D : +/- 1 px",
                "W/S : +/- 10 px",
                "ENTER=Confirm"
            ])
            cv2.imshow(window_name, display_image)

            key = cv2.waitKey(30) & 0xFF
            if key == 13: 
                cv2.destroyAllWindows(); return 'confirm', current_pipette_start_x
            if key == 27: 
                cv2.destroyAllWindows(); return 'stop', None
            if key in (ord('r'), ord('R')): 
                cv2.destroyAllWindows(); return 'restart', None
            
            # Fine Adjustment (1 px)
            elif key in (ord('a'), ord('A')): current_pipette_start_x = max(1, current_pipette_start_x - 1)
            elif key in (ord('d'), ord('D')): current_pipette_start_x = min(test_image.shape[1] - 1, current_pipette_start_x + 1)
            
            # Coarse Adjustment (10 px)
            elif key in (ord('s'), ord('S')): current_pipette_start_x = max(1, current_pipette_start_x - 10)
            elif key in (ord('w'), ord('W')): current_pipette_start_x = min(test_image.shape[1] - 1, current_pipette_start_x + 10)
        
        cv2.destroyAllWindows()
        return 'stop', None

    def _interactive_threshold_selection(self, test_image: np.ndarray, pipette_start_x: int, clip_walls: bool) -> Tuple[str, Optional[int]]:
        """
        Opens a GUI window. User adjusts a threshold to separate Cell from Background.
        Can operate in Protrusion Mode (clip_walls=True) or Body Mode (clip_walls=False).
        Includes mouse interaction for dragging the threshold line.
        """
        test_image_8bit = utils.normalize_to_8bit(test_image)
        h, w = test_image_8bit.shape[:2]
        
        # 1. Determine ROI for statistics/histogram based on mode
        margin = int(h * self.wall_clip_margin) if clip_walls else 0
        
        if clip_walls:
             # PROTRUSION: Left of pipette, clipped walls
             roi_for_stats = test_image_8bit[margin:h-margin, :pipette_start_x] if margin > 0 else test_image_8bit[:, :pipette_start_x]
        else:
             # BODY: Right of pipette, full height
             roi_for_stats = test_image_8bit[:, pipette_start_x:]

        # 2. Initial Threshold Guess (Otsu)
        if roi_for_stats.size > 0:
            otsu_val, _ = cv2.threshold(roi_for_stats, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            initial_threshold = int(otsu_val)
        else:
            initial_threshold = 50
            
        window_name = f"Threshold Selection ({'Protrusion' if clip_walls else 'Cell Body'})"
        utils.create_centered_window(window_name, self.window_width, self.window_height)
        
        # Define panel widths here to ensure sync between Drawing and Mouse Logic
        img_panel_w = int(self.window_width * 0.45)
        hist_panel_w = int(self.window_width * 0.55)

        # Shared state for mouse callback
        state = {
            'threshold': initial_threshold,
            'dragging': False,
            'offset_x': img_panel_w,   # X-coordinate where histogram starts
            'width': hist_panel_w      # Width of the histogram panel
        }

        # --- Mouse Callback Function ---
        def threshold_mouse_callback(event, x, y, flags, param):
            hist_x_start = param['offset_x']
            hist_w = param['width']
            
            # Helper: Map mouse X position to Threshold (0-255)
            def update_threshold_from_mouse(mouse_x):
                rel_x = mouse_x - hist_x_start
                val = int((rel_x / hist_w) * 255)
                param['threshold'] = max(0, min(255, val))

            # 1. Click to jump
            if event == cv2.EVENT_LBUTTONDOWN:
                if x >= hist_x_start: # Check if click is inside histogram
                    param['dragging'] = True
                    update_threshold_from_mouse(x)
            
            # 2. Drag to slide
            elif event == cv2.EVENT_MOUSEMOVE:
                if param['dragging']:
                    # Update even if mouse strays slightly outside panel while dragging
                    update_threshold_from_mouse(x)
            
            # 3. Release to stop
            elif event == cv2.EVENT_LBUTTONUP:
                param['dragging'] = False

        cv2.setMouseCallback(window_name, threshold_mouse_callback, state)
        
        while True:
            current_threshold = state['threshold'] # Read latest value from state

            # 3. Generate Mask
            if clip_walls:
                # PROTRUSION: Clip everything to the RIGHT (keep left)
                binary_mask = self._segment_mask(test_image, current_threshold, clip_walls=True, limit_x_max=pipette_start_x)
            else:
                # BODY: Generate full mask, then manually zero out the LEFT side (keep right)
                binary_mask = self._segment_mask(test_image, current_threshold, clip_walls=False, limit_x_max=None)
                binary_mask[:, :pipette_start_x] = 0
            
            # 4. Create Panels
            panel_height = self.window_height
            
            # Left: Image Overlay
            display_img = self._create_overlay_panel(
                test_image, binary_mask, pipette_start_x, clip_walls, 
                width=img_panel_w, height=panel_height
            )
            
            # Right: Histogram
            hist_img = self._draw_histogram(
                roi_for_stats, current_threshold, 
                width=hist_panel_w, height=panel_height
            )
            
            # 5. Combine and Display
            combined_display = np.hstack((display_img, hist_img))
            
            utils.add_text_overlay(combined_display, [
                f"Threshold: {current_threshold}",
                "Mouse: Drag yellow line to adjust",
                "W/S: +/- 1 | A/D: +/- 10",
                "ENTER: Confirm | R: Restart"
            ])
            cv2.imshow(window_name, combined_display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 13: cv2.destroyAllWindows(); return 'confirm', current_threshold
            if key == 27: cv2.destroyAllWindows(); return 'stop', None
            if key in (ord('r'), ord('R')): cv2.destroyAllWindows(); return 'restart', None
            
            # Keyboard controls update the shared state
            if key in (ord('w'), ord('W')): state['threshold'] = min(255, current_threshold + 1)
            elif key in (ord('s'), ord('S')): state['threshold'] = max(0, current_threshold - 1)
            elif key in (ord('d'), ord('D')): state['threshold'] = min(255, current_threshold + 10)
            elif key in (ord('a'), ord('A')): state['threshold'] = max(0, current_threshold - 10)

    # Visualization Helper Methods

    def _create_overlay_panel(self, image: np.ndarray, binary_mask: np.ndarray, pipette_start_x: int, clip_walls: bool, width: int, height: int) -> np.ndarray:
        """
        Helper to draw the cyan overlay for threshold selection.
        Returns an image formatted for the display window.
        """
        image_8u = utils.normalize_to_8bit(image)
        display_base = cv2.cvtColor(image_8u, cv2.COLOR_GRAY2BGR)
        
        # Create Cyan Overlay (BGR: 255, 255, 0)
        overlay = np.zeros_like(display_base, dtype=np.uint8)
        overlay[binary_mask == 255] = (255, 255, 0) 
        
        # Blend: 70% Image, 30% Overlay
        blended = cv2.addWeighted(display_base, 0.7, overlay, 0.3, 0)
        
        # Draw Pipette Line (Cyan)
        safe_x = max(0, min(image.shape[1], int(pipette_start_x)))
        cv2.line(blended, (safe_x, 0), (safe_x, blended.shape[0]), (0, 255, 255), 1)
        
        # Draw Wall Lines (WHITE) if clipping is active
        if clip_walls:
            h = blended.shape[0]
            margin = int(h * self.wall_clip_margin)
            if margin > 0:
                # Changed from Red (0,0,255) to White (255,255,255)
                cv2.line(blended, (0, margin), (blended.shape[1], margin), (255, 255, 255), 1)
                cv2.line(blended, (0, h-margin), (blended.shape[1], h-margin), (255, 255, 255), 1)

        return utils.prepare_display_image(blended, width, height)

    def _draw_histogram(self, image: np.ndarray, threshold_line: int, width: int = 400, height: int = 500) -> np.ndarray:
        """
        Helper to draw an intensity histogram with a yellow line marking the current threshold.
        """
        hist_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if image is None or image.size == 0:
            cv2.putText(hist_img, "No Data", (10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            return hist_img
            
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        # Normalize to fit in the window height, leaving space for labels
        cv2.normalize(hist, hist, alpha=0, beta=height-80, norm_type=cv2.NORM_MINMAX)
        
        # Draw gray histogram bars
        bin_w = width / 256
        for i in range(1, 256):
            # Previous point
            pt1 = (int((i-1) * bin_w), height - 40 - int(hist[i-1]))
            # Current point
            pt2 = (int(i * bin_w), height - 40 - int(hist[i]))
            cv2.line(hist_img, pt1, pt2, (200, 200, 200), 2)
            
        # Draw yellow threshold line
        x_thresh = int(threshold_line * width / 255)
        cv2.line(hist_img, (x_thresh, 0), (x_thresh, height - 40), (0, 255, 255), 2)
        cv2.putText(hist_img, f"T={threshold_line}", (x_thresh + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Axis labels
        # X-Axis
        cv2.putText(hist_img, "Intensity (0-255)", (width//3, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        # Y-Axis (Approximate)
        cv2.putText(hist_img, "Frequency", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        
        return hist_img
    
    # =========================================================================
    #                       IMAGE PROCESSING LOGIC
    # =========================================================================

    def _segment_mask(self, image: np.ndarray, threshold: int, clip_walls: bool = True, limit_x_max: Optional[int] = None) -> np.ndarray:
        """
        Creates a binary mask based on the provided threshold and geometry settings.
        
        Args:
            image: Raw image.
            threshold: Binary threshold value.
            clip_walls: If True, blacks out top/bottom margins (Standard for protrusion).
            limit_x_max: If provided, blacks out everything to the right of this X.
        """
        image_8bit = utils.normalize_to_8bit(image)
        gray = image_8bit if len(image_8bit.shape) == 2 else cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
        
        # 1. Enhance & Blur
        clahe = cv2.createCLAHE(clipLimit=self.params.get('clahe_clip_limit', 2.0), tileGridSize=tuple(self.params.get('clahe_tile_grid_size', (8, 8))))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, tuple(self.params.get('gaussian_kernel_size', (5, 5))), 0)
        
        # 2. Threshold
        _, binary_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # 3. Geometric Constraints
        h, w = binary_mask.shape
        
        # A. Clip Walls (Standard for protrusion detection)
        if clip_walls:
            margin = int(h * self.wall_clip_margin) 
            if margin > 0:
                binary_mask[:margin, :] = 0
                binary_mask[h-margin:, :] = 0
        
        # B. Clip Right side (if we only want protrusion length)
        if limit_x_max is not None:
            safe_limit = max(0, min(w, limit_x_max))
            binary_mask[:, safe_limit:] = 0

        # 4. Filter Noise (small specks)     
        return self._clean_binary_mask(binary_mask)
    
    def _clean_binary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Standard morphological cleaning for masks."""
        # Filter small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        filtered = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_threshold: 
                filtered[labels == i] = 255
        
        # Close holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filled = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove artifacts on left border
        cleaned = self._remove_left_border_objects(filled)
        
        # Final cleanup
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        return opened

    def _remove_left_border_objects(self, binary_image: np.ndarray) -> np.ndarray:
        """Helper: Removes blobs that are touching the left edge of the image (artifacts)."""
        padded = np.pad(binary_image, ((0, 0), (1, 0)), mode='constant', constant_values=255)
        h, w = padded.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for y in range(h):
            if padded[y, 0] == 255: cv2.floodFill(padded, mask, (0, y), 0)
        return padded[:, 1:]

    def _refine_edge_subpixel(self, image: np.ndarray, int_edge_x: int, stat_entry: np.ndarray, search_window: int = 10) -> float:
        """
        Calculates the edge position with sub-pixel accuracy.
        
        How it works:
        1. Takes an integer edge position (e.g., pixel 50).
        2. Looks at the intensity profile around that pixel (e.g., pixels 45-55).
        3. Finds the "50% intensity drop" point using linear interpolation.
        
        Result: Returns a float (e.g., 50.4) instead of an integer.
        """
        h, w = image.shape
        y_top = stat_entry[cv2.CC_STAT_TOP]
        y_height = stat_entry[cv2.CC_STAT_HEIGHT]
        center_y = y_top + y_height / 2.0
        
        # Define a narrow horizontal strip through the center of the protrusion
        keep_height = max(1, int(y_height * 0.30)) 
        y_start = max(0, int(center_y - keep_height // 2))
        y_end = min(h, int(center_y + keep_height // 2 + 1))
        
        # Look around the detected integer edge
        x_start = max(0, int_edge_x - 30)
        x_end = min(w, int_edge_x + 5)
        
        if x_end - x_start < 5: return float(int_edge_x)
        roi = image[y_start:y_end, x_start:x_end]
        if roi.size == 0: return float(int_edge_x)
        
        # Average intensity profile along the strip
        profile = np.mean(roi, axis=0)
        bg_floor = np.min(profile)
        
        # Calculate dynamic range (contrast)
        obj_level = np.median(profile[-5:]) if len(profile) >= 5 else np.max(profile)
        dynamic_range = obj_level - bg_floor
        
        if dynamic_range < 3.0: return float(int_edge_x) # Contrast too low for subpixel
        
        # Determine 50% threshold for the edge
        threshold = bg_floor + (0.5 * dynamic_range)
        
        # Scan RIGHT-TO-LEFT to find the exact drop-off point
        found_edge = False
        edge_idx = 0
        for i in range(len(profile) - 1, -1, -1):
            if profile[i] < threshold:
                edge_idx = i
                found_edge = True
                break
        
        if not found_edge: return float(x_start)
        
        # Linear interpolation between pixels
        if edge_idx < len(profile) - 1:
            val_low = profile[edge_idx]
            val_high = profile[edge_idx + 1]
            fraction = (threshold - val_low) / (val_high - val_low) if val_high != val_low else 0.5
            offset = edge_idx + fraction
        else:
            offset = float(edge_idx)
            
        return max(0.0, min(float(w), x_start + offset))

    def _measure_downstream_intensity(self, image: np.ndarray, tip_x: float, pipette_x: int) -> float:
        """
        Measures brightness *ahead* of the cell tip (inside the empty pipette).
        High brightness here indicates a leak/rupture (cytoplasm spraying out).
        """
        offset = self.params.get('rupture_offset_from_tip_px', 10)
        x_end = int(tip_x - offset)
        
        width = self.params.get('rupture_window_width_px', 15)
        x_start = max(0, x_end - width)
        
        if x_end <= x_start: return 0.0 
        
        h, w = image.shape[:2]
        margin = int(h * self.wall_clip_margin)
        y_start, y_end = margin, h - margin
        if y_end <= y_start: y_start, y_end = 0, h
            
        roi = image[y_start:y_end, x_start:x_end]
        if roi.size == 0: return 0.0
        return np.mean(roi)

    def _process_all_frames(self, pipette_start_x: int, threshold: int) -> None:
        """
        Loops through frames to calculate Protrusion Length (Membrane Channel).
        Does NOT handle dye uptake.
        """
        scale_factor = self.params.get('scale_factor', 0.63)
        protrusion_px_list = []
        downstream_int_list = []
        
        for i, image in enumerate(self.roi_images):
            if image is None:
                protrusion_px_list.append(0); downstream_int_list.append(0)
                self.results['debug_images'].append(None); self.results['detection_confidence'].append(0.0)
                continue

            image_8bit = utils.normalize_to_8bit(image)
            gray_image = image_8bit if len(image_8bit.shape) == 2 else cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
            
            # 1. PROTRUSION DETECTION (Membrane Channel)
            # Use original logic: clip walls, clip to the right of pipette start
            mask_prot_clipped = self._segment_mask(
                image, 
                threshold, 
                clip_walls=True, 
                limit_x_max=pipette_start_x
            )
            
            # Measure length
            protrusion_len_px = self._measure_protrusion_from_mask(mask_prot_clipped, pipette_start_x, gray_image)
            protrusion_px_list.append(protrusion_len_px)
            
            # Measure Rupture Intensity
            tip_x = pipette_start_x - protrusion_len_px
            downstream_int_list.append(self._measure_downstream_intensity(gray_image, tip_x, pipette_start_x))

            # 2. DEBUG VISUALIZATION
            debug_image = self._create_debug_visualization(image, mask_prot_clipped, pipette_start_x, protrusion_len_px)
            self.results['debug_images'].append(debug_image)
            self.results['detection_confidence'].append(1.0 if protrusion_len_px > 0 else 0.0)

        # Post-Processing: Smoothing
        if self.params.get('enable_smoothing', True):
            self.results['protrusion_lengths_px'] = self._simple_smoothing_filter(protrusion_px_list)
        else:
            self.results['protrusion_lengths_px'] = protrusion_px_list
            
        self.results['protrusion_lengths_um'] = [p * scale_factor for p in self.results['protrusion_lengths_px']]
        self.results['downstream_intensities'] = downstream_int_list

    def _simple_smoothing_filter(self, values: List[float], window_size: int = 3) -> List[float]:
        """Applies a small Rolling Median filter to smooth out jitter in frame-by-frame data."""
        if len(values) < window_size: return values
        return pd.Series(values).rolling(window=window_size, center=True, min_periods=1).median().tolist()

    def _measure_protrusion_from_mask(self, binary_mask: np.ndarray, pipette_start_x: int, gray_image: Optional[np.ndarray] = None) -> float:
        """Finds the left-most edge of the largest object in the mask and calculates distance from pipette entrance."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        if num_labels <= 1: return 0.0
        
        # Assumes largest blob is the cell
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        edge_x = stats[largest_label, cv2.CC_STAT_LEFT]
        
        # Use sub-pixel refinement if original image is provided
        if gray_image is not None:
             return float(pipette_start_x - self._refine_edge_subpixel(gray_image, edge_x, stats[largest_label]))
        return float(pipette_start_x - edge_x)

    def _create_debug_visualization(self, image: np.ndarray, binary_mask: np.ndarray, pipette_x: int, protrusion_len: float) -> np.ndarray:
        """
        Creates the 'Debug Image' seen in results:
        - Blue overlay: Detected Cell
        - Cyan Line 1: Pipette Entrance
        - Cyan Line 2: Cell Tip
        - Cyan Box: Rupture monitoring zone
        """
        image_8bit = utils.normalize_to_8bit(image)
        debug_img = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        
        # Draw Blue Mask Overlay
        colors = utils.get_color_scheme('blue_red')
        mask_color = utils.hex_to_bgr(colors['fitting_data'])
        overlay = np.zeros_like(debug_img)
        overlay[binary_mask == 255] = mask_color
        debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
        
        # Draw Vertical Lines (Cyan)
        cv2.line(debug_img, (pipette_x, 0), (pipette_x, debug_img.shape[0]), (0, 255, 255), 1)
        tip_x = int(pipette_x - protrusion_len)
        cv2.line(debug_img, (tip_x, 0), (tip_x, debug_img.shape[0]), (0, 255, 255), 1)

        # Draw Wall Limit Lines (White/Black Dashed)
        h, w = debug_img.shape[:2]
        margin = int(h * self.wall_clip_margin)
        if margin > 0:
            cv2.line(debug_img, (0, margin), (w, margin), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(debug_img, (0, h - margin), (w, h - margin), (255, 255, 255), 1, cv2.LINE_AA)
            overlay_lines = debug_img.copy()
            cv2.line(overlay_lines, (0, margin), (w, margin), (0, 0, 0), 1)
            cv2.line(overlay_lines, (0, h - margin), (w, h - margin), (0, 0, 0), 1)
            debug_img = cv2.addWeighted(debug_img, 0.8, overlay_lines, 0.2, 0)
            
        # Draw Rupture Monitoring Box (Cyan)
        offset = self.params.get('rupture_offset_from_tip_px', 10)
        width = self.params.get('rupture_window_width_px', 15)
        
        x_end = int(tip_x - offset)
        x_start = max(0, x_end - width)
        y_start, y_end = margin, h - margin
        
        if x_end > x_start:
            overlay_box = debug_img.copy()
            cv2.rectangle(overlay_box, (x_start, y_start), (x_end, y_end), (0, 165, 255), -1) # Orange
            debug_img = cv2.addWeighted(debug_img, 0.8, overlay_box, 0.2, 0)
            cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (0, 165, 255), 1)
        
        return debug_img

    # =========================================================================
    #                       RUPTURE DETECTION LOGIC
    # =========================================================================
    
    def _detect_rupture_from_haze(self, intensity_trace: List[float]) -> Tuple[bool, Optional[int]]:
        """
        Analyzes the intensity timeline to find sudden, sustained increases in brightness.
        
        Algorithm: CUSUM (Cumulative Sum)
        CUSUM is better than a simple threshold at detecting "drifts" in the mean,
        which is what a slow leak or haze buildup looks like.
        """
        # SAFETY: Need at least a few frames to analyze
        if not intensity_trace or len(intensity_trace) < self.min_cusum_baseline: 
            return False, None

        # 1. READ PARAMETERS
        settling_buffer = self.params.get('cusum_settling_buffer', 3)
        baseline_len_req = self.params.get('cusum_baseline_len', 5)
        sensitivity = self.params.get('cusum_sensitivity_sigma', 0.7413)
        
        # 2. Slice the trace to ignore the entry artifact
        valid_trace = intensity_trace[settling_buffer:]
        if len(valid_trace) < baseline_len_req: return False, None
        
        # 3. Establish Baseline
        #    We must lock in the baseline BEFORE the rupture happens.
        baseline_len = min(baseline_len_req, len(valid_trace) // 2)
        baseline = valid_trace[:baseline_len]
        
        mu = np.median(baseline)
        q75, q25 = np.percentile(baseline, [75 ,25])
        iqr = q75 - q25
        
        # Estimate noise (Sigma)
        sigma = max(iqr * sensitivity, self.min_sigma)
        
        # CUSUM Parameters (now configurable via config.yaml)
        drift_factor = self.params.get('cusum_drift_tolerance_factor', 0.5)
        threshold_factor = self.params.get('cusum_threshold_factor', 10.0)
        
        k = max(self.min_drift, drift_factor * sigma)             # Drift tolerance
        h = max(self.min_cusum_thresh, threshold_factor * sigma)  # Detection threshold
        
        S_pos = 0.0
        start_cand = None
        
        # Scan the valid trace
        for i, x in enumerate(valid_trace):
            deviation = x - mu - k
            if deviation > 0:
                S_pos += deviation
                if start_cand is None: start_cand = i
            else:
                S_pos = max(0, S_pos + deviation)
                if S_pos == 0: start_cand = None
            
            if S_pos > h:
                found_idx = start_cand if start_cand is not None else i
                return True, found_idx + settling_buffer

        return False, None

    def _generate_comprehensive_results(self) -> None:
        """
        Final data aggregation step.
        Determines exactly WHEN to start looking for rupture by analyzing cell entry speed.
        """
        self.results['protrusion_lengths_um'] = [p * self.params.get('scale_factor', 0.63) for p in self.results['protrusion_lengths_px']]
        
        protrusions = self.results['protrusion_lengths_px']
        intensities = self.results['downstream_intensities']
        
        # Calculate velocity (change in length between frames)
        velocity = np.diff(protrusions)
        
        # --- RESIDUE FILTER ---
        # Find the first frame where the protrusion grows by more than ENTRY_VELOCITY_THRESHOLD_PX pixels in one step
        # This ignores static residue (dirt), which has a velocity of ~0.
        entry_idx = next((i for i, v in enumerate(velocity) if v > self.entry_velocity_threshold), 0)
        
        # Pass the trace starting from entry. 
        # The function `_detect_rupture_from_haze` will now apply 
        # an additional "settling buffer" to this slice.
        is_ruptured, local_idx = self._detect_rupture_from_haze(intensities[entry_idx:])
        
        self.results['rupture_detected'] = is_ruptured
        self.results['rupture_frame_index'] = entry_idx + local_idx if is_ruptured and local_idx is not None else None

    def export_results_to_csv(self, output_dir: Union[str, Path], experiment_id: str = "experiment", 
                              dir_full: Optional[Path] = None, dir_filtered: Optional[Path] = None) -> List[Path]:
        """
        Saves two CSV files:
        1. `..._full.csv`: Contains every frame of data.
        2. `..._filtered.csv`: Cuts off data after a rupture is detected (cleaner for plotting).
        """
        frame_interval = self.params.get('frame_interval', 0.2)
        num_frames = len(self.results['protrusion_lengths_px'])
        time_points = [i * frame_interval for i in range(num_frames)]

        df = pd.DataFrame({
            'Frame': list(range(num_frames)),
            'Time_s': time_points,
            'Protrusion_Length_px': self.results['protrusion_lengths_px'],
            'Protrusion_Length_um': self.results['protrusion_lengths_um'],
            'Downstream_Intensity': self.results['downstream_intensities'],
            'Detection_Confidence': self.results['detection_confidence']
        })

        created_files = []
        
        # --- 1. Export Full Data ---
        # Use specific directory if provided, else fallback to main output_dir
        path_full = dir_full if dir_full else Path(output_dir)
        path_full.mkdir(parents=True, exist_ok=True)  
        
        filename_full = f"{experiment_id}_detection_full.csv" 
        full_path = path_full / filename_full
        df.to_csv(full_path, index=False)
        created_files.append(full_path)

        # --- 2. Export Filtered Data ---
        path_filtered = dir_filtered if dir_filtered else Path(output_dir)
        path_filtered.mkdir(parents=True, exist_ok=True) 
        
        filename_filtered = f"{experiment_id}_detection_filtered.csv" # No timestamp
        filtered_path = path_filtered / filename_filtered
        
        if self.results.get('rupture_detected') and self.results.get('rupture_frame_index') is not None:
            rup_idx = self.results['rupture_frame_index']
            df.iloc[:rup_idx+1].to_csv(filtered_path, index=False)
        else:
            df.to_csv(filtered_path, index=False)
        created_files.append(filtered_path)
        
        return created_files