# -*- coding: utf-8 -*-
"""
Automated Detection of Membrane Protrusion in Micropipette Aspiration (MFA)

ROLE IN PIPELINE:
This module is the "eyes" of the analysis. It takes raw image frames and 
outputs the physical length of the cell protrusion over time.

KEY ALGORITHMS:
1.  Adaptive Preprocessing: Uses CLAHE to normalize lighting differences.
2.  Sub-pixel Edge Detection: Uses gradient analysis to measure lengths with 
    accuracy greater than the pixel grid.
3.  CUSUM Rupture Detection: Monitors "haze" intensity to find rupture events.
"""

import logging
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

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
        """
        self.roi_images = roi_images
        self.pipette_coords = pipette_coords
        self.params = params or {}

        # --- EXTRACT SUB-DICTIONARIES ---
        r_params = self.params.get('rupture_detection', {})
        wf_settings = self.params.get('workflow_settings', {})
        img_params = self.params.get('image_processing', {})

        # Interactive Window parameters (Updated to use workflow_settings)
        self.window_scale = wf_settings.get('window_scale_factor', 1.0)
        base_w = wf_settings.get('interactive_window_width', 1000)
        base_h = wf_settings.get('interactive_window_height', 800)
        self.window_width = int(base_w * self.window_scale)
        self.window_height = int(base_h * self.window_scale)
        
        # --- Rupture Parameters (Read from r_params) ---
        self.entry_velocity_threshold = r_params.get('entry_velocity_threshold_px', 2.0)
        self.rupture_offset = r_params.get('rupture_offset_from_tip_px', 5)
        self.rupture_width = r_params.get('rupture_window_width_px', 5)       
        
        # 1. Spike (Transient Burst)
        self.enable_spike = r_params.get('enable_spike_check', True)
        self.spike_sigma = r_params.get('spike_sigma_threshold', 6.0)

        # 2. Step (Fast Leak)
        self.enable_step = r_params.get('enable_step_check', True)
        self.step_sigma = r_params.get('step_sigma_threshold', 6.0)

        # 3. CUSUM (Slow Drift)
        self.min_cusum_baseline = r_params.get('min_cusum_baseline_frames', 5)
        self.min_intensity_noise_floor = r_params.get('min_intensity_noise_floor', 1.0)
        
        self.cusum_drift_tol = r_params.get('cusum_drift_tolerance_factor', 0.5)
        self.cusum_thresh_fac = r_params.get('cusum_threshold_factor', 8.0)
        self.min_drift = r_params.get('min_drift_tolerance', 0.2)
        self.min_cusum_thresh = r_params.get('min_cusum_threshold', 2.0)
        
        # Processing Parameters (Updated to use image_processing)
        self.min_area_threshold = img_params.get('min_area_threshold', 100)         
        self.small_object_threshold = img_params.get('small_object_threshold', 50)  
        self.wall_clip_margin = img_params.get('wall_clip_margin', 0.40)         

        # --- Results Container ---
        self.results: Dict[str, Any] = {
            # Pre-processing
            'signal': 'stop',               # Flow control signal (confirm/restart/stop)
            'protrusion_lengths_px': [],    # Raw length data in pixels
            'protrusion_lengths_um': [],    # Converted length in microns
            'total_area_um2': [],           # Total cell area
            'body_solidity': [],            # Cell body solidity
            'entry_frame_index': None,
            # Rupture Detection
            'downstream_intensities': [],   # Brightness values inside the pipette (for rupture)'
            'rupture_detected': False,      # Boolean flag for rupture event
            'rupture_frame_index': None,    # Frame index where rupture occurred
            'rupture_time': None,           # Time in seconds
            'debug_images': [],             # Visualizations with overlays
            'rupture_reason': None,
            # Protrusion Detection
            'pipette_start_x_used': None,   # The X-coordinate used as "Zero"
            'threshold_prot': None,         # Primary threshold (clipped)
            'threshold_body': None,         # Secondary threshold (full height)
            # Data
            'processing_metadata': {},      # Extra info
            'analysis_timestamp': datetime.now().isoformat(),
            'detection_confidence': [],     # Metric 0.0 or 1.0 indicating tracking success
            'r_eff': 0.0,                   # Placeholder for calculation results
            'time_seconds': []
        }
        
        # Expose debug frames for preview
        self.debug_frames = []
        
    def run(self) -> Dict[str, Any]:
        """
        Facade method that routes to either interactive or automatic detection
        based on the 'verify_traps_interactively' setting in params.
        """
        is_interactive = self.params.get('workflow_settings', {}).get('verify_traps_interactively', True)
        
        if is_interactive:
            return self.run_detection()
        else:
            return self.run_automatic()
        
    def run_automatic(self) -> Dict[str, Any]:
        """
        Runs detection without user intervention (Batch Mode).
        Automatically calculates thresholds using Otsu's method on the middle frame.
        """
        if not self.roi_images or all(frame is None for frame in self.roi_images):
            self.results['signal'] = 'stop'
            return self.results

        # 1. Select Reference Frame (Middle of sequence is usually best)
        mid_idx = len(self.roi_images) // 2
        ref_image = self.roi_images[mid_idx] if self.roi_images[mid_idx] is not None else self.roi_images[0]
        
        if ref_image is None:
            self.results['signal'] = 'stop'
            return self.results

        # 2. Determine Pipette Start X
        # Use the value passed in __init__ (from setup phase)
        # Ensure it's within bounds
        pipette_x = self.pipette_coords[0]
        w = ref_image.shape[1]
        pipette_x = max(1, min(w - 1, pipette_x))
        
        # 3. Determine Threshold (Otsu)
        # Extract ROI for stats (similar to interactive mode logic)
        img_8bit = utils.normalize_to_8bit(ref_image)
        h, w = img_8bit.shape[:2]
        margin = int(h * self.wall_clip_margin)
        
        # Use protrusion area for statistics
        if margin > 0:
            roi_stats = img_8bit[margin:h-margin, :pipette_x]
        else:
            roi_stats = img_8bit[:, :pipette_x]
            
        if roi_stats.size > 0:
            otsu_val, _ = cv2.threshold(roi_stats, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold_prot = int(otsu_val)
        else:
            threshold_prot = 50 # Fallback
            
        logger.debug(f"Auto-calculated detection threshold: {threshold_prot}")
        
        # 4. Run Detection
        return self.run_detection_with_parameters(pipette_x, threshold_prot)

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
        # Alias debug_images to debug_frames for consistency with Plotting/Preview
        self.debug_frames = self.results['debug_images']
        
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
        self.debug_frames = self.results['debug_images']
        
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
        # Safety check for coordinates
        if not (0 <= current_pipette_start_x < image_width): current_pipette_start_x = image_width // 2

        window_name = 'Pipette Entrance Positioning'
        utils.create_centered_window(window_name, self.window_width, self.window_height)

        while True:
            display_image = utils.prepare_display_image(test_image, self.window_width, self.window_height)
            h, w = display_image.shape[:2]
            original_h, original_w = test_image.shape[:2]
            
            # Scale logic to ensure overlay matches mouse/drawing coordinates
            if original_h > 0 and original_w > 0:
                 display_scale = min(self.window_width / original_w, self.window_height / original_h) * 0.9
            else:
                 display_scale = 1.0
                 
            # Calculate display offset (centering)
            # The prepare_display_image function centers the image on a black canvas
            new_w = int(original_w * display_scale) if original_w > 0 else 100
            x_offset = (self.window_width - new_w) // 2
            
            display_pipette_x = x_offset + int(current_pipette_start_x * display_scale)
            
            # Use Pipette Color (Dark Blue)
            pip_color = utils.get_ui_color('pipette')
            cv2.line(display_image, (display_pipette_x, 0), (display_pipette_x, h), pip_color, 3)
            
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
                "W/S: +/- 10 | A/D: +/- 1",
                "ENTER: Confirm | R: Restart"
            ])
            cv2.imshow(window_name, combined_display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 13: cv2.destroyAllWindows(); return 'confirm', current_threshold
            if key == 27: cv2.destroyAllWindows(); return 'stop', None
            if key in (ord('r'), ord('R')): cv2.destroyAllWindows(); return 'restart', None
            
            # Keyboard controls update the shared state
            if key in (ord('w'), ord('W')): state['threshold'] = min(255, current_threshold + 10)
            elif key in (ord('s'), ord('S')): state['threshold'] = max(0, current_threshold - 10)
            elif key in (ord('d'), ord('D')): state['threshold'] = min(255, current_threshold + 1)
            elif key in (ord('a'), ord('A')): state['threshold'] = max(0, current_threshold - 1)

    # Visualization Helper Methods

    def _create_overlay_panel(self, image: np.ndarray, binary_mask: np.ndarray, pipette_start_x: int, clip_walls: bool, width: int, height: int) -> np.ndarray:
        """
        Helper to draw the cyan overlay for threshold selection.
        Returns an image formatted for the display window.
        """
        image_8u = utils.normalize_to_8bit(image)
        display_base = cv2.cvtColor(image_8u, cv2.COLOR_GRAY2BGR)
        
        # 1. Draw Threshold Mask (Dark Red)
        # This corresponds to 'ui_mask' in the palette
        mask_color = utils.get_ui_color('mask')
        overlay = np.zeros_like(display_base, dtype=np.uint8)
        overlay[binary_mask == 255] = mask_color
        blended = cv2.addWeighted(display_base, 0.7, overlay, 0.3, 0)
        
        # 2. Draw Pipette Line (Dark Blue)
        pip_color = utils.get_ui_color('pipette')
        safe_x = max(0, min(image.shape[1], int(pipette_start_x)))
        cv2.line(blended, (safe_x, 0), (safe_x, blended.shape[0]), pip_color, 2)
        
        # 3. Draw Wall Clip Lines (Medium Blue - Same as ROI/Guides)
        if clip_walls:
            guide_color = utils.get_ui_color('guide')
            h = blended.shape[0]
            margin = int(h * self.wall_clip_margin)
            if margin > 0:
                cv2.line(blended, (0, margin), (blended.shape[1], margin), guide_color, 1)
                cv2.line(blended, (0, h-margin), (blended.shape[1], h-margin), guide_color, 1)

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
            pt1 = (int((i-1) * bin_w), height - 40 - int(hist[i-1]))
            pt2 = (int(i * bin_w), height - 40 - int(hist[i]))
            cv2.line(hist_img, pt1, pt2, (200, 200, 200), 2)
        
        # Draw threshold line 
        guide_color = utils.get_ui_color('guide')
        x_thresh = int(threshold_line * width / 255)
        cv2.line(hist_img, (x_thresh, 0), (x_thresh, height - 40), guide_color, 2)
        
        utils.draw_ui_text(hist_img, f"T={threshold_line}", (x_thresh + 5, 30), scale=0.6, color=guide_color)
        
        # Axis labels
        cv2.putText(hist_img, "Intensity (0-255)", (width//3, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
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
        
        # 1. Enhance & Blur (Updated to use image_processing parameters)
        image_params = self.params.get('image_processing', {})
        clahe = cv2.createCLAHE(
            clipLimit=image_params.get('clahe_clip_limit', 2.0), 
            tileGridSize=tuple(image_params.get('clahe_tile_grid_size', (8, 8)))
        )
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(
            enhanced, 
            tuple(image_params.get('gaussian_kernel_size', (5, 5))), 
            0
        )
        
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
        """
        Removes blobs touching the left edge of the image (wall/border artifacts).

        The trick: OpenCV's floodFill needs a seed pixel that is already foreground (255)
        to spread and paint things background (0). Rather than iterating the actual
        image border (which might be background), we *pad* a 1-pixel-wide foreground
        column onto the left edge. This guarantees every left-border pixel is 255,
        so floodFill can reach and erase any blob that touches the true edge.
        The padding column is then stripped before returning.

        Why not just zero-out the left column directly? That would only delete the
        border pixel itself, not the entire connected blob attached to it.
        """
        padded = np.pad(binary_image, ((0, 0), (1, 0)), mode='constant', constant_values=255)
        h, w = padded.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for y in range(h):
            if padded[y, 0] == 255:
                cv2.floodFill(padded, mask, (0, y), 0)
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
        
        # Vectorized Search
        edge_indices = np.where(profile < threshold)[0]
        if edge_indices.size == 0:
            return float(int_edge_x)
            
        # The rightmost index where signal drops below threshold
        edge_idx = edge_indices[-1]
        
        # Linear interpolation for sub-pixel cross point
        if edge_idx < len(profile) - 1:
            v_low, v_high = profile[edge_idx], profile[edge_idx + 1]
            fraction = (threshold - v_low) / (v_high - v_low) if v_high != v_low else 0.5
            offset = edge_idx + fraction
        else:
            offset = float(edge_idx)
            
        return max(0.0, min(float(w), x_start + offset))

    def _process_all_frames(self, pipette_start_x: int, threshold: int) -> None:
        """
        Loops through frames to calculate Protrusion Length (Membrane Channel).
        Does NOT handle dye uptake.
        """
        # Updated to extract scale factor using proper nesting
        scale_factor = self.params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        protrusion_px_list = []
        downstream_int_list = []
        
        for i, image in enumerate(self.roi_images):
            if image is None:
                protrusion_px_list.append(0); downstream_int_list.append(0)
                self.results['total_area_um2'].append(0.0)    
                self.results['body_solidity'].append(0.0)     
                self.results['debug_images'].append(None)
                self.results['detection_confidence'].append(0.0)
                continue

            image_8bit = utils.normalize_to_8bit(image)
            gray_image = image_8bit if len(image_8bit.shape) == 2 else cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
            
            # 1. GENERATE MASKS & PROTRUSION DETECTION
            thr_prot = self.results.get('threshold_prot', threshold)
            thr_body = self.results.get('threshold_body', threshold)
            
            # A) Generate standard masks (This correctly captures the body without deleting it)
            _, mask_body_standard = utils.generate_dual_masks(
                image, pipette_start_x, thr_prot, thr_body, self.params
            )
            
            # B) Generate the highly-tuned protrusion mask using your internal segmenter
            mask_prot= self._segment_mask(
                image, 
                threshold, 
                clip_walls=True, 
                limit_x_max=pipette_start_x
            )
            
            # C) Combine for true total cell calculations
            mask_total = cv2.bitwise_or(mask_prot, mask_body_standard)
            
            # Measure length and area
            protrusion_len_px, cell_area = self._measure_protrusion_from_mask(mask_prot, pipette_start_x, gray_image)
            protrusion_px_list.append(protrusion_len_px)
            
            # Measure Morphology (Total Area & Body Solidity) using the un-erased body mask
            total_area, body_solidity = self._calculate_morphology(mask_total, mask_body_standard)
            self.results['total_area_um2'].append(total_area)
            self.results['body_solidity'].append(body_solidity)
            
            # Measure Rupture Intensity
            tip_x = pipette_start_x - protrusion_len_px
            downstream_int_list.append(self._measure_downstream_intensity(gray_image, tip_x, pipette_start_x))
            
            # 3. DEBUG VISUALIZATION
            debug_image = self._create_debug_visualization(image, mask_prot, pipette_start_x, protrusion_len_px)
            self.results['debug_images'].append(debug_image)
            self.results['detection_confidence'].append(1.0 if protrusion_len_px > 0 else 0.0)

        # Post-Processing: Smoothing (Updated to use rupture_detection parameters)
        if self.params.get('rupture_detection', {}).get('enable_smoothing', True):
            self.results['protrusion_lengths_px'] = self._simple_smoothing_filter(protrusion_px_list)
        else:
            self.results['protrusion_lengths_px'] = protrusion_px_list
            
        self.results['protrusion_lengths_um'] = [p * scale_factor for p in self.results['protrusion_lengths_px']]
        self.results['downstream_intensities'] = downstream_int_list

    def _simple_smoothing_filter(self, values: List[float], window_size: int = 3) -> List[float]:
        """Applies a small Rolling Median filter to smooth out jitter in frame-by-frame data."""
        if len(values) < window_size: return values
        return pd.Series(values).rolling(window=window_size, center=True, min_periods=1).median().tolist()

    def _measure_protrusion_from_mask(self, binary_mask: np.ndarray, pipette_start_x: int, gray_image: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Finds the left-most edge of the largest object in the mask and calculates distance from pipette entrance."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        if num_labels <= 1: return 0.0, 0.0
        
        # Assumes largest blob is the cell
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        edge_x = stats[largest_label, cv2.CC_STAT_LEFT]
        area = float(stats[largest_label, cv2.CC_STAT_AREA])
        
        # Use sub-pixel refinement if original image is provided
        if gray_image is not None:
             return float(pipette_start_x - self._refine_edge_subpixel(gray_image, edge_x, stats[largest_label])), area
        return float(pipette_start_x - edge_x), area

    def _create_debug_visualization(self, image: np.ndarray, binary_mask: np.ndarray, pipette_x: int, protrusion_len: float) -> np.ndarray:
        """
        Creates the 'Debug Image' seen in results.
        Includes transparent wall lines and high-contrast tip tracking.
        """
        image_8bit = utils.normalize_to_8bit(image)
        debug_img = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        
        # 1. Overlay (Light Blue - Tertiary)
        mask_color = utils.get_bgr_color('tertiary')
        overlay = np.zeros_like(debug_img)
        overlay[binary_mask == 255] = mask_color
        debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
        
        # 2. Draw Wall Limit Lines (TRANSPARENT/DIM)
        h, w = debug_img.shape[:2]
        margin = int(h * self.wall_clip_margin)
        if margin > 0:
            # Create a separate layer for lines
            line_overlay = debug_img.copy()
            white = (255, 255, 255)
            cv2.line(line_overlay, (0, margin), (w, margin), white, 1, cv2.LINE_AA)
            cv2.line(line_overlay, (0, h - margin), (w, h - margin), white, 1, cv2.LINE_AA)
            
            # Blend lines to make them transparent (0.4 alpha)
            debug_img = cv2.addWeighted(debug_img, 0.6, line_overlay, 0.4, 0)
        
        # 3. Pipette Line (Dark Blue - Pipette)
        c_pip = utils.get_ui_color('pipette')
        cv2.line(debug_img, (pipette_x, 0), (pipette_x, debug_img.shape[0]), c_pip, 1)
        
        # 4. Detected Tip (Medium Red - Secondary)
        c_tip = utils.get_ui_color('pipette')
        tip_x = int(pipette_x - protrusion_len)
        cv2.line(debug_img, (tip_x, 0), (tip_x, debug_img.shape[0]), c_tip, 1)

        # 5. Rupture Monitoring Box (Dark Red - Mask)
        c_rupture = utils.get_ui_color('mask')
                
        x_end = int(tip_x - self.rupture_offset)
        x_start = max(0, x_end - self.rupture_width)
        y_start, y_end = margin, h - margin
        
        if x_end > x_start:
            # Draw semi-transparent fill
            box_overlay = debug_img.copy()
            cv2.rectangle(box_overlay, (x_start, y_start), (x_end, y_end), c_rupture, -1)
            debug_img = cv2.addWeighted(debug_img, 0.7, box_overlay, 0.3, 0)
            
            # Draw solid border
            cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), c_rupture, 1)
        
        return debug_img
    
    
    def _calculate_morphology(self, mask_total: np.ndarray, mask_body: np.ndarray) -> Tuple[float, float]:
        """Calculates total cell area and body-only solidity with robust fallbacks."""
        scale_factor = self.params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        
        # 1. Total Area
        total_area_px = cv2.countNonZero(mask_total)
        total_area_um2 = total_area_px * (scale_factor ** 2)
        
        # 2. Body Solidity
        mask_body_u8 = mask_body.astype(np.uint8)
        contours, _ = cv2.findContours(mask_body_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return total_area_um2, 0.0
            
        cnt = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(cnt)
        
        # Fallback to pixel count if geometric area is 0
        if area_px == 0:
            area_px = cv2.countNonZero(mask_body_u8)
            
        if len(cnt) >= 3: # Convex hull requires at least 3 points
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = float(area_px) / hull_area
                return total_area_um2, solidity
                
        return total_area_um2, 0.0

    # =========================================================================
    #                       RUPTURE DETECTION LOGIC
    # =========================================================================
    
    def _generate_comprehensive_results(self) -> None:
        """
        Runs multiple detectors to find the FIRST failure point.
        Includes logic to handle 'Already Inside' cases (Trap 2, 14).
        Note: protrusion_lengths_um is already set by _process_all_frames; no recalculation needed.
        """
        protrusions_um = self.results['protrusion_lengths_um']
        intensities = self.results['downstream_intensities']
        
        # --- 1. Determine Entry Point & Candidates ---
        candidates = []
        
        if intensities:
            baseline_mean = np.median(intensities[:5])
            # Use config threshold
            abs_threshold = self.params.get('rupture_detection', {}).get('absolute_intensity_threshold', 6.5)
            if baseline_mean >= abs_threshold: 
                 candidates.append((0, 'Immediate High Haze (DOA)'))

        start_len = protrusions_um[0] if protrusions_um else 0
        
        if start_len > 20.0:
            entry_idx = 0
        else:
            velocity = np.diff(self.results['protrusion_lengths_px'], prepend=0)
            entry_idx = next((i for i, v in enumerate(velocity) if v > self.entry_velocity_threshold), 0)
            if entry_idx >= len(protrusions_um): entry_idx = 0
            
        self.results['entry_frame_index'] = entry_idx
        valid_intensities = intensities[entry_idx:]

        # B. Spike (Transient Burst)
        if self.enable_spike:
            is_spike, spike_idx = self._detect_intensity_anomaly(valid_intensities, mode='spike')
            if is_spike: candidates.append((entry_idx + spike_idx, 'Intensity Spike'))
        
        # C. Step (Fast Leak)
        if self.enable_step:
            is_step, step_idx = self._detect_intensity_anomaly(valid_intensities, mode='step')
            if is_step: candidates.append((entry_idx + step_idx, 'Intensity Step'))

        # D. CUSUM (Slow Drift)
        is_cusum, cusum_idx = self._detect_cusum_drift(valid_intensities)
        if is_cusum: candidates.append((entry_idx + cusum_idx, 'Haze Drift'))
        
        # E. LATE SAFETY CHECK
        if not candidates and valid_intensities:
            final_mean = np.mean(valid_intensities[-5:])
            if final_mean > 25.0:
                grad = np.gradient(valid_intensities)
                max_grad_idx = np.argmax(grad)
                candidates.append((entry_idx + max_grad_idx, 'Late High Haze'))

        # Arbitrate: Pick the earliest detected event
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_idx, reason = candidates[0]
            self.results['rupture_detected'] = True
            self.results['rupture_frame_index'] = best_idx
            self.results['rupture_reason'] = reason
        else:
            self.results['rupture_detected'] = False
            self.results['rupture_frame_index'] = None
            self.results['rupture_reason'] = None
        
    def _detect_intensity_anomaly(self, trace: List[float], mode: str = 'spike') -> Tuple[bool, Optional[int]]:
        """
        Unified detector for Spikes (transient) and Steps (sustained).
        """
        r_params = self.params.get('rupture_detection', {})
        
        settling = r_params.get('cusum_settling_buffer', 3)
        baseline_len = r_params.get('cusum_baseline_len', 5)
        lag = r_params.get('cusum_baseline_lag', 0)
        
        min_len = settling + baseline_len + lag + 1
        if len(trace) < min_len: return False, None
        
        threshold_sigma = self.spike_sigma if mode == 'spike' else self.step_sigma
        
        for i in range(settling + baseline_len + lag, len(trace)):
            b_end = i - lag
            b_start = b_end - baseline_len
            
            baseline = trace[b_start : b_end]
            median = np.median(baseline)
            
            q75, q25 = np.percentile(baseline, [75, 25])
            iqr = q75 - q25
            
            raw_sigma = iqr * 0.7413
            max_sigma = r_params.get('max_baseline_sigma', 1.0)
            capped_sigma = min(raw_sigma, max_sigma)
            sigma = max(capped_sigma, self.min_intensity_noise_floor)
            
            current_val = trace[i]
            deviation = current_val - median
            
            if deviation > (threshold_sigma * sigma):
                return True, i
                
        return False, None

    def _detect_cusum_drift(self, trace: List[float]) -> Tuple[bool, Optional[int]]:
        """
        Dynamic CUSUM Control Chart.
        Uses a rolling baseline to prevent initial entry vibration from permanently inflating 
        the detection threshold.
        """
        r_params = self.params.get('rupture_detection', {})
        settling = r_params.get('cusum_settling_buffer', 3)
        baseline_len = r_params.get('cusum_baseline_len', 5)
        lag = r_params.get('cusum_baseline_lag', 0)
        
        min_len = settling + baseline_len + lag + 1
        if len(trace) < min_len: 
            return False, None
        
        S_pos = 0.0
        
        for i in range(settling + baseline_len + lag, len(trace)):
            # 1. Establish Rolling Baseline
            b_end = i - lag
            b_start = b_end - baseline_len
            baseline = trace[b_start : b_end]
            
            # 2. Calculate Local Statistics
            mu = np.median(baseline)
            q75, q25 = np.percentile(baseline, [75, 25])
            iqr = q75 - q25
            
            # Prevent Sigma Inflation
            raw_sigma = iqr * 0.7413
            max_sigma = r_params.get('max_baseline_sigma', 1.0)
            capped_sigma = min(raw_sigma, max_sigma)
            sigma = max(capped_sigma, self.min_intensity_noise_floor)
            
            # 3. Dynamic Thresholds
            k = max(self.min_drift, self.cusum_drift_tol * sigma)
            h = max(self.min_cusum_thresh, self.cusum_thresh_fac * sigma)
            
            # 4. Accumulate Deviation
            deviation = trace[i] - mu - k
            if deviation > 0:
                S_pos += deviation
            else:
                S_pos = 0.0 # Reset accumulation if signal drops back to noise floor
            
            if S_pos > h:
                return True, i
                
        return False, None

    def _measure_downstream_intensity(self, image: np.ndarray, tip_x: float, pipette_x: int) -> float:
        """
        Measures the mean brightness in a small window *ahead* of the cell tip.
        Uses the settings loaded in __init__.
        """        
        x_end = int(tip_x - self.rupture_offset)
        x_start = max(0, x_end - self.rupture_width)
        
        if x_end <= x_start: return 0.0
        
        h, w = image.shape[:2]
        margin = int(h * self.wall_clip_margin)
        roi = image[max(0, margin):min(h, h-margin), max(0, x_start):min(w, x_end)]
        return np.mean(roi) if roi.size > 0 else 0.0
    
    def export_results_to_csv(self, output_dir: Union[str, Path], experiment_id: str = "experiment", 
                              dir_full: Optional[Path] = None, dir_filtered: Optional[Path] = None) -> List[Path]:
        """
        Saves two CSV files:
        1. `..._full.csv`: Contains every frame of data.
        2. `..._filtered.csv`: Cuts off data after a rupture is detected (cleaner for plotting).
        """
        frame_interval = self.params.get('experiment_parameters', {}).get('frame_interval', 0.2)
        num_frames = len(self.results['protrusion_lengths_px'])
        time_points = [i * frame_interval for i in range(num_frames)]

        df = pd.DataFrame({
            'Frame': list(range(num_frames)),
            'Time_s': time_points,
            'Protrusion_Length_px': self.results['protrusion_lengths_px'],
            'Protrusion_Length_um': self.results['protrusion_lengths_um'],
            'Total_Area_um2': self.results['total_area_um2'],    
            'Body_Solidity': self.results['body_solidity'],      
            'Downstream_Intensity': self.results['downstream_intensities'],
            'Detection_Confidence': self.results['detection_confidence']
        })

        created_files = []
        
        # --- 1. Export Full Data ---
        path_full = dir_full if dir_full else Path(output_dir)
        path_full.mkdir(parents=True, exist_ok=True)  
        
        filename_full = f"{experiment_id}_detection_full.csv" 
        full_path = path_full / filename_full
        df.to_csv(full_path, index=False)
        created_files.append(full_path)

        # --- 2. Export Filtered Data ---
        path_filtered = dir_filtered if dir_filtered else Path(output_dir)
        path_filtered.mkdir(parents=True, exist_ok=True) 
        
        filename_filtered = f"{experiment_id}_detection_filtered.csv" 
        filtered_path = path_filtered / filename_filtered
        
        if self.results.get('rupture_detected') and self.results.get('rupture_frame_index') is not None:
            rup_idx = self.results['rupture_frame_index']
            df.iloc[:rup_idx+1].to_csv(filtered_path, index=False)
        else:
            df.to_csv(filtered_path, index=False)
        created_files.append(filtered_path)
        
        return created_files