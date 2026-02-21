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

        # GUV mode: read fill_membrane_holes from guv_settings (master location).
        # Falls back to False so standard cell experiments are unaffected.
        guv_settings = self.params.get('guv_settings', {})
        self.fill_membrane_holes = (
            guv_settings.get('enable', False) and
            guv_settings.get('fill_membrane_holes', True)
        )

        # Interactive Window parameters (Updated to use workflow_settings)
        self.window_scale = wf_settings.get('window_scale_factor', 1.0)
        base_w = wf_settings.get('interactive_window_width', 1000)
        base_h = wf_settings.get('interactive_window_height', 800)
        self.window_width = int(base_w * self.window_scale)
        self.window_height = int(base_h * self.window_scale)
        
        # --- Rupture Parameters (Read from r_params) ---
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
        
        # Processing Parameters
        self.min_area_threshold = img_params.get('min_area_threshold', 100)         
        self.small_object_threshold = img_params.get('small_object_threshold', 50) 
        self.wall_clip_margin = img_params.get('wall_clip_margin', 0.25)         

        # --- Results Container ---
        self.results: Dict[str, Any] = {
            # Pre-processing
            'signal': 'stop',               # Flow control signal (confirm/restart/stop)
            'protrusion_lengths_px': [],    # Raw length data in pixels
            'protrusion_lengths_um': [],    # Converted length in microns
            'protrusion_area_um2': [],      # Protrusion Area
            'body_area_um2': [],            # Body Area
            'total_area_um2': [],           # Total cell area
            'body_solidity': [],            # Cell body solidity
            'entry_frame_index': None,      # Cell enters trap
            'exit_frame_index': None,       # Cell Exit Handling
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
            'time_seconds': [],
            'frame_masks': [],              # Per-frame (mask_prot, mask_body) tuples for downstream use
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
        pipette_x = self.pipette_coords[0]
        w = ref_image.shape[1]
        pipette_x = max(1, min(w - 1, pipette_x))
        
        # 3. Determine Threshold (Otsu)
        img_8bit = utils.normalize_to_8bit(ref_image)
        h, w = img_8bit.shape[:2]
        margin = int(h * self.wall_clip_margin)
        
        # Protrusion threshold
        roi_stats = img_8bit[margin:h-margin, :pipette_x] if margin > 0 else img_8bit[:, :pipette_x]
        threshold_prot = int(cv2.threshold(roi_stats, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]) if roi_stats.size > 0 else 50
            
        # Body threshold (Ensure this always runs for morphology)
        roi_body = img_8bit[:, pipette_x:]
        threshold_body = int(cv2.threshold(roi_body, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]) if roi_body.size > 0 else 50
            
        logger.debug(f"Auto-calculated thresholds: Prot={threshold_prot}, Body={threshold_body}")
        
        self.results['threshold_body'] = threshold_body
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
        
        # --- Step 3: Cell Body Threshold  ---
        # Separate threshold for the body (Full ROI Height)
        logger.info("Step 3: Cell Body Threshold (Full ROI Height)")
        signal, thr_body = self._interactive_threshold_selection(test_image, pipette_start_x, clip_walls=False)
        if signal in ('restart', 'stop'):
            self.results['signal'] = signal
            return self.results
            self.results['threshold_body'] = thr_body        
            
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
        """Standard morphological cleaning for masks.
        
        For GUVs, the membrane signal is ring-like: bright rim, dark interior.
        A simple threshold produces a broken ring, not a filled disc.
        When fill_membrane_holes is True, we find the outer contour of each blob
        and draw it as a solid filled shape, which fills the lumen regardless
        of how dark or patchy it is.
        """
        # --- Step 1: Remove tiny specks (noise) ---
        # connectedComponentsWithStats labels every separate white blob and gives us
        # statistics about each one (position, area, etc.). We keep only blobs that
        # are at least min_area_threshold pixels in size.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        filtered = np.zeros_like(binary_mask)
        for i in range(1, num_labels):  # start at 1 to skip the background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_threshold:
                filtered[labels == i] = 255
    
        # --- Step 2: Morphological close (bridges small gaps) ---
        # This is a two-step operation: dilate (grow) then erode (shrink).
        # Net effect: small gaps in the membrane ring get bridged.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filled = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close)
    
        # --- Step 3 (GUV mode): Fill the interior completely ---
        # For cells with a hollow lumen (GUVs), the close above is not enough.
        # We find the outer contour of each blob and draw it as a solid filled
        # polygon. This works even when the interior is completely dark, because
        # it doesn't rely on pixel intensity at all — only the outer boundary shape.
        # GUV mode: fill the hollow lumen if enabled in guv_settings.
        guv_settings = self.params.get('guv_settings', {})
        if guv_settings.get('enable', False) and guv_settings.get('fill_membrane_holes', True):
            filled = self._fill_contour_interiors(filled)
    
        # --- Step 4: Remove any blob touching the left image border ---
        # These are usually the channel walls leaking into the ROI, not the cell.
        cleaned = self._remove_left_border_objects(filled)
    
        # --- Step 5: Final gentle cleanup ---
        # The open operation (erode then dilate) removes thin spurs and tiny specks
        # that survived Step 1, without shrinking the main blob.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    
        return opened

    def _fill_contour_interiors(self, binary_mask: np.ndarray) -> np.ndarray:
        """Fill all enclosed holes inside detected blobs.

        How it works (the 'flood from outside' trick):
            1. Invert the mask so background = white (255), cell blobs = black (0).
            2. Flood-fill starting from the top-left corner (guaranteed exterior).
               This paints all connected exterior background a temporary grey (128).
            3. After flooding, anything still white (255) in the inverted image is
               an interior hole — surrounded by cell, unreachable from outside.
            4. Invert back. Those interior holes become white and get OR'd into
               the original mask, filling them.

        This is more reliable than filling via contours when the membrane is very
        patchy, because it makes no assumption about contour shape.
        """
        # Step 1: invert — background (0) becomes white (255), blobs become black
        inverted = cv2.bitwise_not(binary_mask)

        # Step 2: flood fill from the top-left corner outward
        # The mask argument to floodFill must be 2 pixels larger in each dimension
        h, w = inverted.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        # We paint the exterior with grey (128) so we can identify it later
        cv2.floodFill(inverted, flood_mask, seedPoint=(0, 0), newVal=128)

        # Step 3: anything still white (255) in 'inverted' is an interior hole
        interior_holes = (inverted == 255).astype(np.uint8) * 255

        # Step 4: add the holes back into the original mask
        result = cv2.bitwise_or(binary_mask, interior_holes)
        return result

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
        """
        # Updated to extract scale factor using proper nesting
        scale_factor = self.params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        protrusion_px_list = []
        downstream_int_list = []
        
        for i, image in enumerate(self.roi_images):
            if image is None:
                protrusion_px_list.append(0); downstream_int_list.append(0)
                self.results['protrusion_area_um2'].append(0.0)
                self.results['body_area_um2'].append(0.0)
                self.results['total_area_um2'].append(0.0)    
                self.results['body_solidity'].append(0.0)     
                self.results['debug_images'].append(None)
                self.results['detection_confidence'].append(0.0)
                self.results['frame_masks'].append((None, None))
                continue

            image_8bit = utils.normalize_to_8bit(image)
            gray_image = image_8bit if len(image_8bit.shape) == 2 else cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)
            
            # 1. GENERATE MASKS & PROTRUSION DETECTION
            thr_prot = self.results.get('threshold_prot', threshold)
            # 'threshold_body' is initialised to None and only gets a real
            # value during interactive runs.  dict.get(key, default) does NOT
            # fire the default when the key exists but holds None — so we use
            # 'or' to fall back to thr_prot in non-interactive batch mode.
            thr_body = self.results.get('threshold_body') or threshold
            
            # A) Generate masks for cell body, protrusion, and total cell
            _, mask_body_standard = utils.generate_dual_masks(image, pipette_start_x, thr_prot, thr_body, self.params)
            mask_prot= self._segment_mask(image, threshold, clip_walls=True, limit_x_max=pipette_start_x)
            mask_total = cv2.bitwise_or(mask_prot, mask_body_standard)
            
            # Measure length and area
            protrusion_len_px, _ = self._measure_protrusion_from_mask(mask_prot, pipette_start_x, gray_image)
            protrusion_px_list.append(protrusion_len_px)
            
            # Measure Morphology (Total Area & Body Solidity) using the un-erased body mask
            a_prot, a_body, a_tot, solidity = self._calculate_morphology(mask_prot, mask_body_standard, mask_total)
            self.results['protrusion_area_um2'].append(a_prot)
            self.results['body_area_um2'].append(a_body)
            self.results['total_area_um2'].append(a_tot)
            self.results['body_solidity'].append(solidity)
            
            # Measure Rupture Intensity
            tip_x = pipette_start_x - protrusion_len_px
            downstream_int_list.append(self._measure_downstream_intensity(gray_image, tip_x, pipette_start_x))
            
            # 3. DEBUG VISUALIZATION
            debug_image = self._create_debug_visualization(image, mask_prot, pipette_start_x, protrusion_len_px)
            self.results['debug_images'].append(debug_image)
            self.results['detection_confidence'].append(1.0 if protrusion_len_px > 0 else 0.0)
            self.results['frame_masks'].append((mask_prot, mask_body_standard))

        # Post-Processing: Smoothing
        if self.params.get('rupture_detection', {}).get('enable_smoothing', True):
            self.results['protrusion_lengths_px'] = self._simple_smoothing_filter(protrusion_px_list)
        else:
            self.results['protrusion_lengths_px'] = protrusion_px_list
        
        # --- CONVERT TO MICRONS BEFORE ENTRY/EXIT DETECTION ---
        self.results['protrusion_lengths_um'] = [p * scale_factor for p in self.results['protrusion_lengths_px']]
        
        # --- ENTRY AND EXIT SYNCHRONIZATION ---
        protrusions = self.results['protrusion_lengths_um']
        entry_thresh = self.params.get('rupture_detection', {}).get('entry_protrusion_threshold_um', 0.5)
        exit_thresh = self.params.get('rupture_detection', {}).get('exit_protrusion_threshold_um', 0.5)
        exit_ratio = self.params.get('rupture_detection', {}).get('exit_drop_ratio', 0.2)
        
        # 1. Entry: Target the frame prior to initial detection spike
        entry_idx = 0
        for i, p in enumerate(protrusions):
            if p > entry_thresh:
                entry_idx = max(0, i - 1)
                break
        self.results['entry_frame_index'] = entry_idx
        
        # 2. Exit: Detect cell slip or massive loss after stable period.
        #
        # PULSE GUARD: During an electroporation pulse, the electric field can briefly
        # disrupt the image — either the cell appears to vanish (triggering exit_thresh)
        # or the length spikes then collapses (triggering exit_ratio). Both are artefacts,
        # not real exits. We skip the exit check for a short blanking window around the
        # known pulse frame so these frames cannot cause a false early exit.
        pulse_frame = self.params.get('dye_uptake_parameters', {}).get('pulse_frame', None)
        pulse_blanking = self.params.get('rupture_detection', {}).get('pulse_exit_blanking_frames', 5)

        # Build the set of frames to skip (only when a pulse frame is known)
        if pulse_frame is not None:
            blanked_frames = set(range(pulse_frame - 1, pulse_frame + pulse_blanking + 1))
        else:
            blanked_frames = set()

        exit_idx = len(protrusions)
        for i in range(entry_idx + 5, len(protrusions)):
            if i in blanked_frames:
                continue  # Skip — pulse artefact window, not a real exit
            p = protrusions[i]
            p_prev = protrusions[i-1]
            if p < exit_thresh or (p_prev > 3.0 and p < exit_ratio * p_prev):
                exit_idx = i
                break
        self.results['exit_frame_index'] = exit_idx
        
        # Nullify out-of-bounds data physically isolated to the active period
        for i in range(len(protrusions)):
            if i < entry_idx or i > exit_idx:
                self.results['protrusion_lengths_px'][i] = np.nan
                self.results['protrusion_lengths_um'][i] = np.nan
                self.results['protrusion_area_um2'][i] = np.nan
                self.results['body_area_um2'][i] = np.nan
                self.results['total_area_um2'][i] = np.nan
                self.results['body_solidity'][i] = np.nan
        
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
    
    def _abort(self, signal: str) -> Dict[str, Any]:
        self.results['signal'] = signal
        return self.results    
    
    def _calculate_morphology(self, mask_prot: np.ndarray, mask_body: np.ndarray, mask_total: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculates areas and strictly isolates solidity to the cell body."""
        scale_factor = self.params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        sf2 = scale_factor ** 2
        
        # 1. Individual Areas
        prot_area = cv2.countNonZero(mask_prot) * sf2
        body_area = cv2.countNonZero(mask_body) * sf2
        total_area = cv2.countNonZero(mask_total) * sf2
        
        # 2. Body Solidity
        # Performed specifically on the cell body to bypass empty spatial defects surrounding protrusions
        mask_body_u8 = mask_body.astype(np.uint8)
        contours, _ = cv2.findContours(mask_body_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        solidity = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(cnt)
            if area_px == 0:
                area_px = cv2.countNonZero(mask_body_u8)
                
            if len(cnt) >= 3: 
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area_px) / hull_area
                    
        return prot_area, body_area, total_area, solidity

    # =========================================================================
    #                       RUPTURE DETECTION LOGIC
    # =========================================================================
    
    def _generate_comprehensive_results(self) -> None:
        """Runs multiple detectors strictly inside the validated cell-presence bounds."""
        intensities = self.results['downstream_intensities']
        
        entry_idx = self.results.get('entry_frame_index', 0)
        exit_idx = self.results.get('exit_frame_index', len(intensities))
        
        # Expand the window slightly past the physical exit ---
        # When a cell pops, it disappears (exit_idx), but the haze flush 
        # happens at that exact frame or slightly after. 
        analysis_end = min(len(intensities), exit_idx + 3)
        valid_intensities = intensities[entry_idx:analysis_end]
        
        candidates = []
        
        # --- 1. Run temporal event detectors first ---
        
        # A. Spike (Transient Burst)
        if self.enable_spike:
            is_spike, spike_idx = self._detect_intensity_anomaly(valid_intensities, mode='spike')
            if is_spike: candidates.append((entry_idx + spike_idx, 'Intensity Spike'))
        
        # B. Step (Fast Leak)
        if self.enable_step:
            is_step, step_idx = self._detect_intensity_anomaly(valid_intensities, mode='step')
            if is_step: candidates.append((entry_idx + step_idx, 'Intensity Step'))

        # C. Direct Difference (Catches acute haze jumps smoothed out by CUSUM buffer logic)
        # the entry ramp is never included in the diff calculation.
        jump_thresh = self.params.get('rupture_detection', {}).get('sudden_jump_threshold', 1.0)
        jump_settling = self.params.get('rupture_detection', {}).get('cusum_settling_buffer', 0)

        if len(valid_intensities) > 1 + jump_settling:
            # Only look at the trace AFTER the settling period
            scan_region = valid_intensities[jump_settling:]
            diffs = np.diff(scan_region)
            max_diff_idx = np.argmax(diffs)
            # Use a conservative floor here. This detector finds the single largest
            # frame-to-frame jump in the entire trace — a low threshold makes it prone
            # to noise over long recordings. Pulse-context transient bursts are handled
            # instead by the peak-based check inside _detect_pulse_context_rupture,
            # which is gated to the pulse frame and therefore far less prone to false positives.
            if diffs[max_diff_idx] > max(jump_thresh, self.min_drift):
                # Translate the index back into the full valid_intensities coordinate space
                adjusted_idx = max_diff_idx + jump_settling + 1
                candidates.append((entry_idx + adjusted_idx, 'Sudden Haze Jump'))

        # D. CUSUM (Slow Drift)
        is_cusum, cusum_idx = self._detect_cusum_drift(valid_intensities)
        if is_cusum: candidates.append((entry_idx + cusum_idx, 'Haze Drift'))

        # E. Pulse Context (Pulse-Induced Rupture)
        # If dye_uptake is disabled, the key won't exist and this block is skipped.
        pulse_frame_idx = self.params.get('rupture_detection', {}).get('pulse_frame_idx', None)
        if pulse_frame_idx is not None:
            is_pulse_rupture, pulse_rupt_idx = self._detect_pulse_context_rupture(
                intensities,      # Pass the FULL array — pulse_frame_idx is an absolute index
                entry_idx,
                pulse_frame_idx
            )
            if is_pulse_rupture:
                candidates.append((pulse_rupt_idx, 'Pulse-Induced Rupture'))

        # --- 2. Evaluate DOA Context ---
        abs_threshold = self.params.get('rupture_detection', {}).get('absolute_intensity_threshold', 6.5)

        if len(valid_intensities) > 0 and np.median(valid_intensities[:3]) >= abs_threshold:
            # High initial intensity detected. We need to decide: did this cell rupture
            # upon entry (DOA), or did it enter intact and rupture later (e.g. at the pulse)?
            # guard here: only count acute events that occur AFTER a minimum
            doa_min_delay = self.params.get('rupture_detection', {}).get('doa_min_acute_delay_frames', 3)

            acute_events = [
                c for c in candidates
                if c[1] in ('Intensity Spike', 'Intensity Step', 'Sudden Haze Jump', 'Pulse-Induced Rupture')
                and c[0] >= entry_idx + doa_min_delay   # Must be meaningfully after entry
            ]

            if not acute_events:
                candidates.append((entry_idx, 'DOA: Sustained High Haze'))

        # --- 3. Arbitrate ---
        # Pick the earliest detected event among valid candidates
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_idx, reason = candidates[0]
            
            # Align the detected haze anomaly with the physical disappearance.
            final_rupture_idx = min(best_idx, max(0, exit_idx - 1))
            
            self.results['rupture_detected'] = True
            self.results['rupture_frame_index'] = final_rupture_idx
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
        baseline_len = r_params.get('min_cusum_baseline_frames', 5)
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
        baseline_len = r_params.get('min_cusum_baseline_frames', 5)
        lag = r_params.get('cusum_baseline_lag', 0)
        
        min_len = settling + baseline_len + lag + 1
        S_pos = 0.0
        
        # Fallback mechanism for very short traces
        if len(trace) < min_len: 
            if len(trace) < 3:
                return False, None
            
            baseline = trace[:3]
            mu = np.median(baseline)
            q75, q25 = np.percentile(baseline, [75, 25])
            iqr = q75 - q25
            raw_sigma = iqr * 0.7413
            capped_sigma = min(raw_sigma, r_params.get('max_baseline_sigma', 1.0))
            sigma = max(capped_sigma, self.min_intensity_noise_floor)
            
            k = max(self.min_drift, self.cusum_drift_tol * sigma)
            h = max(self.min_cusum_thresh, self.cusum_thresh_fac * sigma)
            
            for i in range(3, len(trace)):
                deviation = trace[i] - mu - k
                if deviation > 0:
                    S_pos += deviation
                else:
                    S_pos = 0.0
                if S_pos > h:
                    return True, i
            return False, None

        # Standard dynamic processing
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

    def _detect_pulse_context_rupture(
        self,
        intensities: List[float],
        entry_idx: int,
        pulse_frame_idx: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Detects membrane rupture caused by an electric pulse.
        """
        r_params = self.params.get('rupture_detection', {})

        # How many frames before and after the pulse to sample.
        pre_window  = r_params.get('pulse_context_pre_window',  5)
        post_window = r_params.get('pulse_context_post_window', 5)

        # How many frames immediately after the pulse to use for the peak check.
        # Kept small (1-2) so we only catch the burst itself, not later noise.
        peak_window = r_params.get('pulse_context_peak_window', 2)

        # Shared thresholds for both checks:
        # - fold_thresh: post signal must be at least this multiple of pre signal.
        # - abs_thresh:  post signal must also exceed pre signal by this absolute amount.
        fold_thresh = r_params.get('pulse_context_fold_threshold', 1.3)
        abs_thresh  = r_params.get('pulse_context_abs_threshold', 0.5)

        n = len(intensities)

        # Safety: pulse must sit inside the trace with room for both windows.
        if pulse_frame_idx <= entry_idx:
            return False, None
        if pulse_frame_idx - pre_window < entry_idx:
            return False, None
        if pulse_frame_idx + post_window >= n:
            return False, None

        # --- Shared baseline (used by both checks) ---
        pre_vals   = intensities[pulse_frame_idx - pre_window : pulse_frame_idx]
        pre_median = float(np.median(pre_vals))

        # Guard against near-zero baselines (very dark images)
        if pre_median < 0.1:
            return False, None

        # --- CHECK 1: Sustained leak (median of full post-window) ---
        post_vals   = intensities[pulse_frame_idx + 1 : pulse_frame_idx + 1 + post_window]
        post_median = float(np.median(post_vals))

        if (post_median / pre_median >= fold_thresh and
                post_median - pre_median >= abs_thresh):
            return True, pulse_frame_idx

        # --- CHECK 2: Transient burst (peak of first few frames after pulse) ---
        peak_end  = min(n, pulse_frame_idx + 1 + peak_window)
        peak_vals = intensities[pulse_frame_idx + 1 : peak_end]

        if len(peak_vals) == 0:
            return False, None

        post_peak = float(np.max(peak_vals))

        if (post_peak / pre_median >= fold_thresh and
                post_peak - pre_median >= abs_thresh):
            return True, pulse_frame_idx

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