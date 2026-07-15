# -*- coding: utf-8 -*-
"""
Handles interactive image processing for microfluidic aspiration setup.

ROLE IN PIPELINE:
This module provides the Graphical User Interface (GUI) for the setup phase.
It allows the researcher to visually:
1.  Correct image rotation (aligning channels horizontally).
2.  Define the Region of Interest (ROI) for the first trap.
3.  Identify the pipette tip (zero point).
4.  Preview and confirm the positions of all subsequent traps.
5.  Select which specific traps to analyze.

TECHNICAL IMPLEMENTATION:
-   Uses OpenCV highgui for window management.
-   Handles mouse callbacks for dragging (ROI selection) and clicking (point selection).
-   Manages coordinate transformations between the displayed window (resized)
    and the raw image data (original resolution).
"""

import os
import logging
from typing import List, Optional, Union, TYPE_CHECKING, Tuple, Any, Dict
from pathlib import Path

import cv2
import numpy as np

import Utils_MFA as utils
if TYPE_CHECKING:
    from FileHandling_MFA import FileRead

logger = logging.getLogger(__name__)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 8,
    gap_length: int = 6,
) -> None:
    """Draw a dashed straight line between pt1 and pt2 on `img`.

    OpenCV has no native dashed-line primitive, so we walk from pt1 toward pt2
    in steps of (dash_length + gap_length), drawing a solid segment of length
    dash_length each step.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    total_len = float(np.hypot(dx, dy))
    if total_len < 1e-6:
        return
    # Unit vector along the line.
    ux = dx / total_len
    uy = dy / total_len
    step = dash_length + gap_length
    n_steps = int(total_len // step) + 1
    for i in range(n_steps):
        start_dist = i * step
        end_dist = min(start_dist + dash_length, total_len)
        seg_start = (int(round(x1 + ux * start_dist)), int(round(y1 + uy * start_dist)))
        seg_end = (int(round(x1 + ux * end_dist)), int(round(y1 + uy * end_dist)))
        cv2.line(img, seg_start, seg_end, color, thickness)


def _draw_dashed_rectangle(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
    gap_length: int = 6,
) -> None:
    """Draw a dashed rectangle by dashing each of its four sides."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    corners = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for p1, p2 in corners:
        _draw_dashed_line(img, p1, p2, color, thickness, dash_length, gap_length)


class _RoiSelector:
    """
    Helper class to manage the interactive dragging/resizing of the ROI box.
    Handles low-level mouse events (click, drag, release).
    """
    def __init__(self, initial_rect: Tuple[int, int, int, int], image_dims: Tuple[int, int]):
        self.x, self.y, self.w, self.h = initial_rect
        self.img_h, self.img_w = image_dims
        self.is_dragging = False
        self.is_resizing = False
        self.resize_corner = None
        self.drag_start_pos = None
        self.rect_at_drag_start = None

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any):
        """OpenCV mouse callback to handle dragging and resizing."""
        corners = {
            "top_left": (self.x, self.y), "top_right": (self.x + self.w, self.y),
            "bottom_left": (self.x, self.y + self.h), "bottom_right": (self.x + self.w, self.y + self.h)
        }
        
        # 1. Check if user clicked a corner (Resize) or the center (Move)
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (cx, cy) in corners.items():
                if abs(x - cx) < 15 and abs(y - cy) < 15:
                    self.is_resizing, self.resize_corner = True, name
                    self.drag_start_pos = (x, y)
                    self.rect_at_drag_start = (self.x, self.y, self.w, self.h)
                    return
            if self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h:
                self.is_dragging = True
                self.drag_start_pos = (x, y)
                self.rect_at_drag_start = (self.x, self.y, self.w, self.h)
        
        # 2. Update coordinates while dragging
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_resizing and self.drag_start_pos:
                self._perform_resize(x, y)
            elif self.is_dragging and self.drag_start_pos:
                dx = x - self.drag_start_pos[0]
                dy = y - self.drag_start_pos[1]
                self.x = self.rect_at_drag_start[0] + dx
                self.y = self.rect_at_drag_start[1] + dy
        
        # 3. Stop dragging on release
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging, self.is_resizing = False, False

        self._constrain_to_image_bounds()

    def _perform_resize(self, x: int, y: int):
        """Calculates new ROI dimensions during a resize operation."""
        dx = x - self.drag_start_pos[0]
        dy = y - self.drag_start_pos[1]
        orig_x, orig_y, orig_w, orig_h = self.rect_at_drag_start

        if self.resize_corner == "top_left":
            if orig_w - dx > 20 and orig_h - dy > 20:
                self.x, self.y = orig_x + dx, orig_y + dy
                self.w, self.h = orig_w - dx, orig_h - dy
        elif self.resize_corner == "bottom_right":
            if orig_w + dx > 20 and orig_h + dy > 20:
                self.w, self.h = orig_w + dx, orig_h + dy
        elif self.resize_corner == "top_right":
            if orig_w + dx > 20 and orig_h - dy > 20:
                self.y = orig_y + dy
                self.w, self.h = orig_w + dx, orig_h - dy
        elif self.resize_corner == "bottom_left":
            if orig_w - dx > 20 and orig_h + dy > 20:
                self.x = orig_x + dx
                self.w, self.h = orig_w - dx, orig_h + dy

    def _constrain_to_image_bounds(self):
        """Ensures the ROI does not go outside the image boundaries."""
        self.x = max(0, min(self.img_w - self.w, self.x))
        self.y = max(0, min(self.img_h - self.h, self.y))

    def get_coords(self) -> List[int]:
        """Returns the final ROI coordinates in [ymin, ymax, xmin, xmax] format."""
        return [self.y, self.y + self.h, self.x, self.x + self.w]

class _PointSelector:
    """Helper class for clicking a single point (Pipette Tip)."""
    def __init__(self, display_dims: Tuple[int, int, int, int], scale: float):
        self.x, self.y = -1, -1
        self.disp_w, self.disp_h, self.x_off, self.y_off = display_dims
        self.scale = scale

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any):
        """
        Captures click position and converts it from 'Display Coordinates'
        back to 'Original Image Coordinates'.
        """
        is_in_x_bounds = self.x_off <= x < self.x_off + self.disp_w
        is_in_y_bounds = self.y_off <= y < self.y_off + self.disp_h

        if event == cv2.EVENT_LBUTTONDOWN and is_in_x_bounds and is_in_y_bounds:
            self.x = int((x - self.x_off) / self.scale)
            self.y = int((y - self.y_off) / self.scale)

    def get_coords(self) -> List[int]:
        return [self.x, self.y]


class CropImage():
    """
    Main controller for the setup workflow.
    Manages the sequence of UI steps.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initializes the state for the image processing workflow."""
        self.params = params or {}
        self.rotation_angle: float = -90.0
        self.roi_position: List[int] = [0, 0]
        self.roi_size: Optional[Tuple[int, int]] = None
        self.pipette_coords: Optional[List[int]] = None
        self.current_trap: int = 0
        self.scale_factor: float = self.params.get('scale_factor', 0.629)
        self.vertical_line_pos: Optional[int] = None
        self.moving_line: bool = False
        self.trap_spacing_factor: float = self.params.get('trap_spacing_factor', 1.43)
        # Read verify_traps_interactively from the nested workflow_settings sub-dict,
        # matching how LineDetectionMFA.run() reads it. Falls back to a flat-dict
        # lookup so the worker path (which merges workflow_settings into a flat dict)
        # still works without change.
        self.verify_traps_interactively: bool = (
            self.params.get('workflow_settings', {}).get('verify_traps_interactively')
            if 'workflow_settings' in self.params
            else self.params.get('verify_traps_interactively', True)
        )
        self.all_trap_rois: List[Optional[List[int]]] = []
        self.max_traps: int = self.params.get('max_traps', 18)

        # Window dimensions
        self.window_width: int = self.params.get('interactive_window_width', 1000)
        self.window_height: int = self.params.get('interactive_window_height', 800)

        # Contrast/Brightness controls
        self.display_alpha = 1.0   # Contrast
        self.display_beta = 0      # Brightness
        
        self.file_reader: 'FileRead' = None
        self.first_image: np.ndarray = None
        self.img_height: int = 0
        self.img_width: int = 0
        self.rotated_first_image: np.ndarray = None
        self.roi_coords: List[int] = [0, 0, 0, 0]


    def run(self, file_reader: 'FileRead') -> str:
        """
        Runs the 4-step setup wizard.
        Returns 'confirm' if successful, or 'stop'/'restart'.
        """
        self.file_reader = file_reader
        
        # --- Step 0. Load Frame 0 for Geometry Setup ---
        logger.info("Interactive Setup: Loading Frame 0 for geometry adjustment.")
        self.first_image = self.file_reader.read_img(self.file_reader.tif_files[0])
        
        if self.first_image is None:
            logger.error("Failed to load Frame 0 for setup.")
            return 'stop'

        self.img_height, self.img_width = self.first_image.shape[:2]
        
        # Step 1: Rotate
        logger.info("--- Step 1: Adjust Image Rotation ---")
        signal = self._adjust_rotation_interactively()
        if signal in ('restart', 'stop'):
            cv2.destroyAllWindows()
            return signal
        
        # Step 2: ROI (Identify First Trap)
        logger.info("--- Step 2: ROI Positioning and Sizing ---")
        signal = self._position_and_resize_roi()
        if signal in ('restart', 'stop'):
            cv2.destroyAllWindows()
            return signal
        
        # Step 3: Spacing
        logger.info("--- Step 3: Trap Position Preview & Spacing ---")
        signal = self.preview_all_traps()
        if signal in ('restart', 'stop'):
            cv2.destroyAllWindows()
            return signal
        
        # Step 4: Pipette Tip
        logger.info("--- Step 4: Pipette Tip Selection ---")
        signal = self._get_pipette_position()
        if signal in ('restart', 'stop'):
            cv2.destroyAllWindows()
            return signal
            
        # Step 5: Selection
        num_frames = len(self.file_reader.tif_files)

        # GUV mode: use the same early frame index as the threshold-selection step
        # so the user sees consistent, intact GUV images throughout setup.
        # Cell mode: fall back to the fractional default (e.g. middle of video).
        guv = self.params.get('guv_settings', {})
        if guv.get('enable', False):
            target_idx = max(0, min(int(guv.get('selection_frame_index', 2)), num_frames - 1))
        else:
            fraction = self.params.get('selection_frame_fraction', 0.5)
            target_idx = max(0, min(int(num_frames * fraction), num_frames - 1))

        logger.info("--- Step 5: Select Traps to Analyze ---")

        selection_image = self.file_reader.read_img(self.file_reader.tif_files[target_idx])
        if selection_image is None:
            logger.warning(f"Could not load frame {target_idx}, falling back to Frame 0 for selection.")
            selection_image = self.first_image

        signal, selected = self.select_traps_interactively(selection_image)
        if signal in ('restart', 'stop'):
            cv2.destroyAllWindows()
            return signal
        self.selected_traps = selected

        logger.info("Setup complete!")
        return 'confirm'
    
    def _handle_ui_keypress(self, key: int) -> str:
        """Centralized hotkey handler for contrast/brightness and flow control."""
        if key == 13: return 'confirm'
        if key == 27: return 'stop'
        if key in (ord('r'), ord('R')): return 'restart'
    
        if key in (ord('l'), ord('L')): self.display_alpha = min(3.0, self.display_alpha + 0.1)
        elif key in (ord('j'), ord('J')): self.display_alpha = max(0.1, self.display_alpha - 0.1)
        elif key in (ord('k'), ord('K')): self.display_beta = min(100, self.display_beta + 5)
        elif key in (ord('i'), ord('I')): self.display_beta = max(-100, self.display_beta - 5)
        
        return 'continue'
    
    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhances image contrast using CLAHE and manual controls."""
        clip_limit = self.params.get('clahe_clip_limit', 2.0)
        grid_size = tuple(self.params.get('clahe_tile_grid_size', (8, 8)))
        
        img_8bit = utils.normalize_to_8bit(image)
        
        # 1. Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced = clahe.apply(img_8bit)
        
        # 2. Apply manual controls
        adjusted = cv2.convertScaleAbs(enhanced, alpha=self.display_alpha, beta=self.display_beta)
        return adjusted

    def _adjust_rotation_interactively(self) -> str:
        """
        Displays a window with a vertical guideline.
        """
        rotated_image = utils.rotate_image(self.first_image, self.rotation_angle)
        utils.create_centered_window('Angle Adjustment', self.window_width, self.window_height)
        h, w = rotated_image.shape[:2]
        self.vertical_line_pos = w // 2
        
        # State dictionary to safely manage mouse dragging
        state = {'moving': False}

        def move_line(event, x, y, flags, param):
            # 1. Automatically match the exact 95% margin used by Utils_MFA.prepare_display_image
            scale = min((self.window_width * 0.95) / w, (self.window_height * 0.95) / h)
            
            display_w = int(w * scale)
            x_offset = (self.window_width - display_w) // 2
            
            # 2. Map the mouse's Window X back to the Original Image X
            if scale > 0:
                orig_x = int((x - x_offset) / scale)
            else:
                orig_x = x
                
            # 3. Clamp the value so dragging out of bounds doesn't crash the UI
            orig_x = max(0, min(w - 1, orig_x))

            if event == cv2.EVENT_LBUTTONDOWN: 
                state['moving'] = True
                self.vertical_line_pos = orig_x
            elif event == cv2.EVENT_MOUSEMOVE and state['moving']: 
                self.vertical_line_pos = orig_x
            elif event == cv2.EVENT_LBUTTONUP: 
                state['moving'] = False
        
        cv2.setMouseCallback('Angle Adjustment', move_line)

        while True:
            # Re-apply enhancement *inside* the loop
            rotated_enhanced = self._adjust_brightness_contrast(rotated_image)
            display_image = cv2.cvtColor(rotated_enhanced, cv2.COLOR_GRAY2BGR)

            # Standardized Guide Color (Medium Blue)
            guide_color = utils.get_ui_color('guide')
            cv2.line(display_image, (self.vertical_line_pos, 0), (self.vertical_line_pos, h), guide_color, 2)
            
            utils.add_text_overlay(display_image, [
                f"Angle: {self.rotation_angle:.1f} | Contrast: {self.display_alpha:.1f} | Bright: {self.display_beta}",
                "W/S: +/- 10 deg | A/D: +/- 0.1 deg",
                "Contrast: J/L | Brightness: I/K",
                "ENTER: Confirm | R: Restart Setup | ESC: Stop Analysis"
            ])
            display_resized = utils.prepare_display_image(display_image, self.window_width, self.window_height)
            cv2.imshow('Angle Adjustment', display_resized)

            key = cv2.waitKey(30) & 0xFF
            angle_changed = False
            
            if key == 13: break             # Enter
            if key == 27: return 'stop'     # ESC
            if key in (ord('r'), ord('R')): return 'restart'

            elif key in (ord('w'), ord('W')): self.rotation_angle -= 10.0; angle_changed = True
            elif key in (ord('s'), ord('S')): self.rotation_angle += 10.0; angle_changed = True
            elif key in (ord('a'), ord('A')): self.rotation_angle -= 0.1; angle_changed = True
            elif key in (ord('d'), ord('D')): self.rotation_angle += 0.1; angle_changed = True
            
            elif key in (ord('l'), ord('L')): self.display_alpha = min(3.0, self.display_alpha + 0.1)
            elif key in (ord('j'), ord('J')): self.display_alpha = max(0.1, self.display_alpha - 0.1)
            elif key in (ord('k'), ord('K')): self.display_beta = min(100, self.display_beta + 5)
            elif key in (ord('i'), ord('I')): self.display_beta = max(-100, self.display_beta - 5)

            if angle_changed:
                rotated_image = utils.rotate_image(self.first_image, self.rotation_angle)

        self.rotated_first_image = rotated_image
        cv2.destroyAllWindows()
        return 'confirm'

    def _position_and_resize_roi(self) -> str:
        """
        User drags a box over the FIRST trap.
        Initial size is set by config (default_roi_width_um).
        """
        self.rotated_first_image = utils.rotate_image(self.first_image, self.rotation_angle)
        h, w = self.rotated_first_image.shape[:2]
        
        # Calculate pixel dimensions from config microns
        default_width_um = self.params.get('default_roi_width_um', 60.0)
        default_height_um = self.params.get('default_roi_height_um', 25.0)

        # Convert micron-based size to pixel-based size
        default_width_px = int(default_width_um / self.scale_factor)
        default_height_px = int(default_height_um / self.scale_factor)
        initial_x = w // 2 - default_width_px // 2
        initial_y = h // 2 - default_height_px // 2

        roi_selector = _RoiSelector((initial_x, initial_y, default_width_px, default_height_px), (h, w))
        window_name = 'Position and Resize ROI'
        utils.create_centered_window(window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(window_name, roi_selector.mouse_callback)

        while True:
            display_base_image = self._adjust_brightness_contrast(self.rotated_first_image)
            display_image = cv2.cvtColor(display_base_image, cv2.COLOR_GRAY2BGR)

            x, y, rw, rh = roi_selector.x, roi_selector.y, roi_selector.w, roi_selector.h
            
            # Standardized Guide Color (Medium Blue) for ROI Box
            guide_color = utils.get_ui_color('guide')
            cv2.rectangle(display_image, (x, y), (x + rw, y + rh), guide_color, 2)
            
            utils.draw_ui_text(display_image, 
                               f"Size: {rw * self.scale_factor:.1f} x {rh * self.scale_factor:.1f} um", 
                               (x, y - 10), color=guide_color)
            
            utils.add_text_overlay(display_image, [
                f"Contrast: {self.display_alpha:.1f} | Bright: {self.display_beta}",
                "Drag box/corners. WASD to move.",
                "Contrast: J/L | Brightness: I/K",
                "ENTER: Confirm | R: Restart Setup | ESC: Stop Analysis"
            ])
            cv2.imshow(window_name, display_image)

            key = cv2.waitKey(20) & 0xFF
            if key == 13: break
            if key == 27: return 'stop'
            if key in (ord('r'), ord('R')): return 'restart'
            
            if key in (ord('w'), ord('W')): roi_selector.y -= 1
            elif key in (ord('s'), ord('S')): roi_selector.y += 1
            elif key in (ord('a'), ord('A')): roi_selector.x -= 1
            elif key in (ord('d'), ord('D')): roi_selector.x += 1
            
            elif key in (ord('l'), ord('L')): self.display_alpha = min(3.0, self.display_alpha + 0.1)
            elif key in (ord('j'), ord('J')): self.display_alpha = max(0.1, self.display_alpha - 0.1)
            elif key in (ord('k'), ord('K')): self.display_beta = min(100, self.display_beta + 5)
            elif key in (ord('i'), ord('I')): self.display_beta = max(-100, self.display_beta - 5)

            roi_selector._constrain_to_image_bounds()

        self.roi_size = (roi_selector.w, roi_selector.h)
        self.roi_coords = roi_selector.get_coords()
        cv2.destroyAllWindows()
        
        return 'confirm'

    def _get_pipette_position(self) -> str:
        """Launches a window for the user to select the pipette tip."""
        y_min, y_max, x_min, x_max = self.roi_coords
        roi_image = self.rotated_first_image[y_min:y_max, x_min:x_max]
        
        window_name = 'Select Pipette Tip'
        utils.create_centered_window(window_name, self.window_width, self.window_height)

        enhanced_roi_init = self._adjust_brightness_contrast(roi_image)
        roi_h, roi_w = enhanced_roi_init.shape[:2]
        
        scale = min((self.window_width * 0.9) / roi_w, (self.window_height * 0.8) / roi_h)
        disp_w, disp_h = int(roi_w * scale), int(roi_h * scale)

        x_off = (self.window_width - disp_w) // 2
        y_off = (self.window_height - disp_h - 60) // 2
        
        point_selector = _PointSelector((disp_w, disp_h, x_off, y_off), scale)
        cv2.setMouseCallback(window_name, point_selector.mouse_callback)

        while True:
            enhanced_roi = self._adjust_brightness_contrast(roi_image)
            image_8bit = cv2.cvtColor(enhanced_roi, cv2.COLOR_GRAY2BGR)

            display_roi_image = cv2.resize(image_8bit, (disp_w, disp_h), interpolation=cv2.INTER_CUBIC)

            display_image = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            display_image[y_off:y_off + disp_h, x_off:x_off + disp_w] = display_roi_image
            
            px, py = point_selector.get_coords()
            
            # Use Pipette Color (Dark Blue)
            pip_color = utils.get_ui_color('pipette')
            
            if px >= 0 and py >= 0:
                disp_x, disp_y = int(px * scale) + x_off, int(py * scale) + y_off
                cv2.circle(display_image, (disp_x, disp_y), 8, pip_color, -1)
                cv2.line(display_image, (disp_x, y_off), (disp_x, y_off + disp_h), pip_color, 2)
            
            utils.add_text_overlay(display_image, [
                f"Contrast: {self.display_alpha:.1f} | Bright: {self.display_beta}",
                "Click on the pipette tip.", 
                "Contrast: J/L | Brightness: I/K",
                "ENTER: Confirm | R: Restart Setup | ESC: Stop Analysis"
                ], bottom_margin=20)
            
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 13: break  # Enter
            if key == 27: return 'stop'     # ESC
            if key in (ord('r'), ord('R')): return 'restart'
            
            elif key in (ord('l'), ord('L')): self.display_alpha = min(3.0, self.display_alpha + 0.1)
            elif key in (ord('j'), ord('J')): self.display_alpha = max(0.1, self.display_alpha - 0.1)
            elif key in (ord('k'), ord('K')): self.display_beta = min(100, self.display_beta + 5)
            elif key in (ord('i'), ord('I')): self.display_beta = max(-100, self.display_beta - 5)

        self.pipette_coords = point_selector.get_coords()
        cv2.destroyAllWindows()
        
        return 'confirm'

    def get_next_trap_roi(self) -> Optional[Union[List[int], str]]:
        """
        DEPRECATED in batch-selection workflow, but kept for compatibility.
        Calculates the next trap ROI and optionally asks for user verification.
        """
        if self.current_trap >= self.max_traps - 1:
            return None
    
        last_confirmed_roi = next((roi for roi in reversed(self.all_trap_rois) if roi is not None), None)
    
        if last_confirmed_roi is None:
            last_confirmed_roi = self.roi_coords
    
        prev_y_min, prev_y_max, prev_x_min, prev_x_max = last_confirmed_roi
    
        trap_spacing = int(self.roi_size[1] * self.trap_spacing_factor)
        
        num_skipped = 0
        for roi in reversed(self.all_trap_rois):
            if roi is None: num_skipped += 1
            else: break
        
        vertical_shift = trap_spacing * (num_skipped + 1)
        new_y_min = prev_y_min + vertical_shift
        new_y_max = prev_y_max + vertical_shift
    
        if new_y_max > self.img_height:
            return None
        new_roi = [new_y_min, new_y_max, prev_x_min, prev_x_max]
    
        if self.verify_traps_interactively:
            verification_result = self._verify_trap_position(last_confirmed_roi, new_roi)
            if verification_result is None:
                self.current_trap += 1
                self.all_trap_rois.append(None)
                return None
            elif verification_result == "stop":
                return "stop"
            else:
                self.roi_coords = verification_result
                self.all_trap_rois.append(self.roi_coords)
                self.current_trap += 1
                return self.roi_coords
        else:
            self.roi_coords = new_roi
            self.all_trap_rois.append(new_roi)
            self.current_trap += 1
            return new_roi


    def _verify_trap_position(self, prev_roi: List[int], new_roi: List[int]) -> Optional[Union[List[int], str]]:
        """Displays a window for the user to verify or adjust the next trap's ROI."""
        frame = self.file_reader.read_img(self.file_reader.tif_files[self.setup_frame_index])
        rotated = utils.rotate_image(frame, self.rotation_angle)
        
        utils.create_centered_window("Verify Next Trap", self.window_width, self.window_height)

        adjusting = False
        roi_width = new_roi[3] - new_roi[2]
        roi_height = new_roi[1] - new_roi[0]
        roi_position = [new_roi[2], new_roi[0]]

        def adjust_roi(event, x, y, flags, param):
            nonlocal adjusting, roi_position
            if event == cv2.EVENT_LBUTTONDOWN: adjusting = True
            elif event == cv2.EVENT_MOUSEMOVE and adjusting:
                roi_position = [x - roi_width // 2, y - roi_height // 2]
            elif event == cv2.EVENT_LBUTTONUP: adjusting = False

        cv2.setMouseCallback("Verify Next Trap", adjust_roi)

        while True:
            enhanced_frame = self._adjust_brightness_contrast(rotated)
            display = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
            
            display_copy = display.copy()
            cv2.rectangle(display_copy, (prev_roi[2], prev_roi[0]), (prev_roi[3], prev_roi[1]), (0, 255, 0), 2)

            curr_x_min, curr_y_min = roi_position
            curr_x_max, curr_y_max = curr_x_min + roi_width, curr_y_min + roi_height
            cv2.rectangle(display_copy, (curr_x_min, curr_y_min), (curr_x_max, curr_y_max), (255, 255, 0), 2)
            
            utils.add_text_overlay(display_copy, [
                f"Contrast: {self.display_alpha:.1f} | Bright: {self.display_beta}",
                "Y: Accept | N: Skip | ESC: Stop | Drag or WASD: Adjust",
                "Contrast: J/L | Brightness: I/K"
            ], bottom_margin=20)
            
            cv2.imshow("Verify Next Trap", display_copy)

            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('y'):
                cv2.destroyAllWindows()
                return [curr_y_min, curr_y_max, curr_x_min, curr_x_max]
            elif key == ord('n'):
                cv2.destroyAllWindows()
                return None
            elif key == 27:
                cv2.destroyAllWindows()
                return "stop"
                
            elif key in (ord('w'), ord('W')): roi_position[1] -= 1
            elif key in (ord('s'), ord('S')): roi_position[1] += 1
            elif key in (ord('a'), ord('A')): roi_position[0] -= 1
            elif key in (ord('d'), ord('D')): roi_position[0] += 1
            
            elif key in (ord('l'), ord('L')): self.display_alpha = min(3.0, self.display_alpha + 0.1)
            elif key in (ord('j'), ord('J')): self.display_alpha = max(0.1, self.display_alpha - 0.1)
            elif key in (ord('k'), ord('K')): self.display_beta = min(100, self.display_beta + 5)
            elif key in (ord('i'), ord('I')): self.display_beta = max(-100, self.display_beta - 5)

    def preview_all_traps(self) -> str:
        """
        Calculates ROIs for ALL traps based on the first trap and the spacing factor.
        Displays them all overlayed on the image.
        User can adjust 'trap_spacing_factor' live using WASD keys.
        """
        rotated = utils.rotate_image(self.first_image, self.rotation_angle)
        
        y_min_int, y_max_int, x_min_int, x_max_int = self.roi_coords
        roi_height = self.roi_size[1]
        
        while True:
            enhanced_frame = self._adjust_brightness_contrast(rotated)
            display = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
            
            # Use Standard Colors
            guide_color = utils.get_ui_color('guide')
            
            # Draw First Trap
            cv2.rectangle(display, (x_min_int, y_min_int), (x_max_int, y_max_int), guide_color, 2)
            cv2.putText(display, "Trap #1", (x_min_int, y_min_int - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)

            y_min_float = float(y_min_int)
            for trap_idx in range(2, self.max_traps + 1):
                trap_spacing_float = roi_height * self.trap_spacing_factor
                y_min_float += trap_spacing_float
                
                y_min_draw = int(y_min_float)
                y_max_draw = int(y_min_float + roi_height)

                if y_max_draw > rotated.shape[0]: break
                
                cv2.rectangle(display, (x_min_int, y_min_draw), (x_max_int, y_max_draw), guide_color, 2)
                cv2.putText(display, f"#{trap_idx}", (x_min_int + 5, y_min_draw + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, guide_color, 2)

            utils.add_text_overlay(display, [
                f"Spacing: {self.trap_spacing_factor:.4f}",
                "WASD: Adjust Spacing",
                "ENTER: Confirm"
            ])
            
            utils.create_centered_window("Trap Positions Preview", self.window_width, self.window_height)
            cv2.imshow("Trap Positions Preview", display)

            key = cv2.waitKey(0) & 0xFF
            if key == 13: break 
            if key == 27: return 'stop' 
            if key in (ord('r'), ord('R')): return 'restart'

            elif key in (ord('w'), ord('W')): self.trap_spacing_factor += 0.001
            elif key in (ord('s'), ord('S')): self.trap_spacing_factor -= 0.001
            elif key in (ord('d'), ord('D')): self.trap_spacing_factor += 0.01
            elif key in (ord('a'), ord('A')): self.trap_spacing_factor -= 0.01
            
            elif key in (ord('l'), ord('L')): self.display_alpha = min(3.0, self.display_alpha + 0.1)
            elif key in (ord('j'), ord('J')): self.display_alpha = max(0.1, self.display_alpha - 0.1)
            elif key in (ord('k'), ord('K')): self.display_beta = min(100, self.display_beta + 5)
            elif key in (ord('i'), ord('I')): self.display_beta = max(-100, self.display_beta - 5)
        
        # Populate final list
        self.all_trap_rois = [self.roi_coords]
        y_min_float = float(self.roi_coords[0])
        for trap_idx in range(2, self.max_traps + 1):
            trap_spacing_float = roi_height * self.trap_spacing_factor
            y_min_float += trap_spacing_float
            y_min_draw = int(y_min_float)
            y_max_draw = int(y_min_float + roi_height)
            if y_max_draw > rotated.shape[0]:
                self.all_trap_rois.append(None)
            else:
                self.all_trap_rois.append([y_min_draw, y_max_draw, x_min_int, x_max_int])
        
        cv2.destroyAllWindows()
        return 'confirm'
    
    def select_traps_interactively(self, test_image: np.ndarray) -> Tuple[str, List[int]]:
        """
        Shows a full view with all potential traps and allows the user
        to CLICK-TO-SELECT which ones to analyze.
        
        Returns:
            Tuple: (Status Signal, List of Selected Trap Indices)
        """
        logger.info("   Batch selecting traps for analysis...")
        rotated_image = utils.rotate_image(test_image, self.rotation_angle)
        
        selected_indices = set()        
        
        h, w = rotated_image.shape[:2]
        scale = min((self.window_width * 0.95) / w, (self.window_height * 0.95) / h)
        new_w, new_h = int(w * scale), int(h * scale)
        x_offset = (self.window_width - new_w) // 2
        y_offset = (self.window_height - new_h) // 2
        
        callback_data = {
            'selected': selected_indices, 
            'needs_redraw': True,
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset
        }
        
        # Callback to handle clicking
        def click_to_select(event, x, y, flags, param):
            """Mouse callback to toggle trap selection."""
            if event == cv2.EVENT_LBUTTONDOWN:
                scale = param['scale']
                x_off = param['x_offset']
                y_off = param['y_offset']

                # Convert click coordinates to original image coordinates
                orig_x = int((x - x_off) / scale)
                orig_y = int((y - y_off) / scale)
                
                # Check if click falls inside any trap ROI
                for i, roi in enumerate(self.all_trap_rois):
                    if roi is None:
                        continue
                    y_min, y_max, x_min, x_max = roi
                    
                    if x_min <= orig_x <= x_max and y_min <= orig_y <= y_max:
                        if i in param['selected']:
                            param['selected'].remove(i)
                        else:
                            param['selected'].add(i)
                        param['needs_redraw'] = True
                        break

        window_name = "Batch Trap Selection"
        utils.create_centered_window(window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(window_name, click_to_select, callback_data)
        
        # Pre-fetch colors for the loop
        color_selected = utils.get_bgr_color('secondary') # Red
        color_unselected = utils.get_ui_color('guide')    # Medium Blue

        while True:
            display_base = self._adjust_brightness_contrast(rotated_image)
            display_base = cv2.cvtColor(display_base, cv2.COLOR_GRAY2BGR)

            if callback_data['needs_redraw']:
                display = display_base.copy()
                overlay = display.copy()

                for i, roi in enumerate(self.all_trap_rois):
                    if roi is None:
                        continue
                    y_min, y_max, x_min, x_max = roi
                    
                    if i in callback_data['selected']:
                        # Selected: Filled Red box
                        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color_selected, -1)
                        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), color_selected, 2)
                        cv2.putText(display, f"#{i+1}", (x_min + 5, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_selected, 2)
                    else:
                        # Unselected: Thin Blue box
                        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), color_unselected, 1)
                        cv2.putText(display, f"#{i+1}", (x_min + 5, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_unselected, 1)

                alpha = 0.4
                display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)
                
                instructions = [
                    f"Contrast: {self.display_alpha:.1f} | Bright: {self.display_beta}",
                    "Click on traps to select/deselect.",
                    "Contrast: J/L | Brightness: I/K",
                    "ENTER: Confirm | R: Restart Setup | ESC: Stop Analysis"
                ]
                utils.add_text_overlay(display, instructions)
                
                display_resized = utils.prepare_display_image(display, self.window_width, self.window_height)
                cv2.imshow(window_name, display_resized)
                callback_data['needs_redraw'] = False

            key = cv2.waitKey(30) & 0xFF
            
            if key == 13: # Enter
                cv2.destroyAllWindows()
                final_selection = sorted(list(callback_data['selected']))
                logger.info(f"   Selected {len(final_selection)} traps for processing: {[i+1 for i in final_selection]}")
                return 'confirm', final_selection
            if key == 27: # ESC
                cv2.destroyAllWindows()
                return 'stop', []
            if key in (ord('r'), ord('R')): # R
                cv2.destroyAllWindows()
                return 'restart', []
            
            elif key in (ord('l'), ord('L')): 
                self.display_alpha = min(3.0, self.display_alpha + 0.1)
                callback_data['needs_redraw'] = True
            elif key in (ord('j'), ord('J')): 
                self.display_alpha = max(0.1, self.display_alpha - 0.1)
                callback_data['needs_redraw'] = True
            elif key in (ord('k'), ord('K')): 
                self.display_beta = min(100, self.display_beta + 5)
                callback_data['needs_redraw'] = True
            elif key in (ord('i'), ord('I')): 
                self.display_beta = max(-100, self.display_beta - 5)
                callback_data['needs_redraw'] = True
        
        # Fallback
        cv2.destroyAllWindows()
        return 'stop', []


    def save_trap_map(self, results_dir: Path) -> None:
        """Saves an image showing the final positions of all confirmed ROIs.

        Styling rules:
          - Traps that were selected for analysis are drawn in blue; traps that
            were detected but not selected for analysis are drawn in gray.
          - Trap #1 (index 0) is drawn with a dashed border regardless of whether
            it was selected. Its color follows the rule above.
        """
        frame = self.file_reader.read_img(self.file_reader.tif_files[0])
        rotated = utils.rotate_image(frame, self.rotation_angle)

        # Use the default, not the manually adjusted, enhancement for saving
        self.display_alpha = 1.0
        self.display_beta = 0
        enhanced_frame = self._adjust_brightness_contrast(rotated)
        display = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

        # Pre-fetch colors
        c_processed = utils.get_ui_color('guide')  # Blue — traps chosen for analysis
        c_unprocessed = (150, 150, 150)            # Neutral gray — traps skipped

        # `selected_traps` may not exist if setup was skipped; treat as empty set.
        processed = set(getattr(self, 'selected_traps', None) or [])

        thickness = 2

        for trap_idx, roi in enumerate(self.all_trap_rois):
            if roi is None:
                continue
            y_min, y_max, x_min, x_max = roi

            is_first = (trap_idx == 0)
            color = c_processed if trap_idx in processed else c_unprocessed

            if is_first:
                # Dashed border for the first trap.
                _draw_dashed_rectangle(
                    display, (x_min, y_min), (x_max, y_max),
                    color, thickness=thickness, dash_length=8, gap_length=6
                )
            else:
                cv2.rectangle(display, (x_min, y_min), (x_max, y_max), color, thickness)

            cv2.putText(
                display, f"Trap #{trap_idx+1}",
                (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        save_path = results_dir / "trap_map.png"
        cv2.imwrite(str(save_path), display)
        logger.info(f"Trap map saved to: {save_path.name}")

    def process_frame(self, frame: np.ndarray, trap_idx: int = 0, skip_rotation: bool = False) -> Optional[np.ndarray]:
        """
        Applies rotation and extracts the specified trap ROI from a single frame.
        Optimization: skip_rotation=True skips the expensive cv2.warpAffine call.
        """
        if frame is None:
            logger.warning(f"process_frame received a None frame for trap {trap_idx}.")
            return None
            
        # Only rotate if the flag is False and an angle is actually set
        if skip_rotation or self.rotation_angle == 0:
            rotated = frame
        else:
            rotated = utils.rotate_image(frame, self.rotation_angle)
            
        y_min, y_max, x_min, x_max = self.all_trap_rois[trap_idx]
        return rotated[y_min:y_max, x_min:x_max].copy()