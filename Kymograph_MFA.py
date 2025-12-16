# -*- coding: utf-8 -*-
"""
Generates Kymographs (Space-Time plots) from analyzing image sequences.

ROLE IN PIPELINE:
This module creates visual summaries of the cell entry process.
A "Kymograph" takes a 1D slice (center of the channel) from every frame
and stacks them vertically.

-   Y-Axis: Time (Top = Start, Bottom = End)
-   X-Axis: Position along the channel (Left = Inside, Right = Outside)

This allows researchers to see the cell's trajectory at a glance.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Utils_MFA as utils

logger = logging.getLogger(__name__)

class KymographAnalysis:
    """
    Generates kymographs from image sequences and overlays detection results.
    """

    def __init__(self, roi_images: List[np.ndarray], detection_results: Dict[str, Any], 
                 params: Dict[str, Any], time_data: Optional[List[float]] = None) -> None:
        self.roi_images = roi_images
        self.results = detection_results
        self.params = params
        self.time_data = time_data
        
        self.kymograph_matrix: Optional[np.ndarray] = None
        # self.growth_rate removed
        
        # Extract necessary detection data
        self.pipette_start_x = self.results.get('pipette_start_x_used')
        self.protrusions_px = self.results.get('protrusion_lengths_px', [])

    def run_analysis(self, output_dir: Path, trap_index: int) -> 'KymographAnalysis':
        """Executes the kymograph generation and saving workflow."""
        if not self.roi_images or not self.pipette_start_x:
            logger.warning(f"Skipping kymograph for Trap {trap_index}: Missing images or pipette info.")
            return self

        # 1. Generate the Kymograph Image (Matrix)
        self.kymograph_matrix = self._create_kymograph_matrix()
        
        # 2. Save the visualization (Single panel with trace)
        save_name = f"trap_{trap_index+1:02d}_kymograph.png"
        save_path = output_dir / save_name
        self.save_kymograph_visualization(save_path, trap_index)

        return self

    def _create_kymograph_matrix(self) -> np.ndarray:
        """
        Creates a kymograph by taking the centerline of each frame and stacking them vertically.
        """
        # Determine dimensions based on the first frame
        first_frame = self.roi_images[0]
        h, w = first_frame.shape[:2]
        center_y = h // 2
        
        # Define the slice thickness (averaging over a few lines reduces noise)
        thickness = self.params.get('kymograph_line_thickness', 3)
        y_start = max(0, center_y - thickness // 2)
        y_end = min(h, center_y + thickness // 2 + 1)
        
        lines = []
        for frame in self.roi_images:
            if frame is None:
                # Handle missing frames with a black line
                lines.append(np.zeros((1, w), dtype=np.uint8))
                continue
            
            img_8bit = utils.normalize_to_8bit(frame)
            if len(img_8bit.shape) == 3:
                img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
            
            # Extract horizontal strip and average vertically
            strip = img_8bit[y_start:y_end, :]
            avg_line = np.mean(strip, axis=0).astype(np.uint8)
            lines.append(avg_line)
            
        # Stack lines vertically: (Time, Space)
        kymograph = np.vstack(lines)
        return kymograph

    def save_kymograph_visualization(self, save_path: Path, trap_index: int) -> None:
        """
        Plots the Kymograph and overlays the red detection trace in MICRONS.
        X-Axis: Positive to the Left (Extension), Negative to the Right (Outside).
        Y-Axis: Time in Seconds (Using real extracted timestamps).
        """
        if self.kymograph_matrix is None: return

        scale_factor = self.params.get('scale_factor', 0.629)
        h, w = self.kymograph_matrix.shape
        
        if self.time_data and len(self.time_data) == h:
             total_time_s = self.time_data[-1]
             y_times = np.array(self.time_data)
        else:
             frame_interval = self.params.get('frame_interval', 0.2)
             total_time_s = h * frame_interval
             y_times = np.arange(len(self.protrusions_px)) * frame_interval

        utils.set_paper_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # --- 1. Calculate Extents for the Image ---
        # Map pixels to microns: Left=Inside(Positive), Right=Outside(Negative)
        # Note: Detection kymograph logic (Geometric) might differ slightly, but assuming
        # standard orientation: Left=Deep Inside.
        left_limit_um = (self.pipette_start_x - 0) * scale_factor
        right_limit_um = (self.pipette_start_x - w) * scale_factor
        
        # Show image with Time growing downwards
        ax.imshow(self.kymograph_matrix, cmap='gray', aspect='auto', 
                  extent=[left_limit_um, right_limit_um, total_time_s, 0])
        
        # --- 2. Overlay the Detection Trace ---
        if self.protrusions_px and len(self.protrusions_px) == h:
            tip_x_coords_um = [length * scale_factor for length in self.protrusions_px]
            
            # Use 'rupture_point' color from Utils for the trace
            colors = utils.get_color_scheme('blue_red')
            ax.plot(tip_x_coords_um, y_times, color=colors['rupture_point'], linewidth=1.5, label='Detected Tip', alpha=0.8)
            ax.axvline(x=0, color='cyan', linestyle='--', linewidth=1, label='Pipette Entrance', alpha=0.6)

        ax.set_title(f'Trap #{trap_index+1} Kymograph')
        ax.set_xlabel('Position relative to channel entrance (µm)')
        ax.set_ylabel('Time (s)')
        ax.legend(loc='upper right', fontsize='small', framealpha=0.7)
        
        # Save using Utils to ensure directory existence and DPI
        utils.save_plot_png(save_path, dpi=300)
        plt.close(fig)

def create_kymograph_for_trap(roi_images: List[np.ndarray], detection_results: Dict[str, Any], 
                              params: Dict[str, Any], output_dir: Path, trap_index: int, 
                              time_data: Optional[List[float]] = None) -> KymographAnalysis:
    """Helper function to instantiate and run the analysis."""
    # Dummy object that mocks the old 'growth_rate' property to prevent errors in MFA_analysis.py
    analyzer = KymographAnalysis(roi_images, detection_results, params, time_data)
    analyzer.run_analysis(output_dir, trap_index)
    
    # We monkey-patch a growth_rate attribute of None so the main loop doesn't crash
    analyzer.growth_rate = None 
    return analyzer