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
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np

import Utils_MFA as utils
import Plotting_MFA

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
        self.growth_rate = None # Placeholder compatibility
        
        # Extract necessary detection data
        self.pipette_start_x = self.results.get('pipette_start_x_used')
        self.protrusions_px = self.results.get('protrusion_lengths_px', [])

    def run_analysis(self, output_dir: Union[str, Path], trap_index: int) -> 'KymographAnalysis':
        """Executes the kymograph generation and saving workflow."""
        if not self.roi_images or not self.pipette_start_x:
            logger.warning(f"Skipping kymograph for Trap {trap_index}: Missing images or pipette info.")
            return self

        # 1. Generate the Kymograph Image (Matrix)
        self.kymograph_matrix = self._create_kymograph_matrix()
        
        # 2. Save the Kymograph Plot
        if self.kymograph_matrix is not None:
             Plotting_MFA.plot_kymograph(
                kymograph_matrix=self.kymograph_matrix,
                trap_index=trap_index,
                output_dir=output_dir,
                pipette_x=self.pipette_start_x,
                protrusions_px=self.protrusions_px,
                time_data=self.time_data,
                params=self.params
            )

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
        thickness = self.params.get('kymograph_parameters', {}).get('default_line_width_px', 10)
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
            
        return np.vstack(lines)

def create_kymograph_for_trap(roi_images: List[np.ndarray], detection_results: Dict[str, Any], 
                              params: Dict[str, Any], output_dir: Union[str, Path], trap_index: int, 
                              time_data: Optional[List[float]] = None) -> KymographAnalysis:
    """Helper function to instantiate and run the analysis."""
    analyzer = KymographAnalysis(roi_images, detection_results, params, time_data)
    analyzer.run_analysis(output_dir, trap_index)
    return analyzer