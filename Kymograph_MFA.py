# -*- coding: utf-8 -*-
"""
Generates the membrane-channel kymograph (space-time plot) for a trap.

ROLE IN PIPELINE
----------------
Takes a 1D horizontal slice from the centre of each frame in the membrane
channel and stacks the slices vertically to produce a kymograph:
    Y-axis: time (top = start, bottom = end)
    X-axis: position along the aspiration channel

This module also writes the raw intensity values (before the 8-bit
normalization used for display) to a long-format CSV so the same signal
that appears in the kymograph image is available as data.

Outputs per trap:
    Trap_XX_Membrane_Kymograph.png   (plot; via Plotting_MFA.plot_kymograph)
    Trap_XX_Membrane_Kymograph.csv   (Time_s, Position_um, Intensity)

The dye-channel kymograph is produced separately by
UptakeQuantification.export_kymograph_csv, because that one is
column-averaged within the cell mask rather than a centre-line slice.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import Utils_MFA as utils
import Plotting_MFA

logger = logging.getLogger(__name__)


class KymographAnalysis:
    """
    Generates a kymograph and its underlying data CSV for a single channel.
    Only the membrane channel is processed here; see module docstring for
    why the dye channel is handled elsewhere.
    """

    def __init__(
        self,
        roi_images: List[np.ndarray],
        detection_results: Dict[str, Any],
        params: Dict[str, Any],
        time_data: Optional[List[float]] = None,
    ) -> None:
        self.roi_images = roi_images
        self.results = detection_results
        self.params = params
        self.time_data = time_data

        # Populated by run_analysis()
        self.display_matrix: Optional[np.ndarray] = None
        self.raw_matrix:     Optional[np.ndarray] = None

        self.growth_rate = None  # placeholder for legacy compatibility

        # Detection info reused for overlays and axes
        self.pipette_start_x = self.results.get('pipette_start_x_used')
        self.protrusions_px = self.results.get('protrusion_lengths_px', [])
        self.scale_factor = params.get('experiment_parameters', {}).get('scale_factor', 0.629)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run_analysis(self, output_dir: Union[str, Path], trap_index: int) -> 'KymographAnalysis':
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.roi_images or not self.pipette_start_x:
            logger.warning(
                f"Skipping kymograph for Trap {trap_index}: "
                f"missing membrane images or pipette info."
            )
            return self

        self.display_matrix, self.raw_matrix = self._create_kymograph_matrices(self.roi_images)
        if self.display_matrix is None:
            return self

        # Plot uses the 8-bit display matrix so contrast is comparable across
        # experiments regardless of absolute intensity.
        Plotting_MFA.plot_kymograph(
            kymograph_matrix=self.display_matrix,
            trap_index=trap_index,
            output_dir=output_dir,
            pipette_x=self.pipette_start_x,
            protrusions_px=self.protrusions_px,
            time_data=self.time_data,
            params=self.params,
            channel_label="Membrane",
        )

        # CSV stores the raw floating-point intensity so downstream code can
        # apply its own normalization if needed.
        self._export_csv(
            raw_matrix=self.raw_matrix,
            trap_index=trap_index,
            output_dir=output_dir,
        )

        return self

    # ------------------------------------------------------------------ #
    # Matrix construction
    # ------------------------------------------------------------------ #
    def _create_kymograph_matrices(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build display and raw matrices from a list of frames.
        Both have shape (n_frames, width).  Each row is the vertically
        averaged centre strip of one frame.
        """
        if not frames or frames[0] is None:
            return None, None

        first = frames[0]
        h, w = first.shape[:2]
        center_y = h // 2
        thickness = self.params.get('kymograph_parameters', {}).get('default_line_width_px', 10)
        y_start = max(0, center_y - thickness // 2)
        y_end   = min(h, center_y + thickness // 2 + 1)

        display_lines: List[np.ndarray] = []
        raw_lines:     List[np.ndarray] = []

        for frame in frames:
            if frame is None:
                display_lines.append(np.zeros((1, w), dtype=np.uint8))
                raw_lines.append(np.zeros((1, w), dtype=np.float32))
                continue

            # --- Raw row: original intensity, pre-normalization ------------
            frame_gray = frame
            if len(frame_gray.shape) == 3:
                frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
            raw_strip = frame_gray[y_start:y_end, :].astype(np.float32)
            raw_lines.append(raw_strip.mean(axis=0)[np.newaxis, :])

            # --- Display row: 8-bit-normalized for imshow ------------------
            img_8bit = utils.normalize_to_8bit(frame)
            if len(img_8bit.shape) == 3:
                img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
            disp_strip = img_8bit[y_start:y_end, :]
            display_lines.append(disp_strip.mean(axis=0).astype(np.uint8)[np.newaxis, :])

        return np.vstack(display_lines), np.vstack(raw_lines)

    # ------------------------------------------------------------------ #
    # CSV export
    # ------------------------------------------------------------------ #
    def _export_csv(
        self,
        raw_matrix: np.ndarray,
        trap_index: int,
        output_dir: Path,
    ) -> None:
        """
        Write the membrane kymograph to a long-format CSV.

        Columns
        -------
        Time_s
            Timestamp of the frame that generated the row.
        Position_um
            Distance from the pipette entrance in micrometres, using the
            same sign convention as the actin and uptake kymograph CSVs
            (negative = inside the channel, positive = outside).
        Intensity
            Vertically averaged centre-strip intensity of the raw frame at
            that column.
        """
        if raw_matrix is None or raw_matrix.size == 0:
            return

        n_frames, n_cols = raw_matrix.shape

        # Time axis: use time_data when we have enough entries, otherwise
        # fall back to frame indices with NaN padding on the tail.
        if self.time_data is not None and len(self.time_data) >= n_frames:
            t_axis = np.asarray(self.time_data[:n_frames], dtype=float)
        else:
            t_axis = np.arange(n_frames, dtype=float)
            if self.time_data is not None:
                fill = np.asarray(self.time_data, dtype=float)
                t_axis[:len(fill)] = fill
                t_axis[len(fill):] = np.nan

        pip_x = float(self.pipette_start_x)
        col_indices = np.arange(n_cols, dtype=float)
        position_um = (col_indices - pip_x) * self.scale_factor

        time_col = np.repeat(t_axis, n_cols)
        pos_col  = np.tile(position_um, n_frames)
        int_col  = raw_matrix.reshape(-1)

        df = pd.DataFrame({
            'Time_s':      time_col,
            'Position_um': pos_col,
            'Intensity':   int_col,
        })

        save_path = output_dir / f"Trap_{trap_index:02d}_Membrane_Kymograph.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Membrane kymograph CSV saved: {save_path.name}")


def create_kymograph_for_trap(
    roi_images: List[np.ndarray],
    detection_results: Dict[str, Any],
    params: Dict[str, Any],
    output_dir: Union[str, Path],
    trap_index: int,
    time_data: Optional[List[float]] = None,
) -> KymographAnalysis:
    """Helper function to instantiate and run the analysis."""
    analyzer = KymographAnalysis(
        roi_images=roi_images,
        detection_results=detection_results,
        params=params,
        time_data=time_data,
    )
    analyzer.run_analysis(output_dir, trap_index)
    return analyzer