# -*- coding: utf-8 -*-
"""
Quantification of Dye Uptake for Electroporation Experiments.
"""

import logging
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import Utils_MFA as utils

logger = logging.getLogger(__name__)

# NumPy 2.0 moved RankWarning into np.exceptions; fall back for older versions.
try:
    _NP_RANK_WARNING = np.exceptions.RankWarning
except AttributeError:
    _NP_RANK_WARNING = np.RankWarning

class DyeUptakeAnalyzer:
    """
    Analyzes fluorescence intensity within the mechanically trapped cell.
    """
    def __init__(self, 
                 membrane_rois: List[np.ndarray], 
                 dye_rois: List[np.ndarray], 
                 pipette_x: int, 
                 threshold_prot: int,
                 threshold_body: int,
                 rupture_idx: Optional[int],
                 params: Dict[str, Any],
                 start_idx: int = 0,
                 frame_masks: List[Tuple] = None):
        
        self.mem_imgs = membrane_rois
        self.dye_imgs = dye_rois
        self.pipette_x = pipette_x
        
        # Accept two distinct thresholds for segmentation
        self.threshold_prot = threshold_prot
        self.threshold_body = threshold_body if threshold_body is not None else threshold_prot
        
        self.rupture_idx = rupture_idx if rupture_idx is not None else len(membrane_rois)
        self.params = params
        self.start_idx = start_idx

        # Pre-computed per-frame masks from LineDetection — same source used by
        # protrusion length, kymograph, and actin quantification.
        self.frame_masks = frame_masks or [] 
        
        # Extract specific dye parameters from config
        dye_params = params.get('dye_uptake_parameters', {})
        # Pulse frame (0-based array index into the acquisition stack).
        # MFA_analysis resolves this once per experiment using the auto-
        # detected filename numbering scheme and stores the result under
        # dye_uptake_parameters -> pulse_frame_idx.  Prefer that resolved
        # value; fall back to the legacy "subtract one" rule with a warning.
        resolved_pf = dye_params.get('pulse_frame_idx', None)
        if resolved_pf is not None:
            self.pulse_frame = int(max(0, resolved_pf))
        else:
            self.pulse_frame = max(0, int(dye_params.get('pulse_frame', 10)) - 1)
            logger.warning(
                "DyeUptakeAnalyzer: pulse_frame_idx not provided; assuming "
                "filenames start at 1.  If the acquisition starts at t0000 "
                "the baseline window and pulse timing will be off by one frame."
            )
        self.baseline_len = dye_params.get('baseline_frames', 5)
        self.scale_factor = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        
        # Initialize results dictionary
        self.results = {
            'time_s': [],

            # ----- Primary CSV outputs (raw absolute intensities) -----
            # Un-subtracted mean per-pixel dye intensity in each region and
            # the raw pixel count for that region's mask.  These plus the
            # scalar F0 stored on self are the only quantities exported to
            # Trap_XX_Uptake_Data.csv.  All the other lists below are still
            # populated internally for the retained dashboard / debug plots.
            'raw_mean_body': [],
            'raw_mean_prot': [],
            'raw_mean_tip':  [],
            'mask_count_body': [],
            'mask_count_prot': [],
            'mask_count_tip':  [],

            # Baseline-subtracted intensities used by the dashboard plot.
            # Kept in memory only; not exported to CSV.
            'uptake_protrusion': [],
            'uptake_cell_body': [],
            'uptake_total': [],
            'uptake_tip': [],
            
            # Standard Deviations (Spatial Heterogeneity)
            'uptake_protrusion_std': [],
            'uptake_cell_body_std': [],
            'uptake_total_std': [],
            'uptake_tip_std': [],
            
            # Pixel Counts (For SEM calculation)
            'count_protrusion': [],
            'count_cell_body': [],
            'count_total': [],
            'count_tip': [],

            # Size metrics — identical formulas to LineDetection._calculate_morphology
            # (sf = scale_factor in μm/px, n = countNonZero pixels)
            'area_protrusion_um2': [],      # n × sf²          [μm²]
            'area_cell_body_um2': [],       # n × sf²          [μm²]
            'area_total_um2': [],           # n × sf²          [μm²]
            'linear_size_prot_um': [],      # sqrt(n) × sf     [μm]
            'linear_size_body_um': [],      # sqrt(n) × sf     [μm]
            'linear_size_total_um': [],     # sqrt(n) × sf     [μm]
            'volume_prot_um3': [],          # (n × sf²)^1.5    [μm³]
            'volume_body_um3': [],          # (n × sf²)^1.5    [μm³]
            'volume_total_um3': [],         # (n × sf²)^1.5    [μm³]
            
            # Baseline Normalized (dF/F0)
            'uptake_protrusion_norm': [],  
            'uptake_cell_body_norm': [],
            'uptake_total_norm': [],
            'uptake_tip_norm': [],
            
            # Normalized StdDevs
            'uptake_protrusion_norm_std': [],
            'uptake_cell_body_norm_std': [],
            'uptake_total_norm_std': [],
            'uptake_tip_norm_std': [],
            
            # Min-Max Normalized (0 to 1)
            'uptake_protrusion_minmax': [],
            'uptake_cell_body_minmax': [],
            'uptake_total_minmax': [],
            'uptake_tip_minmax': [],

            # Min-Max Scaled StdDevs
            'uptake_protrusion_minmax_std': [],
            'uptake_cell_body_minmax_std': [],
            'uptake_total_minmax_std': [],
            'uptake_tip_minmax_std': [],

            # Volume-Normalized (a.u./μm³)
            # Formula: val / volume_um3  where volume_um3 = (area_px × sf²)^1.5
            # Removes the cell-size confound: a larger cell accumulates more raw
            # signal simply because it contains more volume, independent of how
            # permeable its membrane is.  Dividing by volume makes uptake
            # directly comparable across cells of different sizes.
            'uptake_protrusion_vol_norm': [],
            'uptake_cell_body_vol_norm': [],
            'uptake_total_vol_norm': [],
            'uptake_protrusion_vol_norm_std': [],
            'uptake_cell_body_vol_norm_std': [],
            'uptake_total_vol_norm_std': [],

            'spatial_profiles': [],
            'baseline_intensity': None,  
            'pipette_x_px': pipette_x,
        }
        
    def _get_masks(self, frame_idx: int, mem_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mask_prot, mask_body) for a given frame index.

        Uses the pre-computed masks from LineDetection when available so that
        all modules (LineDetection, DyeUptake, ActinQuantification) operate on
        identical cell regions. Falls back to recomputing only if no pre-computed
        masks were supplied.
        """
        if self.frame_masks and frame_idx < len(self.frame_masks):
            mp, mb = self.frame_masks[frame_idx]
            if mp is not None and mb is not None:
                return mp, mb
        
        # Check is_guv before triggering the threshold fallback
        is_guv = self.params.get('guv_settings', {}).get('enable', False)
        if is_guv:
            logger.warning(f"Missing pre-computed masks for GUV at frame {frame_idx}. Skipping frame fallback.")
            return None, None
        
        return utils.generate_dual_masks(
            mem_img, self.pipette_x,
            self.threshold_prot, self.threshold_body, self.params
        )

    def run(self, time_data: List[float]) -> Dict[str, Any]:
        """
        Main execution loop with region-specific baseline correction and normalization.

        The analysis now processes every acquired frame regardless of whether
        a rupture was detected; the rupture frame is used only as a display
        annotation in the downstream plots.  Previously the frame loop was
        truncated at ``rupture_idx + 1``, which cut the kymograph and dye
        traces short and made post-rupture dye equilibration invisible.
        """

        # 1. Determine Processing Range
        # Cover the full acquisition (both channels).  rupture_idx is retained
        # on self so plots can still draw a red marker line.
        valid_frames = min(len(self.mem_imgs), len(self.dye_imgs))
        
        # 2. Calculate Baseline (Cell-Specific F0 for each region)
        dye_params = self.params.get('dye_uptake_parameters', {})
        has_pulse = dye_params.get('enable', False) and dye_params.get('has_pulse', True)
        
        if has_pulse:
            # Use frames immediately before the shock
            baseline_end = self.pulse_frame
            baseline_start = max(0, baseline_end - self.baseline_len)
        else:
            # For control experiments, use the first few frames after entry
            baseline_start = self.start_idx
            baseline_end = min(self.start_idx + self.baseline_len, valid_frames)
        
        base_vals_prot = []
        base_vals_body = []
        base_vals_total = []
        base_vals_tip = []

        for k in range(baseline_start, baseline_end):
            mem_ref_img = self.mem_imgs[k]
            dye_ref_img = self.dye_imgs[k]

            # Safety check: skip missing frames in baseline
            if mem_ref_img is None or dye_ref_img is None:
                logger.warning(f"Baseline frame {k} is missing. Skipping.")
                continue

            # Define the masks for baseline
            mask_prot_ref, mask_body_ref = self._get_masks(k, mem_ref_img)

            # Define mask_total for the baseline period
            mask_total_ref = cv2.bitwise_or(mask_prot_ref, mask_body_ref)

            # Geometric tip = far half of the protrusion mask (deepest in
            # channel).  Uses the shared helper so ActinQuantification and
            # DyeUptakeAnalyzer stay geometrically consistent.
            mask_tip_ref = utils.build_tip_mask(mask_prot_ref)

            # Collect mean intensities for region-specific F0
            base_vals_prot.append(cv2.mean(dye_ref_img, mask=mask_prot_ref)[0])
            base_vals_body.append(cv2.mean(dye_ref_img, mask=mask_body_ref)[0])
            base_vals_total.append(cv2.mean(dye_ref_img, mask=mask_total_ref)[0])
            if cv2.countNonZero(mask_tip_ref) > 0:
                base_vals_tip.append(cv2.mean(dye_ref_img, mask=mask_tip_ref)[0])

        # --- Compute the F0 values ---
        bg_total = np.mean(base_vals_total) if base_vals_total else 0.0
        bg_body = np.mean(base_vals_body) if base_vals_body else bg_total
        bg_prot = np.mean(base_vals_prot) if base_vals_prot else bg_total
        bg_tip  = np.mean(base_vals_tip)  if base_vals_tip  else bg_prot

        if bg_total == 0.0:
            logger.warning("Complete baseline failure. Fallback to global mean.")
            valid_imgs = [img for img in self.dye_imgs[baseline_start:baseline_end] if img is not None]
            fallback = np.mean([np.mean(img) for img in valid_imgs]) if valid_imgs else 0.0
            bg_total = bg_body = bg_prot = bg_tip = fallback

        # Persist F0 for downstream code (kept as a dict for parity with
        # ActinQuantification).
        self.f0 = {
            'body':  float(bg_body),
            'prot':  float(bg_prot),
            'tip':   float(bg_tip),
            'total': float(bg_total),
        }

        # Store for export
        self.results['baseline_intensity'] = bg_total
        
        # Calculate timing for logging
        pulse_time = time_data[min(self.pulse_frame, len(time_data)-1)] if time_data else 0
        
        logger.info(
            f"Dye Uptake Analysis:\n"
            f"   Start Frame: {self.start_idx+1} (Cell Entry)\n"
            + (f"   Pulse Frame: {self.pulse_frame+1} (t={pulse_time:.1f}s)\n" if has_pulse else "   No pulse applied (control experiment)\n")
            + f"   Baseline Intensity (Total F0): {bg_total:.2f} a.u."
        )
        
        # 3. Process Frames (Loop starts from start_idx)
        for i in range(self.start_idx, valid_frames):
            current_mem = self.mem_imgs[i]
            current_dye = self.dye_imgs[i]
            
            # Safety check to prevent NoneType propagation
            if current_mem is None or current_dye is None:
                self._record_empty_frame()
                continue
            
            # Call centralized utility
            mask_prot, mask_body = self._get_masks(i, current_mem)
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
                        
            # B. Quantify Pixel Counts (N) - Required for SEM calculation
            n_prot  = cv2.countNonZero(mask_prot)
            n_body  = cv2.countNonZero(mask_body)
            n_total = cv2.countNonZero(mask_total)

            self.results['count_protrusion'].append(n_prot)
            self.results['count_cell_body'].append(n_body)
            self.results['count_total'].append(n_total)

            # B2. Size metrics — same formulas as LineDetection._calculate_morphology
            # so values are directly comparable between the two modules.
            sf  = self.scale_factor       # μm/px
            sf2 = sf ** 2                 # (μm/px)²

            area_prot  = n_prot  * sf2          # μm²
            area_body  = n_body  * sf2          # μm²
            area_total = n_total * sf2          # μm²

            self.results['area_protrusion_um2'].append(area_prot)
            self.results['area_cell_body_um2'].append(area_body)
            self.results['area_total_um2'].append(area_total)

            self.results['linear_size_prot_um'].append(np.sqrt(n_prot)  * sf)
            self.results['linear_size_body_um'].append(np.sqrt(n_body)  * sf)
            self.results['linear_size_total_um'].append(np.sqrt(n_total) * sf)

            self.results['volume_prot_um3'].append(area_prot  ** 1.5)
            self.results['volume_body_um3'].append(area_body  ** 1.5)
            self.results['volume_total_um3'].append(area_total ** 1.5)
            
            # C. Quantify Dye Signal with Region-Specific Background Subtraction
            dye_float = current_dye.astype(float)

            # D. Raw and baseline-subtracted means.
            # The raw values feed the trimmed CSV; the baseline-subtracted
            # values feed the retained dashboard plot.
            raw_prot  = float(np.mean(dye_float[mask_prot  > 0])) if n_prot  > 0 else 0.0
            raw_body  = float(np.mean(dye_float[mask_body  > 0])) if n_body  > 0 else 0.0
            raw_total = float(np.mean(dye_float[mask_total > 0])) if n_total > 0 else 0.0

            val_prot  = raw_prot  - bg_prot  if n_prot  > 0 else 0.0
            val_body  = raw_body  - bg_body  if n_body  > 0 else 0.0
            val_total = raw_total - bg_total if n_total > 0 else 0.0

            std_prot  = float(np.std(dye_float[mask_prot  > 0])) if n_prot  > 0 else 0.0
            std_body  = float(np.std(dye_float[mask_body  > 0])) if n_body  > 0 else 0.0
            std_total = float(np.std(dye_float[mask_total > 0])) if n_total > 0 else 0.0

            # Store the raw versions -- these are the ones exported to CSV.
            self.results['raw_mean_body'].append(raw_body)
            self.results['raw_mean_prot'].append(raw_prot)
            self.results['mask_count_body'].append(int(n_body))
            self.results['mask_count_prot'].append(int(n_prot))
            
            # Geometric tip = far half of the protrusion mask.  Same
            # definition used by ActinQuantification.  Replaces the older
            # "top 5% brightest pixels" heuristic so tip intensity is a
            # physical region rather than a signal-selected subset.
            mask_tip = utils.build_tip_mask(mask_prot)
            n_tip = cv2.countNonZero(mask_tip)
            if n_tip > 0:
                tip_pixels = dye_float[mask_tip > 0]
                raw_tip = float(np.mean(tip_pixels))
                val_tip = raw_tip - bg_tip
                std_tip = float(np.std(tip_pixels))
            else:
                raw_tip, val_tip, std_tip = 0.0, 0.0, 0.0

            self.results['raw_mean_tip'].append(raw_tip)
            self.results['mask_count_tip'].append(int(n_tip))
            
            # E. Store Absolute Values
            self.results['uptake_protrusion'].append(val_prot)
            self.results['uptake_cell_body'].append(val_body)
            self.results['uptake_total'].append(val_total)
            self.results['uptake_tip'].append(val_tip)
            
            self.results['uptake_protrusion_std'].append(std_prot)
            self.results['uptake_cell_body_std'].append(std_body)
            self.results['uptake_total_std'].append(std_total)
            self.results['uptake_tip_std'].append(std_tip)
            
            self.results['count_tip'].append(n_tip)
            
            # F. Volume-Normalized Values (a.u./μm³)
            # volume_um3 = (area_px × sf²)^1.5 — already computed above.
            # Guard against zero area (cell not yet in frame / mask failure).
            vol_prot_frame  = area_prot  ** 1.5   # same value appended to volume_prot_um3
            vol_body_frame  = area_body  ** 1.5
            vol_total_frame = area_total ** 1.5

            self.results['uptake_protrusion_vol_norm'].append(
                val_prot  / vol_prot_frame  if vol_prot_frame  > 0 else 0.0)
            self.results['uptake_cell_body_vol_norm'].append(
                val_body  / vol_body_frame  if vol_body_frame  > 0 else 0.0)
            self.results['uptake_total_vol_norm'].append(
                val_total / vol_total_frame if vol_total_frame > 0 else 0.0)

            self.results['uptake_protrusion_vol_norm_std'].append(
                std_prot  / vol_prot_frame  if vol_prot_frame  > 0 else 0.0)
            self.results['uptake_cell_body_vol_norm_std'].append(
                std_body  / vol_body_frame  if vol_body_frame  > 0 else 0.0)
            self.results['uptake_total_vol_norm_std'].append(
                std_total / vol_total_frame if vol_total_frame > 0 else 0.0)

            # G. Calculate dF/F₀ Normalized Values
            epsilon = 1e-6
            self.results['uptake_protrusion_norm'].append(val_prot / bg_prot if bg_prot > epsilon else 0.0)
            self.results['uptake_cell_body_norm'].append(val_body / bg_body if bg_body > epsilon else 0.0)
            self.results['uptake_total_norm'].append(val_total / bg_total if bg_total > epsilon else 0.0)
            self.results['uptake_tip_norm'].append(val_tip / bg_prot if bg_prot > epsilon else 0.0)
            
            self.results['uptake_protrusion_norm_std'].append(std_prot / bg_prot if bg_prot > epsilon else 0.0)
            self.results['uptake_cell_body_norm_std'].append(std_body / bg_body if bg_body > epsilon else 0.0)
            self.results['uptake_total_norm_std'].append(std_total / bg_total if bg_total > epsilon else 0.0)
            self.results['uptake_tip_norm_std'].append(std_tip / bg_prot if bg_prot > epsilon else 0.0)
            
            # H. Spatial Profile
            dye_corrected_total = np.maximum(dye_float - bg_total, 0)
            profile = utils.calculate_spatial_profile(dye_corrected_total, mask_total)
            self.results['spatial_profiles'].append(profile)  
       
        # 4. Sync Time Vector
        processed_count = len(self.results['uptake_total'])
        self.results['time_s'] = time_data[self.start_idx : self.start_idx + processed_count]
        
        # 5. Min-Max Normalization (Means AND Stds)
        def normalize_pair(mean_key: str, std_key: str, out_mean: str, out_std: str):
            raw_mean = np.array(self.results[mean_key])
            raw_std = np.array(self.results[std_key])
            if len(raw_mean) == 0: return

            d_min, d_max = np.min(raw_mean), np.max(raw_mean)
            rng = d_max - d_min
            
            if rng == 0:
                self.results[out_mean] = np.zeros_like(raw_mean).tolist()
                self.results[out_std] = np.zeros_like(raw_std).tolist()
            else:
                self.results[out_mean] = ((raw_mean - d_min) / rng).tolist()
                self.results[out_std] = (raw_std / rng).tolist()

        normalize_pair('uptake_total', 'uptake_total_std', 'uptake_total_minmax', 'uptake_total_minmax_std')
        normalize_pair('uptake_protrusion', 'uptake_protrusion_std', 'uptake_protrusion_minmax', 'uptake_protrusion_minmax_std')
        normalize_pair('uptake_cell_body', 'uptake_cell_body_std', 'uptake_cell_body_minmax', 'uptake_cell_body_minmax_std')
        normalize_pair('uptake_tip', 'uptake_tip_std', 'uptake_tip_minmax', 'uptake_tip_minmax_std')
        
        return self.results
    
    def _record_empty_frame(self):
        """Appends zero values for all metrics when a frame is missing or invalid."""
        keys_to_zero = [
            # New primary CSV outputs
            'raw_mean_body', 'raw_mean_prot', 'raw_mean_tip',
            'mask_count_body', 'mask_count_prot', 'mask_count_tip',
            # Legacy internal metrics (still populated for the retained plots)
            'count_protrusion', 'count_cell_body', 'count_total', 'count_tip',
            'area_protrusion_um2', 'area_cell_body_um2', 'area_total_um2',
            'linear_size_prot_um', 'linear_size_body_um', 'linear_size_total_um',
            'volume_prot_um3', 'volume_body_um3', 'volume_total_um3',
            'uptake_protrusion', 'uptake_cell_body', 'uptake_total', 'uptake_tip',
            'uptake_protrusion_std', 'uptake_cell_body_std', 'uptake_total_std', 'uptake_tip_std',
            'uptake_protrusion_norm', 'uptake_cell_body_norm', 'uptake_total_norm', 'uptake_tip_norm',
            'uptake_protrusion_norm_std', 'uptake_cell_body_norm_std', 'uptake_total_norm_std', 'uptake_tip_norm_std',
            'uptake_protrusion_vol_norm', 'uptake_cell_body_vol_norm', 'uptake_total_vol_norm',
            'uptake_protrusion_vol_norm_std', 'uptake_cell_body_vol_norm_std', 'uptake_total_vol_norm_std',
        ]
        for key in keys_to_zero:
            self.results[key].append(0.0)
        self.results['spatial_profiles'].append(np.zeros(10))
    # =========================================================================
    # EXPORT DATA
    # =========================================================================
        
    def save_debug_video(self, trap_idx: int, output_dir: Path):
        """
        Save a video overlaying the generated masks onto the dye channel.

        The video now runs through every frame (rupture no longer
        truncates the loop).  When the acquisition reaches the detected
        rupture frame, a red vertical bar on the left edge and the label
        'RUPTURED' are drawn on every subsequent frame so the rupture
        onset is visible without cutting the sequence short.
        """
        if not self.results['time_s']:
            return
        save_path = output_dir / f"Trap_{trap_idx:02d}_Mask_Debug.avi"

        if not self.mem_imgs:
            return
        h, w = self.mem_imgs[0].shape[:2]
        fps = 5
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h), isColor=True)
        num_frames = len(self.results['time_s'])

        color_prot = utils.get_bgr_color('secondary')
        color_body = utils.get_bgr_color('tertiary')
        color_line = utils.get_bgr_color('text')
        if color_line == (0, 0, 0):
            color_line = (255, 255, 255)
        # BGR red for the rupture marker.  Kept independent of the palette
        # so it stays visible regardless of any theme swap.
        color_rupture = (0, 0, 255)

        # Rupture frame in the same index space we iterate over below.
        # self.rupture_idx is an index into the full mem/dye stack.
        rupture_local = None
        if self.rupture_idx is not None and self.rupture_idx < len(self.mem_imgs):
            rupture_local = self.rupture_idx - self.start_idx

        for i in range(num_frames):
            actual_idx = self.start_idx + i
            if actual_idx >= len(self.mem_imgs):
                break
            mem_img = self.mem_imgs[actual_idx]
            dye_img = self.dye_imgs[actual_idx]
            if mem_img is None or dye_img is None:
                continue

            mask_prot, mask_body = self._get_masks(actual_idx, mem_img)

            norm_dye = cv2.normalize(dye_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.cvtColor(norm_dye, cv2.COLOR_GRAY2BGR)

            overlay = display.copy()
            overlay[mask_prot > 0] = color_prot
            overlay[mask_body > 0] = color_body
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

            cv2.line(display, (int(self.pipette_x), 0),
                     (int(self.pipette_x), h), color_line, 1)

            # Post-rupture annotation.
            if rupture_local is not None and i >= rupture_local:
                # 4-px-wide red bar down the left edge of the frame.
                display[:, :4] = color_rupture
                cv2.putText(display, "RUPTURED", (10, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rupture, 1)

            if i < len(self.results['time_s']):
                time_s = self.results['time_s'][i]
                cv2.putText(display, f"t={time_s:.1f}s", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line, 1)

            out.write(display)

        out.release()
        logger.info(f"Saved mask debug video: {save_path.name}")

    def export_csv(self, trap_idx: int, output_dir: Path):
        """
        Save the trimmed uptake measurement to CSV.

        Columns
        -------
        Time_s
            Per-frame timestamp.
        Body_Intensity, Prot_Intensity, Tip_Intensity
            Raw un-subtracted mean dye intensity per pixel inside the
            corresponding mask.  Un-subtracted so downstream code can
            decide whether to apply an F0 correction.
        Body_Mask_Count, Prot_Mask_Count, Tip_Mask_Count
            Number of pixels contributing to the mean for each region.
            Multiply by scale_factor^2 to recover the mask area in um^2.
        F0_Body, F0_Prot, F0_Tip
            Per-cell pre-pulse (or post-entry, for control experiments)
            baseline of the dye signal in each region.  Scalar per trap,
            broadcast across all rows.  Divide Intensity by F0 to get the
            dF/F0 signal; subtract F0 from Intensity to get dF.

        What was removed
        ----------------
        All area, linear-size, volume, volume-normalized, dF/F0, min-max,
        Total-region, and standard-deviation columns.  Volume normalization
        moves to bulk_file_handling, which now derives area from the mask
        counts and applies region-appropriate geometry there.
        """
        n = len(self.results.get('time_s', []))
        if n == 0:
            logger.warning(f"Trap {trap_idx}: no uptake frames to export.")
            return

        def _col(key: str) -> list:
            v = self.results.get(key, [])
            return list(v) + [np.nan] * max(0, n - len(v))

        def _scalar(key: str) -> list:
            val = float(getattr(self, 'f0', {}).get(key, np.nan))
            return [val] * n

        df = pd.DataFrame({
            'Time_s':          self.results['time_s'],

            # Raw absolute intensities (mean per pixel in each mask)
            'Body_Intensity':  _col('raw_mean_body'),
            'Prot_Intensity':  _col('raw_mean_prot'),
            'Tip_Intensity':   _col('raw_mean_tip'),

            # Mask sizes (pixel counts) -- feed downstream area / volume math
            'Body_Mask_Count': _col('mask_count_body'),
            'Prot_Mask_Count': _col('mask_count_prot'),
            'Tip_Mask_Count':  _col('mask_count_tip'),

            # Per-region pre-pulse baseline (scalar broadcast to every row)
            'F0_Body':         _scalar('body'),
            'F0_Prot':         _scalar('prot'),
            'F0_Tip':          _scalar('tip'),

            # Scale factor (um/px) is broadcast to every row so downstream
            # code can convert mask counts to area without loading the
            # experiment's config.  Constant per acquisition.
            'Scale_Factor_um_per_px': [float(self.scale_factor)] * n,
        })

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"Trap_{trap_idx:02d}_Uptake_Data.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Uptake data saved: {save_path.name}")

    def export_kymograph_csv(self, trap_idx: int, output_dir: Path) -> None:
        """
        Save the dye kymograph values in long format.

        Format
        ------
        Time_s, Position_um, Intensity

        Data source
        -----------
        results['spatial_profiles'] -- one 1D array per frame containing the
        mean dye intensity per pixel column, averaged within the total
        (body + protrusion) mask.  This is the same data underlying the
        Trap_XX_Uptake_Kymograph.png plot from plot_dye_uptake_dashboard,
        so the CSV and the image show the same signal.

        The Position axis is in micrometres relative to the pipette
        entrance, using the same sign convention as the actin and
        membrane kymograph CSVs: negative = inside the channel (deeper),
        positive = outside.  This keeps the three kymograph CSVs directly
        overlayable frame by frame.
        """
        profiles = self.results.get('spatial_profiles', [])
        times    = self.results.get('time_s', [])
        if not profiles or len(times) == 0:
            logger.debug("No dye spatial profiles to export as kymograph CSV.")
            return

        n_frames = min(len(profiles), len(times))
        # Frames omitted from the loop (e.g. baseline-only frames) still
        # produce an entry in spatial_profiles via _record_empty_frame(),
        # so the row count matches results['time_s'] one-to-one.
        max_w = max((len(p) for p in profiles[:n_frames]), default=0)
        if max_w == 0:
            return

        pip_x = float(self.pipette_x)
        col_indices = np.arange(max_w, dtype=float)
        position_um = (col_indices - pip_x) * self.scale_factor

        kymo = np.full((n_frames, max_w), np.nan, dtype=float)
        for i in range(n_frames):
            p = np.asarray(profiles[i], dtype=float)
            kymo[i, :len(p)] = p

        t_axis = np.asarray(times[:n_frames], dtype=float)
        time_col = np.repeat(t_axis, max_w)
        pos_col  = np.tile(position_um, n_frames)
        int_col  = kymo.reshape(-1)

        df = pd.DataFrame({
            'Time_s':      time_col,
            'Position_um': pos_col,
            'Intensity':   int_col,
        })

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"Trap_{trap_idx:02d}_Uptake_Kymograph.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Uptake kymograph CSV saved: {save_path.name}")

   
    def compute_volume_correction(self):
        """
        Adds volume-corrected uptake to self.results.

        Uses 'volume_prot_um3' / 'volume_body_um3' computed in the frame loop:
            volume (μm³) = (count_px × sf²)^1.5

        Result is intensity per unit volume [a.u./μm³], removing the confound
        of cells with different sizes showing different raw uptake simply because
        they contain more volume.

        Call after run(). Adds keys:
            'uptake_protrusion_vol_corr'
            'uptake_cell_body_vol_corr'
        """
        vol_prot = np.array(self.results['volume_prot_um3'],   dtype=float)
        vol_body = np.array(self.results['volume_body_um3'],   dtype=float)
        upt_prot = np.array(self.results['uptake_protrusion'], dtype=float)
        upt_body = np.array(self.results['uptake_cell_body'],  dtype=float)

        corr_prot = np.where(vol_prot > 0, upt_prot / vol_prot, 0.0)
        corr_body = np.where(vol_body > 0, upt_body / vol_body, 0.0)

        self.results['uptake_protrusion_vol_corr'] = corr_prot.tolist()
        self.results['uptake_cell_body_vol_corr']  = corr_body.tolist()

        logger.info(
            "Volume correction applied.\n"
            f"   Protrusion: mean = {np.nanmean(corr_prot[corr_prot > 0]):.4e} a.u./\u03bcm\u00b3\n"
            f"   Cell body:  mean = {np.nanmean(corr_body[corr_body > 0]):.4e} a.u./\u03bcm\u00b3"
        )