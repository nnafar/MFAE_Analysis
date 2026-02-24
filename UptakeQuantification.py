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
        self.pulse_frame = max(0, dye_params.get('pulse_index', 9))
        self.baseline_len = dye_params.get('baseline_frames', 5)
        self.scale_factor = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        
        # Initialize results dictionary
        self.results = {
            'time_s': [],
            
            # Absolute Means (Corrected Intensity)
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
        return utils.generate_dual_masks(
            mem_img, self.pipette_x,
            self.threshold_prot, self.threshold_body, self.params
        )

    def run(self, time_data: List[float]) -> Dict[str, Any]:
        """
        Main execution loop with region-specific baseline correction and normalization.
        """
        
        # 1. Determine Processing Range
        valid_frames = min(len(self.mem_imgs), len(self.dye_imgs))
        if self.rupture_idx is not None:
             valid_frames = min(valid_frames, self.rupture_idx + 1)
        
        # 2. Calculate Baseline (Cell-Specific F0 for each region)
        dye_params = self.params.get('dye_uptake_parameters', {})
        has_pulse = dye_params.get('enable', False) and dye_params.get('has_pulse', True)
        
        if has_pulse:
            baseline_end = self.pulse_frame
            baseline_start = max(0, baseline_end - self.baseline_len)
        else:
            baseline_start = self.start_idx
            baseline_end = min(self.start_idx + self.baseline_len, valid_frames)
        
        base_vals_prot = []
        base_vals_body = []
        base_vals_total = []
        
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
            
            # Collect mean intensities for region-specific F0
            base_vals_prot.append(cv2.mean(dye_ref_img, mask=mask_prot_ref)[0])
            base_vals_body.append(cv2.mean(dye_ref_img, mask=mask_body_ref)[0])
            base_vals_total.append(cv2.mean(dye_ref_img, mask=mask_total_ref)[0])
            
        # --- Compute the F0 values ---
        bg_total = np.mean(base_vals_total) if base_vals_total else 0.0
        bg_body = np.mean(base_vals_body) if base_vals_body else bg_total
        bg_prot = np.mean(base_vals_prot) if base_vals_prot else bg_total

        if bg_total == 0.0:
            logger.warning("Complete baseline failure. Fallback to global mean.")
            valid_imgs = [img for img in self.dye_imgs[baseline_start:baseline_end] if img is not None]
            bg_total = bg_body = bg_prot = np.mean([np.mean(img) for img in valid_imgs]) if valid_imgs else 0.0
        
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
            n_prot = cv2.countNonZero(mask_prot)
            n_body = cv2.countNonZero(mask_body)
            n_total = cv2.countNonZero(mask_total)
            
            self.results['count_protrusion'].append(n_prot)
            self.results['count_cell_body'].append(n_body)
            self.results['count_total'].append(n_total)
            
            # C. Quantify Dye Signal with Region-Specific Background Subtraction
            dye_float = current_dye.astype(float)
            
            # D. Calculate Means and Standard Deviations Inline
            val_prot = (np.mean(dye_float[mask_prot > 0]) - bg_prot) if n_prot > 0 else 0.0
            val_body = (np.mean(dye_float[mask_body > 0]) - bg_body) if n_body > 0 else 0.0
            val_total = (np.mean(dye_float[mask_total > 0]) - bg_total) if n_total > 0 else 0.0
            
            std_prot = np.std(dye_float[mask_prot > 0]) if n_prot > 0 else 0.0
            std_body = np.std(dye_float[mask_body > 0]) if n_body > 0 else 0.0
            std_total = np.std(dye_float[mask_total > 0]) if n_total > 0 else 0.0
            
            # Isolate the leading edge (Top 5% brightest pixels in the protrusion)
            if n_prot > 10:
                prot_pixels = dye_float[mask_prot > 0]
                threshold_95 = np.percentile(prot_pixels, 95)
                tip_pixels = prot_pixels[prot_pixels >= threshold_95]
                val_tip = np.mean(tip_pixels) - bg_prot
                std_tip = np.std(tip_pixels)
                n_tip = len(tip_pixels)
            else:
                val_tip, std_tip, n_tip = 0.0, 0.0, 0
            
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
            
            # F. Calculate Normalized Values (ΔF/F₀)
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
            'count_protrusion', 'count_cell_body', 'count_total', 'count_tip',
            'uptake_protrusion', 'uptake_cell_body', 'uptake_total', 'uptake_tip',
            'uptake_protrusion_std', 'uptake_cell_body_std', 'uptake_total_std', 'uptake_tip_std',
            'uptake_protrusion_norm', 'uptake_cell_body_norm', 'uptake_total_norm', 'uptake_tip_norm',
            'uptake_protrusion_norm_std', 'uptake_cell_body_norm_std', 'uptake_total_norm_std', 'uptake_tip_norm_std'
        ]
        for key in keys_to_zero:
            self.results[key].append(0.0)
        self.results['spatial_profiles'].append(np.zeros(10))
    # =========================================================================
    # EXPORT DATA
    # =========================================================================
        
    def save_debug_video(self, trap_idx: int, output_dir: Path):
        """Saves a video overlaying the generated masks onto the dye channel."""
        if not self.results['time_s']: return
        save_path = output_dir / f"Trap_{trap_idx:02d}_Mask_Debug.avi"
        
        if not self.mem_imgs: return
        h, w = self.mem_imgs[0].shape[:2]
        fps = 5
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h), isColor=True)
        num_frames = len(self.results['uptake_total'])
        
        color_prot = utils.get_bgr_color('secondary') 
        color_body = utils.get_bgr_color('tertiary')
        color_line = utils.get_bgr_color('text') 
        if color_line == (0,0,0): color_line = (255,255,255)
        
        for i in range(num_frames):
            actual_idx = self.start_idx + i
            if actual_idx >= len(self.mem_imgs): break
            mem_img = self.mem_imgs[actual_idx]
            dye_img = self.dye_imgs[actual_idx]
            if mem_img is None or dye_img is None: continue

            mask_prot, mask_body = self._get_masks(actual_idx, mem_img)
            
            norm_dye = cv2.normalize(dye_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.cvtColor(norm_dye, cv2.COLOR_GRAY2BGR)
            
            overlay = display.copy()
            overlay[mask_prot > 0] = color_prot
            overlay[mask_body > 0] = color_body
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            
            cv2.line(display, (int(self.pipette_x), 0), (int(self.pipette_x), h), color_line, 1)
            
            if i < len(self.results['time_s']):
                time_s = self.results['time_s'][i]
                cv2.putText(display, f"t={time_s:.1f}s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line, 1)
            
            out.write(display)
            
        out.release()
        logger.info(f"Saved mask debug video: {save_path.name}")

    def export_csv(self, trap_idx: int, output_dir: Path):
        """Saves the primary uptake kinetics to CSV."""
        df = pd.DataFrame({
            'Time_s': self.results['time_s'],
            
            # Absolute Values (Background Subtracted)
            'Total_Intensity': self.results['uptake_total'],
            'Protrusion_Intensity': self.results['uptake_protrusion'],
            'Tip_Intensity': self.results['uptake_tip'],
            'Body_Intensity': self.results['uptake_cell_body'],
            
            # Baseline Normalized (dF/F0)
            'Total_Normalized_dF_F0': self.results['uptake_total_norm'],
            'Protrusion_Normalized_dF_F0': self.results['uptake_protrusion_norm'],
            'Tip_Normalized_dF_F0': self.results['uptake_tip_norm'],
            'Body_Normalized_dF_F0': self.results['uptake_cell_body_norm'],
            
            # Min-Max Normalized (0-1)
            'Total_MinMax': self.results['uptake_total_minmax'],
            'Protrusion_MinMax': self.results['uptake_protrusion_minmax'],
            'Tip_MinMax': self.results['uptake_tip_minmax'],
            'Body_MinMax': self.results['uptake_cell_body_minmax'],

        })
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"Trap_{trap_idx:02d}_Uptake_Data.csv"
        df.to_csv(save_path, index=False)