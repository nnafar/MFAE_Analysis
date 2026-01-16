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
                 start_idx: int = 0):
        
        self.mem_imgs = membrane_rois
        self.dye_imgs = dye_rois
        self.pipette_x = pipette_x
        
        # Accept two distinct thresholds for segmentation
        self.threshold_prot = threshold_prot
        self.threshold_body = threshold_body if threshold_body is not None else threshold_prot
        
        self.rupture_idx = rupture_idx if rupture_idx is not None else len(membrane_rois)
        self.params = params
        self.start_idx = start_idx 
        
        # Extract specific dye parameters from config
        dye_params = params.get('dye_uptake_parameters', {})
        self.pulse_frame = max(0, dye_params.get('pulse_frame', 10) - 1)
        self.baseline_len = dye_params.get('baseline_frames', 5)
        self.scale_factor = params.get('scale_factor', 0.629)
        
        # --- INITIALIZE PROCESSING PARAMS (Required for Mask Gen) ---
        image_params = params.get('image_processing', {})
        self.clip_margin = image_params.get('wall_clip_margin', 0.40)
        self.clahe_limit = image_params.get('clahe_clip_limit', 2.0)
        
        kernel = image_params.get('gaussian_kernel_size', (5,5))
        self.blur_kernel = tuple(kernel) if isinstance(kernel, (list, tuple)) else (5,5)
        
        # Initialize results dictionary
        self.results = {
            'time_s': [],
            
            # Absolute Means (Corrected Intensity)
            'uptake_protrusion': [],
            'uptake_cell_body': [],
            'uptake_total': [],
            
            # Standard Deviations (Spatial Heterogeneity)
            'uptake_protrusion_std': [],
            'uptake_cell_body_std': [],
            'uptake_total_std': [],
            
            # Pixel Counts (For SEM calculation)
            'count_protrusion': [],
            'count_cell_body': [],
            'count_total': [],
            
            # Baseline Normalized (dF/F0)
            'uptake_protrusion_norm': [],  
            'uptake_cell_body_norm': [],
            'uptake_total_norm': [],
            
            # Normalized StdDevs
            'uptake_protrusion_norm_std': [],
            'uptake_cell_body_norm_std': [],
            'uptake_total_norm_std': [],
            
            # Min-Max Normalized (0 to 1)
            'uptake_protrusion_minmax': [],
            'uptake_cell_body_minmax': [],
            'uptake_total_minmax': [],

             # Min-Max Scaled StdDevs (NEW)
            'uptake_protrusion_minmax_std': [],
            'uptake_cell_body_minmax_std': [],
            'uptake_total_minmax_std': [],
            
            'spatial_profiles': [],
            'baseline_intensity': None,  # Store for export
            'pipette_x_px': pipette_x,
        }
        
    def run(self, time_data: List[float]) -> Dict[str, Any]:
        """
        Main execution loop with baseline correction and StdDev calculation.
        """
        
        # 1. Calculate Baseline (Cell-Specific F0)
        baseline_start = max(0, self.pulse_frame - self.baseline_len)
        baseline_end = self.pulse_frame
        
        baseline_vals = []
        
        # Iterate through specific baseline indices
        for k in range(baseline_start, baseline_end):
            # Check bounds
            if k >= len(self.mem_imgs) or k >= len(self.dye_imgs): continue
            
            mem_ref = self.mem_imgs[k]
            dye_ref = self.dye_imgs[k]
            
            if mem_ref is not None and dye_ref is not None:
                # Generate mask for this specific baseline frame
                mask_prot_ref, mask_body_ref = self._generate_dual_masks(mem_ref)
                mask_total_ref = cv2.bitwise_or(mask_prot_ref, mask_body_ref)
                
                # Measure dye intensity ONLY inside the cell mask
                mean_val = cv2.mean(dye_ref, mask=mask_total_ref)[0]
                
                # Only add valid measurements
                if mean_val > 0:
                    baseline_vals.append(mean_val)
        
        # Compute F0 (Average initial brightness of the cell)
        if baseline_vals:
            bg_level = np.mean(baseline_vals)
        else:
            # Fallback to global mean if detection failed entirely in baseline
            logger.warning("Could not detect cell in baseline frames. Fallback to Global Mean.")
            valid_baseline_imgs = [img for img in self.dye_imgs[baseline_start:baseline_end] if img is not None]
            if valid_baseline_imgs:
                bg_level = np.mean([np.mean(img) for img in valid_baseline_imgs])
            else:
                bg_level = 0.0

        # Store for export
        self.results['baseline_intensity'] = bg_level
        
        # Calculate timing for logging
        pulse_time = time_data[min(self.pulse_frame, len(time_data)-1)] if time_data else 0
        
        logger.info(
            f"Dye Uptake Analysis:\n"
            f"   Start Frame: {self.start_idx+1} (Cell Entry)\n"
            f"   Pulse Frame: {self.pulse_frame+1} (t={pulse_time:.1f}s)\n"
            f"   Baseline Intensity (F0): {bg_level:.2f} a.u."
        )

       # 2. Determine Processing Range
        valid_frames = min(len(self.mem_imgs), len(self.dye_imgs))
        if self.rupture_idx is not None:
             valid_frames = min(valid_frames, self.rupture_idx + 1)
        
        # 3. Process Frames (Loop starts from start_idx)
        for i in range(self.start_idx, valid_frames):
            mem_img = self.mem_imgs[i]
            dye_img = self.dye_imgs[i]
            
            if mem_img is None or dye_img is None:
                self._record_empty_frame()
                continue
            
            # Stop if image is effectively black/empty
            if np.mean(mem_img) < 1.0 or np.mean(dye_img) < 1.0:
                break
            
            # A. Create Dual Masks
            mask_prot, mask_body = self._generate_dual_masks(mem_img)
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            
            # B. Quantify Pixel Counts (N) - Required for SEM calculation
            n_prot = cv2.countNonZero(mask_prot)
            n_body = cv2.countNonZero(mask_body)
            n_total = cv2.countNonZero(mask_total)
            
            # C. Quantify Dye Signal (Baseline Subtracted)
            dye_float = dye_img.astype(float)
            dye_corrected = dye_float - bg_level
            dye_corrected[dye_corrected < 0] = 0
            
            # D. Quantify Mean AND Standard Deviation
            m_prot, s_prot = cv2.meanStdDev(dye_corrected, mask=mask_prot)
            m_body, s_body = cv2.meanStdDev(dye_corrected, mask=mask_body)
            m_total, s_total = cv2.meanStdDev(dye_corrected, mask=mask_total)
            
            # Extract scalar values
            val_prot = m_prot[0][0]; std_prot = s_prot[0][0]
            val_body = m_body[0][0]; std_body = s_body[0][0]
            val_total = m_total[0][0]; std_total = s_total[0][0]
            
            # E. Store Absolute Values
            self.results['uptake_protrusion'].append(val_prot)
            self.results['uptake_cell_body'].append(val_body)
            self.results['uptake_total'].append(val_total)
            
            self.results['uptake_protrusion_std'].append(std_prot)
            self.results['uptake_cell_body_std'].append(std_body)
            self.results['uptake_total_std'].append(std_total)
            
            self.results['count_protrusion'].append(n_prot)
            self.results['count_cell_body'].append(n_body)
            self.results['count_total'].append(n_total)
            
            # F. Calculate Normalized Values (ΔF/F₀)
            epsilon = 1e-6
            if bg_level > epsilon:
                norm_prot = val_prot / bg_level
                norm_body = val_body / bg_level
                norm_total = val_total / bg_level
                
                # Standard deviation scales linearly
                norm_std_prot = std_prot / bg_level
                norm_std_body = std_body / bg_level
                norm_std_total = std_total / bg_level
            else:
                if i == 0:
                    logger.warning("Baseline intensity is near zero. Normalization (dF/F0) disabled.")
                norm_prot = 0.0; norm_body = 0.0; norm_total = 0.0
                norm_std_prot = 0.0; norm_std_body = 0.0; norm_std_total = 0.0
            
            self.results['uptake_protrusion_norm'].append(norm_prot)
            self.results['uptake_cell_body_norm'].append(norm_body)
            self.results['uptake_total_norm'].append(norm_total)
            
            self.results['uptake_protrusion_norm_std'].append(norm_std_prot)
            self.results['uptake_cell_body_norm_std'].append(norm_std_body)
            self.results['uptake_total_norm_std'].append(norm_std_total)
            
            # G. Spatial Profile
            profile = self._calculate_spatial_profile(dye_corrected, mask_total)
            self.results['spatial_profiles'].append(profile)

        self.results['time_s'] = time_data[:len(self.results['uptake_total'])]
        
        # 4. Sync Time Vector
        processed_count = len(self.results['uptake_total'])
        self.results['time_s'] = time_data[self.start_idx : self.start_idx + processed_count]
        
        # 5. Min-Max Normalization (Means AND Stds)
        # Function to normalize both Mean and scale the Std by the same range
        def normalize_pair(mean_key, std_key, out_mean, out_std):
            raw_mean = np.array(self.results[mean_key])
            raw_std = np.array(self.results[std_key])
            if len(raw_mean) == 0: return

            d_min, d_max = np.min(raw_mean), np.max(raw_mean)
            rng = d_max - d_min
            
            if rng == 0:
                self.results[out_mean] = np.zeros_like(raw_mean).tolist()
                self.results[out_std] = np.zeros_like(raw_std).tolist()
            else:
                # Mean is shifted and scaled
                self.results[out_mean] = ((raw_mean - d_min) / rng).tolist()
                # Std is ONLY scaled (not shifted)
                self.results[out_std] = (raw_std / rng).tolist()

        normalize_pair('uptake_total', 'uptake_total_std', 'uptake_total_minmax', 'uptake_total_minmax_std')
        normalize_pair('uptake_protrusion', 'uptake_protrusion_std', 'uptake_protrusion_minmax', 'uptake_protrusion_minmax_std')
        normalize_pair('uptake_cell_body', 'uptake_cell_body_std', 'uptake_cell_body_minmax', 'uptake_cell_body_minmax_std')
        
        return self.results
    
    def _generate_dual_masks(self, mem_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates separate masks for Protrusion and Cell Body."""
        img_8u = utils.normalize_to_8bit(mem_img)
        gray = img_8u if len(img_8u.shape) == 2 else cv2.cvtColor(img_8u, cv2.COLOR_BGR2GRAY)
        
        # 1. Preprocessing
        clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, self.blur_kernel, 0)
        
        h, w = gray.shape
        margin = int(h * self.clip_margin)
        pip_x = max(0, min(w, int(self.pipette_x)))
        
        # 2. Protrusion Mask (Left)
        _, bin_prot = cv2.threshold(blurred, self.threshold_prot, 255, cv2.THRESH_BINARY)
        mask_prot = self._clean_mask(bin_prot)
        
        if margin > 0:
            mask_prot[:margin, :] = 0
            mask_prot[h-margin:, :] = 0
        mask_prot[:, pip_x:] = 0
        
        # --- Debris Removal ---
        mask_prot = self._keep_connected_to_pipette(mask_prot, pip_x)

        # 3. Body Mask (Right)
        _, bin_body = cv2.threshold(blurred, self.threshold_body, 255, cv2.THRESH_BINARY)
        mask_body = self._clean_mask(bin_body)
        mask_body[:, :pip_x] = 0

        return mask_prot, mask_body
    
    def _keep_connected_to_pipette(self, mask: np.ndarray, pip_x: int, tolerance: int = 5) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1: return mask
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            x, y, w, h_rect, area = stats[i]
            if (x + w) >= (pip_x - tolerance):
                new_mask[labels == i] = 255
        return new_mask
    
    def _clean_mask(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        return cleaned

    def _calculate_spatial_profile(self, dye_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        col_sums = np.sum(dye_img * (mask > 0), axis=0)
        col_counts = np.sum((mask > 0), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = col_sums / col_counts
            profile[col_counts == 0] = 0
        return profile

    def _record_empty_frame(self):
        """Records zeros for a missing frame."""
        for key in ['uptake_protrusion', 'uptake_cell_body', 'uptake_total',
                    'uptake_protrusion_std', 'uptake_cell_body_std', 'uptake_total_std',
                    'count_protrusion', 'count_cell_body', 'count_total',
                    'uptake_protrusion_norm', 'uptake_cell_body_norm', 'uptake_total_norm',
                    'uptake_protrusion_norm_std', 'uptake_cell_body_norm_std', 'uptake_total_norm_std',
                    'uptake_total_minmax', 'uptake_protrusion_minmax', 'uptake_cell_body_minmax',
                    'uptake_total_minmax_std', 'uptake_protrusion_minmax_std', 'uptake_cell_body_minmax_std']:
            self.results[key].append(0)
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

            mask_prot, mask_body = self._generate_dual_masks(mem_img)
            
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
        """Saves numerical results including baseline info and min-max normalization."""
        df = pd.DataFrame({
            'Time_s': self.results['time_s'],
            
            # Absolute Data
            'Total_Intensity_Mean': self.results['uptake_total'],
            'Total_Intensity_Std': self.results['uptake_total_std'],
            'Count_Total_Px': self.results['count_total'],
            
            'Protrusion_Mean': self.results['uptake_protrusion'],
            'Protrusion_Std': self.results['uptake_protrusion_std'],
            'Count_Protrusion_Px': self.results['count_protrusion'],
            
            'Body_Mean': self.results['uptake_cell_body'],
            'Body_Std': self.results['uptake_cell_body_std'],
            'Count_Body_Px': self.results['count_cell_body'],
            
            # Baseline Normalized (dF/F0)
            'Total_Normalized_dF_F0': self.results['uptake_total_norm'],
            'Protrusion_Normalized_dF_F0': self.results['uptake_protrusion_norm'],
            'Body_Normalized_dF_F0': self.results['uptake_cell_body_norm'],
            
            # Min-Max Normalized (0-1)
            'Total_MinMax': self.results['uptake_total_minmax'],
            'Protrusion_MinMax': self.results['uptake_protrusion_minmax'],
            'Body_MinMax': self.results['uptake_cell_body_minmax'],
            
            # Min-Max Scaled StdDevs
            'Total_MinMax_Std': self.results['uptake_total_minmax_std'],
            'Protrusion_MinMax_Std': self.results['uptake_protrusion_minmax_std'],
            'Body_MinMax_Std': self.results['uptake_cell_body_minmax_std']
        })
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"Trap_{trap_idx:02d}_Uptake_Data.csv"
        
        # Add baseline as header comment
        # newline='' prevents double spacing on Windows
        with open(save_path, 'w', newline='') as f:
            f.write(f"# Baseline Intensity (Cell Mask): {self.results['baseline_intensity']:.2f} a.u.\n")
            f.write(f"# Pulse Frame: {self.pulse_frame + 1}\n")
            df.to_csv(f, index=False)
        
        logger.info(f"Saved uptake data: {save_path.name}")