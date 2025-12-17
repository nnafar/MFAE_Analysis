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
    def __init__(self, 
                 membrane_rois: List[np.ndarray], 
                 dye_rois: List[np.ndarray], 
                 pipette_x: int, 
                 threshold_prot: int,
                 threshold_body: int,
                 rupture_idx: Optional[int],
                 params: Dict[str, Any]):
        
        self.mem_imgs = membrane_rois
        self.dye_imgs = dye_rois
        self.pipette_x = pipette_x
        
        # Accept two distinct thresholds
        self.threshold_prot = threshold_prot
        self.threshold_body = threshold_body if threshold_body is not None else threshold_prot
        
        self.rupture_idx = rupture_idx if rupture_idx is not None else len(membrane_rois)
        self.params = params
        
        # Extract specific dye parameters
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
        
        self.results = {
            'time_s': [],
            'uptake_protrusion': [],
            'uptake_cell_body': [],
            'uptake_total': [],
            'uptake_protrusion_norm': [],  # ΔF/F₀ (normalized)
            'uptake_cell_body_norm': [],
            'uptake_total_norm': [],
            'spatial_profiles': [],
            'baseline_intensity': None,  # Store for export
            'pipette_x_px': pipette_x,
        }
        
    def run(self, time_data: List[float]) -> Dict[str, Any]:
        """
        Main execution loop with baseline correction.
        
        BASELINE CALCULATION:
        Takes the N frames immediately BEFORE the pulse as baseline.
        This captures the cell's pre-pulse state, including any mechanical
        permeabilization from aspiration, and isolates the electrical effect.
        """
        
        # 1. Calculate Baseline (Background Level)
        baseline_start = max(0, self.pulse_frame - self.baseline_len)
        baseline_end = self.pulse_frame
        
        baseline_imgs = self.dye_imgs[baseline_start:baseline_end]
        
        if not baseline_imgs:
            logger.warning("Pulse frame is too early; cannot calculate baseline. Using 0.")
            bg_level = 0.0
        else:
            bg_level = np.mean([np.mean(img) for img in baseline_imgs])
        
        # Store for export
        self.results['baseline_intensity'] = bg_level
        
        # Calculate timing for logging
        pulse_time = time_data[min(self.pulse_frame, len(time_data)-1)] if time_data else 0
        baseline_start_time = time_data[baseline_start] if baseline_start < len(time_data) else 0
        baseline_end_time = time_data[baseline_end-1] if baseline_end > 0 and baseline_end-1 < len(time_data) else 0
        
        logger.info(
            f"Dye Uptake Analysis:\n"
            f"   Pulse: Frame {self.pulse_frame+1} (t={pulse_time:.1f}s)\n"
            f"   Baseline: Frames {baseline_start+1}-{baseline_end} "
            f"(t={baseline_start_time:.1f}-{baseline_end_time:.1f}s)\n"
            f"   Baseline Intensity: {bg_level:.2f} a.u."
        )

        # 2. Process Frames
        valid_frames = min(len(self.mem_imgs), len(self.dye_imgs))
        if self.rupture_idx is not None:
             valid_frames = min(valid_frames, self.rupture_idx + 1)
        
        for i in range(valid_frames):
            mem_img = self.mem_imgs[i]
            dye_img = self.dye_imgs[i]
            
            # Skip if None
            if mem_img is None or dye_img is None:
                self._record_empty_frame()
                continue
            
            # Detect and stop at black frames
            if np.mean(mem_img) < 1.0 or np.mean(dye_img) < 1.0:
                break
            
            # A. Create Dual Masks
            mask_prot, mask_body = self._generate_dual_masks(mem_img)
            
            # B. Quantify Dye Signal (Background Corrected)
            dye_float = dye_img.astype(float)
            dye_corrected = dye_float - bg_level
            dye_corrected[dye_corrected < 0] = 0
            
            # C. Quantify Mean Intensity in Each Region
            val_prot = cv2.mean(dye_corrected, mask=mask_prot)[0]
            val_body = cv2.mean(dye_corrected, mask=mask_body)[0]
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            val_total = cv2.mean(dye_corrected, mask=mask_total)[0]
            
            # D. Store Absolute Values
            self.results['uptake_protrusion'].append(val_prot)
            self.results['uptake_cell_body'].append(val_body)
            self.results['uptake_total'].append(val_total)
            
            # E. Calculate and Store Normalized Values (ΔF/F₀)
            epsilon = 1e-6
            if bg_level > epsilon:
                norm_prot = val_prot / bg_level
                norm_body = val_body / bg_level
                norm_total = val_total / bg_level
            else:
                if i == 0:
                    logger.warning("Baseline intensity is near zero. Normalization (dF/F0) disabled.")
                norm_prot = 0.0; norm_body = 0.0; norm_total = 0.0
            
            self.results['uptake_protrusion_norm'].append(norm_prot)
            self.results['uptake_cell_body_norm'].append(norm_body)
            self.results['uptake_total_norm'].append(norm_total)
            
            # F. Spatial Profile (1D projection along X-axis)
            profile = self._calculate_spatial_profile(dye_corrected, mask_total)
            self.results['spatial_profiles'].append(profile)

        self.results['time_s'] = time_data[:len(self.results['uptake_total'])]
        return self.results

    def _generate_dual_masks(self, mem_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates separate masks for Protrusion and Cell Body using specific logic for each.
        Matches LineDetection preprocessing to ensure consistency.
        """
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
            mask_prot[:margin, :] = 0    # Clip Top
            mask_prot[h-margin:, :] = 0  # Clip Bottom
        mask_prot[:, pip_x:] = 0         # Clip Right (Body side)

       # 3. Body Mask (Right)
        _, bin_body = cv2.threshold(blurred, self.threshold_body, 255, cv2.THRESH_BINARY)
        mask_body = self._clean_mask(bin_body)
        mask_body[:, :pip_x] = 0         # Clip Left (Protrusion side)

        return mask_prot, mask_body
    
    def _clean_mask(self, binary: np.ndarray) -> np.ndarray:
        """Helper for morphological cleanup."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        return cleaned

    def _calculate_spatial_profile(self, dye_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Calculates average intensity column-by-column (X-axis)."""
        col_sums = np.sum(dye_img * (mask > 0), axis=0)
        col_counts = np.sum((mask > 0), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = col_sums / col_counts
            profile[col_counts == 0] = 0
        return profile

    def _record_empty_frame(self):
        """Records zeros for a missing frame."""
        self.results['uptake_protrusion'].append(0)
        self.results['uptake_cell_body'].append(0)
        self.results['uptake_total'].append(0)
        self.results['uptake_protrusion_norm'].append(0)
        self.results['uptake_cell_body_norm'].append(0)
        self.results['uptake_total_norm'].append(0)
        self.results['spatial_profiles'].append(np.zeros(10))

    # =========================================================================
    # EXPORT DATA
    # =========================================================================
        
    def save_debug_video(self, trap_idx: int, output_dir: Path):
        """
        Saves a video overlaying the generated masks onto the dye channel.
        """
        if not self.results['time_s']: return
        save_path = output_dir / f"Trap_{trap_idx:02d}_Mask_Debug.avi"
        
        # Get dimensions from the first valid frame
        h, w = self.mem_imgs[0].shape[:2]
        fps = 5
        
        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h), isColor=True)
        num_frames = len(self.results['uptake_total'])
        
        # Use centralized colors
        # Secondary = Red (Protrusion), Tertiary = Blue (Body)
        color_prot = utils.get_bgr_color('secondary') 
        color_body = utils.get_bgr_color('tertiary')
        color_line = utils.get_bgr_color('text') # Black/White contrast depending on palette, default White for OpenCV
        if color_line == (0,0,0): color_line = (255,255,255)
        
        for i in range(num_frames):
            mem_img = self.mem_imgs[i]
            dye_img = self.dye_imgs[i]
            if mem_img is None or dye_img is None: continue

            # Re-generate masks
            mask_prot, mask_body = self._generate_dual_masks(mem_img)
            
            # Create Base Image
            norm_dye = cv2.normalize(dye_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.cvtColor(norm_dye, cv2.COLOR_GRAY2BGR)
            
            # Overlay masks
            overlay = display.copy()
            overlay[mask_prot > 0] = color_prot
            overlay[mask_body > 0] = color_body
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            
            # Draw Pipette Line
            cv2.line(display, (int(self.pipette_x), 0), (int(self.pipette_x), h), color_line, 1)
            
            # Add Frame Info
            time_s = self.results['time_s'][i]
            cv2.putText(display, f"t={time_s:.1f}s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line, 1)
            out.write(display)
            
        out.release()
        logger.info(f"Saved mask debug video: {save_path.name}")

    def export_csv(self, trap_idx: int, output_dir: Path):
        """Saves numerical results including baseline info."""
        df = pd.DataFrame({
            'Time_s': self.results['time_s'],
            'Total_Intensity_Corrected': self.results['uptake_total'],
            'Protrusion_Intensity_Corrected': self.results['uptake_protrusion'],
            'Body_Intensity_Corrected': self.results['uptake_cell_body'],
            'Total_Normalized_dF_F0': self.results['uptake_total_norm'],
            'Protrusion_Normalized_dF_F0': self.results['uptake_protrusion_norm'],
            'Body_Normalized_dF_F0': self.results['uptake_cell_body_norm']
        })
        
        # Force directory creation
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"Trap_{trap_idx:02d}_Uptake_Data.csv"
        
        # Add baseline as header comment
        with open(save_path, 'w') as f:
            f.write(f"# Baseline Intensity: {self.results['baseline_intensity']:.2f} a.u.\n")
            f.write(f"# Pulse Frame: {self.pulse_frame + 1}\n")
            df.to_csv(f, index=False)
        
        logger.info(f"Saved uptake data: {save_path.name}")