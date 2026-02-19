# -*- coding: utf-8 -*-
"""
Quantification of Actin (C3) distribution.
Focuses on Protrusion-to-Body ratio and spatial profiles.
"""
import logging
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import Utils_MFA as utils

logger = logging.getLogger(__name__)

class ActinAnalyzer:
    def __init__(self, 
                 membrane_rois: List[np.ndarray], 
                 actin_rois: List[np.ndarray], 
                 pipette_x: int, 
                 threshold_prot: int,
                 threshold_body: int,
                 params: Dict[str, Any]):
        
        self.mem_imgs = membrane_rois
        self.actin_imgs = actin_rois
        self.pipette_x = pipette_x
        self.threshold_prot = threshold_prot
        self.threshold_body = threshold_body
        self.params = params
        
        # Pulse Timing Logic
        dye_params = params.get('dye_uptake_parameters', {})
        self.pulse_enabled = dye_params.get('enable', False)
        # Convert 1-based config frame to 0-based index
        self.pulse_frame = max(0, dye_params.get('pulse_frame', 10) - 1)
        self.window_size = dye_params.get('baseline_frames', 5)

        self.results = {
            'actin_protrusion_mean': [],
            'actin_body_mean': [],
            'actin_ratio_pb': [], # Protrusion / Body
            'spatial_profiles': [],
            'time_s': [],
            
            # New Comparison Data
            'profile_pre_pulse': None,
            'profile_post_pulse': None,
            'pulse_frame_index': self.pulse_frame if self.pulse_enabled else None
        }

    def run(self, time_data: List[float]) -> Dict[str, Any]:
        valid_frames = min(len(self.mem_imgs), len(self.actin_imgs))
        
        for i in range(valid_frames):
            mem_img = self.mem_imgs[i]
            act_img = self.actin_imgs[i]
            
            if mem_img is None or act_img is None:
                self._record_empty()
                continue
                
            # 1. Generate Masks (Using Membrane Channel)
            mask_prot, mask_body = utils.generate_dual_masks(
                mem_img, self.pipette_x, self.threshold_prot, 
                self.threshold_body, self.params
            )
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            
            # 2. Measure Actin Intensity
            # Note: We do NOT subtract baseline F0 for Actin (it's constitutive)
            mean_prot = cv2.mean(act_img, mask=mask_prot)[0]
            mean_body = cv2.mean(act_img, mask=mask_body)[0]
            
            # 3. Calculate Ratio
            ratio = (mean_prot / mean_body) if mean_body > 1.0 else 0.0
            
            # 4. Spatial Profile (Kymograph data)
            profile = utils.calculate_spatial_profile(act_img.astype(float), mask_total)
            
            self.results['actin_protrusion_mean'].append(mean_prot)
            self.results['actin_body_mean'].append(mean_body)
            self.results['actin_ratio_pb'].append(ratio)
            self.results['spatial_profiles'].append(profile)
            
        self.results['time_s'] = time_data[:len(self.results['actin_protrusion_mean'])]
        
        # --- 5. Calculate Pre/Post Pulse Profiles ---
        if self.pulse_enabled and self.results['spatial_profiles']:
            self._calculate_pulse_comparison()

        return self.results

    def _calculate_pulse_comparison(self):
        """Computes average spatial profiles before and after the pulse."""
        profiles = self.results['spatial_profiles']
        pf = self.pulse_frame
        win = self.window_size
        
        # Align profiles to the same length (max width) for averaging
        max_w = max(len(p) for p in profiles)
        aligned_profiles = np.zeros((len(profiles), max_w))
        for i, p in enumerate(profiles):
            aligned_profiles[i, :len(p)] = p
            
        # Define Indices
        # Pre: [Pulse - Window : Pulse]
        start_pre = max(0, pf - win)
        end_pre = pf
        
        # Post: [Pulse : Pulse + Window]
        start_post = pf
        end_post = min(len(profiles), pf + win)
        
        # Compute Averages
        if end_pre > start_pre:
            self.results['profile_pre_pulse'] = np.mean(aligned_profiles[start_pre:end_pre], axis=0)
        
        if end_post > start_post:
            self.results['profile_post_pulse'] = np.mean(aligned_profiles[start_post:end_post], axis=0)

    def _record_empty(self):
        for k in ['actin_protrusion_mean', 'actin_body_mean', 'actin_ratio_pb']:
            self.results[k].append(0)
        self.results['spatial_profiles'].append(np.zeros(10))

    def export_csv(self, trap_idx: int, output_dir: Path):
        df = pd.DataFrame({
            'Time_s': self.results['time_s'],
            'Actin_Protrusion_Mean': self.results['actin_protrusion_mean'],
            'Actin_Body_Mean': self.results['actin_body_mean'],
            'Actin_Ratio_PB': self.results['actin_ratio_pb']
        })
        save_path = output_dir / f"Trap_{trap_idx:02d}_Actin_Data.csv"
        df.to_csv(save_path, index=False)