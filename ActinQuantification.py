# -*- coding: utf-8 -*-
"""
Quantification of Actin (C3) distribution.
Focuses on Protrusion-to-Body ratio and spatial profiles.
"""
import logging
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Any, Tuple
from pathlib import Path

import Utils_MFA as utils
# We reuse the masking logic from UptakeQuantification to ensure consistency
from UptakeQuantification import DyeUptakeAnalyzer 

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
        self.params = params
        
        # Borrow mask generation from Dye Analyzer (DRY principle)
        # We instantiate it just to access its helper methods
        self._mask_helper = DyeUptakeAnalyzer(
            membrane_rois, [], pipette_x, threshold_prot, threshold_body, None, params
        )

        self.results = {
            'actin_protrusion_mean': [],
            'actin_body_mean': [],
            'actin_ratio_pb': [], # Protrusion / Body
            'spatial_profiles': [],
            'time_s': []
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
            mask_prot, mask_body = self._mask_helper._generate_dual_masks(mem_img)
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            
            # 2. Measure Actin Intensity
            mean_prot = cv2.mean(act_img, mask=mask_prot)[0]
            mean_body = cv2.mean(act_img, mask=mask_body)[0]
            
            # 3. Calculate Ratio
            ratio = (mean_prot / mean_body) if mean_body > 1.0 else 0.0
            
            # 4. Spatial Profile (Kymograph data)
            profile = self._mask_helper._calculate_spatial_profile(act_img.astype(float), mask_total)
            
            self.results['actin_protrusion_mean'].append(mean_prot)
            self.results['actin_body_mean'].append(mean_body)
            self.results['actin_ratio_pb'].append(ratio)
            self.results['spatial_profiles'].append(profile)
            
        self.results['time_s'] = time_data[:len(self.results['actin_protrusion_mean'])]
        return self.results

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