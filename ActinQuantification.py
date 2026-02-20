# -*- coding: utf-8 -*-
"""
Quantification of Actin (C3) distribution.
Calculates Relative Mean Intensity and Integrated Density.
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
        
        self.scale_factor = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
        dye_params = params.get('dye_uptake_parameters', {})
        self.baseline_len = dye_params.get('baseline_frames', 5)

        self.results = {
            'time_s': [],
            
            # Relative Mean Intensity (I_t / I_0)
            'actin_prot_rel': [],
            'actin_body_rel': [],
            'actin_total_rel': [],
            
            # Integrated Density
            'actin_integrated_density_total': [],
            
            # Basic Ratio
            'actin_ratio_pb': [], 
            
            'spatial_profiles': []
        }

    def run(self, time_data: List[float]) -> Dict[str, Any]:
        valid_frames = min(len(self.mem_imgs), len(self.actin_imgs))
        if valid_frames == 0: return self.results
        
        # --- 1. Calculate Baselines (I_0) ---
        base_prot, base_body, base_total = [], [], []
        calc_frames = min(self.baseline_len, valid_frames)
        
        for k in range(calc_frames):
            mem_img = self.mem_imgs[k]
            act_img = self.actin_imgs[k]
            if mem_img is None or act_img is None: continue
                
            mask_prot, mask_body = utils.generate_dual_masks(
                mem_img, self.pipette_x, self.threshold_prot, self.threshold_body, self.params
            )
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            
            base_prot.append(cv2.mean(act_img, mask=mask_prot)[0])
            base_body.append(cv2.mean(act_img, mask=mask_body)[0])
            base_total.append(cv2.mean(act_img, mask=mask_total)[0])
            
        f0_prot = np.mean(base_prot) if base_prot else 1.0
        f0_body = np.mean(base_body) if base_body else 1.0
        f0_total = np.mean(base_total) if base_total else 1.0
        
        # Prevent division by zero
        epsilon = 1e-6
        f0_prot = max(f0_prot, epsilon)
        f0_body = max(f0_body, epsilon)
        f0_total = max(f0_total, epsilon)

        # --- 2. Process Sequence ---
        for i in range(valid_frames):
            mem_img = self.mem_imgs[i]
            act_img = self.actin_imgs[i]
            
            if mem_img is None or act_img is None:
                self._record_empty()
                continue
                
            mask_prot, mask_body = utils.generate_dual_masks(
                mem_img, self.pipette_x, self.threshold_prot, self.threshold_body, self.params
            )
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            
            # Means
            mean_prot = cv2.mean(act_img, mask=mask_prot)[0]
            mean_body = cv2.mean(act_img, mask=mask_body)[0]
            mean_total = cv2.mean(act_img, mask=mask_total)[0]
            
            # Area (Total)
            area_total_um2 = cv2.countNonZero(mask_total) * (self.scale_factor ** 2)
            
            # Store Metrics
            self.results['actin_prot_rel'].append(mean_prot / f0_prot)
            self.results['actin_body_rel'].append(mean_body / f0_body)
            self.results['actin_total_rel'].append(mean_total / f0_total)
            
            self.results['actin_integrated_density_total'].append(mean_total * area_total_um2)
            
            self.results['actin_ratio_pb'].append(mean_prot / mean_body if mean_body > epsilon else 0.0)
            
            # Spatial Profile 
            profile = utils.calculate_spatial_profile(act_img.astype(float), mask_total)
            self.results['spatial_profiles'].append(profile)
            
        self.results['time_s'] = time_data[:len(self.results['actin_prot_rel'])]
        return self.results

    def _record_empty(self):
        for k in ['actin_prot_rel', 'actin_body_rel', 'actin_total_rel', 'actin_integrated_density_total', 'actin_ratio_pb']:
            self.results[k].append(0.0)
        self.results['spatial_profiles'].append(np.zeros(10))

    def export_csv(self, trap_idx: int, output_dir: Path):
        df = pd.DataFrame({
            'Time_s': self.results['time_s'],
            'Actin_Relative_Mean_Total': self.results['actin_total_rel'],
            'Actin_Relative_Mean_Protrusion': self.results['actin_prot_rel'],
            'Actin_Relative_Mean_Body': self.results['actin_body_rel'],
            'Actin_Integrated_Density_Total': self.results['actin_integrated_density_total'],
            'Actin_Ratio_PB': self.results['actin_ratio_pb']
        })
        save_path = output_dir / f"Trap_{trap_idx:02d}_Actin_Data.csv"
        df.to_csv(save_path, index=False)