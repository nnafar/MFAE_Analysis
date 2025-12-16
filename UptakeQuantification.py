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
        # NOTE: params is flattened in MFA_analysis, so keys are at top level
        image_params = params.get('image_processing', {})
        self.clip_margin = image_params.get('wall_clip_margin', 0.40)
        self.clahe_limit = image_params.get('clahe_clip_limit', 2.0)
        
        kernel = image_params.get('gaussian_kernel_size', (5,5))
        self.blur_kernel = tuple(kernel) if isinstance(kernel, (list, tuple)) else (5,5)
        
        # Load standard colors
        self.colors = utils.get_color_scheme('blue_red')
        
        self.results = {
            'time_s': [],
            'uptake_protrusion': [],
            'uptake_cell_body': [],
            'uptake_total': [],
            'spatial_profiles': [] 
        }
        
    def run(self, time_data: List[float]) -> Dict[str, Any]:
        """Main execution loop."""
        
        # 1. Calculate Baseline (Background Level)
        baseline_start = max(0, self.pulse_frame - self.baseline_len)
        baseline_end = self.pulse_frame
        
        baseline_imgs = self.dye_imgs[baseline_start:baseline_end]
        
        if not baseline_imgs:
            logger.warning("Pulse frame is too early; cannot calculate baseline. Using 0.")
            bg_level = 0.0
        else:
            bg_level = np.mean([np.mean(img) for img in baseline_imgs])

        logger.info(f"Dye Analysis: Pulse at frame {self.pulse_frame+1}, Baseline Background: {bg_level:.2f}")

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
            
            # --- Detect and stop at black frames ---
            # If the camera saved an empty frame at the end, break immediately.
            if np.mean(mem_img) < 1.0 or np.mean(dye_img) < 1.0:
                break
            
            # A. Create Dual Masks using the separate thresholds
            mask_prot, mask_body = self._generate_dual_masks(mem_img)
            
            # B. Quantify Dye Signal (Background Corrected)
            dye_float = dye_img.astype(float)
            dye_corrected = dye_float - bg_level
            dye_corrected[dye_corrected < 0] = 0 
            
            val_prot = cv2.mean(dye_corrected, mask=mask_prot)[0]
            val_body = cv2.mean(dye_corrected, mask=mask_body)[0]
            
            # Combine for total (using weighted average or union)
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            val_total = cv2.mean(dye_corrected, mask=mask_total)[0]
            
            self.results['uptake_protrusion'].append(val_prot)
            self.results['uptake_cell_body'].append(val_body)
            self.results['uptake_total'].append(val_total)
            
            # C. Spatial Profile (1D projection along X-axis)
            profile = self._calculate_spatial_profile(dye_corrected, mask_total)
            self.results['spatial_profiles'].append(profile)

        # --- Slice time data to match ACTUAL processed frames ---
        self.results['time_s'] = time_data[:len(self.results['uptake_total'])]
        
        return self.results

    def _generate_dual_masks(self, mem_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates separate masks for Protrusion and Cell Body using specific logic for each.
        Matches LineDetection preprocessing to ensure consistency.
        """
        img_8u = utils.normalize_to_8bit(mem_img)
        gray = img_8u if len(img_8u.shape) == 2 else cv2.cvtColor(img_8u, cv2.COLOR_BGR2GRAY)
        
        # 1. Preprocessing (Matched to LineDetection)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, self.blur_kernel, 0)
        
        h, w = gray.shape
        margin = int(h * self.clip_margin)
        pip_x = max(0, min(w, int(self.pipette_x)))
        
        # 2. Protrusion Mask (Left)
        # Logic: Use threshold_prot + Clip Walls + Clip Right of Pipette
        _, bin_prot = cv2.threshold(blurred, self.threshold_prot, 255, cv2.THRESH_BINARY)
        mask_prot = self._clean_mask(bin_prot)
        
        if margin > 0:
            mask_prot[:margin, :] = 0    # Clip Top
            mask_prot[h-margin:, :] = 0  # Clip Bottom
        mask_prot[:, pip_x:] = 0         # Clip Right (Body side)

        # 3. Body Mask (Right)
        # Logic: Use threshold_body + NO Wall Clipping + Clip Left of Pipette
        _, bin_body = cv2.threshold(blurred, self.threshold_body, 255, cv2.THRESH_BINARY)
        mask_body = self._clean_mask(bin_body)
        
        mask_body[:, :pip_x] = 0         # Clip Left (Protrusion side)

        return mask_prot, mask_body
    
    def _clean_mask(self, binary: np.ndarray) -> np.ndarray:
        """Helper for morphological cleanup."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        # Matches the _clean_binary_mask from LineDetection roughly
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
        self.results['uptake_protrusion'].append(0)
        self.results['uptake_cell_body'].append(0)
        self.results['uptake_total'].append(0)
        self.results['spatial_profiles'].append(np.zeros(10))

    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================

    def plot_results(self, trap_idx: int, output_dir: Path):
        """Generates all requested plots."""
        utils.set_paper_style() # Apply publication fonts/sizes
        
        # Force directory creation to prevent FileNotFoundError
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t = np.array(self.results['time_s'])
        
        if len(t) == 0:
            logger.warning(f"No data to plot for Trap {trap_idx}")
            return

        # 1. Uptake over Time
        self._plot_timecourse(trap_idx, output_dir, t)
        
        # 2. Spatial Kymograph (Heatmap)
        self._plot_kymograph(trap_idx, output_dir, t)
        
        # 3. Diffusion Curves (Line Profiles at specific times)
        self._plot_diffusion_curves(trap_idx, output_dir, t)

    def _plot_timecourse(self, trap_idx, output_dir, t):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        pulse_time = t[min(self.pulse_frame, len(t)-1)]
        ax.axvline(pulse_time, color='gray', linestyle='--', label='Pulse')
        
        # Use standard scheme: Data=Total(Grey), Fit=Protrusion(Blue), Alt=Body(Red)
        ax.plot(t, self.results['uptake_total'], '-', color=self.colors['data_points'], linewidth=2, label='Total')
        ax.plot(t, self.results['uptake_protrusion'], '-', color=self.colors['fitting_data'], alpha=0.8, label='Protrusion')
        ax.plot(t, self.results['uptake_cell_body'], '-', color=self.colors['model_fit_alt'], alpha=0.8, label='Body')
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Fluorescence (a.u.)")
        ax.set_title(f"Trap {trap_idx}: Dye Uptake")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Uptake_Timecourse.png")
        plt.close(fig)

    def _plot_kymograph(self, trap_idx, output_dir, t):
        profiles = self.results['spatial_profiles']
        if not profiles: return
        
        max_w = max(len(p) for p in profiles)
        kymo_matrix = np.zeros((len(profiles), max_w))
        for i, p in enumerate(profiles):
            kymo_matrix[i, :len(p)] = p
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Axis Flip: Protrusion (Left) = Positive, Body (Right) = Negative
        x_left_um = (self.pipette_x - 0) * self.scale_factor
        x_right_um = (self.pipette_x - max_w) * self.scale_factor
        
        extent = [x_left_um, x_right_um, t[-1], t[0]]
        
        im = ax.imshow(kymo_matrix, aspect='auto', extent=extent, cmap='inferno', interpolation='nearest')
        ax.axvline(0, color='cyan', linestyle='--', linewidth=1, label='Pipette Tip')
        ax.axhline(t[min(self.pulse_frame, len(t)-1)], color='white', linestyle='--', linewidth=1, label='Pulse')
        
        plt.colorbar(im, ax=ax, label="Intensity")
        ax.set_xlabel("Distance from Pipette Tip (μm)\n(Positive=Inside/Protrusion, Negative=Outside/Body)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Trap {trap_idx}: Uptake Kymograph")
        
        utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Uptake_Kymograph.png")
        plt.close(fig)

    def _plot_diffusion_curves(self, trap_idx, output_dir, t):
        profiles = self.results['spatial_profiles']
        if not profiles: return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_curves = 6
        indices = np.linspace(0, len(t)-1, n_curves, dtype=int)
        max_w = max(len(p) for p in profiles)
        
        x_axis_px = self.pipette_x - np.arange(max_w)
        x_axis_um = x_axis_px * self.scale_factor

        colors = cm.viridis(np.linspace(0, 1, len(indices)))
        
        for i, idx in enumerate(indices):
            if idx >= len(profiles): continue
            
            p = profiles[idx]
            y_data = np.zeros(max_w)
            y_data[:len(p)] = p
            
            time_label = f"{t[idx]:.1f}s"
            if idx == self.pulse_frame: time_label += " (Pulse)"
            
            ax.plot(x_axis_um, y_data, color=colors[i], linewidth=2, label=time_label)

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Pipette Tip')
        ax.invert_xaxis()
        
        ax.set_xlabel("Distance from Pipette Tip (μm)\n(Positive=Inside/Protrusion)")
        ax.set_ylabel("Fluorescence Intensity (a.u.)")
        ax.set_title(f"Trap {trap_idx}: Diffusion Profiles over Time")
        ax.legend(title="Time")
        ax.grid(True, alpha=0.3)
        
        utils.save_plot_png(output_dir / f"Trap_{trap_idx:02d}_Diffusion_Profiles.png")
        plt.close(fig)
        
    def save_debug_video(self, trap_idx: int, output_dir: Path):
        """
        Saves a video overlaying the generated masks onto the dye channel.
        Cyan = Protrusion Mask
        Magenta = Cell Body Mask
        """
        if not self.results['time_s']:
            return

        save_path = output_dir / f"Trap_{trap_idx:02d}_Mask_Debug.avi"
        
        # Get dimensions from the first valid frame
        h, w = self.mem_imgs[0].shape[:2]
        fps = 5  # Playback speed
        
        # Initialize Video Writer (MJPG is widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h), isColor=True)
        
        # Determine number of processed frames
        num_frames = len(self.results['uptake_total'])
        
        for i in range(num_frames):
            mem_img = self.mem_imgs[i]
            dye_img = self.dye_imgs[i]
            
            if mem_img is None or dye_img is None: 
                continue

            # 1. Re-generate masks
            mask_prot, mask_body = self._generate_dual_masks(mem_img)
            
            # 2. Create Base Image (Dye Channel)
            # Normalize for visibility in video
            norm_dye = cv2.normalize(dye_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.cvtColor(norm_dye, cv2.COLOR_GRAY2BGR)
            
            # 3. Overlay Protrusion (Cyan)
            overlay = display.copy()
            overlay[mask_prot > 0] = (255, 255, 0) # Cyan (BGR)
            
            # 4. Overlay Body (Magenta)
            overlay[mask_body > 0] = (255, 0, 255) # Magenta (BGR)
            
            # Blend
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            
            # 5. Draw Pipette Line
            cv2.line(display, (int(self.pipette_x), 0), (int(self.pipette_x), h), (255, 255, 255), 1)
            
            # 6. Add Frame Info
            time_s = self.results['time_s'][i]
            cv2.putText(display, f"t={time_s:.1f}s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            out.write(display)
            
        out.release()
        logger.info(f"Saved mask debug video: {save_path.name}")

    def export_csv(self, trap_idx: int, output_dir: Path):
        """Saves numerical results."""
        df = pd.DataFrame({
            'Time_s': self.results['time_s'],
            'Total_Mean_Intensity': self.results['uptake_total'],
            'Protrusion_Mean_Intensity': self.results['uptake_protrusion'],
            'Body_Mean_Intensity': self.results['uptake_cell_body']
        })
        
        # Force directory creation
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / f"Trap_{trap_idx:02d}_Uptake_Data.csv"
        df.to_csv(save_path, index=False)