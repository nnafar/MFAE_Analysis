# -*- coding: utf-8 -*-
"""
Orchestrates the Microfluidic Aspiration (MFA) analysis pipeline.

ROLE IN PIPELINE:
This is the main entry point. It coordinates the entire workflow:
1.  Loads configuration and experimental data.
2.  Runs the Interactive Setup (GUI) for users to select traps.
3.  Interactive Preview: Verifies detection on the first trap.
4.  Manages Parallel Processing to analyze all traps.
5.  Aggregates results.
"""

import atexit
import os
import gc
import argparse
import yaml
import multiprocessing
import traceback
import logging
import cv2
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from pydantic import ValidationError

# Internal Imports
from FileHandling_MFA import FileRead, FileSave
from ImageProcessing_MFA import CropImage
from LineDetection_MFA import LineDetectionMFA
from Calculation_MFA import compute_reff, compute_shear_metrics
from Fitting_MFA import FittingMFA
from Kymograph_MFA import create_kymograph_for_trap
from UptakeQuantification import DyeUptakeAnalyzer
from ActinQuantification import ActinAnalyzer
from config_schema import MFAConfig, validate_config
import Plotting_MFA
import Utils_MFA as utils

logger = logging.getLogger(__name__)

# =============================================================================
# TOP-LEVEL STATIC WORKER FUNCTION (Optimized for Shared Memory)
# =============================================================================

def static_parallel_worker(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single trap in an independent process with guaranteed cleanup.
    """
    # Prevent OpenCV from spawning internal threads inside the worker
    cv2.setNumThreads(0)
    
    trap_index = config_dict['trap_index']
    params = config_dict['params']
    
    # Initialize handles to None for safe reference in the finally block
    all_frames = None
    all_dye_frames = None
    all_actin_frames = None
    
    # Unpack Output Directories
    dirs = config_dict['output_dirs']
    worker_logger = logging.getLogger(f"Worker-{trap_index+1:02d}")
    
    try:
        worker_logger.info(f"Starting analysis for Trap {trap_index+1}...")
        
        # 1. Access Shared Image Data (Membrane)
        all_frames = np.memmap(config_dict['mmap_path'], 
                               dtype=config_dict['dtype'], 
                               mode='r', 
                               shape=config_dict['shape'])

        # 2. Re-create minimal Cropper for this worker
        crop_params = {
            **params.get('experiment_parameters', {}),
            **params.get('workflow_settings', {}),
            **params.get('image_processing', {})
        }
        cropper = CropImage(crop_params)
        cropper.rotation_angle = config_dict['rotation_angle']
        cropper.all_trap_rois = [None] * (trap_index + 1)
        cropper.all_trap_rois[trap_index] = config_dict['roi_coords']
        
        # 3. Extract Trap-Specific ROIs (Direct slice from shared memory)
        num_frames = config_dict['shape'][0]
        rois = []
        for j in range(num_frames):
            # Slicing from mmap; rotation is already handled in the shared memory stack
            rois.append(utils.crop_single_trap(all_frames[j], config_dict['roi_coords'], 0.0))

        # 4. Run Length Detection
        pip_x = config_dict['tuned_pipette_x']
        thr_prot = config_dict['tuned_threshold_prot']

        # --- Inject pulse frame index so the rupture detector can use it ---
        dye_params_pre = params.get('dye_uptake_parameters', {})
        if dye_params_pre.get('enable', False):
            pulse_frame_0based = dye_params_pre.get('pulse_index', 9)
            # setdefault creates the 'rupture_detection' dict if it doesn't already exist
            params.setdefault('rupture_detection', {})['pulse_frame_idx'] = pulse_frame_0based

        det_full = LineDetectionMFA(rois, [pip_x, 0], params)
        det_res = det_full.run_detection_with_parameters(pip_x, thr_prot, threshold_body=config_dict.get('tuned_threshold_body'))

        # --- Extract Key Timing Data ---
        rupture_time = None
        rupture_idx = None
        time_data = config_dict['time_data']
        
        if det_res.get('rupture_detected'):
            rupture_idx = det_res.get('rupture_frame_index')
            if rupture_idx is not None and rupture_idx < len(time_data):
                rupture_time = time_data[rupture_idx]
        
        pulse_time = None
        dye_params = params.get('dye_uptake_parameters', {})
        if dye_params.get('enable', False) and dye_params.get('has_pulse', True):
            p_idx = dye_params.get('pulse_index', 9)
            if 0 <= p_idx < len(time_data):
                pulse_time = time_data[p_idx]
        
        # 5. Data Safety Check
        if not (det_res and det_res['protrusion_lengths_um'] and any(p > 0 for p in det_res['protrusion_lengths_um'])):
            return {'trap_index': trap_index, 'status': 'no_detection'}

        # Compute pulse_idx (0-based array position of the electroporation pulse frame),
        # or None if no pulse was applied.  We store this in trap_data so aggregate
        # plots can split metrics into pre- and post-pulse windows.
        pulse_idx_for_data = None
        if pulse_time is not None:
            p_frame = dye_params.get('pulse_frame', 10)
            _candidate = p_frame - 1          # Convert 1-based frame number → 0-based index
            if 0 < _candidate < len(time_data):
                pulse_idx_for_data = _candidate

        trap_data = {
            'trap_index': trap_index,
            'time': time_data,
            'protrusions': det_res['protrusion_lengths_um'],
            'pip': pip_x,
            'thr': thr_prot,
            'rupture_idx': rupture_idx,
            'pulse_idx': pulse_idx_for_data,   # None when no pulse was applied
            'area': det_full.results.get('total_area_um2', []),    
            'area_prot': det_full.results.get('protrusion_area_um2', []),
            'area_body': det_full.results.get('body_area_um2', []),       
            'solidity': det_full.results.get('body_solidity', [])
        }

        # Export Raw CSV and Plot Trace.
        try:
            det_full.export_results_to_csv(dirs['root'], f"trap_{trap_index+1:02d}", dirs['full_csv'], dirs['filtered_csv'])
        except Exception:
            worker_logger.warning(f"Trap {trap_index+1}: CSV export failed.", exc_info=True)

        try:
            Plotting_MFA.plot_protrusion_trace(
                debug_images=det_full.results['debug_images'],
                time_points=np.array(time_data),
                protrusions=np.array(det_res['protrusion_lengths_um']),
                trap_index=trap_index + 1,
                save_path=dirs['tracking_visuals'] / f"trap_{trap_index+1:02d}_detection_trace.png",
                params=params,
                rupture_time=rupture_time,
                intensities=det_res.get('downstream_intensities'),
                pulse_time=pulse_time
            )
        except Exception:
            worker_logger.warning(f"Trap {trap_index+1}: Trace plot failed.", exc_info=True)

        # 6. Kymograph Generation
        if params.get('workflow_settings', {}).get('create_kymographs', True):
            try:
                # Pass dirs['tracking_visuals'] to the Kymograph generator
                create_kymograph_for_trap(rois, det_res, params, dirs['tracking_visuals'], trap_index + 1, time_data) # <-- UPDATED
            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Kymograph generation failed.", exc_info=True)

        # 7. Viscoelastic Fitting
        trap_fit_rows = []
        if params.get('model_parameters', {}).get('perform_fitting', True):
            try:
                dirs['fitting'].mkdir(parents=True, exist_ok=True)

                r_eff = compute_reff(params['model_parameters']['channel_width_um'],
                                     params['model_parameters']['channel_height_um'],
                                     params['model_parameters']['fstar'])
                t_pts = np.array(time_data[:rupture_idx+1]) if rupture_idx is not None else np.array(time_data)
                l_pts = np.array(det_res['protrusion_lengths_um'][:rupture_idx+1]) if rupture_idx is not None else np.array(det_res['protrusion_lengths_um'])
                delta_p = params['experiment_parameters']['constant_pressure']
                fit_run_params = {**params, **params.get('fitting_parameters', {})}

                # --- Determine pulse index (1-indexed frame number → 0-indexed array position) ---
                pulse_idx = None
                if pulse_time is not None:
                    p_frame = params.get('dye_uptake_parameters', {}).get('pulse_frame', 10)
                    # Clamp to the available range after any rupture truncation
                    pulse_idx = min(p_frame - 1, len(t_pts) - 1)
                    if pulse_idx <= 0:
                        pulse_idx = None  # Pulse at or before frame 1: no usable pre-pulse window

                def _run_fitter(t, l, phase, rupture_det, rupt_time):
                    """Instantiate, run, and return a FittingMFA for one time window."""
                    fitter = FittingMFA(t, l, r_eff, delta_p,
                                        rupture_detected=rupture_det,
                                        rupture_time=rupt_time,
                                        phase=phase)
                    fitter.run(params=fit_run_params)
                    return fitter

                def _collect_rows(fitter, phase_label):
                    """Convert a fitted FittingMFA into summary rows for the CSV."""
                    rows = []
                    if fitter.best_fit_model_name:
                        for model_name, res_fit in fitter.fit_results.items():
                            rows.append({
                                'Trap_Number': trap_index + 1,
                                'Phase': phase_label,
                                'Model_Name': model_name,
                                'R_Squared': res_fit['r_squared'],
                                'AIC': res_fit['aic'],
                                'N_Params': res_fit['n_params'],
                                'Is_Best_Fit': model_name == fitter.best_fit_model_name
                            })
                    return rows

                # --- Phase A: Full trace (always run, preserves backward compatibility) ---
                fitter_full = _run_fitter(t_pts, l_pts, 'full',
                                          rupture_det=(rupture_idx is not None),
                                          rupt_time=rupture_time)
                trap_fit_rows.extend(_collect_rows(fitter_full, 'full'))

                # Use the full-trace fitter for the analysis plot (most complete view)
                if fitter_full.best_fit_model_name:
                    try:
                        plotter = Plotting_MFA.MFAPlotter(fitter_full, params=params)
                        plotter.create_analysis_plot(trap_index + 1, dirs['fitting'] / f"trap_{trap_index+1:02d}_analysis_plot.png")
                    except Exception:
                        worker_logger.warning(f"Trap {trap_index+1}: Fit plot failed.", exc_info=True)

                # --- Phases B & C: Pre- and post-pulse (only when a pulse was applied) ---
                if pulse_idx is not None:
                    # Pre-pulse: frames 0 → pulse_idx (exclusive).
                    # Standard time and lengths — this is normal baseline aspiration.
                    try:
                        t_pre = t_pts[:pulse_idx]
                        l_pre = l_pts[:pulse_idx]
                        fitter_pre = _run_fitter(t_pre, l_pre, 'pre_pulse',
                                                  rupture_det=False, rupt_time=None)
                        trap_fit_rows.extend(_collect_rows(fitter_pre, 'pre_pulse'))
                    except Exception:
                        worker_logger.warning(f"Trap {trap_index+1}: Pre-pulse fitting failed.", exc_info=True)

                    # Post-pulse: frames pulse_idx → end.
                    # Time is re-zeroed to the pulse moment so the model sees t=0 at the pulse.
                    # Length is offset by L at the pulse so the model starts from ~0 deformation,
                    # satisfying the physical assumption of deforming from rest under constant pressure.
                    # The extracted E and η are therefore "post-pulse apparent parameters" and can be
                    # directly compared to the pre-pulse values.
                    try:
                        t_post = t_pts[pulse_idx:] - t_pts[pulse_idx]
                        l_post = l_pts[pulse_idx:] - l_pts[pulse_idx]
                        fitter_post = _run_fitter(t_post, l_post, 'post_pulse',
                                                   rupture_det=(rupture_idx is not None),
                                                   rupt_time=(rupture_time - t_pts[pulse_idx]) if rupture_time is not None else None)
                        trap_fit_rows.extend(_collect_rows(fitter_post, 'post_pulse'))
                    except Exception:
                        worker_logger.warning(f"Trap {trap_index+1}: Post-pulse fitting failed.", exc_info=True)

            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Fitting failed.", exc_info=True)

        # 8. Dye Uptake Analysis
        dye_results = None
        dye_config = config_dict.get('dye_mmap')
        if params.get('dye_uptake_parameters', {}).get('enable', False) and dye_config and dye_config['path']:
            try:
                dirs['dye'].mkdir(parents=True, exist_ok=True)
                all_dye_frames = np.memmap(dye_config['path'], dtype=dye_config['dtype'], mode='r', shape=dye_config['shape'])
                dye_rois = [cropper.process_frame(f, trap_index, skip_rotation=True) for f in all_dye_frames]

                uptake_analyzer = DyeUptakeAnalyzer(rois, dye_rois, det_res['pipette_start_x_used'], thr_prot, config_dict['tuned_threshold_body'], rupture_idx, params, frame_masks=det_res.get('frame_masks', []))
                dye_results = uptake_analyzer.run(time_data)
                uptake_analyzer.export_csv(trap_index + 1, dirs['dye'])
                
                # Pass pulse_time down to the updated plotter
                Plotting_MFA.plot_dye_uptake_dashboard(dye_results, trap_index + 1, dirs['dye'], params, det_res['pipette_start_x_used'], pulse_time=pulse_time)
                
                # Add the missing debug video call here
                uptake_analyzer.save_debug_video(trap_index + 1, dirs['dye'])
                
            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Dye uptake analysis failed.", exc_info=True)

        # 9. Actin Quantification
        actin_config = config_dict.get('actin_mmap')
        if params.get('actin_parameters', {}).get('enable', False) and actin_config and actin_config['path']:
            try:
                dirs['actin'].mkdir(parents=True, exist_ok=True)
                all_actin_frames = np.memmap(actin_config['path'], dtype=actin_config['dtype'], mode='r', shape=actin_config['shape'])
                actin_rois = [cropper.process_frame(f, trap_index, skip_rotation=True) for f in all_actin_frames]
                
                actin_analyzer = ActinAnalyzer(
                    rois, actin_rois,
                    det_res['pipette_start_x_used'],
                    thr_prot,
                    config_dict['tuned_threshold_body'],
                    params,
                    frame_masks=det_res.get('frame_masks', []),
                    start_idx=det_res.get('entry_frame_index', 0)
                )
                actin_results = actin_analyzer.run(time_data)
                actin_analyzer.export_csv(trap_index + 1, dirs['actin'])
                actin_analyzer.save_zones_debug_video(trap_index + 1, dirs['actin'])
                actin_analyzer.save_cortex_debug_video(trap_index + 1, dirs['actin'])
                Plotting_MFA.plot_actin_dashboard(actin_results, trap_index + 1, dirs['actin'], params, det_res['pipette_start_x_used'], pulse_time=pulse_time, rupture_time=rupture_time)
                Plotting_MFA.plot_actin_kymograph_and_profiles(actin_results, trap_index + 1, dirs['actin'], params, det_res['pipette_start_x_used'], pulse_time=pulse_time, rupture_time=rupture_time)
                Plotting_MFA.plot_actin_zones(actin_results, trap_index + 1, dirs['actin'], params, pulse_time=pulse_time, rupture_time=rupture_time)
                Plotting_MFA.plot_actin_cortex_structure(actin_results, trap_index + 1, dirs['actin'], params, pulse_time=pulse_time, rupture_time=rupture_time)
            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Actin analysis failed.", exc_info=True)
                
        return {
            'trap_index': trap_index,
            'status': 'success',
            'data': trap_data,
            'fit_rows': trap_fit_rows,
            'dye_data': dye_results
        }
            
    except Exception:
        worker_logger.error("A fatal error occurred in worker", exc_info=True)
        return {'trap_index': trap_index, 'status': 'error', 'error': traceback.format_exc()}

    finally:
        # --- Cleanup: Explicitly release all Shared Memory handles ---
        for mmap_obj in [all_frames, all_dye_frames, all_actin_frames]:
            if mmap_obj is not None:
                try:
                    mmap_obj.flush()
                    # Force OS-level lock release for Windows
                    if hasattr(mmap_obj, '_mmap') and mmap_obj._mmap is not None:
                        mmap_obj._mmap.close()
                    del mmap_obj
                except Exception: 
                    pass
        gc.collect()

class MFAAnalysis:
    """
    Manages the end-to-end analysis workflow for an MFA experiment.
    """

    def __init__(self, path: Path, experiment_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"The specified data path does not exist: {path}")
        self.path = path
        self.experiment_id = experiment_id
        self.params = params or {}
        
        # Cleanup orphaned temp folders from previous aborted or crashed runs to prevent disk bloat
        self.cleanup_orphaned_temp_dirs()
        
        # Helper classes for File I/O
        self.file_reader = FileRead(self.path, self.params)
        self.results_dir, self.subdirs = self._setup_output_directory()
        self.file_saver = FileSave(output_dir=self.results_dir)
        
        utils.setup_logging(self.params.get('workflow_settings', {}).get('debug_mode', False), self.results_dir / f"{self.experiment_id}_analysis.log")
        
        self.trap_results = []
        self.skipped = []
        self.loader = None
        self.cropper = None
        
        # Temp dir for the memory map file
        self.temp_dir = Path(tempfile.mkdtemp(prefix="MFA_temp_"))
        
        # Register cleanup to run at exit
        atexit.register(self.cleanup)

    @staticmethod
    def cleanup_orphaned_temp_dirs(max_age_hours: int = 24) -> None:
        """
        Scans the system temporary directory for any 'MFA_temp_*' folders older 
        than max_age_hours and deletes them. Resolves issues with disk space 
        accumulation resulting from hard-aborted pipeline runs.
        """
        temp_base = Path(tempfile.gettempdir())
        now = time.time()
        
        logger.info("Checking for orphaned temporary folders...")
        
        for p in temp_base.glob("MFA_temp_*"):
            if not p.is_dir(): continue
            try:
                if (now - p.stat().st_mtime) / 3600 > max_age_hours:
                    shutil.rmtree(p, ignore_errors=True)
            except Exception: pass

    def _setup_output_directory(self) -> Tuple[Path, Dict[str, Path]]:
        """Creates the structured folder hierarchy for results."""
        try:
            project_root = Path(__file__).resolve().parent
        except NameError:
            project_root = Path.cwd()
            
        output_base = project_root / "Output"
        results_dir = output_base / self.experiment_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define subfolders
        subdirs = {
            'root': results_dir,
            'filtered_csv': results_dir / "Filtered protrusion detection",
            'full_csv': results_dir / "Full protrusion detection",
            'tracking_visuals': results_dir / "Tracking visuals", 
            'morphology': results_dir / "Cell Morphology",
            'fitting': results_dir / "Fitting results",
            'dye': results_dir / "Dye Uptake",
            'actin': results_dir / "Actin Quantification" 
        }
        
        # Create core CSV folders immediately
        subdirs['filtered_csv'].mkdir(parents=True, exist_ok=True)
        subdirs['full_csv'].mkdir(parents=True, exist_ok=True)
        subdirs['tracking_visuals'].mkdir(parents=True, exist_ok=True)
        subdirs['morphology'].mkdir(parents=True, exist_ok=True)
            
        return results_dir, subdirs
    
    def cleanup(self):
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
                    

    def run_analysis(self) -> bool:
        """
        Main execution method.
        """
        try:
            logger.info("== MFA ANALYSIS START ==")
            
            # --- Phase 0: Load Metadata ---'
            # 1. Load file list
            self.file_reader.run()
            self.loader = utils.MemoryEfficientFrameLoader(self.file_reader, 
                self.params.get('workflow_settings', {}).get('max_cache_size', 50))
         
            trap_configs = []
            
            # --- Phase 1: Interactive Setup (Geometry) ---
            while True:
                crop_gui_params = {
                    **self.params.get('experiment_parameters', {}),
                    **self.params.get('workflow_settings', {}),
                    **self.params.get('image_processing', {}),
                    # Pass the full guv_settings dict so CropImage can apply the
                    # correct frame index for GUV experiments.
                    'guv_settings': self.params.get('guv_settings', {}),
                }
                self.cropper = CropImage(crop_gui_params)
                self.cropper.max_traps = self.params.get('experiment_parameters', {}).get('max_traps', 20)
                
                logger.info("\n== PHASE 1: INTERACTIVE TRAP SETUP ==")
                
                setup_signal = self.cropper.run(self.file_reader)
                if setup_signal in ('stop', 'restart'):
                     if setup_signal == 'stop': return False
                     continue
                
                # --- Phase 1b: Per-Trap Parameter Tuning ---
                setup_signal, trap_configs = self._collect_interactive_parameters()
                
                if setup_signal in ('stop', 'restart'):
                     if setup_signal == 'stop': return False
                     continue
                if setup_signal == 'confirm': break

            if not trap_configs:
                logger.info("No traps selected.")
                return True
                
            if self.cropper: self.cropper.save_trap_map(self.results_dir)

            # --- Phase 2: Parallel Processing ---
            logger.info("\n== PHASE 2: PARALLEL PROCESSING ==")
            
            # 1. Cache Membrane (Standard)
            mem_mmap_path, mem_shape, mem_dtype = self._prepare_shared_memory_stack(self.file_reader.tif_files, "Membrane")
            
            # 2. Cache Dye (Optional)
            dye_mmap_path, dye_shape, dye_dtype = None, None, None
            if self.params.get('dye_uptake_parameters', {}).get('enable', False) and self.file_reader.dye_files:
                dye_mmap_path, dye_shape, dye_dtype = self._prepare_shared_memory_stack(self.file_reader.dye_files, "Dye")

            # 3. Cache Actin (Optional)
            actin_mmap_path, actin_shape, actin_dtype = None, None, None
            if self.params.get('actin_parameters', {}).get('enable', False) and self.file_reader.actin_files:
                actin_mmap_path, actin_shape, actin_dtype = self._prepare_shared_memory_stack(self.file_reader.actin_files, "Actin")
            
            # Prepare arguments for each worker process
            worker_args = []
            for config in trap_configs:
                trap_index = config['trap_index']
                full_config = config.copy()
                full_config.update({
                    'params': self.params,
                    'time_data': self.file_reader.time_data,
                    'rotation_angle': self.cropper.rotation_angle,
                    'roi_coords': self.cropper.all_trap_rois[trap_index],
                    # Pass the specific tuned parameters
                    'tuned_pipette_x': config['pip'],
                    'tuned_threshold_prot': config['thr_prot'],
                    'tuned_threshold_body': config['thr_body'],
                    
                    'results_dir': self.results_dir,
                    'output_dirs': self.subdirs,
                    
                    # MEMBRANE CACHE
                    'mmap_path': mem_mmap_path,
                    'shape': mem_shape,
                    'dtype': mem_dtype,
                    
                    # EXTRA CHANNELS CACHE (Pass dictionaries)
                    'dye_mmap': {'path': dye_mmap_path, 'shape': dye_shape, 'dtype': dye_dtype},
                    'actin_mmap': {'path': actin_mmap_path, 'shape': actin_shape, 'dtype': actin_dtype},
                })
                worker_args.append(full_config)

            n_workers = self.params.get('workflow_settings', {}).get('n_workers', 4)
            all_results = []
            
            if n_workers == 1:
                # Sequential fallback
                for args in tqdm(worker_args, desc="Processing Traps"):
                    all_results.append(static_parallel_worker(args))
            else:
                # Parallel execution
                n_workers = min(n_workers, len(worker_args), (os.cpu_count() or 1))
                with multiprocessing.Pool(n_workers) as pool:
                    all_results = list(tqdm(pool.map(static_parallel_worker, worker_args), total=len(worker_args)))

            # --- Phase 3: Export ---
            logger.info("\n== PHASE 3: AGGREGATION & EXPORT ==")
            self._aggregate_and_export_results(all_results)
            
            return True
            
        finally:
            self.cleanup()
            
    def _prepare_shared_memory_stack(self, file_list: List[Path], tag: str) -> Tuple[Optional[str], Optional[Tuple], Optional[np.dtype]]:
        """
        Writes image frames into a binary file on disk (memmap).
        Generic version for any channel.
        """
        if not file_list:
            return None, None, None

        logger.info(f"Caching {tag} images to shared memory...")
        
        # Determine rotation and final dimensions
        rotation_angle = self.cropper.rotation_angle if self.cropper else 0.0
        first_img = self.file_reader.read_img(file_list[0])
        if rotation_angle != 0:
            first_img = utils.rotate_image(first_img, rotation_angle)
             
        h, w = first_img.shape[:2]
        is_color = len(first_img.shape) == 3
        shape = (len(file_list), h, w, 3) if is_color else (len(file_list), h, w)
        dtype = first_img.dtype
        mmap_path = self.temp_dir / f"image_stack_{tag}.dat"
        
        fp = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)
        
        for i, f in enumerate(tqdm(file_list, desc=f"Caching {tag} (Rotated)")):
            img = self.file_reader.read_img(f)
            if rotation_angle != 0:
                img = utils.rotate_image(img, rotation_angle)
            fp[i] = img if img is not None else np.zeros_like(first_img)
        
        fp.flush()
        
        # Explicitly release the OS file lock in the main process
        if hasattr(fp, '_mmap') and fp._mmap is not None:
            fp._mmap.close()
        del fp
        
        return str(mmap_path), shape, dtype
    
    def _collect_interactive_parameters(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Loops through EACH selected trap to let user tune Pipette & Thresholds.
        """
        configs = []
        if not self.loader or not self.cropper: return 'stop', []
        
        selected_indices = self.cropper.selected_traps
        if not selected_indices:
            logger.warning("No traps were selected in the setup phase.")
            return 'confirm', []
            
        logger.info(f"\nConfiguring {len(selected_indices)} selected traps (Pipette & Thresholds)...")
        num_frames = len(self.file_reader.tif_files)
        
        wf = self.params.get('workflow_settings', {})
        guv = self.params.get('guv_settings', {})

        if guv.get('enable', False):
            # GUV mode: use an early frame so the membrane is still intact.
            # selection_frame_index defaults to 2 if not set in guv_settings.
            test_idx = max(0, min(int(guv.get('selection_frame_index', 2)), num_frames - 1))
        else:
            # Cell mode: pick a frame proportionally through the video.
            frame_frac = wf.get('selection_frame_fraction', 0.5)
            test_idx = max(0, min(int(num_frames * frame_frac), num_frames - 1))
        
        img = self.loader.get_frame(test_idx)
        
        setup_progress = utils.ProgressTracker(len(selected_indices))
        
        for i, trap_index in enumerate(selected_indices):
            setup_progress.start_trap(i)
            try:
                roi_img = self.cropper.process_frame(img, trap_index)
                if roi_img is None:
                    self.skipped.append(trap_index)
                    setup_progress.finish_trap(i, True)
                    continue
        
                det_interactive = LineDetectionMFA([roi_img], self.cropper.pipette_coords, self.params)
                
                interactive_res = det_interactive.run_detection() 
                signal = interactive_res.get('signal')
                
                if signal in ('restart', 'stop'):
                    self.loader.clear_cache()
                    return signal, []
                
                if not interactive_res or 'pipette_start_x_used' not in interactive_res:
                    self.skipped.append(trap_index)
                    setup_progress.finish_trap(i, True)
                    continue
                
                configs.append({
                    'trap_index': trap_index, 
                    'pip': interactive_res['pipette_start_x_used'], 
                    'thr_prot': interactive_res['threshold_prot'],
                    'thr_body': interactive_res.get('threshold_body', interactive_res['threshold_prot'])
                })
                setup_progress.finish_trap(i, False)
                
            except Exception as e:
                logger.error(f"Error in setup trap #{trap_index+1}: {e}", exc_info=True)
                self.skipped.append(trap_index)
                setup_progress.finish_trap(i, True)

        self.loader.clear_cache()
        setup_progress.print_final_summary()
        return 'confirm', configs

    def _aggregate_and_export_results(self, all_results: List[Dict[str, Any]]) -> None:
        """Combines results from all workers into Summary CSVs."""
        self._save_consolidated_protrusions_long(all_results)
        
        all_fits_data = []
        for res in all_results:
            if 'fit_rows' in res and res['fit_rows']:
                 all_fits_data.extend(res['fit_rows'])
        
        if all_fits_data:
            self.subdirs['fitting'].mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(all_fits_data)
            df['Experiment_ID'] = self.experiment_id
            df.to_csv(self.subdirs['fitting'] / f"{self.experiment_id}_summary_fits.csv", index=False)

        self._perform_shear_analysis()
        
        # Call the morphology aggregate plotter (Using the corrected 'self.subdirs')
        Plotting_MFA.plot_aggregate_metrics(
            all_results, 
            self.file_reader.time_data, 
            self.subdirs['morphology'],
            self.experiment_id
        )
        
        if self.params.get('dye_uptake_parameters', {}).get('enable', False):
            Plotting_MFA.plot_aggregate_dye_metrics(
                all_results,
                self.subdirs['dye'],
                self.experiment_id,
                self.params
            )

    def _perform_shear_analysis(self):
        """Calculates Son (2007) shear metrics."""
        model_params = self.params.get('model_parameters', {})
        exp_params = self.params.get('experiment_parameters', {})
        
        W = model_params.get('channel_width_um', 10)
        H = model_params.get('channel_height_um', 10)
        L = model_params.get('channel_length_um', 100) 
        dP = exp_params.get('constant_pressure', 1000)
        visc = model_params.get('fluid_viscosity_pa_s', 0.001)    
        
        metrics = compute_shear_metrics(W, H, L, dP, visc)
        
        shear_df = pd.DataFrame([metrics])
        shear_df['Experiment_ID'] = self.experiment_id
        
        self.subdirs['fitting'].mkdir(parents=True, exist_ok=True)
        save_path = self.subdirs['fitting'] / f"{self.experiment_id}_shear_analysis.csv"
        shear_df.to_csv(save_path, index=False)
        Plotting_MFA.plot_shear_analysis_card(
            metrics, 
            self.subdirs['fitting'] / f"{self.experiment_id}_shear_analysis_card.png"
        )
        
    def _save_consolidated_protrusions_wide(self, all_results: List[Dict[str, Any]]) -> None:
        if not self.file_reader.time_data: return
        df = pd.DataFrame()
        df['Experiment_ID'] = [self.experiment_id] * len(self.file_reader.time_data)
        df['Time_s'] = self.file_reader.time_data
        
        sorted_results = sorted(all_results, key=lambda x: x['trap_index'])
        for res in sorted_results:
            trap_idx = res['trap_index']
            col_name = f'Trap_{trap_idx+1}_Protrusion_um'
            if res.get('status') in ('success', 'detection_only', 'fit_failed') and 'data' in res:
                 protrusions = res['data'].get('protrusions')
                 if protrusions and len(protrusions) == len(self.file_reader.time_data):
                     data_arr = np.array(protrusions, dtype=float)
                     data_arr[data_arr == 0] = np.nan
                     r_idx = res['data'].get('rupture_idx')
                     if r_idx is not None: data_arr[r_idx+1:] = np.nan
                     df[col_name] = data_arr
                 else: df[col_name] = np.nan
            else: df[col_name] = np.nan

        df.to_csv(self.results_dir / f"{self.experiment_id}_all_traps_protrusions.csv", index=False)

    def _save_consolidated_protrusions_long(self, all_results: List[Dict[str, Any]]) -> None:
        records = []
        pressure_pa = self.params.get('experiment_parameters', {}).get('constant_pressure', 0)
        for res in all_results:
            if res.get('status') not in ('success', 'detection_only', 'fit_failed'): continue
            data = res.get('data', {})
            t_pts = data.get('time', [])
            p_pts = data.get('protrusions', [])
            rupture_idx = data.get('rupture_idx')
            trap_id = data.get('trap_index') + 1
            if len(t_pts) != len(p_pts): continue
            for i in range(len(t_pts)):
                val = p_pts[i]
                if val == 0: val = np.nan
                post_rupture = (rupture_idx is not None) and (i > rupture_idx)
                records.append({
                    'Time_s': t_pts[i],
                    'Experiment_ID': self.experiment_id,
                    'Trap_Number': trap_id,
                    'Protrusion_Length_um': val if not post_rupture else np.nan,
                    'Pressure_Pa': pressure_pa,
                    'Rupture_Detected': (rupture_idx is not None),
                    'Is_Post_Rupture': post_rupture
                })
        if records:
            df = pd.DataFrame(records)
            df.to_csv(self.results_dir / f"{self.experiment_id}_all_traps_protrusions_long.csv", index=False)

def main() -> bool:
    utils.setup_logging(debug_mode=False) 
    parser = argparse.ArgumentParser(description="Run MFA analysis from a configuration file.")
    parser.add_argument("config_path", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    config_path = Path(args.config_path)

    try:
        with open(config_path, 'r') as f: raw_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error parsing config: {e}"); return False

    try: 
        params = validate_config(raw_config)
    except ValidationError as e:
        logger.error(f"Invalid configuration: {e}"); return False
    
    analysis = None
    try:
        analysis = MFAAnalysis(Path(params['paths']['data_folder']), params['paths']['experiment_id'], params)
        success = analysis.run_analysis()
        return success
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user.")
        return False
    except Exception:
        logger.error("Fatal error", exc_info=True)
        return False
    finally:
        if analysis:
            analysis.cleanup()

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()