# -*- coding: utf-8 -*-
"""
Orchestrates the Microfluidic Aspiration (MFA) analysis pipeline.

ROLE IN PIPELINE:
This is the main entry point. It coordinates the entire workflow:
1.  Loads configuration and experimental data.
2.  Runs the Interactive Setup (GUI) for users to select traps.
3.  Manages Parallel Processing (using multiprocessing) to analyze traps simultaneously.
4.  Aggregates results from all workers into final CSVs and plots.

KEY TECHNICAL FEATURES:
- Shared Memory: Uses `np.memmap` to share a single copy of the large image stack 
  across parallel processes, preventing RAM explosion.
- Batch Processing: Handles multiple traps in a single run.
- Error Handling: Ensures one failing trap doesn't crash the whole experiment.
"""
import atexit
import os
import gc
import argparse
import yaml
import multiprocessing
import traceback
import logging
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from pydantic import ValidationError

from FileHandling_MFA import FileRead, FileSave
from ImageProcessing_MFA import CropImage
from LineDetection_MFA import LineDetectionMFA
from Calculation_MFA import compute_reff, compute_shear_metrics
from Fitting_MFA import FittingMFA
from Kymograph_MFA import create_kymograph_for_trap
from config_schema import MFAConfig
import Plotting_MFA
import Utils_MFA as utils
from UptakeQuantification import DyeUptakeAnalyzer

logger = logging.getLogger(__name__)

# =============================================================================
# TOP-LEVEL STATIC WORKER FUNCTION (Optimized for Shared Memory)
# =============================================================================
# NOTE: This function must be at the top level (outside the class) to be 
# "picklable" by the multiprocessing module.

def static_parallel_worker(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single trap in an independent process.
    """
    trap_index = config_dict['trap_index']
    params = config_dict['params']
    
    # --- Shared Memory Access ---
    mmap_path = config_dict['mmap_path']
    dtype = config_dict['dtype']
    shape = config_dict['shape']
    
    # --- Unpack Dye Channel if present ---
    all_frames_c2 = None
    if config_dict.get('mmap_path_dye'):
        shape_dye = config_dict['shape_dye'] 
        all_frames_c2 = np.memmap(config_dict['mmap_path_dye'], dtype=dtype, mode='r', shape=shape_dye)

    dirs = config_dict['output_dirs']
    worker_logger = logging.getLogger(f"Worker-{trap_index+1:02d}")
    
    try:
        worker_logger.info(f"Starting...")
        
        # 1. Access Shared Image Data
        all_frames = np.memmap(mmap_path, dtype=dtype, mode='r', shape=shape)

        # 2. Get Coordinates
        rotation_angle = config_dict['rotation_angle']
        roi_coords = config_dict['roi_coords']

        # 3. Process ROIs (Membrane)
        num_frames = shape[0]
        rois = []
        for j in range(num_frames):
            cropped = utils.crop_single_trap(all_frames[j], roi_coords, rotation_angle)
            rois.append(cropped)

        # 4. Run Detection (Core Image Processing)
        pip = config_dict['pip']
        thr_prot = config_dict['thr_prot']
        
        det_full = LineDetectionMFA(rois, config_dict['pipette_coords_roi'], params)
        
        # Run detection using the protrusion threshold (for length)
        det_res = det_full.run_detection_with_parameters(pip, thr_prot)
        
        # --- EXTRACT RUPTURE TIME & INDEX ---
        rupture_time = None
        rupture_idx = None
        time_data = config_dict['time_data']
        
        if det_res.get('rupture_detected'):
            rupture_idx = det_res.get('rupture_frame_index')
            if rupture_idx is not None and rupture_idx < len(time_data):
                rupture_time = time_data[rupture_idx]
        
        # Safety Check
        if not (det_res and det_res['protrusion_lengths_um'] and any(p > 0 for p in det_res['protrusion_lengths_um'])):
            return {'trap_index': trap_index, 'status': 'no_detection'}
        
        # --- DYE UPTAKE LOGIC INSERTION ---
        enable_dye = params.get('dye_uptake_parameters', {}).get('enable', False)
        
        if enable_dye and all_frames_c2 is not None:
            worker_logger.info("Running Dye Uptake Quantification...")
            
            # Generate Dye ROIs
            rois_c2 = []
            for j in range(len(all_frames_c2)):
                crop_c2 = utils.crop_single_trap(all_frames_c2[j], roi_coords, rotation_angle)
                rois_c2.append(crop_c2)
            
            # Retrieve BOTH thresholds
            thr_body = config_dict.get('thr_body', thr_prot)
            
            # Get pipette coordinate
            pipette_x_loc = det_res.get('pipette_start_x_used')

            uptake_analyzer = DyeUptakeAnalyzer(
                membrane_rois=rois,
                dye_rois=rois_c2,
                pipette_x=pipette_x_loc,
                threshold_prot=thr_prot,  
                threshold_body=thr_body,  
                rupture_idx=rupture_idx,
                params=params
            )
            
            uptake_analyzer.run(time_data)
            
            dye_out_dir = dirs['dye_uptake']
            
            Plotting_MFA.plot_dye_uptake_dashboard(
                results=uptake_analyzer.results,
                trap_idx=trap_index + 1,
                output_dir=dye_out_dir,
                params=params,
                pipette_x=pipette_x_loc 
            )
            
            uptake_analyzer.export_csv(trap_index + 1, dye_out_dir)
            uptake_analyzer.save_debug_video(trap_index + 1, dye_out_dir)
        
        # 5. Generate Outputs & Exports
        trap_data = {
            'trap_index': trap_index,
            'time': time_data,
            'protrusions': det_res['protrusion_lengths_um'],
            'pip': det_res.get('pipette_start_x_used'),
            'thr': thr_prot,
            'rupture_idx': rupture_idx
        }

        # Export Raw CSV
        det_full.export_results_to_csv(
            output_dir=dirs['root'], 
            experiment_id=f"trap_{trap_index+1:02d}",
            dir_full=dirs['full_csv'],
            dir_filtered=dirs['filtered_csv']
        )

        # Plot Detection Trace
        Plotting_MFA.plot_protrusion_trace(
            debug_images=det_full.results['debug_images'],
            time_points=np.array(time_data),
            protrusions=np.array(det_res['protrusion_lengths_um']),
            trap_index=trap_index + 1,
            save_path=dirs['traces'] / f"trap_{trap_index+1:02d}_detection_trace.png",
            params=params,
            rupture_time=rupture_time,
            intensities=det_res.get('downstream_intensities')
        )

        growth_rate = None
        if params['create_kymographs']:
            # Create object and calc matrix (No plotting inside)
            kymo_analyzer = create_kymograph_for_trap(
                roi_images=rois,
                detection_results=det_res,
                params=params,
                output_dir=dirs['kymographs'],
                trap_index=trap_index,
                time_data=time_data  
            )
            growth_rate = kymo_analyzer.growth_rate
            trap_data['growth_rate_um_per_s'] = growth_rate
            
            Plotting_MFA.plot_kymograph(
                kymograph_matrix=kymo_analyzer.kymograph_matrix,
                trap_index=trap_index + 1,
                output_dir=dirs['kymographs'],
                pipette_x=det_res.get('pipette_start_x_used'),
                protrusions_px=det_res['protrusion_lengths_px'],
                time_data=time_data,
                params=params
            )
        
        # 6. Run Viscoelastic Fitting
        if not params.get('perform_fitting', True):
            return {'trap_index': trap_index, 'status': 'success', 'data': trap_data, 'fit_rows': []}
        
        channel_width = params['channel_width_um']
        channel_height = params['channel_height_um']
        f_star = params['fstar']
        delta_p = params['constant_pressure']
        r_eff = compute_reff(channel_width, channel_height, f_star)
        
        time_points = np.array(trap_data['time'], dtype=float)
        protrusion_lengths = np.array(trap_data['protrusions'], dtype=float)

        if rupture_idx is not None:
             time_points = time_points[:rupture_idx+1]
             protrusion_lengths = protrusion_lengths[:rupture_idx+1]
        
        fitter = FittingMFA(t_data=time_points, l_data=protrusion_lengths, r_eff=r_eff, delta_p=delta_p)
        fitter.run(params=params)
        
        trap_fit_rows = []
        if fitter.best_fit_model_name:
            plotter = Plotting_MFA.MFAPlotter(fitter, params=params)
            plotter.plot_all_models_comparison(trap_index + 1, dirs['fitting'] / f"trap_{trap_index+1:02d}_model_comparison.png")
            plotter.create_analysis_plot(trap_index + 1, dirs['fitting'] / f"trap_{trap_index+1:02d}_analysis_plot.png")
            
            for model_name, results in fitter.fit_results.items():
                model_row = {
                    'Trap_Number': trap_index + 1,
                    'Model_Name': model_name,
                    'Is_Best_Fit': model_name == fitter.best_fit_model_name,
                    'R_Squared': results.get('r_squared'),
                    'Rupture_Detected': fitter.rupture_detected,
                    'Effective_Radius_um': r_eff, 
                    'F_Star': f_star
                }
                if results.get('params') is not None:
                    params_dict = dict(zip(results['param_names'], results['params']))
                    model_row.update(params_dict)
                trap_fit_rows.append(model_row)
        else:
            worker_logger.warning("Fitting failed for all models.")
            return {'trap_index': trap_index, 'status': 'fit_failed', 'data': trap_data}
        
        del all_frames
        del rois
        gc.collect()
        
        return {'trap_index': trap_index, 'status': 'success', 'data': trap_data, 'fit_rows': trap_fit_rows}

    except Exception:
        worker_logger.error("A fatal error occurred in worker", exc_info=True)
        return {'trap_index': trap_index, 'status': 'error', 'error': traceback.format_exc()}


class MFAAnalysis:
    """
    Manages the end-to-end analysis workflow for an MFA experiment.
    Instantiated once per experiment folder.
    """

    def __init__(self, path: Path, experiment_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"The specified data path does not exist: {path}")
        self.path = path
        self.experiment_id = experiment_id
        self.params = params or {}
        
        # Helper classes for File I/O
        self.file_reader = FileRead(self.path, self.params)
        self.results_dir, self.subdirs = self._setup_output_directory()
        self.file_saver = FileSave(output_dir=self.results_dir)
        
        utils.setup_logging(self.params.get('debug_mode', False), self.results_dir / f"{self.experiment_id}_analysis.log")
        
        self.trap_results = []
        self.skipped = []
        self.progress = None
        self.loader = None
        self.cropper = None
        
        # Temp dir for the memory map file
        self.temp_dir = Path(tempfile.mkdtemp(prefix="MFA_temp_"))
        
        # Register cleanup to run at exit
        atexit.register(self.cleanup)

    def _setup_output_directory(self) -> Tuple[Path, Dict[str, Path]]:
        """Creates the structured folder hierarchy for results."""
        try:
            project_root = Path(__file__).resolve().parent
        except NameError:
            project_root = Path.cwd()
            
        output_base = project_root / "Output"
        results_dir = output_base / self.experiment_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define and create subfolders
        subdirs = {
            'root': results_dir,
            'filtered_csv': results_dir / "Filtered protrusion detection",
            'full_csv': results_dir / "Full protrusion detection",
            'traces': results_dir / "Protrusion traces",
            'fitting': results_dir / "Fitting results",
            'kymographs': results_dir / "Kymographs",
            'dye_uptake': results_dir / "Dye_Uptake",
        }
        
        for p in subdirs.values(): p.mkdir(parents=True, exist_ok=True)
        return results_dir, subdirs
    
    def cleanup(self):
        """Removes temporary files (memory maps) to free disk space."""
        # Check if temp_dir exists before trying to delete to be safe/idempotent
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not cleanup temp dir: {e}")

    def run_analysis(self) -> bool:
        """
        Main execution method.
        Returns: True if analysis completed, False if stopped by user.
        """
        try:
            logger.info("== MFA ANALYSIS START ==")
            
            # --- Phase 0: Load Metadata ---
            self.file_reader.run()
            # The loader handles caching for the interactive phase
            self.loader = utils.MemoryEfficientFrameLoader(self.file_reader, self.params['max_cache_size'])
            trap_configs = []
            
            # --- Phase 1: Interactive Setup ---
            # User uses GUI to define rotation, ROI, and settings per trap.
            while True:
                self.cropper = CropImage(self.params)
                self.cropper.max_traps = self.params['max_traps']
                logger.info("\n== PHASE 1: INTERACTIVE TRAP SETUP ==")
                
                setup_signal = self.cropper.run(self.file_reader)
                if setup_signal in ('stop', 'restart'):
                     if setup_signal == 'stop': return False
                     continue
                
                # Collect specific parameters for each selected trap
                self.params['scale_factor'] = self.cropper.scale_factor
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
            
            # 1. Cache Membrane Images (Original Logic)
            mmap_path, shape, dtype = self._prepare_shared_memory_stack()
            
            # 2. Cache Dye Images (Only if enabled)
            mmap_path_dye = None
            shape_dye = None
            if self.params.get('dye_uptake_parameters', {}).get('enable', False):
                if self.file_reader.dye_files:
                    logger.info("Caching Dye channel (C2) images...")
                    mmap_path_dye, shape_dye, _ = self._prepare_dye_stack(self.file_reader.dye_files)
                else:
                    logger.warning("Dye uptake enabled, but no Dye files found.")
            
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
                    'pipette_coords_roi': self.cropper.pipette_coords,
                    'results_dir': self.results_dir,
                    'mmap_path': mmap_path,
                    'shape': shape,
                    'dtype': dtype,
                    'output_dirs': self.subdirs,
                    'mmap_path_dye': mmap_path_dye,
                    'shape_dye': shape_dye
                })
                worker_args.append(full_config)

            n_workers = self.params.get('n_workers', 1)
            all_results = []
            
            if n_workers == 1:
                # Sequential fallback for debugging
                for args in tqdm(worker_args, desc="Processing Traps"):
                    all_results.append(static_parallel_worker(args))
            else:
                # Parallel execution
                # Cap workers at CPU count to avoid thrashing
                n_workers = min(n_workers, len(worker_args), (os.cpu_count() or 1))
                with multiprocessing.Pool(n_workers) as pool:
                    all_results = list(tqdm(pool.map(static_parallel_worker, worker_args), total=len(worker_args)))

            # --- Phase 3: Export ---
            logger.info("\n== PHASE 3: AGGREGATION & EXPORT ==")
            self._aggregate_and_export_results(all_results)
            return True
        
        finally:
            self.cleanup()
            
    def _prepare_shared_memory_stack(self) -> Tuple[str, Tuple, np.dtype]:
        """
        Writes all image frames into a single binary file on disk (memmap).
        This file is then mapped into memory by all workers, avoiding data duplication.
        """
        logger.info("Caching all images to shared memory...")
        first_img = self.file_reader.read_img(self.file_reader.tif_files[0])
        h, w = first_img.shape[:2]
        is_color = len(first_img.shape) == 3
        n_frames = len(self.file_reader.tif_files)
        shape = (n_frames, h, w, 3) if is_color else (n_frames, h, w)
        dtype = first_img.dtype
        mmap_path = self.temp_dir / "image_stack.dat"
        
        # 'w+' creates or overwrites the file
        fp = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)
        for i, f in enumerate(tqdm(self.file_reader.tif_files, desc="Loading Cache")):
            img = self.file_reader.read_img(f)
            if img is not None: fp[i] = img
            else: fp[i] = np.zeros_like(first_img)
        # Flush to disk to ensure data is written
        fp.flush()
        return str(mmap_path), shape, dtype

    def _prepare_dye_stack(self, file_list: List[Path]) -> Tuple[str, Tuple, np.dtype]:
        """
        New function: Writes dye frames to disk.
        Essentially a copy of the above, but kept separate to minimize diffs.
        """
        first_img = self.file_reader.read_img(file_list[0])
        h, w = first_img.shape[:2]
        is_color = len(first_img.shape) == 3
        n_frames = len(file_list)
        shape = (n_frames, h, w, 3) if is_color else (n_frames, h, w)
        dtype = first_img.dtype
        mmap_path = self.temp_dir / "dye_stack.dat"
        
        # 'w+' creates or overwrites the file
        fp = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)
        for i, f in enumerate(tqdm(file_list, desc="Loading Dye Cache")):
            img = self.file_reader.read_img(f)
            if img is not None: fp[i] = img
            else: fp[i] = np.zeros_like(first_img)
        
        # Flush to disk to ensure data is written
        fp.flush()
        return str(mmap_path), shape, dtype
    
    def _collect_interactive_parameters(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Runs a mini-detection on each selected trap to let the user
        confirm the pipette entrance and threshold *before* the batch run.
        """
        configs = []
        if not self.loader or not self.cropper: return 'stop', []
        logger.info("Loading trap selection window...")
        num_frames = len(self.file_reader.tif_files)
        
        # Pick a frame halfway through the experiment to assist setup
        frame_frac = self.params.get('selection_frame_fraction', 0.5)
        test_idx = int(num_frames * frame_frac)
        test_idx = max(0, min(test_idx, num_frames - 1))
        
        img = self.loader.get_frame(test_idx)
        
        # 1. User selects which traps to analyze
        signal, selected_indices = self.cropper.select_traps_interactively(img)
        
        if signal in ('restart', 'stop'): return signal, []
        if not selected_indices: return 'confirm', []
            
        setup_progress = utils.ProgressTracker(len(selected_indices))
        logger.info("\nConfiguring selected traps (pipette & threshold)...")
        
        # 2. Loop through selected traps for specific tuning
        for i, trap_index in enumerate(selected_indices):
            setup_progress.start_trap(i)
            try:
                roi_img = self.cropper.process_frame(img, trap_index)
                if roi_img is None:
                    self.skipped.append(trap_index)
                    setup_progress.finish_trap(i, True)
                    continue
                
                # Run the interactive detection UI
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
                
                # Store the confirmed parameters (pipette X, threshold)
                configs.append({
                    'trap_index': trap_index, 
                    'pip': interactive_res['pipette_start_x_used'], 
                    'thr_prot': interactive_res['threshold_prot'], # <--- SAVE PROT
                    'thr_body': interactive_res['threshold_body']  # <--- SAVE BODY
                })
                setup_progress.finish_trap(i, False)
            except Exception as e:
                logger.error(f"Error in setup trap #{trap_index+1}: {e}")
                self.skipped.append(trap_index)
                setup_progress.finish_trap(i, True)

        self.loader.clear_cache()
        setup_progress.print_final_summary()
        return 'confirm', configs

    def _aggregate_and_export_results(self, all_results: List[Dict[str, Any]]) -> None:
        """Combines results from all workers into Summary CSVs."""
        self._save_consolidated_protrusions_wide(all_results)
        self._save_consolidated_protrusions_long(all_results)
        
        all_fits_data = []
        for res in all_results:
            if 'fit_rows' in res and res['fit_rows']:
                 all_fits_data.extend(res['fit_rows'])
        
        if all_fits_data:
            df = pd.DataFrame(all_fits_data)
            df['Experiment_ID'] = self.experiment_id
            df.to_csv(self.subdirs['fitting'] / f"{self.experiment_id}_summary_fits.csv", index=False)

        # --- Shear analysis ---
        self._perform_shear_analysis()

    def _perform_shear_analysis(self):
        """Calculates Son (2007) shear metrics and saves them."""
        # 1. Gather Constants from Config
        W = self.params['channel_width_um']
        H = self.params['channel_height_um']
        L = self.params['channel_length_um'] 
        dP = self.params['constant_pressure']
        visc = self.params['fluid_viscosity_pa_s']   
        
        # 2. Calculate
        metrics = compute_shear_metrics(W, H, L, dP, visc)
        
        # 3. Create DataFrame 
        shear_df = pd.DataFrame([metrics])
        shear_df['Experiment_ID'] = self.experiment_id
        
        # 4. Save CSV (Force Directory Creation)
        save_path = self.subdirs['fitting'] / f"{self.experiment_id}_shear_analysis.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shear_df.to_csv(save_path, index=False)
        logger.info(f"Shear analysis CSV saved: {save_path.name}")
        
        # 5. Generate Figure
        plot_path = self.subdirs['fitting'] / f"{self.experiment_id}_shear_analysis_card.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        Plotting_MFA.plot_shear_analysis_card(metrics, plot_path)
        
    def _save_consolidated_protrusions_wide(self, all_results: List[Dict[str, Any]]) -> None:
        """Saves a 'Wide' format CSV: Time | Trap 1 Length | Trap 2 Length ..."""
        if not self.file_reader.time_data: return
        df_consolidated = pd.DataFrame()
        df_consolidated['Experiment_ID'] = [self.experiment_id] * len(self.file_reader.time_data)
        df_consolidated['Time_s'] = self.file_reader.time_data
        
        sorted_results = sorted(all_results, key=lambda x: x['trap_index'])
        for res in sorted_results:
            trap_idx = res['trap_index']
            col_name = f'Trap_{trap_idx+1}_Protrusion_um'
            if res.get('status') in ('success', 'detection_only', 'fit_failed') and 'data' in res:
                 protrusions = res['data'].get('protrusions')
                 if protrusions and len(protrusions) == len(self.file_reader.time_data):
                     data_arr = np.array(protrusions, dtype=float)
                     data_arr[data_arr == 0] = np.nan
                     # If rupture occurred, set subsequent data to NaN
                     r_idx = res['data'].get('rupture_idx')
                     if r_idx is not None: data_arr[r_idx+1:] = np.nan
                     df_consolidated[col_name] = data_arr
                 else: df_consolidated[col_name] = np.nan
            else: df_consolidated[col_name] = np.nan

        save_path = self.results_dir / f"{self.experiment_id}_all_traps_protrusions.csv"
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True) # Force creation
            df_consolidated.to_csv(save_path, index=False)
            logger.info(f"Consolidated Wide CSV saved: {save_path.name}")
        except Exception as e: logger.error(f"Failed to save consolidated CSV: {e}")

    def _save_consolidated_protrusions_long(self, all_results: List[Dict[str, Any]]) -> None:
        """Saves a 'Long' (Tidy) format CSV suitable for statistical plotting packages."""
        logger.info("Exporting Long/Tidy format CSV...")
        records = []
        pressure_pa = self.params.get('constant_pressure', 0)
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
            save_path = self.results_dir / f"{self.experiment_id}_all_traps_protrusions_long.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True) # Force creation
            df.to_csv(save_path, index=False)
            logger.info(f"Long format CSV saved: {save_path.name}")

def main() -> bool:
    utils.setup_logging(debug_mode=False) 
    parser = argparse.ArgumentParser(description="Run MFA analysis from a configuration file.")
    parser.add_argument("config_path", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    config_path = Path(args.config_path)

    # Load and Validate Configuration
    try:
        with open(config_path, 'r') as f: raw_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error parsing config: {e}"); return False

    try: mfa_config = MFAConfig(**raw_config)
    except ValidationError as e:
        logger.error(f"Invalid configuration: {e}"); return False
    
    # Flatten config for easier access in the pipeline
    params = {
        **mfa_config.experiment_parameters.model_dump(),
        **mfa_config.model_parameters.model_dump(),
        **mfa_config.rupture_detection.model_dump(),
        **mfa_config.workflow_settings.model_dump(),
        **mfa_config.image_processing.model_dump(),
        **mfa_config.kymograph_parameters.model_dump(),
        **mfa_config.plotting_parameters.model_dump(),
        **mfa_config.fitting_parameters.model_dump(),
        **mfa_config.validation_parameters.model_dump(),
        'dye_uptake_parameters': mfa_config.dye_uptake_parameters.model_dump(),
    }
    params['enable_rupture_detection'] = mfa_config.rupture_detection.enable
    params['debug_mode'] = mfa_config.workflow_settings.debug_mode
    params['verify_traps_interactively'] = mfa_config.workflow_settings.verify_traps_interactively
    
    # Launch Analysis
    analysis = None
    try:
        analysis = MFAAnalysis(mfa_config.paths.data_folder, mfa_config.paths.experiment_id, params)
        # analysis.mfa_config = mfa_config
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