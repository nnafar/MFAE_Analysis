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
#import sys
import cv2
import tempfile
import shutil
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
from config_schema import MFAConfig, validate_config
import Plotting_MFA
import Utils_MFA as utils

logger = logging.getLogger(__name__)

# =============================================================================
# TOP-LEVEL STATIC WORKER FUNCTION (Optimized for Shared Memory)
# =============================================================================

def static_parallel_worker(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single trap in an independent process.
    """
    trap_index = config_dict['trap_index']
    params = config_dict['params']
    
    # --- Shared Memory Access ---
    # Reconstruct the numpy array from the memory map file on disk.
    mmap_path = config_dict['mmap_path']
    dtype = config_dict['dtype']
    shape = config_dict['shape']

    # Unpack Output Directories
    dirs = config_dict['output_dirs']
    
    worker_logger = logging.getLogger(f"Worker-{trap_index+1:02d}")
    
    try:
        worker_logger.info(f"Starting analysis for Trap {trap_index+1}...")
        
        # 1. Access Shared Image Data (Memmap)
        # 'r' mode ensures we don't accidentally modify the source images
        all_frames = np.memmap(mmap_path, dtype=dtype, mode='r', shape=shape)

        # 2. Re-create minimal Cropper
        # Worker receives full params, so we must flatten for CropImage here too if needed,
        # but workers usually don't use the GUI parts of CropImage.
        # However, to be safe and consistent:
        crop_params = {
            **params.get('experiment_parameters', {}),
            **params.get('workflow_settings', {}),
            **params.get('image_processing', {})
        }
        cropper = CropImage(crop_params)
        
        cropper.rotation_angle = config_dict['rotation_angle']
        # Populate only the specific trap ROI this worker needs
        cropper.all_trap_rois = [None] * (trap_index + 1) 
        cropper.all_trap_rois[trap_index] = config_dict['roi_coords']
        
        # 3. Process ROIs (Direct slice from mmap array)
        num_frames = shape[0]
        rois = []
        for j in range(num_frames):
            rois.append(cropper.process_frame(all_frames[j], trap_index))

        # 4. Run Detection (Using Per-Trap Tuned Parameters)
        pip_x = config_dict['tuned_pipette_x']
        thr_prot = config_dict['tuned_threshold_prot']
        
        det_full = LineDetectionMFA(rois, [pip_x, 0], params)
        det_res = det_full.run_detection_with_parameters(pip_x, thr_prot)

        # --- Extract Rupture Time & Index ---
        rupture_time = None
        rupture_idx = None
        time_data = config_dict['time_data']
        
        if det_res.get('rupture_detected'):
            rupture_idx = det_res.get('rupture_frame_index')
            if rupture_idx is not None and rupture_idx < len(time_data):
                rupture_time = time_data[rupture_idx]
        
        # Safety Check: If no protrusion was detected at all
        if not (det_res and det_res['protrusion_lengths_um'] and any(p > 0 for p in det_res['protrusion_lengths_um'])):
            return {'trap_index': trap_index, 'status': 'no_detection'}

        # 5. Generate Basic Outputs
        trap_data = {
            'trap_index': trap_index,
            'time': time_data,
            'protrusions': det_res['protrusion_lengths_um'],
            'pip': pip_x,
            'thr': thr_prot,
            'rupture_idx': rupture_idx
        }

        # Export Raw CSV (per trap)
        det_full.export_results_to_csv(
            output_dir=dirs['root'], 
            experiment_id=f"trap_{trap_index+1:02d}",
            dir_full=dirs['full_csv'],
            dir_filtered=dirs['filtered_csv']
        )

        # Plot the Detection Trace (Always created)
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

        # 6. Optional: Kymograph
        if params.get('workflow_settings', {}).get('create_kymographs', True):
            dirs['kymographs'].mkdir(parents=True, exist_ok=True)
        
            kymo_params = {**params, **params.get('kymograph_parameters', {})}
            
            det_res['pipette_start_x_used'] = config_dict['tuned_pipette_x']

            create_kymograph_for_trap(
                roi_images=rois,
                detection_results=det_res,
                params=kymo_params,
                output_dir=str(dirs['kymographs']), # <--- FIX: Convert to string
                trap_index=trap_index + 1,
                time_data=time_data  
            )
        
        # 7. Optional: Viscoelastic Fitting
        trap_fit_rows = []
        if params.get('model_parameters', {}).get('perform_fitting', True):
            dirs['fitting'].mkdir(parents=True, exist_ok=True)
            
            # Geometry & Pressure
            model_params = params.get('model_parameters', {})
            exp_params = params.get('experiment_parameters', {})
            
            channel_width = model_params['channel_width_um']
            channel_height = model_params['channel_height_um']
            f_star = model_params['fstar']
            delta_p = exp_params['constant_pressure']
            r_eff = compute_reff(channel_width, channel_height, f_star)
            
            # Prepare data
            t_pts = np.array(trap_data['time'], dtype=float)
            l_pts = np.array(trap_data['protrusions'], dtype=float)
            if rupture_idx is not None:
                 t_pts = t_pts[:rupture_idx+1]
                 l_pts = l_pts[:rupture_idx+1]
            
            # Run Fitter
            fit_config = {**params, **params.get('fitting_parameters', {}), **params.get('rupture_detection', {})}
            
            fitter = FittingMFA(t_data=t_pts, l_data=l_pts, r_eff=r_eff, delta_p=delta_p)
            fitter.run(params=fit_config)
            
            # Save Results
            if fitter.best_fit_model_name:
                plotter = Plotting_MFA.MFAPlotter(fitter, params=params)
                
                # A. Create Summary Card
                plotter.create_analysis_plot(trap_index + 1, dirs['fitting'] / f"trap_{trap_index+1:02d}_analysis_plot.png")
                
                # B. Create Model Comparison
                plotter.plot_all_models_comparison(trap_index + 1, dirs['fitting'] / f"trap_{trap_index+1:02d}_model_comparison.png")
                
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

        # 8. Optional: Dye Uptake Analysis
        dye_files = config_dict.get('dye_files_path')
        if params.get('dye_uptake_parameters', {}).get('enable', False) and dye_files:
            dirs['dye'].mkdir(parents=True, exist_ok=True)
            worker_logger.info("Running Dye Uptake Analysis...")
            
            # Read Dye Frames specific to this trap
            dye_rois = []
            for f_path in dye_files:
                img = cv2.imread(str(f_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Crop using the same geometry
                    dye_rois.append(cropper.process_frame(img, trap_index))
                else:
                    dye_rois.append(None)
            
            # Run Analyzer
            thr_prot = config_dict.get('tuned_threshold_prot', 30)
            thr_body = config_dict.get('tuned_threshold_body', thr_prot)
            
            uptake_analyzer = DyeUptakeAnalyzer(
                membrane_rois=rois,
                dye_rois=dye_rois,
                pipette_x=det_res.get('pipette_start_x_used'),
                threshold_prot=thr_prot,
                threshold_body=thr_body,
                rupture_idx=rupture_idx,
                params=params,
                start_idx=det_res.get('entry_frame_index', 0)
            )
            
            dye_results = uptake_analyzer.run(time_data)
            uptake_analyzer.export_csv(trap_index + 1, dirs['dye'])
            
            Plotting_MFA.plot_dye_uptake_dashboard(
                dye_results, trap_index + 1, dirs['dye'], params, pipette_x=det_res.get('pipette_start_x_used')
            )

        # Cleanup memory
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
        
        utils.setup_logging(self.params.get('workflow_settings', {}).get('debug_mode', False), self.results_dir / f"{self.experiment_id}_analysis.log")
        
        self.trap_results = []
        self.skipped = []
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
        
        # Define subfolders (creation is deferred to worker/usage)
        subdirs = {
            'root': results_dir,
            'filtered_csv': results_dir / "Filtered protrusion detection",
            'full_csv': results_dir / "Full protrusion detection",
            'traces': results_dir / "Protrusion traces",
            'fitting': results_dir / "Fitting results",
            'kymographs': results_dir / "Kymographs",
            'dye': results_dir / "Dye Uptake"
        }
        
        # Create core CSV folders immediately
        subdirs['filtered_csv'].mkdir(parents=True, exist_ok=True)
        subdirs['full_csv'].mkdir(parents=True, exist_ok=True)
        subdirs['traces'].mkdir(parents=True, exist_ok=True)
            
        return results_dir, subdirs
    
    def cleanup(self):
        """Removes temporary files (memory maps) to free disk space."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not cleanup temp dir: {e}")

    def run_analysis(self) -> bool:
        """
        Main execution method.
        """
        try:
            logger.info("== MFA ANALYSIS START ==")
            
            # --- Phase 0: Load Metadata ---
            self.file_reader.run()
            # The loader handles caching for the interactive phase
            max_cache = self.params.get('workflow_settings', {}).get('max_cache_size', 50)
            self.loader = utils.MemoryEfficientFrameLoader(self.file_reader, max_cache)
            
            trap_configs = []
            
            # --- Phase 1: Interactive Setup (Geometry) ---
            while True:
                crop_gui_params = {
                    **self.params.get('experiment_parameters', {}),
                    **self.params.get('workflow_settings', {}),
                    **self.params.get('image_processing', {})
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
            
            # Create the memory map file (Shared Memory)
            mmap_path, shape, dtype = self._prepare_shared_memory_stack()
            
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
                    'mmap_path': mmap_path,
                    'shape': shape,
                    'dtype': dtype,
                    'output_dirs': self.subdirs,
                    # Pass path to dye files for direct loading
                    'dye_files_path': self.file_reader.dye_files 
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
            
    def _prepare_shared_memory_stack(self) -> Tuple[str, Tuple, np.dtype]:
        """
        Writes all image frames into a single binary file on disk (memmap).
        """
        logger.info("Caching all images to shared memory...")
        # Read first file to get dimensions/dtype
        first_img = self.file_reader.read_img(self.file_reader.tif_files[0])
        h, w = first_img.shape[:2]
        is_color = len(first_img.shape) == 3
        n_frames = len(self.file_reader.tif_files)
        
        shape = (n_frames, h, w, 3) if is_color else (n_frames, h, w)
        dtype = first_img.dtype
        mmap_path = self.temp_dir / "image_stack.dat"
        
        # 'w+' creates or overwrites the file
        fp = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)
        
        # Only map the BF files. Dye files are handled separately.
        for i, f in enumerate(tqdm(self.file_reader.tif_files, desc="Loading Cache")):
            img = self.file_reader.read_img(f)
            if img is not None: fp[i] = img
            else: fp[i] = np.zeros_like(first_img)
        
        fp.flush()
        return str(mmap_path), shape, dtype

    def _collect_interactive_parameters(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Loops through EACH selected trap to let user tune Pipette & Thresholds.
        """
        configs = []
        if not self.loader or not self.cropper: return 'stop', []
        
        # Use traps already selected in Phase 1
        selected_indices = self.cropper.selected_traps
        if not selected_indices:
            logger.warning("No traps were selected in the setup phase.")
            return 'confirm', []
            
        logger.info(f"\nConfiguring {len(selected_indices)} selected traps (Pipette & Thresholds)...")
        num_frames = len(self.file_reader.tif_files)
        
        # Pick a representative frame
        frame_frac = self.params.get('workflow_settings', {}).get('selection_frame_fraction', 0.5)
        test_idx = int(num_frames * frame_frac)
        test_idx = max(0, min(test_idx, num_frames - 1))
        
        img = self.loader.get_frame(test_idx)
        
        setup_progress = utils.ProgressTracker(len(selected_indices))
        
        # 2. Loop through traps
        for i, trap_index in enumerate(selected_indices):
            setup_progress.start_trap(i)
            try:
                # Crop representative frame
                roi_img = self.cropper.process_frame(img, trap_index)
                if roi_img is None:
                    self.skipped.append(trap_index)
                    setup_progress.finish_trap(i, True)
                    continue
        
                # Check config to see if we need Dye thresholding too
                dye_enabled = self.params.get('dye_uptake_parameters', {}).get('enable', False)
                
                # Run the interactive UI
                # Note: We pass [roi_img] as a list because LineDetectionMFA expects a list
                det_interactive = LineDetectionMFA([roi_img], self.cropper.pipette_coords, self.params)
                
                # This call triggers the sequence: Pipette -> Thresh Prot -> Thresh Body (if dye enabled)
                interactive_res = det_interactive.run_detection() 
                signal = interactive_res.get('signal')
                
                if signal in ('restart', 'stop'):
                    self.loader.clear_cache()
                    return signal, []
                
                if not interactive_res or 'pipette_start_x_used' not in interactive_res:
                    self.skipped.append(trap_index)
                    setup_progress.finish_trap(i, True)
                    continue
                
                # Store the CONFIRMED parameters for this specific trap
                configs.append({
                    'trap_index': trap_index, 
                    'pip': interactive_res['pipette_start_x_used'], 
                    'thr_prot': interactive_res['threshold_prot'],
                    'thr_body': interactive_res.get('threshold_body', interactive_res['threshold_prot'])
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
            # Ensure directory exists before saving summary
            self.subdirs['fitting'].mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(all_fits_data)
            df['Experiment_ID'] = self.experiment_id
            df.to_csv(self.subdirs['fitting'] / f"{self.experiment_id}_summary_fits.csv", index=False)

        # --- Shear analysis ---
        self._perform_shear_analysis()

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
        
        # Ensure directory exists before saving
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

    # Load and Validate Configuration
    try:
        with open(config_path, 'r') as f: raw_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error parsing config: {e}"); return False

    try: 
        # Validate using Pydantic (returns dict)
        params = validate_config(raw_config)
    except ValidationError as e:
        logger.error(f"Invalid configuration: {e}"); return False
    
    # Launch Analysis
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