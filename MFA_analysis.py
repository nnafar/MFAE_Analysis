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
import sys

# ---------------------------------------------------------------------------
# Silence noisy libtiff/OpenCV C++ warnings BEFORE cv2 is imported.
# These lines (e.g. "cv::TIFF_Warning TIFFReadDirectory: Unknown field with
# tag 18246") come from OpenCV's C++ layer straight to stderr and cannot be
# intercepted by Python's logging module. The env var below is read by
# OpenCV at import time; setting it after `import cv2` has no effect.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import gc
import argparse
import yaml
import multiprocessing
import traceback
import logging
import cv2

# Belt-and-braces: also mute OpenCV's Python-facing logger. The env var
# above is the load-time one; this handles any post-import verbosity.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass  # very old cv2 build without utils.logging

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
from config_schema import MFAConfig, validate_config, format_validation_error
import config_loader
import experiment_naming
import Plotting_MFA
import Plotting_Multipanel
import Utils_MFA as utils

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent



# The lab's standard experiment config; also the template for --init-config.
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def default_output_root() -> Path:
    """Folder that per-experiment results are written to by default."""
    return PROJECT_ROOT / "Output"

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
        # Allocate enough slots for the full trap count, not just up to this trap's index.
        n_slots = max(
            params.get('experiment_parameters', {}).get('max_traps', trap_index + 1),
            trap_index + 1
        )
        cropper.all_trap_rois = [None] * n_slots
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

        # Filename numbering scheme detected once at pipeline setup and
        # passed through the worker config.  Falls back to 1 (t0001-start,
        # the historical assumption) if the field is missing.
        first_frame_number = int(config_dict.get('first_frame_number', 1))

        # --- Resolve the pulse frame once for this worker ---
        # A pulse exists only when has_pulse is True (set from the experiment
        # name: 0 V = no pulse), whether or not the dye channel is analysed.
        # Aspiration-only controls may still carry a pulse_frame value; it must
        # not reach the rupture detector, exit blanking, or the CSV/plot pulse
        # annotations, otherwise controls get a spurious "Pulse-Induced Rupture".
        dye_params_pre = params.get('dye_uptake_parameters', {})
        pulse_active = bool(dye_params_pre.get('has_pulse', True))
        # Work on a per-trap copy of the sections we modify so injected keys
        # never leak between traps (n_workers == 1 shares self.params).
        params = {**params,
                  'rupture_detection': dict(params.get('rupture_detection', {})),
                  'dye_uptake_parameters': dict(dye_params_pre)}
        params['rupture_detection'].pop('pulse_frame_idx', None)
        params['dye_uptake_parameters'].pop('pulse_frame_idx', None)

        pf_idx: Optional[int] = None
        if pulse_active:
            pf_idx = utils.pulse_frame_to_index(dye_params_pre.get('pulse_frame', 10),
                                                first_frame_number)
            params['rupture_detection']['pulse_frame_idx'] = pf_idx
            # Expose the resolved index so downstream analyzers can reuse it
            # without re-doing the offset calculation.
            params['dye_uptake_parameters']['pulse_frame_idx'] = pf_idx

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
        if pf_idx is not None and 0 <= pf_idx < len(time_data):
            pulse_time = time_data[pf_idx]

        # 5. Data Safety Check
        if not (det_res and det_res['protrusion_lengths_um'] and any(p > 0 for p in det_res['protrusion_lengths_um'])):
            return {'trap_index': trap_index, 'status': 'no_detection'}

        # --- Build the pulse-window frame set ---
        # Used to annotate exported CSVs with an Is_Pulse_Window flag.
        # Empty for experiments without a pulse.
        pulse_blanking = params.get('rupture_detection', {}).get('pulse_exit_blanking_frames', 5)
        if pf_idx is not None:
            # One frame before the pulse position through the blanking window after it.
            blanked_frames: set = set(range(max(0, pf_idx - 1), pf_idx + pulse_blanking + 1))
        else:
            blanked_frames = set()

        # pulse_idx_for_data: 0-based index stored in trap_data for aggregate plots
        # that split metrics at the pulse boundary.
        pulse_idx_for_data = pf_idx

        trap_data = {
            'trap_index':    trap_index,
            'time':          time_data,
            'protrusions':   det_res['protrusion_lengths_um'],
            'pip':           pip_x,
            'thr':           thr_prot,
            'rupture_idx':   rupture_idx,
            'pulse_idx':     pulse_idx_for_data,
            'is_pulse_window': [i in blanked_frames for i in range(len(time_data))],
            'area':          det_full.results.get('total_area_um2', []),
            'area_prot':     det_full.results.get('protrusion_area_um2', []),
            'area_body':     det_full.results.get('body_area_um2', []),
            'solidity':      det_full.results.get('body_solidity', [])
        }

        # Per-trap figures are combined into multipanel PDFs (all traps on one
        # page per plot type) after all workers finish. Each worker therefore
        # returns the small arrays those panels need in `plot_data`, and only
        # writes its own single-trap PDFs when explicitly requested.
        save_singles = params.get('plotting_parameters', {}).get('save_individual_trap_plots', False)
        plot_data: Dict[str, Any] = {
            'pipette_x':    det_res['pipette_start_x_used'],
            'pulse_time':   pulse_time,
            'rupture_time': rupture_time,
            'trace': {
                'time':        list(time_data),
                'protrusions': list(det_res['protrusion_lengths_um']),
                'intensities': det_res.get('downstream_intensities'),
            },
        }

        # Export Raw CSV and Plot Trace.
        try:
            det_full.export_results_to_csv(
                dirs['root'], f"trap_{trap_index+1:02d}",
                dir_full=dirs['full_csv'],
                time_points=time_data,
            )
        except Exception:
            worker_logger.warning(f"Trap {trap_index+1}: CSV export failed.", exc_info=True)

        if save_singles:
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
        # Membrane kymograph (centre-line slice of the membrane channel):
        # writes Trap_XX_Membrane_Kymograph.csv (plus a single-trap PDF only when
        # save_individual_trap_plots is on); the matrix goes into plot_data.
        # The dye-channel kymograph is written separately by
        # DyeUptakeAnalyzer.export_kymograph_csv (see section 8), because it
        # uses mask-scoped column averages rather than a centre-line slice.
        if params.get('workflow_settings', {}).get('create_kymographs', True):
            try:
                kymo = create_kymograph_for_trap(
                    rois, det_res, params, dirs['tracking_visuals'],
                    trap_index + 1, time_data, save_plot=save_singles,
                )
                if kymo.display_matrix is not None:
                    plot_data['membrane_kymo'] = {
                        'matrix':         kymo.display_matrix,
                        'protrusions_px': list(kymo.protrusions_px),
                    }
            except Exception:
                worker_logger.warning(
                    f"Trap {trap_index+1}: Kymograph generation failed.",
                    exc_info=True,
                )

        # 7. Viscoelastic Fitting
        # Default: protrusion was detected but fitting was not performed.
        # Updated to 'success' or 'fit_failed' below depending on outcome.
        fit_status = 'detection_only'
        trap_fit_rows = []
        
        if params.get('model_parameters', {}).get('perform_fitting', True):
            if pulse_time is not None:
                # The Jeffreys / Burgers / Kelvin-Voigt models all assume the cell is
                # incompressible.  Electroporation causes osmotic swelling (volume increase),
                # which violates that assumption.  Fitted E and η values from a pulsed
                # experiment are not physically interpretable, so we skip fitting rather
                # than silently produce misleading numbers.
                worker_logger.info(
                    f"Trap {trap_index+1}: Viscoelastic fitting skipped — "
                    f"electroporation pulse applied; incompressibility assumption violated."
                )
            else:
                try:
                    dirs['fitting'].mkdir(parents=True, exist_ok=True)
        
                    r_eff    = compute_reff(params['model_parameters']['channel_width_um'],
                                            params['model_parameters']['channel_height_um'],
                                            params['model_parameters']['fstar'])
                    t_pts    = np.array(time_data[:rupture_idx+1]) if rupture_idx is not None else np.array(time_data)
                    l_pts    = np.array(det_res['protrusion_lengths_um'][:rupture_idx+1]) if rupture_idx is not None else np.array(det_res['protrusion_lengths_um'])
                    delta_p  = params['experiment_parameters']['constant_pressure']
                    fit_run_params = {**params, **params.get('fitting_parameters', {})}
        
                    def _run_fitter(t, l, phase, rupture_det, rupt_time):
                        fitter = FittingMFA(t, l, r_eff, delta_p,
                                            rupture_detected=rupture_det,
                                            rupture_time=rupt_time,
                                            phase=phase)
                        fitter.run(params=fit_run_params)
                        return fitter
        
                    def _collect_rows(fitter, phase_label):
                        rows = []
                        if fitter.best_fit_model_name:
                            for model_name, res_fit in fitter.fit_results.items():
                                rows.append({
                                    'Trap_Number': trap_index + 1,
                                    'Phase':       phase_label,
                                    'Model_Name':  model_name,
                                    'R_Squared':   res_fit['r_squared'],
                                    'AIC':         res_fit['aic'],
                                    'N_Params':    res_fit['n_params'],
                                    'Is_Best_Fit': model_name == fitter.best_fit_model_name
                                })
                        return rows
        
                    # Full trace — only run for non-pulsed experiments
                    fitter_full = _run_fitter(t_pts, l_pts, 'full',
                                              rupture_det=(rupture_idx is not None),
                                              rupt_time=rupture_time)
                    trap_fit_rows.extend(_collect_rows(fitter_full, 'full'))
        
                    if fitter_full.best_fit_model_name:
                        try:
                            plotter = Plotting_MFA.MFAPlotter(fitter_full, params=params)
                            plotter.create_analysis_plot(
                                trap_index + 1,
                                dirs['fitting'] / f"trap_{trap_index+1:02d}_analysis_plot.png"
                            )
                        except Exception:                          # ← this was missing
                            worker_logger.warning(f"Trap {trap_index+1}: Fit plot failed.", exc_info=True)

                    fit_status = 'success'

                except Exception:
                    fit_status = 'fit_failed'
                    worker_logger.warning(f"Trap {trap_index+1}: Fitting failed.", exc_info=True)

        # 8. Dye Uptake Analysis
        dye_results = None
        dye_config = config_dict.get('dye_mmap')
        if params.get('dye_uptake_parameters', {}).get('enable', False) and dye_config and dye_config['path']:
            try:
                dirs['dye'].mkdir(parents=True, exist_ok=True)
                all_dye_frames = np.memmap(dye_config['path'], dtype=dye_config['dtype'], mode='r', shape=dye_config['shape'])
                dye_rois = [cropper.process_frame(f, trap_index, skip_rotation=True) for f in all_dye_frames]

                uptake_analyzer = DyeUptakeAnalyzer(rois, dye_rois, det_res['pipette_start_x_used'], thr_prot, config_dict['tuned_threshold_body'], rupture_idx, params, frame_masks=det_res.get('frame_masks', []), start_idx=det_res.get('entry_frame_index', 0))
                dye_results = uptake_analyzer.run(time_data)
                uptake_analyzer.export_csv(trap_index + 1, dirs['dye'])
                uptake_analyzer.export_kymograph_csv(trap_index + 1, dirs['dye'])
                
                # Pass pulse_time AND rupture_time down.  Uptake analysis
                # no longer truncates at rupture; the dashboard now marks
                # rupture as a red horizontal line instead.
                if save_singles:
                    Plotting_MFA.plot_dye_uptake_dashboard(
                        dye_results, trap_index + 1, dirs['dye'], params,
                        det_res['pipette_start_x_used'],
                        pulse_time=pulse_time,
                        rupture_time=rupture_time,
                    )
                
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
                actin_analyzer.export_kymograph_csv(trap_index + 1, dirs['actin'])
                actin_analyzer.save_zones_debug_video(trap_index + 1, dirs['actin'])
                actin_analyzer.save_cortex_debug_video(trap_index + 1, dirs['actin'])
                # Retained plots: kymograph (image) and zone dynamics.
                # Removed plots: dashboard (_Ratio), cortex-structure, profiles.
                if save_singles:
                    Plotting_MFA.plot_actin_kymograph(actin_results, trap_index + 1, dirs['actin'], params, det_res['pipette_start_x_used'], pulse_time=pulse_time, rupture_time=rupture_time)
                    Plotting_MFA.plot_actin_zones(actin_results, trap_index + 1, dirs['actin'], params, pulse_time=pulse_time, rupture_time=rupture_time)
                plot_data['actin'] = {
                    k: v for k, v in actin_results.items()
                    if k in ('time_s', 'spatial_profiles') or k.startswith('zone_')
                }
            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Actin analysis failed.", exc_info=True)
                
        return {
            'trap_index': trap_index,
            'status':     fit_status,   # 'success' | 'detection_only' | 'fit_failed'
            'data':       trap_data,
            'fit_rows':   trap_fit_rows,
            'dye_data':   dye_results,
            'plot_data':  plot_data,
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
                    if hasattr(mmap_obj, '_mmap') and mmap_obj._mmap is not None:
                        mmap_obj._mmap.close()
                except Exception:
                    pass
        del all_frames, all_dye_frames, all_actin_frames
        gc.collect()

class MFAAnalysis:
    """
    Manages the end-to-end analysis workflow for an MFA experiment.
    """

    def __init__(self, path: Path, experiment_id: str,
                 params: Optional[Union['MFAConfig', Dict[str, Any]]] = None,
                 output_root: Optional[Path] = None,
                 use_session_cache: bool = True) -> None:
        """
        Args:
            path: Folder containing the raw .tif image sequence.
            experiment_id: Name of the results folder created under output_root.
            params: Validated MFAConfig (preferred) or a plain config dict.
            output_root: Where result folders are written. Defaults to the
                'Output' folder next to this script (the bulk analysis
                scripts read from there by default).
            use_session_cache: If False, ignore any saved session_cache.json
                and run the interactive trap setup again.
        """
        if not path.exists():
            raise FileNotFoundError(f"The specified data path does not exist: {path}")
        self.path = path
        self.experiment_id = experiment_id
        self.output_root = Path(output_root) if output_root is not None else None
        self.use_session_cache = use_session_cache
        
        # Accept a validated MFAConfig object (typed) or a plain dict (backward compat).
        if hasattr(params, 'model_dump'):
            self.config: Optional[MFAConfig] = params
            self.params: Dict[str, Any] = params.model_dump()
        elif hasattr(params, 'dict'):
            self.config = params
            self.params = params.dict()
        else:
            self.config = None
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
        output_base = self.output_root if self.output_root is not None else default_output_root()
        results_dir = output_base / self.experiment_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define subfolders
        subdirs = {
            'root': results_dir,
            'full_csv': results_dir / "Full protrusion detection",
            'tracking_visuals': results_dir / "Tracking visuals", 
            'morphology': results_dir / "Cell Morphology",
            'fitting': results_dir / "Fitting results",
            'dye': results_dir / "Dye Uptake",
            'actin': results_dir / "Actin Quantification" 
        }
        
        # Create core CSV folders immediately
        subdirs['full_csv'].mkdir(parents=True, exist_ok=True)
        subdirs['tracking_visuals'].mkdir(parents=True, exist_ok=True)
        subdirs['morphology'].mkdir(parents=True, exist_ok=True)
            
        return results_dir, subdirs
    
    def cleanup(self):
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
                    


    # =========================================================================
    # SESSION CACHE  (save/load interactive-setup results between runs)
    # =========================================================================

    def _save_session_cache(self, trap_configs: List[Dict[str, Any]]) -> None:
        """
        Writes a JSON file to the output folder that captures every value that
        was set interactively in Phase 1:

          - rotation_angle
          - per-trap ROI coordinates, pipette X, and detection thresholds
          - the list of source membrane TIF files (used for an integrity check
            when the cache is reloaded)

        The file is human-readable and can be edited by hand if you want to
        adjust a threshold or ROI without rerunning the GUI.
        """
        import json

        # roi_coords are tuples/lists of 4 ints: (x, y, width, height).
        # json.dump cannot handle numpy integers, so cast everything explicitly.
        cache = {
            'experiment_id':  self.experiment_id,
            'rotation_angle': float(self.cropper.rotation_angle),
            'trap_configs': [
                {
                    'trap_index': int(cfg['trap_index']),
                    'roi_coords': [int(v) for v in self.cropper.all_trap_rois[cfg['trap_index']]],
                    'pip':        float(cfg['pip']),
                    'thr_prot':   float(cfg['thr_prot']),
                    'thr_body':   float(cfg['thr_body']),
                }
                for cfg in trap_configs
            ],
            # Store the source-file list so we can warn the user if the data
            # folder has changed since the cache was written.
            'source_files': [str(p) for p in self.file_reader.tif_files],
        }

        cache_path = self.results_dir / 'session_cache.json'
        with open(cache_path, 'w') as fh:
            json.dump(cache, fh, indent=2)
        logger.info(f"Session cache saved → {cache_path.name}")

    def _load_session_cache(self) -> Optional[Dict[str, Any]]:
        """
        Reads session_cache.json from the output folder, if it exists.

        Returns the parsed dictionary, or None when no cache file is present.
        Emits a warning (but does not abort) when the cached source-file list
        differs from the files currently found by the file reader — this happens
        when the data folder is reorganised after the original run.
        """
        import json

        cache_path = self.results_dir / 'session_cache.json'
        if not cache_path.exists():
            return None

        with open(cache_path, 'r') as fh:
            cache = json.load(fh)

        # Integrity check: compare the membrane TIF list recorded at save time
        # with the one the file reader discovered right now.
        cached_files  = set(cache.get('source_files', []))
        current_files = set(str(p) for p in self.file_reader.tif_files)
        if cached_files != current_files:
            logger.warning(
                "Session cache was created with a different set of source files.\n"
                "  Cached :  %d file(s)\n"
                "  Current:  %d file(s)\n"
                "Proceeding with cached setup — verify results carefully.",
                len(cached_files), len(current_files),
            )

        return cache

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
            # If session_cache.json exists in the output folder, Phase 1 is
            # skipped automatically and the pipeline jumps to Phase 2.
            # Delete session_cache.json to force a fresh interactive setup.
            logger.info("\n== PHASE 1: INTERACTIVE TRAP SETUP ==")
            existing_cache = self._load_session_cache() if self.use_session_cache else None
            if existing_cache is not None:
                logger.info(
                    "Re-using the trap setup saved in session_cache.json "
                    "(run with --fresh to redo the interactive setup)."
                )

            if existing_cache is not None:
                # ── Restore cropper state from cache ──────────────────────────
                # Rebuild a minimal CropImage so that the rest of the pipeline
                # (which reads self.cropper.rotation_angle and
                # self.cropper.all_trap_rois) behaves identically to a live run.
                crop_gui_params = {
                    **self.params.get('experiment_parameters', {}),
                    **self.params.get('workflow_settings', {}),
                    **self.params.get('image_processing', {}),
                    'guv_settings': self.params.get('guv_settings', {}),
                }
                self.cropper = CropImage(crop_gui_params)
                self.cropper.max_traps = self.params.get('experiment_parameters', {}).get('max_traps', 20)
                self.cropper.rotation_angle = existing_cache['rotation_angle']
                self.cropper.file_reader = self.file_reader  # needed by save_trap_map

                # Allocate the ROI list and fill in only the cached traps.
                n_slots = max(
                    self.params.get('experiment_parameters', {}).get('max_traps', 20),
                    max(c['trap_index'] for c in existing_cache['trap_configs']) + 1,
                )
                self.cropper.all_trap_rois = [None] * n_slots
                for cfg in existing_cache['trap_configs']:
                    # roi_coords was serialised as a plain list; restore as tuple
                    # so it matches what the live GUI produces.
                    self.cropper.all_trap_rois[cfg['trap_index']] = tuple(cfg['roi_coords'])

                trap_configs = [
                    {
                        'trap_index': int(cfg['trap_index']),
                        'pip':        int(cfg['pip']),   # JSON deserialises ints as float; cast back
                        'thr_prot':   int(cfg['thr_prot']),
                        'thr_body':   int(cfg['thr_body']),
                    }
                    for cfg in existing_cache['trap_configs']
                ]
                logger.info(
                    f"Session cache loaded: {len(trap_configs)} trap(s), "
                    f"rotation = {existing_cache['rotation_angle']:.2f}°"
                )

            else:
                # ── Normal interactive Phase 1 ─────────────────────────────────
                while True:
                    crop_gui_params = {
                        **self.params.get('experiment_parameters', {}),
                        **self.params.get('workflow_settings', {}),
                        **self.params.get('image_processing', {}),
                        'guv_settings': self.params.get('guv_settings', {}),
                    }
                    self.cropper = CropImage(crop_gui_params)
                    self.cropper.max_traps = self.params.get('experiment_parameters', {}).get('max_traps', 20)

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

                # Save everything that was just configured so the next run
                # skips Phase 1 automatically.
                self._save_session_cache(trap_configs)

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
            
            # Auto-detect filename numbering scheme so the pulse_frame config
            # value (which matches the filename suffix, e.g. t0010) maps to
            # the correct 0-based position in the sorted file list.
            first_frame_number = utils.get_first_frame_number(self.file_reader.tif_files)
            logger.info(
                f"First frame's numeric suffix: {first_frame_number}. "
                f"pulse_frame in config will be treated as filename suffix "
                f"and offset by this amount to get the array position."
            )

            # Prepare arguments for each worker process
            worker_args = []
            for config in trap_configs:
                trap_index = config['trap_index']
                full_config = config.copy()
                full_config.update({
                    'params': self.params,
                    'time_data': self.file_reader.time_data,
                    'first_frame_number': first_frame_number,
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
            # Guard against two failure modes:
            #   1. img is None  — complete read failure (e.g. missing file).
            #   2. img has the wrong shape — partial/corrupt read where OpenCV
            #      returns a small array instead of the expected frame size.
            # In both cases we substitute a zero frame so the stack stays
            # consistent and downstream code can handle the gap gracefully.
            if img is None or img.shape[:2] != (h, w):
                if img is not None:
                    logger.warning(
                        f"{tag} frame {i} ({f.name}): unexpected shape "
                        f"{img.shape[:2]} (expected {(h, w)}) — substituting zeros."
                    )
                fp[i] = np.zeros(shape[1:], dtype=dtype)
            else:
                fp[i] = img
        
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

        # All traps' per-trap plots, one multipanel PDF per plot type.
        Plotting_Multipanel.plot_trap_multipanels(
            all_results, self.subdirs, self.experiment_id, self.params
        )
        
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
        
# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python MFA_analysis.py",
        description="Microfluidic aspiration (MFA) analysis of one experiment folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical workflow:\n"
            "  1. python MFA_analysis.py --init-config my_exp.yaml --data-folder D:/data/260128_...\n"
            "  2. python MFA_analysis.py my_exp.yaml --check     (no windows; checks config and images)\n"
            "  3. python MFA_analysis.py my_exp.yaml             (interactive setup, then analysis)\n"
            "\n"
            "Your file holds the experiment's settings; everything else comes from\n"
            "config/defaults.yaml. See README.md for details."
        ),
    )
    parser.add_argument("config_path", nargs="?", type=Path,
                        help="Path to the YAML config file.")
    parser.add_argument("--data-folder", type=Path,
                        help="Override paths.data_folder (folder with the .tif images).")
    parser.add_argument("--experiment-id",
                        help="Override paths.experiment_id (name of the results folder). "
                             "Defaults to the data folder's name when --data-folder is given.")
    parser.add_argument("--output-dir", type=Path,
                        help=f"Where result folders are written (default: {default_output_root()}).")
    parser.add_argument("--check", action="store_true",
                        help="Validate the config and scan the images, then exit. Opens no windows.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore a saved session_cache.json and redo the interactive trap setup.")
    parser.add_argument("--debug", action="store_true",
                        help="Show detailed (DEBUG) log messages on screen.")
    parser.add_argument("--show-config", action="store_true",
                        help="Print every setting this run would use, and which file set it, then exit.")
    parser.add_argument("--init-config", type=Path, metavar="NEW_CONFIG",
                        help="Create a new config by copying an existing one with the new "
                             "--data-folder / --experiment-id filled in, then exit.")
    parser.add_argument("--from", dest="copy_from", type=Path, default=DEFAULT_CONFIG,
                        metavar="CONFIG",
                        help=f"Config copied by --init-config (default: {DEFAULT_CONFIG.name}).")
    return parser


def _set_yaml_value(text: str, key: str, value: str) -> str:
    """Replace the value of the first `  key: ...` line, keeping comments and layout."""
    import re
    quoted = "'" + str(value).replace("'", "''") + "'"
    pattern = re.compile(rf"^(\s+{key}:)[^\n]*", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"no '{key}:' entry to fill in")
    return pattern.sub(lambda m: f"{m.group(1)} {quoted}", text, count=1)


def init_config(dest: Path, source: Path, data_folder: Optional[Path],
                experiment_id: Optional[str]) -> bool:
    """
    Copy an existing config (by default config.yaml) to `dest`, replacing its
    data folder and experiment name. All other settings are kept, so copying
    the config of a similar experiment is the quickest way to start a new one.
    """
    if dest.exists():
        logger.error(f"{dest} already exists; choose another name or delete it first.")
        return False
    if not source.exists():
        logger.error(f"Config to copy not found: {source}")
        return False

    text = source.read_text(encoding="utf-8")
    if data_folder is not None:
        experiment_id = experiment_id or data_folder.name
    try:
        if data_folder is not None:
            text = _set_yaml_value(text, "data_folder", str(data_folder))
        if experiment_id:
            text = _set_yaml_value(text, "experiment_id", experiment_id)
    except ValueError as e:
        logger.error(f"Cannot fill in {source.name}: {e}.")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    logger.info(f"Created {dest} (copy of {source.name}).")
    if data_folder is None and not experiment_id:
        logger.info("Next: open it and set paths.data_folder and paths.experiment_id.")
    logger.info(f"Then check it with:  python MFA_analysis.py {dest} --check")
    return True


def load_config(config_path: Path, data_folder: Optional[Path] = None,
                experiment_id: Optional[str] = None
                ) -> Tuple[Optional[MFAConfig], Optional[config_loader.LayeredConfig]]:
    """
    Combine the defaults, the user's file and command-line overrides, then
    validate. Logs a readable error and returns (None, None) on failure.
    """
    overrides: Dict[str, Any] = {}
    if data_folder is not None:
        overrides["data_folder"] = str(data_folder)
        experiment_id = experiment_id or data_folder.name
    if experiment_id is not None:
        overrides["experiment_id"] = experiment_id

    try:
        layered = config_loader.load_layers(config_path, {"paths": overrides} if overrides else None)
    except config_loader.ConfigFileError as e:
        logger.error(str(e))
        return None, None

    try:
        return validate_config(layered.data), layered
    except ValidationError as e:
        logger.error(f"{config_path} has {e.error_count()} problem(s):\n"
                     f"{format_validation_error(e, layered.sources)}")
    except ValueError as e:
        logger.error(f"{config_path}: {e}")
    return None, None


def show_config(config: MFAConfig, layered: config_loader.LayeredConfig) -> str:
    """All settings of a run, each value annotated with the file that set it (blank = defaults)."""
    header = (f"# Settings from: {' <- '.join(layered.layers)}\n"
              f"# Values without a comment come from config/defaults.yaml.\n\n")
    return header + config_loader.render_settings(
        config.model_dump(mode="json"), layered.sources, hide_source="defaults")


def save_effective_config(config: MFAConfig, layered: config_loader.LayeredConfig,
                          config_path: Path, results_dir: Path) -> Path:
    """Record exactly which settings produced this run's results."""
    from datetime import datetime
    path = Path(results_dir) / "effective_config.yaml"
    header = (f"# Complete settings used for this run (written {datetime.now():%Y-%m-%d %H:%M}).\n"
              f"# Config file: {Path(config_path).resolve()}\n"
              f"# This file is a full config: run it again with\n"
              f"#     python MFA_analysis.py {path.name}\n")
    path.write_text(header + show_config(config, layered).split("\n", 2)[2], encoding="utf-8")
    return path


def check_setup(config: MFAConfig, output_root: Optional[Path] = None) -> bool:
    """
    Dry run for new users: scans the image folder with the same code the
    pipeline uses and reports what it found, without opening any windows
    or writing results. Returns False if a problem would stop the analysis.
    """
    params = config.model_dump()
    problems: List[str] = []
    notes: List[str] = []

    reader = FileRead(config.paths.data_folder, params)
    try:
        reader.run()
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    dye = params["dye_uptake_parameters"]
    actin = params["actin_parameters"]
    n = len(reader.tif_files)

    logger.info("== SETUP CHECK ==")
    logger.info(f"Data folder   : {config.paths.data_folder}")
    logger.info(f"Results folder: {(output_root or default_output_root()) / config.paths.experiment_id}")
    logger.info(f"Membrane      : {n} frames (pattern '{dye['membrane_channel_pattern']}')")
    if dye["enable"]:
        logger.info(f"Dye           : {len(reader.dye_files)} frames (pattern '{dye['dye_channel_pattern']}')")
        if not reader.dye_files:
            problems.append(f"Dye analysis is enabled but no file names contain "
                            f"'{dye['dye_channel_pattern']}' (dye_uptake_parameters.dye_channel_pattern).")
    if actin["enable"]:
        logger.info(f"Actin         : {len(reader.actin_files)} frames (pattern '{actin['channel_pattern']}')")
        if not reader.actin_files:
            problems.append(f"Actin analysis is enabled but no file names contain "
                            f"'{actin['channel_pattern']}' (actin_parameters.channel_pattern).")
        elif actin["channel_pattern"] == dye["membrane_channel_pattern"]:
            notes.append("Actin and membrane use the same channel pattern; actin is measured "
                         "on the membrane images. Change actin_parameters.channel_pattern if "
                         "you have a separate actin channel.")

    if n < 2:
        problems.append("Fewer than 2 membrane frames found; nothing to analyse.")
    else:
        dt = (reader.time_data[-1] - reader.time_data[0]) / (n - 1)
        logger.info(f"Frame interval: {dt:.4f} s ({reader.metadata_method_used})")
        cfg_dt = params["experiment_parameters"]["frame_interval"]
        if reader.metadata_method_used and not reader.metadata_method_used.startswith("Manual"):
            if abs(dt - cfg_dt) > 0.1 * cfg_dt:
                notes.append(f"Image metadata says {dt:.4f} s per frame but the config's "
                             f"frame_interval is {cfg_dt} s. The metadata value is used.")

    first = utils.get_first_frame_number(reader.tif_files)
    logger.info(f"First frame number in file names: {first}")
    if dye["has_pulse"]:
        idx = utils.pulse_frame_to_index(dye["pulse_frame"], first)
        if 0 <= idx < n:
            logger.info(f"Pulse         : frame {dye['pulse_frame']:g} -> position {idx} "
                        f"(t = {reader.time_data[idx]:.2f} s)")
        else:
            problems.append(f"pulse_frame {dye['pulse_frame']:g} is outside the recording "
                            f"(only {n} frames).")
    else:
        logger.info("Pulse         : none (aspiration-only control)")

    name = experiment_naming.parse_experiment_name(config.paths.experiment_id)
    if name is None:
        notes.append(
            "experiment_id does not follow the naming rule "
            f"({experiment_naming.NAMING_RULE}), so the pulse settings come from the "
            "config instead of the name, and Bulk_Analysis will skip this experiment. "
            "See README.md.")
    else:
        logger.info(f"Experiment    : {name.describe()} (from the experiment name)")
        cfg_p = params["experiment_parameters"]["constant_pressure"]
        if abs(cfg_p - name.pressure_pa) > 1e-6:
            notes.append(f"constant_pressure is {cfg_p:g} Pa but the experiment name says "
                         f"{name.pressure_pa} Pa. The config value is used for the analysis.")

    cache = (output_root or default_output_root()) / config.paths.experiment_id / "session_cache.json"
    if cache.exists():
        notes.append(f"A saved trap setup exists ({cache.name}); it will be reused. "
                     "Add --fresh to redo the interactive setup.")

    for note in notes:
        logger.warning(f"NOTE: {note}")
    for problem in problems:
        logger.error(f"PROBLEM: {problem}")
    if problems:
        logger.error(f"Setup check found {len(problems)} problem(s). Fix them before running.")
        return False
    logger.info("Setup check passed. Run again without --check to start the analysis.")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point. Returns a process exit code (0 = success)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    utils.setup_logging(debug_mode=args.debug)

    if args.init_config is not None:
        return 0 if init_config(args.init_config, args.copy_from,
                                args.data_folder, args.experiment_id) else 1

    if args.config_path is None:
        parser.print_help()
        return 2

    config, layered = load_config(args.config_path, args.data_folder, args.experiment_id)
    if config is None:
        return 1

    if args.show_config:
        print(show_config(config, layered))
        return 0

    if args.check:
        return 0 if check_setup(config, args.output_dir) else 1

    if args.debug:
        config.workflow_settings.debug_mode = True

    analysis = None
    try:
        analysis = MFAAnalysis(config.paths.data_folder, str(config.paths.experiment_id), config,
                               output_root=args.output_dir,
                               use_session_cache=not args.fresh)
        saved = save_effective_config(config, layered, args.config_path, analysis.results_dir)
        logger.info(f"Settings for this run saved to {saved}")
        return 0 if analysis.run_analysis() else 1
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user.")
        return 130
    except Exception:
        logger.error("Fatal error", exc_info=True)
        return 1
    finally:
        if analysis:
            analysis.cleanup()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
