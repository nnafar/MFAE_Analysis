# -*- coding: utf-8 -*-
"""
Analysis of one experiment (run it with run_experiment.py; command line in mfae.cli).

1. Reads the images (.tif folder or .czi file) and the settings.
2. Interactive setup: rotation, trap boxes, channel entrances and thresholds
   (or reuses Run_Info/session_cache.json from an earlier run).
3. Analyses every trap in parallel (static_parallel_worker): membrane masks and
   tip tracking, kymographs, tracking movies, dye uptake, actin, fits.
4. Writes the all-trap figures, trap_summary.csv and README.txt
   (folder layout: mfae.files.output_layout).
"""

import atexit
import os

# ---------------------------------------------------------------------------
# Silence noisy libtiff/OpenCV C++ warnings BEFORE cv2 is imported.
# These lines (e.g. "cv::TIFF_Warning TIFFReadDirectory: Unknown field with
# tag 18246") come from OpenCV's C++ layer straight to stderr and cannot be
# intercepted by Python's logging module. The env var below is read by
# OpenCV at import time; setting it after `import cv2` has no effect.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import gc
import multiprocessing
import traceback
import logging
import logging.handlers
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

# Internal Imports
from mfae.files.readers import FileSave, make_file_reader
from mfae.imaging.trap_rois import CropImage
from mfae.membrane.detection import LineDetectionMFA
from mfae.mechanics.models import compute_reff, compute_shear_metrics
from mfae.mechanics.fitting import FittingMFA
from mfae.membrane.kymograph import create_kymograph_for_trap
from mfae.quantification.dye_uptake import DyeUptakeAnalyzer
from mfae.quantification.actin import ActinAnalyzer
from mfae.membrane.tracking_movie import save_tracking_movie
from mfae.files import output_layout
from mfae.config.schema import MFAConfig
from mfae.config import loader as config_loader
from mfae.plotting import figures
from mfae.plotting import multipanel
from mfae import utils

logger = logging.getLogger(__name__)

from mfae.paths import DEFAULT_OUTPUT_ROOT  # noqa: E402



# The lab's standard experiment configs; templates for --init-config.
DEFAULT_CONFIG = config_loader.CONFIG_DIR / "config_tiff_template.yaml"    # TIFF folders
DEFAULT_CZI_CONFIG = config_loader.CONFIG_DIR / "config_czi_template.yaml"  # .czi files


def default_experiment_id(data_folder: Path) -> str:
    """Experiment name implied by the data location: the folder name, or a .czi file's name."""
    data_folder = Path(data_folder)
    return data_folder.stem if data_folder.suffix.lower() == ".czi" else data_folder.name


def threshold_scale(params: Dict[str, Any]) -> str:
    """The scale the interactive thresholds are measured on (recorded in session_cache.json)."""
    if utils.membrane_enhancement_enabled(params):
        fs = params.get('image_processing', {}).get('membrane_snr_full_scale', 40.0)
        return f"membrane SNR, 255 = {float(fs):g} sigma"
    return "legacy (CLAHE)"


def default_output_root() -> Path:
    """Folder that per-experiment results are written to by default."""
    return DEFAULT_OUTPUT_ROOT

# =============================================================================
# TOP-LEVEL STATIC WORKER FUNCTION (Optimized for Shared Memory)
# =============================================================================

def trap_summary_row(trap_index: int, det_res: Optional[Dict[str, Any]], time_data: List[float],
                     pulse_idx: Optional[int], status: str, dye_done: bool = False,
                     actin_done: bool = False) -> Dict[str, Any]:
    """
    One row of 0_Overview/trap_summary.csv: what happened in this trap.
    Frame numbers count from 1, as in Fiji and the movie labels.
    """
    def frame(i):
        return int(i) + 1 if i is not None else None

    def t(i):
        return round(float(time_data[i]), 3) if i is not None and 0 <= i < len(time_data) else None

    row: Dict[str, Any] = {'Trap': trap_index + 1, 'Status': status}
    if not det_res:
        return row
    segs = det_res.get('cell_segments') or []
    e, x = det_res.get('entry_frame_index'), det_res.get('exit_frame_index')
    lengths = np.asarray(det_res.get('protrusion_lengths_um') or [], dtype=float)
    finite = lengths[np.isfinite(lengths)] if lengths.size else lengths
    at_pulse = (float(lengths[pulse_idx]) if pulse_idx is not None and 0 <= pulse_idx < lengths.size
                and np.isfinite(lengths[pulse_idx]) else None)
    row.update({
        'Cells_seen_in_trap':   len(segs),
        'Analysed_cell':        det_res.get('selected_cell'),
        'Cell_present_at_pulse': det_res.get('cell_at_pulse'),
        'Entry_confirmed':      det_res.get('entry_confirmed'),
        'First_frame':          frame(e),
        'First_time_s':         t(e),
        'Last_frame':           frame(x),
        'Last_time_s':          t(x),
        'Why_tracking_ended':   det_res.get('exit_reason') or 'end of recording',
        'Pulse_frame':          frame(pulse_idx),
        'Protrusion_at_pulse_um': round(at_pulse, 3) if at_pulse is not None else None,
        'Max_protrusion_um':    round(float(finite.max()), 3) if finite.size else None,
        'Rupture_detected':     bool(det_res.get('rupture_detected')),
        'Dye_analysed':         bool(dye_done),
        'Actin_analysed':       bool(actin_done),
    })
    return row


def _init_worker_logging(log_queue) -> None:
    """Pool initializer: route every log record of this worker to the main process."""
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)
    for noisy in ('matplotlib', 'PIL', 'fontTools'):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
    # Fill in every sub-folder of the output layout (see mfae/files/output_layout.py).
    dirs = output_layout.complete(config_dict['output_dirs'])
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
        # Per-trap membrane smoothing chosen during setup (F key / auto).
        if config_dict.get('tuned_denoise_sigma_along') is not None:
            params['image_processing'] = {**params.get('image_processing', {}),
                                          'membrane_denoise_sigma_along':
                                              float(config_dict['tuned_denoise_sigma_along'])}
        # Per-trap 8-bit full scale fitted to the cell's brightness during setup.
        if config_dict.get('tuned_snr_full_scale') is not None:
            params['image_processing'] = {**params.get('image_processing', {}),
                                          'membrane_snr_full_scale':
                                              float(config_dict['tuned_snr_full_scale'])}
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
        det_full.trap_label = f"Trap {trap_index + 1}"
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
            return {'trap_index': trap_index, 'status': 'no_detection',
                    'summary': trap_summary_row(trap_index, None, time_data, pf_idx,
                                                'no cell detected')}

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
                full_name=output_layout.trap_file('protrusion_csv', trap_index + 1),
            )
        except Exception:
            worker_logger.warning(f"Trap {trap_index+1}: CSV export failed.", exc_info=True)

        if save_singles:
            try:
                figures.plot_protrusion_trace(
                    debug_images=det_full.results['debug_images'],
                    time_points=np.array(time_data),
                    protrusions=np.array(det_res['protrusion_lengths_um']),
                    trap_index=trap_index + 1,
                    save_path=dirs['protrusion'] / output_layout.trap_file('protrusion_fig', trap_index + 1),
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
                    rois, det_res, params, dirs['protrusion'],
                    trap_index + 1, time_data, save_plot=save_singles,
                    csv_dir=dirs['membrane_kymo'], pulse_time=pulse_time,
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

        # 6b. Tracking movie: membrane image with the masks, tip and length
        # trace, frame by frame (1_Protrusion_and_Shape/Movies).
        if params.get('workflow_settings', {}).get('save_tracking_movies', True):
            try:
                save_tracking_movie(
                    rois, det_res, time_data, params,
                    dirs['movies'] / output_layout.trap_file('tracking_movie', trap_index + 1),
                    trap_number=trap_index + 1, pulse_idx=pf_idx,
                )
            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Tracking movie failed.", exc_info=True)

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
                            plotter = figures.MFAPlotter(fitter_full, params=params)
                            plotter.create_analysis_plot(
                                trap_index + 1,
                                dirs['fitting'] / output_layout.trap_file('fit_fig', trap_index + 1)
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

                uptake_analyzer = DyeUptakeAnalyzer(rois, dye_rois, det_res['pipette_start_x_used'], thr_prot, config_dict['tuned_threshold_body'], rupture_idx, params, frame_masks=det_res.get('frame_masks', []), start_idx=det_res.get('entry_frame_index', 0),
                                                    end_idx=det_res.get('exit_frame_index'))
                dye_results = uptake_analyzer.run(time_data)
                for key in ('dye_traces', 'dye_kymo', 'dye_movies'):
                    dirs[key].mkdir(parents=True, exist_ok=True)
                uptake_analyzer.export_csv(trap_index + 1, dirs['dye_traces'])
                uptake_analyzer.export_kymograph_csv(trap_index + 1, dirs['dye_kymo'])
                
                # Pass pulse_time AND rupture_time down.  Uptake analysis
                # no longer truncates at rupture; the dashboard now marks
                # rupture as a red horizontal line instead.
                if save_singles:
                    figures.plot_dye_uptake_dashboard(
                        dye_results, trap_index + 1, dirs['dye'], params,
                        det_res['pipette_start_x_used'],
                        pulse_time=pulse_time,
                        rupture_time=rupture_time,
                    )
                
                # Add the missing debug video call here
                uptake_analyzer.save_debug_video(trap_index + 1, dirs['dye_movies'])
                
            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Dye uptake analysis failed.", exc_info=True)

        # 9. Actin Quantification
        actin_results = None
        actin_config = config_dict.get('actin_mmap')
        if params.get('actin_parameters', {}).get('enable', False) and actin_config and actin_config['path']:
            try:
                dirs['actin'].mkdir(parents=True, exist_ok=True)
                if actin_config['path'] == config_dict['mmap_path']:
                    # Same images as the membrane channel: reuse the trap crops
                    # (read-only memmap slices, so neither analysis can alter them).
                    actin_rois = rois
                else:
                    all_actin_frames = np.memmap(actin_config['path'], dtype=actin_config['dtype'], mode='r', shape=actin_config['shape'])
                    actin_rois = [cropper.process_frame(f, trap_index, skip_rotation=True) for f in all_actin_frames]
                
                actin_analyzer = ActinAnalyzer(
                    rois, actin_rois,
                    det_res['pipette_start_x_used'],
                    thr_prot,
                    config_dict['tuned_threshold_body'],
                    params,
                    frame_masks=det_res.get('frame_masks', []),
                    start_idx=det_res.get('entry_frame_index', 0),
                    end_idx=det_res.get('exit_frame_index'),
                )
                actin_results = actin_analyzer.run(time_data)
                for key in ('actin_traces', 'actin_kymo', 'actin_movies'):
                    dirs[key].mkdir(parents=True, exist_ok=True)
                actin_analyzer.export_csv(trap_index + 1, dirs['actin_traces'])
                actin_analyzer.export_kymograph_csv(trap_index + 1, dirs['actin_kymo'])
                actin_analyzer.save_zones_debug_video(trap_index + 1, dirs['actin_movies'])
                actin_analyzer.save_cortex_debug_video(trap_index + 1, dirs['actin_movies'])
                # Retained plots: kymograph (image) and zone dynamics.
                # Removed plots: dashboard (_Ratio), cortex-structure, profiles.
                if save_singles:
                    figures.plot_actin_kymograph(actin_results, trap_index + 1, dirs['actin'], params, det_res['pipette_start_x_used'], pulse_time=pulse_time, rupture_time=rupture_time)
                    figures.plot_actin_zones(actin_results, trap_index + 1, dirs['actin'], params, pulse_time=pulse_time, rupture_time=rupture_time)
                plot_data['actin'] = {
                    k: v for k, v in actin_results.items()
                    if k in ('time_s', 'spatial_profiles') or k.startswith('zone_')
                }
            except Exception:
                worker_logger.warning(f"Trap {trap_index+1}: Actin analysis failed.", exc_info=True)
                
        status_text = {'success': 'analysed', 'detection_only': 'analysed',
                       'fit_failed': 'analysed (fit failed)'}.get(fit_status, fit_status)
        if not det_res.get('entry_confirmed', True):
            status_text = 'entry not confirmed (check movie)'
        return {
            'summary':    trap_summary_row(trap_index, det_res, time_data, pf_idx, status_text,
                                           dye_done=dye_results is not None,
                                           actin_done=actin_results is not None),
            'trap_index': trap_index,
            'status':     fit_status,   # 'success' | 'detection_only' | 'fit_failed'
            'data':       trap_data,
            'fit_rows':   trap_fit_rows,
            'dye_data':   dye_results,
            'plot_data':  plot_data,
        }
            
    except Exception:
        worker_logger.error("A fatal error occurred in worker", exc_info=True)
        return {'trap_index': trap_index, 'status': 'error', 'error': traceback.format_exc(),
                'summary': {'Trap': trap_index + 1, 'Status': 'error (see Run_Info/analysis.log)'}}

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
                 use_session_cache: bool = True,
                 log_mode: str = 'w',
                 retune_traps: Optional[List[int]] = None) -> None:
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
            log_mode: 'w' starts a new analysis.log, 'a' appends to it (batch
                runs do the setup and the analysis in separate passes).
            retune_traps: Trap numbers (1-based, as in the movies and plots)
                whose pipette and thresholds are set again interactively; all
                other traps keep the saved setup. Needs a session_cache.json.
        """
        if not path.exists():
            raise FileNotFoundError(f"The specified data path does not exist: {path}")
        self.path = path
        self.experiment_id = experiment_id
        self.output_root = Path(output_root) if output_root is not None else None
        self.use_session_cache = use_session_cache
        self.retune_traps = sorted({int(t) for t in retune_traps}) if retune_traps else []
        self.log_mode = log_mode
        self.stopped_by_user = False       # Esc during the interactive setup
        
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
        self.file_reader = make_file_reader(self.path, self.params)
        self.results_dir, self.subdirs = self._setup_output_directory()
        self.file_saver = FileSave(output_dir=self.results_dir)
        
        utils.setup_logging(self.params.get('workflow_settings', {}).get('debug_mode', False),
                            output_layout.experiment_file(self.subdirs, 'log'), mode=self.log_mode)
        
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
        
        # Folder layout and file names are defined in mfae/files/output_layout.py:
        #   0_Overview/, 1_Protrusion_and_Shape/{Figures,Data,Movies},
        #   2_Dye_Uptake/..., 3_Actin/..., 4_Mechanics/, Run_Info/
        subdirs = output_layout.build(results_dir)

        # Create core folders immediately (dye/actin/mechanics are created on demand)
        for key in ('overview', 'run_info', 'protrusion', 'full_csv', 'membrane_kymo'):
            subdirs[key].mkdir(parents=True, exist_ok=True)
            
        return results_dir, subdirs
    
    def cleanup(self):
        if hasattr(self, 'file_reader') and hasattr(self.file_reader, 'close'):
            self.file_reader.close()
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

        # roi_coords are tuples/lists of 4 ints: (y_min, y_max, x_min, x_max),
        # as used by utils.crop_single_trap.
        # json.dump cannot handle numpy integers, so cast everything explicitly.
        cache = {
            'experiment_id':  self.experiment_id,
            # What the thresholds below are measured in; a cache whose scale
            # differs from the current settings is not reused.
            'threshold_scale': self._threshold_scale(),
            'rotation_angle': float(self.cropper.rotation_angle),
            'trap_configs': [
                {
                    'trap_index': int(cfg['trap_index']),
                    'roi_coords': [int(v) for v in self.cropper.all_trap_rois[cfg['trap_index']]],
                    'pip':        float(cfg['pip']),
                    'thr_prot':   float(cfg['thr_prot']),
                    'thr_body':   float(cfg['thr_body']),
                    'smooth':     (float(cfg['smooth']) if cfg.get('smooth') is not None else None),
                    'snr_scale':  (float(cfg['snr_scale']) if cfg.get('snr_scale') is not None else None),
                }
                for cfg in trap_configs
            ],
            # Store the source-file list so we can warn the user if the data
            # folder has changed since the cache was written.
            'source_files': [str(p) for p in self.file_reader.tif_files],
        }

        cache_path = output_layout.experiment_file(self.subdirs, 'session_cache')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as fh:
            json.dump(cache, fh, indent=2)
        logger.info(f"Session cache saved → {cache_path.name}")

    def _threshold_scale(self) -> str:
        """Describes the scale the setup thresholds are on (stored in the session cache)."""
        return threshold_scale(self.params)

    def _load_session_cache(self) -> Optional[Dict[str, Any]]:
        """
        Reads session_cache.json from the output folder, if it exists.

        Returns the parsed dictionary, or None when no cache file is present.
        Emits a warning (but does not abort) when the cached source-file list
        differs from the files currently found by the file reader — this happens
        when the data folder is reorganised after the original run.
        """
        import json

        # Run_Info/session_cache.json; results from older versions kept it in
        # the experiment folder itself
        found = [c for c in output_layout.session_cache_candidates(self.results_dir) if c.exists()]
        if not found:
            return None
        cache_path = max(found, key=lambda c: c.stat().st_mtime)   # newest, if both exist
        new_path = output_layout.experiment_file(output_layout.build(self.results_dir), 'session_cache')
        if cache_path != new_path:
            # move a cache from an older version into Run_Info/
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path = Path(cache_path).replace(new_path)
            except OSError:
                pass

        with open(cache_path, 'r') as fh:
            cache = json.load(fh)

        # Integrity check: compare the membrane TIF list recorded at save time
        # with the one the file reader discovered right now.
        # The thresholds only mean something on the scale they were picked on.
        # Caches written before the scale was recorded were made with the CLAHE method.
        cached_scale = cache.get('threshold_scale', 'legacy (CLAHE)')
        if cached_scale != self._threshold_scale():
            logger.warning(
                "session_cache.json was saved with thresholds on a different scale, so it is "
                "not reused and the interactive setup runs again.\n"
                "  Cache  : %s\n  Current: %s",
                cached_scale, self._threshold_scale(),
            )
            return None

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

    def run_analysis(self, setup_only: bool = False) -> bool:
        """
        Run the experiment: setup (interactive, or the saved one), then analysis.
        setup_only: stop once the setup is done and saved (batch runs do every
        setup first, then every analysis without interaction).
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
            if existing_cache is not None and not self.retune_traps:
                logger.info(
                    "Re-using the trap setup saved in session_cache.json "
                    "(run with --fresh to redo the interactive setup, or "
                    "--traps N to redo single traps)."
                )
            elif self.retune_traps and existing_cache is None:
                logger.warning(
                    "--traps %s: there is no usable saved setup for this experiment, so "
                    "the full interactive setup runs instead.",
                    ",".join(map(str, self.retune_traps)))

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
                        # Older caches have no per-trap smoothing -> config default.
                        'smooth':     cfg.get('smooth'),
                        # Older caches: thresholds were on the configured full scale.
                        'snr_scale':  cfg.get('snr_scale'),
                    }
                    for cfg in existing_cache['trap_configs']
                ]
                logger.info(
                    f"Session cache loaded: {len(trap_configs)} trap(s), "
                    f"rotation = {existing_cache['rotation_angle']:.2f}°"
                )
                if self.retune_traps:
                    ok = self._retune_cached_traps(trap_configs)
                    if not ok:
                        return False

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
                        if setup_signal == 'stop':
                            self.stopped_by_user = True
                            return False
                        continue

                    # --- Phase 1b: Per-Trap Parameter Tuning ---
                    setup_signal, trap_configs = self._collect_interactive_parameters()

                    if setup_signal in ('stop', 'restart'):
                        if setup_signal == 'stop':
                            self.stopped_by_user = True
                            return False
                        continue
                    if setup_signal == 'confirm': break

                # Save everything that was just configured so the next run
                # skips Phase 1 automatically.
                self._save_session_cache(trap_configs)

            if not trap_configs:
                logger.info("No traps selected.")
                return True

            if self.cropper: self.cropper.save_trap_map(self.subdirs['overview'])

            if setup_only:
                logger.info(f"Setup saved for {self.experiment_id} ({len(trap_configs)} trap(s)).")
                return True

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
                if list(self.file_reader.actin_files) == list(self.file_reader.tif_files):
                    # Actin is imaged in the membrane channel: reuse those
                    # frames instead of reading and caching them a second time.
                    logger.info("Actin channel is the membrane channel: reusing the membrane images.")
                    actin_mmap_path, actin_shape, actin_dtype = mem_mmap_path, mem_shape, mem_dtype
                else:
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
                    'tuned_denoise_sigma_along': config.get('smooth'),
                    'tuned_snr_full_scale': config.get('snr_scale'),
                    
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
                # Worker processes have no log handlers of their own (on Windows
                # they start fresh), so without this their messages - warnings
                # about failed steps included - never reached the log file.
                # Workers send records through a queue; the main process writes
                # them with its own handlers (console + <experiment>_analysis.log).
                log_queue = multiprocessing.Queue(-1)
                listener = logging.handlers.QueueListener(
                    log_queue, *logging.getLogger().handlers, respect_handler_level=True)
                listener.start()
                try:
                    with multiprocessing.Pool(n_workers, initializer=_init_worker_logging,
                                              initargs=(log_queue,)) as pool:
                        all_results = list(tqdm(pool.map(static_parallel_worker, worker_args), total=len(worker_args)))
                finally:
                    listener.stop()

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
    
    def _retune_cached_traps(self, trap_configs: List[Dict[str, Any]]) -> bool:
        """
        Redo the pipette and threshold windows for the traps in self.retune_traps
        (1-based numbers) and replace their entries in trap_configs and in
        session_cache.json. The trap boxes and all other traps are kept.
        Returns False when the user stopped (Esc).
        """
        by_index = {c['trap_index']: c for c in trap_configs}
        unknown = [t for t in self.retune_traps if (t - 1) not in by_index]
        if unknown:
            logger.warning(
                "--traps: trap(s) %s are not in the saved setup (saved: %s) and are skipped. "
                "To add a trap, run with --fresh.",
                ",".join(map(str, unknown)),
                ",".join(str(i + 1) for i in sorted(by_index)))
        todo = [t - 1 for t in self.retune_traps if (t - 1) in by_index]
        if not todo:
            return True
        logger.info("Redoing the setup of trap(s) %s; the other traps keep the saved setup.",
                    ",".join(str(i + 1) for i in todo))
        while True:
            signal, new = self._collect_interactive_parameters(
                indices=todo, start_pip={i: by_index[i]['pip'] for i in todo})
            if signal == 'stop':
                self.stopped_by_user = True
                return False
            if signal == 'confirm':
                break                       # 'restart': show the same traps again
        for cfg in new:
            by_index[cfg['trap_index']].update(cfg)
        self._save_session_cache(trap_configs)
        return True

    def _collect_interactive_parameters(self, indices: Optional[List[int]] = None,
                                        start_pip: Optional[Dict[int, int]] = None
                                        ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Loops through EACH selected trap to let user tune Pipette & Thresholds.

        indices / start_pip: redo only these traps (0-based), starting the
        pipette window at their saved entrance position (used by --traps).
        """
        configs = []
        if not self.loader or not self.cropper: return 'stop', []
        
        selected_indices = indices if indices is not None else self.cropper.selected_traps
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
        
                pip0 = ([int(start_pip[trap_index]), 0] if start_pip and trap_index in start_pip
                        else self.cropper.pipette_coords)
                det_interactive = LineDetectionMFA([roi_img], pip0, self.params)
                
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
                    'thr_body': interactive_res.get('threshold_body', interactive_res['threshold_prot']),
                    'smooth': interactive_res.get('denoise_sigma_along'),
                    'snr_scale': interactive_res.get('snr_full_scale'),
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
            df.to_csv(output_layout.experiment_file(self.subdirs, 'fits_csv'), index=False)

        self._perform_shear_analysis()

        # 0_Overview/trap_summary.csv: one row per trap
        rows = [r['summary'] for r in sorted(all_results, key=lambda r: r.get('trap_index', 0))
                if r.get('summary')]
        if rows:
            path = output_layout.experiment_file(self.subdirs, 'trap_summary')
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info(f"Trap summary saved: {path.relative_to(self.results_dir)}")

        # All traps' per-trap plots, one multipanel PDF per plot type.
        multipanel.plot_trap_multipanels(
            all_results, self.subdirs, self.experiment_id, self.params
        )
        
        # Call the morphology aggregate plotter (Using the corrected 'self.subdirs')
        figures.plot_aggregate_metrics(
            all_results, 
            self.file_reader.time_data, 
            self.subdirs['morphology'],
            self.experiment_id
        )
        
        if self.params.get('dye_uptake_parameters', {}).get('enable', False):
            figures.plot_aggregate_dye_metrics(
                all_results,
                self.subdirs['dye'],
                self.experiment_id,
                self.params
            )

        # README.txt: what every folder and file in this result holds
        try:
            output_layout.write_readme(self.results_dir, self.experiment_id)
        except Exception:
            logger.warning("Could not write README.txt", exc_info=True)

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
        save_path = output_layout.experiment_file(self.subdirs, 'flow_csv')
        shear_df.to_csv(save_path, index=False)
        figures.plot_shear_analysis_card(
            metrics, 
            output_layout.experiment_file(self.subdirs, 'flow_card')
        )
