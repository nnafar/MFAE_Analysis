# -*- coding: utf-8 -*-
"""
Tests for the multipanel (all traps on one page) figures and for the worker
output they are built from.
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import Plotting_Multipanel as pmu  # noqa: E402
from config_schema import MFAConfig  # noqa: E402

N_FRAMES = 40
DT = 0.5


def _fake_trap(trap_index: int, with_dye=True, with_actin=True, rupture=False):
    rng = np.random.default_rng(trap_index)
    t = [i * DT for i in range(N_FRAMES)]
    entry = 4
    length = np.full(N_FRAMES, np.nan)
    length[entry:] = np.linspace(1, 12, N_FRAMES - entry)
    prof_t = t[entry:]
    profiles = [np.linspace(5, 1, 50) + rng.normal(0, 0.1, 50) for _ in prof_t]
    result = {
        'trap_index': trap_index,
        'status': 'detection_only',
        'dye_data': {'time_s': prof_t, 'spatial_profiles': profiles} if with_dye else None,
        'plot_data': {
            'pipette_x': 60,
            'pulse_time': t[10],
            'rupture_time': t[25] if rupture else None,
            'trace': {'time': t, 'protrusions': list(length),
                      'intensities': list(10 + rng.normal(0, 0.2, N_FRAMES))},
            'membrane_kymo': {'matrix': rng.integers(0, 255, (N_FRAMES, 80)).astype(np.uint8),
                              'protrusions_px': list(np.nan_to_num(length) / 0.3)},
        },
    }
    if with_actin:
        actin = {'time_s': prof_t, 'spatial_profiles': profiles}
        for zone in ('tip', 'base', 'perinuclear', 'distal'):
            actin[f'zone_{zone}_norm'] = np.ones(len(prof_t))
            actin[f'zone_{zone}_didt'] = np.r_[np.nan, np.zeros(len(prof_t) - 1)]
        result['plot_data']['actin'] = actin
    return result


def _subdirs(tmp_path):
    return {'tracking_visuals': tmp_path / 'tv', 'dye': tmp_path / 'dye', 'actin': tmp_path / 'actin'}


PARAMS = {'experiment_parameters': {'scale_factor': 0.3},
          'dye_uptake_parameters': {'has_pulse': True},
          'plotting_parameters': {'multipanel_columns': 3}}


def test_all_six_figures_written(tmp_path):
    results = [_fake_trap(i, rupture=(i == 2)) for i in (0, 1, 2, 5, 7)]
    written = pmu.plot_trap_multipanels(results, _subdirs(tmp_path), 'EXP', PARAMS)
    names = sorted(p.name for p in written)
    assert names == sorted([
        'EXP_Detection_Traces.pdf', 'EXP_Membrane_Kymographs.pdf',
        'EXP_Uptake_Kymographs.pdf', 'EXP_Diffusion_Profiles.pdf',
        'EXP_Actin_Kymographs.pdf', 'EXP_Actin_Zones.pdf'])
    assert all(p.stat().st_size > 0 for p in written)


def test_figures_skipped_when_channel_missing(tmp_path):
    results = [_fake_trap(i, with_dye=False, with_actin=False) for i in range(3)]
    written = pmu.plot_trap_multipanels(results, _subdirs(tmp_path), 'EXP', PARAMS)
    assert sorted(p.name for p in written) == ['EXP_Detection_Traces.pdf',
                                               'EXP_Membrane_Kymographs.pdf']


def test_failed_traps_are_ignored(tmp_path):
    results = [_fake_trap(0), {'trap_index': 1, 'status': 'error', 'error': 'boom'},
               {'trap_index': 2, 'status': 'no_detection'}]
    assert len(pmu.plot_trap_multipanels(results, _subdirs(tmp_path), 'EXP', PARAMS)) == 6


def test_nothing_to_plot(tmp_path):
    assert pmu.plot_trap_multipanels([{'trap_index': 0, 'status': 'error'}],
                                     _subdirs(tmp_path), 'EXP', PARAMS) == []


# -----------------------------------------------------------------------------
# Worker end-to-end on a synthetic image stack
# -----------------------------------------------------------------------------

@pytest.mark.parametrize('save_singles', [False, True])
def test_worker_outputs(tmp_path, save_singles):
    from synthetic_data_generators import create_synthetic_protrusion_image
    import MFA_analysis as mfa

    import config_loader
    raw = config_loader.load_layers(ROOT / 'config.yaml', {
        'paths': {'data_folder': tempfile.gettempdir()},
        'plotting_parameters': {'save_individual_trap_plots': save_singles},
    }).data
    params = MFAConfig(**raw).model_dump()

    n = 30
    frames = []
    for i in range(n):
        length = 0.5 if i < 5 else min(45.0, 8 + 1.2 * (i - 5))
        img, _ = create_synthetic_protrusion_image(protrusion_length_px=length)
        frames.append(img.astype(np.uint8))
    stack = np.stack(frames)
    mmap_path = tmp_path / 'stack.dat'
    fp = np.memmap(mmap_path, dtype=stack.dtype, mode='w+', shape=stack.shape)
    fp[:] = stack
    fp.flush()
    del fp

    dirs = {k: tmp_path / k for k in
            ('root', 'full_csv', 'tracking_visuals', 'morphology', 'fitting', 'dye', 'actin')}
    for d in dirs.values():
        d.mkdir()
    channel = {'path': str(mmap_path), 'shape': stack.shape, 'dtype': stack.dtype}
    result = mfa.static_parallel_worker({
        'trap_index': 0, 'params': params, 'output_dirs': dirs,
        'mmap_path': str(mmap_path), 'dtype': stack.dtype, 'shape': stack.shape,
        'rotation_angle': 0.0, 'roi_coords': (0, 50, 0, 200),
        'tuned_pipette_x': 150, 'tuned_threshold_prot': 110, 'tuned_threshold_body': 110,
        'time_data': [i * DT for i in range(n)], 'first_frame_number': 1,
        'dye_mmap': channel, 'actin_mmap': channel,
    })

    assert result['status'] in ('success', 'detection_only'), result.get('error')
    assert {'trace', 'membrane_kymo', 'actin', 'pipette_x'} <= set(result['plot_data'])
    assert (dirs['full_csv'] / 'trap_01_detection_full.csv').exists()
    assert not list(tmp_path.rglob('*filtered*'))       # filtered CSV no longer written

    single_pdfs = list(tmp_path.rglob('*.pdf'))
    if save_singles:
        assert len(single_pdfs) >= 6
    else:
        assert single_pdfs == []
