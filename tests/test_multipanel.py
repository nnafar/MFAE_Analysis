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
import matplotlib.pyplot  # noqa: E402

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from mfae.plotting import multipanel as pmu  # noqa: E402
from mfae.config.schema import MFAConfig  # noqa: E402

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
    return {'protrusion': tmp_path / 'prot', 'dye': tmp_path / 'dye', 'actin': tmp_path / 'actin'}


PARAMS = {'experiment_parameters': {'scale_factor': 0.3},
          'dye_uptake_parameters': {'has_pulse': True},
          'plotting_parameters': {'multipanel_columns': 3}}


def test_all_six_figures_written(tmp_path):
    results = [_fake_trap(i, rupture=(i == 2)) for i in (0, 1, 2, 5, 7)]
    written = pmu.plot_trap_multipanels(results, _subdirs(tmp_path), 'EXP', PARAMS)
    names = sorted(p.name for p in written)
    assert names == sorted([
        'All_traps_protrusion_length.pdf', 'All_traps_membrane_kymographs.pdf',
        'All_traps_dye_kymographs.pdf', 'All_traps_dye_profiles_along_cell.pdf',
        'All_traps_actin_kymographs.pdf', 'All_traps_actin_zones.pdf'])
    assert all(p.stat().st_size > 0 for p in written)


def test_figures_skipped_when_channel_missing(tmp_path):
    results = [_fake_trap(i, with_dye=False, with_actin=False) for i in range(3)]
    written = pmu.plot_trap_multipanels(results, _subdirs(tmp_path), 'EXP', PARAMS)
    assert sorted(p.name for p in written) == ['All_traps_membrane_kymographs.pdf',
                                               'All_traps_protrusion_length.pdf']


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
    from mfae import pipeline

    from mfae.config import loader as config_loader
    raw = config_loader.load_layers(ROOT / 'config' / 'pipeline/template_tiff.yaml', {
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
            ('root', 'full_csv', 'protrusion', 'morphology', 'fitting', 'dye', 'actin')}
    for d in dirs.values():
        d.mkdir()
    channel = {'path': str(mmap_path), 'shape': stack.shape, 'dtype': stack.dtype}
    result = pipeline.static_parallel_worker({
        'trap_index': 0, 'params': params, 'output_dirs': dirs,
        'mmap_path': str(mmap_path), 'dtype': stack.dtype, 'shape': stack.shape,
        'rotation_angle': 0.0, 'roi_coords': (0, 50, 0, 200),
        'tuned_pipette_x': 150, 'tuned_threshold_prot': 110, 'tuned_threshold_body': 110,
        'time_data': [i * DT for i in range(n)], 'first_frame_number': 1,
        'dye_mmap': channel, 'actin_mmap': channel,
    })

    assert result['status'] in ('success', 'detection_only'), result.get('error')
    assert {'trace', 'membrane_kymo', 'actin', 'pipette_x'} <= set(result['plot_data'])
    from mfae.files import output_layout as ol
    d = ol.complete(dirs)
    assert (d['full_csv'] / 'Trap_01_protrusion_trace.csv').exists()
    assert not list(tmp_path.rglob('*filtered*'))       # filtered CSV no longer written

    # Output layout: CSVs, kymograph CSVs and movies in their own sub-folders.
    prot = dirs['protrusion']
    assert (d['movies'] / 'Trap_01_tracking.avi').stat().st_size > 0
    assert (d['membrane_kymo'] / 'Trap_01_membrane_kymograph.csv').exists()
    assert (d['actin_traces'] / 'Trap_01_actin.csv').exists()
    assert (d['actin_kymo'] / 'Trap_01_actin_kymograph.csv').exists()
    assert (d['actin_movies'] / 'Trap_01_actin_zones.avi').exists()
    assert (d['actin_movies'] / 'Trap_01_actin_cortex.avi').exists()
    assert (d['dye_traces'] / 'Trap_01_dye_uptake.csv').exists()
    assert (d['dye_kymo'] / 'Trap_01_dye_kymograph.csv').exists()
    assert (d['dye_movies'] / 'Trap_01_dye_masks.avi').exists()
    for top in (prot, dirs['dye'], dirs['actin']):   # only PDFs directly in the main folders
        assert all(f.suffix == '.pdf' for f in top.iterdir() if f.is_file())

    single_pdfs = list(tmp_path.rglob('*.pdf'))
    if save_singles:
        assert len(single_pdfs) >= 6
    else:
        assert single_pdfs == []


def test_shared_actin_channel_gives_identical_results(tmp_path):
    """Reusing the membrane frames for actin must match reading a separate copy."""
    import shutil
    from synthetic_data_generators import create_synthetic_protrusion_image
    from mfae import pipeline
    from mfae.config import loader as config_loader

    params = MFAConfig(**config_loader.load_layers(
        ROOT / 'config' / 'pipeline/template_tiff.yaml', {'paths': {'data_folder': tempfile.gettempdir()}}).data).model_dump()
    frames = [create_synthetic_protrusion_image(
        protrusion_length_px=0.5 if i < 5 else 8 + 1.2 * (i - 5))[0].astype(np.uint8) for i in range(25)]
    stack = np.stack(frames)
    mem = tmp_path / 'membrane.dat'
    fp = np.memmap(mem, dtype=stack.dtype, mode='w+', shape=stack.shape)
    fp[:] = stack
    fp.flush()
    del fp
    copy = tmp_path / 'actin_copy.dat'
    shutil.copy(mem, copy)

    def run(actin_path, out):
        dirs = {k: out / k for k in ('root', 'full_csv', 'protrusion', 'morphology',
                                     'fitting', 'dye', 'actin')}
        for d in dirs.values():
            d.mkdir(parents=True)
        chan = {'path': str(actin_path), 'shape': stack.shape, 'dtype': stack.dtype}
        pipeline.static_parallel_worker({
            'trap_index': 0, 'params': params, 'output_dirs': dirs, 'mmap_path': str(mem),
            'dtype': stack.dtype, 'shape': stack.shape, 'rotation_angle': 0.0,
            'roi_coords': (0, 50, 0, 200), 'tuned_pipette_x': 150, 'tuned_threshold_prot': 110,
            'tuned_threshold_body': 110, 'time_data': [i * DT for i in range(len(frames))],
            'first_frame_number': 1, 'dye_mmap': {'path': None}, 'actin_mmap': chan,
        })
        from mfae.files import output_layout as ol
        return (ol.complete(dirs)['actin_traces'] / 'Trap_01_actin.csv').read_text()

    assert run(mem, tmp_path / 'shared') == run(copy, tmp_path / 'separate')


def test_file_reader_reuses_membrane_files_for_actin(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mfae.files.readers import FileRead
    for t in range(1, 6):
        for c in ('C1', 'C2'):
            cv2.imwrite(str(tmp_path / f"exp_{c}_t{t:04d}.tif"), np.full((8, 8), 50, np.uint8))
    reader = FileRead(tmp_path, {'dye_uptake_parameters': {'enable': True, 'membrane_channel_pattern': 'C1',
                                                           'dye_channel_pattern': 'C2'},
                                 'actin_parameters': {'enable': True, 'channel_pattern': 'C1'}})
    reader.run()
    assert reader.actin_files == reader.tif_files and reader.actin_files is not reader.tif_files


def test_membrane_kymographs_mark_the_pulse_only_when_there_is_one():
    pulsed = _fake_trap(0)
    unpulsed = _fake_trap(1)
    unpulsed['plot_data']['pulse_time'] = None
    fig = pmu._membrane_kymographs([pulsed, unpulsed], ncols=2, scale=0.3)
    labels = [[ln.get_label() for ln in ax.get_lines()] for ax in fig.axes[:2]]
    assert 'Pulse' in labels[0]
    assert 'Pulse' not in labels[1]
    matplotlib.pyplot.close(fig)


def _superplot_df(levels):
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = []
    for t_i, (cond, level) in enumerate(levels.items()):
        for exp in range(2):
            for cell in range(8):
                rows.append({'Date': 260101 + 10 * t_i + exp, 'Treatment': 'WT', 'Cond': cond,
                             'Experiment_Folder': f"2601{t_i}{exp}_MDAMB231_WT_Chip{exp + 1}_Experiment{exp}-1100Pa-0V-0ms-frame0",
                             'E': level * rng.lognormal(0, 0.1)})
    return pd.DataFrame(rows)


def test_superplot_draws_a_bracket_for_every_significant_pair(tmp_path, monkeypatch):
    """The bracket code used to run once after the pair loop: only the last pair got one."""
    import matplotlib
    matplotlib.use('Agg')
    from mfae.bulk.plots import common
    drawn = []
    real = common._add_bracket
    monkeypatch.setattr(common, '_add_bracket',
                        lambda ax, x1, x2, y_top, label, **k: drawn.append((x1, x2, y_top)) or real(ax, x1, x2, y_top, label, **k))
    df = _superplot_df({'A': 1.0, 'B': 3.0, 'C': 9.0})
    common.render_visco_parameter_superplot(df, [('E', 'E', 'E')], tmp_path / 'sp.pdf', category_col='Cond', categories=['A', 'B', 'C'],
                               grid_shape=(1, 2), figsize=(8, 4), bracket_pairs=[(0, 1), (1, 2), (0, 2)])
    assert [(a, b) for a, b, _ in drawn] == [(0, 1), (1, 2), (0, 2)]
    assert drawn[2][2] > max(drawn[0][2], drawn[1][2])     # spanning bracket stacked above the others
    assert (tmp_path / 'sp.pdf').exists()


def test_superplot_with_one_category_does_not_crash(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from mfae.bulk.plots import common
    common.render_visco_parameter_superplot(_superplot_df({'A': 1.0}), [('E', 'E', 'E')], tmp_path / 'one.pdf',
                               category_col='Cond', categories=['A'], grid_shape=(1, 2), figsize=(8, 4))
    assert (tmp_path / 'one.pdf').exists()


def test_trap_image_export_is_off_by_default_and_follows_the_trap_list(tmp_path):
    """Developer export: written only when switched on, only for the chosen traps,
    with the cropped frames of every loaded channel (actin = membrane is not repeated)
    and the trap's setup; independent of the detection result."""
    import shutil
    from synthetic_data_generators import create_synthetic_protrusion_image
    from mfae import pipeline
    from mfae.config import loader as config_loader
    from mfae.files import output_layout as ol, trap_export

    base = MFAConfig(**config_loader.load_layers(
        ROOT / 'config' / 'pipeline/template_tiff.yaml', {'paths': {'data_folder': tempfile.gettempdir()}}).data).model_dump()
    assert base['workflow_settings']['export_trap_images'] is False
    frames = [create_synthetic_protrusion_image(protrusion_length_px=8 + i)[0].astype(np.uint16) for i in range(12)]
    stack = np.stack(frames)
    mem, dye = tmp_path / 'membrane.dat', tmp_path / 'dye.dat'
    fp = np.memmap(mem, dtype=stack.dtype, mode='w+', shape=stack.shape)
    fp[:] = stack
    fp.flush()
    del fp
    shutil.copy(mem, dye)

    def run(out, export, traps):
        params = {**base, 'workflow_settings': {**base['workflow_settings'], 'export_trap_images': export,
                                                'export_trap_images_traps': traps, 'save_tracking_movies': False}}
        dirs = {'root': out}
        out.mkdir()
        chan = {'path': str(dye), 'shape': stack.shape, 'dtype': stack.dtype}
        same = {'path': str(mem), 'shape': stack.shape, 'dtype': stack.dtype}
        pipeline.static_parallel_worker({
            'trap_index': 0, 'params': params, 'output_dirs': dirs, 'mmap_path': str(mem),
            'dtype': stack.dtype, 'shape': stack.shape, 'rotation_angle': 0.0,
            'roi_coords': (5, 45, 10, 190), 'tuned_pipette_x': 140, 'tuned_threshold_prot': 110,
            'tuned_threshold_body': 120, 'tuned_denoise_sigma_along': 2.0,
            'time_data': [i * DT for i in range(len(frames))], 'first_frame_number': 1,
            'dye_mmap': chan, 'actin_mmap': same, 'results_dir': str(out),
        })
        return ol.complete(dirs)['trap_images'] / 'Trap_01_images.npz'

    assert not run(tmp_path / 'off', False, []).exists()
    assert not run(tmp_path / 'other', True, [2, 3]).exists()
    f = run(tmp_path / 'on', True, [1])
    arrays, meta = trap_export.read(f)
    assert set(arrays) == {'membrane', 'dye', 'time_s'}                 # actin = membrane: not repeated
    assert arrays['membrane'].shape == (12, 40, 180) and arrays['membrane'].dtype == np.uint16
    assert np.array_equal(arrays['membrane'][3], stack[3, 5:45, 10:190])
    assert np.allclose(arrays['time_s'], [i * DT for i in range(12)])
    assert (meta['trap_number'], meta['pip'], meta['thr_prot'], meta['thr_body'], meta['smooth']) == (1, 140, 110, 120, 2.0)
    assert meta['roi_coords'] == [5, 45, 10, 190] and 'image_processing' in meta
    assert run(tmp_path / 'all', True, []).exists()                     # [] = every trap
