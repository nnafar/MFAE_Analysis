# -*- coding: utf-8 -*-
"""Output folder layout and the readers' fallback to the old flat layout."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mfae.files import output_layout as ol  # noqa: E402


def test_build_layout(tmp_path):
    d = ol.build(tmp_path)
    assert d['protrusion'] == tmp_path / '1_Protrusion_and_Shape' / 'Figures'
    assert d['full_csv'] == tmp_path / '1_Protrusion_and_Shape' / 'Data' / 'Traces'
    assert d['membrane_kymo'] == tmp_path / '1_Protrusion_and_Shape' / 'Data' / 'Kymographs'
    assert d['movies'] == tmp_path / '1_Protrusion_and_Shape' / 'Movies'
    assert d['morphology'] == tmp_path / '1_Protrusion_and_Shape' / 'Figures'
    assert d['dye'] == tmp_path / '2_Dye_Uptake' / 'Figures'
    assert d['dye_traces'] == tmp_path / '2_Dye_Uptake' / 'Data' / 'Traces'
    assert d['dye_kymo'] == tmp_path / '2_Dye_Uptake' / 'Data' / 'Kymographs'
    assert d['dye_movies'] == tmp_path / '2_Dye_Uptake' / 'Movies'
    assert d['actin_traces'] == tmp_path / '3_Actin' / 'Data' / 'Traces'
    assert d['actin_kymo'] == tmp_path / '3_Actin' / 'Data' / 'Kymographs'
    assert d['actin_movies'] == tmp_path / '3_Actin' / 'Movies'
    assert d['fitting'] == tmp_path / '4_Mechanics'
    assert d['overview'] == tmp_path / '0_Overview'
    assert d['run_info'] == tmp_path / 'Run_Info'


def test_complete_fills_missing_keys_from_parents(tmp_path):
    d = ol.complete({'root': tmp_path, 'tracking_visuals': tmp_path / 'tv', 'dye': tmp_path / 'dye'})
    assert d['protrusion'] == tmp_path / 'tv'
    assert d['movies'] == tmp_path / 'tv' / 'Movies'
    assert d['dye_movies'] == tmp_path / 'dye' / 'Movies'
    assert d['actin'] == tmp_path / '3_Actin' / 'Figures'
    # a full layout is left as it is
    assert ol.complete(ol.build(tmp_path)) == ol.build(tmp_path)


def test_file_names():
    assert ol.trap_file('dye_csv', 3) == 'Trap_03_dye_uptake.csv'
    assert ol.trap_file('protrusion_csv', 12) == 'Trap_12_protrusion_trace.csv'
    assert 'Trap_12_protrusion_and_shape.csv' in ol.trap_file_candidates('protrusion_csv', 12)
    assert ol.trap_file_candidates('dye_csv', 3)[1:] == ['Trap_03_Uptake_Data.csv', 'Trap_3_Uptake_Data.csv']


def test_readers_fall_back_to_old_layouts(tmp_path):
    flat = tmp_path / 'Full protrusion detection'
    flat.mkdir()
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == flat
    older = tmp_path / 'Protrusion Length Detection' / 'Full protrusion detection'
    older.mkdir(parents=True)
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == older
    flat_data = tmp_path / '1_Protrusion_and_Shape' / 'Data'           # previous layout
    flat_data.mkdir(parents=True)
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == flat_data
    new = flat_data / 'Traces'
    new.mkdir()
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == new
    assert ol.first_existing(ol.session_cache_candidates(tmp_path)) == tmp_path / 'Run_Info' / 'session_cache.json'


def test_session_cache_with_other_threshold_scale_is_not_reused(tmp_path):
    import json
    import types
    from mfae import pipeline

    def fake(enhance, full_scale=40.0):
        obj = types.SimpleNamespace()
        obj.params = {'image_processing': {'membrane_enhancement': enhance,
                                           'membrane_snr_full_scale': full_scale},
                      'guv_settings': {'enable': False}}
        obj.results_dir = tmp_path
        obj.file_reader = types.SimpleNamespace(tif_files=[])
        obj._threshold_scale = types.MethodType(pipeline.MFAAnalysis._threshold_scale, obj)
        return obj

    load = pipeline.MFAAnalysis._load_session_cache
    cache = {'trap_configs': [], 'source_files': []}          # old cache: no scale recorded
    (tmp_path / 'session_cache.json').write_text(json.dumps(cache))
    assert load(fake(enhance=True)) is None                   # CLAHE thresholds, SNR pipeline
    assert load(fake(enhance=False)) is not None              # still fine for the CLAHE pipeline

    cache['threshold_scale'] = fake(True)._threshold_scale()
    (tmp_path / 'session_cache.json').write_text(json.dumps(cache))
    assert load(fake(enhance=True)) is not None
    assert load(fake(enhance=True, full_scale=20.0)) is None  # scale changed since the setup
    # a cache from an older version (experiment folder) is moved into Run_Info/
    assert not (tmp_path / 'session_cache.json').exists()
    assert (tmp_path / 'Run_Info' / 'session_cache.json').exists()


def test_trap_summary_row_counts_frames_from_one():
    from mfae import pipeline
    det = {'cell_segments': [(0, 9, 'cell change'), (12, 29, None)], 'selected_cell': 2,
           'cell_at_pulse': True, 'entry_confirmed': True, 'entry_frame_index': 12,
           'exit_frame_index': 29, 'exit_reason': None, 'rupture_detected': False,
           'protrusion_lengths_um': [float('nan')] * 12 + [float(i) for i in range(18)]}
    row = pipeline.trap_summary_row(4, det, [0.5 * i for i in range(30)], 20, 'analysed', dye_done=True)
    assert row['Trap'] == 5 and row['Cells_seen_in_trap'] == 2 and row['Analysed_cell'] == 2
    assert (row['First_frame'], row['Last_frame'], row['Pulse_frame']) == (13, 30, 21)
    assert row['First_time_s'] == 6.0 and row['Why_tracking_ended'] == 'end of recording'
    assert row['Protrusion_at_pulse_um'] == 8.0 and row['Max_protrusion_um'] == 17.0
    assert row['Dye_analysed'] and not row['Actin_analysed']


def test_readme_describes_the_files_present(tmp_path):
    d = ol.build(tmp_path)
    for key, name in (('dye_traces', 'Trap_01_dye_uptake.csv'), ('dye_traces', 'Trap_02_dye_uptake.csv'),
                      ('overview', 'trap_summary.csv'), ('dye', 'All_traps_dye_uptake.pdf')):
        d[key].mkdir(parents=True, exist_ok=True)
        (d[key] / name).write_text('x')
    text = ol.write_readme(tmp_path, 'EXP').read_text()
    assert 'Trap_XX_dye_uptake.csv' in text and 'Trap_01_dye_uptake.csv' not in text
    assert 'mean dye intensity in body' in text
    assert '2_Dye_Uptake/' in text and '3_Actin/' not in text      # only folders that exist


def test_bulk_layout_and_readme(tmp_path):
    from mfae.bulk import layout as bl
    out = bl.build(tmp_path / 'run', treatments=['WT', 'LifeAct'], pulses=[(100, '1ms'), (100, '5ms')])
    assert out['asp_by_treatment']['LifeAct'] == tmp_path / 'run' / '1_Aspiration_Mechanics' / 'LifeAct'
    assert out['pre_by_pulse']['ASP'] == tmp_path / 'run' / '2_Before_Pulse' / 'Aspiration'
    assert out['at_by_pulse']['1ms'] == tmp_path / 'run' / '3_At_Pulse' / '100V_1ms'
    assert out['whole_all'] == tmp_path / 'run' / '4_Whole_Trace' / 'All_conditions'
    (out['at_by_pulse']['1ms'] / 'Thesis_Fate_Pie_WT_1ms.pdf').write_text('x')
    bl.table(out, 'results').write_text('x')
    text = bl.write_readme(out['root']).read_text()
    assert 'mechanics_results_all_traps.csv' in text
    assert '100V_1ms/Thesis_Fate_Pie_WT_1ms.pdf' in text and 'intact vs ruptured' in text
    # folders nothing was written to are removed and not listed
    assert '4_Whole_Trace' not in text and not (tmp_path / 'run' / '4_Whole_Trace').exists()
    assert not (tmp_path / 'run' / '3_At_Pulse' / '100V_5ms').exists()


def test_thesis_routing_follows_the_conditions_present(tmp_path):
    from mfae.bulk.plots import figures as tp
    m = {'ASP': tmp_path / 'a', '5ms': tmp_path / 'b', '15ms': tmp_path / 'c', '1ms': tmp_path / 'd'}
    assert tp._resolve_output_dir('EP-pre 100V 15ms (intact)', tmp_path, m) == tmp_path / 'c'
    assert tp._resolve_output_dir('EP-pre 100V 5ms', tmp_path, m) == tmp_path / 'b'
    assert tp._resolve_output_dir('EP-pre 100V 1ms', tmp_path, m) == tmp_path / 'd'
    assert tp._resolve_output_dir('ASP', tmp_path, m) == tmp_path / 'a'
    assert tp._resolve_output_dir('EP-pre 100V 2ms', tmp_path, m) == tmp_path   # unknown -> combined
    t = {'WT': tmp_path / 'wt', 'LifeAct': tmp_path / 'la'}
    assert tp._resolve_treatment_dir('MDAMB231_LifeAct_1100Pa_ASP', tmp_path, t) == tmp_path / 'la'
    assert tp._resolve_treatment_dir('MDAMB231_WT_1100Pa_ASP', tmp_path, t) == tmp_path / 'wt'


def test_retune_replaces_only_the_chosen_traps(monkeypatch):
    from mfae import pipeline
    obj = pipeline.MFAAnalysis.__new__(pipeline.MFAAnalysis)
    obj.retune_traps = [7, 30]                       # 30 is not in the saved setup
    obj.stopped_by_user = False
    saved = []
    asked = []

    def fake_collect(indices=None, start_pip=None):
        asked.append((indices, start_pip))
        return 'confirm', [{'trap_index': 6, 'pip': 101, 'thr_prot': 60, 'thr_body': 90,
                            'smooth': 2.0, 'snr_scale': 300.0}]

    obj._collect_interactive_parameters = fake_collect
    obj._save_session_cache = lambda cfgs: saved.append([dict(c) for c in cfgs])
    configs = [{'trap_index': i, 'pip': 100, 'thr_prot': 120, 'thr_body': 150,
                'smooth': 1.0, 'snr_scale': None} for i in (2, 6, 9)]
    assert obj._retune_cached_traps(configs)
    assert asked == [([6], {6: 100})]                # starts at the saved entrance
    assert configs[1]['thr_prot'] == 60 and configs[1]['snr_scale'] == 300.0
    assert configs[0]['thr_prot'] == 120 and configs[2]['thr_prot'] == 120
    assert saved and saved[0][1]['thr_prot'] == 60


def test_retune_stop_aborts():
    from mfae import pipeline
    obj = pipeline.MFAAnalysis.__new__(pipeline.MFAAnalysis)
    obj.retune_traps, obj.stopped_by_user = [3], False
    obj._collect_interactive_parameters = lambda **k: ('stop', [])
    obj._save_session_cache = lambda cfgs: pytest.fail("must not save")
    assert not obj._retune_cached_traps([{'trap_index': 2, 'pip': 50}])
    assert obj.stopped_by_user


def test_thresholds_only_redoes_every_saved_trap_and_keeps_entrances():
    from mfae import pipeline
    obj = pipeline.MFAAnalysis.__new__(pipeline.MFAAnalysis)
    obj.retune_traps, obj.thresholds_only, obj.stopped_by_user = [], True, False
    asked = []

    def fake_collect(indices=None, start_pip=None, thresholds_only=False):
        asked.append((indices, start_pip, thresholds_only))
        return 'confirm', [{'trap_index': i, 'pip': start_pip[i], 'thr_prot': 50, 'thr_body': 80,
                            'smooth': 1.0, 'snr_scale': None} for i in indices]

    obj._collect_interactive_parameters = fake_collect
    obj._save_session_cache = lambda cfgs: None
    configs = [{'trap_index': i, 'pip': 100 + i, 'thr_prot': 120, 'thr_body': 150,
                'smooth': 1.0, 'snr_scale': None} for i in (2, 6, 9)]
    assert obj._retune_cached_traps(configs)
    assert asked == [([2, 6, 9], {2: 102, 6: 106, 9: 109}, True)]
    assert [c['thr_prot'] for c in configs] == [50, 50, 50]
    assert [c['pip'] for c in configs] == [102, 106, 109]          # entrances kept


def test_flat_data_files_of_the_previous_layout_are_tidied(tmp_path):
    data = tmp_path / '1_Protrusion_and_Shape' / 'Data'
    (data / 'Traces').mkdir(parents=True)
    for name in ('Trap_01_protrusion_and_shape.csv', 'Trap_01_membrane_kymograph.csv', 'notes.csv',
                 'Trap_03_protrusion_and_shape_R.csv'):
        (data / name).write_text(name)
    dye = tmp_path / '2_Dye_Uptake' / 'Data'
    dye.mkdir(parents=True)
    (dye / 'Trap_03_dye_uptake.csv').write_text('x')
    assert ol.remove_flat_data_files(tmp_path) == 4
    assert sorted(p.name for p in data.iterdir()) == ['Traces', 'notes.csv']      # own files kept
    tagged = data / 'Traces' / 'Trap_03_protrusion_trace_R.csv'                  # tag kept, moved
    assert tagged.read_text() == 'Trap_03_protrusion_and_shape_R.csv'
    # the analysis then writes trap 3 into the tagged file, trap 1 under the plain name
    assert ol.protrusion_csv_name(data / 'Traces', 3) == 'Trap_03_protrusion_trace_R.csv'
    assert ol.protrusion_csv_name(data / 'Traces', 1) == 'Trap_01_protrusion_trace.csv'


def test_readme_lists_the_data_subfolders(tmp_path):
    d = ol.build(tmp_path)
    for key, name in (('full_csv', 'Trap_01_protrusion_trace.csv'),
                      ('membrane_kymo', 'Trap_01_membrane_kymograph.csv')):
        d[key].mkdir(parents=True, exist_ok=True)
        (d[key] / name).write_text('x')
    text = ol.write_readme(tmp_path, 'X').read_text()
    assert 'Data/Traces/' in text and 'Data/Kymographs/' in text
    assert 'Trap_XX_protrusion_trace.csv' in text and 'protrusion length' in text



def _setup_with_script(monkeypatch, script):
    """MFAAnalysis whose per-trap windows answer from `script` (trap -> list of signals)."""
    from mfae import pipeline, utils
    obj = pipeline.MFAAnalysis.__new__(pipeline.MFAAnalysis)
    obj.params, obj.skipped = {'workflow_settings': {}}, []
    obj.file_reader = types.SimpleNamespace(tif_files=[0] * 10)
    obj.loader = types.SimpleNamespace(get_frame=lambda i: 'frame', clear_cache=lambda: None)
    obj.cropper = types.SimpleNamespace(selected_traps=[0, 3, 5], pipette_coords=[50, 0],
                                        process_frame=lambda img, t: f'roi{t}')
    calls = []

    class FakeDet:
        def __init__(self, rois, pip, params):
            self.trap = int(rois[0][3:]); self.pip = pip

        def run_detection(self, skip_pipette=False, initial_thresholds=None):
            calls.append((self.trap, initial_thresholds))
            sig, thr = script[self.trap].pop(0)
            if sig != 'confirm':
                return {'signal': sig}
            return {'signal': 'confirm', 'pipette_start_x_used': self.pip[0],
                    'threshold_prot': thr, 'threshold_body': thr + 10}

    monkeypatch.setattr(pipeline, 'LineDetectionMFA', FakeDet)
    return obj, calls


def test_setup_back_returns_to_the_previous_trap_at_its_values(monkeypatch):
    obj, calls = _setup_with_script(monkeypatch, {
        0: [('confirm', 70), ('confirm', 75)],
        3: [('back', None), ('confirm', 90)],
        5: [('confirm', 100)]})
    signal, configs = obj._collect_interactive_parameters()
    assert signal == 'confirm'
    assert calls == [(0, None), (3, None), (0, (70, 80)), (3, None), (5, None)]
    assert [(c['trap_index'], c['thr_prot']) for c in configs] == [(0, 75), (3, 90), (5, 100)]


def test_setup_restart_goes_to_the_first_trap_and_keeps_the_boxes(monkeypatch):
    obj, calls = _setup_with_script(monkeypatch, {
        0: [('confirm', 70), ('confirm', 71)],
        3: [('confirm', 90), ('confirm', 91)],
        5: [('restart', None), ('confirm', 100)]})
    signal, configs = obj._collect_interactive_parameters()
    assert signal == 'confirm'                      # not 'restart': the trap boxes stay
    assert [c[0] for c in calls] == [0, 3, 5, 0, 3, 5]
    assert calls[3] == (0, (70, 80))               # earlier choices pre-filled
    assert [(c['trap_index'], c['thr_prot']) for c in configs] == [(0, 71), (3, 91), (5, 100)]


def test_threshold_windows_show_the_cell_at_the_pulse_as_the_analysis_sees_it(monkeypatch):
    """Exp 2 trap 11: the pulsed cell is faint and arrives after a bright one left. The
    windows open just before the pulse (not at selection_frame_fraction) and show the
    median of membrane_temporal_window frames, as the analysis thresholds it."""
    import numpy as np
    from mfae import pipeline
    obj = pipeline.MFAAnalysis.__new__(pipeline.MFAAnalysis)
    obj.params = {'workflow_settings': {'selection_frame_fraction': 0.4},
                  'image_processing': {'membrane_enhancement': True, 'membrane_temporal_window': 5},
                  'dye_uptake_parameters': {'has_pulse': True, 'pulse_frame': 304}}
    obj.skipped = []
    obj.file_reader = types.SimpleNamespace(tif_files=[f"x_t{i:04d}.tif" for i in range(1, 506)])
    loaded = []

    def get_frame(i):
        loaded.append(i)
        return np.full((4, 4), float(i))
    obj.loader = types.SimpleNamespace(get_frame=get_frame, clear_cache=lambda: None)
    obj.cropper = types.SimpleNamespace(selected_traps=[10], pipette_coords=[2, 0],
                                        process_frame=lambda img, t: img)
    seen = []

    class FakeDet:
        def __init__(self, rois, pip, params):
            seen.append(rois[0])

        def run_detection(self, skip_pipette=False, initial_thresholds=None):
            return {'signal': 'confirm', 'pipette_start_x_used': 2, 'threshold_prot': 50, 'threshold_body': 40}

    monkeypatch.setattr(pipeline, 'LineDetectionMFA', FakeDet)
    assert obj._collect_interactive_parameters()[0] == 'confirm'
    # pulse_frame 304 with files from t0001 -> index 303; 3 before -> 300; median of 298-302
    assert sorted(loaded) == [298, 299, 300, 301, 302]
    assert np.all(seen[0] == 300.0)
    obj.params['dye_uptake_parameters'] = {'has_pulse': False}                # aspiration only
    assert obj._threshold_frame_index(505) == int(505 * 0.4)
