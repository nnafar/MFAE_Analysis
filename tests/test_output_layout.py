# -*- coding: utf-8 -*-
"""Output folder layout and the readers' fallback to the old flat layout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mfae.files import output_layout as ol  # noqa: E402


def test_build_layout(tmp_path):
    d = ol.build(tmp_path)
    assert d['protrusion'] == tmp_path / '1_Protrusion_and_Shape' / 'Figures'
    assert d['full_csv'] == tmp_path / '1_Protrusion_and_Shape' / 'Data'
    assert d['membrane_kymo'] == tmp_path / '1_Protrusion_and_Shape' / 'Data'
    assert d['movies'] == tmp_path / '1_Protrusion_and_Shape' / 'Movies'
    assert d['morphology'] == tmp_path / '1_Protrusion_and_Shape' / 'Figures'
    assert d['dye'] == tmp_path / '2_Dye_Uptake' / 'Figures'
    assert d['dye_traces'] == tmp_path / '2_Dye_Uptake' / 'Data'
    assert d['dye_movies'] == tmp_path / '2_Dye_Uptake' / 'Movies'
    assert d['actin_traces'] == tmp_path / '3_Actin' / 'Data'
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
    assert ol.trap_file('protrusion_csv', 12) == 'Trap_12_protrusion_and_shape.csv'
    assert ol.trap_file_candidates('dye_csv', 3)[1:] == ['Trap_03_Uptake_Data.csv', 'Trap_3_Uptake_Data.csv']


def test_readers_fall_back_to_old_layouts(tmp_path):
    flat = tmp_path / 'Full protrusion detection'
    flat.mkdir()
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == flat
    older = tmp_path / 'Protrusion Length Detection' / 'Full protrusion detection'
    older.mkdir(parents=True)
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == older
    new = tmp_path / '1_Protrusion_and_Shape' / 'Data'
    new.mkdir(parents=True)
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
