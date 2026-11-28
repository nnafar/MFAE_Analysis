# -*- coding: utf-8 -*-
"""Cell tags: Run_Info/cell_annotations.csv, the --review mode and the bulk analysis reading them."""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mfae.files import annotations, output_layout  # noqa: E402

EXP = "261119_MDAMB231_LifeAct_Chip1_Experiment5-1100Pa-50V-1ms-frame300"


def _summary(first3=265):
    return [
        {'Trap': 1, 'Status': 'analysed', 'Analysed_cell': 1, 'First_frame': 67, 'Last_frame': 600,
         'Cells_seen_in_trap': 1, 'Rupture_detected': False, 'Cell_present_at_pulse': True},
        {'Trap': 3, 'Status': 'analysed', 'Analysed_cell': 2, 'First_frame': first3, 'Last_frame': 314,
         'Cells_seen_in_trap': 3, 'Rupture_detected': True, 'Rupture_frame': 302,
         'Rupture_reason': 'Intensity Step', 'Cell_present_at_pulse': True},
        {'Trap': 6, 'Status': 'entry not confirmed (check movie)'},
    ]


def test_table_is_prefilled_and_keeps_your_tags(tmp_path):
    exp = tmp_path / EXP
    annotations.update_from_summary(exp, _summary())
    rows = {int(r['Trap']): r for r in annotations.read(exp)}
    assert list(rows) == [1, 3, 6]
    assert (rows[3]['Analysed_cell'], rows[3]['First_frame'], rows[3]['Last_frame']) == ('2', '265', '314')
    assert 'rupture? frame 302 (Intensity Step)' in rows[3]['Pipeline_hint'] and '3 cells' in rows[3]['Pipeline_hint']
    assert rows[6]['Pipeline_hint'] == 'entry not confirmed (check movie)'
    assert all(r['Tag'] == '' for r in rows.values())                 # the pipeline never tags

    annotations.set_tag(exp, 3, 'r', note=None)
    annotations.set_tag(exp, 6, 'MASK')
    rows = {int(r['Trap']): r for r in annotations.read(exp)}
    rows[1]['Note'] = 'looked fine'
    rows[1]['My column'] = 'x'                                          # a column you added
    annotations.write(exp, rows.values())

    annotations.update_from_summary(exp, _summary())                   # re-analysis
    rows = {int(r['Trap']): r for r in annotations.read(exp)}
    assert (rows[3]['Tag'], rows[6]['Tag'], rows[1]['Note'], rows[1]['My column']) == ('R', 'MASK', 'looked fine', 'x')
    assert rows[3]['Check'] == ''


def test_a_tag_on_a_cell_that_is_no_longer_analysed_is_flagged(tmp_path, caplog):
    exp = tmp_path / EXP
    annotations.update_from_summary(exp, _summary())
    annotations.set_tag(exp, 3, 'R')
    with caplog.at_level('WARNING'):
        annotations.update_from_summary(exp, _summary(first3=400))       # another cell now
    row = annotations.tags(exp)[3]
    assert row['Tag'] == 'R' and row['Check'].startswith('CHECK: analysed cell changed')
    assert 'frames 265-314' in row['Check'] and 'changed since it was tagged' in caplog.text
    annotations.set_tag(exp, 3, 'R')                                   # confirmed by reviewing again
    assert annotations.tags(exp)[3]['Check'] == ''


def test_file_name_tags_are_copied_in_once(tmp_path):
    exp = tmp_path / EXP
    traces = output_layout.build(exp)['full_csv']
    traces.mkdir(parents=True)
    (traces / 'Trap_03_protrusion_trace_DOA.csv').write_text("Frame\n0\n")
    annotations.update_from_summary(exp, _summary())
    row = annotations.tags(exp)[3]
    assert row['Tag'] == 'DOA' and 'file name' in row['Note']
    annotations.set_tag(exp, 3, 'OK')                                   # the table wins from now on
    annotations.update_from_summary(exp, _summary())
    assert annotations.tags(exp)[3]['Tag'] == 'OK'


def test_excel_saved_tables_are_read(tmp_path):
    """Excel (European settings) saves with ';' and a BOM; tags in any case, with or without '_'."""
    exp = tmp_path / EXP
    p = annotations.path_for(exp)
    p.parent.mkdir(parents=True)
    p.write_bytes("﻿Trap;Analysed_cell;First_frame;Last_frame;Tag;Note\n1;1;67;600;_r0;\n2;;;;irregular;odd\n"
                  .encode("utf-8"))
    t = annotations.tags(exp)
    assert annotations.normalise_tag(t[1]['Tag']) == 'R0' and annotations.normalise_tag(t[2]['Tag']) == 'IRREGULAR'
    assert annotations.normalise_tag('nonsense') is None and annotations.normalise_tag('') == ''


def test_table_open_in_excel_is_reported_not_lost(tmp_path, monkeypatch, caplog):
    exp = tmp_path / EXP
    annotations.update_from_summary(exp, _summary())
    annotations.set_tag(exp, 1, 'R')

    def locked(*a, **k):
        raise PermissionError("locked")
    monkeypatch.setattr(annotations, "write", locked)
    with caplog.at_level('WARNING'):
        assert annotations.update_from_summary(exp, _summary()) is None
    assert 'open in another program' in caplog.text
    monkeypatch.undo()
    assert annotations.tags(exp)[1]['Tag'] == 'R'


# --- review --------------------------------------------------------------------
def _movie(path: Path, n=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (120, 80))
    for k in range(n):
        w.write(np.full((80, 120, 3), 20 * k, np.uint8))
    w.release()


def _experiment(tmp_path, traps=(1, 3, 6)):
    exp = tmp_path / "Output" / "Symmetric_Efield" / EXP
    movies = output_layout.complete(output_layout.build(exp))['movies']
    for t in traps:
        _movie(movies / f"Trap_{t:02d}_tracking.avi")
    annotations.update_from_summary(exp, _summary())
    return exp


def test_review_keys_tag_and_move(tmp_path):
    from mfae import review
    exp = _experiment(tmp_path)
    rv = review.Review(review.build_items([exp]))
    assert [it.trap for it in rv.items] == [1, 3, 6]
    assert rv.handle_key(ord('o')) == 'next' and rv.item.trap == 3   # tag saves and advances
    assert rv.handle_key(ord('R')) == 'next' and rv.item.trap == 6
    assert rv.handle_key(ord('b')) == 'next' and rv.item.trap == 3   # back
    assert rv.handle_key(ord('c')) == '' and rv.item.trap == 3       # clear, stay
    assert rv.handle_key(13) == 'next' and rv.item.trap == 6         # Enter: next, unchanged
    assert rv.handle_key(ord('m')) == 'quit' and rv.done             # past the last trap
    t = annotations.tags(exp)
    assert (t[1]['Tag'], t[3]['Tag'], t[6]['Tag']) == ('OK', '', 'MASK')
    assert review.first_untagged(review.build_items([exp])) == 1     # trap 3 is untagged again


def test_review_window_runs_headless(tmp_path, monkeypatch):
    """The real player loop with scripted keys: pause, step, speed, tag, quit."""
    from mfae import review
    exp = _experiment(tmp_path)
    keys = iter([-1, ord(' '), ord('.'), ord(','), ord(']'), ord('['), ord('d'), -1, ord('q')])
    shown = []
    monkeypatch.setattr(review.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(review.cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(review.cv2, "waitKey", lambda ms: next(keys, ord('q')))
    monkeypatch.setattr(review.cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(review.cv2, "destroyAllWindows", lambda: None)
    assert review.run(review.find_experiments(tmp_path / "Output", "Experiment5")) == 0
    assert shown and shown[0][0] == 70 + 80 + 24                       # banner + movie + help
    assert annotations.tags(exp)[1]['Tag'] == 'DOA'


def test_review_option_finds_experiments_without_config_or_data(tmp_path, monkeypatch):
    from mfae import cli, review
    _experiment(tmp_path)
    seen = {}
    monkeypatch.setattr(review, "run", lambda exps, traps=None: seen.update(exps=exps, traps=traps) or 0)
    assert cli.main(["--review", "--only", "LifeAct_Chip1_Experiment5", "--traps", "3,6",
                     "--output-dir", str(tmp_path / "Output")]) == 0
    assert [e.name for e in seen['exps']] == [EXP] and seen['traps'] == [3, 6]
    assert cli.main(["--review", "--only", "Experiment99", "--output-dir", str(tmp_path / "Output")]) == 1


# --- bulk analysis ---------------------------------------------------------------
def test_bulk_analysis_reads_tags_from_the_table_first(tmp_path, caplog):
    from mfae.bulk.loader import BulkDataLoader
    exp = tmp_path / "Output" / "Symmetric_Efield" / EXP
    traces = output_layout.build(exp)['full_csv']
    traces.mkdir(parents=True)
    csv = "Frame,Time_s,Protrusion_Length_um\n0,0,1\n1,3,2\n"
    (traces / "Trap_01_protrusion_trace.csv").write_text(csv)        # table: R0 -> excluded
    (traces / "Trap_02_protrusion_trace_R.csv").write_text(csv)      # table empty -> file name R
    (traces / "Trap_03_protrusion_trace_DOA.csv").write_text(csv)    # table R overrides the file name
    (traces / "Trap_04_protrusion_trace.csv").write_text(csv)        # unknown tag -> detection failure
    (traces / "Trap_05_protrusion_trace.csv").write_text(csv)        # OK -> analysed intact
    annotations.write(exp, [{'Trap': 1, 'Tag': 'R0'}, {'Trap': 2, 'Tag': ''}, {'Trap': 3, 'Tag': 'r'},
                            {'Trap': 4, 'Tag': 'banana'}, {'Trap': 5, 'Tag': 'OK', 'Check': 'CHECK: x'}])
    with caplog.at_level('WARNING'):
        loader = BulkDataLoader(str(exp.parent))
        traps = {t.trap_id: t.fate_status for _, ts in loader.iter_groups() for t in ts}
    assert traps == {2: 'ruptured_post', 3: 'ruptured_post', 5: 'intact'}
    counts = loader.attrition_records[0].bucket_counts
    assert counts['ruptured_pre_pulse'] == 1 and counts['detection_failure'] == 1
    assert "unknown tag 'banana'" in caplog.text and 'CHECK: x' in caplog.text
