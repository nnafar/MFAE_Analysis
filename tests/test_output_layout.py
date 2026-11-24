# -*- coding: utf-8 -*-
"""Output folder layout and the readers' fallback to the old flat layout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import output_layout as ol  # noqa: E402


def test_build_layout(tmp_path):
    d = ol.build(tmp_path)
    assert d['full_csv'] == tmp_path / 'Protrusion Length Detection' / 'Full protrusion detection'
    assert d['membrane_kymo'] == tmp_path / 'Protrusion Length Detection' / 'Kymographs'
    assert d['movies'] == tmp_path / 'Protrusion Length Detection' / 'Movies'
    assert d['dye_traces'] == tmp_path / 'Dye Uptake' / 'Uptake traces'
    assert d['dye_kymo'] == tmp_path / 'Dye Uptake' / 'Uptake Kymographs'
    assert d['dye_movies'] == tmp_path / 'Dye Uptake' / 'Uptake Movies'
    assert d['actin_traces'] == tmp_path / 'Actin Quantification' / 'Actin traces'
    assert d['actin_kymo'] == tmp_path / 'Actin Quantification' / 'Actin Kymographs'
    assert d['actin_movies'] == tmp_path / 'Actin Quantification' / 'Actin Movies'


def test_complete_fills_missing_keys_from_parents(tmp_path):
    d = ol.complete({'root': tmp_path, 'tracking_visuals': tmp_path / 'tv', 'dye': tmp_path / 'dye'})
    assert d['protrusion'] == tmp_path / 'tv'
    assert d['movies'] == tmp_path / 'tv' / 'Movies'
    assert d['dye_movies'] == tmp_path / 'dye' / 'Uptake Movies'
    assert d['actin'] == tmp_path / 'Actin Quantification'


def test_readers_fall_back_to_old_layout(tmp_path):
    old = tmp_path / 'Full protrusion detection'
    old.mkdir()
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == old
    new = tmp_path / 'Protrusion Length Detection' / 'Full protrusion detection'
    new.mkdir(parents=True)
    assert ol.first_existing(ol.full_csv_candidates(tmp_path)) == new
