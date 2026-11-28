# -*- coding: utf-8 -*-
"""Bulk analysis: uptake rows get the protrusion length of the same frame."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mfae.bulk.loader import BulkDataLoader  # noqa: E402


def _loader():
    ld = BulkDataLoader.__new__(BulkDataLoader)
    ld.r_eff_um, ld.V_prot_cap_um3 = 2.0, 0.0
    return ld


def test_uptake_rows_are_matched_to_the_protrusion_trace_by_time():
    """The protrusion trace has every frame; the uptake CSV starts at the cell's entry
    (frame 6 here). Pairing by row index used the length of frame k for uptake row k,
    i.e. 6 frames too early."""
    dt, entry = 3.0, 6
    t_all = np.arange(40) * dt
    L = t_all.copy()                                   # length = time: a mismatch is obvious
    L[:entry] = np.nan                                 # before entry: no length
    n_up = 34
    uptake = {'Time_s': t_all[entry:entry + n_up], 'Body_Intensity': np.full(n_up, 10.0),
              'Prot_Intensity': np.full(n_up, 10.0), 'Body_Mask_Count': np.full(n_up, 100.0),
              'Prot_Mask_Count': np.full(n_up, 50.0), 'F0_Body': np.full(n_up, 5.0),
              'F0_Prot': np.full(n_up, 5.0), 'Scale_Factor_um_per_px': np.full(n_up, 0.5)}
    prot = {'Time_s': t_all, 'Protrusion_Length_um': L}
    _loader()._recompute_volumes_and_norms(uptake, prot)
    expected = math.pi * 2.0 ** 2 * t_all[entry:entry + n_up]
    assert len(uptake['Volume_Prot_um3']) == n_up
    assert np.allclose(uptake['Volume_Prot_um3'], expected)


def test_older_csvs_without_times_still_load_by_row_order():
    uptake = {'Body_Intensity': np.ones(5), 'Prot_Intensity': np.ones(5), 'Body_Mask_Count': np.ones(5) * 10,
              'Prot_Mask_Count': np.ones(5) * 5, 'Scale_Factor_um_per_px': np.full(5, 0.5)}
    prot = {'Protrusion_Length_um': np.arange(7, dtype=float)}
    _loader()._recompute_volumes_and_norms(uptake, prot)
    assert np.allclose(uptake['Volume_Prot_um3'], math.pi * 4.0 * np.arange(5))


def test_unknown_treatments_get_colours_and_other_durations_keep_their_label():
    """Symmetric setup: LifeAct cells, 1 ms pulses. The plots raised 'No colour ramp
    for LifeAct' and styled 1 ms cohorts as aspiration-only."""
    from mfae.bulk.plots import common, figures
    ramp = common.treatment_ramp("LifeAct")
    assert len(ramp) == 6 and len(set(ramp)) == 6 and common.treatment_ramp("LifeAct") == ramp
    assert common.treatment_ramp("WT") == common.WT_DATE_RAMP                   # known ones unchanged
    assert figures._parse_bucket_to_style('EP-whole 100V 1ms (ruptured)') == ('1ms', 'ruptured_post')
    assert figures._parse_bucket_to_style('EP-pre 50V 1ms') == ('1ms', 'intact')
    assert figures._parse_bucket_to_style('EP-pre 100V 5ms') == ('5ms', 'intact')
    assert figures._parse_bucket_to_style('ASP') == ('ASP', 'intact')
