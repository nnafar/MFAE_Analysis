# -*- coding: utf-8 -*-
"""
Regression tests for:
  * config_schema: unknown keys must not be silently dropped, and settings the
    rupture detector reads (e.g. enable_length_veto) must survive validation.
  * Rupture detection: the pulse-context detector must only run when the
    experiment really has a pulse (dye enabled AND has_pulse True).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config_schema import MFAConfig  # noqa: E402
from LineDetection_MFA import LineDetectionMFA  # noqa: E402


# =============================================================================
# Config schema
# =============================================================================

def _load_config(name: str) -> dict:
    raw = yaml.safe_load((ROOT / name).read_text(encoding='utf-8'))
    raw['paths']['data_folder'] = tempfile.gettempdir()  # must exist
    return raw


@pytest.mark.parametrize('name', ['config.yaml', 'config_100us.yaml',
                                  'config_5ms.yaml', 'config_asp.yaml'])
def test_shipped_configs_validate(name):
    MFAConfig(**_load_config(name))


@pytest.mark.parametrize('name', ['config_100us.yaml', 'config_5ms.yaml', 'config_asp.yaml'])
def test_length_veto_setting_survives_validation(name):
    """These configs set enable_length_veto: False; it used to be dropped."""
    dumped = MFAConfig(**_load_config(name)).model_dump()
    assert dumped['rupture_detection']['enable_length_veto'] is False


def test_unknown_key_in_section_is_rejected():
    raw = _load_config('config.yaml')
    raw['rupture_detection']['enable_lenght_veto'] = False  # typo
    with pytest.raises(ValidationError):
        MFAConfig(**raw)


def test_unimplemented_keys_are_dropped_with_warning(caplog):
    raw = _load_config('config.yaml')
    with caplog.at_level('WARNING'):
        dumped = MFAConfig(**raw).model_dump()
    assert 'doa_haze_collapse_ratio' not in dumped['rupture_detection']
    assert any('doa_haze_collapse_ratio' in r.getMessage() for r in caplog.records)


# =============================================================================
# Rupture detection
# =============================================================================

N_FRAMES = 60
PULSE_IDX = 9  # 0-based index of the pulse frame


def _pulse_step_trace():
    """
    Flat haze, then a modest sustained rise right after the pulse frame.
    The rise (10 -> 13.5, fold 1.35) is enough for the pulse-context
    detector but well below both length-veto haze bypasses (fold 2.0 / 1.7),
    so with a flat protrusion length every non-pulse candidate is vetoed.
    """
    rng = np.random.default_rng(0)
    trace = 10.0 + rng.normal(0.0, 0.05, N_FRAMES)
    trace[PULSE_IDX + 1:] += 3.5
    return trace.tolist()


def _run_detector(dye_params: dict, rupture_overrides: dict = None) -> dict:
    r_params = {'enable_length_veto': True}
    r_params.update(rupture_overrides or {})
    det = LineDetectionMFA([np.zeros((10, 10))], [0, 0],
                           {'rupture_detection': r_params,
                            'dye_uptake_parameters': dye_params})
    det.results.update({
        'downstream_intensities': _pulse_step_trace(),
        'entry_frame_index': 0,
        'entry_confirmed': True,
        'exit_frame_index': N_FRAMES,
        'body_solidity': [1.0] * N_FRAMES,
        'protrusion_lengths_px': [20.0] * N_FRAMES,  # flat length: no volume loss
    })
    det._generate_comprehensive_results()
    return det.results


def test_pulsed_experiment_flags_pulse_induced_rupture():
    res = _run_detector(
        {'enable': True, 'has_pulse': True, 'pulse_frame': 10},
        {'pulse_frame_idx': PULSE_IDX},
    )
    assert res['rupture_detected'] is True
    assert res['rupture_reason'] == 'Pulse-Induced Rupture'
    assert res['rupture_frame_index'] == PULSE_IDX


def test_control_without_pulse_has_no_pulse_induced_rupture():
    """has_pulse: False controls keep pulse_frame in the config; it must be ignored."""
    res = _run_detector(
        {'enable': True, 'has_pulse': False, 'pulse_frame': 10},
        # Even if a stale index is present it must not be used.
        {'pulse_frame_idx': PULSE_IDX},
    )
    assert res['rupture_detected'] is False
    assert res['rupture_reason'] is None


def test_disabling_length_veto_keeps_haze_candidates():
    """With the veto off, the same haze rise is reported on a control."""
    res = _run_detector(
        {'enable': True, 'has_pulse': False, 'pulse_frame': 10},
        {'enable_length_veto': False},
    )
    assert res['rupture_detected'] is True
    assert res['rupture_reason'] != 'Pulse-Induced Rupture'
    assert abs(res['rupture_frame_index'] - (PULSE_IDX + 1)) <= 2


@pytest.mark.parametrize('dye_params, r_params, expected', [
    ({'enable': True, 'has_pulse': True, 'pulse_frame': 10}, {'pulse_frame_idx': 10}, 10),
    ({'enable': True, 'has_pulse': True, 'pulse_frame': 10}, {}, 9),     # t0001 fallback
    ({'enable': True, 'has_pulse': False, 'pulse_frame': 10}, {'pulse_frame_idx': 9}, None),
    ({'enable': False, 'has_pulse': True, 'pulse_frame': 10}, {'pulse_frame_idx': 9}, None),
])
def test_resolve_pulse_frame_idx(dye_params, r_params, expected):
    det = LineDetectionMFA([np.zeros((10, 10))], [0, 0],
                           {'rupture_detection': r_params,
                            'dye_uptake_parameters': dye_params})
    assert det._resolve_pulse_frame_idx() == expected
