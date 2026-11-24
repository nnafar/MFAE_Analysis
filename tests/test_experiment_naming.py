# -*- coding: utf-8 -*-
"""Reading pulse settings from the experiment name, and the checks around it."""

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config_loader  # noqa: E402
from config_schema import MFAConfig  # noqa: E402
from experiment_naming import parse_experiment_name  # noqa: E402

EP_1MS = "260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-50V-1ms-frame12"
ASP = "260326_MDAMB231_WT_Chip2_Experiment283-1100Pa-0V-0ms-frame0"


@pytest.mark.parametrize("name, voltage, duration_ms, frame, pulse", [
    (EP_1MS, 50, 1.0, 12, True),
    ("260402_MDAMB231_WT_Chip2_Experiment3-1100Pa-100V-100us-frame111", 100, 0.1, 111, True),
    ("260402_MDAMB231_WT_Chip2_Experiment3-1100Pa-80V-2s-frame5", 80, 2000.0, 5, True),
    (ASP, 0, 0.0, 0, False),
])
def test_parse(name, voltage, duration_ms, frame, pulse):
    info = parse_experiment_name(name)
    assert (info.voltage_v, info.duration_ms, info.pulse_frame, info.has_pulse) == \
        (voltage, pytest.approx(duration_ms), frame, pulse)


def test_bulk_loader_uses_the_same_rule():
    sys.path.insert(0, str(ROOT / "Bulk_Analysis"))
    import bulk_file_handling
    import experiment_naming
    assert bulk_file_handling.BulkDataLoader.FOLDER_PATTERN is experiment_naming.FOLDER_PATTERN


def _write(tmp_path, experiment_id, extra=""):
    path = tmp_path / "exp.yaml"
    path.write_text(f"paths:\n  data_folder: '{tempfile.gettempdir()}'\n"
                    f"  experiment_id: '{experiment_id}'\n{extra}", encoding="utf-8")
    return path


def test_pulse_settings_come_from_the_name(tmp_path):
    layered = config_loader.load_layers(_write(tmp_path, EP_1MS))
    config = MFAConfig(**layered.data)
    assert config.dye_uptake_parameters.has_pulse is True
    assert config.dye_uptake_parameters.pulse_frame == 12
    assert layered.sources["dye_uptake_parameters.pulse_frame"] == "experiment name"


def test_aspiration_name_means_no_pulse(tmp_path):
    config = MFAConfig(**config_loader.load_layers(_write(tmp_path, ASP)).data)
    assert config.dye_uptake_parameters.has_pulse is False


def test_config_contradicting_the_name_is_an_error(tmp_path):
    path = _write(tmp_path, EP_1MS, "dye_uptake_parameters:\n  pulse_frame: 10\n")
    with pytest.raises(config_loader.ConfigFileError, match="experiment name"):
        config_loader.load_layers(path)


def test_matching_explicit_value_is_fine(tmp_path):
    path = _write(tmp_path, EP_1MS, "dye_uptake_parameters:\n  pulse_frame: 12\n")
    assert MFAConfig(**config_loader.load_layers(path).data).dye_uptake_parameters.pulse_frame == 12


def test_name_without_naming_rule_uses_config_values(tmp_path):
    path = _write(tmp_path, "my_test_run",
                  "dye_uptake_parameters:\n  pulse_frame: 7\n")
    config = MFAConfig(**config_loader.load_layers(path).data)
    assert config.dye_uptake_parameters.pulse_frame == 7
