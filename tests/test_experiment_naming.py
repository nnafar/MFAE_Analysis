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


USER_SETTINGS = """
experiment_parameters:
  scale_factor: 0.315
  constant_pressure: 1100
  frame_interval: 0.5
  max_traps: 18
dye_uptake_parameters:
  enable: True
  membrane_channel_pattern: 'C1'
  dye_channel_pattern: 'C2'
actin_parameters:
  enable: False
  channel_pattern: 'C3'
rupture_detection:
  enable: False
  enable_smoothing: False
  enable_outlier_rejection: False
model_parameters:
  perform_fitting: False
workflow_settings:
  create_kymographs: True
  selection_frame_fraction: 0.3
  default_roi_width_um: 115.0
  default_roi_height_um: 45.0
  n_workers: 1
"""


def _write(tmp_path, experiment_id, extra="", experiment_type=None):
    """A minimal valid user config; `extra` is merged in as more YAML."""
    import yaml
    data = yaml.safe_load(USER_SETTINGS)
    name = parse_experiment_name(experiment_id)
    data["experiment_type"] = experiment_type or ("asp" if name and not name.has_pulse else "ep")
    data["paths"] = {"data_folder": tempfile.gettempdir(), "experiment_id": experiment_id}
    data = config_loader.deep_merge(data, yaml.safe_load(extra) or {})
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
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
    path = _write(tmp_path, "my_test_run", "dye_uptake_parameters:\n  pulse_frame: 7\n")
    config = MFAConfig(**config_loader.load_layers(path).data)
    assert config.dye_uptake_parameters.has_pulse is True
    assert config.dye_uptake_parameters.pulse_frame == 7


def test_name_without_naming_rule_needs_pulse_frame_for_ep(tmp_path):
    with pytest.raises(config_loader.ConfigFileError, match="pulse_frame"):
        config_loader.load_layers(_write(tmp_path, "my_test_run"))
    # asp needs no pulse frame
    path = _write(tmp_path, "my_test_run", experiment_type="asp")
    assert MFAConfig(**config_loader.load_layers(path).data).dye_uptake_parameters.has_pulse is False


@pytest.mark.parametrize("experiment_type, name", [("asp", EP_1MS), ("ep", ASP)])
def test_experiment_type_must_match_the_voltage(tmp_path, experiment_type, name):
    with pytest.raises(config_loader.ConfigFileError, match="experiment_type"):
        config_loader.load_layers(_write(tmp_path, name, experiment_type=experiment_type))


def test_invalid_experiment_type(tmp_path):
    with pytest.raises(config_loader.ConfigFileError, match="'ep'"):
        config_loader.load_layers(_write(tmp_path, EP_1MS, experiment_type="electro"))


# -----------------------------------------------------------------------------
# The user config and defaults.yaml never share a setting
# -----------------------------------------------------------------------------

def test_missing_user_setting_is_reported(tmp_path):
    path = _write(tmp_path, EP_1MS)
    text = path.read_text(encoding="utf-8").replace("  n_workers: 1\n", "")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(config_loader.ConfigFileError, match="workflow_settings.n_workers"):
        config_loader.load_layers(path)


def test_developer_setting_in_user_config_is_refused(tmp_path):
    path = _write(tmp_path, EP_1MS, "image_processing:\n  wall_clip_margin: 0.3\n")
    with pytest.raises(config_loader.ConfigFileError, match="image_processing.wall_clip_margin"):
        config_loader.load_layers(path)


def test_defaults_file_holds_no_user_settings():
    defaults = config_loader.read_yaml(config_loader.DEFAULTS_FILE)
    user_owned = set(config_loader.USER_KEYS) | set(config_loader.OPTIONAL_USER_KEYS)
    assert not set(config_loader._leaf_keys(defaults)) & user_owned


def test_user_setting_in_defaults_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / "defaults.yaml"
    bad.write_text("workflow_settings:\n  n_workers: 2\n", encoding="utf-8")
    monkeypatch.setattr(config_loader, "DEFAULTS_FILE", bad)
    with pytest.raises(config_loader.ConfigFileError, match="belong in the experiment"):
        config_loader.load_layers(_write(tmp_path, EP_1MS))


@pytest.mark.parametrize("name", ["config.yaml", "config_5ms.yaml",
                                  "config_100us.yaml", "config_asp.yaml"])
def test_shipped_configs_hold_exactly_the_user_settings(name):
    keys = set(config_loader._leaf_keys(config_loader.read_yaml(ROOT / "config" / name)))
    assert keys == set(config_loader.USER_KEYS)
