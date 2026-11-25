# -*- coding: utf-8 -*-
"""Tests for the user-facing command line of run_experiment.py (no GUI involved)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mfae import cli  # noqa: E402

EXP_NAME = "260101_MDAMB231_WT_Chip1_Experiment1-1100Pa-100V-5ms-frame10"


@pytest.fixture
def data_folder(tmp_path):
    """A tiny two-channel image sequence named like a real acquisition."""
    cv2 = pytest.importorskip("cv2")
    folder = tmp_path / EXP_NAME
    folder.mkdir()
    rng = np.random.default_rng(0)
    for t in range(1, 21):
        for channel in ("C1", "C2"):
            img = (rng.random((32, 32)) * 200 + 20).astype(np.uint16)
            cv2.imwrite(str(folder / f"exp_{channel}_t{t:04d}.tif"), img)
    return folder


def test_init_config_copies_config_yaml_with_new_paths(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    assert cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)]) == 0
    data = yaml.safe_load(dest.read_text(encoding="utf-8"))
    template = yaml.safe_load((ROOT / "config" / "config_tiff_template.yaml").read_text(encoding="utf-8"))
    assert Path(data["paths"]["data_folder"]) == data_folder
    assert data["paths"]["experiment_id"] == EXP_NAME  # defaults to the folder name
    # everything except the paths is copied unchanged
    assert {k: v for k, v in data.items() if k != "paths"} == \
           {k: v for k, v in template.items() if k != "paths"}


def test_init_config_from_another_config(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    other = tmp_path / "other.yaml"
    other.write_text((ROOT / "config" / "config_tiff_template.yaml").read_text(encoding="utf-8")
                     .replace("default_roi_width_um: 115.0", "default_roi_width_um: 70.0"), encoding="utf-8")
    assert cli.main(["--init-config", str(dest), "--from", str(other),
                     "--data-folder", str(data_folder)]) == 0
    data = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert data["workflow_settings"]["default_roi_width_um"] == 70.0


def test_init_config_refuses_to_overwrite(tmp_path):
    dest = tmp_path / "exists.yaml"
    dest.write_text("keep me")
    assert cli.main(["--init-config", str(dest)]) == 1
    assert dest.read_text() == "keep me"


def test_check_passes_on_valid_setup(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    assert cli.main([str(dest), "--check", "--output-dir", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out").exists()  # --check writes nothing


def test_check_fails_when_pulse_is_outside_recording(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    late = EXP_NAME.replace("frame10", "frame500")      # pulse frame comes from the name
    assert cli.main([str(dest), "--check", "--experiment-id", late]) == 1


def test_invalid_config_gives_readable_error(tmp_path, data_folder, capsys):
    dest = tmp_path / "bad.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    text = dest.read_text(encoding="utf-8")
    dest.write_text(text.replace("scale_factor: 0.315", "scale_factor: -1"), encoding="utf-8")
    assert cli.main([str(dest), "--check"]) == 1
    # setup_logging() replaces the root handlers, so read the console output.
    assert "experiment_parameters.scale_factor: Input should be greater than 0" \
        in capsys.readouterr().out


def test_developer_setting_in_user_config_is_refused(tmp_path, data_folder, capsys):
    """User configs may only contain user settings (typos land here too)."""
    dest = tmp_path / "bad.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    text = dest.read_text(encoding="utf-8").replace(
        "  n_workers:", "  max_cache_size: 30\n  n_workers:")
    dest.write_text(text, encoding="utf-8")
    assert cli.main([str(dest), "--check"]) == 1
    out = capsys.readouterr().out
    assert "not user settings" in out and "workflow_settings.max_cache_size" in out


def test_missing_config_file(tmp_path):
    assert cli.main([str(tmp_path / "nope.yaml"), "--check"]) == 1


def test_no_arguments_prints_help(capsys):
    assert cli.main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_show_config_marks_where_values_come_from(tmp_path, data_folder, capsys):
    dest = tmp_path / "my.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    capsys.readouterr()
    assert cli.main([str(dest), "--show-config"]) == 0
    out = capsys.readouterr().out
    assert "has_pulse: true  # experiment_type" in out
    assert "pulse_frame: 10.0  # experiment name" in out
    assert "scale_factor: 0.315  # my.yaml" in out
    assert "min_area_threshold: 100\n" in out          # from defaults: no comment


def test_effective_config_reproduces_the_run(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    config, layered = cli.load_config(dest)
    saved = cli.save_effective_config(config, layered, dest, tmp_path)
    again, _ = cli.load_config(saved)
    assert again.model_dump() == config.model_dump()


def test_leftover_preset_line_is_ignored(tmp_path, data_folder, capsys):
    dest = tmp_path / "my.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    dest.write_text("preset: ep\n" + dest.read_text(encoding="utf-8"), encoding="utf-8")
    capsys.readouterr()
    assert cli.main([str(dest), "--check"]) == 0
    assert "'preset' is no longer used" in capsys.readouterr().out


def test_legacy_full_config_gives_identical_settings():
    """An old 400-line config (no preset) must still load, with the same result."""
    import json
    import tempfile
    data = ROOT / "tests" / "data"
    config, _ = cli.load_config(data / "legacy_full_config.yaml",
                                data_folder=Path(tempfile.gettempdir()),
                                experiment_id="260128_MDAMB231_WT_Device1_Experiment3-1100Pa-100V-5ms-frame10")
    expected = json.loads((data / "legacy_full_config_expected.json").read_text())
    assert json.loads(json.dumps(config.model_dump(mode="json"))) == expected


def test_check_reports_whether_saved_setup_will_be_reused(tmp_path, data_folder, capsys):
    import json
    dest = tmp_path / "my.yaml"
    cli.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    cache = tmp_path / "out" / EXP_NAME / "session_cache.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"trap_configs": []}))            # old cache: no scale recorded
    assert cli.main([str(dest), "--check", "--output-dir", str(tmp_path / "out")]) == 0
    assert "interactive setup will run again" in capsys.readouterr().out
