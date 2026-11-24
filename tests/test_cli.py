# -*- coding: utf-8 -*-
"""Tests for the user-facing command line of MFA_analysis.py (no GUI involved)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import MFA_analysis as mfa  # noqa: E402

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


def test_init_config_fills_paths_and_keeps_comments(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    assert mfa.main(["--init-config", str(dest), "--template", "asp",
                     "--data-folder", str(data_folder)]) == 0
    text = dest.read_text(encoding="utf-8")
    assert "# CONFIGURATION FOR MICROFLUIDIC ASPIRATION" in text  # comments kept
    paths = yaml.safe_load(text)["paths"]
    assert Path(paths["data_folder"]) == data_folder
    assert paths["experiment_id"] == EXP_NAME  # defaults to the folder name


def test_init_config_refuses_to_overwrite(tmp_path):
    dest = tmp_path / "exists.yaml"
    dest.write_text("keep me")
    assert mfa.main(["--init-config", str(dest)]) == 1
    assert dest.read_text() == "keep me"


def test_check_passes_on_valid_setup(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    mfa.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    assert mfa.main([str(dest), "--check", "--output-dir", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out").exists()  # --check writes nothing


def test_check_fails_when_pulse_is_outside_recording(tmp_path, data_folder):
    dest = tmp_path / "my.yaml"
    mfa.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    dest.write_text(dest.read_text(encoding="utf-8").replace(
        "pulse_frame: 10 ", "pulse_frame: 500 "), encoding="utf-8")
    assert mfa.main([str(dest), "--check"]) == 1


def test_invalid_config_gives_readable_error(tmp_path, data_folder, capsys):
    dest = tmp_path / "bad.yaml"
    mfa.main(["--init-config", str(dest), "--data-folder", str(data_folder)])
    text = dest.read_text(encoding="utf-8")
    text = text.replace("scale_factor: 0.315", "scale_factor: -1")
    text = text.replace("  enable_spike_check:", "  enable_spike_chek:")
    dest.write_text(text, encoding="utf-8")

    assert mfa.main([str(dest), "--check"]) == 1
    # setup_logging() replaces the root handlers, so read the console output.
    log = capsys.readouterr().out
    assert "experiment_parameters.scale_factor" in log
    assert "rupture_detection.enable_spike_chek: unknown setting" in log


def test_missing_config_file(tmp_path):
    assert mfa.main([str(tmp_path / "nope.yaml"), "--check"]) == 1


def test_no_arguments_prints_help(capsys):
    assert mfa.main([]) == 2
    assert "usage:" in capsys.readouterr().out
