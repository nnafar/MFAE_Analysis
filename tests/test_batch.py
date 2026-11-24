# -*- coding: utf-8 -*-
"""Batch mode: every experiment in one parent folder with one config (--batch)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import MFA_analysis as mfa  # noqa: E402

EP = "260101_MDAMB231_WT_Chip1_Experiment1-1100Pa-100V-5ms-frame10"
ASP = "260101_MDAMB231_WT_Chip1_Experiment2-1100Pa-0V-0ms-frame0"


def _tif_experiment(parent: Path, name: str) -> Path:
    cv2 = pytest.importorskip("cv2")
    folder = parent / name
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for t in range(1, 21):
        for channel in ("C1", "C2"):
            img = (rng.random((32, 32)) * 200 + 20).astype(np.uint16)
            cv2.imwrite(str(folder / f"exp_{channel}_t{t:04d}.tif"), img)
    return folder


@pytest.fixture
def batch_folder(tmp_path):
    parent = tmp_path / "100us_100V"
    _tif_experiment(parent, EP)
    _tif_experiment(parent, ASP)
    (parent / "notes").mkdir()                      # no .tif files: not an experiment
    return parent


@pytest.fixture
def config(tmp_path, batch_folder):
    dest = tmp_path / "series.yaml"
    assert mfa.main(["--init-config", str(dest), "--data-folder", str(batch_folder / EP)]) == 0
    return dest


def test_find_experiments(tmp_path, batch_folder):
    assert [p.name for p in mfa.find_experiments(batch_folder, "tiff")] == sorted([EP, ASP])
    czi_dir = tmp_path / "czi"
    czi_dir.mkdir()
    for n in (EP, ASP):
        (czi_dir / f"{n}.czi").write_bytes(b"")
    (czi_dir / "readme.txt").write_text("x")
    assert [p.stem for p in mfa.find_experiments(czi_dir, "czi")] == sorted([EP, ASP])


def test_batch_check_uses_each_name_for_pulse_and_type(tmp_path, batch_folder, config, capsys):
    # One ep config for a folder holding an ep and an asp (0 V) experiment.
    rc = mfa.main([str(config), "--data-folder", str(batch_folder), "--batch", "--check",
                   "--output-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"{EP}: check passed" in out and f"{ASP}: check passed" in out


def test_badly_named_folder_is_skipped(tmp_path, batch_folder, config, capsys):
    _tif_experiment(batch_folder, "some_folder")
    rc = mfa.main([str(config), "--data-folder", str(batch_folder), "--batch", "--check"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "some_folder: skipped: name doesn't follow the naming rule" in out
    assert f"{EP}: check passed" in out


def test_batch_runs_all_setups_then_all_analyses(tmp_path, batch_folder, config, monkeypatch):
    calls = []

    def fake_run(self, setup_only=False):
        calls.append(("setup" if setup_only else "analysis", self.experiment_id))
        return True

    monkeypatch.setattr(mfa.MFAAnalysis, "run_analysis", fake_run)
    rc = mfa.main([str(config), "--data-folder", str(batch_folder), "--batch",
                   "--output-dir", str(tmp_path / "out")])
    assert rc == 0
    names = sorted([EP, ASP])
    assert calls == [("setup", n) for n in names] + [("analysis", n) for n in names]
    assert list((tmp_path / "out").glob("batch_summary_*.txt"))
    for n in names:
        assert (tmp_path / "out" / n / "effective_config.yaml").exists()


def test_esc_during_setup_stops_the_batch(tmp_path, batch_folder, config, monkeypatch):
    calls = []

    def fake_run(self, setup_only=False):
        calls.append(("setup" if setup_only else "analysis", self.experiment_id))
        if setup_only and len(calls) == 2:
            self.stopped_by_user = True
            return False
        return True

    monkeypatch.setattr(mfa.MFAAnalysis, "run_analysis", fake_run)
    rc = mfa.main([str(config), "--data-folder", str(batch_folder), "--batch",
                   "--output-dir", str(tmp_path / "out")])
    first, second = sorted([EP, ASP])
    assert rc == 1
    # first experiment set up, Esc in the second: only the first is analysed
    assert calls == [("setup", first), ("setup", second), ("analysis", first)]
