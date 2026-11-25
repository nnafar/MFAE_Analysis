# -*- coding: utf-8 -*-
"""Batch runs (run_experiment.py --batch) and the per-setup bulk analysis."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mfae import batch, cli  # noqa: E402
from mfae.files import output_layout  # noqa: E402

PULSED = "260114_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10"
ASP = "260325_MDAMB231_CytD_Chip1_Experiment272-1100Pa-0V-0ms-frame0"
CZI = "261119_MDAMB231_LifeAct_Chip1_Experiment2-1100Pa-100V-1ms-frame304"


def _tree(tmp_path):
    """Asymmetric setup with .tif folders at two depths, plus things to skip."""
    setup = tmp_path / "Asymmetric_Efield"
    for rel in (f"5ms_100V/{PULSED}", f"0ms_0V/{ASP}", "junk/not_an_experiment"):
        d = setup / rel
        d.mkdir(parents=True)
        (d / "C1_t0001.tif").write_bytes(b"")
    (setup / "1ms_100V").mkdir()
    (setup / "1ms_100V" / f"{CZI}.czi").write_bytes(b"")
    return setup


def test_finds_experiments_at_any_depth(tmp_path):
    setup = _tree(tmp_path)
    found = batch.find_experiments(setup, "tiff")
    assert sorted(p.name for p in found.experiments) == sorted([PULSED, ASP])
    assert [p.name for p in found.badly_named] == ["not_an_experiment"]
    assert [p.name for p in found.other_format] == [f"{CZI}.czi"]

    czi = batch.find_experiments(setup, "czi")
    assert [p.name for p in czi.experiments] == [f"{CZI}.czi"]
    assert len(czi.other_format) == 3


def test_type_follows_the_name_in_a_batch(tmp_path):
    setup = _tree(tmp_path)
    cfg = ROOT / "config" / "config_asymmetric_tiff.yaml"       # says experiment_type: ep
    asp_dir = setup / "0ms_0V" / ASP
    config, _ = cli.load_config(cfg, asp_dir, type_from_name=True)
    assert config is not None and config.dye_uptake_parameters.has_pulse is False
    config, _ = cli.load_config(cfg, setup / "5ms_100V" / PULSED, type_from_name=True)
    assert config.dye_uptake_parameters.has_pulse is True


def test_batch_does_all_setups_then_all_analyses_and_skips_finished(tmp_path, monkeypatch):
    setup = _tree(tmp_path)
    out = tmp_path / "Output"
    done_dir = output_layout.build(out / "Asymmetric_Efield" / ASP)   # ASP already analysed
    done_dir['overview'].mkdir(parents=True)
    (done_dir['overview'] / "trap_summary.csv").write_text("Trap\n1\n")

    calls = []

    class FakeAnalysis:
        def __init__(self, path, name, config, output_root=None, use_session_cache=True, log_mode='w'):
            self.name, self.results_dir, self.stopped_by_user = name, Path(output_root) / name, False
            assert Path(output_root) == out / "Asymmetric_Efield"       # setup kept separate
            self.results_dir.mkdir(parents=True, exist_ok=True)

        def run_analysis(self, setup_only=False):
            calls.append(("setup" if setup_only else "analysis", self.name))
            return True

        def cleanup(self):
            pass

    monkeypatch.setattr(batch, "MFAAnalysis", FakeAnalysis)
    monkeypatch.setattr(cli, "save_effective_config", lambda *a, **k: None)
    args = types.SimpleNamespace(config_path=ROOT / "config" / "config_asymmetric_tiff.yaml",
                                 batch=str(setup), setup=None, output_dir=out, redo=False,
                                 fresh=False, check=False, show_config=False, debug=False,
                                 experiment_id=None)
    assert batch.run_batch(args) == 0
    assert calls == [("setup", PULSED), ("analysis", PULSED)]         # ASP skipped
    assert list((out / "Asymmetric_Efield").glob("batch_summary_*.txt"))

    calls.clear()
    args.redo = True
    batch.run_batch(args)
    kinds = [k for k, _ in calls]
    assert kinds == ["setup", "setup", "analysis", "analysis"]       # setups first, then analyses


def test_bulk_analyses_each_setup_separately(tmp_path):
    from mfae.bulk.run import find_setups
    out = tmp_path / "Output"
    for setup in ("Symmetric_Efield", "Asymmetric_Efield"):
        (out / setup / PULSED).mkdir(parents=True)
    (out / "Bulk_Analysis_results").mkdir()
    assert [d.name for d in find_setups(out)] == ["Asymmetric_Efield", "Symmetric_Efield"]
    (out / ASP).mkdir()                                               # experiments directly in Output/
    assert [d.name for d in find_setups(out)] == ["Output", "Asymmetric_Efield", "Symmetric_Efield"]


def test_cli_dispatches_batch(monkeypatch):
    seen = {}
    import mfae.batch as mb
    monkeypatch.setattr(mb, "run_batch", lambda args: seen.setdefault("folder", args.batch) and 0)
    cli.main([str(ROOT / "config" / "config_asymmetric_tiff.yaml"), "--batch", "D:/data/X"])
    assert seen["folder"] == "D:/data/X"


def test_aspiration_controls_join_each_setup_but_only_aspiration_runs(tmp_path, monkeypatch):
    from mfae.bulk.loader import BulkDataLoader
    out = tmp_path / "Output"
    (out / "Symmetric_Efield" / CZI).mkdir(parents=True)
    (out / "Aspiration" / ASP).mkdir(parents=True)
    (out / "Aspiration" / PULSED).mkdir(parents=True)               # misplaced pulsed run: ignored
    monkeypatch.setattr(BulkDataLoader, "_load_experiment_traps", lambda self, meta: [meta])
    loader = BulkDataLoader(str(out / "Symmetric_Efield"), control_dirs=[out / "Aspiration"])
    names = sorted(m.full_path.name for _, traps in loader.iter_groups() for m in traps)
    assert names == sorted([CZI, ASP])
    alone = BulkDataLoader(str(out / "Symmetric_Efield"))
    assert [m.full_path.name for _, t in alone.iter_groups() for m in t] == [CZI]


def _setup_config(tmp_path, overrides_yaml):
    src = (ROOT / "config" / "config_asymmetric_tiff.yaml").read_text()
    src = src[:src.index("\noverrides:")] + "\n" + overrides_yaml
    path = tmp_path / "setup.yaml"
    path.write_text(src)
    return path


def test_overrides_apply_by_folder_or_name_and_later_entries_win(tmp_path):
    setup = _tree(tmp_path)
    cfg = _setup_config(tmp_path, """overrides:
  - match: 5ms_100V
    rupture_detection:
      enable_smoothing: True
    experiment_parameters:
      frame_interval: 2.0
  - match: [Experiment3, Experiment999]
    experiment_parameters:
      frame_interval: 3.0
""")
    pulsed = setup / "5ms_100V" / PULSED
    config, layered = cli.load_config(cfg, pulsed, type_from_name=True)
    assert config.rupture_detection.enable_smoothing is True
    assert config.experiment_parameters.frame_interval == 3.0            # later entry wins
    assert layered.sources["experiment_parameters.frame_interval"].endswith(
        "overrides[Experiment3, Experiment999]")
    other, _ = cli.load_config(cfg, setup / "0ms_0V" / ASP, type_from_name=True)
    assert other.rupture_detection.enable_smoothing is False             # not matched
    # the saved settings hold the final values and no 'overrides' section
    saved = cli.save_effective_config(config, layered, cfg, tmp_path / "res")
    again, _ = cli.load_config(saved)
    assert again.experiment_parameters.frame_interval == 3.0
    assert "\noverrides:" not in saved.read_text()


@pytest.mark.parametrize("bad", [
    "  - match: 5ms\n    paths:\n      image_format: czi\n",
    "  - match: 5ms\n    image_processing:\n      membrane_snr_full_scale: 20\n",
    "  - rupture_detection:\n      enable_smoothing: True\n",
])
def test_overrides_refuse_paths_developer_settings_and_missing_match(tmp_path, bad):
    setup = _tree(tmp_path)
    cfg = _setup_config(tmp_path, "overrides:\n" + bad)
    config, _ = cli.load_config(cfg, setup / "5ms_100V" / PULSED, type_from_name=True)
    assert config is None
