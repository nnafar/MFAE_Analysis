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
    cfg = ROOT / "config" / "config_asymmetric.yaml"       # says experiment_type: ep
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
    args = types.SimpleNamespace(config_path=ROOT / "config" / "config_asymmetric.yaml",
                                 batch=[str(setup)], setup=None, output_dir=out, redo=False,
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
    cli.main([str(ROOT / "config" / "config_asymmetric.yaml"), "--batch", "D:/data/X", "D:/data/Y"])
    assert seen["folder"] == ["D:/data/X", "D:/data/Y"]


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


def test_condition_folders_file_into_their_setup():
    assert batch.setup_name(Path("MFAE/Asymmetric_Efield/5ms_100V")) == "Asymmetric_Efield"
    assert batch.setup_name(Path("MFAE/Asymmetric_Efield/100us_100V")) == "Asymmetric_Efield"
    assert batch.setup_name(Path("MFAE/Symmetric_Efield")) == "Symmetric_Efield"
    assert batch.setup_name(Path("MFAE/Aspiration")) == "Aspiration"


def test_two_configs_on_parts_of_one_setup_fill_the_same_setup_folder(tmp_path, monkeypatch):
    """5 ms with one config, 100 us with another: both end up in Output/Asymmetric_Efield/."""
    setup = tmp_path / "Asymmetric_Efield"
    names = {"5ms_100V": PULSED, "100us_100V": "260114_MDAMB231_WT_Chip2_Experiment1-1100Pa-100V-100us-frame10"}
    for cond, name in names.items():
        (setup / cond / name).mkdir(parents=True)
        (setup / cond / name / "C1_t0001.tif").write_bytes(b"")
    out = tmp_path / "Output"
    roots = []

    class FakeAnalysis:
        def __init__(self, path, name, config, output_root=None, use_session_cache=True, log_mode='w'):
            roots.append((name, Path(output_root)))
            self.name, self.results_dir, self.stopped_by_user = name, Path(output_root) / name, False

        def run_analysis(self, setup_only=False):
            return True

        def cleanup(self):
            pass

    monkeypatch.setattr(batch, "MFAAnalysis", FakeAnalysis)
    monkeypatch.setattr(cli, "save_effective_config", lambda *a, **k: None)
    base = ROOT / "config" / "config_asymmetric.yaml"
    other = tmp_path / "config_100us.yaml"                     # e.g. another box size
    other.write_text(base.read_text(encoding="utf-8")
                     .replace("default_roi_width_um: 75.0", "default_roi_width_um: 70.0"),
                     encoding="utf-8")
    for cfg, cond in ((base, "5ms_100V"), (other, "100us_100V")):
        args = types.SimpleNamespace(config_path=cfg, batch=[str(setup / cond)],
                                     setup=None, output_dir=out, redo=False, fresh=False,
                                     check=False, show_config=False, debug=False, experiment_id=None)
        assert batch.run_batch(args) == 0
    assert {r for _, r in roots} == {out / "Asymmetric_Efield"}
    assert {n for n, _ in roots} == set(names.values())


def test_folders_can_be_listed_in_the_config(tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"paths:\n  image_format: tiff\n  data_folder:\n    - '{a}'\n    - '{b}'\n")
    roots, fmt = batch.batch_folders_and_format(cfg, [])
    assert roots == [a, b] and fmt == "tiff"
    roots, _ = batch.batch_folders_and_format(cfg, [str(tmp_path / "C")])   # command line wins
    assert roots == [tmp_path / "C"]


def test_a_plain_run_of_a_folder_list_config_asks_for_batch(capsys):
    assert cli.main([str(ROOT / "config" / "config_asymmetric.yaml")]) == 1
    assert "add --batch" in capsys.readouterr().out


def test_single_experiment_with_a_batch_config_goes_where_the_batch_would(tmp_path, monkeypatch):
    setup = _tree(tmp_path)
    cfg = tmp_path / "asym.yaml"
    src = (ROOT / "config" / "config_asymmetric.yaml").read_text()
    cfg.write_text(src.replace(r"M:\tnw\bn\gk\NN\2_Data-Analysis\MFAE\Asymmetric_Efield'",
                               str(setup) + "'"))
    out = tmp_path / "Output"
    seen = {}

    class FakeAnalysis:
        def __init__(self, path, name, config, output_root=None, use_session_cache=True):
            seen.update(root=Path(output_root), name=name, pulse=config.dye_uptake_parameters.has_pulse)
            self.results_dir = Path(output_root) / name
            self.results_dir.mkdir(parents=True, exist_ok=True)

        def run_analysis(self):
            return True

        def cleanup(self):
            pass

    monkeypatch.setattr(cli, "MFAAnalysis", FakeAnalysis)
    # an experiment inside the config's folder
    assert cli.main([str(cfg), "--data-folder", str(setup / "5ms_100V" / PULSED), "--output-dir", str(out)]) == 0
    assert seen == {"root": out / "Asymmetric_Efield", "name": PULSED, "pulse": True}
    # one outside it, in another condition folder of the setup; 0 V -> aspiration only
    assert cli.main([str(cfg), "--data-folder", str(setup / "0ms_0V" / ASP), "--output-dir", str(out)]) == 0
    assert seen == {"root": out / "Asymmetric_Efield", "name": ASP, "pulse": False}


def test_traps_option_redoes_one_experiment_only(tmp_path, monkeypatch):
    setup = _tree(tmp_path)
    out = tmp_path / "Output"
    for name in (PULSED, ASP):                                  # both already analysed
        d = output_layout.build(out / "Asymmetric_Efield" / name)
        d['overview'].mkdir(parents=True)
        (d['overview'] / "trap_summary.csv").write_text("Trap\n1\n")
    calls = []

    class FakeAnalysis:
        def __init__(self, path, name, config, output_root=None, use_session_cache=True,
                     log_mode='w', retune_traps=None):
            self.name, self.results_dir, self.stopped_by_user = name, Path(output_root) / name, False
            calls.append(("init", name, use_session_cache, retune_traps))

        def run_analysis(self, setup_only=False):
            calls.append(("setup" if setup_only else "analysis", self.name))
            return True

        def cleanup(self):
            pass

    monkeypatch.setattr(batch, "MFAAnalysis", FakeAnalysis)
    monkeypatch.setattr(cli, "save_effective_config", lambda *a, **k: None)
    args = types.SimpleNamespace(config_path=ROOT / "config" / "config_asymmetric.yaml",
                                 batch=[str(setup)], setup=None, output_dir=out, redo=False,
                                 fresh=False, check=False, show_config=False, debug=False,
                                 experiment_id=None, traps=[7], only=None)
    assert batch.run_batch(args) == 1                          # two experiments: ambiguous
    assert not calls
    args.only = "experiment3"
    assert batch.run_batch(args) == 0
    assert calls[0] == ("init", PULSED, True, [7])             # saved setup kept, trap 7 redone
    assert [c[0] for c in calls] == ["init", "setup", "init", "analysis"]
    args.fresh = True
    assert batch.run_batch(args) == 1                          # --fresh and --traps conflict


def test_trap_list_parsing():
    assert cli._trap_list("7") == [7]
    assert cli._trap_list("12, 7,7") == [7, 12]
    with pytest.raises(Exception):
        cli._trap_list("0")
    with pytest.raises(Exception):
        cli._trap_list("seven")


def test_thresholds_option_reopens_every_experiment(tmp_path, monkeypatch):
    setup = _tree(tmp_path)
    out = tmp_path / "Output"
    for name in (PULSED, ASP):                                  # both already analysed
        d = output_layout.build(out / "Asymmetric_Efield" / name)
        d['overview'].mkdir(parents=True)
        (d['overview'] / "trap_summary.csv").write_text("Trap\n1\n")
    calls = []

    class FakeAnalysis:
        def __init__(self, path, name, config, output_root=None, use_session_cache=True,
                     log_mode='w', retune_traps=None, thresholds_only=False):
            self.name, self.results_dir, self.stopped_by_user = name, Path(output_root) / name, False
            calls.append(("init", name, use_session_cache, thresholds_only))

        def run_analysis(self, setup_only=False):
            return True

        def cleanup(self):
            pass

    monkeypatch.setattr(batch, "MFAAnalysis", FakeAnalysis)
    monkeypatch.setattr(cli, "save_effective_config", lambda *a, **k: None)
    args = types.SimpleNamespace(config_path=ROOT / "config" / "config_asymmetric.yaml",
                                 batch=[str(setup)], setup=None, output_dir=out, redo=False,
                                 fresh=False, check=False, show_config=False, debug=False,
                                 experiment_id=None, traps=None, only=None, thresholds=True)
    assert batch.run_batch(args) == 0
    setups = {c[1] for c in calls if c[3]}                       # setup pass: thresholds only
    assert setups == {PULSED, ASP}                              # both, though already analysed
    assert all(c[2] for c in calls)                             # saved setups kept
    args.fresh = True
    assert batch.run_batch(args) == 1


def test_tip_tracking_option_reaches_the_analysis(tmp_path, monkeypatch):
    setup = _tree(tmp_path)
    out = tmp_path / "Output"
    seen = []

    class FakeAnalysis:
        def __init__(self, path, name, config, output_root=None, use_session_cache=True, log_mode='w'):
            seen.append(config.image_processing.tip_tracking)
            self.name, self.results_dir, self.stopped_by_user = name, Path(output_root) / name, False

        def run_analysis(self, setup_only=False):
            return True

        def cleanup(self):
            pass

    monkeypatch.setattr(batch, "MFAAnalysis", FakeAnalysis)
    monkeypatch.setattr(cli, "save_effective_config", lambda *a, **k: None)
    args = types.SimpleNamespace(config_path=ROOT / "config" / "config_asymmetric.yaml",
                                 batch=[str(setup)], setup=None, output_dir=out, redo=True,
                                 fresh=False, check=False, show_config=False, debug=False,
                                 experiment_id=None, traps=None, only=None, thresholds=False,
                                 tip_tracking="kymograph")
    assert batch.run_batch(args) == 0
    assert seen and set(seen) == {"kymograph"}
    assert cli.build_arg_parser().parse_args(["x.yaml", "--tip-tracking", "mask"]).tip_tracking == "mask"


def test_condition_folders_in_either_order_and_a_warning_for_odd_names(caplog):
    for name in ("100V_5ms", "5 ms 100 V", "100us-100V", "1.5ms_50V"):
        assert batch.setup_name(Path("MFAE/Asymmetric_Efield") / name) == "Asymmetric_Efield", name
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert batch.setup_name(Path("MFAE/Asymmetric_Efield/5ms_100V_new")) == "5ms_100V_new"
    assert "looks like a pulse-condition folder" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        batch.setup_name(Path("MFAE/Symmetric_Efield"))
        batch.setup_name(Path("MFAE") / PULSED)            # an experiment folder listed on its own
    assert caplog.text == ""


def test_bulk_says_when_there_are_no_shared_controls(tmp_path, monkeypatch, caplog):
    from mfae.bulk import run as bulk_run
    out = tmp_path / "Output"
    (out / "Symmetric_Efield" / CZI).mkdir(parents=True)
    seen = []
    monkeypatch.setattr(bulk_run, "run_setup", lambda d, extra: seen.append((d.name, [e.name for e in extra])))
    monkeypatch.setattr(bulk_run, "_configure_logging", lambda: None)
    with caplog.at_level("INFO"):
        assert bulk_run.main(["--output-root", str(out)]) == 0
    assert seen == [("Symmetric_Efield", [])]
    assert "No shared control folder 'Aspiration'" in caplog.text
    caplog.clear()
    (out / "Controls" / ASP).mkdir(parents=True)
    seen.clear()
    with caplog.at_level("WARNING"):
        assert bulk_run.main(["--output-root", str(out), "--controls", "Ctrl"]) == 0
    assert "No shared control folder 'Ctrl'" in caplog.text
    seen.clear()
    assert bulk_run.main(["--output-root", str(out), "--controls", "Controls"]) == 0
    assert ("Symmetric_Efield", ["Controls"]) in seen
