# -*- coding: utf-8 -*-
"""Bulk results: one Bulk_Analysis_results/ folder with a sub-folder per setup, plain file names."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mfae.bulk import layout as bl  # noqa: E402


def _names_in_the_code():
    """Every literal output name the bulk code can save (f-string fields filled with a cohort)."""
    names = set()
    rx = re.compile(r"f?[\"']((?:Thesis_|MI_|ASP_|spatial_|spearman)[^\"'\n]*\.(?:pdf|png|csv|tex))[\"']")
    for py in (ROOT / "mfae" / "bulk").rglob("*.py"):
        for m in rx.finditer(py.read_text(encoding="utf-8")):
            n = re.sub(r"\{[^}]*\}", "LifeAct", m.group(1))
            names.add(n)
    return sorted(names)


@pytest.mark.parametrize("old", _names_in_the_code())
def test_every_output_gets_a_plain_name(old):
    new = bl.plain_name(old)
    assert not new.startswith(("Thesis_", "MI_", "ASP_")), (old, new)
    assert "(" not in new and "EP-pre" not in new and Path(new).suffix == Path(old).suffix
    assert bl._note(new), f"no README description for {new} (from {old})"


@pytest.mark.parametrize("old", ["Thesis_MI_Boxplot_EP-pre_100V_1ms_(intact).pdf",
                                 "Thesis_Fate_Pie_LifeAct_EP-whole_1ms.pdf",
                                 "spatial_heatmap_EP-pre_50V_1ms_(ruptured)_Body.png"])
def test_cohort_part_is_cleaned(old):
    new = bl.plain_name(old)
    assert "EP-" not in new and "(" not in new and "__" not in new, new


def test_examples_read_as_what_they_show():
    assert bl.plain_name("Thesis_PrePulse_Visco_Parameter_Boxplots.pdf") == \
        "Viscoelastic_parameters_before_pulse_by_condition.pdf"
    assert bl.plain_name("Thesis_Fate_PrePulse_Visco_Boxplots.pdf") == "Viscoelastic_parameters_before_pulse_by_fate.pdf"
    assert bl.plain_name("Thesis_Combined_Protrusion_Dynamics_MI.pdf") == "Protrusion_length_before_pulse_by_condition.pdf"
    assert bl.plain_name("Thesis_Combined_PrePulse_Trace_by_Fate.pdf") == "Protrusion_length_before_pulse_by_fate.pdf"
    assert bl.plain_name("Thesis_MI_Boxplot_EP-pre_100V_1ms_(intact).pdf") == "Max_protrusion_length_per_trap_100V_1ms_intact.pdf"
    assert bl.plain_name("Thesis_MI_Boxplot_ASP.pdf") == "Max_protrusion_length_per_trap_aspiration.pdf"
    assert bl.plain_name("Thesis_Uptake_vs_ProtLength_WT_U100.pdf") == "Uptake_at_100s_vs_protrusion_length_WT.pdf"
    assert bl.plain_name("README.txt") == "README.txt" and bl.plain_name("mechanics_results_all_traps.csv") == \
        "mechanics_results_all_traps.csv"


def test_rename_outputs_keeps_both_files_when_two_names_meet(tmp_path):
    d = tmp_path / "1_Aspiration_Mechanics" / "All_treatments"
    d.mkdir(parents=True)
    (d / "Thesis_ASP_Actin_F0_Boxplots.pdf").write_text("a")
    (d / "Thesis_ASP_Actin_Baselines.pdf").write_text("b")
    (tmp_path / "0_Tables" / "Figure_Data").mkdir(parents=True)
    (tmp_path / "0_Tables" / "Figure_Data" / "Mechanics.csv").write_text("x")
    done = bl.rename_outputs(tmp_path)
    assert sorted(p.name for p in d.iterdir()) == ["Actin_baseline_by_treatment.pdf", "Actin_baseline_by_treatment_2.pdf"]
    assert len(done) == 2 and (tmp_path / "0_Tables" / "Figure_Data" / "Mechanics.csv").exists()


def test_bulk_results_go_to_one_folder_with_a_folder_per_setup(tmp_path, monkeypatch):
    from mfae.bulk import run as bulk_run
    out = tmp_path / "Output"
    exp = "261119_MDAMB231_LifeAct_Chip1_Experiment2-1100Pa-100V-1ms-frame304"
    (out / "Symmetric_Efield" / exp).mkdir(parents=True)
    (out / "Asymmetric_Efield" / "260114_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-5ms-frame10").mkdir(parents=True)
    made = []

    def fake_run(setup_dir, results_dir, cache_root, controls):
        made.append(results_dir)
        (results_dir / "2_Before_Pulse" / "All_conditions").mkdir(parents=True)
        (results_dir / "2_Before_Pulse" / "All_conditions" / "Thesis_Combined_Protrusion_Dynamics_MI.pdf").write_text("x")
    monkeypatch.setattr(bulk_run, "_run", fake_run)
    monkeypatch.setattr(bulk_run, "_configure_logging", lambda: None)
    assert bulk_run.main(["--output-root", str(out)]) == 0
    rel = sorted(p.relative_to(out).parts[:2] for p in made)
    assert rel == [("Bulk_Analysis_results", "Asymmetric_Efield"), ("Bulk_Analysis_results", "Symmetric_Efield")]
    for r in made:
        assert (r / "2_Before_Pulse" / "All_conditions" / "Protrusion_length_before_pulse_by_condition.pdf").exists()
        assert "Protrusion_length_before_pulse_by_condition.pdf" in (r / "README.txt").read_text()
    # a second run does not treat Bulk_Analysis_results as a setup
    assert [d.name for d in bulk_run.find_setups(out)] == ["Asymmetric_Efield", "Symmetric_Efield"]


def test_figure_data_of_older_runs_still_loads(tmp_path):
    import pandas as pd
    from mfae.bulk import figure_data as fd
    pd.DataFrame({'a': [1]}).to_csv(tmp_path / "Ch3_Data_Mechanics.csv", index=False)
    pd.DataFrame({'b': [2]}).to_csv(tmp_path / "Traces_Protrusion.csv", index=False)
    data = fd.load_all(tmp_path)
    assert list(data['mechanics'].columns) == ['a'] and list(data['protrusion'].columns) == ['b']


def test_the_main_treatment_is_that_of_the_pulsed_cells():
    """Symmetric setup: LifeAct cells are pulsed; figures for 'the' treatment were fixed to WT
    (empty, and named _WT), also when the shared aspiration controls are WT."""
    import pandas as pd
    from mfae.bulk.plots import common
    sym = pd.DataFrame({'Treatment': ['LifeAct'] * 5 + ['WT'] * 8,
                        'Condition_Type': ['EP'] * 5 + ['ASP'] * 8})
    assert common.base_treatment(sym) == 'LifeAct'
    assert common.treatments_with_pulse(sym) == ('LifeAct',)
    asym = pd.DataFrame({'Treatment': ['WT'] * 4 + ['CytD'] * 4, 'Condition_Type': ['EP'] * 6 + ['ASP'] * 2})
    assert common.base_treatment(asym) == 'WT' and common.treatments_with_pulse(asym) == ('WT', 'CytD')
    asp_only = pd.DataFrame({'Treatment': ['CytD', 'WT'], 'Condition_Type': ['ASP', 'ASP']})
    assert common.base_treatment(asp_only) == 'WT'
    assert bl.plain_name("Thesis_LifeAct_MeanUptakeTrace.pdf") == "Uptake_over_time_by_condition_LifeAct.pdf"
