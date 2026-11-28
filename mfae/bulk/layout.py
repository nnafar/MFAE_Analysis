# -*- coding: utf-8 -*-
"""
Folder layout of one bulk-analysis run (mfae/bulk/run.py).

Output/Bulk_Analysis_results/<setup>/<YYMMDD_HHMM>/   (setup e.g. Asymmetric_Efield)
    README.txt                    what every folder and file holds
    0_Tables/                     the numbers: one row per trap, attrition, Figure_Data/
    1_Aspiration_Mechanics/       aspiration-only cells (no pulse): viscoelastic mechanics
        <Treatment>/              single-treatment figures (e.g. WT/, CytD/)
        All_treatments/           treatments compared
    2_Before_Pulse/               pulsed vs aspiration-only cells, fitted before the pulse
        Aspiration/  <pulse>/     single-cohort figures (<pulse> e.g. 100V_5ms)
        All_conditions/           cohorts compared
    3_At_Pulse/                   what happened at the pulse: fate, uptake, actin
        <pulse>/  All_conditions/
    4_Whole_Trace/                whole recording: model-independent fits, uptake
        <pulse>/  All_conditions/

Treatment and pulse sub-folders are made for the conditions present in the
data, so any treatment or pulse gets its own folder.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

ALL_TREATMENTS = "All_treatments"
ALL_CONDITIONS = "All_conditions"
ASPIRATION = "Aspiration"

FOLDERS = {
    # key:      (folder name,              description)
    'tables':   ("0_Tables",               "The numbers behind everything: one row per trap, how many "
                                           "cells were found/analysed/excluded and why, and every value "
                                           "plotted in the figures (Figure_Data/)."),
    'asp':      ("1_Aspiration_Mechanics", "Aspiration-only cells (no pulse): viscoelastic fits per cell, "
                                           "parameters compared between treatments, and controls (trap "
                                           "position, body volume, actin). <Treatment>/ = one treatment, "
                                           f"{ALL_TREATMENTS}/ = treatments compared."),
    'pre':      ("2_Before_Pulse",         "Pulsed vs aspiration-only cells, fitted on the same window before "
                                           "the pulse: were the cohorts mechanically alike before the pulse, "
                                           "and does pre-pulse mechanics predict fate? "
                                           f"{ASPIRATION}/ and <pulse>/ = one cohort, {ALL_CONDITIONS}/ = compared."),
    'at':       ("3_At_Pulse",             "What happened at the pulse: cell fate (intact / ruptured) per "
                                           "condition and per trap, dye uptake and actin around the pulse."),
    'whole':    ("4_Whole_Trace",          "The whole recording: model-independent fits (also for cohorts too "
                                           "short for a pre-pulse fit), mean uptake traces, per-trap uptake."),
}

TABLES = {
    'results':   ("mechanics_results_all_traps.csv",
                  "one row per trap: metadata, fate, every fitted parameter and fit-quality flag"),
    'attrition': ("attrition_per_experiment.csv",
                  "per experiment: cells found, analysed and excluded (with reason)"),
}

FIGURE_DATA_NOTE = ("every value behind the figures (traces per channel, mechanics, attrition), "
                    "to redraw figures without rerunning: thesis_figure_data.load_all(<this folder>)")

# --------------------------------------------------------------------------
# File names
# --------------------------------------------------------------------------
# The plot code saves under its historical names (Thesis_..., MI_..., ...).
# At the end of a run every file is renamed by the table below to a plain
# name: what is shown, then how it is grouped, then the cohort. The folder
# already says the phase (before the pulse / at the pulse / whole recording)
# and the condition. Abbreviations are written out:
#     ASP -> aspiration, EP -> pulsed, Visco -> viscoelastic,
#     MI -> model-free (fits of L(t) without a material model: linear, power law),
#     U100 -> uptake at 100 s after the pulse, A -> uptake plateau (fitted).
# (regex on the file stem, replacement); first match wins. \1 etc. is the
# cohort part of the old name, cleaned by _cohort().
RENAMES = [
    # 1_Aspiration_Mechanics
    (r"^Thesis_ASP_Actin_(?:F0_Boxplots?|Baselines)$",   "Actin_baseline_by_treatment"),
    (r"^Thesis_ASP_Actin_Trace_by_Condition$",           "Actin_over_time_by_treatment"),
    (r"^Thesis_ASP_Actin_vs_E$",                         "Actin_vs_viscoelastic_parameters"),
    (r"^Thesis_ASP_BodyVolume_vs_E$",                    "Body_volume_vs_stiffness_E"),
    (r"^Thesis_ASP_ProtLength_vs_Mechanics$",            "Max_protrusion_length_vs_viscoelastic_parameters"),
    (r"^Thesis_ASP_Trap_Dependency$",                    "Mechanics_by_trap_position"),
    (r"^Thesis_ASP_Visco_Model_Selection_Frequency$",    "Viscoelastic_model_chosen_counts"),
    (r"^MI_Model_Selection_Frequency$",                  "Model-free_model_chosen_counts"),
    (r"^Thesis_ASP_Visco_Parameter_Boxplots$",           "Viscoelastic_parameters_by_treatment"),
    (r"^Thesis_Combined_Protrusion_Dynamics_ASP$",       "Protrusion_length_over_time_by_treatment"),
    (r"^(?:Thesis_)?ASP_BestFit_Panel_(.+)$",            "Viscoelastic_fit_per_cell_{1}"),
    (r"^Thesis_Boxplot_(.+?)_ASP$",                      "Max_protrusion_length_per_trap_{1}"),
    (r"^Thesis_Boxplot_(.+)$",                           "Max_protrusion_length_per_trap_{1}"),
    # 2_Before_Pulse
    (r"^Thesis_Combined_Protrusion_Dynamics_MI$",        "Protrusion_length_before_pulse_by_condition"),
    (r"^Thesis_Combined_PrePulse_Trace_by_Fate$",        "Protrusion_length_before_pulse_by_fate"),
    (r"^Thesis_PrePulse_Visco_Parameter_Boxplots$",      "Viscoelastic_parameters_before_pulse_by_condition"),
    (r"^Thesis_Fate_PrePulse_Visco_Boxplots$",           "Viscoelastic_parameters_before_pulse_by_fate"),
    (r"^Thesis_MI_Boxplot_(.+)$",                        "Max_protrusion_length_per_trap_{1}"),
    (r"^Thesis_MI_PerTrap_Params_(.+)$",                 "Model-free_parameters_per_trap_before_pulse_{1}"),
    (r"^Thesis_MI_Parameter_Boxplots$",                  "Model-free_parameters_before_pulse_by_condition"),
    (r"^MI_Fits_Panel_(.+)$",                            "Model-free_fit_per_cell_before_pulse_{1}"),
    (r"^Thesis_PrePulse_Mechanics_Correlations_(.+?)_U100$", "Uptake_at_100s_vs_viscoelastic_parameters_{1}"),
    (r"^Thesis_PrePulse_Mechanics_Correlations_(.+)$",   "Uptake_plateau_vs_viscoelastic_parameters_{1}"),
    # 3_At_Pulse
    (r"^Thesis_Fate_Pie_(.+)$",                          "Fate_pie_{1}"),
    (r"^Thesis_Fate_PerTrap_(.+)$",                      "Fate_per_trap_{1}"),
    (r"^Thesis_Fate_Counts_by_Pulse$",                   "Fate_counts_by_pulse"),
    (r"^Thesis_EP_Actin_Around_Pulse_(.+)_ByFate$",      "Actin_around_pulse_by_fate_{1}"),
    (r"^Thesis_EP_Actin_Around_Pulse_(.+)$",             "Actin_around_pulse_{1}"),
    (r"^Thesis_EP_Protrusion_Around_Pulse_(.+)$",        "Protrusion_length_around_pulse_{1}"),
    (r"^Thesis_EP_Actin_PrePost_Paired_ByFate$",         "Actin_before_vs_after_pulse_by_fate"),
    (r"^Thesis_EP_Actin_PrePost_Paired$",                "Actin_before_vs_after_pulse"),
    (r"^Thesis_(.+?)_MeanUptakeTrace$",                  "Uptake_over_time_by_condition_{1}"),
    (r"^Thesis_Uptake_Mean_Trace_BodyProt_(.+)$",        "Uptake_over_time_body_and_protrusion_{1}"),
    (r"^Thesis_Uptake_Mean_Trace_ByFitSubset_(.+)$",     "Uptake_over_time_by_fit_quality_{1}"),
    (r"^Thesis_Uptake_Amplitude_Per_Trap_(.+)$",         "Uptake_plateau_per_trap_{1}"),
    (r"^Thesis_Uptake_BestFit_Panel_(.+)$",              "Uptake_fit_per_cell_{1}"),
    (r"^Thesis_Uptake_Kinetics_(.+?)_tau_only_(.+)$",    "Uptake_kinetics_table_{1}_{2}"),
    (r"^Thesis_Uptake_Kinetics_(.+)$",                   "Uptake_kinetics_table_{1}"),
    (r"^Thesis_Uptake_vs_ProtLength_(.+?)_U100$",        "Uptake_at_100s_vs_protrusion_length_{1}"),
    (r"^Thesis_Uptake_vs_ProtLength_(.+)$",              "Uptake_plateau_vs_protrusion_length_{1}"),
    (r"^Thesis_Uptake_Correlation_Sweep_(.+?)_U100_dilution_check$",
                                                         "Uptake_at_100s_correlations_{1}_dilution_check"),
    (r"^Thesis_Uptake_Correlation_Sweep_(.+?)_U100$",    "Uptake_at_100s_correlations_{1}"),
    (r"^Thesis_Uptake_Correlation_Sweep_(.+?)_A$",       "Uptake_plateau_correlations_{1}"),
    (r"^Thesis_Mechanics_vs_Uptake_(.+)$",               "Uptake_vs_mechanics_{1}"),
    (r"^Thesis_Spatial_Mech_Uptake_3D_(.+)$",            "Mechanics_uptake_and_trap_position_3D_{1}"),
    # 4_Whole_Trace
    (r"^Thesis_Combined_WholeTrace_Trace_by_Fate$",      "Protrusion_length_whole_recording_by_fate"),
    (r"^Thesis_EP_WholeTrace_by_Fate$",                  "Protrusion_length_whole_recording_by_pulse_and_fate"),
    (r"^Thesis_Fate_WholeTrace_MI_Boxplots$",            "Model-free_parameters_whole_recording_by_fate"),
    (r"^Thesis_EP_Fate_MI_Boxplots$",                    "Model-free_parameters_by_fate"),
    (r"^Thesis_MI_WholeTrace_Parameter_Boxplots$",       "Model-free_parameters_whole_recording_by_condition"),
    (r"^Thesis_MI_WholeTrace_Protrusion_Dynamics$",      "Protrusion_length_whole_recording_by_condition"),
    (r"^Thesis_MI_WholeTrace_PerTrap_Params_(.+)$",      "Model-free_parameters_per_trap_whole_recording_{1}"),
    (r"^MI_WholeTrace_Fits_Panel_(.+)$",                 "Model-free_fit_per_cell_whole_recording_{1}"),
    (r"^Thesis_MI_Whole_Mechanics_Correlations_(.+?)_U100$", "Uptake_at_100s_vs_model-free_parameters_{1}"),
    (r"^Thesis_MI_Whole_Mechanics_Correlations_(.+)$",   "Uptake_plateau_vs_model-free_parameters_{1}"),
    # trap position, correlations
    (r"^spatial_cell_size_(.+)$",                        "Cell_size_by_trap_position_{1}"),
    (r"^spatial_heatmap_(.+)$",                          "Trap_position_heatmap_{1}"),
    (r"^spatial_trap_gradient_(.+)$",                    "Trend_along_the_trap_row_{1}"),
    (r"^spearman_parameter_pairplot(.*)$",               "Parameter_correlations{1}"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in RENAMES]


def _cohort(text: str) -> str:
    """Cohort part of an old name, in plain words: 'EP-pre_100V_1ms_(intact)' -> '100V_1ms_intact'."""
    t = str(text)
    t = re.sub(r"(?i)(^|[_ ])EP-(?:pre|whole)(?=[_ ]|$)", r"\1", t)   # the folder says before / whole
    t = re.sub(r"(?i)(^|_)ASP(?=_|$)", r"\1aspiration", t)
    t = t.replace("(", "").replace(")", "")
    t = re.sub(r"[^\w\-]+", "_", t)
    return re.sub(r"_+", "_", t).strip("_")


def plain_name(filename: str) -> str:
    """New name of one output file (the extension is kept); unknown names are cleaned only."""
    p = Path(filename)
    stem = p.stem
    for rx, repl in _COMPILED:
        m = rx.match(stem)
        if m:
            parts = [_cohort(g) for g in m.groups()]
            new = repl
            for k, g in enumerate(parts, 1):
                new = new.replace("{%d}" % k, g)
            return re.sub(r"_+", "_", new).strip("_") + p.suffix
    if stem.startswith("Thesis_"):                       # not in the table: drop the prefix at least
        return _cohort(stem[len("Thesis_"):]) + p.suffix
    return p.name


def rename_outputs(root: Union[str, Path]) -> Dict[str, str]:
    """Rename every figure/table of a run to its plain name; returns {old relative path: new name}."""
    root = Path(root)
    done: Dict[str, str] = {}
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        if f.name == "README.txt" or "Figure_Data" in f.parts:
            continue
        new = plain_name(f.name)
        if new == f.name:
            continue
        target, k = f.with_name(new), 2
        while target.exists():                           # two old names -> one new name: keep both
            target = f.with_name(f"{Path(new).stem}_{k}{Path(new).suffix}")
            k += 1
        f.rename(target)
        done[f.relative_to(root).as_posix()] = target.name
    return done


# File-name prefix -> what the figure shows (for README.txt). First match wins.
PLOT_NOTES = [
    ("Viscoelastic_fit_per_cell",         "viscoelastic best fit for every aspirated cell of one treatment"),
    ("Max_protrusion_length_per_trap",    "maximum protrusion length per trap, one condition"),
    ("Protrusion_length_over_time_by_treatment", "mean protrusion length over time, treatments overlaid"),
    ("Protrusion_length_before_pulse_by_condition", "mean protrusion length before the pulse, cohorts overlaid"),
    ("Protrusion_length_before_pulse_by_fate", "mean protrusion length before the pulse, intact vs ruptured"),
    ("Protrusion_length_whole_recording_by_fate", "mean protrusion length over the recording, intact vs ruptured"),
    ("Protrusion_length_whole_recording_by_pulse_and_fate",
                                          "protrusion length over the recording per pulse, intact vs ruptured"),
    ("Protrusion_length_whole_recording_by_condition", "mean protrusion length over the whole recording"),
    ("Protrusion_length_around_pulse",    "protrusion length aligned to the pulse"),
    ("Mechanics_by_trap_position",        "fitted parameters against trap position (control)"),
    ("Body_volume_vs_stiffness_E",        "cell body volume against stiffness E (control)"),
    ("Max_protrusion_length_vs_viscoelastic", "maximum protrusion length against fitted parameters"),
    ("Actin_vs_viscoelastic",             "actin intensity against fitted parameters"),
    ("Actin_baseline_by_treatment",       "actin baseline (F0) per treatment"),
    ("Actin_over_time_by_treatment",      "actin intensity over time per treatment"),
    ("Actin_around_pulse",                "actin intensity aligned to the pulse"),
    ("Actin_before_vs_after_pulse",       "actin before vs after the pulse, paired per cell"),
    ("Viscoelastic_model_chosen_counts",  "how often each viscoelastic model fitted best"),
    ("Model-free_model_chosen_counts",    "how often each model-free fit (linear, power law) fitted best"),
    ("Viscoelastic_parameters_by_treatment", "viscoelastic parameters, treatments compared"),
    ("Viscoelastic_parameters_before_pulse_by_condition", "viscoelastic parameters before the pulse, cohorts compared"),
    ("Viscoelastic_parameters_before_pulse_by_fate", "viscoelastic parameters before the pulse, intact vs ruptured"),
    ("Model-free_parameters_per_trap",    "model-free fit parameters per trap"),
    ("Model-free_parameters_before_pulse_by_condition", "model-free parameters before the pulse, cohorts compared"),
    ("Model-free_parameters_whole_recording_by_condition", "model-free parameters of the whole recording, cohorts compared"),
    ("Model-free_parameters_whole_recording_by_fate", "model-free parameters of the whole recording, intact vs ruptured"),
    ("Model-free_parameters_by_fate",     "model-free parameters, intact vs ruptured"),
    ("Model-free_fit_per_cell",           "model-free fit of L(t) for every cell of one condition"),
    ("Uptake_plateau_vs_viscoelastic",    "dye uptake plateau against pre-pulse viscoelastic parameters"),
    ("Uptake_at_100s_vs_viscoelastic",    "dye uptake at 100 s against pre-pulse viscoelastic parameters"),
    ("Uptake_plateau_vs_model-free",      "dye uptake plateau against whole-recording model-free parameters"),
    ("Uptake_at_100s_vs_model-free",      "dye uptake at 100 s against whole-recording model-free parameters"),
    ("Uptake_plateau_vs_protrusion_length", "dye uptake plateau against protrusion length"),
    ("Uptake_at_100s_vs_protrusion_length", "dye uptake at 100 s against protrusion length"),
    ("Uptake_plateau_correlations",       "dye uptake plateau correlated with each parameter (table)"),
    ("Uptake_at_100s_correlations",       "dye uptake at 100 s correlated with each parameter (table)"),
    ("Uptake_over_time_by_condition",     "mean dye uptake over time, aspiration vs each pulse"),
    ("Uptake_over_time_body_and_protrusion", "mean dye uptake over time, body and protrusion"),
    ("Uptake_over_time_by_fit_quality",   "mean dye uptake over time, cells split by fit quality"),
    ("Uptake_plateau_per_trap",           "dye uptake plateau per trap position"),
    ("Uptake_fit_per_cell",               "dye uptake fit for every cell"),
    ("Uptake_kinetics_table",             "uptake time constants and amplitudes (table)"),
    ("Uptake_vs_mechanics",               "mechanics against dye uptake"),
    ("Mechanics_uptake_and_trap_position_3D", "mechanics, uptake and trap position together"),
    ("Fate_pie",                          "intact vs ruptured cells for one treatment and pulse"),
    ("Fate_per_trap",                     "intact vs ruptured per trap position"),
    ("Fate_counts_by_pulse",              "intact / ruptured counts per pulse (table)"),
    ("Cell_size_by_trap_position",        "cell size against trap position"),
    ("Trap_position_heatmap",             "parameter per trap position"),
    ("Trend_along_the_trap_row",          "trend along the row of traps (field gradient check)"),
    ("Parameter_correlations",            "pairwise correlations between parameters"),
    ("Traces_",                           "time series of every cell (long format)"),
    ("Mechanics.csv",                     "per-cell table used for the figures"),
    ("Attrition.csv",                     "attrition table used for the figures"),
    ("Manifest.csv",                      "list of values that could not be exported, with the reason"),
]


def pulse_folder(voltage, duration_label: str) -> str:
    """Folder name of one pulse condition, e.g. (100, '5ms') -> '100V_5ms'."""
    return f"{voltage}V_{duration_label}"


def safe_name(text: str) -> str:
    return re.sub(r'[^\w.\-]+', '_', str(text)).strip('_') or 'condition'


def build(results_dir: Union[str, Path], treatments: Iterable[str] = (),
          pulses: Iterable[Tuple[int, str]] = ()) -> Dict[str, object]:
    """
    Folders of one run plus the routing maps the plot suites take.

    treatments: aspiration-only treatments present (e.g. 'WT', 'CytD').
    pulses:     (voltage, duration_label) of the pulsed conditions present.
    Map keys are what the plot code looks up: treatment names, and duration
    labels ('5ms', '100us', ...) plus 'ASP' for the aspiration cohort.
    """
    root = Path(results_dir)
    out: Dict[str, object] = {'root': root}
    out.update({k: root / name for k, (name, _) in FOLDERS.items()})
    out['figure_data_parent'] = out['tables']

    out['asp_all'] = out['asp'] / ALL_TREATMENTS
    out['asp_by_treatment'] = {t: out['asp'] / safe_name(t) for t in sorted(set(treatments))}

    # duration label -> folder (several voltages with the same duration share it)
    by_dur: Dict[str, list] = {}
    for v, d in sorted(set(pulses)):
        by_dur.setdefault(d, []).append(v)
    names = {d: (pulse_folder(vs[0], d) if len(vs) == 1
                 else "_".join(f"{v}V" for v in vs) + f"_{d}") for d, vs in by_dur.items()}

    for key in ('pre', 'at', 'whole'):
        out[f'{key}_all'] = out[key] / ALL_CONDITIONS
        out[f'{key}_by_pulse'] = {d: out[key] / n for d, n in names.items()}
    out['pre_by_pulse'] = {'ASP': out['pre'] / ASPIRATION, **out['pre_by_pulse']}

    # Create every folder now: some plot functions write without creating
    # their folder. write_readme() removes the ones left empty at the end.
    folders = [out['tables']]
    for v in list(out.values()):
        if isinstance(v, dict):
            folders += list(v.values())
        elif isinstance(v, Path) and v != root:
            folders.append(v)
    for d in folders:
        d.mkdir(parents=True, exist_ok=True)
    return out


def table(out: Dict[str, object], key: str) -> Path:
    """Path of one of the result tables in 0_Tables/."""
    return Path(out['tables']) / TABLES[key][0]


def _note(name: str) -> str:
    for fname, desc in TABLES.values():
        if name == fname:
            return desc
    for candidate in (name, plain_name(name)):          # old names are described too
        low = candidate.lower()
        for prefix, desc in PLOT_NOTES:
            if low.startswith(prefix.lower()):
                return desc
    return ""


def remove_empty_folders(root: Union[str, Path]) -> None:
    """Remove folders a run wrote nothing to (deepest first; only empty ones)."""
    for d in sorted((p for p in Path(root).rglob('*') if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()


def write_readme(results_dir: Union[str, Path], source: Union[str, Path, None] = None,
                 controls=None) -> Path:
    """README.txt listing the folders and files of this run, with what they show."""
    root = Path(results_dir)
    remove_empty_folders(root)
    title = f"Bulk analysis run {root.name}"
    lines = [title, "=" * len(title), ""]
    if source is not None:
        lines += [f"Pooled from the per-experiment results in: {source}"]
    for c in controls or []:
        lines += [f"Aspiration-only controls (no pulse) added from: {c}"]
    if source is not None or controls:
        lines += [""]
    lines += [
        "Start with 0_Tables/mechanics_results_all_traps.csv (one row per trap) and",
        "0_Tables/attrition_per_experiment.csv (which cells were analysed and why others were not).",
        "Aspiration = no pulse; pulsed = electroporated. Pulse folders are named <voltage>_<duration>.",
        "Model-free = fits of the protrusion length L(t) without a material model (linear, power law);",
        "viscoelastic = Kelvin-Voigt / Jeffreys / Burgers fits. Uptake plateau = fitted amplitude A;",
        "uptake at 100 s = measured 100 s after the pulse.",
        "Each figure folder has single-condition sub-folders and an All_... folder with the comparisons.",
        "",
    ]
    for key, (name, desc) in FOLDERS.items():
        d = root / name
        if not d.is_dir():
            continue
        lines += [f"{name}/", f"    {desc}"]
        files = sorted(p for p in d.rglob('*') if p.is_file())
        for f in files:
            rel = f.relative_to(d).as_posix()
            lines.append(f"    {rel:<58} {_note(f.name)}".rstrip())
        if (d / "Figure_Data").is_dir():
            lines.append(f"    Figure_Data/: {FIGURE_DATA_NOTE}")
        lines.append("")
    path = root / "README.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
