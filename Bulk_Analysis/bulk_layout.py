# -*- coding: utf-8 -*-
"""
Folder layout of one bulk-analysis run (master_bulk_thesis.py).

Bulk_Analysis_results/<YYMMDD_HHMM>/
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

# File-name prefix -> what the figure shows (for README.txt). First match wins.
PLOT_NOTES = [
    ("Thesis_ASP_BestFit_Panel",          "viscoelastic best fit for every aspirated cell of one treatment"),
    ("Thesis_Boxplot",                    "fitted parameters per trap for one condition"),
    ("Thesis_Combined_Protrusion_Dynamics_ASP", "mean protrusion length over time, treatments overlaid"),
    ("Thesis_Combined_Protrusion_Dynamics_MI",  "mean protrusion length before the pulse, cohorts overlaid"),
    ("Thesis_ASP_Trap_Dependency",        "parameters against trap position (control)"),
    ("Thesis_ASP_BodyVolume_vs_E",        "cell body volume against stiffness E (control)"),
    ("Thesis_ASP_ProtLength_vs_Mechanics", "protrusion length against fitted parameters"),
    ("Thesis_ASP_Actin_vs_E",             "actin intensity against stiffness E"),
    ("Thesis_ASP_Actin",                  "actin baseline / F0 per treatment"),
    ("Thesis_MI_Boxplot",                 "model-independent parameters for one cohort"),
    ("Thesis_MI_PerTrap_Params",          "model-independent parameters per trap"),
    ("Thesis_MI_Parameter_Boxplots",      "model-independent parameters, cohorts compared"),
    ("Thesis_MI_WholeTrace_PerTrap",      "whole-trace parameters per trap"),
    ("Thesis_MI_WholeTrace_Parameter",    "whole-trace parameters, cohorts compared"),
    ("Thesis_MI_WholeTrace_Protrusion",   "mean protrusion length over the whole recording"),
    ("Thesis_MI_Whole_Mechanics_Corr",    "correlations between whole-trace parameters"),
    ("Thesis_PrePulse_Mechanics_Corr",    "correlations between pre-pulse parameters"),
    ("Thesis_Fate_PrePulse_Visco",        "pre-pulse viscoelastic parameters, intact vs ruptured"),
    ("Thesis_Fate_WholeTrace_MI",         "whole-trace parameters, intact vs ruptured"),
    ("Thesis_Combined_PrePulse_Trace_by_Fate",   "protrusion length before the pulse, intact vs ruptured"),
    ("Thesis_Combined_WholeTrace_Trace_by_Fate", "protrusion length over the recording, intact vs ruptured"),
    ("Thesis_EP_WholeTrace_by_Fate",      "protrusion length over the recording, intact vs ruptured"),
    ("Thesis_EP_Fate_MI_Boxplots",        "model-independent parameters, intact vs ruptured"),
    ("Thesis_Fate_Pie",                   "intact vs ruptured cells for one treatment and pulse"),
    ("Thesis_Fate_PerTrap",               "intact vs ruptured per trap position"),
    ("Thesis_Fate_Counts_by_Pulse",       "intact / ruptured counts per pulse"),
    ("Thesis_EP_Actin_Around_Pulse",      "actin intensity aligned to the pulse"),
    ("Thesis_EP_Protrusion_Around_Pulse", "protrusion length aligned to the pulse"),
    ("Thesis_EP_Actin_PrePost_Paired",    "actin before vs after the pulse, paired per cell"),
    ("Thesis_WT_MeanUptakeTrace",         "mean dye uptake over time, aspiration vs each pulse"),
    ("Thesis_Uptake_Mean_Trace",          "mean dye uptake over time (body / protrusion)"),
    ("Thesis_Uptake_Amplitude_Per_Trap",  "dye uptake amplitude per trap position"),
    ("Thesis_Uptake_BestFit_Panel",       "uptake fit for every cell"),
    ("Thesis_Uptake_Kinetics",            "uptake time constants and amplitudes"),
    ("Thesis_Uptake_vs_ProtLength",       "dye uptake against protrusion length"),
    ("Thesis_Uptake_Correlation_Sweep",   "uptake correlated with each mechanical parameter"),
    ("Thesis_Mechanics_vs_Uptake",        "mechanics against dye uptake"),
    ("Thesis_Spatial_Mech_Uptake_3D",     "mechanics, uptake and trap position together"),
    ("MI_WholeTrace_Fits_Panel",          "whole-trace fit for every cell of one condition"),
    ("Ch3_Data_Traces",                   "time series of every cell (long format)"),
    ("Ch3_Data_Mechanics",                "per-cell table used for the figures"),
    ("Ch3_Data_Attrition",                "attrition table used for the figures"),
    ("Ch3_Data_Manifest",                 "list of the figure-data files"),
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
    low = name.lower()
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


def write_readme(results_dir: Union[str, Path], source: Union[str, Path, None] = None) -> Path:
    """README.txt listing the folders and files of this run, with what they show."""
    root = Path(results_dir)
    remove_empty_folders(root)
    title = f"Bulk analysis run {root.name}"
    lines = [title, "=" * len(title), ""]
    if source is not None:
        lines += [f"Pooled from the per-experiment results in: {source}", ""]
    lines += [
        "Start with 0_Tables/mechanics_results_all_traps.csv (one row per trap) and",
        "0_Tables/attrition_per_experiment.csv (which cells were analysed and why others were not).",
        "ASP = aspiration only (no pulse); EP = electroporated. Pulse folders are named <voltage>_<duration>.",
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
