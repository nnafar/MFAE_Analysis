# -*- coding: utf-8 -*-
"""
Folder layout of one bulk-analysis run (master_bulk_pipeline.py).

Bulk_Analysis_results/<YYMMDD_HHMM>/
    README.txt                 what every folder and file holds
    0_Tables/                  the numbers: one row per trap (all conditions)
    1_Per_Condition/<cond>/    checks per condition: every trap's fit, traces, distributions
    2_Mechanics/               mechanics compared across conditions (ASP vs EP)
    3_Dye_Uptake/              uptake time constants across conditions
    4_EP_Spatial/              electroporation: uptake vs trap position (field gradient)
    5_Correlations/            correlations between per-trap parameters
    6_Thesis_Figures/          the thesis figure set

<cond> is CellType_Treatment_<P>Pa_<V>V_<duration> (or ..._ASP for aspiration only).
"""

import re
from pathlib import Path
from typing import Dict, Union

FOLDERS = {
    # key:           (folder name,          description)
    'tables':        ("0_Tables",           "The numbers: one row per trap, all conditions."),
    'per_condition': ("1_Per_Condition",    "Per condition: every trap's fit and trace, so fits can be "
                                            "checked by eye, plus per-trap distributions."),
    'mechanics':     ("2_Mechanics",        "Mechanics compared across conditions (aspiration vs "
                                            "electroporation): model-independent fits, slopes, "
                                            "post-pulse behaviour."),
    'dye':           ("3_Dye_Uptake",       "Dye uptake time constants compared across conditions."),
    'spatial':       ("4_EP_Spatial",       "Electroporation only: uptake against trap position "
                                            "(field gradient), per experiment, and cell size as a confound."),
    'correlations':  ("5_Correlations",     "Spearman correlations and pair plots between per-trap parameters."),
    'thesis':        ("6_Thesis_Figures",   "The thesis figure set."),
}

TABLES = {
    'results':   ("mechanics_results.csv",
                  "one row per trap after pre-pulse QC: fitted mechanics, uptake fits, metadata"),
    'all_traps': ("mechanics_results_all_traps.csv",
                  "the same before QC (audit trail)"),
    'excluded':  ("excluded_traps_prepulse_QC.csv",
                  "traps removed because dye was already inside before the pulse (Pre_Leaky / Pre_Loaded)"),
}

# What the plot file names mean (prefix -> description), for README.txt
PLOT_NOTES = [
    ("Fit_Quality_Panel_",        "uptake fit for every trap of the condition"),
    ("Uptake_Traces_Panel_",      "uptake trace for every trap"),
    ("Visco",                     "viscoelastic fit for every trap"),
    ("MI_Fits_Panel_",            "model-independent fit for every trap"),
    ("Recoil_Fits_Panel_",        "recoil after the pulse, every trap (EP)"),
    ("Uptake_MeanSD_",            "mean +- SD uptake over time"),
    ("Uptake_Dist_BodyVsProt_",   "uptake per trap, body vs protrusion"),
    ("Protrusion_Dist_",          "maximum protrusion per trap and experiment"),
    ("Uptake_Exponential_",       "exponential fit to the mean uptake (time constant)"),
    ("Model_Independent_",        "model-independent parameters, ASP vs EP"),
    ("MI_Model_Selection",        "how often each model was preferred"),
    ("Slope_Comparison",          "protrusion growth slopes, ASP vs EP"),
    ("EP_PrePost_Slope",          "slope before vs after the pulse (EP)"),
    ("EP_PostPulse_Behavior_Counts", "number of cells per post-pulse behaviour"),
    ("EP_PostPulse_Behavior",     "what cells did after the pulse"),
    ("Uptake_Tau",                "uptake time constants per condition"),
    ("spatial_trap_gradient_",    "uptake against trap number (position in the field)"),
    ("spatial_heatmap_",          "uptake per trap x experiment"),
    ("spatial_cell_size_",        "cell size against trap position (confound check)"),
    ("Spearman",                  "Spearman correlation matrix between per-trap parameters"),
    ("Pairplot",                  "pairwise scatter plots of per-trap parameters"),
]


def build(results_dir: Union[str, Path]) -> Dict[str, Path]:
    """Folders of one run (only the run folder and 0_Tables are created here)."""
    root = Path(results_dir)
    out = {'root': root}
    out.update({k: root / name for k, (name, _) in FOLDERS.items()})
    root.mkdir(parents=True, exist_ok=True)
    out['tables'].mkdir(parents=True, exist_ok=True)
    return out


def ensure(out: Dict[str, Path], key: str) -> Path:
    out[key].mkdir(parents=True, exist_ok=True)
    return out[key]


def file(out: Dict[str, Path], key: str) -> Path:
    """Path of one of the result tables in 0_Tables/."""
    return ensure(out, 'tables') / TABLES[key][0]


def condition_dir(out: Dict[str, Path], cond_label: str) -> Path:
    """1_Per_Condition/<condition>/ (created)."""
    safe = re.sub(r'[^\w.\-]+', '_', str(cond_label)).strip('_') or 'condition'
    d = out['per_condition'] / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _note(name: str) -> str:
    for prefix, desc in PLOT_NOTES:
        if name.lower().startswith(prefix.lower().rstrip('_')):
            return desc
    for fname, desc in TABLES.values():
        if name == fname:
            return desc
    return ""


def write_readme(results_dir: Union[str, Path], source: Union[str, Path, None] = None) -> Path:
    """README.txt listing the folders and files of this run, with what they show."""
    root = Path(results_dir)
    lines = [f"Bulk analysis run {root.name}", "=" * (17 + len(root.name)), ""]
    if source is not None:
        lines += [f"Pooled from the per-experiment results in: {source}", ""]
    lines += ["Start with 0_Tables/mechanics_results.csv (one row per trap) and the",
              "cross-condition figures in 2_ to 5_. Use 1_Per_Condition/ to check fits by eye.",
              "Conditions are named CellType_Treatment_<P>Pa_<V>V_<duration> (..._ASP = aspiration only).",
              ""]
    for key, (name, desc) in FOLDERS.items():
        d = root / name
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()          # nothing was written there in this run
        if not d.is_dir():
            continue
        lines += [f"{name}/", f"    {desc}"]
        if key == 'per_condition':
            conds = sorted(p.name for p in d.iterdir() if p.is_dir())
            for c in conds:
                lines.append(f"    {c}/")
            kinds = sorted({re.sub(r'_?' + re.escape(c) + r'.*$', '', f.stem)
                            for c in conds for f in (d / c).iterdir() if f.is_file()})
            if kinds:
                lines.append("    each holds:")
                for k in kinds:
                    lines.append(f"        {k + '_*':<36} {_note(k)}".rstrip())
        else:
            for f in sorted(p for p in d.rglob('*') if p.is_file()):
                rel = f.relative_to(d).as_posix()
                lines.append(f"    {rel:<44} {_note(f.name)}".rstrip())
        lines.append("")
    path = root / "README.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
