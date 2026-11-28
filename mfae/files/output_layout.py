# -*- coding: utf-8 -*-
"""
Folder layout and file names of one experiment's results (single source of truth).

<experiment>/
    README.txt                    what every folder and file holds
    0_Overview/                   trap_map.png, trap_summary.csv (one row per trap)
    1_Protrusion_and_Shape/
        Figures/                  All_traps_*.pdf (protrusion length, kymographs, area, solidity)
        Data/
            Traces/               Trap_XX_protrusion_trace.csv (length, areas, solidity per frame)
            Kymographs/           Trap_XX_membrane_kymograph.csv
        Movies/                   Trap_XX_tracking.avi
    2_Dye_Uptake/
        Figures/  Movies/         (..._dye_masks.avi)
        Data/Traces/              Trap_XX_dye_uptake.csv
        Data/Kymographs/          Trap_XX_dye_kymograph.csv
    3_Actin/
        Figures/  Movies/         (..._actin_zones.avi, ...)
        Data/Traces/              Trap_XX_actin.csv
        Data/Kymographs/          Trap_XX_actin_kymograph.csv
    4_Mechanics/                  Flow_conditions.csv/.pdf, Viscoelastic_fits.csv, per-trap fit plots
    Run_Info/                     effective_config.yaml, session_cache.json, analysis.log
        Trap_images/              Trap_XX_images.npz (developer export, only when switched on)

Everything that writes a result takes its folder from build()/complete() and
its file name from the helpers below, so the layout is defined only here.
Readers (bulk analysis) use the *_candidates helpers, which also find results
written with the older layouts (CSVs directly in Data/, "Protrusion Length
Detection/...", flat).
"""

import re
from pathlib import Path
from typing import Dict, List, Union

# ---------------------------------------------------------------------------
# Folder names
# ---------------------------------------------------------------------------
OVERVIEW = "0_Overview"
PROTRUSION_TOP = "1_Protrusion_and_Shape"
DYE_TOP = "2_Dye_Uptake"
ACTIN_TOP = "3_Actin"
MECHANICS = "4_Mechanics"
RUN_INFO = "Run_Info"
FIGURES, DATA, MOVIES = "Figures", "Data", "Movies"
TRACES, KYMOGRAPHS = "Traces", "Kymographs"     # sub-folders of each Data folder

# Older layout (kept for readers of earlier results)
OLD_PROTRUSION, OLD_FULL_CSV = "Protrusion Length Detection", "Full protrusion detection"
OLD_DYE, OLD_DYE_TRACES = "Dye Uptake", "Uptake traces"
OLD_ACTIN, OLD_ACTIN_TRACES = "Actin Quantification", "Actin traces"
OLD_FITTING = "Fitting results"

# key -> path relative to the experiment folder. The keys are what the
# pipeline code uses; figure keys ('protrusion', 'dye', 'actin', 'morphology',
# 'fitting') point at the Figures folders.
_PATHS = {
    'overview':      (OVERVIEW,),
    'run_info':      (RUN_INFO,),
    'protrusion':    (PROTRUSION_TOP, FIGURES),
    'morphology':    (PROTRUSION_TOP, FIGURES),
    'full_csv':      (PROTRUSION_TOP, DATA, TRACES),
    'membrane_kymo': (PROTRUSION_TOP, DATA, KYMOGRAPHS),
    'movies':        (PROTRUSION_TOP, MOVIES),
    'dye':           (DYE_TOP, FIGURES),
    'dye_traces':    (DYE_TOP, DATA, TRACES),
    'dye_kymo':      (DYE_TOP, DATA, KYMOGRAPHS),
    'dye_movies':    (DYE_TOP, MOVIES),
    'actin':         (ACTIN_TOP, FIGURES),
    'actin_traces':  (ACTIN_TOP, DATA, TRACES),
    'actin_kymo':    (ACTIN_TOP, DATA, KYMOGRAPHS),
    'actin_movies':  (ACTIN_TOP, MOVIES),
    'fitting':       (MECHANICS,),
    'trap_images':   (RUN_INFO, "Trap_images"),     # developer export (off by default)
}
# parent of each key when only some folders are given (older callers, tests)
_PARENT = {
    'full_csv': 'protrusion', 'membrane_kymo': 'protrusion', 'movies': 'protrusion',
    'dye_traces': 'dye', 'dye_kymo': 'dye', 'dye_movies': 'dye',
    'actin_traces': 'actin', 'actin_kymo': 'actin', 'actin_movies': 'actin',
}
_SIBLING_NAME = {'full_csv': (DATA, TRACES), 'membrane_kymo': (DATA, KYMOGRAPHS), 'movies': (MOVIES,),
                 'dye_traces': (DATA, TRACES), 'dye_kymo': (DATA, KYMOGRAPHS), 'dye_movies': (MOVIES,),
                 'actin_traces': (DATA, TRACES), 'actin_kymo': (DATA, KYMOGRAPHS),
                 'actin_movies': (MOVIES,)}


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
_TRAP_FILES = {
    # kind:              (template,                          description for README.txt)
    'protrusion_csv':    ("Trap_{n:02d}_protrusion_trace.csv",
                          "per frame: protrusion length, areas, volumes, body solidity, which cell"),
    'membrane_kymo_csv': ("Trap_{n:02d}_membrane_kymograph.csv",
                          "membrane intensity along the channel centre line over time"),
    'tracking_movie':    ("Trap_{n:02d}_tracking.avi",
                          "membrane image with the masks, the tracked tip and the length trace"),
    'dye_csv':           ("Trap_{n:02d}_dye_uptake.csv",
                          "per frame: mean dye intensity in body, protrusion and tip, mask sizes, F0"),
    'dye_kymo_csv':      ("Trap_{n:02d}_dye_kymograph.csv",
                          "dye intensity along the cell over time"),
    'dye_movie':         ("Trap_{n:02d}_dye_masks.avi",
                          "dye image with the same masks as the tracking movie"),
    'actin_csv':         ("Trap_{n:02d}_actin.csv",
                          "per frame: mean actin intensity per region/zone/cortex band, and F0"),
    'actin_kymo_csv':    ("Trap_{n:02d}_actin_kymograph.csv",
                          "actin intensity along the cell over time"),
    'actin_zones_movie': ("Trap_{n:02d}_actin_zones.avi",
                          "actin image with the tip/base/perinuclear/distal zones"),
    'actin_cortex_movie': ("Trap_{n:02d}_actin_cortex.avi",
                           "actin image with the cortex and lumen bands"),
    # optional single-trap figures (plotting_parameters.save_individual_trap_plots)
    'protrusion_fig':    ("Trap_{n:02d}_protrusion_length.pdf", "single-trap figure"),
    'membrane_kymo_fig': ("Trap_{n:02d}_membrane_kymograph.pdf", "single-trap figure"),
    'dye_kymo_fig':      ("Trap_{n:02d}_dye_kymograph.pdf", "single-trap figure"),
    'actin_kymo_fig':    ("Trap_{n:02d}_actin_kymograph.pdf", "single-trap figure"),
    'actin_zones_fig':   ("Trap_{n:02d}_actin_zones.pdf", "single-trap figure"),
    'fit_fig':           ("Trap_{n:02d}_viscoelastic_fit.pdf", "single-trap figure"),
    # developer export (workflow_settings.export_trap_images / --export-traps)
    'trap_images':       ("Trap_{n:02d}_images.npz",
                          "developer export: the trap's frames (rotated, cropped, original bit depth) "
                          "and its setup, for rerunning the detection elsewhere"),
}

# Experiment-level files: kind -> (layout key, file name, description)
EXPERIMENT_FILES = {
    'trap_map':            ('overview', "trap_map.png", "frame with every trap box and its number"),
    'trap_summary':        ('overview', "trap_summary.csv",
                            "one row per trap: which cell was analysed, entry/exit, pulse, flags"),
    'protrusion_all':      ('protrusion', "All_traps_protrusion_length.pdf",
                            "protrusion length over time, one panel per trap"),
    'membrane_kymo_all':   ('protrusion', "All_traps_membrane_kymographs.pdf",
                            "membrane kymograph with the tracked tip, one panel per trap"),
    'area_change':         ('morphology', "All_traps_area_change.pdf",
                            "change in protrusion / body / total area (before vs after pulse)"),
    'solidity_change':     ('morphology', "All_traps_body_solidity.pdf",
                            "cell body solidity (before vs after pulse)"),
    'dye_kymo_all':        ('dye', "All_traps_dye_kymographs.pdf",
                            "dye intensity along the cell over time, one panel per trap"),
    'dye_profiles_all':    ('dye', "All_traps_dye_profiles_along_cell.pdf",
                            "dye intensity along the cell, one line per time point"),
    'dye_uptake_all':      ('dye', "All_traps_dye_uptake.pdf",
                            "mean dye intensity over time in protrusion, body and whole cell"),
    'actin_kymo_all':      ('actin', "All_traps_actin_kymographs.pdf",
                            "actin intensity along the cell over time, one panel per trap"),
    'actin_zones_all':     ('actin', "All_traps_actin_zones.pdf",
                            "actin per zone (normalised to F0) and its rate of change"),
    'flow_csv':            ('fitting', "Flow_conditions.csv",
                            "channel flow: wall shear stress/rate, flow rate (Son 2007)"),
    'flow_card':           ('fitting', "Flow_conditions.pdf", "the same, as a summary card"),
    'fits_csv':            ('fitting', "Viscoelastic_fits.csv",
                            "fitted viscoelastic parameters per trap (when fitting is on)"),
    'effective_config':    ('run_info', "effective_config.yaml",
                            "every setting this run used; rerun with it to reproduce"),
    'session_cache':       ('run_info', "session_cache.json",
                            "trap boxes, entrances and thresholds from the interactive setup"),
    'log':                 ('run_info', "analysis.log", "full log of the run"),
}

FOLDER_DESCRIPTIONS = {
    OVERVIEW:       "Start here: where each trap is and what happened in it.",
    PROTRUSION_TOP: "Membrane channel: protrusion length into the channel and cell shape.",
    DYE_TOP:        "Dye channel: uptake into the cell, measured inside the membrane masks.",
    ACTIN_TOP:      "Actin channel: intensity per region, zone and cortex band.",
    MECHANICS:      "Flow conditions in the channel and viscoelastic fits.",
    RUN_INFO:       "How this result was made (settings, setup, log).",
}


def trap_file(kind: str, trap_number: int) -> str:
    """File name of a per-trap output, e.g. trap_file('dye_csv', 3) -> 'Trap_03_dye_uptake.csv'."""
    return _TRAP_FILES[kind][0].format(n=int(trap_number))


def experiment_file(dirs: Dict[str, Path], kind: str) -> Path:
    """Full path of an experiment-level output (see EXPERIMENT_FILES)."""
    key, name, _ = EXPERIMENT_FILES[kind]
    return Path(complete(dirs)[key]) / name


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
def build(results_dir: Union[str, Path]) -> Dict[str, Path]:
    """All output folders of one experiment (nothing is created on disk)."""
    root = Path(results_dir)
    dirs: Dict[str, Path] = {'root': root}
    dirs.update({k: root.joinpath(*parts) for k, parts in _PATHS.items()})
    dirs['tracking_visuals'] = dirs['protrusion']    # backwards-compatible alias
    return dirs


def complete(dirs: Dict[str, Path]) -> Dict[str, Path]:
    """
    Return a copy of `dirs` with every layout key present. A data/movie folder
    whose figure folder was given explicitly goes next to it (inside it, if
    that folder is not called 'Figures'); everything else comes from the
    standard layout under 'root'.
    """
    out = {k: Path(v) for k, v in dirs.items()}
    if 'protrusion' not in out and 'tracking_visuals' in out:
        out['protrusion'] = out['tracking_visuals']
    root = out.get('root')
    std = build(root) if root is not None else None
    for key in _PATHS:
        if key in out:
            continue
        parent = _PARENT.get(key)
        if parent and parent in out and (std is None or out[parent] != std[parent]):
            p = out[parent]
            out[key] = (p.parent if p.name == FIGURES else p).joinpath(*_SIBLING_NAME[key])
        elif std is not None:
            out[key] = std[key]
        else:
            out[key] = Path(*_PATHS[key])
    out.setdefault('tracking_visuals', out['protrusion'])
    return out


def ensure(dirs: Dict[str, Path], key: str) -> Path:
    """Create (if needed) and return one output folder."""
    p = Path(complete(dirs)[key])
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_cache_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    """session_cache.json: Run_Info/ first, then the experiment folder (older results)."""
    exp_dir = Path(exp_dir)
    return [exp_dir / RUN_INFO / "session_cache.json", exp_dir / "session_cache.json"]


# ---------------------------------------------------------------------------
# Reader helpers: current layout first, then the older ones.
# ---------------------------------------------------------------------------
def full_csv_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    exp_dir = Path(exp_dir)
    return [exp_dir / PROTRUSION_TOP / DATA / TRACES, exp_dir / PROTRUSION_TOP / DATA,
            exp_dir / OLD_PROTRUSION / OLD_FULL_CSV, exp_dir / OLD_FULL_CSV]


def dye_traces_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    exp_dir = Path(exp_dir)
    return [exp_dir / DYE_TOP / DATA / TRACES, exp_dir / DYE_TOP / DATA,
            exp_dir / OLD_DYE / OLD_DYE_TRACES, exp_dir / OLD_DYE]


def actin_traces_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    exp_dir = Path(exp_dir)
    return [exp_dir / ACTIN_TOP / DATA / TRACES, exp_dir / ACTIN_TOP / DATA,
            exp_dir / OLD_ACTIN / OLD_ACTIN_TRACES, exp_dir / OLD_ACTIN]


# Per-trap CSVs that earlier versions wrote directly into the Data folders
# (exact, untagged names only; they are rewritten by the analysis).
_FLAT_DATA_NAMES = ("protrusion_and_shape", "membrane_kymograph", "dye_uptake", "dye_kymograph",
                    "actin", "actin_kymograph")
_OLD_PROTRUSION_TAGGED = re.compile(r"^(Trap_\d{2})_protrusion_and_shape_(.+)\.csv$")
_TAGGED_TRACE = "Trap_{n:02d}_protrusion_trace_*.csv"


def protrusion_csv_name(folder: Union[str, Path], trap_number: int) -> str:
    """
    File name to write a trap's protrusion trace under: the hand-tagged name
    (e.g. Trap_03_protrusion_trace_R.csv) when one exists in `folder`, so a
    re-analysis keeps the tag and leaves one file per trap; else the plain name.
    """
    tagged = sorted(Path(folder).glob(_TAGGED_TRACE.format(n=int(trap_number)))) if Path(folder).is_dir() else []
    return tagged[0].name if tagged else trap_file('protrusion_csv', trap_number)


def remove_flat_data_files(results_dir: Union[str, Path]) -> int:
    """
    Tidy the Data folders written by an earlier version before re-analysing:
    hand-tagged protrusion files (Trap_03_protrusion_and_shape_R.csv) move to
    Data/Traces as Trap_03_protrusion_trace_R.csv (the tag is kept, and the
    analysis then writes into that file); the plain per-trap CSVs, which the
    analysis rewrites into Data/Traces and Data/Kymographs, are deleted.
    Any other file is left alone. Returns how many files were moved or deleted.
    """
    root, n = Path(results_dir), 0
    for top in (PROTRUSION_TOP, DYE_TOP, ACTIN_TOP):
        data = root / top / DATA
        if not data.is_dir():
            continue
        for f in sorted(data.iterdir()):
            if not f.is_file():
                continue
            m = _OLD_PROTRUSION_TAGGED.match(f.name)
            if top == PROTRUSION_TOP and m:
                dest = data / TRACES / f"{m.group(1)}_protrusion_trace_{m.group(2)}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                f.replace(dest)
                n += 1
            elif re.fullmatch(r"Trap_\d{2}_(%s)\.csv" % "|".join(_FLAT_DATA_NAMES), f.name):
                f.unlink()
                n += 1
    return n


def mechanics_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    exp_dir = Path(exp_dir)
    return [exp_dir / MECHANICS, exp_dir / OLD_FITTING]


def trap_file_candidates(kind: str, trap_number: int) -> List[str]:
    """Current file name first, then the names older versions wrote."""
    n = int(trap_number)
    old = {
        'protrusion_csv': [f"Trap_{n:02d}_protrusion_and_shape.csv", f"trap_{n:02d}_detection_full.csv"],
        'dye_csv':        [f"Trap_{n:02d}_Uptake_Data.csv", f"Trap_{n}_Uptake_Data.csv"],
        'actin_csv':      [f"Trap_{n:02d}_Actin_Data.csv", f"Trap_{n}_Actin_Data.csv"],
    }.get(kind, [])
    return [trap_file(kind, n)] + old


def first_existing(candidates: List[Path]) -> Path:
    """First folder that exists, else the first (current-layout) candidate."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# README.txt for each experiment
# ---------------------------------------------------------------------------
COLUMN_NOTES = {
    'trap_summary.csv': [
        ("Status", "analysed / no cell detected / entry not confirmed / error"),
        ("Cells_seen_in_trap", "cells that entered this trap during the recording"),
        ("Analysed_cell", "which of them was analysed (the one present at the pulse)"),
        ("First_frame, Last_frame", "frames of the analysed cell, counted from 1 (as in Fiji and the movies)"),
        ("Why_tracking_ended", "cell change / left ROI (tip reached the box edge) / end of recording"),
        ("Protrusion_at_pulse_um", "protrusion length at the pulse frame"),
    ],
    'protrusion_trace.csv': [
        ("Frame", "frame index counted from 0"),
        ("Protrusion_Length_um", "tip position to channel entrance; NaN outside the analysed cell"),
        ("Protrusion_Length_raw_um", "the same before the spike filter (and smoothing)"),
        ("Protrusion_Length_mask_um", "length from each frame's own mask (before kymograph tip tracking)"),
        ("Spike_Removed", "True where a short spike was replaced by the local median"),
        ("Protrusion/Body/Total_Area_um2", "mask areas; Volume_*_um3 assume a cylinder / sphere"),
        ("Body_Solidity", "body area / convex hull area (1 = convex)"),
        ("Tracked", "True for frames of the analysed cell"),
        ("Cell", "which cell (1, 2, ...) occupied the trap in this frame"),
    ],
    'dye_uptake.csv': [
        ("Body/Prot/Tip_Intensity", "mean raw dye intensity inside the membrane masks (a.u.)"),
        ("*_Mask_Count", "mask size in pixels"),
        ("F0_*", "baseline: mean over the frames just before the pulse (inside the cell's frames)"),
    ],
    'actin.csv': [
        ("Actin_*_Mean", "mean raw actin intensity per region, zone or cortex band (a.u.)"),
        ("F0_*", "baseline, as for the dye"),
    ],
    'kymograph.csv': [
        ("Time_s, Position_um, Intensity", "long format; Position_um is relative to the channel "
                                           "entrance (negative = into the channel)"),
    ],
}


def _describe(name: str) -> str:
    """Description of a file; per-trap files are passed without their 'Trap_XX' prefix."""
    for template, desc in _TRAP_FILES.values():
        if name == template[len("Trap_{n:02d}"):]:
            return desc
    for key, fname, desc in EXPERIMENT_FILES.values():
        if name == fname:
            return desc
    return ""


def write_readme(results_dir: Union[str, Path], experiment_id: str) -> Path:
    """Write <experiment>/README.txt describing the folders and files actually present."""
    root = Path(results_dir)
    lines = [
        f"Results of {experiment_id}",
        "=" * (11 + len(experiment_id)),
        "",
        "Start with 0_Overview/trap_summary.csv (what happened in each trap) and the",
        "All_traps_*.pdf figures. Per-trap numbers are in the Data folders, the movies",
        "show the masks the numbers were measured in (the same mask in every channel).",
        "",
    ]
    for folder in (OVERVIEW, PROTRUSION_TOP, DYE_TOP, ACTIN_TOP, MECHANICS, RUN_INFO):
        top = root / folder
        if not top.is_dir():
            continue
        lines += [f"{folder}/", f"    {FOLDER_DESCRIPTIONS[folder]}"]
        subs = [top] + sorted(p for p in top.rglob("*") if p.is_dir())
        for sub in subs:
            files = sorted(p.name for p in sub.iterdir() if p.is_file())
            if not files:
                continue
            indent = "    " if sub == top else "        "
            if sub != top:
                lines.append(f"    {sub.relative_to(top).as_posix()}/")
            seen = set()
            for name in files:
                shown = name
                if name.startswith("Trap_") and len(name) > 8 and name[5:7].isdigit():
                    shown = "Trap_XX" + name[7:]
                    if shown in seen:
                        continue
                seen.add(shown)
                desc = _describe(name if not shown.startswith("Trap_XX") else shown[7:])
                lines.append(f"{indent}{shown:<44} {desc}".rstrip())
        lines.append("")
    lines += ["Columns", "-------"]
    for table, notes in COLUMN_NOTES.items():
        lines.append(f"{table}")
        for col, note in notes:
            lines.append(f"    {col:<34} {note}")
    lines += ["", "Times are in seconds from the first frame; lengths in micrometres.",
              "Rerun exactly this analysis with Run_Info/effective_config.yaml.", ""]
    path = root / "README.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
