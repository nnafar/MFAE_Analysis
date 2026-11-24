# -*- coding: utf-8 -*-
"""
Folder layout of one experiment's results (single source of truth).

<experiment>/
    Protrusion Length Detection/            PDFs: detection traces, membrane kymographs
        Full protrusion detection/          trap_XX_detection_full.csv
        Kymographs/                         Trap_XX_Membrane_Kymograph.csv
        Movies/                             Trap_XX_Tracking.avi
    Dye Uptake/                             PDFs: uptake dashboards, kymographs, aggregates
        Uptake traces/                      Trap_XX_Uptake_Data.csv
        Uptake Kymographs/                  Trap_XX_Uptake_Kymograph.csv
        Uptake Movies/                      Trap_XX_Mask_Debug.avi
    Actin Quantification/                   PDFs: actin kymographs, zones
        Actin traces/                       Trap_XX_Actin_Data.csv
        Actin Kymographs/                   Trap_XX_Actin_Kymograph.csv
        Actin Movies/                       Trap_XX_Actin_Debug_Zones.avi / _Cortex.avi
    Cell Morphology/
    Fitting results/

Readers (bulk analysis) should use the *_CANDIDATES helpers so results
written with the older flat layout keep loading.
"""

from pathlib import Path
from typing import Dict, List, Union

PROTRUSION = "Protrusion Length Detection"
FULL_CSV = "Full protrusion detection"
MEMBRANE_KYMO = "Kymographs"
MOVIES = "Movies"

DYE = "Dye Uptake"
DYE_TRACES = "Uptake traces"
DYE_KYMO = "Uptake Kymographs"
DYE_MOVIES = "Uptake Movies"

ACTIN = "Actin Quantification"
ACTIN_TRACES = "Actin traces"
ACTIN_KYMO = "Actin Kymographs"
ACTIN_MOVIES = "Actin Movies"

MORPHOLOGY = "Cell Morphology"
FITTING = "Fitting results"

# key -> (parent key, folder name)
_CHILDREN = {
    'full_csv':      ('protrusion', FULL_CSV),
    'membrane_kymo': ('protrusion', MEMBRANE_KYMO),
    'movies':        ('protrusion', MOVIES),
    'dye_traces':    ('dye', DYE_TRACES),
    'dye_kymo':      ('dye', DYE_KYMO),
    'dye_movies':    ('dye', DYE_MOVIES),
    'actin_traces':  ('actin', ACTIN_TRACES),
    'actin_kymo':    ('actin', ACTIN_KYMO),
    'actin_movies':  ('actin', ACTIN_MOVIES),
}
_TOP = {'protrusion': PROTRUSION, 'dye': DYE, 'actin': ACTIN,
        'morphology': MORPHOLOGY, 'fitting': FITTING}


def build(results_dir: Union[str, Path]) -> Dict[str, Path]:
    """All output folders of one experiment (nothing is created on disk)."""
    root = Path(results_dir)
    dirs: Dict[str, Path] = {'root': root}
    dirs.update({k: root / name for k, name in _TOP.items()})
    return complete(dirs)


def complete(dirs: Dict[str, Path]) -> Dict[str, Path]:
    """
    Return a copy of `dirs` with every layout key present. Missing keys are
    derived from their parent folder, so callers that only pass the top-level
    folders (older code, tests) still get the full layout.
    """
    out = {k: Path(v) for k, v in dirs.items()}
    root = out.get('root')
    if 'protrusion' not in out:
        # 'tracking_visuals' was the previous name of the protrusion PDF folder.
        out['protrusion'] = out.get('tracking_visuals') or (root / PROTRUSION if root else Path(PROTRUSION))
    for key, name in _TOP.items():
        if key not in out:
            out[key] = (root / name) if root else Path(name)
    for key, (parent, name) in _CHILDREN.items():
        out.setdefault(key, out[parent] / name)
    out.setdefault('tracking_visuals', out['protrusion'])  # backwards-compatible alias
    return out


# ---------------------------------------------------------------------------
# Reader helpers: new location first, then the old flat layout.
# ---------------------------------------------------------------------------
def full_csv_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    exp_dir = Path(exp_dir)
    return [exp_dir / PROTRUSION / FULL_CSV, exp_dir / FULL_CSV]


def dye_traces_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    exp_dir = Path(exp_dir)
    return [exp_dir / DYE / DYE_TRACES, exp_dir / DYE]


def actin_traces_candidates(exp_dir: Union[str, Path]) -> List[Path]:
    exp_dir = Path(exp_dir)
    return [exp_dir / ACTIN / ACTIN_TRACES, exp_dir / ACTIN]


def first_existing(candidates: List[Path]) -> Path:
    """First folder that exists, else the first (new-layout) candidate."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]
