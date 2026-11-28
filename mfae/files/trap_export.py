# -*- coding: utf-8 -*-
"""
Developer export of a trap's images (Run_Info/Trap_images/Trap_XX_images.npz).

For developing the detection on a dataset whose raw file is too large to
share: the frames exactly as the analysis works on them (rotated, cropped to
the trap box, original bit depth, lossless), plus what is needed to rerun the
detection on them: the trap's setup (entrance, thresholds, smoothing), the
frame times, the pixel size and the image-processing settings.

Switched on per run with --export-traps 1,5,6 (or all), or permanently in
config/pipeline/defaults.yaml: workflow_settings.export_trap_images and
export_trap_images_traps. tools/export_trap_stacks.py writes the same files
afterwards from a finished analysis.

Reading one back:
    arrays, meta = trap_export.read("Run_Info/Trap_images/Trap_05_images.npz")
    arrays["membrane"]   # (frames, height, width), e.g. uint16
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


def wanted(params: Dict[str, Any], trap_number: int) -> bool:
    """Is the export on for this trap (1-based, as in the movies)?"""
    wf = params.get('workflow_settings', {}) or {}
    if not wf.get('export_trap_images', False):
        return False
    traps = wf.get('export_trap_images_traps') or []
    return not traps or int(trap_number) in {int(t) for t in traps}


def write(path: Union[str, Path], channels: Dict[str, Sequence[Optional[np.ndarray]]],
          time_s: Optional[Iterable[float]], meta: Dict[str, Any]) -> Path:
    """
    Save the frames of each channel (name -> list of 2-D frames; a missing
    frame becomes zeros) with the frame times and `meta` (JSON-able) to one
    compressed .npz. Returns the path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: Dict[str, np.ndarray] = {}
    for name, frames in channels.items():
        frames = list(frames or [])
        ref = next((f for f in frames if f is not None), None)
        if ref is None:
            continue
        ref = np.asarray(ref)
        out[name] = np.stack([np.asarray(f)[..., 0] if np.ndim(f) == 3 else np.asarray(f)
                              if f is not None else np.zeros(ref.shape[:2], ref.dtype) for f in frames])
    out['time_s'] = np.asarray(list(time_s) if time_s is not None else [], dtype=float)
    out['meta_json'] = np.array(json.dumps(meta, default=_jsonable))
    np.savez_compressed(path, **out)
    return path


def read(path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """(arrays: channel name -> frames, plus 'time_s'; meta) of a file written by write()."""
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files if k != 'meta_json'}
        meta = json.loads(str(d['meta_json'])) if 'meta_json' in d.files else {}
    return arrays, meta


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)
