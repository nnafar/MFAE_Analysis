# -*- coding: utf-8 -*-
"""
Your tags for the cells of one experiment: Run_Info/cell_annotations.csv.

One row per trap. The pipeline fills in which cell was analysed (Trap,
Analysed_cell, First_frame, Last_frame) and what it suspects (Pipeline_hint),
and keeps everything you typed (Tag, Note) when the experiment is analysed
again. Tags are set while reviewing the tracking movies (run_experiment.py
... --review) or typed in a spreadsheet; the bulk analysis reads them from
here.

Tags (case does not matter):

    OK          reviewed, nothing wrong             analysed
    R           ruptured after the pulse            analysed
    R0          ruptured before the pulse           excluded
    DOA, PI     not viable                          excluded
    R_DOA, R0_DOA                                   excluded (not viable)
    after, POST arrived after the pulse             excluded
    MASK, SHAPE, IRREGULAR  detection failed        excluded

An empty Tag means "not reviewed" and counts as intact, as before. The
pipeline never decides a tag: its automatic detections (rupture, entry not
confirmed, ...) only appear in Pipeline_hint.

If a re-analysis analyses a different cell in a trap than the one you
tagged, the tag is kept but the Check column says so, the log warns, and the
bulk analysis warns too, so a tag never silently moves to another cell.
Clear the Check text (or review the trap again) once you have confirmed.
"""

import csv
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)

FILE_NAME = "cell_annotations.csv"
COLUMNS = ["Trap", "Analysed_cell", "First_frame", "Last_frame", "Tag", "Note", "Pipeline_hint", "Check"]

# tag -> meaning; the bulk analysis maps them to fates (mfae/bulk/loader.py)
TAGS: Dict[str, str] = {
    "OK": "reviewed, nothing wrong",
    "R": "ruptured after the pulse",
    "R0": "ruptured before the pulse",
    "DOA": "not viable (dead on arrival)",
    "PI": "not viable (PI positive)",
    "R_DOA": "ruptured after the pulse, not viable",
    "R0_DOA": "ruptured before the pulse, not viable",
    "after": "arrived after the pulse",
    "POST": "arrived after the pulse",
    "MASK": "detection failed (mask)",
    "SHAPE": "detection failed (shape)",
    "IRREGULAR": "detection failed (irregular cell)",
}
_CANON = {k.lower(): k for k in TAGS}

# a new analysed cell counts as "different" when its first frame moved by more than this
CELL_CHANGE_FRAMES = 10


def path_for(results_dir: Union[str, Path]) -> Path:
    from mfae.files import output_layout
    return Path(results_dir) / output_layout.RUN_INFO / FILE_NAME


def normalise_tag(tag: Any) -> Optional[str]:
    """'r', ' R ', '_R' -> 'R'; '' / None -> ''; an unknown tag -> None."""
    if tag is None or (isinstance(tag, float) and tag != tag):
        return ""
    t = str(tag).strip().lstrip("_")
    if not t:
        return ""
    return _CANON.get(t.lower())


def read(results_dir: Union[str, Path]) -> List[Dict[str, str]]:
    """The rows of the table (empty list when there is none)."""
    p = path_for(results_dir)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Excel's BOM
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")   # Excel may save with ';'
        except csv.Error:
            dialect = csv.excel
        rows = []
        for r in csv.DictReader(f, dialect=dialect):
            r = {(k or "").strip(): ("" if v is None else str(v).strip()) for k, v in r.items()}
            if r.get("Trap", "").strip():
                rows.append(r)
    return rows


def write(results_dir: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> Path:
    """
    Write the table (sorted by trap). Raises PermissionError when the file is
    open in Excel (Windows locks it); callers say so.
    """
    p = path_for(results_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: _int(r.get("Trap")) or 0)
    extra = [k for r in rows for k in r if k not in COLUMNS]
    cols = COLUMNS + [k for k in dict.fromkeys(extra) if k]   # keep columns you added yourself
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: "" if r.get(c) is None else r.get(c) for c in cols})
    os.replace(tmp, p)
    return p


def tags(results_dir: Union[str, Path]) -> Dict[int, Dict[str, str]]:
    """{trap: row} for every trap with a tag, a note or a Check message."""
    out = {}
    for r in read(results_dir):
        t = _int(r.get("Trap"))
        if t is not None:
            out[t] = r
    return out


def hint(summary_row: Dict[str, Any]) -> str:
    """What the pipeline suspects about a trap (never used as a tag)."""
    h = []
    status = str(summary_row.get("Status", ""))
    if status and not status.startswith("analysed"):
        h.append(status)
    elif "fit failed" in status:
        h.append("fit failed")
    if summary_row.get("Rupture_detected"):
        rf = summary_row.get("Rupture_frame")
        why = summary_row.get("Rupture_reason")
        h.append("rupture?" + (f" frame {rf}" if rf else "") + (f" ({why})" if why else ""))
    if summary_row.get("Cell_present_at_pulse") is False:
        h.append("no cell at the pulse (cell before it analysed)")
    n = _int(summary_row.get("Cells_seen_in_trap"))
    if n and n > 1:
        h.append(f"{n} cells in trap")
    return "; ".join(h)


def update_from_summary(results_dir: Union[str, Path], summary_rows: List[Dict[str, Any]]) -> Optional[Path]:
    """
    After an analysis: one row per trap with the analysed cell and the hints,
    keeping every Tag / Note (and any column you added). Tags written in the
    protrusion trace file names (Trap_03_protrusion_trace_R.csv) are copied in
    once. Returns the path, or None when the file could not be written.
    """
    results_dir = Path(results_dir)
    old = {(_int(r.get("Trap"))): r for r in read(results_dir) if _int(r.get("Trap")) is not None}
    file_tags = _file_name_tags(results_dir)
    rows = []
    seen = set()
    for s in summary_rows:
        trap = _int(s.get("Trap"))
        if trap is None:
            continue
        seen.add(trap)
        prev = dict(old.get(trap, {}))
        row = {**prev, "Trap": trap, "Analysed_cell": _blank(s.get("Analysed_cell")),
               "First_frame": _blank(s.get("First_frame")), "Last_frame": _blank(s.get("Last_frame")),
               "Pipeline_hint": hint(s)}
        if not prev.get("Tag") and trap in file_tags:
            row["Tag"] = file_tags[trap]
            row["Note"] = (prev.get("Note") or "") or "tag copied from the file name"
        tagged = bool(prev.get("Tag"))
        if tagged and _changed(prev, row):
            was = f"{prev.get('First_frame') or '?'}-{prev.get('Last_frame') or '?'}"
            row["Check"] = f"CHECK: analysed cell changed (tag was set for frames {was})"
            logger.warning(f"Trap {trap}: the analysed cell changed since it was tagged "
                           f"'{prev.get('Tag')}' (was frames {was}, now {row['First_frame']}-"
                           f"{row['Last_frame']}); check the tag in {FILE_NAME}.")
        rows.append(row)
    rows += [r for t, r in old.items() if t not in seen]          # traps not in this run: kept
    try:
        p = write(results_dir, rows)
    except PermissionError:
        logger.warning(f"Could not update {path_for(results_dir)}: it is open in another program "
                       "(Excel?). Close it and run again; your tags in it are unchanged.")
        return None
    bad = [(r["Trap"], r["Tag"]) for r in rows if r.get("Tag") and normalise_tag(r["Tag"]) is None]
    for trap, tag in bad:
        logger.warning(f"Trap {trap}: unknown tag '{tag}' in {FILE_NAME} (known: {', '.join(TAGS)}).")
    return p


def set_tag(results_dir: Union[str, Path], trap: int, tag: str, note: Optional[str] = None) -> Path:
    """Set (or clear, tag='') one trap's tag; clears its Check message. Used by the review."""
    t = normalise_tag(tag)
    if t is None:
        raise ValueError(f"unknown tag '{tag}' (known: {', '.join(TAGS)})")
    rows = {(_int(r.get("Trap"))): r for r in read(results_dir) if _int(r.get("Trap")) is not None}
    row = rows.setdefault(int(trap), {"Trap": int(trap)})
    row["Tag"] = t
    row["Check"] = ""
    if note is not None:
        row["Note"] = note
    return write(results_dir, rows.values())


# --- helpers ---------------------------------------------------------------
def _int(v: Any) -> Optional[int]:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def _blank(v: Any) -> Any:
    return "" if v is None or (isinstance(v, float) and v != v) else v


def _changed(prev: Dict[str, Any], new: Dict[str, Any]) -> bool:
    a, b = _int(prev.get("First_frame")), _int(new.get("First_frame"))
    if a is None or b is None:
        return (a is None) != (b is None)
    return abs(a - b) > CELL_CHANGE_FRAMES


_FILE_TAG = re.compile(r"^Trap_(\d+)_protrusion_trace_(.+)\.csv$", re.IGNORECASE)


def _file_name_tags(results_dir: Path) -> Dict[int, str]:
    """Tags in Trap_XX_protrusion_trace_<TAG>.csv file names (the earlier way of tagging)."""
    from mfae.files import output_layout
    out = {}
    for folder in output_layout.full_csv_candidates(results_dir):
        if not Path(folder).is_dir():
            continue
        for f in Path(folder).glob("Trap_*_protrusion_trace_*.csv"):
            m = _FILE_TAG.match(f.name)
            if m and normalise_tag(m.group(2)):
                out.setdefault(int(m.group(1)), normalise_tag(m.group(2)))
    return out
