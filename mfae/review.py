# -*- coding: utf-8 -*-
"""
Review the tracking movies and tag each trap's cell (run_experiment.py --review).

Plays 1_Protrusion_and_Shape/Movies/Trap_XX_tracking.avi of every trap, one
after another (from just before the analysed cell arrives, looping), and writes the tag you press into Run_Info/cell_annotations.csv
(see mfae/files/annotations.py) right away. It needs only the results folder,
not the config or the raw data.

Keys (shown in the window):
    R  ruptured after the pulse      0  ruptured before the pulse (R0)
    D  not viable (DOA)              P  not viable (PI)
    X  ruptured after the pulse and not viable (R_DOA)
    Z  ruptured before the pulse and not viable (R0_DOA)
    A  arrived after the pulse       M / S / I  detection failed (MASK / SHAPE / IRREGULAR)
    O  reviewed, nothing wrong (OK)  C  clear the tag
    A tag key saves and goes to the next trap.
    Enter / N  next trap (tag unchanged)     B  previous trap
    Space  pause / play     , .  one frame back / forward     [ ]  slower / faster
    Q / Esc  quit (every tag is already saved)
"""

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mfae.files import annotations, output_layout

logger = logging.getLogger(__name__)

KEY_TAGS = {'r': 'R', '0': 'R0', 'd': 'DOA', 'x': 'R_DOA', 'z': 'R0_DOA', 'p': 'PI', 'a': 'after', 'm': 'MASK',
            's': 'SHAPE', 'i': 'IRREGULAR', 'o': 'OK'}
HELP = ("R rupt  0 R0  D DOA  X R+DOA  Z R0+DOA  P PI  A after  M mask  S shape  I irregular  O ok  C clear | "
        "Enter next  B back  Space pause  ,/. step  [/] speed  Q quit")
WINDOW = "MFAE review"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Finding what to review
# --------------------------------------------------------------------------
def find_experiments(output_root: Path, only: Optional[str] = None) -> List[Path]:
    """Analysed experiments (results folders with tracking movies) under output_root, by name."""
    output_root = Path(output_root)
    found = set()
    for movie in output_root.glob(f"**/{output_layout.PROTRUSION_TOP}/{output_layout.MOVIES}/Trap_*_tracking.avi"):
        exp = movie.parent.parent.parent
        if "Bulk_Analysis_results" in exp.parts:
            continue
        if only and only.lower() not in exp.name.lower():
            continue
        found.add(exp)
    return sorted(found)


def trap_movies(results_dir: Path) -> Dict[int, Path]:
    movies = output_layout.complete(output_layout.build(results_dir))['movies']
    out = {}
    for p in sorted(Path(movies).glob("Trap_*_tracking.avi")):
        try:
            out[int(p.name.split("_")[1])] = p
        except (IndexError, ValueError):
            continue
    return out


def ensure_table(results_dir: Path) -> None:
    """Results analysed before the table existed: build it from trap_summary.csv (and file-name tags)."""
    if annotations.path_for(results_dir).exists():
        return
    summary = output_layout.experiment_file(output_layout.build(results_dir), 'trap_summary')
    rows = []
    if summary.exists():
        with open(summary, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r = {k: v for k, v in r.items()}
                for k in ('Rupture_detected', 'Cell_present_at_pulse'):
                    if k in r:
                        r[k] = {'true': True, 'false': False}.get(str(r[k]).strip().lower(), r[k] or None)
                rows.append(r)
    else:
        rows = [{'Trap': t} for t in trap_movies(results_dir)]
    annotations.update_from_summary(results_dir, rows)


# --------------------------------------------------------------------------
# The review state (no window code, so it can be tested)
# --------------------------------------------------------------------------
@dataclass
class Item:
    results_dir: Path
    trap: int
    movie: Path


@dataclass
class Review:
    items: List[Item]
    pos: int = 0
    done: bool = False
    message: str = ""
    message_until: float = 0.0
    saved: List[Tuple[str, int, str]] = field(default_factory=list)

    @property
    def item(self) -> Item:
        return self.items[self.pos]

    def row(self) -> Dict[str, str]:
        return annotations.tags(self.item.results_dir).get(self.item.trap, {})

    def say(self, text: str, seconds: float = 1.5) -> None:
        self.message, self.message_until = text, time.monotonic() + seconds

    def move(self, step: int) -> bool:
        """Go to another trap; returns True when the trap changed. Past the last one: done."""
        new = self.pos + step
        if new >= len(self.items):
            self.done = True
            return False
        if new < 0:
            self.say("This is the first trap.")
            return False
        self.pos = new
        return True

    def tag(self, tag: str, advance: bool = True) -> bool:
        """Save a tag ('' clears) for the current trap; advance on success."""
        it = self.item
        try:
            annotations.set_tag(it.results_dir, it.trap, tag)
        except PermissionError:
            self.say(f"Could not save: close {annotations.FILE_NAME} (open in Excel?), then press the key again.", 6)
            return False
        self.saved.append((it.results_dir.name, it.trap, tag))
        self.say(f"Saved: Trap {it.trap:02d} = {tag or '(cleared)'}")
        if advance:
            self.move(+1)
        return True

    def handle_key(self, key: int) -> str:
        """
        Apply one key press. Returns what the player has to do: 'next' (load
        the new trap's movie), 'pause', 'back1' / 'fwd1' (one frame), 'slower',
        'faster', 'quit' or '' (nothing).
        """
        if key < 0:
            return ''
        k = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ''
        if key in (27,) or k == 'q':
            self.done = True
            return 'quit'
        if k in KEY_TAGS:
            before = self.pos
            ok = self.tag(KEY_TAGS[k])
            return 'quit' if self.done else ('next' if ok and self.pos != before else '')
        if k == 'c':
            self.tag('', advance=False)
            return ''
        if key in (13, 10) or k == 'n':
            moved = self.move(+1)
            return 'quit' if self.done else ('next' if moved else '')
        if k == 'b' or key == 8:
            return 'next' if self.move(-1) else ''
        if k == ' ':
            return 'pause'
        if k == ',':
            return 'back1'
        if k == '.':
            return 'fwd1'
        if k == '[':
            return 'slower'
        if k == ']':
            return 'faster'
        return ''


def build_items(experiments: Sequence[Path], traps: Optional[Sequence[int]] = None) -> List[Item]:
    items = []
    for exp in experiments:
        ensure_table(exp)
        for t, movie in trap_movies(exp).items():
            if traps and t not in traps:
                continue
            items.append(Item(exp, t, movie))
    return items


def first_untagged(items: List[Item]) -> int:
    cache: Dict[Path, Dict[int, Dict[str, str]]] = {}
    for i, it in enumerate(items):
        rows = cache.setdefault(it.results_dir, annotations.tags(it.results_dir))
        r = rows.get(it.trap, {})
        if not r.get('Tag') or r.get('Check'):
            return i
    return 0


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
def _banner(rv: Review, width: int, paused: bool, speed: float) -> np.ndarray:
    it, r = rv.item, rv.row()
    strip = np.full((70, width, 3), 30, np.uint8)
    cell = (f"cell {r.get('Analysed_cell')}: frames {r.get('First_frame')}-{r.get('Last_frame')}"
            if r.get('First_frame') else "no analysed cell")
    tag = r.get('Tag') or '-'
    line1 = f"{it.results_dir.name[:46]} | Trap {it.trap:02d} ({rv.pos + 1}/{len(rv.items)}) | {cell}"
    cv2.putText(strip, line1, (8, 20), _FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(strip, f"Tag: {tag}", (8, 44), _FONT, 0.65,
                (90, 220, 90) if r.get('Tag') else (170, 170, 170), 2, cv2.LINE_AA)
    info = []
    if r.get('Pipeline_hint'):
        info.append(f"hint: {r['Pipeline_hint']}")
    if r.get('Check'):
        info.append(r['Check'])
    if paused:
        info.append("PAUSED")
    if abs(speed - 1.0) > 1e-6:
        info.append(f"speed x{speed:g}")
    cv2.putText(strip, " | ".join(info)[:120], (130, 44), _FONT, 0.45,
                (80, 200, 255) if r.get('Check') else (200, 200, 200), 1, cv2.LINE_AA)
    if rv.message and time.monotonic() < rv.message_until:
        cv2.putText(strip, rv.message[:130], (8, 64), _FONT, 0.45, (120, 220, 255), 1, cv2.LINE_AA)
    return strip


def _help(width: int) -> np.ndarray:
    strip = np.full((24, width, 3), 20, np.uint8)
    cv2.putText(strip, HELP, (8, 16), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


def run(experiments: Sequence[Path], traps: Optional[Sequence[int]] = None) -> int:
    """Review the given results folders. Returns 0 when done or quit, 1 when nothing to review."""
    items = build_items(experiments, traps)
    if not items:
        logger.error("No tracking movies to review (1_Protrusion_and_Shape/Movies/Trap_XX_tracking.avi). "
                     "Analyse the experiment first (workflow_settings.save_tracking_movies must be on).")
        return 1
    rv = Review(items, pos=0 if traps else first_untagged(items))
    logger.info(f"Reviewing {len(items)} trap(s) in {len(experiments)} experiment(s); starting at "
                f"{rv.item.results_dir.name} trap {rv.item.trap}. Tags go to Run_Info/{annotations.FILE_NAME}.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cap, n_frames, fps, frame_i, frame = None, 0, 10.0, 0, None
    paused, speed, load = False, 1.0, True
    try:
        while not rv.done:
            if load:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(str(rv.item.movie))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
                frame_i, frame, load, paused = 0, None, False, False
                # start just before the analysed cell arrives (frames count from 1)
                first = annotations._int(rv.row().get('First_frame'))
                if first and first > 11:
                    frame_i = min(first - 11, n_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
            if frame is None or not paused:
                ok, img = cap.read()
                if not ok:                                   # end of the movie: loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_i = 0
                    ok, img = cap.read()
                if ok:
                    frame, frame_i = img, frame_i + 1
            if frame is None:
                rv.say(f"Could not read {rv.item.movie.name}", 3)
                frame = np.zeros((200, 600, 3), np.uint8)
            w = frame.shape[1]
            cv2.imshow(WINDOW, np.vstack([_banner(rv, w, paused, speed), frame, _help(w)]))
            key = cv2.waitKey(max(1, int(1000 / (fps * speed))))
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break                                    # window closed
            except cv2.error:
                break
            action = rv.handle_key(key)
            if action == 'next':
                load = True
            elif action == 'pause':
                paused = not paused
            elif action in ('back1', 'fwd1'):
                paused = True
                target = max(0, frame_i - 2) if action == 'back1' else frame_i
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(target, n_frames - 1))
                frame_i = min(target, n_frames - 1)
                ok, img = cap.read()
                if ok:
                    frame, frame_i = img, frame_i + 1
            elif action == 'slower':
                speed = max(0.25, speed / 2)
            elif action == 'faster':
                speed = min(8.0, speed * 2)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    n = len(rv.saved)
    logger.info(f"Review finished: {n} tag(s) saved." + ("" if n else " Nothing changed."))
    for exp in dict.fromkeys(it.results_dir for it in items):
        tagged = sum(1 for r in annotations.read(exp) if r.get('Tag'))
        logger.info(f"  {exp.name}: {tagged} trap(s) tagged -> {annotations.path_for(exp)}")
    return 0
