# -*- coding: utf-8 -*-
"""
Per-trap tracking movie (1_Protrusion_and_Shape/Movies/Trap_XX_tracking.avi).

Each video frame shows
  * the membrane image, enlarged, with a FIXED per-trap contrast window
    (no frame-to-frame flicker, so real intensity changes stay visible),
  * outlines of the protrusion and cell-body masks actually used for the
    measurements, the pipette entrance, the tracked protrusion tip, and the
    rupture-monitoring box,
  * a status line (frame, time, length, tracking state, pulse, rupture),
  * the protrusion-length trace with a cursor at the current frame.

It reads only what the detection already produced (frame_masks,
protrusion_lengths, entry/exit/rupture indices), so it shows exactly what the
analysis measured.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

import Utils_MFA as utils

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _display_window(rois: Sequence[Optional[np.ndarray]]) -> tuple:
    """Fixed contrast window for the whole trap (1st / 99.8th percentile of a frame sample)."""
    valid = [r for r in rois if r is not None]
    if not valid:
        return 0.0, 1.0
    step = max(1, len(valid) // 40)
    sample = np.concatenate([np.asarray(r, dtype=np.float32).ravel() for r in valid[::step]])
    lo, hi = np.percentile(sample, (1.0, 99.8))
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _to_bgr(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.clip((np.asarray(img, dtype=np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if g.ndim == 3:
        g = g[..., 0]
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _draw_mask_outline(canvas: np.ndarray, mask: Optional[np.ndarray], scale: int,
                       color: tuple, fill_alpha: float = 0.18) -> None:
    if mask is None or not np.any(mask):
        return
    big = cv2.resize((mask > 0).astype(np.uint8) * 255, (canvas.shape[1], canvas.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    if fill_alpha > 0:
        layer = canvas.copy()
        layer[big > 0] = color
        cv2.addWeighted(layer, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)


def _trace_panel(lengths_um: np.ndarray, idx: int, width: int, height: int,
                 time_s: Sequence[float], marks: Dict[str, Optional[int]],
                 colors: Dict[str, tuple]) -> np.ndarray:
    panel = np.full((height, width, 3), 20, np.uint8)
    n = len(lengths_um)
    finite = lengths_um[np.isfinite(lengths_um)]
    y_max = max(float(finite.max()) * 1.1, 1.0) if finite.size else 1.0
    pad_l, pad_r, pad_t, pad_b = 52, 10, 10, 22
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def xy(i, v):
        x = pad_l + (i / max(n - 1, 1)) * pw
        y = pad_t + ph - (v / y_max) * ph
        return int(round(x)), int(round(y))

    # axes
    cv2.line(panel, (pad_l, pad_t), (pad_l, pad_t + ph), (110, 110, 110), 1)
    cv2.line(panel, (pad_l, pad_t + ph), (pad_l + pw, pad_t + ph), (110, 110, 110), 1)
    cv2.putText(panel, f"{y_max:.0f}", (4, pad_t + 10), _FONT, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, "0", (pad_l - 14, pad_t + ph), _FONT, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, "L [um]", (4, pad_t + ph // 2 + 4), _FONT, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    if time_s is not None and len(time_s) == n and n:
        cv2.putText(panel, f"{time_s[-1]:.0f} s", (pad_l + pw - 40, height - 6), _FONT, 0.4,
                    (170, 170, 170), 1, cv2.LINE_AA)

    # event markers
    for key, label in (('entry', 'entry'), ('exit', 'stop'), ('pulse', 'pulse'), ('rupture', 'rupture')):
        i_ev = marks.get(key)
        if i_ev is None or not (0 <= i_ev < n):
            continue
        x, _ = xy(i_ev, 0)
        cv2.line(panel, (x, pad_t), (x, pad_t + ph), colors[key], 1, cv2.LINE_AA)
        tw = cv2.getTextSize(label, _FONT, 0.35, 1)[0][0]
        tx = x + 3 if x + 3 + tw < width else x - 3 - tw   # keep label inside the panel
        cv2.putText(panel, label, (tx, pad_t + 10), _FONT, 0.35, colors[key], 1, cv2.LINE_AA)

    # full trace (dim) and trace up to the current frame (bright)
    def polyline(upto, color, thick):
        seg = []
        for i in range(min(upto + 1, n)):
            v = lengths_um[i]
            if np.isfinite(v):
                seg.append(xy(i, v))
            elif len(seg) > 1:
                cv2.polylines(panel, [np.array(seg, np.int32)], False, color, thick, cv2.LINE_AA)
                seg = []
            else:
                seg = []
        if len(seg) > 1:
            cv2.polylines(panel, [np.array(seg, np.int32)], False, color, thick, cv2.LINE_AA)

    polyline(n - 1, (85, 85, 85), 1)
    polyline(idx, colors['trace'], 2)
    x, _ = xy(idx, 0)
    cv2.line(panel, (x, pad_t), (x, pad_t + ph), (230, 230, 230), 1)
    if 0 <= idx < n and np.isfinite(lengths_um[idx]):
        cv2.circle(panel, xy(idx, lengths_um[idx]), 4, colors['trace'], -1, cv2.LINE_AA)
    return panel


def save_tracking_movie(rois: Sequence[Optional[np.ndarray]],
                        det_results: Dict[str, Any],
                        time_data: Sequence[float],
                        params: Dict[str, Any],
                        save_path: Path,
                        trap_number: int,
                        pulse_idx: Optional[int] = None,
                        fps: Optional[int] = None) -> Optional[Path]:
    """Write the tracking movie for one trap. Returns the path, or None if nothing was written."""
    if not rois or all(r is None for r in rois):
        return None
    wf = params.get('workflow_settings', {}) or {}
    fps = int(fps or wf.get('tracking_movie_fps', 10))
    target_w = int(wf.get('tracking_movie_width_px', 900))

    first = next(r for r in rois if r is not None)
    h0, w0 = first.shape[:2]
    scale = max(1, int(round(target_w / max(w0, 1))))
    img_w, img_h = w0 * scale, h0 * scale
    strip_h, trace_h = 46, 150
    frame_w, frame_h = img_w, img_h + strip_h + trace_h
    frame_w += frame_w % 2
    frame_h += frame_h % 2

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'MJPG'), fps,
                             (frame_w, frame_h), isColor=True)
    if not writer.isOpened():
        logger.warning(f"Trap {trap_number}: could not open video writer for {save_path.name}")
        return None

    sf = params.get('experiment_parameters', {}).get('scale_factor', 0.629)
    r_p = params.get('rupture_detection', {}) or {}
    rup_offset = int(r_p.get('rupture_offset_from_tip_px', 5))
    rup_width = int(r_p.get('rupture_window_width_px', 5))
    blanking = int(r_p.get('pulse_exit_blanking_frames', 5))
    margin_frac = float(params.get('image_processing', {}).get('wall_clip_margin', 0.25))

    pip_x = det_results.get('pipette_start_x_used')
    pip_x = int(pip_x) if pip_x is not None else None
    lengths_px = np.asarray(det_results.get('protrusion_lengths_px', []), dtype=float)
    lengths_um = np.asarray(det_results.get('protrusion_lengths_um', []), dtype=float)
    masks = det_results.get('frame_masks', []) or []
    n = len(rois)
    if lengths_um.size < n:
        lengths_um = np.r_[lengths_um, np.full(n - lengths_um.size, np.nan)]
        lengths_px = np.r_[lengths_px, np.full(n - lengths_px.size, np.nan)]

    entry = det_results.get('entry_frame_index')
    exit_ = det_results.get('exit_frame_index')
    rupture = det_results.get('rupture_frame_index') if det_results.get('rupture_detected') else None
    entry_ok = det_results.get('entry_confirmed', True)
    marks = {'entry': entry if entry_ok else None,
             'exit': exit_ if (exit_ is not None and det_results.get('exit_reason')) else None,
             'pulse': pulse_idx, 'rupture': rupture}

    # Fixed, clearly distinct colours (BGR): magenta protrusion, cyan body.
    c_prot = utils.MASK_OUTLINE_PROT
    c_body = utils.MASK_OUTLINE_BODY
    c_pip = utils.get_ui_color('pipette')
    colors = {'trace': (80, 200, 255), 'entry': (120, 220, 120), 'exit': (200, 200, 200),
              'pulse': (255, 200, 60), 'rupture': (60, 60, 255)}
    c_tip = (80, 200, 255)   # yellow-orange

    lo, hi = _display_window(rois)
    times = list(time_data) if time_data is not None else []

    for i, roi in enumerate(rois):
        frame = np.zeros((frame_h, frame_w, 3), np.uint8)
        if roi is not None:
            img = cv2.resize(_to_bgr(roi, lo, hi), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            mp, mb = masks[i] if i < len(masks) else (None, None)
            _draw_mask_outline(img, mb, scale, c_body)
            _draw_mask_outline(img, mp, scale, c_prot)

            margin = int(h0 * margin_frac) * scale
            if margin > 0:
                for y in (margin, img_h - margin):
                    cv2.line(img, (0, y), (img_w, y), (90, 90, 90), 1, cv2.LINE_AA)
            if pip_x is not None:
                cv2.line(img, (pip_x * scale, 0), (pip_x * scale, img_h), c_pip, 2)
                L = lengths_px[i] if i < lengths_px.size else np.nan
                if np.isfinite(L) and L > 0:
                    tip = int(round((pip_x - L) * scale))
                    cv2.line(img, (tip, margin), (tip, img_h - margin), c_tip, 2, cv2.LINE_AA)
                    x_end = int(round((pip_x - L - rup_offset) * scale))
                    x_start = max(0, x_end - rup_width * scale)
                    if x_end > x_start:
                        cv2.rectangle(img, (x_start, margin), (x_end, img_h - margin), (60, 60, 200), 1)
            frame[:img_h, :img_w] = img

        # --- status line ---
        t = times[i] if i < len(times) else float('nan')
        L_um = lengths_um[i] if i < lengths_um.size else np.nan
        segs = det_results.get('cell_segments') or []
        sel_cell = det_results.get('selected_cell')
        other = next((k + 1 for k, (e, x, _) in enumerate(segs)
                      if e <= i <= x and (k + 1) != sel_cell), None)
        if not entry_ok:
            state, s_col = "ENTRY NOT CONFIRMED", (60, 160, 255)
        elif other is not None:
            state, s_col = f"cell {other} (not analysed)", (170, 170, 170)
        elif entry is not None and i < entry:
            state, s_col = "pre-entry", (170, 170, 170)
        elif exit_ is not None and i > exit_:
            reason = det_results.get('exit_reason')
            state = {"left ROI": "tip left the ROI - tracking stopped",
                     "slip": "cell exited - tracking stopped",
                     "cell change": "cell left - tracking stopped"}.get(reason, "exited")
            s_col = (170, 170, 170)
        else:
            state = f"tracking cell {sel_cell}" if len(segs) > 1 else "tracking"
            s_col = (120, 220, 120)
        y0 = img_h + 18
        txt = f"Trap {trap_number:02d} | frame {i + 1}/{n} | t = {t:.1f} s | L = " + \
              (f"{L_um:.2f} um" if np.isfinite(L_um) else "--")
        cv2.putText(frame, txt, (8, y0), _FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        x_state = 8
        cv2.putText(frame, state, (x_state, y0 + 22), _FONT, 0.5, s_col, 1, cv2.LINE_AA)
        x_state += 12 + cv2.getTextSize(state, _FONT, 0.5, 1)[0][0]
        if pulse_idx is not None and pulse_idx - 1 <= i <= pulse_idx + blanking:
            cv2.putText(frame, "PULSE", (x_state, y0 + 22), _FONT, 0.5, colors['pulse'], 1, cv2.LINE_AA)
            x_state += 70
        if rupture is not None and i >= rupture:
            cv2.putText(frame, "RUPTURED", (x_state, y0 + 22), _FONT, 0.5, colors['rupture'], 1, cv2.LINE_AA)
            frame[:img_h, :5] = colors['rupture']
        legend = [("protrusion", c_prot), ("body", c_body), ("tip", c_tip)]
        x_leg = frame_w - 250
        for name, col in legend:
            cv2.rectangle(frame, (x_leg, y0 + 12), (x_leg + 12, y0 + 24), col, -1)
            cv2.putText(frame, name, (x_leg + 16, y0 + 23), _FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            x_leg += 24 + cv2.getTextSize(name, _FONT, 0.42, 1)[0][0]

        # --- length trace ---
        frame[img_h + strip_h:img_h + strip_h + trace_h, :img_w] = _trace_panel(
            lengths_um[:n], i, img_w, trace_h, times[:n] if len(times) >= n else None, marks, colors)
        writer.write(frame)

    writer.release()
    logger.info(f"Saved tracking movie: {save_path.name}")
    return save_path
