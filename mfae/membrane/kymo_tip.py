# -*- coding: utf-8 -*-
"""
Protrusion tip tracked on the membrane kymograph (centre-line profile x time).

Why
---
Measured frame by frame from the mask, the tip of a faint protrusion flickers:
in one frame the dim far end drops below the threshold (tip jumps towards the
entrance), in the next faint signal ahead of the tip is counted as cell (tip
jumps towards the box edge). On the kymograph the edge is still obvious to the
eye, because the eye follows it through time. This module does the same:

1. One profile per frame along the channel centre line (the kymograph rows),
   lightly smoothed over neighbouring frames.
2. Per frame and position an edge score: how well that position looks like
   the protrusion's OUTER edge (protrusion signal on the entrance side,
   background on the channel side). An internal step, such as a bright base
   next to a dim far end, scores low because the channel side is not
   background there.
3. The tip path through all frames that maximises the summed edge score minus
   a penalty per pixel of jump between consecutive frames (dynamic
   programming / Viterbi). A wrong edge in single frames, or in short runs,
   cannot pull the path away; a change that persists can.
4. Sub-pixel refinement on the frame's own profile, at the same level as the
   mask-based tip (tip_level_fraction of the protrusion contrast).

Coordinates: x = column in the ROI, the entrance at x = pipette_x, the
channel to the left (x < pipette_x). Tip x = leftmost protrusion column;
length = pipette_x - tip x.
"""

from typing import List, Optional, Tuple

import numpy as np


def centre_band_profile(band: np.ndarray, centre_row: float, half_height: float) -> np.ndarray:
    """Median over the rows centre_row +/- half_height of a (rows x columns) array."""
    rows = band.shape[0]
    r0 = int(max(0, np.floor(centre_row - half_height)))
    r1 = int(min(rows, np.ceil(centre_row + half_height) + 1))
    if r1 <= r0:
        r0, r1 = max(0, rows // 2 - 1), min(rows, rows // 2 + 2)
    return np.median(band[r0:r1, :], axis=0)


def _smooth_time(K: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along time (axis 0), NaN rows left out of the average."""
    if sigma <= 0:
        return K.copy()
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    valid = np.isfinite(K).all(axis=1)
    Kz = np.where(valid[:, None], K, 0.0)
    num = np.zeros_like(Kz)
    den = np.zeros(len(K))
    n = len(K)
    for k, wk in zip(range(-r, r + 1), w):
        lo, hi = max(0, -k), min(n, n - k)
        num[lo:hi] += wk * Kz[lo + k:hi + k]
        den[lo:hi] += wk * valid[lo + k:hi + k]
    out = np.full_like(K, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok, None]
    return out


def _box_mean(a: np.ndarray, lo_off: int, hi_off: int) -> np.ndarray:
    """mean(a[x+lo_off : x+hi_off]) for every x (columns outside the array are skipped)."""
    n = a.shape[-1]
    c = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, axis=-1)], axis=-1)
    x = np.arange(n)
    lo = np.clip(x + lo_off, 0, n)
    hi = np.clip(x + hi_off, 0, n)
    cnt = np.maximum(hi - lo, 1)
    return (c[..., hi] - c[..., lo]) / cnt


def track_tip(K: np.ndarray, pipette_x: int, active: np.ndarray,
              level_fraction: float = 0.25, min_contrast: float = 3.0,
              time_sigma: float = 0.0, jump_cost: float = 1.0,
              base_px: int = 12, ahead_weight: float = 10,
              jump_cap_px: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tip x per frame from the kymograph K (frames x columns, columns 0..pipette_x-1
    used; signal in noise units or any linear scale).

    active: frames to track (a protrusion is present, e.g. its mask exists);
    the path is found separately in each run of consecutive active frames.
    level_fraction: tip level, as in the mask-based refinement.
    min_contrast: protrusion base minus background (in K units) below which a
    frame has no usable edge; also the noise units a far end must reach to
    count as protrusion and a clear edge must step.
    jump_cost: penalty per pixel of tip movement between frames, in noise
    units of edge strength (1.0 -> a jump of jump_cap_px = 10 px or more costs
    as much as a 10-noise-unit edge; a bright edge is worth far more).
    time_sigma: optional Gaussian smoothing over frames (0 = off; smoothing
    leaks a moving tip into neighbouring frames, so it is off by default).

    Returns (tip_x, significant): tip_x is the sub-pixel tip column (NaN where
    not tracked); significant marks frames whose chosen edge is a clear one
    (enough contrast and an edge score of at least 0.3).
    """
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    W = int(min(pipette_x, K.shape[1]))
    tip = np.full(n, np.nan)
    significant = np.zeros(n, dtype=bool)
    if n == 0 or W < 2 * base_px:
        return tip, significant
    K = K[:, :W]
    Ks = _smooth_time(K, time_sigma)

    # per-frame background (channel ahead) and protrusion base level
    finite = np.isfinite(Ks).all(axis=1)
    bg = np.full(n, np.nan)
    obj = np.full(n, np.nan)
    obj[finite] = np.median(Ks[finite, W - base_px:W - 2], axis=1)
    # Frame noise (in K units) from pixel-to-pixel differences, which steps
    # barely affect; "signal" below means a few noise units above background.
    noise = np.ones(n)
    d = np.diff(Ks, axis=1)
    noise[finite] = np.maximum(1.4826 * np.median(np.abs(d[finite]), axis=1) / np.sqrt(2), 1e-6)
    # Background: the 20th percentile of the profile (the channel ahead of the
    # tip), raised by the 0.84 noise units a 20th percentile of noise sits below
    # its mean.
    bg[finite] = np.percentile(Ks[finite], 20, axis=1) + 0.84 * noise[finite]
    contrast = obj - bg
    usable = np.asarray(active, dtype=bool)[:n] & finite & (contrast >= min_contrast)

    Kn = (Ks - bg[:, None]) / np.where(contrast > 0, contrast, np.nan)[:, None]
    Kn = np.nan_to_num(Kn, nan=0.0)
    f = level_fraction
    # Beyond the box edge nothing is known: repeat the edge column there
    # instead of treating it as background (otherwise faint haze filling the
    # channel looks like a protrusion edge at the box edge).
    pad = 10
    Kp = np.pad(Kn, ((0, 0), (pad, 0)), mode='edge')
    inside = _box_mean(Kp, 0, 3)[:, pad:]            # columns x .. x+2  (protrusion side)
    outside = _box_mean(Kp, -3, 0)[:, pad:]          # columns x-3 .. x-1 (channel side)
    ahead = _box_mean(Kp, -9, -2)[:, pad:]           # further out in the channel
    inner = _box_mean(Kp, 1, 7)[:, pad:]
    snr_scale = (contrast / noise)[:, None]            # Kn -> noise units above bg (1 px)
    # Outer edge: a step from protrusion to channel, with no signal left ahead
    # of it (signal further out means an internal step, e.g. a bright base
    # next to a dim far end) and protrusion signal just inside it. Judged in
    # noise units, so a faint far end (several noise units) and faint glow
    # ahead of the tip (about background) are told apart even when both are
    # small next to the bright base.
    # averages over 7 / 6 px: their noise is sqrt(7) / sqrt(6) times lower
    ahead_snr = ahead * snr_scale * np.sqrt(7)
    inner_snr = inner * snr_scale * np.sqrt(6)
    # Everything in noise units: a bright edge is worth far more than any jump
    # penalty (followed however fast it moves), a faint one has to beat it
    # (so one-frame excursions are not taken).
    step_snr = (inside - outside) * snr_scale * np.sqrt(1.5)     # 3-px means on both sides
    score = step_snr \
        - ahead_weight * np.maximum(0.0, ahead_snr - 2.0) \
        - np.maximum(0.0, min_contrast - inner_snr)
    score[:, W - base_px:] = -1e3           # the base itself is not a tip

    xs = np.arange(W)
    # Jump penalty grows with distance up to jump_cap_px, then stays flat: a
    # real, lasting change (slip, a new cell, fast growth) pays it once and is
    # followed; a one-frame excursion pays it twice (out and back) for one
    # frame of gain, so it is not taken.
    jump = jump_cost * np.minimum(np.abs(xs[:, None] - xs[None, :]), jump_cap_px)   # [x, x_prev]

    t = 0
    while t < n:
        if not usable[t]:
            t += 1
            continue
        t1 = t
        while t1 < n and usable[t1]:
            t1 += 1
        # Viterbi over frames t .. t1-1
        acc = score[t].copy()
        back = np.zeros((t1 - t, W), dtype=np.int32)
        for k in range(t + 1, t1):
            cand = acc[None, :] - jump                      # [x, x_prev]
            back[k - t] = np.argmax(cand, axis=1)
            acc = cand[xs, back[k - t]] + score[k]
        x = int(np.argmax(acc))
        path = np.empty(t1 - t, dtype=np.int32)
        for k in range(t1 - 1, t - 1, -1):
            path[k - t] = x
            x = int(back[k - t][x]) if k > t else x
        for k, xk in zip(range(t, t1), path):
            tip[k] = _refine(K[k], bg[k], obj[k], xk, f)
            # a clear edge: protrusion signal just inside it (6-px mean at least
            # min_contrast noise units) and a step down to the channel
            significant[k] = (inner_snr[k, xk] >= min_contrast) and (step_snr[k, xk] >= 1.5)
        t = t1
    return tip, significant


def _refine(profile: np.ndarray, bg: float, obj: float, x: int, f: float, reach: int = 4) -> float:
    """Sub-pixel crossing of bg + f*(obj-bg) on the frame's own profile, nearest to x."""
    if not (np.isfinite(bg) and np.isfinite(obj)) or obj <= bg:
        return float(x)
    level = bg + f * (obj - bg)
    lo, hi = max(1, x - reach), min(len(profile) - 1, x + reach)
    best, best_d = float(x), None
    for j in range(lo, hi + 1):
        a, b = profile[j - 1], profile[j]          # outside (left) -> inside (right)
        if np.isfinite(a) and np.isfinite(b) and a < level <= b:
            pos = j - 1 + (level - a) / (b - a) if b != a else j - 0.5
            d = abs(pos - x)
            if best_d is None or d < best_d:
                best, best_d = pos, d              # same convention as the mask-based tip
    return float(best)


def lengths_from_tips(tip_x: np.ndarray, pipette_x: int) -> List[Optional[float]]:
    """Protrusion length in px per frame (None where not tracked)."""
    return [None if not np.isfinite(x) else float(pipette_x - x) for x in tip_x]
