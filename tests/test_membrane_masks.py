# -*- coding: utf-8 -*-
"""
Membrane-channel mask generation.

The membrane channel is only used to build masks, so it may be enhanced
freely; intensity analyses (actin, dye) must keep reading raw pixels.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).parent.parent))

import Utils_MFA as utils  # noqa: E402

H, W, PIP = 72, 216, 116


def _trap_image(gain: float = 1.0, tip_gain: float = 1.0, seed: int = 0):
    """Ring-stained cell body + protrusion in a channel with dark walls."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 400.0)
    for y0 in (22, 50):
        img[y0:y0 + 2, :PIP] = 150
    cx, cy, r = PIP + 24, 36, 23
    d = np.hypot(xx - cx, yy - cy)
    body = d <= r
    sig = np.where(body, 250 + 450 * np.exp(-((d - r + 2) ** 2) / 8), 0.0)
    prot = (xx >= PIP - 30) & (xx < PIP) & (np.abs(yy - 36) <= 6)
    sig += np.where(prot, 380, 0.0)
    tip = (np.hypot(xx - (PIP - 22), yy - 36) <= 5) & prot
    sig += np.where(tip, 380 * tip_gain, 0.0)
    sig = cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2)
    img = (img + sig) * gain + rng.normal(0, 20 * gain, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16), body, prot


def _params(enhance: bool = True):
    return {
        'image_processing': {
            'membrane_enhancement': enhance,
            'wall_clip_margin': 0.25,
            'clahe_clip_limit': 3.0,
            'clahe_tile_grid_size': [16, 16],
            'gaussian_kernel_size': [5, 5],
        },
        'guv_settings': {'enable': False},
    }


def _iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max((a | b).sum(), 1)


def test_enhancement_does_not_modify_input():
    img, _, _ = _trap_image()
    before = img.copy()
    utils.enhance_membrane_image(img, _params())
    utils.generate_dual_masks(img, PIP, 100, 100, _params())
    assert np.array_equal(img, before)


def test_body_mask_is_filled_and_stable_under_exposure_change():
    """Same threshold must give the same mask at full and half brightness."""
    params = _params()
    thr = 100  # ~7.8 sigma above background
    masks = []
    for gain in (1.0, 0.5):
        img, body, _ = _trap_image(gain=gain)
        _, mb = utils.generate_dual_masks(img, PIP, thr, thr, params)
        gt = body.copy()
        gt[:, :PIP] = False
        assert _iou(mb, gt) > 0.8, "ring-stained body should be filled"
        masks.append(mb)
    assert _iou(masks[0], masks[1]) > 0.9


def test_bright_protrusion_tip_does_not_shrink_body_mask():
    params = _params()
    _, mb_plain = utils.generate_dual_masks(_trap_image(tip_gain=1.0)[0], PIP, 100, 100, params)
    _, mb_tip = utils.generate_dual_masks(_trap_image(tip_gain=5.0)[0], PIP, 100, 100, params)
    assert _iou(mb_plain, mb_tip) > 0.9


def test_threshold_to_snr_mapping():
    params = _params()
    params['image_processing']['membrane_snr_full_scale'] = 25.5
    assert utils.threshold_to_snr(100, params) == pytest.approx(10.0)


def test_legacy_path_still_available():
    img, _, _ = _trap_image()
    out = utils.prepare_membrane_for_threshold(img, _params(enhance=False))
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_actin_uses_raw_pixels_when_sharing_membrane_channel():
    """Actin on the membrane channel: masks from the enhanced image, means from raw pixels."""
    from ActinQuantification import ActinAnalyzer

    img, _, _ = _trap_image()
    rois = [img] * 6
    params = _params()
    params.update({
        'experiment_parameters': {'scale_factor': 0.629},
        'dye_uptake_parameters': {'has_pulse': False, 'baseline_frames': 2},
        'actin_parameters': {},
    })
    mp, mb = utils.generate_dual_masks(img, PIP, 100, 100, params)
    analyzer = ActinAnalyzer(rois, rois, PIP, 100, 100, params,
                             frame_masks=[(mp, mb)] * 6, start_idx=0)
    res = analyzer.run(list(range(6)))

    raw_body_mean = float(img[mb > 0].mean())
    raw_prot_mean = float(img[mp > 0].mean())
    body_key, prot_key = 'actin_body_mean', 'actin_prot_mean'
    assert res[body_key][0] == pytest.approx(raw_body_mean, rel=1e-6)
    assert res[prot_key][0] == pytest.approx(raw_prot_mean, rel=1e-6)
    assert np.array_equal(rois[0], img)
