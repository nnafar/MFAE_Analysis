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


def _detector(params):
    from LineDetection_MFA import LineDetectionMFA
    return LineDetectionMFA([np.zeros((H, W), np.uint16)], [PIP, 0], params)


def _channel_scene(prot_left, wall_rows=2, prot_amp=380.0, line=False, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 400.0)
    img[22 - wall_rows:22, :PIP] = 120
    img[50:50 + wall_rows, :PIP] = 120
    if wall_rows > 4:  # large dark PDMS areas, as in many real traps
        img[:22 - wall_rows, :PIP] = 220
        img[50 + wall_rows:, :PIP] = 220
    prot = (xx >= prot_left) & (xx < PIP) & (np.abs(yy - 36) <= 6)
    sig = np.where(prot, prot_amp, 0.0)
    if line:
        sig[:, 2:4] += 200
    sig = cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2)
    img = img + sig + rng.normal(0, 25, (H, W))
    gt = prot.copy()
    gt[:18] = gt[H - 18:] = False
    return np.clip(img, 0, 65535).astype(np.uint16), gt


def test_noise_estimate_not_inflated_by_dark_walls():
    """Thick dark walls must not inflate the noise (dim cells looked like 2 sigma)."""
    params = _params()
    thin, _ = _channel_scene(60, wall_rows=2)
    thick, _ = _channel_scene(60, wall_rows=10)
    f = lambda im: cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.0)
    _, n_thin = utils.estimate_background_and_noise(f(thin))
    _, n_thick = utils.estimate_background_and_noise(f(thick))
    assert n_thick == pytest.approx(n_thin, rel=0.25)
    assert n_thin < 12  # true smoothed noise is ~7 for sigma=25 raw noise


@pytest.mark.parametrize("prot_left,line", [(0, False), (6, True)])
def test_protrusion_reaching_left_border_is_kept(prot_left, line):
    """A protrusion spanning the whole channel used to be deleted as a 'wall artefact'."""
    params = _params()
    img, gt = _channel_scene(prot_left, line=line)
    det = _detector(params)
    m = det._keep_closest_blob(det._segment_mask(img, 64, True, PIP), PIP, True)
    assert _iou(m, gt) > 0.6


def test_left_border_debris_still_removed():
    """Debris touching the left border but not the entrance is still removed."""
    params = _params()
    img, _ = _channel_scene(60)
    img = img.copy()
    img[30:42, 0:25] += 400  # bright blob at the left edge, far from the entrance
    det = _detector(params)
    m = det._segment_mask(img, 64, True, PIP)
    assert not m[:, :25].any()


def test_along_channel_smoothing_raises_faint_protrusion_snr():
    """Anisotropic smoothing must lift a faint protrusion well above the noise."""
    img, gt = _channel_scene(60, prot_amp=45.0)
    iso = _params(); iso['image_processing']['membrane_denoise_sigma_along'] = 1.0
    aniso = _params(); aniso['image_processing']['membrane_denoise_sigma_along'] = 3.0
    snr = lambda p: np.median(utils.enhance_membrane_image(img, p)[gt]) * utils.threshold_to_snr(1, p)
    assert snr(aniso) > 1.5 * snr(iso)


def test_smoothing_suggestion_follows_brightness():
    params = _params()
    bright, _ = _channel_scene(60, prot_amp=600.0)
    faint, _ = _channel_scene(60, prot_amp=30.0)
    s_bright = utils.suggest_membrane_smoothing(bright, PIP, params)
    s_faint = utils.suggest_membrane_smoothing(faint, PIP, params)
    assert s_bright == utils.MEMBRANE_SMOOTHING_LEVELS[0]
    assert s_faint > s_bright


def test_per_trap_smoothing_does_not_leak_between_traps():
    params = _params()
    det = _detector(params)
    det.params['image_processing']['membrane_denoise_sigma_along'] = 5.0
    assert 'membrane_denoise_sigma_along' not in params['image_processing']


def test_temporal_median_frame_skips_missing_and_keeps_input():
    a = np.full((4, 4), 10, np.uint16)
    b = np.full((4, 4), 100, np.uint16)
    c = np.full((4, 4), 20, np.uint16)
    frames = [a, b, c, None]
    assert np.all(utils.temporal_median_frame(frames, 1, 3) == 20)   # outlier frame b removed
    assert np.all(utils.temporal_median_frame(frames, 3, 3) == 20)   # None skipped
    assert b[0, 0] == 100


def test_temporal_window_steadies_faint_protrusion_length():
    """Faint protrusion: masks from a 3-frame median jitter less than per-frame masks."""
    from LineDetection_MFA import LineDetectionMFA
    rois = []
    for t in range(40):
        img, _ = _channel_scene(int(70 - t * 0.5), prot_amp=55.0, seed=100 + t)
        rois.append(img)

    def jitter(window):
        params = _params()
        params['image_processing'].update({'membrane_temporal_window': window,
                                           'membrane_denoise_sigma_along': 3.0,
                                           'protrusion_leak_fraction': 0.0})   # temporal effect alone
        params['rupture_detection'] = {'enable_smoothing': False}
        det = LineDetectionMFA(rois, [PIP, 0], params)
        det.run_detection_with_parameters(PIP, 40, threshold_body=40)
        L = np.array([det._measure_protrusion_from_mask(m[0], PIP)[0] if m[0] is not None else 0
                      for m in det.results['frame_masks']])
        return np.abs(np.diff(L)).mean()

    assert jitter(3) < 0.8 * jitter(1)


def test_tracking_stops_when_tip_leaves_roi():
    """Tip at the left ROI edge for 3+ frames: tracking stops before it; the next cell is not tracked."""
    from LineDetection_MFA import LineDetectionMFA
    yy, xx = np.mgrid[0:H, 0:W]
    lengths = [0] * 3 + list(range(20, 110, 10)) + [PIP] * 6 + [25, 30, 35, 40, 45]
    rois = []
    for k, L in enumerate(lengths):
        rng = np.random.default_rng(k)
        img = np.full((H, W), 400.0)
        sig = np.where(np.hypot(xx - PIP - 24, yy - 36) <= 23, 500.0, 0.0)
        if L:
            sig += np.where((xx >= PIP - L) & (xx < PIP) & (np.abs(yy - 36) <= 6), 500.0, 0.0)
        img = img + cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2) + rng.normal(0, 20, (H, W))
        rois.append(np.clip(img, 0, 65535).astype(np.uint16))
    params = _params()
    params['image_processing']['membrane_temporal_window'] = 1
    params['rupture_detection'] = {'enable_smoothing': False}
    params['experiment_parameters'] = {'scale_factor': 0.1}   # keep lengths below the clamp limit
    det = LineDetectionMFA(rois, [PIP, 0], params)
    res = det.run_detection_with_parameters(PIP, 100, threshold_body=100)
    first_edge = lengths.index(PIP)
    assert res['exit_reason'] == 'left ROI'
    assert res['exit_frame_index'] == first_edge - 1
    assert all(np.isnan(v) for v in res['protrusion_lengths_um'][first_edge:])
    assert np.isfinite(res['protrusion_lengths_um'][first_edge - 1])


def test_protrusion_mask_does_not_leak_into_faint_glow_ahead():
    """Faint glow in the channel ahead of the tip must not extend the protrusion to the box edge."""
    from LineDetection_MFA import LineDetectionMFA
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(1)
    L = 50
    sig = np.zeros((H, W))
    sig += np.where((xx >= PIP - L) & (xx < PIP) & (np.abs(yy - 36) <= 6), 150.0, 0.0)
    sig += np.where((xx < PIP - L - 2) & (np.abs(yy - 36) <= 7), 45.0, 0.0)      # glow, 30 %
    img = 300 + cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2) + rng.normal(0, 20, (H, W))
    img = np.clip(img, 0, 65535).astype(np.uint16)
    params = _params()
    det = LineDetectionMFA([img], [PIP, 0], params)
    thr = 45                                   # ~7 sigma: the glow is above it
    m = det._keep_closest_blob(det._segment_mask(img, thr, True, PIP), PIP, True)
    assert m[:, :3].any()                      # without the cut the mask reaches the edge
    cut = det._cut_protrusion_leak(m, utils.membrane_snr_image(img, params), PIP)
    assert not cut[:, :3].any()
    assert abs(det._measure_protrusion_from_mask(cut, PIP)[0] - L) <= 3


def _cell_sequence(lengths):
    """Frames with a cell body at the entrance and a protrusion of the given length [px]."""
    yy, xx = np.mgrid[0:H, 0:W]
    rois = []
    for k, L in enumerate(lengths):
        rng = np.random.default_rng(200 + k)
        sig = np.where(np.hypot(xx - PIP - 24, yy - 36) <= 23, 500.0, 0.0)
        if L:
            sig += np.where((xx >= PIP - L) & (xx < PIP) & (np.abs(yy - 36) <= 6), 500.0, 0.0)
        img = 400 + cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2) + rng.normal(0, 20, (H, W))
        rois.append(np.clip(img, 0, 65535).astype(np.uint16))
    return rois


def _run_cells(lengths, pulse_idx=None):
    from LineDetection_MFA import LineDetectionMFA
    params = _params()
    params['image_processing']['membrane_temporal_window'] = 1
    params['rupture_detection'] = {'enable_smoothing': False, 'pulse_frame_idx': pulse_idx}
    params['dye_uptake_parameters'] = {'has_pulse': pulse_idx is not None}
    params['experiment_parameters'] = {'scale_factor': 0.1}
    det = LineDetectionMFA(_cell_sequence(lengths), [PIP, 0], params)
    return det.run_detection_with_parameters(PIP, 100, threshold_body=100)


def test_pass_through_then_pulsed_second_cell_is_analysed():
    # cell 1 grows to the edge for 1 frame and passes through; cell 2 arrives and gets the pulse
    lengths = [0] * 3 + list(range(30, 110, 10)) + [PIP] + [20, 25, 30, 35, 40, 45, 50, 55, 60, 62, 64]
    second = lengths.index(PIP) + 1
    res = _run_cells(lengths, pulse_idx=len(lengths) - 4)
    assert len(res['cell_segments']) == 2
    assert res['selected_cell'] == 2 and res['cell_at_pulse'] is True
    assert second - 1 <= res['entry_frame_index'] <= second
    assert all(np.isnan(v) for v in res['protrusion_lengths_um'][:second - 1])   # cell 1 not measured
    assert np.isfinite(res['protrusion_lengths_um'][-1])


def test_cell_leaving_roi_then_new_cell_before_pulse():
    # like Trap 14: tip at the edge for several frames, empty trap, new cell present at the pulse
    lengths = [0] * 3 + list(range(30, 110, 10)) + [PIP] * 4 + [0] * 3 + [20, 30, 40, 45, 50, 52, 54, 56]
    res = _run_cells(lengths, pulse_idx=len(lengths) - 3)
    assert res['cell_segments'][0][2] == 'left ROI'
    assert res['selected_cell'] == 2 and res['cell_at_pulse'] is True


def test_without_pulse_the_longest_cell_is_analysed():
    lengths = [0] * 3 + list(range(30, 110, 10)) + [PIP] + [20, 25, 30, 35]
    res = _run_cells(lengths, pulse_idx=None)
    assert res['selected_cell'] == 1 and res['cell_at_pulse'] is None


def test_no_cell_at_pulse_keeps_last_cell_before_and_flags_it():
    # like Trap 12: cell leaves before the pulse, trap empty at the pulse
    lengths = [0] * 3 + list(range(30, 110, 10)) + [PIP] * 4 + [0] * 8
    res = _run_cells(lengths, pulse_idx=len(lengths) - 3)
    assert res['selected_cell'] == 1 and res['cell_at_pulse'] is False



def test_leak_cut_works_with_a_high_threshold_on_a_bright_protrusion():
    """High thresholds used to disable the cut (8-bit image saturates at 255)."""
    from LineDetection_MFA import LineDetectionMFA
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(2)
    L = 50
    sig = np.zeros((H, W))
    sig += np.where((xx >= PIP - L) & (xx < PIP) & (np.abs(yy - 36) <= 6), 1500.0, 0.0)   # ~200 sigma
    sig += np.where((xx < PIP - L - 2) & (np.abs(yy - 36) <= 7), 250.0, 0.0)              # glow ~17 %
    img = 300 + cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2) + rng.normal(0, 20, (H, W))
    img = np.clip(img, 0, 65535).astype(np.uint16)
    params = _params()
    det = LineDetectionMFA([img], [PIP, 0], params)
    thr = 130                                  # ~20 sigma, like Traps 1 and 3; glow is above it
    m = det._keep_closest_blob(det._segment_mask(img, thr, True, PIP), PIP, True)
    assert m[:, :3].any()
    cut = det._cut_protrusion_leak(m, utils.membrane_snr_image(img, params), PIP)
    assert not cut[:, :3].any()
    assert abs(det._measure_protrusion_from_mask(cut, PIP)[0] - L) <= 3


def test_tip_refinement_stays_near_the_mask_edge():
    """Faint signal ahead of the tip must not pull the refined tip far past the mask edge (Trap 6)."""
    params = _params()
    det = _detector(params)
    img = np.full((H, W), 40, np.uint8)
    edge = 60
    img[30:42, edge:PIP] = 200          # protrusion (mask starts at `edge`)
    img[30:42, edge - 24:edge] = 150    # faint band ahead of it, above 50 % of the protrusion
    img[30:42, edge - 26] = 40          # one dark column far ahead
    stat = np.array([edge, 30, PIP - edge, 12, (PIP - edge) * 12])
    refined = det._refine_edge_subpixel(img, edge, stat)
    assert refined == edge                    # crossing ~25 px out is not trusted -> mask edge


def test_thin_tip_inside_the_box_is_not_cut():
    """A narrow dim tip (Trap 15) is real protrusion; the leak cut only acts at the box edge."""
    from LineDetection_MFA import LineDetectionMFA
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(5)
    sig = np.where((xx >= PIP - 30) & (xx < PIP) & (np.abs(yy - 36) <= 8), 900.0, 0.0)   # wide bright base
    sig += np.where((xx >= PIP - 50) & (xx < PIP - 30) & (np.abs(yy - 36) <= 2), 250.0, 0.0)  # thin dim tip
    img = 300 + cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.0) + rng.normal(0, 15, (H, W))
    img = np.clip(img, 0, 65535).astype(np.uint16)
    params = _params()
    det = LineDetectionMFA([img], [PIP, 0], params)
    m = det._keep_closest_blob(det._segment_mask(img, 40, True, PIP), PIP, True)
    cut = det._cut_protrusion_leak(m, utils.membrane_snr_image(img, params), PIP)
    assert np.array_equal(cut, m)
    assert det._measure_protrusion_from_mask(cut, PIP)[0] >= 47


def test_cell_leaving_at_the_pulse_is_separated():
    """Trap 12: the cell leaves right at the pulse; the pulse window must not hide the cell change."""
    lengths = [0] * 3 + list(range(30, 100, 5)) + [100, 100] + [8] * 12 + [20, 30, 40, 45, 50, 52, 54]
    drop = lengths.index(8)
    res = _run_cells(lengths, pulse_idx=drop)          # the drop falls inside the pulse window
    assert len(res['cell_segments']) >= 2
    assert res['cell_segments'][0][1] < drop


def test_tip_refinement_extends_onto_a_dim_tip_within_the_limit():
    """A dim tip just beyond the mask edge: the half-maximum point (a few px out) is kept."""
    params = _params()
    det = _detector(params)
    img = np.full((H, W), 40, np.uint8)
    edge = 60
    img[30:42, edge:PIP] = 200
    for k in range(1, 6):                       # tip fading out over 5 px beyond the mask
        img[30:42, edge - k] = 200 - 32 * k
    stat = np.array([edge, 30, PIP - edge, 12, 0])
    refined = det._refine_edge_subpixel(img, edge, stat)
    assert edge - 5 <= refined < edge - 1

