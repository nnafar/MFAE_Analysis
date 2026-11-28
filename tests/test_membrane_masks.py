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

from mfae import utils  # noqa: E402

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
    from mfae.quantification.actin import ActinAnalyzer

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
    from mfae.membrane.detection import LineDetectionMFA
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
    from mfae.membrane.detection import LineDetectionMFA
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
    from mfae.membrane.detection import LineDetectionMFA
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
    from mfae.membrane.detection import LineDetectionMFA
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


def _run_cells(lengths, pulse_idx=None, tip_tracking='mask'):
    from mfae.membrane.detection import LineDetectionMFA
    params = _params()
    params['image_processing']['membrane_temporal_window'] = 1
    params['image_processing']['tip_tracking'] = tip_tracking
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
    from mfae.membrane.detection import LineDetectionMFA
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
    assert abs(refined - edge) <= 1           # the band ahead is background: tip stays at the mask edge


def test_thin_tip_inside_the_box_is_not_cut():
    """A narrow dim tip (Trap 15) is real protrusion; the leak cut only acts at the box edge."""
    from mfae.membrane.detection import LineDetectionMFA
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



def test_tip_refinement_is_stable_on_a_dim_tip_in_a_lit_channel():
    """Trap 1: dim tip, faintly lit channel ahead, bright channel-end edge at the box edge.

    The old refinement put the half level inside the background noise (minimum of
    the window vs. the dim mask margin), so the tip jumped by up to 15 px between
    near-identical frames. The tip must now stay on the ramp in every noisy frame.
    """
    params = _params()
    det = _detector(params)
    edge = 30
    tips = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        img = np.full((H, W), 14.0)
        img[:, :2] = 7.0                            # dark box edge
        img[:, 3:5] = 24.0                          # channel-end edge
        ramp = np.clip((np.arange(W) - (edge - 2)) / 8.0, 0, 1)   # tip ramps up over 8 px
        img[30:42, :PIP] += (ramp[:PIP] * 50.0)
        img += rng.normal(0, 1.0, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        stat = np.array([edge + rng.integers(-2, 3), 30, PIP - edge, 12, 0])
        tips.append(det._refine_edge_subpixel(img, int(stat[0]), stat))
    tips = np.array(tips)
    assert tips.min() > edge - 4 and tips.max() < edge + 6
    assert tips.std() < 1.0


def _halo_image(seed: int = 0):
    """Bright protrusion (12 rows) with a wide out-of-focus halo at ~12 % of its brightness,
    a bright spot inside it, and a dim shaft next to the spot."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 400.0)
    core = (xx >= PIP - 60) & (xx < PIP) & (np.abs(yy - 36) <= 6)
    halo = (xx >= PIP - 66) & (xx < PIP) & (np.abs(yy - 36) <= 14)
    sig = np.where(halo, 150.0, 0.0) + np.where(core, 1100.0, 0.0)
    spot = (np.hypot(xx - (PIP - 40), yy - 36) <= 5)
    sig += np.where(spot, 2500.0, 0.0)
    sig = cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.0)
    img = img + sig + rng.normal(0, 15, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16), core


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_relative_level_drops_the_halo_around_a_bright_protrusion(seed):
    img, core = _halo_image(seed)
    thr = 20                                   # ~3 sigma: the halo (10 sigma) passes the noise test
    p_off = _params(); p_off['image_processing']['membrane_relative_level'] = 0.0
    p_on = _params(); p_on['image_processing']['membrane_relative_level'] = 0.3
    off = utils.threshold_membrane(img, thr, p_off)[:, :PIP] > 0
    on = utils.threshold_membrane(img, thr, p_on)[:, :PIP] > 0
    # width across the channel along the protrusion (away from its end, where only
    # halo is in the column and the tip refinement sets the length)
    cols = slice(PIP - 55, PIP - 5)
    core = core[:, cols]
    assert _iou(off[:, cols], core) < 0.6      # noise threshold alone: halo included
    assert _iou(on[:, cols], core) > 0.8       # relative level: mask on the protrusion itself
    # the dim shaft right next to the bright spot is kept (columns PIP-50 .. PIP-47)
    assert on[30:43, PIP - 50:PIP - 46].mean() > 0.9


def test_debug_movies_show_the_linedetection_mask_of_the_same_frame(tmp_path):
    """Dye and actin debug movies draw LineDetection's own mask for each frame, and
    frame i of the results (which start at the cell's entry) is frame start_idx + i."""
    from mfae.quantification.actin import ActinAnalyzer
    from mfae.quantification.dye_uptake import DyeUptakeAnalyzer

    img, _, _ = _trap_image()
    n, start = 8, 3
    rois = [img] * n
    params = _params()
    params.update({
        'experiment_parameters': {'scale_factor': 0.629},
        'dye_uptake_parameters': {'enable': True, 'has_pulse': False, 'baseline_frames': 2},
        'actin_parameters': {},
    })
    mp, mb = utils.generate_dual_masks(img, PIP, 100, 100, params)
    masks = [(mp.copy(), mb.copy()) for _ in range(n)]

    seen = {}
    for cls, args in ((ActinAnalyzer, (rois, rois, PIP, 100, 100, params)),
                      (DyeUptakeAnalyzer, (rois, rois, PIP, 100, 100, None, params))):
        a = cls(*args, frame_masks=masks, start_idx=start)
        a.run(list(range(n)))
        calls = []
        orig = a._get_masks

        def spy(k, m, _orig=orig, _calls=calls):
            out = _orig(k, m)
            assert out[0] is masks[k][0] and out[1] is masks[k][1]   # LineDetection's mask, not a new one
            _calls.append(k)
            return out
        a._get_masks = spy
        if cls is ActinAnalyzer:
            a.save_zones_debug_video(1, tmp_path)
            a.save_cortex_debug_video(1, tmp_path)
        else:
            a.save_debug_video(1, tmp_path)
        seen[cls.__name__] = calls

    assert set(seen['ActinAnalyzer']) == set(range(start, n))
    assert set(seen['DyeUptakeAnalyzer']) == set(range(start, n))


def test_actin_times_start_at_the_cell_entry():
    """Actin per-frame values start at start_idx; their time stamps must too (as for dye)."""
    from mfae.quantification.actin import ActinAnalyzer
    from mfae.quantification.dye_uptake import DyeUptakeAnalyzer

    img, _, _ = _trap_image()
    n, start = 8, 3
    rois = [img] * n
    params = _params()
    params.update({
        'experiment_parameters': {'scale_factor': 0.629},
        'dye_uptake_parameters': {'enable': True, 'has_pulse': False, 'baseline_frames': 2},
        'actin_parameters': {},
    })
    mp, mb = utils.generate_dual_masks(img, PIP, 100, 100, params)
    times = [10.0 * k for k in range(n)]
    act = ActinAnalyzer(rois, rois, PIP, 100, 100, params,
                        frame_masks=[(mp, mb)] * n, start_idx=start).run(times)
    dye = DyeUptakeAnalyzer(rois, rois, PIP, 100, 100, None, params,
                            frame_masks=[(mp, mb)] * n, start_idx=start).run(times)
    assert list(act['time_s']) == times[start:]
    assert list(act['time_s']) == list(dye['time_s'])


def test_downstream_never_resegments_a_frame_without_a_linedetection_mask(monkeypatch, tmp_path):
    """Dye and actin only use LineDetection's masks; a frame without one is skipped."""
    from mfae.quantification.actin import ActinAnalyzer
    from mfae.quantification.dye_uptake import DyeUptakeAnalyzer

    img, _, _ = _trap_image()
    n = 6
    rois = [img] * n
    params = _params()
    params.update({
        'experiment_parameters': {'scale_factor': 0.629},
        'dye_uptake_parameters': {'enable': True, 'has_pulse': False, 'baseline_frames': 2},
        'actin_parameters': {},
    })
    mp, mb = utils.generate_dual_masks(img, PIP, 100, 100, params)
    masks = [(mp, mb)] * n
    masks[4] = (None, None)                       # LineDetection had no mask here

    def boom(*a, **k):
        raise AssertionError("downstream module re-segmented the membrane image")
    monkeypatch.setattr(utils, 'generate_dual_masks', boom)
    monkeypatch.setattr(utils, 'threshold_membrane', boom)

    act = ActinAnalyzer(rois, rois, PIP, 100, 100, params, frame_masks=masks, start_idx=0)
    res_a = act.run(list(range(n)))
    act.save_zones_debug_video(1, tmp_path); act.save_cortex_debug_video(1, tmp_path)
    dye = DyeUptakeAnalyzer(rois, rois, PIP, 100, 100, None, params, frame_masks=masks, start_idx=0)
    res_d = dye.run(list(range(n)))
    dye.save_debug_video(1, tmp_path)
    assert res_d['mask_count_prot'][4] == 0 and res_d['mask_count_prot'][3] > 0
    assert len(res_a['time_s']) == n


def test_masks_exist_only_for_the_tracked_cell():
    """Two cells in one trap: only the selected (pulsed) cell's frames keep masks, so dye,
    actin and the movies never see the other cell."""
    lengths = [0] * 3 + list(range(30, 110, 10)) + [PIP] + [20, 25, 30, 35, 40, 45, 50, 55, 60, 62, 64]
    res = _run_cells(lengths, pulse_idx=len(lengths) - 4)
    e, x = res['entry_frame_index'], res['exit_frame_index']
    fm = res['frame_masks']
    assert all(fm[i] == (None, None) for i in range(len(fm)) if i < e or i > x)
    assert all(fm[i][0] is not None for i in range(e, x + 1))


def test_dye_and_actin_only_measure_the_tracked_cell(tmp_path):
    """Frames outside entry..exit (another cell, empty trap) are never measured or
    used for the baseline, even if they are bright."""
    from mfae.quantification.actin import ActinAnalyzer
    from mfae.quantification.dye_uptake import DyeUptakeAnalyzer

    img, _, _ = _trap_image()
    n, entry, exit_ = 12, 3, 7
    bright = np.full_like(img, 60000)                 # 'other cell' frames: would wreck F0
    # cell frames get +100 per frame, so which frames went into F0 is visible
    dye = [bright if (i < entry or i > exit_) else (img + 100 * i).astype(np.uint16) for i in range(n)]
    params = _params()
    params.update({
        'experiment_parameters': {'scale_factor': 0.629},
        'dye_uptake_parameters': {'enable': True, 'has_pulse': True, 'pulse_frame_idx': 10,
                                  'baseline_frames': 3},
        'actin_parameters': {},
    })
    mp, mb = utils.generate_dual_masks(img, PIP, 100, 100, params)
    masks = [((mp, mb) if entry <= i <= exit_ else (None, None)) for i in range(n)]
    times = [float(i) for i in range(n)]
    d = DyeUptakeAnalyzer([img] * n, dye, PIP, 100, 100, None, params, frame_masks=masks,
                          start_idx=entry, end_idx=exit_)
    rd = d.run(times)
    a = ActinAnalyzer([img] * n, dye, PIP, 100, 100, params, frame_masks=masks,
                      start_idx=entry, end_idx=exit_)
    ra = a.run(times)
    # cell left before the pulse (frame 10): baseline = its own last 3 frames, not frames 7-9
    assert d.f0['body'] == pytest.approx(float(img[mb > 0].mean()) + 600, rel=1e-6)   # frames 5, 6, 7
    assert list(rd['time_s']) == times[entry:exit_ + 1]
    assert list(ra['time_s']) == times[entry:exit_ + 1]
    assert max(rd['raw_mean_body']) < 30000 and max(ra['actin_body_mean']) < 30000


def test_baseline_window_stays_inside_the_cell():
    bw = utils.baseline_window
    assert bw(0, 100, 50, 5, True) == (45, 50)       # just before the pulse
    assert bw(48, 100, 50, 5, True) == (48, 50)      # entered just before the pulse
    assert bw(0, 20, 50, 5, True) == (15, 20)        # left before the pulse: its last frames
    assert bw(50, 100, 50, 5, True) == (50, 55)      # entered at the pulse: its first frames
    assert bw(10, 100, None, 5, False) == (10, 15)   # no pulse: first frames after entry


def _bright_cell_in_fluorescent_medium():
    """Dark PDMS (most pixels), dimly fluorescent medium, very bright cell."""
    import cv2
    rng = np.random.default_rng(0)
    h, w, pip = 90, 200, 120
    img = np.full((h, w), 100.0)
    img[30:60, :pip] = 300                      # channel lumen (medium)
    cv2.circle(img, (160, 45), 40, 300, -1)     # chamber medium
    cell = np.zeros((h, w), np.uint8)
    cv2.circle(cell, (160, 45), 28, 1, -1)
    cell[36:54, 80:pip + 5] = 1
    img[cell > 0] = 3000
    img = cv2.GaussianBlur(img, (0, 0), 2) + rng.normal(0, 10, img.shape)
    return img, cell > 0, pip


def test_full_scale_fitted_so_bright_cell_and_medium_do_not_clip():
    """With background = PDMS, medium sits at ~70 sigma and the cell at ~1000;
    a fixed 40-sigma scale clips both, so the mask swallowed the channel lumen."""
    import cv2
    img, cell, pip = _bright_cell_in_fluorescent_medium()
    p = _params()
    fs = utils.suggest_membrane_full_scale(img, p)
    assert fs > 40.0
    p['image_processing']['membrane_snr_full_scale'] = fs
    enh = utils.enhance_membrane_image(img, p)
    assert (enh[cell] == 255).mean() < 0.05
    otsu, _ = cv2.threshold(enh[:, pip:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = utils.threshold_membrane(img, int(otsu), p) > 0
    assert (mask & cell).sum() / (mask | cell).sum() > 0.95
    assert mask[32:34, 10:60].mean() == 0      # medium in the lumen is not cell


def test_full_scale_unchanged_for_dim_cells():
    rng = np.random.default_rng(1)
    img = 100 + rng.normal(0, 10, (60, 120))
    img[25:35, 20:100] += 40                    # ~14 sigma after smoothing
    assert utils.suggest_membrane_full_scale(img, _params()) == 40.0


def _dimming_sequence(n_bright, n_dim, L=60, far_level=60.0):
    """Trap 7: same cell throughout; after n_bright frames the far half of the
    protrusion dims below the threshold while the near third stays bright."""
    yy, xx = np.mgrid[0:H, 0:W]
    rois = []
    for k in range(n_bright + n_dim):
        rng = np.random.default_rng(300 + k)
        sig = np.where(np.hypot(xx - PIP - 24, yy - 36) <= 23, 500.0, 0.0)
        near = (xx >= PIP - L // 3) & (xx < PIP) & (np.abs(yy - 36) <= 6)
        far = (xx >= PIP - L) & (xx < PIP - L // 3) & (np.abs(yy - 36) <= 6)
        sig += np.where(near, 300.0, 0.0)
        sig += np.where(far, 300.0 if k < n_bright else far_level, 0.0)
        img = 400 + cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2) + rng.normal(0, 20, (H, W))
        rois.append(np.clip(img, 0, 65535).astype(np.uint16))
    return rois


def _run_rois(rois, tip_tracking='kymograph'):
    from mfae.membrane.detection import LineDetectionMFA
    params = _params()
    params['image_processing']['membrane_temporal_window'] = 1
    params['image_processing']['tip_tracking'] = tip_tracking
    params['rupture_detection'] = {'enable_smoothing': False, 'pulse_frame_idx': None}
    params['dye_uptake_parameters'] = {'has_pulse': False}
    params['experiment_parameters'] = {'scale_factor': 0.1}
    det = LineDetectionMFA(rois, [PIP, 0], params)
    return det.run_detection_with_parameters(PIP, 100, threshold_body=100)


def test_dimming_protrusion_end_is_not_a_cell_change():
    """Trap 7: the mask loses the dim far part of the protrusion; the cell never left."""
    # mask-based tip: the length halves, but the dimming check keeps tracking
    res = _run_rois(_dimming_sequence(12, 12), tip_tracking='mask')
    L = res['protrusion_lengths_um']
    assert np.nanmedian(L[14:]) < 0.6 * np.nanmedian(L[3:12])     # the mask did shorten
    assert len(res['cell_segments']) == 1
    assert res['exit_reason'] is None and res['exit_frame_index'] == 23
    assert res['dimmed_frames']


def test_kymograph_tip_keeps_the_dim_far_end():
    """Same cell, kymograph tracking: the dim far end still counts, the length stays."""
    res = _run_rois(_dimming_sequence(12, 12))
    L = np.asarray(res['protrusion_lengths_um'])
    mask = np.asarray(res['protrusion_lengths_mask_px']) * 0.1
    assert np.nanmedian(mask[14:]) < 0.6 * np.nanmedian(mask[3:12])   # the mask lost it ...
    assert abs(np.nanmedian(L[14:]) - np.nanmedian(L[3:12])) < 0.5   # ... the tracked tip did not
    assert len(res['cell_segments']) == 1 and res['exit_reason'] is None


def test_real_cell_change_still_detected_when_old_protrusion_is_gone():
    for method in ('kymograph', 'mask'):
        res = _run_rois(_dimming_sequence(12, 12, far_level=0.0), tip_tracking=method)
        assert res['cell_segments'][0][2] == 'cell change', method


def test_skip_pipette_opens_only_the_threshold_windows(monkeypatch):
    """--thresholds: the saved entrance is used, the entrance window never opens."""
    from mfae.membrane.detection import LineDetectionMFA
    det = LineDetectionMFA(_cell_sequence([40]), [PIP - 3, 0], _params())
    monkeypatch.setattr(det, "_interactive_pipette_positioning",
                        lambda img: pytest.fail("entrance window must not open"))
    windows = []
    monkeypatch.setattr(det, "_interactive_threshold_selection",
                        lambda img, pip, clip_walls, initial=None: windows.append((pip, clip_walls)) or ('confirm', 90))
    res = det.run_detection(skip_pipette=True)
    assert res['pipette_start_x_used'] == PIP - 3
    assert windows == [(PIP - 3, True), (PIP - 3, False)]
    assert res['threshold_prot'] == 90 and res['threshold_body'] == 90



def test_back_steps_through_the_windows_of_a_trap(monkeypatch):
    """B in the body window reopens the protrusion window at the value chosen there;
    B in a trap's first window hands 'back' to the caller (previous trap)."""
    from mfae.membrane.detection import LineDetectionMFA
    det = LineDetectionMFA(_cell_sequence([40]), [PIP, 0], _params())
    script = iter([('confirm', PIP - 2),            # entrance
                   ('confirm', 80),                  # protrusion
                   ('back', None),                   # body -> back to protrusion
                   ('confirm', 85),                  # protrusion again
                   ('confirm', 120)])                # body
    seen = []
    monkeypatch.setattr(det, "_interactive_pipette_positioning", lambda img: next(script))
    monkeypatch.setattr(det, "_interactive_threshold_selection",
                        lambda img, pip, clip_walls, initial=None: seen.append((clip_walls, initial)) or next(script))
    res = det.run_detection()
    assert seen == [(True, None), (False, None), (True, 80), (False, None)]
    assert (res['pipette_start_x_used'], res['threshold_prot'], res['threshold_body']) == (PIP - 2, 85, 120)

    det2 = LineDetectionMFA(_cell_sequence([40]), [PIP, 0], _params())
    monkeypatch.setattr(det2, "_interactive_threshold_selection",
                        lambda img, pip, clip_walls, initial=None: ('back', None))
    assert det2.run_detection(skip_pipette=True)['signal'] == 'back'


def test_spike_filter_removes_short_jumps_and_keeps_real_changes():
    from mfae.membrane.detection import LineDetectionMFA
    f = LineDetectionMFA._remove_length_spikes
    rng = np.random.default_rng(0)
    base = list(60 + rng.normal(0, 1.0, 40))                  # steady tip, 1 px jitter
    trace = list(base)
    trace[10] = 20                                             # lost the faint far end
    trace[20] = trace[21] = 110                                # jumped onto signal ahead (2 frames)
    trace[30:33] = [15, 15, 15]                                # 3-frame run (= window // 2)
    out, flags = f(trace, window=7, min_jump_px=6)
    assert [i for i, s in enumerate(flags) if s] == [10, 20, 21, 30, 31, 32]
    assert all(abs(out[i] - 60) < 3 for i in (10, 20, 21, 30, 31, 32))
    assert all(out[i] == trace[i] for i in range(40) if not flags[i])   # jitter untouched

    step = [60.0] * 15 + [20.0] * 15                           # real, lasting change (slip)
    out, flags = f(step, window=7, min_jump_px=6)
    assert out == step and not any(flags)                      # kept, at its exact frame

    ramp = [30 + 2.0 * i for i in range(30)]                   # steady growth
    assert f(ramp, window=7, min_jump_px=6)[0] == ramp
    assert f(trace, window=0)[0] == trace                      # 0 = off


def test_single_frame_spikes_do_not_end_tracking():
    """Faint cells: one frame at the box edge or one frame of a short mask used to
    count towards 'left ROI' / 'cell change'; now they are spikes."""
    L = 60
    lengths = [0] * 3 + [L] * 30
    lengths[12] = PIP                     # one frame: mask leaks to the box edge
    lengths[13] = PIP
    lengths[20] = 20                      # one frame: far end lost
    res = _run_cells(lengths)
    assert len(res['cell_segments']) == 1 and res['exit_reason'] is None
    um = np.asarray(res['protrusion_lengths_um'])
    assert np.nanmax(um[4:]) < (L + 6) * 0.1 and np.nanmin(um[4:]) > (L - 6) * 0.1
    assert res['spike_removed'][12] and res['spike_removed'][13] and res['spike_removed'][20]


def _flicker_sequence(far=90.0, n=60, seed=0):
    """Slowly growing protrusion whose far half sits at the threshold: the mask flickers."""
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(seed)
    rois, truth = [], []
    for k in range(n):
        L = int(round(50 + 20 * k / n))
        truth.append(L)
        sig = np.where(np.hypot(xx - PIP - 24, yy - 36) <= 23, 500.0, 0.0)
        sig += np.where((xx >= PIP - L // 2) & (xx < PIP) & (np.abs(yy - 36) <= 6), 400.0, 0.0)
        sig += np.where((xx >= PIP - L) & (xx < PIP - L // 2) & (np.abs(yy - 36) <= 5), far, 0.0)
        img = 400 + cv2.GaussianBlur(sig.astype(np.float32), (0, 0), 1.2) + rng.normal(0, 20, (H, W))
        rois.append(np.clip(img, 0, 65535).astype(np.uint16))
    return rois, np.array(truth, float)


def test_kymograph_tip_does_not_flicker_on_a_faint_far_end():
    from mfae.membrane.detection import LineDetectionMFA
    rois, truth = _flicker_sequence()
    jumps = {}
    for method in ('mask', 'kymograph'):
        p = _params()
        p['image_processing'].update(membrane_temporal_window=1, tip_tracking=method)
        p['rupture_detection'] = {'enable_smoothing': False, 'pulse_frame_idx': None,
                                  'spike_filter_window': 0}       # the tracker alone
        p['dye_uptake_parameters'] = {'has_pulse': False}
        p['experiment_parameters'] = {'scale_factor': 0.1}
        r = LineDetectionMFA(rois, [PIP, 0], p).run_detection_with_parameters(PIP, 100, threshold_body=100)
        L = np.array(r['protrusion_lengths_raw_px'], float)
        jumps[method] = np.mean(np.abs(np.diff(L)) > 5)
        if method == 'kymograph':
            assert np.median(np.abs(L - truth)) < 2 and np.mean(np.abs(L - truth) > 5) < 0.05
    assert jumps['mask'] > 0.2 and jumps['kymograph'] < 0.03


def test_kymograph_tip_follows_fast_growth(monkeypatch):
    """Fast entry (10 px per frame) must be followed, not smoothed away."""
    lengths = [0] * 3 + list(range(20, 110, 10)) + [100] * 6
    res = _run_cells(lengths, tip_tracking='kymograph')
    L = np.array(res['protrusion_lengths_raw_px'], float)
    assert np.all(np.abs(L[3:12] - np.array(lengths[3:12])) <= 3)



def test_kymograph_tip_does_not_jump_to_the_box_edge_in_a_hazy_channel():
    """Exp 2 trap 15: faint haze filling the channel must not look like an edge at the box edge."""
    from mfae.membrane.kymo_tip import track_tip
    rng = np.random.default_rng(3)
    n, Wd, tip = 40, 120, 70
    x = np.arange(Wd)
    # faint protrusion; the channel ahead is dark near the tip and hazier
    # towards the far end of the box
    prof = np.where(x >= tip, 14.0, np.where(x < 25, 9.0, 0.0))
    K = np.array([prof + rng.normal(0, 1.0, Wd) for _ in range(n)])
    tips, _ = track_tip(K, Wd, np.ones(n, bool))
    assert np.all(np.abs(tips - tip) <= 2)



def test_kymograph_tip_at_the_box_edge_alone_does_not_end_tracking(monkeypatch):
    """Exp 2 trap 5: the tracker put the tip at the box edge on haze; 'left ROI'
    ended tracking at frame 0. The mask must reach the edge too."""
    from mfae.membrane import kymo_tip
    real = kymo_tip.track_tip

    def edge_for_a_while(K, pipette_x, active, **kw):
        tips, clear = real(K, pipette_x, active, **kw)
        tips[12:18] = 0.0                     # false edge at the box edge, 6 frames, mid-cell
        return tips, clear

    monkeypatch.setattr(kymo_tip, 'track_tip', edge_for_a_while)
    res = _run_cells([0] * 3 + [60] * 25, tip_tracking='kymograph')
    assert len(res['cell_segments']) == 1, res['cell_segments']     # not split at the false edge
    assert res['cell_segments'][0][2] is None and res['exit_frame_index'] == 27


def test_kymograph_ignores_the_channel_end_line():
    """Real Exp 2 kymographs: a narrow, steady bright line at the far end of the
    channel (next to the box edge) pulled the tip there (traps 3, 5)."""
    from mfae.membrane.kymo_tip import track_tip, mask_channel_end
    rng = np.random.default_rng(4)
    n, Wd = 120, 80
    x = np.arange(Wd)
    tips_true = np.linspace(55, 40, n)                  # faint protrusion, slowly growing
    K = np.empty((n, Wd))
    for k in range(n):
        prof = np.full(Wd, 60.0)
        prof[x >= tips_true[k]] = 68.0                   # ~4 noise units, like Exp 2 trap 5
        prof[x >= Wd - 12] = 80.0                        # base at the entrance
        prof[1:4] = 72.0                                 # channel-end line, 3 px, steady, a bit brighter
        K[k] = prof + rng.normal(0, 2.0, Wd)
    tips, _ = track_tip(K, Wd, np.ones(n, bool))
    assert np.median(np.abs(tips - tips_true)) < 2 and np.all(tips > 10)

    full = np.full((n, Wd), 70.0) + rng.normal(0, 2.0, (n, Wd))    # protrusion up to the box edge
    assert np.allclose(mask_channel_end(full), full)                # left alone


@pytest.mark.parametrize('lengths, pulse', [
    ([0] * 3 + list(range(30, 110, 10)) + [PIP] + [20, 25, 30, 35, 40, 45, 50, 55, 60, 62, 64], 'end'),
    ([0] * 3 + list(range(30, 110, 10)) + [PIP] * 4 + [0] * 3 + [20, 30, 40, 45, 50, 52, 54, 56], 'end'),
    ([0] * 3 + list(range(30, 110, 10)) + [PIP] + [20, 25, 30, 35], None),
    ([0] * 3 + list(range(30, 110, 10)) + [PIP] * 4 + [0] * 8, 'end'),
    ([0] * 3 + list(range(30, 100, 5)) + [100, 100] + [8] * 12 + [20, 30, 40, 45, 50, 52, 54], 'drop'),
])
def test_kymograph_mode_finds_the_same_cells_as_mask_mode(lengths, pulse):
    """Entry, exit and new cells are decided from the masks in both modes; the
    kymograph only moves the tip inside the analysed cell."""
    p = {'end': len(lengths) - 3, 'drop': lengths.index(8) if 8 in lengths else None, None: None}[pulse]
    a = _run_cells(lengths, pulse_idx=p, tip_tracking='mask')
    b = _run_cells(lengths, pulse_idx=p, tip_tracking='kymograph')
    assert a['cell_segments'] == b['cell_segments']
    assert (a['entry_frame_index'], a['exit_frame_index'], a['selected_cell']) == \
           (b['entry_frame_index'], b['exit_frame_index'], b['selected_cell'])
    La, Lb = np.asarray(a['protrusion_lengths_px'], float), np.asarray(b['protrusion_lengths_px'], float)
    assert np.array_equal(np.isnan(La), np.isnan(Lb))                  # same frames tracked
    both = ~np.isnan(La)
    assert np.median(np.abs(La[both] - Lb[both])) < 3                  # same tip on clean cells


def test_kymograph_tip_starts_at_the_cell_not_at_faint_haze_ahead():
    """Exp 2 traps 3, 4, 8, 10: a new cell enters with a soft leading edge and a
    faint glow in the channel ahead (about 1 % of the cell's contrast). On the
    smoothed signal image that glow is many 'noise units'; the tracked tip sat
    at its outer edge, 8-20 um ahead of the cell, until the cell grew into it."""
    from scipy.ndimage import gaussian_filter1d
    from mfae.membrane.kymo_tip import track_tip
    rng = np.random.default_rng(5)
    n, Wd = 80, 80
    x = np.arange(Wd, dtype=float)
    tips_true = np.linspace(58, 40, n)                       # growing from the entrance
    K = np.empty((n, Wd))
    for k in range(n):
        ramp = np.clip((x - tips_true[k]) / 8.0, 0, 1)       # soft edge, ~8 px wide
        haze = 1.2 * np.clip((x - 20) / 30.0, 0, 1)          # faint glow rising towards the cell
        prof = 100.0 * ramp + haze + rng.normal(0, 1.0, Wd)
        K[k] = gaussian_filter1d(prof, 3.0)                  # along-channel smoothing
    tips, _ = track_tip(K, Wd, np.ones(n, bool))
    level = tips_true + 0.25 * 8                              # where the ramp reaches 25 %
    assert np.all(np.abs(tips - level) <= 4), np.round(tips - level, 1)


def test_speck_at_the_entrance_does_not_empty_the_body_mask():
    """Exp 2 trap 5: a speck of a few px at the entrance, higher up than the body,
    was kept as 'the body' (same left edge, found first), then dropped as too small:
    the body mask of that frame was empty."""
    m = np.zeros((H, W), np.uint8)
    m[2:7, PIP:PIP + 5] = 255                                   # speck, top, at the entrance
    yy, xx = np.mgrid[0:H, 0:W]
    m[(np.hypot(xx - PIP - 24, yy - 36) <= 23) & (xx >= PIP)] = 255   # body
    out = utils._keep_blob_nearest_to_x(m, PIP, min_area=100, max_gap=8)
    assert out[36, PIP + 20] and not out[4, PIP + 2]
    # the old call (no area limit) still picks by left edge only
    assert utils._keep_blob_nearest_to_x(m, PIP)[4, PIP + 2]


def test_medium_joined_to_a_faint_body_is_trimmed():
    """Exp 2 trap 5: in single frames (and runs of frames) the medium next to the
    trap crosses the body threshold, joins the body and the body mask balloons;
    it is trimmed back to the body of the frames around it. Frames that are not
    too large are left exactly as they are."""
    from mfae.membrane.detection import LineDetectionMFA
    yy, xx = np.mgrid[0:H, 0:W]
    body = ((np.hypot(xx - PIP - 24, yy - 36) <= 20) & (xx >= PIP)).astype(np.uint8) * 255
    medium = ((xx >= PIP + 40) & (yy < 50)).astype(np.uint8) * 255   # broad contact with the body
    merged_frames = {5, 9, 10, 11, 12, 13, 14, 20}                     # singles and a run of 6
    fm = [(np.zeros((H, W), np.uint8), cv2.bitwise_or(body, medium) if k in merged_frames else body.copy())
          for k in range(30)]

    class Det:
        min_area_threshold = 100
        results = {'frame_masks': fm, 'body_area_um2': [0.0] * 30}

        def _calculate_morphology(self, mp, mb, mt):
            return (0.0, float((mb > 0).sum()),) + (0.0,) * 8

    d = Det()
    changed = LineDetectionMFA._trim_merged_body_masks(d)
    assert sorted(changed) == sorted(merged_frames)
    for k in range(30):
        mb = d.results['frame_masks'][k][1] > 0
        assert not mb[10, PIP + 80]                                  # medium gone
        assert (mb & (body > 0)).sum() == (body > 0).sum()           # body intact
        assert mb.sum() <= (body > 0).sum() * 1.15
    assert d.results['body_area_um2'][5] == float((d.results['frame_masks'][5][1] > 0).sum())


def test_kymograph_tip_on_a_protrusion_filling_most_of_the_channel():
    """Exp 2 trap 6: the protrusion fills ~90 % of the channel. The channel-end
    line blanking wiped the empty channel where it ends (anything within 12 px a
    little brighter than the protrusion counted as 'the line'), and the
    background (20th percentile of the profile) sat inside the protrusion; the
    tip landed on an internal step ~12-20 px inside the real end."""
    from mfae.membrane.kymo_tip import track_tip
    rng = np.random.default_rng(6)
    n, Wd, tip = 60, 80, 9
    x = np.arange(Wd)
    prof = np.where(x >= tip, 18.0, 2.0)
    prof[x >= 32] = 30.0                                  # internal step
    prof[(x >= 54) & (x < 62)] = 19.0                     # dimmer stretch
    prof[x >= 70] = 36.0                                  # base at the entrance
    prof[1:4] += 3.0                                      # channel-end line
    K = np.array([prof + rng.normal(0, 1.0, Wd) for _ in range(n)])
    tips, _ = track_tip(K, Wd, np.ones(n, bool))
    assert np.all(np.abs(tips - tip) <= 2), np.round(tips, 1)


def test_kymograph_tip_meets_the_mask_threshold_on_a_dim_end_next_to_a_bright_base():
    """Exp 2 trap 9: the protrusion end is ~45 SNR, the base at the entrance ~200.
    25 % of the base was never reached near the tip, so the tip stayed at the outer
    foot of the edge (whole pixels, ~1 px ahead of the mask). With the mask
    threshold as the level it sits where the mask edge is, to sub-pixel."""
    from scipy.ndimage import gaussian_filter1d
    from mfae.membrane.kymo_tip import track_tip
    rng = np.random.default_rng(9)
    n, Wd, tip = 50, 80, 56.0
    x = np.arange(Wd, dtype=float)
    prof = np.where(x >= tip, 45.0, 2.0)
    prof[x >= 66] = 200.0                                  # bright spot at the entrance
    prof = gaussian_filter1d(prof, 1.5)                    # soft edge
    thr = 35.0                                             # mask threshold (SNR)
    edge = float(np.interp(thr, prof[52:59], x[52:59]))    # where the mask edge is
    K = np.array([prof + rng.normal(0, 0.7, Wd) for _ in range(n)])
    old, _ = track_tip(K, Wd, np.ones(n, bool))
    new, _ = track_tip(K, Wd, np.ones(n, bool), level_floor=thr)
    assert np.median(np.abs(new - edge)) < 0.4
    assert np.median(np.abs(new - edge)) < np.median(np.abs(old - edge))
    # a dim far end below the mask threshold keeps the tracked edge
    far = np.where(x >= 30, 8.0, 0.0); far[x >= 60] = 60.0
    Kf = np.array([far + rng.normal(0, 0.7, Wd) for _ in range(n)])
    tf, _ = track_tip(Kf, Wd, np.ones(n, bool), level_floor=20.0)
    assert np.all(np.abs(tf - 30) <= 2)
