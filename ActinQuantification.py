# -*- coding: utf-8 -*-
"""
Quantification of Actin (C3) distribution.

Three levels of analysis
------------------------
1. Whole-region relative means (protrusion / body / total).
   These are the same metrics, normalised to the pre-pulse baseline.

2. Cortex vs. lumen structure (cell body only).
   The body mask is eroded inward to create a cortex shell and an interior lumen.
   Their mean intensities are compared and the cortex texture (CV) classifies the
   actin organisation for each frame:
       ratio < ~1  → interior enriched  → label "interior"
       ratio ≈ 1   → uniform            → label "uniform"
       ratio > ~1, low  CV → even ring  → label "cortex"   (continuous cortex)
       ratio > ~1, high CV → uneven ring→ label "patches"  (patchy actin)

   *** CV note ***
   LOW  CV = small spread around the ring mean = even signal = continuous cortex.
   HIGH CV = large spread = bright spots and dark gaps = patches.
   The threshold is self.cortex_cv_threshold (default 0.4, configurable in
   config.yaml under actin_parameters → cortex_cv_threshold).

3. Four spatial zones.
   Defined dynamically each frame from the mask extents:
       Protrusion tip       – far  half of the protrusion (deepest in channel)
       Protrusion base      – near half of the protrusion (near the entrance)
       Body perinuclear     – near 1/3 of the body (adjacent to entrance)
       Body distal          – far  2/3 of the body (away from entrance)

   For each zone, three metrics are produced:
       a) Absolute mean intensity per frame.
       b) Normalised mean (I / I_0, where I_0 = pre-pulse baseline average).
       c) Rate of change dI/dt (finite differences, units = a.u. / second).

   Plus one scalar remodelling score per zone:
       Post-pulse mean / Pre-pulse mean   (ratio > 1 = actin increased)
       Post-pulse mean − Pre-pulse mean   (difference, signed)
"""

import logging
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Any, Tuple
from pathlib import Path

import Utils_MFA as utils

logger = logging.getLogger(__name__)


class ActinAnalyzer:

    def __init__(self, 
                 membrane_rois:  List[np.ndarray], 
                 actin_rois:     List[np.ndarray], 
                 pipette_x:      int,
                 threshold_prot: int,
                 threshold_body: int,
                 params:         Dict[str, Any],
                 frame_masks:    List[Tuple] = None,
                 start_idx:      int = 0):

        self.mem_imgs       = membrane_rois
        self.actin_imgs     = actin_rois
        self.pipette_x      = pipette_x
        self.threshold_prot = threshold_prot
        self.threshold_body = threshold_body
        self.params         = params

        # Pre-computed per-frame masks from LineDetection.
        # Each entry is (mask_prot, mask_body) — the exact same masks used for
        # protrusion length measurement and kymograph generation, so all metrics
        # are spatially consistent with each other.
        # Falls back to recomputing via generate_dual_masks() if not provided
        # (e.g. when called from older code paths).
        self.frame_masks = frame_masks or []
        self.start_idx = start_idx

        # Pull sub-sections once so callers don't repeat .get() chains.
        exp_p   = params.get('experiment_parameters', {})
        dye_p   = params.get('dye_uptake_parameters', {})
        actin_p = params.get('actin_parameters', {})

        self.scale_factor = exp_p.get('scale_factor', 0.629)

        # Number of frames before the pulse used to compute I_0.
        self.baseline_len = dye_p.get('baseline_frames', 5)

        # Pulse frame (0-based index).  Mirrors how DyeUptakeAnalyzer reads it.
        self.pulse_frame = max(0, dye_p.get('pulse_frame', 10) - 1)

        # ---- Cortex parameters ----
        # How many pixels to erode inward when drawing the cortex shell.
        # Default 3 px ≈ 1.9 µm at 0.629 µm/px.  Override in config:
        #   actin_parameters → cortex_thickness_px
        self.cortex_thickness_px = actin_p.get('cortex_thickness_px', 3)

        # CV boundary separating continuous cortex from patches.
        # LOW  CV ≤ threshold → smooth, continuous ring → "cortex"
        # HIGH CV >  threshold → spotty, fragmented     → "patches"
        # Override in config: actin_parameters → cortex_cv_threshold
        self.cortex_cv_threshold = actin_p.get('cortex_cv_threshold', 0.4)

        # ---- Pulse remodelling windows ----
        # Override in config: actin_parameters → pre/post_pulse_window_frames
        self.pre_pulse_window  = actin_p.get('pre_pulse_window_frames',   5)
        self.post_pulse_window = actin_p.get('post_pulse_window_frames', 10)

        # ---- Results container ----
        # All per-frame lists grow together inside run().
        # Scalar remodelling scores are stored as single-element lists so they
        # can be broadcast to the full frame count in export_csv().
        self.results: Dict[str, Any] = {
            'time_s': [],

            # --- 1. Whole-region relative means (I_t / I_0) ---
            'actin_prot_rel':                 [],
            'actin_body_rel':                 [],
            'actin_total_rel':                [],
            'actin_integrated_density_total': [],
            'actin_ratio_pb':                 [],

            # --- 2. Body cortex vs. lumen ---
            'actin_body_cortex_mean':        [],
            'actin_body_lumen_mean':         [],
            'actin_body_cortex_lumen_ratio': [],
            'actin_body_cortex_cv':          [],
            'actin_body_structure_label':    [],

            # --- 2b. Protrusion cortex vs. lumen ---
            'actin_prot_cortex_mean':        [],
            'actin_prot_lumen_mean':         [],
            'actin_prot_cortex_lumen_ratio': [],
            'actin_prot_cortex_cv':          [],
            'actin_prot_structure_label':    [],


            # --- 3. Spatial zones (absolute means) ---
            'zone_tip_mean':         [],
            'zone_base_mean':        [],
            'zone_perinuclear_mean': [],
            'zone_distal_mean':      [],

            # Normalised to pre-pulse baseline (I_t / I_0)
            'zone_tip_norm':         [],
            'zone_base_norm':        [],
            'zone_perinuclear_norm': [],
            'zone_distal_norm':      [],

            # Rate of change dI/dt [a.u./s] — frame 0 is NaN (no previous frame).
            # These are filled in after the loop by _compute_didt().
            'zone_tip_didt':         [],
            'zone_base_didt':        [],
            'zone_perinuclear_didt': [],
            'zone_distal_didt':      [],

            # Pulse remodelling scores (one scalar per trap, stored as
            # single-element list and broadcast in export_csv).
            # None = pulse not configured or out of range.
            'pulse_remodel_ratio_tip':          [None],
            'pulse_remodel_ratio_base':         [None],
            'pulse_remodel_ratio_perinuclear':  [None],
            'pulse_remodel_ratio_distal':       [None],
            'pulse_remodel_diff_tip':           [None],
            'pulse_remodel_diff_base':          [None],
            'pulse_remodel_diff_perinuclear':   [None],
            'pulse_remodel_diff_distal':        [None],

            # Spatial profiles (column-averaged, for kymograph / profile plots)
            'spatial_profiles': [],
        }

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def run(self, time_data: List[float]) -> Dict[str, Any]:
        """
        Main processing loop.  Called once per trap by MFA_analysis.py.
        Returns self.results.
        """
        valid_frames = min(len(self.mem_imgs), len(self.actin_imgs))
        if valid_frames == 0:
            return self.results

        epsilon = 1e-6

        # ----------------------------------------------------------------
        # STEP 1 — Compute per-region and per-zone baselines (I_0).
        #
        # We average the `baseline_len` frames that come just before
        # pulse_frame.  If the pulse hasn't happened yet in the sequence
        # (pulse_frame == 0), baseline defaults to the first few frames.
        # ----------------------------------------------------------------
        dye_params = self.params.get('dye_uptake_parameters', {})
        dye_enabled = dye_params.get('enable', False)
        # Pulse exists ONLY if dye module is enabled AND pulse is flagged true
        has_pulse = dye_enabled and dye_params.get('has_pulse', True)
        
        if has_pulse:
            baseline_end   = min(self.pulse_frame, valid_frames)
            baseline_start = max(0, baseline_end - self.baseline_len)
        else:
            # Anchor to cell entry, not frame 0
            baseline_start = self.start_idx
            baseline_end   = min(self.start_idx + self.baseline_len, valid_frames)

        # Collect mean values across the baseline window for each region/zone.
        base_prot, base_body, base_total      = [], [], []
        base_tip, base_base_z, base_pn, base_dist = [], [], [], []
        base_prot_cortex, base_prot_lumen = [], []

        for k in range(baseline_start, baseline_end):
            mem_img = self.mem_imgs[k]
            act_img = self.actin_imgs[k]
            if mem_img is None or act_img is None:
                continue

            mask_prot, mask_body = self._get_masks(k, mem_img)
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            act_f      = act_img.astype(float)

            # Whole-region baselines
            base_prot.append( cv2.mean(act_img, mask=mask_prot)[0]  )
            base_body.append( cv2.mean(act_img, mask=mask_body)[0]  )
            base_total.append(cv2.mean(act_img, mask=mask_total)[0] )

            # Zone baselines
            zones = self._build_zone_masks(mask_prot, mask_body)
            base_tip.append(   self._masked_mean(act_f, zones['tip'])        )
            base_base_z.append(self._masked_mean(act_f, zones['base'])       )
            base_pn.append(    self._masked_mean(act_f, zones['perinuclear']))
            base_dist.append(  self._masked_mean(act_f, zones['distal'])     )

            # Protrusion cortex / lumen baselines
            _pc, _pl = utils.generate_cortex_masks(mask_prot, self.cortex_thickness_px)
            base_prot_cortex.append(self._masked_mean(act_f, _pc))
            base_prot_lumen.append( self._masked_mean(act_f, _pl))

        # Convert to scalar I_0 values (mean across baseline frames).
        # max(..., epsilon) prevents division-by-zero later.
        f0_prot  = max(np.mean(base_prot)    if base_prot    else 1.0, epsilon)
        f0_body  = max(np.mean(base_body)    if base_body    else 1.0, epsilon)
        f0_total = max(np.mean(base_total)   if base_total   else 1.0, epsilon)
        f0_tip   = max(np.mean(base_tip)     if base_tip     else 1.0, epsilon)
        f0_base_z= max(np.mean(base_base_z)  if base_base_z  else 1.0, epsilon)
        f0_pn    = max(np.mean(base_pn)      if base_pn      else 1.0, epsilon)
        f0_dist         = max(np.mean(base_dist)        if base_dist        else 1.0, epsilon)
        f0_prot_cortex  = max(np.mean(base_prot_cortex) if base_prot_cortex else 1.0, epsilon)
        f0_prot_lumen   = max(np.mean(base_prot_lumen)  if base_prot_lumen  else 1.0, epsilon)

        # ----------------------------------------------------------------
        # STEP 2 — Per-frame analysis
        # ----------------------------------------------------------------
        for i in range(self.start_idx, valid_frames):
            mem_img = self.mem_imgs[i]
            act_img = self.actin_imgs[i]

            if mem_img is None or act_img is None:
                self._record_empty()
                continue

            mask_prot, mask_body = self._get_masks(i, mem_img)
            mask_total = cv2.bitwise_or(mask_prot, mask_body)
            act_f      = act_img.astype(float)

            # ---- 2a. Whole-region means ----
            mean_prot  = cv2.mean(act_img, mask=mask_prot)[0]
            mean_body  = cv2.mean(act_img, mask=mask_body)[0]
            mean_total = cv2.mean(act_img, mask=mask_total)[0]
            area_um2   = cv2.countNonZero(mask_total) * (self.scale_factor ** 2)

            self.results['actin_prot_rel'].append(  mean_prot  / f0_prot  )
            self.results['actin_body_rel'].append(  mean_body  / f0_body  )
            self.results['actin_total_rel'].append( mean_total / f0_total )
            self.results['actin_integrated_density_total'].append(mean_total * area_um2)
            self.results['actin_ratio_pb'].append(
                mean_prot / mean_body if mean_body > epsilon else 0.0
            )

            # ---- 2b. Body cortex vs. lumen ----
            # We erode the BODY mask only (not the total mask).
            # The protrusion is excluded: it is too narrow for a meaningful
            # cortex shell and its edge is defined by the channel walls, not
            # the cell membrane.
            mask_b_cortex, mask_b_lumen = utils.generate_cortex_masks(
                mask_body, self.cortex_thickness_px
            )
            cortex_mean, cortex_cv = self._mean_and_cv(act_f, mask_b_cortex)
            lumen_mean,  _         = self._mean_and_cv(act_f, mask_b_lumen)
            cl_ratio = cortex_mean / max(lumen_mean, epsilon)
            label    = self._classify_structure(cl_ratio, cortex_cv)

            self.results['actin_body_cortex_mean'].append(cortex_mean)
            self.results['actin_body_lumen_mean'].append( lumen_mean)
            self.results['actin_body_cortex_lumen_ratio'].append(cl_ratio)
            self.results['actin_body_cortex_cv'].append(cortex_cv)
            self.results['actin_body_structure_label'].append(label)

            # ---- 2c. Protrusion cortex vs. lumen ----
            # The protrusion is narrow; the eroded lumen may be empty on frames
            # where cortex_thickness_px >= half the protrusion width.
            # _mean_and_cv returns (0.0, 0.0) for empty masks gracefully.
            mask_p_cortex, mask_p_lumen = utils.generate_cortex_masks(
                mask_prot, self.cortex_thickness_px
            )
            p_cortex_mean, p_cortex_cv = self._mean_and_cv(act_f, mask_p_cortex)
            p_lumen_mean, _            = self._mean_and_cv(act_f, mask_p_lumen)
            p_cl_ratio = p_cortex_mean / max(p_lumen_mean, epsilon)
            p_label    = self._classify_structure(p_cl_ratio, p_cortex_cv)

            self.results['actin_prot_cortex_mean'].append(p_cortex_mean)
            self.results['actin_prot_lumen_mean'].append( p_lumen_mean)
            self.results['actin_prot_cortex_lumen_ratio'].append(p_cl_ratio)
            self.results['actin_prot_cortex_cv'].append(p_cortex_cv)
            self.results['actin_prot_structure_label'].append(p_label)

            # ---- 2d. Four spatial zones ----
            zones = self._build_zone_masks(mask_prot, mask_body)

            tip_m  = self._masked_mean(act_f, zones['tip'])
            base_m = self._masked_mean(act_f, zones['base'])
            pn_m   = self._masked_mean(act_f, zones['perinuclear'])
            dist_m = self._masked_mean(act_f, zones['distal'])

            self.results['zone_tip_mean'].append(tip_m)
            self.results['zone_base_mean'].append(base_m)
            self.results['zone_perinuclear_mean'].append(pn_m)
            self.results['zone_distal_mean'].append(dist_m)

            self.results['zone_tip_norm'].append(       tip_m  / f0_tip   )
            self.results['zone_base_norm'].append(      base_m / f0_base_z)
            self.results['zone_perinuclear_norm'].append(pn_m  / f0_pn    )
            self.results['zone_distal_norm'].append(    dist_m / f0_dist  )

            # Spatial profile for kymograph / profile plots (unchanged from before)
            profile = utils.calculate_spatial_profile(act_f, mask_total)
            self.results['spatial_profiles'].append(profile)

        # Sync time vector to however many frames were actually processed
        n_frames = len(self.results['actin_prot_rel'])
        self.results['time_s'] = time_data[:n_frames]

        # ----------------------------------------------------------------
        # STEP 3 — Post-loop derived metrics
        # ----------------------------------------------------------------
        self._compute_didt()
        self._compute_pulse_remodelling()

        return self.results

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _get_masks(self, frame_idx: int, mem_img: np.ndarray):
        """
        Returns (mask_prot, mask_body) for a given frame.

        Uses the pre-computed masks from LineDetection when available — these
        are the same masks used for protrusion length, kymograph, and dye uptake,
        ensuring all metrics refer to identical cell regions.

        Falls back to recomputing via generate_dual_masks() only if no pre-
        computed masks were provided (e.g. legacy call paths).
        """
        if self.frame_masks and frame_idx < len(self.frame_masks):
            mp, mb = self.frame_masks[frame_idx]
            if mp is not None and mb is not None:
                return mp, mb
        # Fallback: recompute (less accurate for protrusion, but safe)
        return utils.generate_dual_masks(
            mem_img, self.pipette_x,
            self.threshold_prot, self.threshold_body, self.params
        )

    @staticmethod
    def _masked_mean(image: np.ndarray, mask: np.ndarray) -> float:
        """
        Returns the mean of image pixels that fall inside mask.
        Returns 0.0 if mask is empty (no cell in that zone this frame).
        """
        if cv2.countNonZero(mask) == 0:
            return 0.0
        return float(np.mean(image[mask > 0]))

    @staticmethod
    def _mean_and_cv(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        """
        Returns (mean, CV) of image pixels inside mask.

        CV = std / mean, which measures how spread-out the pixel values are
        relative to the average.  A value of 0.4 means the standard deviation
        is 40% of the mean — a useful threshold between even and patchy actin.

        Returns (0.0, 0.0) if mask has no pixels.
        """
        if cv2.countNonZero(mask) == 0:
            return 0.0, 0.0
        pixels = image[mask > 0].astype(float)
        mean   = float(np.mean(pixels))
        cv     = float(np.std(pixels) / max(mean, 1e-6))
        return mean, cv

    def _classify_structure(self, cl_ratio: float, cortex_cv: float) -> str:
        """
        Assigns a human-readable actin structure label for one frame.

        Decision tree
        -------------
        cl_ratio < 0.9               → "interior"
            The inside of the cell is brighter than the edge.

        0.9 ≤ cl_ratio ≤ 1.1         → "uniform"
            Edge and interior are indistinguishable.

        cl_ratio > 1.1, CV ≤ threshold → "cortex"
            Edge is brighter AND the brightness is evenly spread around
            the ring → continuous actin cortex.
            (LOW CV = small variation = even signal)

        cl_ratio > 1.1, CV > threshold → "patches"
            Edge is brighter BUT the brightness is concentrated in spots
            with dark gaps between them → patchy actin.
            (HIGH CV = large variation = uneven signal)

        *** Why LOW CV = cortex and HIGH CV = patches ***
        A continuous cortex fills the whole ring with similar intensity →
        small standard deviation relative to the mean → LOW CV.
        Patches leave most of the ring dark while a few hot-spots are very
        bright → large standard deviation → HIGH CV.
        The threshold (default 0.4) means "is the std more than 40% of the mean?"
        """
        if cl_ratio < 0.9:
            return "interior"
        if cl_ratio <= 1.1:
            return "uniform"
        # Edge-enriched: distinguish by patchiness of the cortex ring
        if cortex_cv <= self.cortex_cv_threshold:
            return "cortex"    # even ring → continuous cortex
        return "patches"       # uneven ring → patchy actin

    def _build_zone_masks(
        self,
        mask_prot: np.ndarray,
        mask_body: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Splits the protrusion and body into four spatial zones.

        How the split works
        -------------------
        Protrusion (pixels to the LEFT of pipette_x):
            Find the leftmost (deepest in channel) and rightmost (closest
            to entrance) x-columns that contain cell pixels.
            Split at the midpoint:
                tip  → left  half (deeper, farther from entrance)
                base → right half (nearer to entrance)

        Body (pixels to the RIGHT of pipette_x):
            Find the leftmost and rightmost x-columns with cell pixels.
            Split at 1/3 of the body's x-extent:
                perinuclear → near 1/3 (adjacent to entrance)
                distal      → far  2/3 (away from entrance)

        If a mask has no pixels (e.g. cell not yet in channel), the
        corresponding zone masks will be empty (all zeros).  _masked_mean()
        handles this gracefully by returning 0.0.
        """
        # --- Protrusion zones ---
        # np.where(mask.any(axis=0)) finds all x-columns that have at least
        # one non-zero pixel (i.e. the column is "occupied" by the cell).
        prot_cols = np.where(mask_prot.any(axis=0))[0]

        if prot_cols.size >= 2:
            tip_x_start = int(prot_cols[0])            # deepest column
            prot_x_end  = int(prot_cols[-1])           # column nearest entrance
            midpoint    = (tip_x_start + prot_x_end) // 2

            # Start from a full copy then zero out the columns we don't want.
            mask_tip  = mask_prot.copy()
            mask_tip[:, midpoint + 1:] = 0             # keep only up-to midpoint

            mask_base = mask_prot.copy()
            mask_base[:, :midpoint + 1] = 0            # keep only past midpoint
        else:
            mask_tip  = np.zeros_like(mask_prot)
            mask_base = np.zeros_like(mask_prot)

        # --- Body zones ---
        body_cols = np.where(mask_body.any(axis=0))[0]

        if body_cols.size >= 2:
            body_x_start  = int(body_cols[0])          # column nearest entrance
            body_x_end    = int(body_cols[-1])          # farthest column
            body_extent   = body_x_end - body_x_start
            split_x       = body_x_start + max(1, body_extent // 3)

            mask_perinuclear = mask_body.copy()
            mask_perinuclear[:, split_x + 1:] = 0      # keep near 1/3

            mask_distal = mask_body.copy()
            mask_distal[:, :split_x + 1] = 0           # keep far 2/3
        else:
            mask_perinuclear = np.zeros_like(mask_body)
            mask_distal      = np.zeros_like(mask_body)

        return {
            'tip':         mask_tip,
            'base':        mask_base,
            'perinuclear': mask_perinuclear,
            'distal':      mask_distal,
        }

    def _compute_didt(self) -> None:
        """
        Computes dI/dt (rate of actin intensity change) for each zone.

        Method: finite differences.
            dI/dt[i] = (I[i] - I[i-1]) / (t[i] - t[i-1])

        Frame 0 has no previous frame, so it is set to NaN.
        A positive value means actin is accumulating in that zone.
        A negative value means actin is leaving.

        The resulting lists REPLACE whatever placeholders were added by
        _record_empty(), so the length stays consistent.
        """
        time = np.array(self.results['time_s'], dtype=float)

        for zone in ('tip', 'base', 'perinuclear', 'distal'):
            means = np.array(self.results[f'zone_{zone}_mean'], dtype=float)
            n     = len(means)
            didt  = np.full(n, np.nan)   # start with all NaN

            if n >= 2:
                # np.diff(time) gives the time gap between consecutive frames.
                # We clamp to 1e-6 to handle the edge case of identical timestamps.
                dt       = np.diff(time)
                dt       = np.where(dt == 0, 1e-6, dt)
                didt[1:] = np.diff(means) / dt

            self.results[f'zone_{zone}_didt'] = didt.tolist()

    def _compute_pulse_remodelling(self) -> None:
        """
        Computes a scalar pre/post-pulse actin remodelling score for each zone.

        Pre-pulse window  : `pre_pulse_window` frames ending just before
                            pulse_frame (inclusive).
        Post-pulse window : `post_pulse_window` frames starting the frame
                            after pulse_frame.

        Remodelling ratio      = mean(post) / mean(pre)
            >1 means actin accumulated after the pulse.
            <1 means actin was lost after the pulse.

        Remodelling difference = mean(post) − mean(pre)
            Positive = actin increased.  Negative = actin decreased.

        Stored as single-element lists; export_csv() broadcasts them to
        all rows so the CSV has one consistent value per column.
        """
        n_frames = len(self.results['zone_tip_mean'])
        pulse    = self.pulse_frame
        epsilon  = 1e-6
        
        dye_params = self.params.get('dye_uptake_parameters', {})
        has_pulse = dye_params.get('enable', False) and dye_params.get('has_pulse', True)
        
        # If a pulse is configured but outside the data range, skip gracefully.
        if has_pulse and (pulse <= 0 or pulse >= n_frames):
            logger.debug(
                f"Pulse frame {pulse} is outside processed range "
                f"(0–{n_frames-1}); skipping pulse remodelling."
            )
            return

        for zone in ('tip', 'base', 'perinuclear', 'distal'):
            means = np.array(self.results[f'zone_{zone}_mean'], dtype=float)

            if has_pulse:
                # Pre-window: up to pre_pulse_window frames just before the pulse
                pre_start = max(0, pulse - self.pre_pulse_window)
                pre_vals  = means[pre_start : pulse]   # does NOT include pulse frame

                # Post-window: first post_pulse_window frames after the pulse
                post_end  = min(n_frames, pulse + 1 + self.post_pulse_window)
                post_vals = means[pulse + 1 : post_end]
            else:
                # Aspiration Remodeling (Entry vs. Plateau)
                pre_end = min(self.pre_pulse_window, n_frames)
                pre_vals = means[0 : pre_end]
                
                post_start = max(0, n_frames - self.post_pulse_window)
                post_vals = means[post_start : n_frames]

            if pre_vals.size == 0 or post_vals.size == 0:
                ratio = None
                diff  = None
            else:
                pre_mean  = float(np.mean(pre_vals))
                post_mean = float(np.mean(post_vals))
                ratio = post_mean / max(pre_mean, epsilon)
                diff  = post_mean - pre_mean

            self.results[f'pulse_remodel_ratio_{zone}'] = [ratio]
            self.results[f'pulse_remodel_diff_{zone}']  = [diff]

    def _record_empty(self) -> None:
        """
        Appends 0.0 placeholder values for a frame where images are missing.
        Keeps all per-frame lists the same length so nothing desynchronises.
        Note: dI/dt lists are overwritten entirely after the loop, so the 0.0
        placed here for those keys does not affect the final result.
        """
        scalar_keys = [
            'actin_prot_rel',  'actin_body_rel', 'actin_total_rel',
            'actin_integrated_density_total', 'actin_ratio_pb',
            'actin_body_cortex_mean', 'actin_body_lumen_mean',
            'actin_body_cortex_lumen_ratio', 'actin_body_cortex_cv',
            'actin_prot_cortex_mean', 'actin_prot_lumen_mean',
            'actin_prot_cortex_lumen_ratio', 'actin_prot_cortex_cv',
            'zone_tip_mean',    'zone_base_mean',
            'zone_perinuclear_mean', 'zone_distal_mean',
            'zone_tip_norm',    'zone_base_norm',
            'zone_perinuclear_norm', 'zone_distal_norm',
            'zone_tip_didt',   'zone_base_didt',
            'zone_perinuclear_didt', 'zone_distal_didt',
        ]
        for k in scalar_keys:
            self.results[k].append(0.0)
        self.results['actin_body_structure_label'].append('unknown')
        self.results['actin_prot_structure_label'].append('unknown')
        self.results['spatial_profiles'].append(np.zeros(10))

    # =========================================================================
    # =========================================================================
    # DEBUG VIDEO HELPERS
    # =========================================================================

    def _make_video_writer(self, save_path: Path, w: int, total_h: int, fps: int = 5) -> cv2.VideoWriter:
        """Creates an MJPG VideoWriter at the given path and canvas size."""
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        return cv2.VideoWriter(str(save_path), fourcc, fps, (w, total_h), isColor=True)

    def _build_base_frame(self, act_img: np.ndarray) -> np.ndarray:
        """Normalises the actin image to 8-bit and converts to BGR for display."""
        norm = cv2.normalize(act_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    def _draw_masks_on_frame(self, display: np.ndarray,
                              layers: list, alpha: float = 0.4) -> np.ndarray:
        """
        Draws coloured semi-transparent fills + contour outlines for each
        (mask, colour) pair in `layers`. Layers drawn first are underneath.
        Returns a new array — the input is not modified.
        """
        out = display.copy()
        overlay = display.copy()
        for mask, colour in layers:
            if cv2.countNonZero(mask) > 0:
                overlay[mask > 0] = colour
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
        for mask, colour in layers:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, colour, 1)
        return out

    def _attach_info_strip(self, display: np.ndarray, h: int, w: int,
                            info_lines: list, legend_items: list,
                            strip_h: int = 36) -> np.ndarray:
        """
        Attaches a black strip below `display`.
        Left side: up to 2 short text lines.
        Right side: colour-square legend (one row per item).
        Returns the combined frame ready to write.
        """
        strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
        frame = np.vstack([display, strip])

        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = 0.36
        thick = 1
        sq    = 9
        pad   = 5

        for li, line in enumerate(info_lines):
            y = h + 11 + li * 13
            cv2.putText(frame, line, (4, y), font, fs, (255, 255, 255), thick, cv2.LINE_AA)

        max_lw  = max(cv2.getTextSize(lbl, font, fs, thick)[0][0] for _, lbl in legend_items)
        block_w = sq + pad + max_lw + pad
        x_start = w - block_w - 4

        for idx, (colour, lbl) in enumerate(legend_items):
            n              = len(legend_items)
            total_block_h  = n * (sq + pad) - pad
            y_top          = h + (strip_h - total_block_h) // 2 + idx * (sq + pad)
            y_text         = y_top + sq - 1
            cv2.rectangle(frame, (x_start, y_top),
                          (x_start + sq, y_top + sq), colour, -1)
            cv2.putText(frame, lbl, (x_start + sq + pad, y_text),
                        font, fs, (255, 255, 255), thick, cv2.LINE_AA)

        return frame

    # =========================================================================
    # DEBUG VIDEOS
    # =========================================================================

    def save_zones_debug_video(self, trap_idx: int, output_dir: Path) -> None:
        """
        Debug video 1 of 2 — spatial zones.

        Shows the four zone masks overlaid on the actin channel, using the
        same (mask_prot, mask_body) from LineDetection as all other metrics:
            DARK BLUE   – Protrusion tip  (far half, deepest in channel)
            MEDIUM BLUE – Protrusion base (near half, closest to entrance)
            PALE RED    – Perinuclear body (nearest 1/3 of body)
            LIGHT RED   – Distal body     (far 2/3 of body)

        Info strip shows current time and pixel count of each zone.
        Saved as Trap_XX_Actin_Debug_Zones.avi
        """
        n_frames = len(self.results.get('time_s', []))
        if n_frames == 0:
            logger.warning(f"Trap {trap_idx}: no frames; skipping zones debug video.")
            return

        ref_img = next((img for img in self.actin_imgs if img is not None), None)
        if ref_img is None:
            return

        h, w    = ref_img.shape[:2]
        # 2 info lines + 4 legend items → max(42, 70) = 70px
        STRIP_H = max(6 + 2*15 + 6, 6 + 4*(10+6) - 6 + 6)  # = 70px
        total_h = h + STRIP_H
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"Trap_{trap_idx:02d}_Actin_Debug_Zones.avi"
        writer    = self._make_video_writer(save_path, w, total_h)

        col_tip         = utils.get_bgr_color('dark_blue')
        col_base        = utils.get_bgr_color('medium_blue')
        col_perinuclear = utils.get_bgr_color('pale_red')
        col_distal      = utils.get_bgr_color('light_red')
        col_line        = (255, 255, 255)

        legend_items = [
            (col_tip,         "Prot tip"),
            (col_base,        "Prot base"),
            (col_perinuclear, "Perinuclear"),
            (col_distal,      "Distal body"),
        ]

        for i in range(n_frames):
            mem_img = self.mem_imgs[i]
            act_img = self.actin_imgs[i]

            if mem_img is None or act_img is None:
                writer.write(np.zeros((total_h, w, 3), dtype=np.uint8))
                continue

            display               = self._build_base_frame(act_img)
            mask_prot, mask_body  = self._get_masks(i, mem_img)
            zones                 = self._build_zone_masks(mask_prot, mask_body)

            layers = [
                (zones['distal'],      col_distal),
                (zones['perinuclear'], col_perinuclear),
                (zones['base'],        col_base),
                (zones['tip'],         col_tip),
            ]
            display = self._draw_masks_on_frame(display, layers, alpha=0.45)
            cv2.line(display, (int(self.pipette_x), 0),
                     (int(self.pipette_x), h), col_line, 1)

            time_s = self.results['time_s'][i] if i < n_frames else 0.0
            n_tip  = int(cv2.countNonZero(zones['tip']))
            n_base = int(cv2.countNonZero(zones['base']))
            n_pn   = int(cv2.countNonZero(zones['perinuclear']))
            n_dist = int(cv2.countNonZero(zones['distal']))

            info_lines = [
                f"t = {time_s:.1f}s",
                f"tip={n_tip}px  base={n_base}px  pn={n_pn}px  dist={n_dist}px",
            ]
            frame_out = self._attach_info_strip(
                display, h, w, info_lines, legend_items, STRIP_H
            )
            writer.write(frame_out)

        writer.release()
        logger.info(f"Zones debug video saved: {save_path.name}")

    def save_cortex_debug_video(self, trap_idx: int, output_dir: Path) -> None:
        """
        Debug video 2 of 2 — body cortex structure.

        Shows the cortex shell and lumen masks on the cell body, using the
        same mask_body from LineDetection as all other metrics:
            DARK RED    – Body cortex shell (outer ring after erosion)
            LIGHT BLUE  – Body lumen       (interior after erosion)

        How the masks are built:
            1. mask_body comes directly from LineDetection (consistent source).
            2. Erode mask_body inward by cortex_thickness_px → lumen.
            3. Cortex shell = mask_body − lumen.

        If lumen=0px in the strip, the body is too narrow for the erosion —
        reduce cortex_thickness_px in config.yaml.
        Saved as Trap_XX_Actin_Debug_Cortex.avi
        """
        n_frames = len(self.results.get('time_s', []))
        if n_frames == 0:
            logger.warning(f"Trap {trap_idx}: no frames; skipping cortex debug video.")
            return

        ref_img = next((img for img in self.actin_imgs if img is not None), None)
        if ref_img is None:
            return

        h, w    = ref_img.shape[:2]
        # 4 legend items: body cortex/lumen + protrusion cortex/lumen
        STRIP_H = max(6 + 3*15 + 6, 6 + 4*(10+6) - 6 + 6)  # = 69px
        total_h = h + STRIP_H
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"Trap_{trap_idx:02d}_Actin_Debug_Cortex.avi"
        writer    = self._make_video_writer(save_path, w, total_h)

        col_b_cortex = utils.get_bgr_color('dark_red')
        col_b_lumen  = utils.get_bgr_color('light_blue')
        col_p_cortex = utils.get_bgr_color('medium_red')
        col_p_lumen  = utils.get_bgr_color('medium_blue')
        col_line     = (255, 255, 255)

        legend_items = [
            (col_b_cortex, "Body cortex"),
            (col_b_lumen,  "Body lumen"),
            (col_p_cortex, "Prot cortex"),
            (col_p_lumen,  "Prot lumen"),
        ]

        for i in range(n_frames):
            mem_img = self.mem_imgs[i]
            act_img = self.actin_imgs[i]

            if mem_img is None or act_img is None:
                writer.write(np.zeros((total_h, w, 3), dtype=np.uint8))
                continue

            display                  = self._build_base_frame(act_img)
            mask_prot, mask_body     = self._get_masks(i, mem_img)
            mask_b_cortex, mask_b_lumen = utils.generate_cortex_masks(
                mask_body, self.cortex_thickness_px
            )
            mask_p_cortex, mask_p_lumen = utils.generate_cortex_masks(
                mask_prot, self.cortex_thickness_px
            )

            # Lumen fills first (large areas), cortex shells on top.
            layers = [
                (mask_b_lumen,  col_b_lumen),
                (mask_p_lumen,  col_p_lumen),
                (mask_b_cortex, col_b_cortex),
                (mask_p_cortex, col_p_cortex),
            ]
            display = self._draw_masks_on_frame(display, layers, alpha=0.45)
            cv2.line(display, (int(self.pipette_x), 0),
                     (int(self.pipette_x), h), col_line, 1)

            time_s   = self.results['time_s'][i] if i < n_frames else 0.0
            b_cl  = (self.results['actin_body_cortex_lumen_ratio'][i]
                     if i < len(self.results['actin_body_cortex_lumen_ratio']) else 0.0)
            b_cv  = (self.results['actin_body_cortex_cv'][i]
                     if i < len(self.results['actin_body_cortex_cv']) else 0.0)
            b_lbl = (self.results['actin_body_structure_label'][i]
                     if i < len(self.results['actin_body_structure_label']) else '-')
            p_cl  = (self.results['actin_prot_cortex_lumen_ratio'][i]
                     if i < len(self.results['actin_prot_cortex_lumen_ratio']) else 0.0)
            p_cv  = (self.results['actin_prot_cortex_cv'][i]
                     if i < len(self.results['actin_prot_cortex_cv']) else 0.0)
            p_lbl = (self.results['actin_prot_structure_label'][i]
                     if i < len(self.results['actin_prot_structure_label']) else '-')

            info_lines = [
                f"t={time_s:.1f}s",
                f"Body  C/L={b_cl:.2f}  CV={b_cv:.2f}  [{b_lbl}]",
                f"Prot  C/L={p_cl:.2f}  CV={p_cv:.2f}  [{p_lbl}]",
            ]
            frame_out = self._attach_info_strip(
                display, h, w, info_lines, legend_items, STRIP_H
            )
            writer.write(frame_out)

        writer.release()
        logger.info(f"Cortex debug video saved: {save_path.name}")

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_csv(self, trap_idx: int, output_dir: Path) -> None:
        """
        Saves all actin metrics to a CSV file.

        Scalar remodelling scores (one value per trap) are broadcast to every
        row so the table is rectangular and easy to read in Excel or pandas.
        """
        n = len(self.results['time_s'])

        def _col(key: str) -> list:
            """
            Safely fetch a result list.
            If it is a single-element list (scalar result), broadcast it to
            length n so every row has the same value.
            """
            v = self.results.get(key, [])
            if isinstance(v, list) and len(v) == 1 and n > 1:
                return v * n           # e.g. [1.23] → [1.23, 1.23, ..., 1.23]
            return list(v) + [None] * max(0, n - len(v))

        df = pd.DataFrame({
            'Time_s':                         _col('time_s'),

            # Whole-region relatives
            'Actin_Total_Rel':                _col('actin_total_rel'),
            'Actin_Prot_Rel':                 _col('actin_prot_rel'),
            'Actin_Body_Rel':                 _col('actin_body_rel'),
            'Actin_Integrated_Density':       _col('actin_integrated_density_total'),
            'Actin_Ratio_PB':                 _col('actin_ratio_pb'),

            # Body cortex vs. lumen
            'Body_Cortex_Mean':               _col('actin_body_cortex_mean'),
            'Body_Lumen_Mean':                _col('actin_body_lumen_mean'),
            'Body_Cortex_Lumen_Ratio':        _col('actin_body_cortex_lumen_ratio'),
            'Body_Cortex_CV':                 _col('actin_body_cortex_cv'),
            'Body_Structure_Label':           _col('actin_body_structure_label'),

            # Protrusion cortex vs. lumen
            'Prot_Cortex_Mean':               _col('actin_prot_cortex_mean'),
            'Prot_Lumen_Mean':                _col('actin_prot_lumen_mean'),
            'Prot_Cortex_Lumen_Ratio':        _col('actin_prot_cortex_lumen_ratio'),
            'Prot_Cortex_CV':                 _col('actin_prot_cortex_cv'),
            'Prot_Structure_Label':           _col('actin_prot_structure_label'),

            # Zone means (absolute)
            'Zone_Tip_Mean':                  _col('zone_tip_mean'),
            'Zone_Base_Mean':                 _col('zone_base_mean'),
            'Zone_Perinuclear_Mean':          _col('zone_perinuclear_mean'),
            'Zone_Distal_Mean':               _col('zone_distal_mean'),

            # Zone means (normalised to pre-pulse baseline)
            'Zone_Tip_Norm':                  _col('zone_tip_norm'),
            'Zone_Base_Norm':                 _col('zone_base_norm'),
            'Zone_Perinuclear_Norm':          _col('zone_perinuclear_norm'),
            'Zone_Distal_Norm':               _col('zone_distal_norm'),

            # Rate of change [a.u./s]
            'Zone_Tip_dIdt':                  _col('zone_tip_didt'),
            'Zone_Base_dIdt':                 _col('zone_base_didt'),
            'Zone_Perinuclear_dIdt':          _col('zone_perinuclear_didt'),
            'Zone_Distal_dIdt':               _col('zone_distal_didt'),

            # Pulse remodelling scores (broadcast from single value)
            'Pulse_Remodel_Ratio_Tip':        _col('pulse_remodel_ratio_tip'),
            'Pulse_Remodel_Ratio_Base':       _col('pulse_remodel_ratio_base'),
            'Pulse_Remodel_Ratio_Perinuclear':_col('pulse_remodel_ratio_perinuclear'),
            'Pulse_Remodel_Ratio_Distal':     _col('pulse_remodel_ratio_distal'),
            'Pulse_Remodel_Diff_Tip':         _col('pulse_remodel_diff_tip'),
            'Pulse_Remodel_Diff_Base':        _col('pulse_remodel_diff_base'),
            'Pulse_Remodel_Diff_Perinuclear': _col('pulse_remodel_diff_perinuclear'),
            'Pulse_Remodel_Diff_Distal':      _col('pulse_remodel_diff_distal'),
        })

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"Trap_{trap_idx:02d}_Actin_Data.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Actin data saved: {save_path.name}")