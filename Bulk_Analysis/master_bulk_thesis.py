# -*- coding: utf-8 -*-
"""
Master Pipeline for Bulk MFAE Analysis (Thesis Plotting Mode).
"""

import logging
import warnings
import pickle
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

import bulk_file_handling as bfh
import thesis_figure_data as tfd
import bulk_mechanics     as bm
import thesis_plotting    as tp  
from thesis_plotting_uptake import register_uptake_plots
import bulk_utils as utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger('fontTools').setLevel(logging.WARNING) 
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*DataFrame concatenation with empty or all-NA entries.*",
)

logger = logging.getLogger(__name__)

class DiskBackedDict:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True)
        self.keys_map = {}

    def update(self, d):
        for k, v in d.items():
            safe_k = "_".join(str(x) for x in k).replace("/", "-").replace(" ", "")
            filepath = self.cache_dir / f"{safe_k}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(v, f)
            self.keys_map[k] = filepath

    def items(self):
        for k, filepath in self.keys_map.items():
            with open(filepath, 'rb') as f:
                yield k, pickle.load(f)

    def keys(self):
        return self.keys_map.keys()

    def __getitem__(self, k):
        with open(self.keys_map[k], 'rb') as f:
            return pickle.load(f)

def main():
    output_root_dir = Path(r"C:\GitHub\MFAE_Analysis\Output")

    date_stamp  = datetime.now().strftime("%y%m%d")
    results_dir = output_root_dir / "Bulk_Analysis_results" / date_stamp
    results_dir.mkdir(parents=True, exist_ok=True)

    mechanics_dir     = results_dir / "mechanics"
    claim1_dir        = results_dir / "claim1_asp"
    claim2_dir        = results_dir / "claim2_ep_vs_asp"

    # claim1_asp/ is split by treatment. WT/ and CytD/ hold single-
    # condition figures (per-condition trap boxplots, per-condition
    # best-fit multipanels); combined/ holds every cross-treatment
    # figure (protrusion overlay, parameter boxplots, actin-vs-mechanics,
    # trap dependency, body volume vs E, protrusion length vs mechanics).
    claim1_wt_dir       = claim1_dir / "WT"
    claim1_cytd_dir     = claim1_dir / "CytD"
    claim1_combined_dir = claim1_dir / "combined"

    # claim2_ep_vs_asp/ is split three ways along the analysis timeline
    # (pre-pulse / at-pulse / whole-trace). pre-pulse/ has a per-
    # condition subfolder (100V_100us, 100V_5ms), an ASP/ subfolder for
    # the ASP baseline plots that are pulse-independent but shown next
    # to the pulse cohorts, and a combined/ subfolder for cross-cohort
    # comparisons. at-pulse/ and whole-trace/ follow the same shape but
    # without an ASP subfolder (they are pulse-aligned by definition).
    prepulse_dir       = claim2_dir / "pre-pulse"
    at_pulse_dir       = claim2_dir / "at_pulse"
    whole_trace_dir    = claim2_dir / "whole-trace"

    prepulse_100us_dir   = prepulse_dir / "100V_100us"
    prepulse_5ms_dir     = prepulse_dir / "100V_5ms"
    prepulse_asp_dir     = prepulse_dir / "ASP"
    prepulse_combined_dir= prepulse_dir / "combined"

    at_pulse_100us_dir   = at_pulse_dir / "100V_100us"
    at_pulse_5ms_dir     = at_pulse_dir / "100V_5ms"
    at_pulse_combined_dir= at_pulse_dir / "combined"

    whole_trace_100us_dir   = whole_trace_dir / "100V_100us"
    whole_trace_5ms_dir     = whole_trace_dir / "100V_5ms"
    whole_trace_combined_dir= whole_trace_dir / "combined"

    for d in (mechanics_dir,
              claim1_wt_dir, claim1_cytd_dir, claim1_combined_dir,
              prepulse_100us_dir, prepulse_5ms_dir, prepulse_asp_dir, prepulse_combined_dir,
              at_pulse_100us_dir, at_pulse_5ms_dir, at_pulse_combined_dir,
              whole_trace_100us_dir, whole_trace_5ms_dir, whole_trace_combined_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Routing maps consumed by the plot suites. Each map keys a small
    # folder-token onto the destination Path. Plot functions fall back
    # to their `output_dir` argument when the token is missing, so
    # additions of new keys don't break anything.
    # ------------------------------------------------------------------
    claim1_treatment_map = {
        'WT'  : claim1_wt_dir,
        'CytD': claim1_cytd_dir,
    }
    prepulse_bucket_map = {
        'ASP'  : prepulse_asp_dir,
        '100us': prepulse_100us_dir,
        '5ms'  : prepulse_5ms_dir,
    }
    at_pulse_duration_map = {
        '100us': at_pulse_100us_dir,
        '5ms'  : at_pulse_5ms_dir,
    }

    logger.info(f"Results will be saved to: {results_dir}")

    r_eff = bm.DEFAULT_R_EFF
    r2_floor = 0.85

    logger.info(f"Using r_eff = {r_eff:.3f} um  (channel: 6.7 x 5.0 um, C=1.0)")
    logger.info(f"R² quality floor = {r2_floor}")

    def _try(label, func, *args):
        try:
            func(*args)
        except Exception as e:
            logger.error(f"{label} failed: {e}", exc_info=False)

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: DATA LOADING & GLOBAL DURATION CALCULATION")
    logger.info("=" * 50)

    try:
        loader = bfh.BulkDataLoader(
            str(output_root_dir),
            channel_width_um=bm.CHANNEL_WIDTH_UM,
            channel_height_um=bm.CHANNEL_HEIGHT_UM,
            r_eff_um=r_eff,
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return

    all_grouped_data = DiskBackedDict(results_dir / "cache_all")
    for group_key, traps in loader.iter_groups():
        all_grouped_data.update({group_key: traps})

    # --- Phase 1.5: Compute shortest median duration for each trace type ---
    asp_meds, pre_meds, whole_meds = [], [], []
    
    for gk, traps in all_grouped_data.items():
        if not traps: continue
        ctype = traps[0].metadata.condition_type
        
        # Only evaluate traces that meet the 15-point limit
        valid_traps = [t for t in traps if len(t.protrusion_data.get('Time_s', [])) >= 15]
        if not valid_traps: continue

        # Whole-trace durations (for MI Whole)
        w_durs = [float(t.protrusion_data['Time_s'][-1] - t.protrusion_data['Time_s'][0]) for t in valid_traps]
        if w_durs: whole_meds.append(np.median(w_durs))

        if ctype == 'ASP':
            # ASP Viscoelastic durations
            if w_durs: asp_meds.append(np.median(w_durs))
            
        elif ctype == 'EP':
            # Pre-pulse durations (for MI Pre-pulse)
            p_durs = []
            for t in valid_traps:
                pf = t.metadata.pulse_frame
                times = t.protrusion_data['Time_s']
                if 15 <= pf < len(times): 
                    p_durs.append(float(times[pf] - times[0]))
            if p_durs: pre_meds.append(np.median(p_durs))

    global_asp_dur   = min(asp_meds) if asp_meds else None
    global_whole_dur = min(whole_meds) if whole_meds else None
    global_pre_dur   = min(pre_meds) if pre_meds else None

    logger.info(f"Global Common ASP Duration: {global_asp_dur:.1f} s" if global_asp_dur else "Global Common ASP Duration: None")
    logger.info(f"Global Common MI Pre-Pulse Duration: {global_pre_dur:.1f} s" if global_pre_dur else "Global Common MI Pre-Pulse Duration: None")
    logger.info(f"Global Common MI Whole-Trace Duration: {global_whole_dur:.1f} s" if global_whole_dur else "Global Common MI Whole-Trace Duration: None")

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: PER-GROUP MECHANICS")
    logger.info("=" * 50)

    all_mechanics_dfs = []
    all_scalars_dfs   = []

    for group_key, traps in all_grouped_data.items():
        grouped_data_chunk = {group_key: traps}

        mech_df = bm.run_all_mechanics(
            grouped_data_chunk, 
            r_eff=r_eff, 
            C=bm.HALFSPACE_C, 
            r2_floor=r2_floor,
            mi_r2_floor_asp=0.85,
            mi_r2_floor_ep=0.80,
            global_asp_dur=global_asp_dur,
            global_pre_dur=global_pre_dur,
            global_whole_dur=global_whole_dur
        )
        all_mechanics_dfs.append(mech_df)

        scalar_df = bfh.extract_all_scalars(grouped_data_chunk)
        all_scalars_dfs.append(scalar_df)

    if not all_mechanics_dfs:
        logger.error("No valid experiment folders found or processed. Aborting pipeline.")
        return

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: GLOBAL AGGREGATION & THESIS PLOTS")
    logger.info("=" * 50)

    mechanics_df = pd.concat(all_mechanics_dfs, ignore_index=True)
    df_scalars   = pd.concat(all_scalars_dfs,   ignore_index=True)

    raw_path = mechanics_dir / "mechanics_results_all_traps.csv"
    mechanics_df.to_csv(raw_path, index=False)
    logger.info(f"Saved results to: mechanics/{raw_path.name}  ({len(mechanics_df)} traps)")

    # ------------------------------------------------------------------
    # Filter all_grouped_data to the pure ASP viscoelastic cohort
    # ------------------------------------------------------------------
    asp_pass_mask = (
        (mechanics_df['Condition_Type'] == 'ASP')
        & mechanics_df['Best_Model'].notna()
        & mechanics_df['E_Pa'].notna()
        & (mechanics_df['Visco_R2_Flag'] == True)
    )
    accepted_pairs = set(zip(
        mechanics_df.loc[asp_pass_mask, 'Experiment_Folder'],
        mechanics_df.loc[asp_pass_mask, 'Trap_ID'],
    ))
    n_asp = int((mechanics_df.loc[asp_pass_mask, 'Condition_Type'] == 'ASP').sum())
    logger.info(
        f"Viscoelastic-passing cohort: {len(accepted_pairs)} "
        f"(Experiment_Folder, Trap_ID) pairs accepted "
        f"[{n_asp} pure ASP cells]."
    )

    filtered_grouped_data = DiskBackedDict(results_dir / "cache_visco")
    for gk, traps in all_grouped_data.items():
        if traps and traps[0].metadata.condition_type != "ASP":
            continue
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) in accepted_pairs
        ]
        if kept:
            filtered_grouped_data.update({gk: kept})

    if filtered_grouped_data.keys():
        # Cross-treatment ASP figures land in claim1_combined_dir; per-
        # treatment trap boxplots and best-fit multipanels are routed
        # via claim1_treatment_map into claim1_wt_dir / claim1_cytd_dir.
        _try("Thesis Plot Suite",
             tp.run_thesis_plots,
             filtered_grouped_data, mechanics_df,
             claim1_combined_dir, r_eff, global_asp_dur,
             claim1_treatment_map)

    # ------------------------------------------------------------------
    # Claim 1 completion — actin, trap-dependency, body-volume sanity.
    # These are ASP-only figures showing that the platform detects the
    # CytD softening and that the readout doesn't confound with trap
    # position; body volume vs E is intentionally shown even though it
    # is not null (see chapter interpretation). All are cross-treatment
    # so they go to claim1_combined_dir.
    # ------------------------------------------------------------------
    if filtered_grouped_data.keys():
        _try("Thesis Claim 1 Plots",
             tp.run_thesis_claim1_plots,
             filtered_grouped_data, mechanics_df,
             claim1_combined_dir, global_asp_dur,
             claim1_treatment_map)

    # ------------------------------------------------------------------
    # Pre-pulse cohort filter (both ASP and EP-pre): now uses the
    # matched-window viscoelastic fits (PrePulse_Visco_R2_Flag) instead
    # of the old model-independent pre-pulse fits.
    # ------------------------------------------------------------------
    pp_pass_mask = mechanics_df['PrePulse_Best_Model'].notna() & (mechanics_df['PrePulse_Visco_R2_Flag'] == True)

    ep_treatments = set(
        mechanics_df.loc[mechanics_df['Condition_Type'] == 'EP', 'Treatment'].dropna().unique()
    )
    if ep_treatments:
        asp_treatment_matches = (
            (mechanics_df['Condition_Type'] == 'ASP')
            & mechanics_df['Treatment'].isin(ep_treatments)
        )
        keep_non_asp = mechanics_df['Condition_Type'] != 'ASP'
        common_conditions_mask = asp_treatment_matches | keep_non_asp
        pp_pass_mask = pp_pass_mask & common_conditions_mask

    pp_accepted_pairs = set(zip(
        mechanics_df.loc[pp_pass_mask, 'Experiment_Folder'],
        mechanics_df.loc[pp_pass_mask, 'Trap_ID'],
    ))

    mi_filtered_grouped_data = DiskBackedDict(results_dir / "cache_prepulse")
    for gk, traps in all_grouped_data.items():
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) in pp_accepted_pairs
        ]
        if kept:
            mi_filtered_grouped_data.update({gk: kept})

    # ------------------------------------------------------------------
    # MI_Whole cohort filter (whole-trace comparison)
    # ------------------------------------------------------------------
    mi_whole_pass_mask = mechanics_df['MI_Whole_Best_Model'].notna() & (mechanics_df['MI_Whole_R2_Flag'] == True)
    
    if ep_treatments:
        asp_treatment_matches_w = (
            (mechanics_df['Condition_Type'] == 'ASP')
            & mechanics_df['Treatment'].isin(ep_treatments)
        )
        keep_non_asp_w = mechanics_df['Condition_Type'] != 'ASP'
        mi_whole_pass_mask = mi_whole_pass_mask & (asp_treatment_matches_w | keep_non_asp_w)

    mi_whole_accepted_pairs = set(zip(
        mechanics_df.loc[mi_whole_pass_mask, 'Experiment_Folder'],
        mechanics_df.loc[mi_whole_pass_mask, 'Trap_ID'],
    ))

    mi_whole_filtered_grouped_data = DiskBackedDict(results_dir / "cache_mi_whole")
    for gk, traps in all_grouped_data.items():
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) in mi_whole_accepted_pairs
        ]
        if kept:
            mi_whole_filtered_grouped_data.update({gk: kept})

    # ------------------------------------------------------------------
    # Attrition CSV (per-experiment tally of every candidate detection file)
    # ------------------------------------------------------------------
    attrition_df = loader.get_attrition_df()
    if not attrition_df.empty:
        attrition_path = results_dir / "attrition_per_experiment.csv"
        attrition_df.to_csv(attrition_path, index=False)
        logger.info(f"Saved attrition tally: {attrition_path.name}  "
                    f"({len(attrition_df)} experiments)")

    # ------------------------------------------------------------------
    # Figure-data bundle: every value behind a Chapter 3 figure, written to
    # CSV so figures can be redrawn without re-running the pipeline. The
    # per-cell scalars already live in mechanics_results_all_traps.csv; this
    # adds the time series, which previously existed only in memory.
    # ------------------------------------------------------------------
    _try("Figure Data Export",
         tfd.export_all,
         all_grouped_data, mechanics_df, attrition_df, results_dir)

    if mi_filtered_grouped_data.keys():
        # Cross-cohort pre-pulse figures land in prepulse_combined_dir;
        # per-bucket Thesis_MI_Boxplot_*.pdf files are routed via
        # prepulse_bucket_map into ASP/, 100V_100us/, 100V_5ms/.
        _try("Thesis Pre-Pulse Plot Suite",
             tp.run_thesis_mi_prepulse_plots,
             mi_filtered_grouped_data, mechanics_df,
             prepulse_combined_dir, global_pre_dur,
             prepulse_bucket_map)

    # ------------------------------------------------------------------
    # Claim 2 core — fate-mechanics story (Batch 3a).
    # Uses the FULL grouped data (not the pre-pulse-filtered subset) so
    # ruptured EP cells are visible. Filtering by fate + fit quality
    # happens inside each plot function. Pre-pulse fate boxplot lands
    # under pre-pulse/combined/, whole-trace fate boxplot and the
    # whole-trace-by-fate trace plot land under whole-trace/combined/.
    # ------------------------------------------------------------------
    _try("Thesis Claim 2 Core Plots",
         tp.run_thesis_claim2_plots,
         all_grouped_data, mechanics_df,
         prepulse_combined_dir, global_whole_dur, whole_trace_combined_dir,
         global_pre_dur)

    # ------------------------------------------------------------------
    # Fate cohort figures: pie chart per pulse condition + per-trap
    # intact/ruptured stacked bar per pulse condition. Both belong to
    # the at-pulse timeline slice; per-condition PDFs go into
    # 100V_100us/ and 100V_5ms/ via at_pulse_duration_map, while the
    # cross-condition CSV (Thesis_Fate_Counts_by_Pulse.csv) stays in
    # at_pulse_combined_dir.
    # ------------------------------------------------------------------
    _try("Fate Pies",       tp.plot_thesis_fate_counts,
         mechanics_df, attrition_df, at_pulse_combined_dir,
         at_pulse_duration_map)
    _try("Fate Per Trap",   tp.plot_thesis_fate_per_trap,
         mechanics_df, at_pulse_combined_dir,
         at_pulse_duration_map)

    # ------------------------------------------------------------------
    # Claim 2 EP-uptake / actin figures (Batch 3b):
    #   - Mean WT volume-normalised uptake trace I(t) across ASP / 100us / 5ms
    #   - Paired pre/post actin boxplots per treatment × pulse
    #   - Time-resolved mean actin trace aligned to the pulse frame
    # These use the full grouped data because they need the per-trap
    # actin / uptake CSVs; filtering to the analysis cohort happens
    # inside each plot function via mechanics_df. Routed to at-pulse
    # combined since all three are aligned to the pulse.
    # ------------------------------------------------------------------
    _try("Thesis Claim 2 Uptake / Actin Plots",
         tp.run_thesis_claim2_uptake_actin_plots,
         all_grouped_data, mechanics_df, at_pulse_combined_dir,
         at_pulse_duration_map)

    _try("Uptake plots (mean trace, per-trap A, correlations)",
         register_uptake_plots,
         all_grouped_data, mechanics_df, whole_trace_combined_dir)

    # ------------------------------------------------------------------
    # Whole-trace MI plots (was: supplementary/, now: whole-trace/combined/).
    # Includes the 5ms cohort which is too short pre-pulse for a matched
    # viscoelastic fit but produces usable whole-trace MI fits.
    # ------------------------------------------------------------------
    if mi_whole_filtered_grouped_data.keys():
        _try("Whole-Trace MI Plot Suite",
             tp.run_thesis_mi_wholetrace_plots,
             mi_whole_filtered_grouped_data, mechanics_df,
             whole_trace_combined_dir, global_whole_dur)

    logger.info("\n=== THESIS ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()