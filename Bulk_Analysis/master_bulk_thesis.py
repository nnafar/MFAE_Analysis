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
import bulk_mechanics     as bm
import thesis_plotting    as tp  
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
    supp_dir          = results_dir / "supplementary"
    for d in (mechanics_dir, claim1_dir, claim2_dir, supp_dir):
        d.mkdir(parents=True, exist_ok=True)

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
        _try("Thesis Plot Suite",
             tp.run_thesis_plots, filtered_grouped_data, mechanics_df, claim1_dir, r_eff, global_asp_dur)

    # ------------------------------------------------------------------
    # Claim 1 completion — actin, trap-dependency, body-volume sanity.
    # These are ASP-only figures showing that the platform detects the
    # CytD softening and that the readout doesn't confound with trap
    # position; body volume vs E is intentionally shown even though it
    # is not null (see chapter interpretation).
    # ------------------------------------------------------------------
    if filtered_grouped_data.keys():
        _try("Thesis Claim 1 Plots",
             tp.run_thesis_claim1_plots,
             filtered_grouped_data, mechanics_df, claim1_dir, global_asp_dur)

    # ------------------------------------------------------------------
    # MI cohort filter (both ASP and EP-pre)
    # ------------------------------------------------------------------
    mi_pass_mask = mechanics_df['MI_Best_Model'].notna() & (mechanics_df['MI_R2_Flag'] == True)

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
        mi_pass_mask = mi_pass_mask & common_conditions_mask

    mi_accepted_pairs = set(zip(
        mechanics_df.loc[mi_pass_mask, 'Experiment_Folder'],
        mechanics_df.loc[mi_pass_mask, 'Trap_ID'],
    ))

    mi_filtered_grouped_data = DiskBackedDict(results_dir / "cache_mi")
    for gk, traps in all_grouped_data.items():
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) in mi_accepted_pairs
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

    if mi_filtered_grouped_data.keys():
        _try("Thesis MI Pre-Pulse Plot Suite",
             tp.run_thesis_mi_prepulse_plots,
             mi_filtered_grouped_data, mechanics_df, claim2_dir, global_pre_dur)

    # ------------------------------------------------------------------
    # Claim 2 core — fate-mechanics story (Batch 3a).
    # Uses the FULL grouped data (not the MI-filtered subset) so ruptured
    # EP cells are visible.  Filtering by fate + MI R² happens inside the
    # plot functions.
    # ------------------------------------------------------------------
    _try("Thesis Claim 2 Core Plots",
         tp.run_thesis_claim2_plots,
         all_grouped_data, mechanics_df, claim2_dir, global_whole_dur)

    # ------------------------------------------------------------------
    # Fate cohort histogram: intact vs ruptured counts per pulse condition.
    # Belongs to Claim 2 (irreversible vs reversible electroporation).
    # ------------------------------------------------------------------
    _try("Fate Histogram", tp.plot_thesis_fate_counts,
         mechanics_df, attrition_df, claim2_dir)

    # ------------------------------------------------------------------
    # SUPPLEMENTARY: MI whole-trace plots.
    # With post-pulse-entry cells now excluded upstream, the whole-trace
    # comparison collapses to a two-bucket (ASP vs EP) view rather than
    # the three-bucket (ASP / EP-pre / EP-post) view it was designed for.
    # Kept here behind RUN_SUPPLEMENTARY so it can be produced on demand,
    # but not part of the default chapter figure set.
    RUN_SUPPLEMENTARY = False
    if RUN_SUPPLEMENTARY and mi_whole_filtered_grouped_data.keys():
        _try("SUPPLEMENTARY: MI Whole-Trace Plot Suite",
             tp.run_thesis_mi_wholetrace_plots,
             mi_whole_filtered_grouped_data, mechanics_df, supp_dir, global_whole_dur)

    logger.info("\n=== THESIS ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()