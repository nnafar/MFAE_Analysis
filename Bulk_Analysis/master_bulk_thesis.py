# -*- coding: utf-8 -*-
"""
Master Pipeline for Bulk MFAE Analysis (Thesis Plotting Mode).
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd

import bulk_file_handling as bfh
import bulk_mechanics     as bm
import bulk_pi_baseline   as bpb
import thesis_plotting    as tp  
import Utils_MFA as utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
# Silence the PDF font embedding spam
logging.getLogger('fontTools').setLevel(logging.WARNING) 

# Matplotlib emits an INFO message every time a categorical axis is used
# ("Using categorical units to plot a list of strings..."). Seaborn boxplots
# with string x-labels trigger it repeatedly; demote to WARNING.
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)

# Pandas 2.x: concatenating DataFrames with empty/all-NA columns is being
# deprecated. Not relevant here (we intentionally have some all-NA columns
# for EP-only fields on ASP rows and vice versa), so silence it.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*DataFrame concatenation with empty or all-NA entries.*",
)

logger = logging.getLogger(__name__)

def main():
    # ------------------------------------------------------------------
    # Path configuration
    # ------------------------------------------------------------------
    output_root_dir = Path(r"C:\GitHub\MFAE_Analysis\Output")

    date_stamp  = datetime.now().strftime("%y%m%d")
    results_dir = output_root_dir / "Bulk_Analysis_results" / date_stamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Output subfolder structure
    # ------------------------------------------------------------------
    # Split output into topic-scoped subfolders so the results directory
    # stays scannable at a glance. Each downstream call receives the
    # relevant subfolder as its output_dir.
    mechanics_dir     = results_dir / "mechanics"          # CSVs
    qc_dir            = results_dir / "qc"                 # PI_Baseline diagnostic
    asp_visco_dir     = results_dir / "asp_viscoelastic"   # Viscoelastic thesis plots
    mi_prepulse_dir   = results_dir / "mi_prepulse"        # MI ASP vs EP-pre (Plot 1)
    mi_wholetrace_dir = results_dir / "mi_wholetrace"      # MI whole-trace (Plot 2)
    for d in (mechanics_dir, qc_dir, asp_visco_dir, mi_prepulse_dir, mi_wholetrace_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(
        f"  Subfolders: mechanics/, qc/, asp_viscoelastic/, "
        f"mi_prepulse/, mi_wholetrace/"
    )

    # ------------------------------------------------------------------
    # Analysis settings
    # ------------------------------------------------------------------
    r_eff = bm.DEFAULT_R_EFF
    r2_floor = 0.85

    logger.info(f"Using r_eff = {r_eff:.3f} um  (channel: 6.7 x 5.0 um, C=1.0)")
    logger.info(f"R² quality floor = {r2_floor}")

    def _try(label, func, *args):
        try:
            func(*args)
        except Exception as e:
            logger.error(f"{label} failed: {e}", exc_info=False)

    # ------------------------------------------------------------------
    # PHASE 1 -- DATA LOADING
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: DATA LOADING")
    logger.info("=" * 50)

    try:
        loader = bfh.BulkDataLoader(str(output_root_dir))
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return

    # ------------------------------------------------------------------
    # PHASE 2 -- PER-GROUP MECHANICS
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: PER-GROUP MECHANICS (Plotting Disabled)")
    logger.info("=" * 50)

    all_mechanics_dfs = []
    all_scalars_dfs   = []
    all_grouped_data  = {}

    for group_key, traps in loader.iter_groups():
        grouped_data_chunk = {group_key: traps}

        # --- 2a. Mechanics fitting ---
        mech_df = bm.run_all_mechanics(
            grouped_data_chunk, r_eff=r_eff, C=bm.HALFSPACE_C, r2_floor=r2_floor
        )
        all_mechanics_dfs.append(mech_df)

        scalar_df = bfh.extract_all_scalars(grouped_data_chunk)
        all_scalars_dfs.append(scalar_df)

        # --- 2b. QC filtering ---
        # EP-only scoping mirrors the DataFrame filter in Phase 3: ASP cells
        # keep their Uptake_PrePulse_QC label as an audit trail but are NOT
        # excluded from downstream plots on the F0/leaky criteria. Without
        # this scope, CytD ASP cells (which routinely trip Pre_Loaded) get
        # dropped from all_grouped_data before any thesis cohort is built,
        # so they never appear in the ASP L(t) curve or per-trap boxplots.
        if 'Uptake_PrePulse_QC' in mech_df.columns:
            contaminated_flag = (
                (mech_df['Condition_Type'] == 'EP')
                & mech_df['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded'])
            )
            contaminated_ids = set(zip(
                mech_df.loc[contaminated_flag, 'Experiment_Folder'],
                mech_df.loc[contaminated_flag, 'Trap_ID'],
            ))
            if contaminated_ids:
                clean_traps = [
                    t for t in traps
                    if (t.metadata.full_path.name, t.trap_id) not in contaminated_ids
                ]
                n_removed = len(traps) - len(clean_traps)
                grouped_data_chunk = {group_key: clean_traps}
                logger.info(
                    f"  QC filter (EP-only): {n_removed} contaminated EP trap(s) "
                    f"removed from plots for group {group_key}."
                )

        # Accumulate for cross-group plots in Phase 3
        all_grouped_data.update(grouped_data_chunk)

    if not all_mechanics_dfs:
        logger.error("No valid experiment folders found or processed. Aborting pipeline.")
        return

    # ------------------------------------------------------------------
    # PHASE 3 -- GLOBAL AGGREGATION & THESIS PLOTS
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: GLOBAL AGGREGATION & THESIS PLOTS")
    logger.info("=" * 50)

    mechanics_df = pd.concat(all_mechanics_dfs, ignore_index=True)
    df_scalars   = pd.concat(all_scalars_dfs,   ignore_index=True)

    # ------------------------------------------------------------------
    # Pre-pulse QC filtering (DataFrame level)
    # ------------------------------------------------------------------
    raw_path = mechanics_dir / "mechanics_results_all_traps.csv"
    mechanics_df.to_csv(raw_path, index=False)
    logger.info(f"Saved unfiltered results to: mechanics/{raw_path.name}  ({len(mechanics_df)} traps)")

    if 'Uptake_PrePulse_QC' in mechanics_df.columns:
        # Full QC breakdown -- helpful to distinguish 'excluded because
        # contaminated' from 'skipped because dataset has no dye channel'.
        qc_counts = mechanics_df['Uptake_PrePulse_QC'].value_counts(dropna=False).to_dict()
        logger.info(f"Pre-pulse QC category counts (pre-exclusion): {qc_counts}")

        # Only EP cells get excluded on QC contamination. ASP cells retain
        # their QC label for the audit trail but stay in the cohort — the
        # F0-based Pre_Loaded check is dye-timing-sensitive in ways that do
        # not translate cleanly to aspiration-only experiments, and manually
        # marked ASP DOA cells are already handled by bulk_pi_baseline via
        # the '_doa' filename tag.
        is_ep_row = mechanics_df['Condition_Type'] == 'EP'
        contaminated_mask = (
            is_ep_row
            & mechanics_df['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded'])
        )
        excluded_df  = mechanics_df[contaminated_mask].copy()
        mechanics_df = mechanics_df[~contaminated_mask].copy()

        # Log ASP cells that WOULD have been flagged, as a diagnostic. If
        # this count is high, revisit whether the F0 threshold makes sense
        # for aspiration-only experiments too.
        asp_would_have_flagged = int(
            ((mechanics_df['Condition_Type'] == 'ASP')
             & mechanics_df['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded'])
            ).sum()
        )
        if asp_would_have_flagged:
            logger.info(
                f"Pre-pulse QC: {asp_would_have_flagged} ASP trap(s) carry a "
                f"Pre_Loaded/Pre_Leaky label but were RETAINED (ASP exempt from "
                f"this filter). See mechanics_results.csv Uptake_PrePulse_QC "
                f"column for details."
            )

        if not excluded_df.empty:
            excl_path = mechanics_dir / "excluded_traps_prepulse_QC.csv"
            excluded_df.to_csv(excl_path, index=False)
            # Split contaminated counts by condition so ASP vs EP is visible.
            by_cond = (
                excluded_df.groupby('Condition_Type')['Uptake_PrePulse_QC']
                .value_counts()
                .to_dict()
            )
            logger.info(
                f"Pre-pulse QC: {len(excluded_df)} EP trap(s) excluded  "
                f"[breakdown by (Condition_Type, QC): {by_cond}].  "
                f"Saved to: mechanics/{excl_path.name}"
            )
        else:
            logger.info("Pre-pulse QC: no EP traps flagged for contamination.")

    if 'Uptake_PrePulse_QC' in df_scalars.columns:
        # Same EP-only scoping for the scalars table.
        is_ep_row_s = df_scalars['Condition_Type'] == 'EP'
        df_scalars = df_scalars[
            ~(is_ep_row_s & df_scalars['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded']))
        ].copy()

    # --- Save master CSV (filtered) ---
    mech_path = mechanics_dir / "mechanics_results.csv"
    mechanics_df.to_csv(mech_path, index=False)
    logger.info(f"Saved mechanics results to: mechanics/{mech_path.name}  ({len(mechanics_df)} traps after QC)")

    # ------------------------------------------------------------------
    # PI-based DOA filter (complements the F0-based Pre_Loaded check)
    # ------------------------------------------------------------------
    # The existing Uptake_PrePulse_QC catches dead-on-arrival cells via an
    # absolute F0 threshold on the raw dye intensity (PRE_LOADED_F0_THRESHOLD
    # = 1500 ADU). This step adds a volume-normalized second pass on
    # Body_VolNorm[:5].mean() so cells that slip through the F0 check --
    # typically datasets where F0 reconstruction from dF/F0 columns is noisy
    # or where the whole dataset was stamped 'No_Dye_Channel' at the group
    # level -- still get caught before they enter any downstream cohort.
    #
    # Threshold is calibrated on manually-inspected DOA examples (see
    # bulk_pi_baseline.DEFAULT_PI_BASELINE_THRESHOLD). Tune based on the
    # PI_Baseline_Distribution.pdf diagnostic in results_dir.
    all_grouped_data, mechanics_df, pi_df = bpb.run_pi_doa_filter(
        all_grouped_data,
        mechanics_df,
        output_dir=qc_dir,
        metric_output_dir=mechanics_dir,
        fallback_threshold=bpb.DEFAULT_PI_BASELINE_THRESHOLD,
        safety_factor=bpb.DEFAULT_SAFETY_FACTOR,
        min_cal_samples=bpb.DEFAULT_MIN_CAL_SAMPLES,
        n_baseline=bpb.DEFAULT_N_BASELINE,
    )

    # Drop DOA-flagged rows from the plotting mechanics table. The raw
    # mechanics_results_all_traps.csv (saved earlier) remains the audit
    # trail; the filtered table below is what feeds cohort masks.
    n_before_pi = len(mechanics_df)
    mechanics_df = mechanics_df[~mechanics_df['PI_DOA_Flag']].copy()
    n_pi_excluded = n_before_pi - len(mechanics_df)
    if n_pi_excluded:
        logger.info(
            f"PI_DOA filter: {n_pi_excluded} row(s) removed from mechanics_df "
            f"(retained: {len(mechanics_df)})."
        )

    # ------------------------------------------------------------------
    # Filter all_grouped_data to the viscoelastic-passing cohort.
    # ------------------------------------------------------------------
    # Thesis plots (per-trap boxplots, combined L(t) mean ± SD, and the
    # parameter boxplot) should all display the SAME cell population, or
    # readers will see e.g. a cell in the L(t) plot that was silently
    # excluded from the parameter boxplot. The accepted set uses exactly
    # the same mask that plot_asp_parameter_boxplots applies internally:
    #   - Condition_Type == 'ASP' OR Post_Pulse_Entry_Flag (both get a
    #     whole-trace viscoelastic fit; see bulk_mechanics is_asp_like)
    #   - Best_Model is set (BIC picked a winner after the identifiability
    #     guard and bound-hit rejection)
    #   - E_Pa is finite (a real parameter, not NaN)
    #   - Visco_R2_Flag is True (the winner cleared the R² floor)
    # The unfiltered CSV (mechanics_results_all_traps.csv) remains as the
    # audit trail; only the plotting cohort is filtered here.
    is_asp_like_row = (
        (mechanics_df['Condition_Type'] == 'ASP')
        | (mechanics_df.get('Post_Pulse_Entry_Flag', False) == True)
    )
    asp_pass_mask = (
        is_asp_like_row
        & mechanics_df['Best_Model'].notna()
        & mechanics_df['E_Pa'].notna()
        & (mechanics_df['Visco_R2_Flag'] == True)
    )
    accepted_pairs = set(zip(
        mechanics_df.loc[asp_pass_mask, 'Experiment_Folder'],
        mechanics_df.loc[asp_pass_mask, 'Trap_ID'],
    ))
    n_asp = int((mechanics_df.loc[asp_pass_mask, 'Condition_Type'] == 'ASP').sum())
    n_post = int(asp_pass_mask.sum()) - n_asp
    logger.info(
        f"Viscoelastic-passing cohort: {len(accepted_pairs)} "
        f"(Experiment_Folder, Trap_ID) pairs accepted "
        f"[{n_asp} ASP + {n_post} EP_Post_Entry] "
        f"from {is_asp_like_row.sum()} ASP-like rows."
    )

    n_before, n_after = 0, 0
    filtered_grouped_data: dict = {}
    for gk, traps in all_grouped_data.items():
        # Non-ASP groups (e.g. EP) pass through untouched -- the filter is
        # only meaningful for cells that had a viscoelastic fit attempted.
        if traps and traps[0].metadata.condition_type != "ASP":
            filtered_grouped_data[gk] = traps
            continue
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) in accepted_pairs
        ]
        n_before += len(traps)
        n_after  += len(kept)
        if kept:
            filtered_grouped_data[gk] = kept
    logger.info(
        f"ASP trap objects: {n_before} before viscoelastic filter, "
        f"{n_after} after ({n_before - n_after} dropped)."
    )

    # --- Execute Thesis Plots for ASP ---
    if filtered_grouped_data:
        _try("Thesis Plot Suite",
             tp.run_thesis_plots, filtered_grouped_data, mechanics_df, asp_visco_dir, r_eff)

    # ------------------------------------------------------------------
    # MI cohort filter (both ASP and EP)
    # ------------------------------------------------------------------
    # Model-independent thesis plots use a separate cohort from the
    # viscoelastic-passing set: EP traps do not receive a viscoelastic fit,
    # but they do receive MI fits over their pre-pulse window, so they
    # belong in the MI cohort. The rule is:
    #   - MI_Best_Model is set (Linear or Power-Law survived BIC selection)
    #   - Winner's R² >= r2_floor
    # This mirrors, on the MI side, what Visco_R2_Flag does on the
    # viscoelastic side. Both the master-level grouped_data filter here and
    # the DataFrame filter inside plot_thesis_mi_parameter_boxplots use the
    # same rule so cell counts across MI plots match.
    def _mi_winner_r2(row):
        m = row.get('MI_Best_Model')
        if m == 'Linear':
            return row.get('Linear_R2')
        if m == 'Power-Law':
            return row.get('PL_R2')
        return None

    mi_winner_r2 = mechanics_df.apply(_mi_winner_r2, axis=1)
    mi_pass_mask = mechanics_df['MI_Best_Model'].notna() & (mi_winner_r2 >= r2_floor)
    # Post-pulse-entry EP cells have a whole-trace MI fit but do not belong
    # in the ASP-vs-EP-pre comparison. Exclude them from the MI cohort here,
    # matching the same filter used inside the MI plot functions.
    if 'Post_Pulse_Entry_Flag' in mechanics_df.columns:
        mi_pass_mask = mi_pass_mask & (~mechanics_df['Post_Pulse_Entry_Flag'].astype(bool))

    # Common-conditions filter: restrict ASP cells to treatments that also
    # appear in EP data. Rationale — the MI comparison is ASP-mechanics vs
    # pre-pulse-EP-mechanics. If a treatment (e.g. CytD) has ASP data but no
    # EP counterpart, including it on the ASP side compares treatment classes
    # that were never actually electroporated in matched conditions. Keep
    # CytD (or any treatment lacking an EP counterpart) in the viscoelastic
    # ASP suite where the aspiration-only comparison is the whole point.
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

        dropped_asp_treatments = set(
            mechanics_df.loc[
                (mechanics_df['Condition_Type'] == 'ASP')
                & ~mechanics_df['Treatment'].isin(ep_treatments),
                'Treatment'
            ].dropna().unique()
        )
        if dropped_asp_treatments:
            logger.info(
                f"MI common-conditions filter: dropping ASP treatment(s) "
                f"{sorted(dropped_asp_treatments)} from MI cohort (no EP "
                f"counterpart). EP treatments present: {sorted(ep_treatments)}."
            )
        mi_pass_mask = mi_pass_mask & common_conditions_mask
    else:
        logger.info(
            "MI common-conditions filter: no EP treatments present, "
            "skipping filter (all treatments retained in MI cohort)."
        )

    mi_accepted_pairs = set(zip(
        mechanics_df.loc[mi_pass_mask, 'Experiment_Folder'],
        mechanics_df.loc[mi_pass_mask, 'Trap_ID'],
    ))
    logger.info(
        f"MI-passing cohort: {len(mi_accepted_pairs)} "
        f"(Experiment_Folder, Trap_ID) pairs accepted "
        f"(ASP={((mechanics_df.loc[mi_pass_mask, 'Condition_Type'] == 'ASP').sum())}, "
        f"EP={((mechanics_df.loc[mi_pass_mask, 'Condition_Type'] == 'EP').sum())})."
    )

    mi_filtered_grouped_data: dict = {}
    for gk, traps in all_grouped_data.items():
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) in mi_accepted_pairs
        ]
        if kept:
            mi_filtered_grouped_data[gk] = kept

    # ------------------------------------------------------------------
    # MI_Whole cohort filter (whole-trace comparison)
    # ------------------------------------------------------------------
    # For the whole-trace MI comparison (ASP whole + EP-whole + EP-post
    # whole), the cohort uses the MI_Whole_* columns instead of the primary
    # MI_* columns. Post-pulse-entry cells ARE included here — that's the
    # whole point of this comparison type.
    def _mi_whole_winner_r2(row):
        m = row.get('MI_Whole_Best_Model')
        if m == 'Linear':
            return row.get('MI_Whole_Linear_R2')
        if m == 'Power-Law':
            return row.get('MI_Whole_PL_R2')
        return None

    mi_whole_winner_r2 = mechanics_df.apply(_mi_whole_winner_r2, axis=1)
    mi_whole_pass_mask = (
        mechanics_df['MI_Whole_Best_Model'].notna()
        & (mi_whole_winner_r2 >= r2_floor)
    )
    # Same common-conditions filter as the pre-pulse MI cohort. `ep_treatments`
    # was computed above and includes both standard EP and post-pulse-entry
    # rows (they share the 'EP' Condition_Type).
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

    # Split counts for the audit log so ASP / EP-whole / EP-post are all
    # visible at a glance.
    mw_sub = mechanics_df.loc[mi_whole_pass_mask]
    n_asp   = int((mw_sub['Condition_Type'] == 'ASP').sum())
    if 'Post_Pulse_Entry_Flag' in mw_sub.columns:
        ep_mask = mw_sub['Condition_Type'] == 'EP'
        n_ep_whole = int((ep_mask & ~mw_sub['Post_Pulse_Entry_Flag'].astype(bool)).sum())
        n_ep_post  = int((ep_mask &  mw_sub['Post_Pulse_Entry_Flag'].astype(bool)).sum())
    else:
        n_ep_whole = int((mw_sub['Condition_Type'] == 'EP').sum())
        n_ep_post  = 0
    logger.info(
        f"MI-Whole-passing cohort: {len(mi_whole_accepted_pairs)} "
        f"(Experiment_Folder, Trap_ID) pairs accepted "
        f"[ASP={n_asp}, EP-whole={n_ep_whole}, EP-post={n_ep_post}]."
    )

    mi_whole_filtered_grouped_data: dict = {}
    for gk, traps in all_grouped_data.items():
        kept = [
            t for t in traps
            if (t.metadata.full_path.name, t.trap_id) in mi_whole_accepted_pairs
        ]
        if kept:
            mi_whole_filtered_grouped_data[gk] = kept

    # --- Execute Thesis Plots for MI (ASP full trace vs EP pre-pulse) ---
    if mi_filtered_grouped_data:
        _try("Thesis MI Pre-Pulse Plot Suite",
             tp.run_thesis_mi_prepulse_plots,
             mi_filtered_grouped_data, mechanics_df, mi_prepulse_dir, r2_floor)

    # --- Execute Thesis Plots for MI (whole-trace comparison: ASP + EP-whole + EP-post) ---
    if mi_whole_filtered_grouped_data:
        _try("Thesis MI Whole-Trace Plot Suite",
             tp.run_thesis_mi_wholetrace_plots,
             mi_whole_filtered_grouped_data, mechanics_df, mi_wholetrace_dir, r2_floor)

    logger.info("\n=== THESIS ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()