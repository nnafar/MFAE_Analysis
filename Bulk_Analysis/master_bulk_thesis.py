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
    logger.info(f"Results will be saved to: {results_dir}")

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
        if 'Uptake_PrePulse_QC' in mech_df.columns:
            contaminated_ids = set(zip(
                mech_df.loc[
                    mech_df['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded']),
                    'Experiment_Folder'
                ],
                mech_df.loc[
                    mech_df['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded']),
                    'Trap_ID'
                ],
            ))
            if contaminated_ids:
                clean_traps = [
                    t for t in traps
                    if (t.metadata.full_path.name, t.trap_id) not in contaminated_ids
                ]
                n_removed = len(traps) - len(clean_traps)
                grouped_data_chunk = {group_key: clean_traps}
                logger.info(
                    f"  QC filter: {n_removed} contaminated trap(s) removed "
                    f"from plots for group {group_key}."
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
    raw_path = results_dir / "mechanics_results_all_traps.csv"
    mechanics_df.to_csv(raw_path, index=False)
    logger.info(f"Saved unfiltered results to: {raw_path.name}  ({len(mechanics_df)} traps)")

    if 'Uptake_PrePulse_QC' in mechanics_df.columns:
        contaminated_mask = mechanics_df['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded'])
        excluded_df  = mechanics_df[contaminated_mask].copy()
        mechanics_df = mechanics_df[~contaminated_mask].copy()

        if not excluded_df.empty:
            excl_path = results_dir / "excluded_traps_prepulse_QC.csv"
            excluded_df.to_csv(excl_path, index=False)
            logger.info(
                f"Pre-pulse QC: {len(excluded_df)} EP trap(s) excluded "
                f"({excluded_df['Uptake_PrePulse_QC'].value_counts().to_dict()}).  "
                f"Saved to: {excl_path.name}"
            )
        else:
            logger.info("Pre-pulse QC: no contaminated traps found — all traps retained.")

    if 'Uptake_PrePulse_QC' in df_scalars.columns:
        df_scalars = df_scalars[
            ~df_scalars['Uptake_PrePulse_QC'].isin(['Pre_Leaky', 'Pre_Loaded'])
        ].copy()

    # --- Save master CSV (filtered) ---
    mech_path = results_dir / "mechanics_results.csv"
    mechanics_df.to_csv(mech_path, index=False)
    logger.info(f"Saved mechanics results to: {mech_path.name}  ({len(mechanics_df)} traps after QC)")

    # ------------------------------------------------------------------
    # Filter all_grouped_data to the viscoelastic-passing cohort.
    # ------------------------------------------------------------------
    # Thesis plots (per-trap boxplots, combined L(t) mean ± SD, and the
    # parameter boxplot) should all display the SAME cell population, or
    # readers will see e.g. a cell in the L(t) plot that was silently
    # excluded from the parameter boxplot. The accepted set uses exactly
    # the same mask that plot_asp_parameter_boxplots applies internally:
    #   - Condition_Type == 'ASP'
    #   - Best_Model is set (BIC picked a winner after the identifiability
    #     guard and bound-hit rejection)
    #   - E_Pa is finite (a real parameter, not NaN)
    #   - Visco_R2_Flag is True (the winner cleared the R² floor)
    # The unfiltered CSV (mechanics_results_all_traps.csv) remains as the
    # audit trail; only the plotting cohort is filtered here.
    asp_pass_mask = (
        (mechanics_df['Condition_Type'] == 'ASP') &
        mechanics_df['Best_Model'].notna() &
        mechanics_df['E_Pa'].notna() &
        (mechanics_df['Visco_R2_Flag'] == True)
    )
    accepted_pairs = set(zip(
        mechanics_df.loc[asp_pass_mask, 'Experiment_Folder'],
        mechanics_df.loc[asp_pass_mask, 'Trap_ID'],
    ))
    logger.info(
        f"Viscoelastic-passing cohort: {len(accepted_pairs)} "
        f"(Experiment_Folder, Trap_ID) pairs accepted from {asp_pass_mask.sum()} "
        f"ASP rows (of {(mechanics_df['Condition_Type'] == 'ASP').sum()} total ASP traps)."
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
             tp.run_thesis_plots, filtered_grouped_data, mechanics_df, results_dir, r_eff)

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

    # --- Execute Thesis Plots for MI (ASP full trace vs EP pre-pulse) ---
    if mi_filtered_grouped_data:
        _try("Thesis MI Plot Suite",
             tp.run_thesis_mi_plots,
             mi_filtered_grouped_data, mechanics_df, results_dir, r2_floor)

    logger.info("\n=== THESIS ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()