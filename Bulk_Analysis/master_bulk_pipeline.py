# -*- coding: utf-8 -*-
"""
Master Pipeline for Bulk MFAE Analysis.

THREE PHASES
------------
Phase 1 -- Data Loading & Initialisation
    Scans the output directory, parses folder names, and prepares the
    iterator that yields groups of CSV data sequentially to prevent
    memory overallocation.

Phase 2 -- Per-Group Processing (iterative)
    For each condition group (CellType × Treatment × Pressure × Voltage × Duration):
      a) Run all mechanics fitting (viscoelastic, model-independent, slopes, uptake tau).
      b) Generate per-group verification plots (multipanel grids, per-trap distributions).
    Results are accumulated in memory for aggregation in Phase 3.

Phase 3 -- Global Aggregation & Cross-Group Plots
    Concatenates all per-group results, saves the master CSV, and generates
    global comparison plots (ASP vs EP uptake, slope comparison, Spearman
    correlation, viscoelastic parameter distributions).

Each plot function is wrapped in _try() so one failure never stops the
rest of the pipeline.
"""

import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

import bulk_file_handling as bfh
import bulk_mechanics     as bm
import bulk_plotting      as bp
import bulk_spatial       as bs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # Path configuration
    # ------------------------------------------------------------------
    output_root_dir = Path(r"C:\GitHub\MFAE_Analysis\Output")

    # NEW: Create a date-stamped subfolder for today's run.
    # Format: YYMMDD (e.g. "250701").  Each run gets its own folder so
    # you can compare results across days without overwriting.
    date_stamp  = datetime.now().strftime("%y%m%d")
    results_dir = output_root_dir / "Bulk_Analysis_results" / date_stamp
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # ------------------------------------------------------------------
    # Analysis settings
    # ------------------------------------------------------------------
    r_eff = bm.DEFAULT_R_EFF
    r2_floor = 0.8

    logger.info(f"Using r_eff = {r_eff:.3f} um  (channel: 6.7 x 5.0 um, C=1.0)")
    logger.info(f"R² quality floor = {r2_floor}")

    # Helper: wrap any plot call so a single failure never kills the pipeline
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
    # PHASE 2 -- PER-GROUP MECHANICS + VERIFICATION PLOTS
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: PER-GROUP MECHANICS & VERIFICATION PLOTS")
    logger.info("=" * 50)

    all_mechanics_dfs = []
    all_scalars_dfs   = []
    all_grouped_data  = {}

    # Track which condition types are present so we can skip
    # EP-only plots when the dataset contains only ASP data.
    has_ep_data  = False
    has_asp_data = False

    for group_key, traps in loader.iter_groups():
        grouped_data_chunk = {group_key: traps}

        # Update condition type flags
        ctype = traps[0].metadata.condition_type if traps else None
        if ctype == "EP":
            has_ep_data = True
        elif ctype == "ASP":
            has_asp_data = True

        # --- 2a. Mechanics fitting ---
        mech_df = bm.run_all_mechanics(
            grouped_data_chunk, r_eff=r_eff, C=bm.HALFSPACE_C, r2_floor=r2_floor
        )
        all_mechanics_dfs.append(mech_df)

        scalar_df = bfh.extract_all_scalars(grouped_data_chunk)
        all_scalars_dfs.append(scalar_df)

        # --- 2b. QC filtering ---
        # run_all_mechanics has already computed Uptake_PrePulse_QC for every
        # trap in this group.  Strip Pre_Leaky / Pre_Loaded traps from
        # grouped_data_chunk NOW so that every plot below — and all_grouped_data
        # in Phase 3 — only ever sees clean cells.
        #
        # all_mechanics_dfs and all_scalars_dfs intentionally receive the full
        # unfiltered data; Phase 3 does its own DataFrame-level filtering before
        # saving the master CSV, keeping both pipelines consistent.
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

        # --- 2c. Per-group verification plots ---
        # Multipanel verification grids
        _try("Uptake fits multipanel",
             bp.plot_uptake_fits_multipanel, grouped_data_chunk, results_dir)
        _try("Uptake traces multipanel",
             bp.plot_uptake_traces_multipanel, grouped_data_chunk, results_dir)
        _try("Viscoelastic fits multipanel",
             bp.plot_viscoelastic_fits_multipanel,
             grouped_data_chunk, results_dir, r_eff, bm.HALFSPACE_C)

        # NEW: Model-independent fits multipanel (both ASP and EP)
        _try("Model-independent fits multipanel",
             bp.plot_model_independent_fits_multipanel,
             grouped_data_chunk, results_dir)

        # EP-only multipanel plots — skip for ASP-only groups
        if ctype == "EP":
            _try("Recoil fits multipanel",
                 bp.plot_recoil_fits_multipanel, grouped_data_chunk, results_dir)

        # Per-group aggregate plots
        _try("Uptake dynamics",
             bp.plot_uptake_dynamics, grouped_data_chunk, results_dir)
        _try("Per-trap protrusion distribution",
             bp.plot_per_trap_protrusion_distribution, grouped_data_chunk, results_dir)
        _try("Per-trap uptake distribution",
             bp.plot_per_trap_uptake_distribution, grouped_data_chunk, results_dir)
        _try("Uptake exponential fit",
             bp.plot_uptake_exponential_fit, grouped_data_chunk, results_dir)

        # Accumulate for cross-group plots in Phase 3
        all_grouped_data.update(grouped_data_chunk)

    if not all_mechanics_dfs:
        logger.error("No valid experiment folders found or processed. Aborting pipeline.")
        return

    # ------------------------------------------------------------------
    # PHASE 3 -- GLOBAL AGGREGATION & CROSS-GROUP PLOTS
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: GLOBAL AGGREGATION & CROSS-GROUP PLOTS")
    logger.info("=" * 50)

    mechanics_df = pd.concat(all_mechanics_dfs, ignore_index=True)
    df_scalars   = pd.concat(all_scalars_dfs,   ignore_index=True)

    # ------------------------------------------------------------------
    # Pre-pulse QC filtering (DataFrame level)
    # ------------------------------------------------------------------
    # Phase 2 already removed contaminated TrapData objects from every plot.
    # Here we do the same for mechanics_df and df_scalars so the master CSV
    # and all Phase 3 plots are consistent.
    #
    # The unfiltered table is saved first as an audit trail.
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

    asp_df = mechanics_df[mechanics_df['Condition_Type'] == 'ASP']
    if not asp_df.empty and 'Best_Model' in asp_df.columns:
        counts = asp_df['Best_Model'].value_counts()
        logger.info("  ASP best-model counts:\n" + counts.to_string())

    # Log MI model selection summary
    if 'MI_Best_Model' in mechanics_df.columns:
        mi_counts = mechanics_df['MI_Best_Model'].value_counts()
        logger.info("  MI best-model counts:\n" + mi_counts.to_string())

    # --- Global mechanics plots ---
    _try("Model-independent comparison (ASP vs EP)",
         bp.plot_model_independent_comparison, mechanics_df, results_dir)
    _try("Uptake tau boxplots",
         bp.plot_uptake_tau_boxplot, mechanics_df, results_dir)
    _try("Slope comparison (ASP vs EP)",
         bp.plot_slope_comparison, mechanics_df, results_dir)

    # --- EP-only global plots (skip when dataset is ASP-only) ---
    if has_ep_data:
        _try("EP post-pulse behavior",
             bp.plot_ep_post_pulse_behavior, mechanics_df, results_dir)

        # --- Spatial field analysis (EP only) ---
        # Run each spatial plot for both the cell body and the protrusion ROI.
        # Body uptake reflects bulk cytoplasmic dye accumulation; protrusion
        # uptake is closer to the pore-entry site and may show a stronger
        # gradient if the field is asymmetric.
        for _region in ('body', 'protrusion'):
            # Q1+Q2: Does uptake vary across trap positions?  Is there a gradient?
            _try(f"Spatial trap gradient ({_region})",
                 bs.plot_trap_gradient, mechanics_df, results_dir, _region)
            # Visual check for replicate-to-replicate consistency of the gradient
            _try(f"Spatial heatmap (trap × experiment) ({_region})",
                 bs.plot_ep_spatial_heatmap, mechanics_df, results_dir, _region)
            # Q3: Is cell body area a confound for the spatial gradient?
            _try(f"Cell size spatial confound ({_region})",
                 bs.plot_cell_size_spatial, mechanics_df, results_dir, _region)
    else:
        logger.info("Skipping EP-only plots (no EP data in this dataset).")

    # --- Cross-group plots (need all conditions together) ---
    if all_grouped_data:
        _try("Uptake ASP vs EP overlay",
             bp.plot_uptake_asp_vs_ep, all_grouped_data, results_dir)
        _try("ASP best-fit multipanel (reduced grouping)",
             bp.plot_asp_best_fit_multipanel,
             all_grouped_data, results_dir, r_eff, bm.HALFSPACE_C)

    # --- ASP parameter boxplots (reduced grouping: CellType × Treatment × Pressure) ---
    _try("ASP parameter boxplots",
         bp.plot_asp_parameter_boxplots, mechanics_df, results_dir)

    # --- Spearman correlation (needs all scalars) ---
    # Merge the fitted exponential amplitude A (ADU) from mechanics_df into
    # df_scalars so the correlation plots use the same uptake metric as the
    # mechanics CSV, rather than the raw nanmax from extract_all_scalars.
    #
    # Join key: (Date, Experiment_Number, Trap_ID) uniquely identifies each
    # trap across all experiments.  how='left' preserves all df_scalars rows;
    # traps where fitting failed get NaN for the amplitude columns.
    if not df_scalars.empty:
        fit_a_cols = ['Date', 'Experiment_Number', 'Trap_ID',
                      'Uptake_Body_Abs_A', 'Uptake_Prot_Abs_A', 'Uptake_Total_Abs_A']
        available_fit_cols = [c for c in fit_a_cols if c in mechanics_df.columns]
        df_scalars = df_scalars.merge(
            mechanics_df[available_fit_cols].drop_duplicates(
                subset=['Date', 'Experiment_Number', 'Trap_ID']
            ),
            on=['Date', 'Experiment_Number', 'Trap_ID'],
            how='left',
        )
        logger.info(
            f"Merged fitted amplitude A columns into df_scalars "
            f"({df_scalars['Uptake_Body_Abs_A'].notna().sum()} / "
            f"{len(df_scalars)} traps have body A)."
        )

        _try("Spearman correlation",
             bp.plot_spearman_correlation, df_scalars, results_dir)
        _try("Parameter pairplot",
             bp.plot_parameter_pairplot, df_scalars, results_dir)

    logger.info("\n=== BULK ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()