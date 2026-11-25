# -*- coding: utf-8 -*-
"""
Bulk analysis: pools the per-experiment results in Output/ into condition-level
statistics and figures.

    python run_bulk_analysis.py [--output-root PATH]

Results: Output/Bulk_Analysis_results/<YYMMDD_HHMM>/ with a README.txt
(layout in mfae/bulk/layout.py):
    0_Tables/                 one row per trap, attrition, Figure_Data/
    1_Aspiration_Mechanics/   aspiration-only mechanics (<Treatment>/, All_treatments/)
    2_Before_Pulse/           pulsed vs aspiration-only before the pulse
    3_At_Pulse/               fate, uptake and actin at the pulse
    4_Whole_Trace/            whole-recording fits and uptake
"""

import logging
import warnings
import pickle
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from mfae.bulk import loader as bfh
from mfae.bulk import figure_data as tfd
from mfae.bulk import mechanics as bm
from mfae.bulk.plots import figures as tp  
from mfae.bulk.plots.uptake import register_uptake_plots
from mfae.bulk import utils
from mfae.bulk import layout as bl

from mfae.paths import DEFAULT_OUTPUT_ROOT

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Console logging for a bulk run (called from main, not on import)."""
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

# Per-experiment results written by run_experiment.py (repo-level "Output" folder).


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="python run_bulk_analysis.py",
        description="Pool per-experiment MFA results into condition-level statistics and figures.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Folder holding the per-experiment result folders from run_experiment.py "
             f"(default: %(default)s). Results go to <output-root>/Bulk_Analysis_results/<YYMMDD_HHMM>.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the bulk analysis; returns 0 on success (exit code)."""
    _configure_logging()
    args = _parse_args(argv)
    output_root_dir = args.output_root
    if not output_root_dir.is_dir():
        logger.error(f"Output folder not found: {output_root_dir}. "
                     "Run run_experiment.py on your experiments first, or pass --output-root.")
        return 1

    # One folder per run, named by date and time, so runs never overwrite
    # each other. The folders are made once the conditions present are known
    # (see mfae/bulk/layout.py and the README.txt written at the end).
    run_stamp   = datetime.now().strftime("%y%m%d_%H%M")
    results_dir = output_root_dir / "Bulk_Analysis_results" / run_stamp
    results_dir.mkdir(parents=True, exist_ok=True)
    # Working caches (pickled trap data); removed when the run ends.
    cache_root  = results_dir / "_cache"
    try:
        _run(output_root_dir, results_dir, cache_root)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)
        if results_dir.is_dir():
            try:
                bl.write_readme(results_dir, output_root_dir)
            except Exception as e:
                logger.error(f"README failed: {e}")
        logger.info(f"Results: {results_dir}  (see README.txt)")
    return 0


def _run(output_root_dir: Path, results_dir: Path, cache_root: Path) -> None:
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

    all_grouped_data = DiskBackedDict(cache_root / "all")
    for group_key, traps in loader.iter_groups():
        all_grouped_data.update({group_key: traps})

    # --- Output folders for the conditions present --------------------------
    asp_treatments, pulses = set(), set()
    for gk, traps in all_grouped_data.items():
        if not traps:
            continue
        m = traps[0].metadata
        if m.condition_type == 'ASP':
            asp_treatments.add(m.treatment)
        elif m.condition_type == 'EP':
            pulses.add((m.voltage, m.duration_label))
    out = bl.build(results_dir, asp_treatments, pulses)
    claim1_combined_dir      = out['asp_all']           # 1_Aspiration_Mechanics/All_treatments
    claim1_treatment_map     = out['asp_by_treatment']  # 1_Aspiration_Mechanics/<Treatment>
    prepulse_combined_dir    = out['pre_all']           # 2_Before_Pulse/All_conditions
    prepulse_bucket_map      = out['pre_by_pulse']      # 2_Before_Pulse/{Aspiration,<pulse>}
    at_pulse_combined_dir    = out['at_all']            # 3_At_Pulse/All_conditions
    at_pulse_duration_map    = out['at_by_pulse']       # 3_At_Pulse/<pulse>
    whole_trace_combined_dir = out['whole_all']         # 4_Whole_Trace/All_conditions
    logger.info(f"Conditions: aspiration {sorted(asp_treatments) or '-'}; "
                f"pulses {[bl.pulse_folder(v, d) for v, d in sorted(pulses)] or '-'}")

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

    raw_path = bl.table(out, 'results')
    mechanics_df.to_csv(raw_path, index=False)
    logger.info(f"Saved results to: {raw_path.relative_to(results_dir)}  ({len(mechanics_df)} traps)")

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

    filtered_grouped_data = DiskBackedDict(cache_root / "visco")
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
        # via claim1_treatment_map into 1_Aspiration_Mechanics/<Treatment>/.
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

    mi_filtered_grouped_data = DiskBackedDict(cache_root / "prepulse")
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

    mi_whole_filtered_grouped_data = DiskBackedDict(cache_root / "mi_whole")
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
        attrition_path = bl.table(out, 'attrition')
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
         all_grouped_data, mechanics_df, attrition_df, out['figure_data_parent'])

    if mi_filtered_grouped_data.keys():
        # Cross-cohort pre-pulse figures land in prepulse_combined_dir;
        # per-bucket Thesis_MI_Boxplot_*.pdf files are routed via
        # prepulse_bucket_map into 2_Before_Pulse/Aspiration/ and /<pulse>/.
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
    # under 2_Before_Pulse/All_conditions/, whole-trace fate boxplot and the
    # whole-trace-by-fate trace plot under 4_Whole_Trace/All_conditions/.
    # ------------------------------------------------------------------
    _try("Thesis Claim 2 Core Plots",
         tp.run_thesis_claim2_plots,
         all_grouped_data, mechanics_df,
         prepulse_combined_dir, global_whole_dur, whole_trace_combined_dir,
         global_pre_dur)

    # ------------------------------------------------------------------
    # Fate cohort figures: pie chart per pulse condition + per-trap
    # intact/ruptured stacked bar per pulse condition. Both belong to
    # 3_At_Pulse; per-condition PDFs go into 3_At_Pulse/<pulse>/
    # via at_pulse_duration_map, while the
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
    # inside each plot function via mechanics_df. Routed to 3_At_Pulse
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
    # Whole-trace MI plots (4_Whole_Trace/All_conditions/).
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
    import sys
    sys.exit(main())