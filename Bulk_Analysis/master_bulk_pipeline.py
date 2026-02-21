# -*- coding: utf-8 -*-
"""
Master Pipeline for Bulk MFAE Analysis.
"""

import sys
import logging
from pathlib import Path

import bulk_file_handling as bfh
import bulk_plotting as bp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def main():
    output_root_dir = Path(r"C:\GitHub\MFAE_Analysis\Output")
    results_dir = output_root_dir / "Bulk_Analysis_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    logger.info("--- PHASE 1: DATA LOADING ---")
    try:
        loader = bfh.BulkDataLoader(str(output_root_dir))
        loader.scan_and_load()
        grouped_data = loader.grouped_data
        if not grouped_data: logger.error("No valid data!"); return
    except Exception as e: logger.error(f"Loading Error: {e}", exc_info=True); return

    logger.info("\nData Summary:"); logger.info(bfh.get_group_stats(grouped_data)); logger.info("-" * 30 + "\n")

    logger.info("--- PHASE 2: VISUALIZATION ---")
    
    # 1. Build Scalar DataFrame and run Spearman Correlation
    df_scalars = bfh.extract_all_scalars(grouped_data)
    if not df_scalars.empty:
        try: bp.plot_spearman_correlation(df_scalars, results_dir)
        except Exception as e: logger.error(f"Spearman Correlation Failed: {e}")
    
    # 2. Run Aggregate Plotting Suite
    try: bp.plot_max_protrusion_distribution(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 1 Failed: {e}")

    try: bp.plot_per_trap_protrusion_distribution(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 2 Failed: {e}")

    try: bp.plot_uptake_dynamics(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 3 Failed: {e}")

    try: bp.plot_per_trap_uptake_distribution(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 4 Failed: {e}")

    try: bp.plot_rupture_probability(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 5 Failed: {e}")

    try: bp.plot_protrusion_recoil_velocity(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 6 Failed: {e}")

    try: bp.plot_uptake_exponential_fit(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 7 Failed: {e}")

    try: bp.plot_correlation_length_vs_uptake(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 8 Failed: {e}")

    try: bp.plot_uptake_fits_multipanel(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 9 Failed: {e}")

    try: bp.plot_recoil_fits_multipanel(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 10 Failed: {e}")

    try: bp.plot_uptake_traces_multipanel(grouped_data, results_dir)
    except Exception as e: logger.error(f"Plot 11 Failed: {e}")

    logger.info("\n=== BULK ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()