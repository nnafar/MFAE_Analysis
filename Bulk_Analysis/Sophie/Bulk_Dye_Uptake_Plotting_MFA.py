# -*- coding: utf-8 -*-
"""
Bulk Dye Uptake Analysis
- Processes multiple experiment folders with dye uptake data
- Generates ensemble average plots for:
  1. Absolute intensity (Total, Protrusion, Body)
  2. Min-Max normalized intensity (Total, Protrusion, Body)
  3. Baseline-normalized intensity ΔF/F₀ (Total, Protrusion, Body)
- Publication-quality visualization
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ENSEMBLE AVERAGING FOR DYE UPTAKE
# =============================================================================
def calculate_dye_ensemble_average(all_dye_data: List[pd.DataFrame], 
                                  columns: List[str]) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Calculates ensemble average for specified dye uptake columns."""
    if not all_dye_data:
        return None
    
    max_time = max(df['Time_s'].max() for df in all_dye_data)
    t_common = np.linspace(0, max_time, 500)
    
    results = {}
    
    for column in columns:
        interpolated_values = []
        for df in all_dye_data:
            if column in df.columns:
                t_data = df['Time_s'].values
                y_data = df[column].values
                
                valid_mask = np.isfinite(t_data) & np.isfinite(y_data)
                t_data, y_data = t_data[valid_mask], y_data[valid_mask]
                
                if len(t_data) >= 2:
                    f_interp = interp1d(t_data, y_data, kind='linear', 
                                      bounds_error=False, fill_value=np.nan)
                    y_interp = f_interp(t_common)
                    interpolated_values.append(y_interp)

        if interpolated_values:
            L_array = np.array([arr for arr in interpolated_values if np.any(np.isfinite(arr))])
            L_mean = np.nanmean(L_array, axis=0)
            L_sem = np.nanstd(L_array, axis=0) / np.sqrt(len(L_array))
            
            finite_mask = np.isfinite(L_mean)
            if np.any(finite_mask):
                results[column] = (t_common[finite_mask], L_mean[finite_mask], L_sem[finite_mask])
    
    return results if results else None

# =============================================================================
# PUBLICATION-QUALITY DYE UPTAKE PLOTTING
# =============================================================================

def plot_dye_ensemble_with_total(ensemble_results: Dict[str, Dict], 
                                 output_dir: Path, analyzer, n_trajectories: int,
                                 pulse_time: Optional[float] = None,
                                 pixel_counts: Optional[Dict[str, np.ndarray]] = None):
    """
    Creates publication-quality ensemble dye uptake plots, including total dye uptake
    (Absolute per-pixel * number of pixels in the region).
    
    Parameters:
        ensemble_results: dict of ensemble averages per region
        output_dir: Path to save plots
        analyzer: BulkDyeAnalyzer instance
        n_trajectories: number of trajectories averaged
        pulse_time: optional pulse time marker
        pixel_counts: dict with region keys and arrays of pixel counts (same length as ensemble arrays)
    """
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black'
    })

    colors = {
        'Total': ('#2E86AB', '#A4C3D2'),
        'Protrusion': ('#A23B72', '#D896B0'),
        'Body': ('#F18F01', '#F8C794')
    }

    # Plot the normal per-pixel mean
    plot_configs = [
        ('Absolute', 'Mean', 'Intensity (a.u.)', False),  # per-pixel
        ('TotalDye', 'Mean', 'Total Dye (a.u.)', True)    # total dye = per-pixel * pixels
    ]

    for suffix, key_suffix, ylabel, use_total in plot_configs:
        fig, ax = plt.subplots(figsize=(8, 6))
        legend_handles = []

        for region in ['Total', 'Protrusion', 'Body']:
            column_key = f'{region}_{key_suffix}'
            if column_key in ensemble_results:
                t, mean, sem = ensemble_results[column_key]

                # Multiply by pixel count if requested
                if use_total and pixel_counts is not None and region in pixel_counts:
                    mean = mean * pixel_counts[region]
                    sem = sem * pixel_counts[region]

                line, = ax.plot(t, mean, color=colors[region][0], linewidth=2.5, 
                                label=region, zorder=3)
                ax.fill_between(t, mean - sem, mean + sem, 
                                color=colors[region][1], alpha=0.3, zorder=2)
                legend_handles.append(line)

        # Pulse marker
        if pulse_time is not None:
            vline = ax.axvline(x=pulse_time, color='#C1292E', linestyle='--',
                                linewidth=2.5, zorder=1)
            legend_handles.append(plt.Line2D([0], [0], color='#C1292E',
                                             linestyle='--', linewidth=2.5,
                                             label='Electroporation Pulse'))

        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')

        title_map = {
            'Absolute': 'Absolute Dye Uptake (per-pixel)',
            'TotalDye': 'Total Dye Uptake (Absolute x Pixels)'
        }
        ax.set_title(f'{title_map[suffix]}: {analyzer.experiment_prefix} (n={n_trajectories})', 
                     fontsize=14, fontweight='bold', pad=15)

        ax.legend(handles=legend_handles, loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()

        save_path = output_dir / f'Ensemble_Dye_Uptake_entireregion{suffix}_{analyzer.experiment_prefix}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {save_path.name}")


def plot_dye_ensemble(ensemble_results: Dict[str, Dict], output_dir: Path, 
                      analyzer, n_trajectories: int, pulse_time: Optional[float] = None):
    """Creates publication-quality ensemble dye uptake plots."""
    
    # Set publication-quality parameters
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black'
    })
    
    # Color scheme
    colors = {
        'Total': ('#2E86AB', '#A4C3D2'),      # Blue
        'Protrusion': ('#A23B72', '#D896B0'),  # Purple
        'Body': ('#F18F01', '#F8C794')         # Orange
    }
    
    # Create three separate plots
    plot_configs = [
        ('Absolute', 'Mean', 'Intensity (a.u.)'),
        ('MinMax', 'MinMax', 'Normalized Intensity (0-1)'),
        ('Normalized_dF_F0', 'Normalized_dF_F0', 'Normalized Fluorescence (ΔF/F₀)')
    ]
    
    for suffix, key_suffix, ylabel in plot_configs:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        legend_handles = []
        
        # Plot each region
        for region in ['Total', 'Protrusion', 'Body']:
            column_key = f'{region}_{key_suffix}'
            
            if column_key in ensemble_results:
                t, mean, sem = ensemble_results[column_key]
                
                line, = ax.plot(t, mean, color=colors[region][0], linewidth=2.5, 
                              label=region, zorder=3)
                ax.fill_between(t, mean - sem, mean + sem, 
                              color=colors[region][1], alpha=0.3, zorder=2)
                legend_handles.append(line)
        
        # Add pulse time marker
        if pulse_time is not None:
            vline = ax.axvline(x=pulse_time, color='#C1292E', linestyle='--', 
                             linewidth=2.5, zorder=1)
            legend_handles.append(plt.Line2D([0], [0], color='#C1292E', 
                                            linestyle='--', linewidth=2.5, 
                                            label='Electroporation Pulse'))
        
        # Formatting
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        
        title_map = {
            'Absolute': 'Absolute Dye Uptake',
            'MinMax': 'Min-Max Normalized Dye Uptake',
            'Normalized_dF_F0': 'Baseline-Normalized Dye Uptake (ΔF/F₀)'
        }
        
        ax.set_title(f'{title_map[suffix]}: {analyzer.experiment_prefix} (n={n_trajectories})', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Legend
        ax.legend(handles=legend_handles, loc='upper left', fontsize=11, 
                 framealpha=0.95)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f'Ensemble_Dye_Uptake_{suffix}_{analyzer.experiment_prefix}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {save_path.name}")

# =============================================================================
# BULK DYE ANALYZER
# =============================================================================
class BulkDyeAnalyzer:
    def __init__(self, base_folder: str, experiment_prefix: str, 
                 selected_traps: Optional[Dict[str, List[int]]] = None, 
                 pulse_time: Optional[float] = None,
                 trap_remap: Optional[Dict[str, Dict[int, int]]] = None):
        
        self.base_folder = Path(base_folder)
        self.experiment_prefix = experiment_prefix
        self.selected_traps = selected_traps
        self.pulse_time = pulse_time
        self.dye_uptake_dir_name = "Dye Uptake"

        self.trap_remap = trap_remap or {}
        self.trap_raw_dye_data: Dict[int, List[pd.DataFrame]] = {}
        
    def get_canonical_trap(self, exp_id: str, trap_number: int) -> int:
        if exp_id in self.trap_remap:
            return self.trap_remap[exp_id].get(trap_number, trap_number)
        return trap_number



    def should_process_trap(self, exp_id: str, trap_number: int) -> bool:
        if self.selected_traps is None:
            return True
        if exp_id in self.selected_traps:
            return trap_number in self.selected_traps[exp_id]
        return False

    def find_experiment_folders(self) -> List[Path]:
        folders = []
        for item in self.base_folder.iterdir():
            if item.is_dir() and item.name.startswith(self.experiment_prefix):
                folders.append(item)
        
        def sort_key(p):
            match = re.search(r'(\d+)', p.name)
            return int(match.group(1)) if match else 0
        
        folders.sort(key=sort_key)
        return folders

    def load_dye_data_df(self, experiment_folder: Path, trap_number: int) -> Optional[pd.DataFrame]:
        """Loads the full dye uptake CSV file."""
        dye_folder = experiment_folder / self.dye_uptake_dir_name
        
        if not dye_folder.exists():
            return None
            
        expected_filename = f"Trap_{trap_number:02d}_Uptake_Data.csv"
        file_path = dye_folder / expected_filename
        
        if not file_path.exists():
            dye_files = list(dye_folder.glob(f"Trap_{trap_number}*_Uptake_Data.*"))
            if not dye_files:
                return None
            file_path = dye_files[0]
            
        try:
            df = pd.read_csv(file_path, comment='#')
            
            # Convert columns to numeric
            numeric_cols = ['Time_s', 'Total_Mean', 'Protrusion_Mean', 'Body_Mean',
                          'Total_MinMax', 'Protrusion_MinMax', 'Body_MinMax',
                          'Total_Normalized_dF_F0', 'Protrusion_Normalized_dF_F0', 
                          'Body_Normalized_dF_F0']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Trap_Number'] = trap_number
            df['Experiment_ID'] = experiment_folder.name
            return df
        except Exception as e:
            print(f"  Warning: Failed to load dye data for Trap {trap_number}: {e}")
            return None

    def run_analysis(self):
        print("="*80)
        print("BULK DYE UPTAKE ANALYSIS")
        print("="*80)
        
        if self.selected_traps is not None:
            print("\nSELECTED TRAPS:")
            for exp_id, traps in self.selected_traps.items():
                print(f"  {exp_id}: Traps {traps}")
        
        # Save results in Output folder with experiment prefix
        output_dir = self.base_folder / f"Bulk_Dye_Uptake_Results_{self.experiment_prefix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nOutput directory: {output_dir}")
        
        experiment_folders = self.find_experiment_folders()
        if not experiment_folders:
            print("No matching experiment folders found.")
            return

        # Process all experiments and traps
        for exp_folder in experiment_folders:
            exp_id = exp_folder.name
            dye_folder = exp_folder / self.dye_uptake_dir_name
            
            if not dye_folder.exists():
                continue
            
            print(f"\n{'='*80}")
            print(f"PROCESSING: {exp_id}")
            print(f"{'='*80}")
            
            # Find all dye uptake files
            dye_files = list(dye_folder.glob("Trap_*_Uptake_Data.*"))
            
            for dye_file in dye_files:
                # Extract trap number
                match = re.search(r'Trap_(\d+)', dye_file.name)
                if not match:
                    continue
                    
                trap_number = int(match.group(1))
                
                canonical_trap = self.get_canonical_trap(exp_id, trap_number)

                if canonical_trap != trap_number:
                    print(f"  ⚠ Remapping {exp_id}: Trap {trap_number} → {canonical_trap}")

                
                if not self.should_process_trap(exp_id, trap_number):
                    continue
                
                print(f"  Loading Trap {trap_number}...", end=" ")
                
                dye_data_df = self.load_dye_data_df(exp_folder, trap_number)
                
                if dye_data_df is not None:
                    dye_data_df['Canonical_Trap'] = canonical_trap

                    if canonical_trap not in self.trap_raw_dye_data:
                        self.trap_raw_dye_data[canonical_trap] = []
                    
                    self.trap_raw_dye_data[canonical_trap].append(dye_data_df)

                    print("SUCCESS")
                else:
                    print("FAILED")

        print(f"\n{'='*80}")
        print("GENERATING ENSEMBLE PLOTS")
        print(f"{'='*80}\n")
        
        if not self.trap_raw_dye_data:
            print("No dye uptake data collected. Exiting.")
            return
        
        for canonical_trap, dye_data_list in self.trap_raw_dye_data.items():
            print(f"\nGenerating ensemble plots for Trap {canonical_trap} "
                  f"(n={len(dye_data_list)})")
        
            trap_output_dir = output_dir / f"Trap_{canonical_trap:02d}"
            trap_output_dir.mkdir(parents=True, exist_ok=True)
        
            columns_absolute = ['Total_Mean', 'Protrusion_Mean', 'Body_Mean']
            columns_minmax = ['Total_MinMax', 'Protrusion_MinMax', 'Body_MinMax']
            columns_normalized = [
                'Total_Normalized_dF_F0',
                'Protrusion_Normalized_dF_F0',
                'Body_Normalized_dF_F0'
            ]
        
            ensemble_absolute = calculate_dye_ensemble_average(dye_data_list, columns_absolute)
            ensemble_minmax = calculate_dye_ensemble_average(dye_data_list, columns_minmax)
            ensemble_normalized = calculate_dye_ensemble_average(dye_data_list, columns_normalized)
        
            all_results = {}
            if ensemble_absolute:
                all_results.update(ensemble_absolute)
            if ensemble_minmax:
                all_results.update(ensemble_minmax)
            if ensemble_normalized:
                all_results.update(ensemble_normalized)
        
            if all_results:
                plot_dye_ensemble(
                    all_results,
                    trap_output_dir,
                    self,
                    len(dye_data_list),
                    self.pulse_time
                )
        
                final_dye_df = pd.concat(dye_data_list, ignore_index=True)
                dye_output_path = trap_output_dir / 'Consolidated_Dye_Uptake_Data.csv'
                final_dye_df.to_csv(dye_output_path, index=False)
        
                print(f"  Saved Trap {canonical_trap} results")
            else:
                print(f"  Warning: No ensemble data for Trap {canonical_trap}")

        # Before plotting, calculate pixel counts for each region
        pixel_counts = {
            'Total': np.array([len(df) for df in dye_data_list[0]['Total_Pixels']]),       # or precomputed
            'Protrusion': np.array([len(df) for df in dye_data_list[0]['Protrusion_Pixels']]),
            'Body': np.array([len(df) for df in dye_data_list[0]['Body_Pixels']])
        }
        
        plot_dye_ensemble_with_total(
            all_results,
            trap_output_dir,
            self,
            len(dye_data_list),
            self.pulse_time,
            pixel_counts=pixel_counts
        )

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*80}\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Configuration
    BASE_FOLDER_PATH = r"C:\GitHub\MFAE_Analysis\Output"
    EXPERIMENT_PREFIX = "100V_100us"
    PULSE_TIME = 10.0  # Time when electroporation pulse was applied (seconds)
    
    # Select specific traps (or set to None for all traps)
    SELECTED_TRAPS = {
        "100V_100us_pulse_Experiment1": [4],
        "100V_100us_pulse_Experiment2": [4],
        "100V_100us_pulse_Experiment3": [4]
    }
    
    trap_remap = {
        "100V_100us_pulse_Experiment1": {4: 4},
        "100V_100us_pulse_Experiment2": {4: 4},
        "100V_100us_pulse_Experiment3": {4: 4},
    }

    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Base folder: {BASE_FOLDER_PATH}")
    print(f"Experiment prefix: {EXPERIMENT_PREFIX}")
    print(f"Pulse time: {PULSE_TIME}s")
    if SELECTED_TRAPS:
        total_traps = sum(len(traps) for traps in SELECTED_TRAPS.values())
        print(f"Selected traps: {total_traps}")
    print("="*80)
    
    analyzer = BulkDyeAnalyzer(
        base_folder=BASE_FOLDER_PATH,
        experiment_prefix=EXPERIMENT_PREFIX,
        selected_traps=SELECTED_TRAPS,
        pulse_time=PULSE_TIME,
        trap_remap = trap_remap
    )
    
    analyzer.run_analysis()