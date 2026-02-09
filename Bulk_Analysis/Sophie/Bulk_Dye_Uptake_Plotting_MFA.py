#!/usr/bin/env python3
"""
Enhanced batch circularity analysis with drug and dosage comparisons
for CK666, SMIFH2, DMSO and CytD TIFF series.
"""

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------
TIFF_ROOT = r"F:\A_Data\A_DNATranslocation\Experiments_2025\MFAE\2.10.25 Actin Inhibitor Dosage Test\LSM710\B_ProcessedImages\TIFFs"
OUT_ROOT = r"F:\A_Data\A_DNATranslocation\Experiments_2025\MFAE\2.10.25 Actin Inhibitor Dosage Test\LSM710\B_ProcessedImages\Circularity_Results"

PIXEL_SIZE_UM = 0.5          # optional, µm per pixel (not used for the dimensionless circularity)
MIN_CELL_AREA = 100          # smallest object to keep (pixels)
MAX_CELL_AREA = 5000         # largest object to keep (pixels)

SAVE_COMBINED_TABLE = True
COMBINED_TABLE_NAME = "all_circularity_summary.csv"

# Plot‑specific settings -------------------------------------------------
BIN_COUNT = 20                # number of circularity bins between 0 and 1
PLOT_DPI = 300
PLOT_FIGSIZE = (8, 5)         # inches
LINE_WIDTH = 2.5
SEM_SHADE_ALPHA = 0.25

# Colour palette – you can edit the hex codes if you prefer other shades
PALETTE = {
    "CK666":  "#1f77b4",   # classic blue
    "SMIFH2": "#ff7f0e",   # orange
    "DMSO":   "#2ca02c",   # green
    "CytD":   "#d62728",   # red
}

# ----------------------------------------------------------------------
# LIBRARY IMPORTS
# ----------------------------------------------------------------------
import pathlib, warnings, sys, re
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import ndimage as ndi, stats
from skimage import io, filters, morphology, measure, exposure, segmentation, util

# ----------------------------------------------------------------------
# OPTIONAL: try to use a seaborn style if it is available
# ----------------------------------------------------------------------
def safe_set_style(style_name: str, fallback: str = "ggplot"):
    """Apply *style_name* if matplotlib knows it; otherwise use *fallback*."""
    if style_name in plt.style.available:
        plt.style.use(style_name)
    else:
        print(f"️  Style '{style_name}' not found – falling back to '{fallback}'.")
        plt.style.use(fallback)

safe_set_style("seaborn-whitegrid", fallback="ggplot")

# ----------------------------------------------------------------------
# CORE FUNCTIONS (unchanged except for distance_transform_edt)
# ----------------------------------------------------------------------
def load_tiff(path: pathlib.Path) -> np.ndarray:
    """Read a TIFF (single‑ or multi‑page) and return a 2‑D float image in [0,1]."""
    img = io.imread(str(path))
    if img.ndim > 2:                 # collapse Z‑stack (max‑projection)
        img = np.max(img, axis=0)
    return util.img_as_float(img)

def preprocess(img: np.ndarray) -> np.ndarray:
    """Background subtraction + contrast stretch + slight smoothing."""
    background = filters.gaussian(img, sigma=50)
    img_corr = img - background
    img_corr = exposure.rescale_intensity(img_corr, out_range=(0, 1))
    return filters.gaussian(img_corr, sigma=1)

def segment_cells(img: np.ndarray) -> np.ndarray:
    """Return a labelled mask (int) where each cell has a unique ID."""
    # 1) Global Otsu threshold
    thresh = filters.threshold_otsu(img)
    binary = img > thresh

    # 2) Clean small artefacts / fill holes
    binary = morphology.remove_small_objects(binary, min_size=MIN_CELL_AREA)
    binary = morphology.remove_small_holes(binary, area_threshold=MIN_CELL_AREA)

    # 3) Distance‑map watershed (fixed import)
    distance = ndi.distance_transform_edt(binary)
    local_max = morphology.h_maxima(distance, h=0.1)
    markers = measure.label(local_max)
    labels = segmentation.watershed(-distance, markers, mask=binary)

    # 4) Keep only objects within the expected area range
    props = measure.regionprops(labels)
    keep = [p.label for p in props if MIN_CELL_AREA <= p.area <= MAX_CELL_AREA]
    filtered = np.isin(labels, keep) * labels
    return filtered.astype(np.int32)

def compute_circularity(labels: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame with Cell_ID, Area_px, Perimeter_px, Circularity."""
    props = measure.regionprops(labels)
    rows = []
    for p in props:
        area = p.area
        peri = p.perimeter
        circ = np.nan if peri == 0 else (4 * np.pi * area) / (peri ** 2)
        rows.append({
            "Cell_ID": p.label,
            "Area_px": area,
            "Perimeter_px": peri,
            "Circularity": circ,
        })
    return pd.DataFrame(rows)

def overlay_labels(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """RGB image with red cell boundaries drawn on the original grayscale."""
    rgb = np.dstack([img, img, img])
    return segmentation.mark_boundaries(rgb, labels, color=(1, 0, 0), mode="outer")

# ----------------------------------------------------------------------
# ENHANCED METADATA EXTRACTION
# ----------------------------------------------------------------------

# Concentration mapping for each drug based on image number
CONCENTRATION_MAP = {
    "CK666": {
        1: 50.0,
        2: 100.0,
        3: 150.0,
        4: 200.0
    },
    "SMIFH2": {
        1: 5.0,
        2: 25.0,
        3: 40.0,
        4: 80.0
    },
    "CYTD": {
        1: 0.1,
        2: 0.25,
        3: 0.50,
        4: 1.0
    },
    "DMSO": {
        1: 0.0,
        2: 50.0,
        3: 100.0,
        4: 200.0
    },
    "CNTRL": {
        1: 0.0
    }
}

def extract_treatment_and_dosage(fname: str) -> tuple:
    """
    Extract drug name and dosage from filename based on image number.
    
    Examples:
        "CK666_Image_1_FL.tif" → ("CK666", 50.0)
        "SMIFH2_Image_3_FL.tif" → ("SMIFH2", 40.0)
        "CytD_Image_2_FL.tif" → ("CytD", 0.25)
        "DMSO_Image_4_FL.tif" → ("DMSO", 200.0)
        "CNTRL_Image_1_FL.tif" → ("CNTRL", 0.0)
    """
    lower = fname.lower()
    original = fname
    
    # Detect drug
    drug = "Unknown"
    for d in ["cntrl", "ck666", "smifh2", "cytd", "dmso"]:  # Check CNTRL first
        if d in lower:
            drug = d.upper()
            break
    
    # Extract image number
    # Look for patterns like "Image_1", "Image1", "image 2", etc.
    image_num_match = re.search(r'image[_\s]*(\d+)', lower)
    
    dosage = 0.0
    if image_num_match and drug in CONCENTRATION_MAP:
        image_num = int(image_num_match.group(1))
        dosage = CONCENTRATION_MAP[drug].get(image_num, 0.0)
    elif "control" in lower or "cntrl" in lower:
        dosage = 0.0
        if drug == "Unknown":
            drug = "CNTRL"
    
    return drug, dosage

def bin_counts(circularities: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Return raw histogram counts (no normalization)."""
    counts, _ = np.histogram(circularities, bins=bins)
    return counts

# ----------------------------------------------------------------------
# COMPARISON PLOTTING FUNCTIONS
# ----------------------------------------------------------------------
def plot_drug_comparison_boxplot(df: pd.DataFrame, out_path: pathlib.Path):
    """Box plot comparing circularity across different drugs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    drugs = ["CK666", "SMIFH2", "DMSO", "CytD"]
    data_to_plot = []
    labels = []
    colors = []
    
    for drug in drugs:
        drug_data = df[df["Drug"] == drug]["Circularity"].dropna()
        if len(drug_data) > 0:
            data_to_plot.append(drug_data)
            labels.append(drug)
            colors.append(PALETTE[drug])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel("Circularity", fontsize=12)
    ax.set_xlabel("Drug Treatment", fontsize=12)
    ax.set_title("Circularity Comparison Across Drug Treatments", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_dosage_comparison(df: pd.DataFrame, out_path: pathlib.Path):
    """Line plot showing mean circularity vs dosage for each drug."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    drugs = ["CK666", "SMIFH2", "CytD"]  # Exclude DMSO as it's typically control
    
    for drug in drugs:
        drug_df = df[df["Drug"] == drug]
        if len(drug_df) == 0:
            continue
        
        # Group by dosage and calculate mean ± SEM
        dosage_stats = drug_df.groupby("Dosage_uM")["Circularity"].agg(['mean', 'sem', 'count']).reset_index()
        dosage_stats = dosage_stats.sort_values("Dosage_uM")
        
        if len(dosage_stats) > 0:
            ax.plot(dosage_stats["Dosage_uM"], dosage_stats["mean"], 
                   marker='o', linewidth=LINE_WIDTH, markersize=8,
                   label=drug, color=PALETTE[drug])
            ax.fill_between(dosage_stats["Dosage_uM"],
                          dosage_stats["mean"] - dosage_stats["sem"],
                          dosage_stats["mean"] + dosage_stats["sem"],
                          color=PALETTE[drug], alpha=SEM_SHADE_ALPHA)
    
    ax.set_xlabel("Dosage (µM)", fontsize=12)
    ax.set_ylabel("Mean Circularity", fontsize=12)
    ax.set_title("Dose-Response: Circularity vs Drug Concentration", fontsize=14, fontweight='bold')
    ax.legend(title="Drug", loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_dosage_heatmap(df: pd.DataFrame, out_path: pathlib.Path):
    """Heatmap showing mean circularity for each drug-dosage combination."""
    # Pivot table: drugs as rows, dosages as columns
    pivot = df.groupby(["Drug", "Dosage_uM"])["Circularity"].mean().reset_index()
    pivot_table = pivot.pivot(index="Drug", columns="Dosage_uM", values="Circularity")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn', vmin=0.7, vmax=0.9)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels([f"{d:.1f}" for d in pivot_table.columns])
    ax.set_yticklabels(pivot_table.index)
    
    # Labels
    ax.set_xlabel("Dosage (µM)", fontsize=12)
    ax.set_ylabel("Drug Treatment", fontsize=12)
    ax.set_title("Circularity Heatmap: Drug × Dosage", fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Circularity", fontsize=11)
    
    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                             color="white" if val < 0.5 else "black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def plot_distribution_by_dosage(df: pd.DataFrame, drug: str, out_path: pathlib.Path):
    """Distribution curves for a specific drug across different dosages."""
    drug_df = df[df["Drug"] == drug]
    dosages = sorted(drug_df["Dosage_uM"].unique())
    
    if len(dosages) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_edges = np.linspace(0, 1, BIN_COUNT + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    cmap = plt.cm.viridis
    colors = [cmap(i / len(dosages)) for i in range(len(dosages))]
    
    for dosage, color in zip(dosages, colors):
        dosage_data = drug_df[drug_df["Dosage_uM"] == dosage]["Circularity"].dropna()
        if len(dosage_data) > 0:
            counts, _ = np.histogram(dosage_data, bins=bin_edges, density=True)
            ax.plot(bin_centers, counts, label=f"{dosage:.1f} µM", 
                   linewidth=LINE_WIDTH, color=color)
    
    ax.set_xlabel("Circularity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{drug}: Circularity Distribution by Dosage", fontsize=14, fontweight='bold')
    ax.legend(title="Dosage", loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

def generate_statistics_report(df: pd.DataFrame, out_path: pathlib.Path):
    """Generate a comprehensive statistical summary report."""
    with open(out_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CIRCULARITY ANALYSIS - STATISTICAL SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total cells analyzed: {len(df)}\n")
        f.write(f"Mean circularity: {df['Circularity'].mean():.4f} ± {df['Circularity'].std():.4f}\n")
        f.write(f"Median circularity: {df['Circularity'].median():.4f}\n\n")
        
        # Statistics by drug
        f.write("STATISTICS BY DRUG TREATMENT\n")
        f.write("-" * 80 + "\n")
        for drug in ["CK666", "SMIFH2", "DMSO", "CytD"]:
            drug_data = df[df["Drug"] == drug]["Circularity"].dropna()
            if len(drug_data) > 0:
                f.write(f"\n{drug}:\n")
                f.write(f"  N cells: {len(drug_data)}\n")
                f.write(f"  Mean: {drug_data.mean():.4f} ± {drug_data.std():.4f}\n")
                f.write(f"  Median: {drug_data.median():.4f}\n")
                f.write(f"  Range: [{drug_data.min():.4f}, {drug_data.max():.4f}]\n")
        
        # Statistics by drug and dosage
        f.write("\n\nSTATISTICS BY DRUG AND DOSAGE\n")
        f.write("-" * 80 + "\n")
        grouped = df.groupby(["Drug", "Dosage_uM"])["Circularity"]
        for (drug, dosage), group in grouped:
            group_clean = group.dropna()
            if len(group_clean) > 0:
                f.write(f"\n{drug} @ {dosage:.1f} µM:\n")
                f.write(f"  N cells: {len(group_clean)}\n")
                f.write(f"  Mean: {group_clean.mean():.4f} ± {group_clean.std():.4f}\n")
                f.write(f"  Median: {group_clean.median():.4f}\n")
        
        # Pairwise comparisons (Kruskal-Wallis test)
        f.write("\n\nSTATISTICAL TESTS (Kruskal-Wallis)\n")
        f.write("-" * 80 + "\n")
        drugs_with_data = []
        drug_groups = []
        for drug in ["CK666", "SMIFH2", "DMSO", "CytD"]:
            drug_data = df[df["Drug"] == drug]["Circularity"].dropna()
            if len(drug_data) > 0:
                drugs_with_data.append(drug)
                drug_groups.append(drug_data.values)
        
        if len(drug_groups) > 1:
            h_stat, p_value = stats.kruskal(*drug_groups)
            f.write(f"\nComparison across all drugs:\n")
            f.write(f"  H-statistic: {h_stat:.4f}\n")
            f.write(f"  p-value: {p_value:.4e}\n")
            if p_value < 0.05:
                f.write(f"  Result: Significant difference detected (p < 0.05)\n")
            else:
                f.write(f"  Result: No significant difference (p ≥ 0.05)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

# ----------------------------------------------------------------------
# MAIN BATCH ROUTINE
# ----------------------------------------------------------------------
def main():
    out_root = pathlib.Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    overlay_dir = out_root / "overlays"
    hist_dir    = out_root / "histograms"
    csv_dir     = out_root / "csv_per_image"
    compare_dir = out_root / "comparisons"  # NEW: comparison folder
    
    for d in (overlay_dir, hist_dir, csv_dir, compare_dir):
        d.mkdir(exist_ok=True)

    tiff_folder = pathlib.Path(TIFF_ROOT)
    tiff_paths = sorted([p for p in tiff_folder.iterdir()
                         if p.suffix.lower() in {".tif", ".tiff"}])
    if not tiff_paths:
        print(f"No TIFF files found in {tiff_folder}")
        sys.exit(0)

    # --------------------------------------------------------------
    # 1) Run the original pipeline for every image
    # --------------------------------------------------------------
    combined_rows = []
    treatment_circ = {k: [] for k in ["CK666", "SMIFH2", "DMSO", "CytD"]}

    for tiff_path in tiff_paths:
        print(f"\nProcessing: {tiff_path.name}")

        # ----- load & preprocess -----
        img = load_tiff(tiff_path)
        img_prep = preprocess(img)

        # ----- segmentation -----
        labels = segment_cells(img_prep)

        # ----- circularity table -----
        df = compute_circularity(labels)
        df.insert(0, "FileName", tiff_path.name)
        
        # ----- extract metadata -----
        drug, dosage = extract_treatment_and_dosage(tiff_path.name)
        df.insert(1, "Drug", drug)
        df.insert(2, "Dosage_uM", dosage)

        # ----- save per‑image CSV -----
        csv_path = csv_dir / f"{tiff_path.stem}_circularity.csv"
        df.to_csv(csv_path, index=False)

        # ----- save overlay -----
        overlay = overlay_labels(img, labels)
        overlay_path = overlay_dir / f"{tiff_path.stem}_overlay.png"
        plt.imsave(str(overlay_path), overlay, cmap="gray")

        # ----- per‑image histogram -----
        plt.figure(figsize=(6, 4))
        plt.hist(df["Circularity"].dropna(), bins=30, edgecolor="black")
        plt.xlabel("Circularity")
        plt.ylabel("Cell count")
        plt.title(f"Circularity distribution – {tiff_path.name}")
        plt.tight_layout()
        hist_path = hist_dir / f"{tiff_path.stem}_hist.png"
        plt.savefig(str(hist_path), dpi=PLOT_DPI)
        plt.close()

        # ----- collect for combined analysis -----
        combined_rows.append(df)
        
        if drug in treatment_circ:
            treatment_circ[drug].append(df["Circularity"].dropna().values)

    # --------------------------------------------------------------
    # 2) Write combined CSV
    # --------------------------------------------------------------
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_path = out_root / COMBINED_TABLE_NAME
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined summary saved to: {combined_path}")
    else:
        print("\nNo data processed!")
        sys.exit(0)

    # --------------------------------------------------------------
    # 3) Original publishable line plot
    # --------------------------------------------------------------
    bin_edges = np.linspace(0, 1, BIN_COUNT + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    means = {}
    sems  = {}
    for treat, circ_lists in treatment_circ.items():
        if not circ_lists:
            continue
        counts_matrix = np.vstack([bin_counts(circ, bin_edges) for circ in circ_lists])
        means[treat] = counts_matrix.mean(axis=0)
        sems[treat]  = counts_matrix.std(axis=0, ddof=1) / np.sqrt(counts_matrix.shape[0])

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    for treat in ["CK666", "SMIFH2", "DMSO", "CytD"]:
        if treat not in means:
            continue
        ax.plot(bin_centers, means[treat],
                label=treat,
                color=PALETTE[treat],
                linewidth=LINE_WIDTH)
        ax.fill_between(bin_centers,
                        means[treat] - sems[treat],
                        means[treat] + sems[treat],
                        color=PALETTE[treat],
                        alpha=SEM_SHADE_ALPHA)

    ax.set_xlabel("Circularity")
    ax.set_ylabel("Mean cell count per bin")
    ax.set_title("Circularity distribution across drug treatments")
    ax.set_xlim(0, 1)
    ax.legend(title="Treatment", loc="upper right")
    plt.tight_layout()

    fig_path_png = out_root / "circularity_distribution_lineplot.png"
    fig_path_pdf = out_root / "circularity_distribution_lineplot.pdf"
    fig.savefig(str(fig_path_png), dpi=PLOT_DPI)
    fig.savefig(str(fig_path_pdf))
    plt.close()

    # --------------------------------------------------------------
    # 4) NEW COMPARISON ANALYSES
    # --------------------------------------------------------------
    print("\n" + "="*60)
    print("GENERATING COMPARISON ANALYSES")
    print("="*60)
    
    # Box plot comparison across drugs
    print("\n[1/6] Creating drug comparison box plot...")
    plot_drug_comparison_boxplot(combined_df, compare_dir / "drug_comparison_boxplot.png")
    
    # Dosage response curves
    print("[2/6] Creating dose-response curves...")
    plot_dosage_comparison(combined_df, compare_dir / "dosage_response_curves.png")
    
    # Heatmap
    print("[3/6] Creating drug × dosage heatmap...")
    plot_dosage_heatmap(combined_df, compare_dir / "drug_dosage_heatmap.png")
    
    # Distribution by dosage for each drug
    print("[4/6] Creating distribution plots by dosage...")
    for drug in ["CK666", "SMIFH2", "CytD"]:
        if drug in combined_df["Drug"].values:
            plot_distribution_by_dosage(combined_df, drug, 
                                       compare_dir / f"{drug}_dosage_distributions.png")
    
    # Statistics report
    print("[5/6] Generating statistical summary report...")
    generate_statistics_report(combined_df, compare_dir / "statistical_summary.txt")
    
    # Summary table
    print("[6/6] Creating summary statistics table...")
    summary = combined_df.groupby(["Drug", "Dosage_uM"])["Circularity"].agg([
        ('N_cells', 'count'),
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Median', 'median'),
        ('Min', 'min'),
        ('Max', 'max')
    ]).reset_index()
    summary.to_csv(compare_dir / "summary_statistics.csv", index=False)

    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {out_root}")
    print(f"\nComparison analyses saved to: {compare_dir}")
    print("\nGenerated files:")
    print("  - drug_comparison_boxplot.png")
    print("  - dosage_response_curves.png")
    print("  - drug_dosage_heatmap.png")
    print("  - [Drug]_dosage_distributions.png")
    print("  - statistical_summary.txt")
    print("  - summary_statistics.csv")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()