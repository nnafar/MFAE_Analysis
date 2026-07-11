"""
plot_mfae_overview.py
---------------------
Reads an MFAE Overview CSV and produces two figures:

  Figure 1 – Per-trap overview
      • Stacked bar chart : counts of I / R / D / E per Trap ID (across all experiments)
      • Pie chart          : overall proportions of I / R / D / E

  Figure 2 – Per-chip overview  (one sub-figure per unique Date × Chip ID combination)
      • Stacked bar chart : counts of I / R / D / E per Trap ID
      • Pie chart          : proportions for that chip

Usage
-----
    python plot_mfae_overview.py

Edit INPUT_FILE below if the path changes.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_FILE = r"C:\GitHub\MFAE_Analysis\Output\MDAMB231_WT-1100Pa-100V-5ms_Overview.csv"

# Output directory: a "Figures" subfolder next to the input CSV.
# It is created automatically if it does not exist yet.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(INPUT_FILE)), "Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette matching the reference figure
COLORS = {
    "I":  "#1B5E8C",   # Intact              – steel blue
    "R0": "#F1948A",   # Ruptured (pre-pulse) – salmon / light red
    "R":  "#C0392B",   # Ruptured (post-pulse)– dark red
    "D":  "#F0A500",   # Debris              – amber / gold
    "E":  "#C8C8C8",   # Empty               – light grey
}
LABELS = {
    "I":  "Intact",
    "R0": "Ruptured pre-pulse",
    "R":  "Ruptured post-pulse",
    "D":  "Debris",
    "E":  "Empty",
}
STATUS_ORDER = ["I", "R0", "R", "D", "E"]   # stack order (bottom → top)

# ── Load & clean data ──────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        header=0,
        encoding="utf-8-sig",   # handles BOM if present
        dtype=str,
    )
    # Rename columns by position so we are robust to header text changes
    col_map = {
        df.columns[0]:  "Date",
        df.columns[3]:  "ChipID",
        df.columns[4]:  "ExpID",
        df.columns[10]: "TrapID",
        df.columns[11]: "Status",
    }
    df = df.rename(columns=col_map)

    # Keep only the columns we need
    df = df[["Date", "ChipID", "ExpID", "TrapID", "Status"]].copy()

    # Strip whitespace
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Normalise Status to uppercase single letter; drop anything else
    df["Status"] = df["Status"].str.upper()
    df = df[df["Status"].isin(STATUS_ORDER)].copy()

    # Numeric trap and chip IDs
    df["TrapID"] = pd.to_numeric(df["TrapID"], errors="coerce")
    df["ChipID"] = pd.to_numeric(df["ChipID"], errors="coerce")
    df = df.dropna(subset=["TrapID", "ChipID"])
    df["TrapID"] = df["TrapID"].astype(int)
    df["ChipID"] = df["ChipID"].astype(int)

    # Unique experiment key = Date × ChipID  (combines all Exp. IDs on the same chip/day)
    df["ChipKey"] = df["Date"] + "  |  Chip " + df["ChipID"].astype(str)

    return df


# ── Counting helpers ───────────────────────────────────────────────────────────

def counts_per_trap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by TrapID with columns I, R, D, E.
    All traps 1-18 are always present (fills 0 for missing combos).
    """
    ct = (
        df.groupby(["TrapID", "Status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
    )
    all_traps = range(1, 19)
    ct = ct.reindex(all_traps, fill_value=0)
    return ct


def counts_totals(df: pd.DataFrame) -> pd.Series:
    return df["Status"].value_counts().reindex(STATUS_ORDER, fill_value=0)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _legend_patches():
    return [
        mpatches.Patch(color=COLORS[s], label=LABELS[s])
        for s in STATUS_ORDER
    ]


def _bar_and_pie(df_subset: pd.DataFrame, fig: plt.Figure,
                 ax_bar: plt.Axes, ax_pie: plt.Axes,
                 title_bar: str = "") -> None:
    """Draw a stacked bar chart and a pie chart into pre-created axes."""

    ct = counts_per_trap(df_subset)
    totals = counts_totals(df_subset)

    # ── stacked bar ────────────────────────────────────────────────────────────
    bottom = [0] * len(ct)
    for status in STATUS_ORDER:
        vals = ct[status].values
        ax_bar.bar(
            ct.index,
            vals,
            bottom=bottom,
            color=COLORS[status],
            label=LABELS[status],
            edgecolor="white",
            linewidth=0.4,
            width=0.7,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax_bar.set_xticks(ct.index)
    ax_bar.set_xticklabels(ct.index, fontsize=8)
    ax_bar.set_xlabel("Trap ID", fontsize=9)
    ax_bar.set_ylabel("Number of events", fontsize=9)
    ax_bar.set_title(title_bar, fontsize=10, fontweight="bold", pad=6)
    ax_bar.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_bar.legend(
        handles=_legend_patches(),
        loc="upper right",
        fontsize=7,
        framealpha=0.7,
    )
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="both", labelsize=8)

    # ── pie ────────────────────────────────────────────────────────────────────
    # Only show slices with at least one event; keep colour order consistent
    present = [(s, totals[s]) for s in STATUS_ORDER if totals[s] > 0]
    pie_vals  = [v for _, v in present]
    pie_cols  = [COLORS[s] for s, _ in present]
    pie_lbls  = [LABELS[s] for s, _ in present]
    total_n   = sum(pie_vals)

    def autopct(pct):
        n = int(round(pct / 100 * total_n))
        return f"{pct:.0f}%\n(n={n})" if pct >= 4 else ""

    wedges, texts, autotexts = ax_pie.pie(
        pie_vals,
        labels=pie_lbls,
        colors=pie_cols,
        autopct=autopct,
        startangle=90,
        pctdistance=0.72,
        labeldistance=1.12,
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
    )
    for t in texts:
        t.set_fontsize(8)
    for at in autotexts:
        at.set_fontsize(7)

    ax_pie.set_title(
        f"Overall  (n={total_n})",
        fontsize=9,
        fontweight="bold",
        pad=4,
    )


# ── Figure 1 : overall ────────────────────────────────────────────────────────

def plot_overall(df: pd.DataFrame, output_dir: str) -> None:
    fig = plt.figure(figsize=(13, 5), constrained_layout=True)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])

    fig.suptitle(
        "All experiments – per-trap cell status overview",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    _bar_and_pie(df, fig, ax_bar, ax_pie, title_bar="Counts per trap (all experiments combined)")

    out = os.path.join(output_dir, "overview_ALL.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2 : per chip × date ────────────────────────────────────────────────

def plot_per_chip(df: pd.DataFrame, output_dir: str) -> None:
    chip_groups = sorted(df["ChipKey"].unique())
    n = len(chip_groups)

    # Layout: 2 rows per chip group (bar + pie), arranged in a single tall figure
    fig = plt.figure(figsize=(14, 5.5 * n), constrained_layout=True)
    outer_gs = GridSpec(n, 1, figure=fig, hspace=0.45)

    fig.suptitle(
        "Per-chip cell status overview  (grouped by Date × Chip ID)",
        fontsize=13,
        fontweight="bold",
        y=1.005,
    )

    for row_idx, chip_key in enumerate(chip_groups):
        sub_df = df[df["ChipKey"] == chip_key]
        inner_gs = outer_gs[row_idx].subgridspec(1, 2, width_ratios=[2.5, 1], wspace=0.30)
        ax_bar = fig.add_subplot(inner_gs[0])
        ax_pie = fig.add_subplot(inner_gs[1])

        # Nicer label: "26.01.14  |  Chip 2"  →  "26 Jan 2014  •  Chip 2"
        _bar_and_pie(sub_df, fig, ax_bar, ax_pie, title_bar=chip_key)

    out = os.path.join(output_dir, "overview_per_chip.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading: {INPUT_FILE}")
    df = load_data(INPUT_FILE)

    n_rows = len(df)
    print(f"Loaded {n_rows} valid events  "
          f"({df['ChipKey'].nunique()} chip×date groups, "
          f"{df['TrapID'].nunique()} unique trap IDs)")
    print("Status distribution (overall):")
    print(counts_totals(df).to_string())
    print()

    plot_overall(df, OUTPUT_DIR)
    plot_per_chip(df, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()