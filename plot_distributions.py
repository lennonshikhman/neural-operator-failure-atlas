#!/usr/bin/env python3
"""
Create publication-quality plots from degradation summary CSVs.

Inputs:
  analysis/*_degradation_summary.csv

Outputs:
  paper_visuals/*.pdf
  paper_visuals/*.png
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

ANALYSIS_DIR = "analysis"
OUT_DIR = "paper_visuals"
os.makedirs(OUT_DIR, exist_ok=True)

# consistent, paper-safe style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (6.5, 4.5),
    "lines.linewidth": 2.0,
})

PDES = [
    "poisson",
    "black_scholes",
    "schrodinger",
    "navier_stokes",
    "kuramoto_sivashinsky",
]


# ============================================================
# Helpers
# ============================================================

def load_summary(pde: str) -> pd.DataFrame:
    path = os.path.join(ANALYSIS_DIR, f"{pde}_degradation_summary.csv")
    return pd.read_csv(path, index_col=0)


def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300)
    plt.close()


# ============================================================
# Plot 1: Per-PDE degradation bar plots (mean ± CI)
# ============================================================

def plot_pde_degradation(pde: str):
    df = load_summary(pde)

    stresses = df.index.tolist()
    means = df["mean"].values
    ci_low = df["ci_low"].values
    ci_high = df["ci_high"].values

    yerr = np.vstack([means - ci_low, ci_high - means])

    plt.figure()
    plt.bar(
        range(len(stresses)),
        means,
        yerr=yerr,
        capsize=4,
    )
    plt.axhline(1.0, linestyle="--", color="black", alpha=0.6)
    plt.xticks(range(len(stresses)), stresses, rotation=30, ha="right")
    plt.ylabel("Degradation (relative to baseline)")
    plt.title(f"{pde.replace('_', ' ').title()}")

    save_fig(f"{pde}_degradation_bar")


# ============================================================
# Plot 2: Cross-PDE heatmap of mean degradation
# ============================================================

def plot_cross_pde_heatmap():
    rows = []
    all_stresses = set()

    for pde in PDES:
        df = load_summary(pde)
        all_stresses |= set(df.index)
        for stress, row in df.iterrows():
            rows.append(dict(
                pde=pde,
                stress=stress,
                mean=row["mean"],
            ))

    df_all = pd.DataFrame(rows)
    pivot = df_all.pivot(index="pde", columns="stress", values="mean")

    plt.figure(figsize=(8, 4.5))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Mean degradation")

    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.title("Mean degradation across PDE families")

    save_fig("cross_pde_degradation_heatmap")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Per-PDE bar plots
    for pde in PDES:
        print(f"[plot] {pde}")
        plot_pde_degradation(pde)

    # Cross-PDE overview
    plot_cross_pde_heatmap()

    print(f"[done] plots saved to '{OUT_DIR}/'")

