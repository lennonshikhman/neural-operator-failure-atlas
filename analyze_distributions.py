#!/usr/bin/env python3
"""
Aggregate multi-seed neural operator failure results and generate paper figures.

Reads per-seed .pkl files produced by run.py and computes:
- baseline-normalized degradation
- mean, std
- 95% confidence intervals

Additionally:
- extracts representative spectral diagnostics
- saves paper-ready spectral figures (no reruns required)
"""

from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# plotting
from plots import plot_spectral_error


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = "results"
OUTPUT_DIR = "analysis"
FIG_DIR = "paper_visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CI_Z = 1.96  # normal 95% CI


# ============================================================
# Helpers
# ============================================================

def scalar_l2(eval_dict: Dict) -> float:
    if "static_l2" in eval_dict:
        return float(eval_dict["static_l2"])
    if "one_step_l2" in eval_dict:
        return float(eval_dict["one_step_l2"])
    raise KeyError("No scalar L2 metric found.")


def worst_error(block: Dict) -> float:
    """
    Extract the worst scalar L2 error from a stress block.
    """
    if "errors" in block:
        return max(scalar_l2(e) for e in block["errors"])

    # rollout case
    if "evals" in block:
        vals = []
        for e in block["evals"]:
            if "one_step_l2" in e:
                vals.append(e["one_step_l2"])
        return max(vals) if vals else np.nan

    return np.nan


def summarize_distribution(xs: List[float]) -> Dict:
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    mean = xs.mean()
    std = xs.std(ddof=1) if n > 1 else 0.0
    ci = CI_Z * std / np.sqrt(n) if n > 1 else 0.0

    return dict(
        n=n,
        mean=mean,
        std=std,
        ci_low=mean - ci,
        ci_high=mean + ci,
    )


def extract_fine_grid_spectral(block: Dict) -> Optional[Dict]:
    """
    Extract the spectral evaluation corresponding to the highest spatial resolution
    in a resolution_shift block.
    """
    if "resolutions" not in block or "errors" not in block:
        return None

    resolutions = block["resolutions"]
    errors = block["errors"]

    if not resolutions or not errors:
        return None

    idx = int(np.argmax(resolutions))
    eval_dict = errors[idx]

    if "spectral" in eval_dict and eval_dict["spectral"] is not None:
        return eval_dict

    return None


# ============================================================
# Main aggregation
# ============================================================

def analyze_pde(pde_name: str):
    """
    Aggregate all seed files for one PDE and generate figures if applicable.
    """
    pattern = re.compile(rf"{pde_name}_seed(\d+)\.pkl")
    files = sorted(f for f in os.listdir(RESULTS_DIR) if pattern.match(f))

    if not files:
        print(f"[WARN] No files found for PDE '{pde_name}'")
        return

    print(f"\n=== PDE: {pde_name} ({len(files)} seeds) ===")

    degradations = defaultdict(list)

    # --------------------------------------------------------
    # Pass 1: aggregate statistics across all seeds
    # --------------------------------------------------------

    for fname in files:
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            data = pickle.load(f)

        baseline = scalar_l2(data["baseline"])

        for key, block in data.items():
            if key == "baseline":
                continue

            worst = worst_error(block)
            if np.isnan(worst) or baseline <= 0:
                continue

            degradations[key].append(worst / baseline)

    rows = []
    for stress, vals in degradations.items():
        stats = summarize_distribution(vals)
        stats["stress"] = stress
        rows.append(stats)

        print(
            f"{stress:<24} "
            f"mean={stats['mean']:.3f}  "
            f"CI95=[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]"
        )

    df = pd.DataFrame(rows).set_index("stress")
    out_csv = os.path.join(OUTPUT_DIR, f"{pde_name}_degradation_summary.csv")
    df.to_csv(out_csv)
    print(f"[saved] {out_csv}")

    # --------------------------------------------------------
    # Pass 2: generate spectral figure (single representative)
    # --------------------------------------------------------

    if pde_name in {"schrodinger", "navier_stokes"}:
        # choose deterministic representative seed
        fname = files[0]

        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            data = pickle.load(f)

        block = data.get("resolution_shift")
        if block is None:
            print(f"[WARN] No resolution_shift block for {pde_name}")
            return

        spec_eval = extract_fine_grid_spectral(block)
        if spec_eval is None:
            print(f"[WARN] No spectral data for fine-grid {pde_name}")
            return

        out_pdf = os.path.join(FIG_DIR, f"spectral_{pde_name}.pdf")
        title = f"{pde_name.replace('_', ' ').title()} spectral error (fine grid)"

        plot_spectral_error(
            evaluation=spec_eval,
            title=title,
            savepath=out_pdf,
        )

        print(f"[saved] {out_pdf}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    PDES = [
        "poisson",
        "black_scholes",
        "schrodinger",
        "navier_stokes",
        "kuramoto_sivashinsky",
    ]

    for pde in PDES:
        analyze_pde(pde)

