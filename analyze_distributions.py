#!/usr/bin/env python3
"""Aggregate multi-seed robustness results and emit paper-ready summaries.

New in this version:
- architecture-aware ingestion (`*_MODEL_seed*.pkl`)
- cross-PDE robustness matrix export
- complementary metrics beyond degradation:
  * absolute baseline error
  * rollout growth-rate and amplification summaries
  * high-frequency spectral error fraction under resolution shifts
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from plots import plot_spectral_error

RESULTS_DIR = "results"
OUTPUT_DIR = "analysis"
FIG_DIR = "paper_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CI_Z = 1.96


def summarize_distribution(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci = CI_Z * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - ci,
        "ci_high": mean + ci,
    }


def scalar_l2(eval_dict: Dict) -> float:
    if "static_l2" in eval_dict:
        return float(eval_dict["static_l2"])
    if "one_step_l2" in eval_dict:
        return float(eval_dict["one_step_l2"])
    raise KeyError("No scalar L2 metric found.")


def worst_error(block: Dict) -> float:
    if "errors" in block:
        return max(scalar_l2(e) for e in block["errors"])
    if "evals" in block:
        vals = [scalar_l2(e) for e in block["evals"] if ("one_step_l2" in e or "static_l2" in e)]
        return max(vals) if vals else np.nan
    return np.nan


def high_frequency_fraction(spec: Dict) -> Optional[float]:
    errs = np.asarray(spec.get("errors", []), dtype=np.float64)
    if errs.size == 0:
        return None
    tail_start = int(0.75 * errs.size)
    denom = float(np.sum(errs))
    if denom <= 0:
        return None
    return float(np.sum(errs[tail_start:]) / denom)


def extract_fine_grid_eval(resolution_block: Dict) -> Optional[Dict]:
    resolutions = resolution_block.get("resolutions", [])
    errors = resolution_block.get("errors", [])
    if not resolutions or not errors:
        return None
    idx = int(np.argmax(resolutions))
    return errors[idx]


def analyze_pde(pde_name: str, model_name: str) -> Optional[pd.DataFrame]:
    pat = re.compile(rf"{pde_name}_{model_name}_seed(\d+)\.pkl")
    files = sorted(f for f in os.listdir(RESULTS_DIR) if pat.match(f))
    if not files:
        print(f"[WARN] No files found for {pde_name} model={model_name}")
        return None

    degradations = defaultdict(list)
    baseline_abs = []
    rollout_growth = []
    rollout_amp = []
    resolution_hf = []

    for fname in files:
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            data = pickle.load(f)

        baseline = scalar_l2(data["baseline"])
        baseline_abs.append(baseline)

        for key, block in data.items():
            if key == "baseline":
                continue
            worst = worst_error(block)
            if not np.isnan(worst) and baseline > 0:
                degradations[key].append(worst / baseline)

        if "rollout_shift" in data:
            block = data["rollout_shift"]
            rollout_growth.extend([float(x) for x in block.get("growth_rates", [])])
            rollout_amp.extend([float(x) for x in block.get("amplifications", [])])

        if "resolution_shift" in data:
            fine_eval = extract_fine_grid_eval(data["resolution_shift"])
            if fine_eval and "spectral" in fine_eval:
                hf = high_frequency_fraction(fine_eval["spectral"])
                if hf is not None:
                    resolution_hf.append(hf)

    rows = []
    for stress, vals in degradations.items():
        s = summarize_distribution(vals)
        s["metric"] = "degradation"
        s["stress"] = stress
        rows.append(s)

    extra_metrics = {
        "baseline_abs_l2": baseline_abs,
        "rollout_growth_rate": rollout_growth,
        "rollout_amplification": rollout_amp,
        "resolution_high_freq_fraction": resolution_hf,
    }
    for metric_name, vals in extra_metrics.items():
        if vals:
            s = summarize_distribution(vals)
            s["metric"] = metric_name
            s["stress"] = "aggregate"
            rows.append(s)

    df = pd.DataFrame(rows).sort_values(["metric", "stress"]).reset_index(drop=True)
    out_csv = os.path.join(OUTPUT_DIR, f"{pde_name}_{model_name}_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # one representative spectral figure for resolution-shifted fine grid
    if pde_name in {"schrodinger", "navier_stokes"}:
        with open(os.path.join(RESULTS_DIR, files[0]), "rb") as f:
            data0 = pickle.load(f)
        fine_eval = extract_fine_grid_eval(data0.get("resolution_shift", {}))
        if fine_eval and "spectral" in fine_eval:
            out_pdf = os.path.join(FIG_DIR, f"spectral_{pde_name}_{model_name}.pdf")
            title = f"{pde_name.replace('_', ' ').title()} spectral error ({model_name})"
            plot_spectral_error(fine_eval, title, out_pdf)
            print(f"[saved] {out_pdf}")

    return df


def save_cross_pde_matrix(pdes: List[str], model_name: str):
    records = []
    for pde in pdes:
        f = os.path.join(OUTPUT_DIR, f"{pde}_{model_name}_summary.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        sub = df[df["metric"] == "degradation"]
        for _, row in sub.iterrows():
            records.append({"pde": pde, "stress": row["stress"], "mean": row["mean"], "ci_high": row["ci_high"], "ci_low": row["ci_low"]})

    if not records:
        return

    long_df = pd.DataFrame(records)
    mat = long_df.pivot(index="stress", columns="pde", values="mean").sort_index()
    out = os.path.join(OUTPUT_DIR, f"cross_pde_degradation_matrix_{model_name}.csv")
    mat.to_csv(out)
    print(f"[saved] {out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-seed robustness distributions")
    parser.add_argument("--model", type=str, default="fno", choices=["fno", "deeponet", "cno"], help="Model architecture tag to aggregate")
    args = parser.parse_args()

    pdes = [
        "poisson",
        "black_scholes",
        "schrodinger",
        "navier_stokes",
        "kuramoto_sivashinsky",
    ]

    for pde in pdes:
        analyze_pde(pde, args.model)
    save_cross_pde_matrix(pdes, args.model)


if __name__ == "__main__":
    main()
