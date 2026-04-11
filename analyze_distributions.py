#!/usr/bin/env python3
"""Aggregate multi-seed robustness results and emit paper-ready summaries.

This version:
- ingests architecture-specific files of the form `PDE_MODEL_seed*.pkl`
- writes per-(PDE, model) summary CSVs
- writes per-model cross-PDE degradation matrices
- emits architecture-specific spectral figures for Schrödinger and Navier--Stokes
- emits comparative paper figures once summaries for all three models are available

Intended usage:
    python analyze_distributions.py --model fno
    python analyze_distributions.py --model deeponet
    python analyze_distributions.py --model cno

You can run the three commands in any order. Comparative figures are generated
whenever the required summary CSVs for all three models are present.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots import plot_spectral_error

RESULTS_DIR = "results"
OUTPUT_DIR = "analysis"
FIG_DIR = "paper_visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CI_Z = 1.96

ALL_MODELS = ["fno", "deeponet", "cno"]
ALL_PDES = [
    "poisson",
    "black_scholes",
    "schrodinger",
    "navier_stokes",
    "kuramoto_sivashinsky",
]


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
        vals = []
        for e in block["errors"]:
            try:
                vals.append(scalar_l2(e))
            except KeyError:
                pass
        return max(vals) if vals else np.nan

    if "evals" in block:
        vals = []
        for e in block["evals"]:
            try:
                vals.append(scalar_l2(e))
            except KeyError:
                pass
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

    if len(resolutions) == 0 or len(errors) == 0:
        return None

    idx = int(np.argmax(resolutions))
    if idx >= len(errors):
        return None
    return errors[idx]


def pretty_pde_name(pde: str) -> str:
    mapping = {
        "poisson": "Poisson",
        "black_scholes": "Black-Scholes",
        "schrodinger": "Nonlinear Schrödinger",
        "navier_stokes": "Navier-Stokes",
        "kuramoto_sivashinsky": "Kuramoto-Sivashinsky",
    }
    return mapping.get(pde, pde.replace("_", " ").title())


def pretty_model_name(model: str) -> str:
    mapping = {
        "fno": "FNO",
        "deeponet": "DeepONet",
        "cno": "CNO",
    }
    return mapping.get(model, model)


def pretty_stress_name(stress: str) -> str:
    mapping = {
        "param_shift": "Parameter shift",
        "boundary_shift": "Boundary shift",
        "resolution_shift": "Resolution",
        "rollout_shift": "Rollout",
        "perturbation_shift": "Perturbation",
        "payoff_shift": "Payoff shift",
        "terminal_shift": "Terminal shift",
        "coeff_shift": "Coefficient shift",
        "coefficient_shift": "Coefficient shift",
    }
    return mapping.get(stress, stress.replace("_", " ").title())


def canonical_stress_name(stress: str) -> str:
    mapping = {
        "param_shift": "param_shift",
        "parameter_shift": "param_shift",
        "coeff_shift": "param_shift",
        "coefficient_shift": "param_shift",
        "boundary_shift": "boundary_shift",
        "terminal_shift": "boundary_shift",
        "payoff_shift": "boundary_shift",
        "resolution_shift": "resolution_shift",
        "rollout_shift": "rollout_shift",
        "perturbation_shift": "perturbation_shift",
    }
    return mapping.get(stress, stress)


def stress_order_for_pde(pde: str) -> List[str]:
    mapping = {
        "poisson": [
            "param_shift",
            "boundary_shift",
            "resolution_shift",
            "perturbation_shift",
        ],
        "black_scholes": [
            "param_shift",
            "boundary_shift",
            "resolution_shift",
            "perturbation_shift",
        ],
        "schrodinger": [
            "param_shift",
            "resolution_shift",
            "rollout_shift",
            "perturbation_shift",
        ],
        "navier_stokes": [
            "param_shift",
            "resolution_shift",
            "rollout_shift",
            "perturbation_shift",
        ],
        "kuramoto_sivashinsky": [
            "rollout_shift",
            "perturbation_shift",
        ],
    }
    return mapping.get(pde, [])


def analyze_pde(pde_name: str, model_name: str) -> Optional[pd.DataFrame]:
    pat = re.compile(rf"{re.escape(pde_name)}_{re.escape(model_name)}_seed(\d+)\.pkl")
    files = sorted(f for f in os.listdir(RESULTS_DIR) if pat.match(f))

    if not files:
        print(f"[WARN] No files found for pde={pde_name} model={model_name}")
        return None

    degradations = defaultdict(list)
    baseline_abs = []
    rollout_growth = []
    rollout_amp = []
    resolution_hf = []

    for fname in files:
        with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
            data = pickle.load(f)

        if "baseline" not in data:
            print(f"[WARN] Missing baseline in {fname}; skipping")
            continue

        try:
            baseline = scalar_l2(data["baseline"])
        except KeyError:
            print(f"[WARN] Could not parse baseline scalar L2 in {fname}; skipping")
            continue

        baseline_abs.append(baseline)

        for key, block in data.items():
            if key == "baseline":
                continue

            worst = worst_error(block)
            if not np.isnan(worst) and baseline > 0:
                degradations[key].append(worst / baseline)

        if "rollout_shift" in data:
            block = data["rollout_shift"]
            rollout_growth.extend(
                [float(x) for x in block.get("growth_rates", []) if np.isfinite(x)]
            )
            rollout_amp.extend(
                [float(x) for x in block.get("amplifications", []) if np.isfinite(x)]
            )

        if "resolution_shift" in data:
            fine_eval = extract_fine_grid_eval(data["resolution_shift"])
            if fine_eval and "spectral" in fine_eval:
                hf = high_frequency_fraction(fine_eval["spectral"])
                if hf is not None and np.isfinite(hf):
                    resolution_hf.append(hf)

    rows = []

    for stress, vals in degradations.items():
        if not vals:
            continue
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
        if not vals:
            continue
        s = summarize_distribution(vals)
        s["metric"] = metric_name
        s["stress"] = "aggregate"
        rows.append(s)

    if not rows:
        print(f"[WARN] No summary rows produced for pde={pde_name} model={model_name}")
        return None

    df = pd.DataFrame(rows).sort_values(["metric", "stress"]).reset_index(drop=True)

    out_csv = os.path.join(OUTPUT_DIR, f"{pde_name}_{model_name}_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # Save one representative spectral figure using the first seed file.
    if pde_name in {"schrodinger", "navier_stokes"}:
        with open(os.path.join(RESULTS_DIR, files[0]), "rb") as f:
            data0 = pickle.load(f)

        fine_eval = extract_fine_grid_eval(data0.get("resolution_shift", {}))
        if fine_eval and "spectral" in fine_eval:
            out_pdf = os.path.join(FIG_DIR, f"spectral_{pde_name}_{model_name}.pdf")
            title = f"{pretty_pde_name(pde_name)} spectral error ({pretty_model_name(model_name)})"
            plot_spectral_error(fine_eval, title, out_pdf)
            print(f"[saved] {out_pdf}")

    return df


def save_cross_pde_matrix(pdes: List[str], model_name: str) -> None:
    records = []

    for pde in pdes:
        f = os.path.join(OUTPUT_DIR, f"{pde}_{model_name}_summary.csv")
        if not os.path.exists(f):
            continue

        df = pd.read_csv(f)
        sub = df[df["metric"] == "degradation"]

        for _, row in sub.iterrows():
            records.append(
                {
                    "pde": pde,
                    "stress": canonical_stress_name(str(row["stress"])),
                    "mean": row["mean"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                }
            )

    if not records:
        print(f"[WARN] No degradation records found for model={model_name}")
        return

    long_df = pd.DataFrame(records)
    mat = long_df.pivot(index="stress", columns="pde", values="mean").sort_index()

    out = os.path.join(OUTPUT_DIR, f"cross_pde_degradation_matrix_{model_name}.csv")
    mat.to_csv(out)
    print(f"[saved] {out}")


def load_summary(pde: str, model_name: str) -> Optional[pd.DataFrame]:
    f = os.path.join(OUTPUT_DIR, f"{pde}_{model_name}_summary.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def collect_degradation_records(pdes: List[str], models: List[str]) -> pd.DataFrame:
    records = []

    for pde in pdes:
        for model in models:
            df = load_summary(pde, model)
            if df is None:
                continue

            sub = df[df["metric"] == "degradation"].copy()
            for _, row in sub.iterrows():
                records.append(
                    {
                        "pde": pde,
                        "model": model,
                        "stress": canonical_stress_name(str(row["stress"])),
                        "mean": float(row["mean"]),
                        "ci_low": float(row["ci_low"]),
                        "ci_high": float(row["ci_high"]),
                    }
                )

    return pd.DataFrame(records)


def collect_baseline_records(pdes: List[str], models: List[str]) -> pd.DataFrame:
    records = []

    for pde in pdes:
        for model in models:
            df = load_summary(pde, model)
            if df is None:
                continue

            sub = df[(df["metric"] == "baseline_abs_l2") & (df["stress"] == "aggregate")]
            if sub.empty:
                continue

            row = sub.iloc[0]
            records.append(
                {
                    "pde": pde,
                    "model": model,
                    "mean": float(row["mean"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    return pd.DataFrame(records)


def have_all_model_summaries(pdes: List[str], models: List[str]) -> bool:
    for pde in pdes:
        for model in models:
            f = os.path.join(OUTPUT_DIR, f"{pde}_{model}_summary.csv")
            if not os.path.exists(f):
                return False
    return True


def plot_pde_degradation_bars(pde: str, all_df: pd.DataFrame) -> None:
    sub = all_df[all_df["pde"] == pde].copy()
    if sub.empty:
        print(f"[WARN] No degradation data available for pde={pde}")
        return

    models = ["fno", "deeponet", "cno"]
    stresses = [s for s in stress_order_for_pde(pde) if s in set(sub["stress"])]

    if not stresses:
        print(f"[WARN] No recognized stress names found for pde={pde}")
        print(f"       Available stresses: {sorted(set(sub['stress']))}")
        return

    x = np.arange(len(stresses), dtype=np.float64)
    width = 0.24
    offsets = [-width, 0.0, width]

    plt.figure(figsize=(8.5, 4.8))

    for offset, model in zip(offsets, models):
        model_sub = sub[sub["model"] == model]
        means = []
        errs_low = []
        errs_high = []

        for stress in stresses:
            row = model_sub[model_sub["stress"] == stress]
            if row.empty:
                means.append(np.nan)
                errs_low.append(0.0)
                errs_high.append(0.0)
            else:
                r = row.iloc[0]
                means.append(float(r["mean"]))
                errs_low.append(float(r["mean"] - r["ci_low"]))
                errs_high.append(float(r["ci_high"] - r["mean"]))

        yerr = np.vstack([errs_low, errs_high])
        plt.bar(x + offset, means, width=width, label=pretty_model_name(model))
        plt.errorbar(x + offset, means, yerr=yerr, fmt="none", capsize=3)

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(x, [pretty_stress_name(s) for s in stresses], rotation=20, ha="right")
    plt.ylabel("Degradation factor")
    plt.title(pretty_pde_name(pde))
    plt.legend()
    plt.tight_layout()

    out_pdf = os.path.join(FIG_DIR, f"{pde}_degradation_bar.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"[saved] {out_pdf}")


def plot_cross_pde_degradation_heatmap(all_df: pd.DataFrame) -> None:
    if all_df.empty:
        print("[WARN] No comparative degradation data available for heatmap")
        return

    pdes = [
        "schrodinger",
        "poisson",
        "navier_stokes",
        "black_scholes",
        "kuramoto_sivashinsky",
    ]
    models = ["fno", "deeponet", "cno"]

    row_specs = []
    for pde in pdes:
        for stress in stress_order_for_pde(pde):
            row_specs.append((pde, stress))

    mat = np.full((len(row_specs), len(models)), np.nan, dtype=np.float64)

    for i, (pde, stress) in enumerate(row_specs):
        for j, model in enumerate(models):
            row = all_df[
                (all_df["pde"] == pde)
                & (all_df["stress"] == stress)
                & (all_df["model"] == model)
            ]
            if not row.empty:
                mat[i, j] = float(row.iloc[0]["mean"])

    plt.figure(figsize=(7.8, max(6.0, 0.34 * len(row_specs))))
    im = plt.imshow(mat, aspect="auto")

    plt.xticks(np.arange(len(models)), [pretty_model_name(m) for m in models])

    ylabels = [
        f"{pretty_pde_name(pde)} | {pretty_stress_name(stress)}"
        for pde, stress in row_specs
    ]
    plt.yticks(np.arange(len(row_specs)), ylabels)

    cbar = plt.colorbar(im)
    cbar.set_label("Mean degradation factor")

    plt.title("Cross-PDE degradation overview")
    plt.tight_layout()

    out_pdf = os.path.join(FIG_DIR, "cross_pde_degradation_heatmap.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"[saved] {out_pdf}")


def plot_baseline_error_summary(baseline_df: pd.DataFrame) -> None:
    if baseline_df.empty:
        print("[WARN] No comparative baseline data available")
        return

    pdes = [
        "schrodinger",
        "poisson",
        "navier_stokes",
        "black_scholes",
        "kuramoto_sivashinsky",
    ]
    models = ["fno", "deeponet", "cno"]

    x = np.arange(len(pdes), dtype=np.float64)
    width = 0.24
    offsets = [-width, 0.0, width]

    plt.figure(figsize=(9.2, 4.9))

    for offset, model in zip(offsets, models):
        sub = baseline_df[baseline_df["model"] == model]
        means = []
        errs_low = []
        errs_high = []

        for pde in pdes:
            row = sub[sub["pde"] == pde]
            if row.empty:
                means.append(np.nan)
                errs_low.append(0.0)
                errs_high.append(0.0)
            else:
                r = row.iloc[0]
                means.append(float(r["mean"]))
                errs_low.append(float(r["mean"] - r["ci_low"]))
                errs_high.append(float(r["ci_high"] - r["mean"]))

        yerr = np.vstack([errs_low, errs_high])
        plt.bar(x + offset, means, width=width, label=pretty_model_name(model))
        plt.errorbar(x + offset, means, yerr=yerr, fmt="none", capsize=3)

    plt.xticks(x, [pretty_pde_name(p) for p in pdes], rotation=20, ha="right")
    plt.ylabel("Baseline absolute $L^2$ error")
    plt.title("In-distribution baseline error across PDE families")
    plt.legend()
    plt.tight_layout()

    out_pdf = os.path.join(FIG_DIR, "baseline_error_summary.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"[saved] {out_pdf}")


def maybe_generate_comparative_figures(pdes: List[str], models: List[str]) -> None:
    if not have_all_model_summaries(pdes, models):
        print("[info] Comparative figures not generated yet; waiting for all three model summary CSVs.")
        return

    all_deg = collect_degradation_records(pdes, models)
    if not all_deg.empty:
        plot_cross_pde_degradation_heatmap(all_deg)
        for pde in pdes:
            plot_pde_degradation_bars(pde, all_deg)

    baseline_df = collect_baseline_records(pdes, models)
    if not baseline_df.empty:
        plot_baseline_error_summary(baseline_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-seed robustness distributions")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fno", "deeponet", "cno"],
        help="Model architecture tag to aggregate",
    )
    args = parser.parse_args()

    for pde in ALL_PDES:
        analyze_pde(pde, args.model)

    save_cross_pde_matrix(ALL_PDES, args.model)
    maybe_generate_comparative_figures(ALL_PDES, ALL_MODELS)


if __name__ == "__main__":
    main()
