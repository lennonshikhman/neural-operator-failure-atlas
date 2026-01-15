# run.py
# ============================================================
# SINGLE ENTRY POINT — NEURAL OPERATOR FAILURE ATLAS
# Deterministic multi-seed sweep
# ============================================================

from __future__ import annotations

import os
import pickle
import random
import numpy as np
import argparse
import torch

from train import train, get_device
from operators import make_fno
from data import make_dataloader
from stress import (
    resolution_shift,
    parameter_shift,
    rollout_horizon_shift,
    perturbation_shift,
    boundary_or_payoff_shift,
)
from eval import evaluate

# ============================================================
# Global determinism
# ============================================================

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_global_seed(seed: int):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Utilities
# ============================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[saved] {path}")


def default_fno_1d(cin, cout):
    return make_fno(
        dim=1,
        in_channels=cin,
        out_channels=cout,
        width=64,
        depth=4,
        modes1=16,
        use_coords=True,
    )


def default_fno_2d(cin, cout):
    return make_fno(
        dim=2,
        in_channels=cin,
        out_channels=cout,
        width=64,
        depth=4,
        modes1=12,
        modes2=12,
        use_coords=True,
    )


def baseline_eval(model, pde_name, n, nt, device, seed):
    loader = make_dataloader(
        pde_name=pde_name,
        batch_size=8,
        n_samples=64,
        n=n,
        nt=nt,
        device=device,
        shuffle=False,
        seed=seed,
    )
    return evaluate(
        model,
        loader,
        rollout_steps=None if nt is None else 5,
    )


# ============================================================
# Per-PDE runners (seed-aware)
# ============================================================

def run_poisson(seed: int):
    set_global_seed(seed)
    device = get_device()

    train_loader = make_dataloader(
        pde_name="poisson",
        batch_size=8,
        n_samples=512,
        n=128,
        nt=None,
        device=device,
        seed=seed,
    )

    model = default_fno_1d(3, 1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device, seed=seed)

    results = {}
    results["baseline"] = baseline_eval(model, "poisson", 128, None, device, seed)

    results["param_a_scale"] = parameter_shift(
        model, "poisson",
        param_name="a_scale",
        param_values=[0.1, 0.5, 1.0, 2.0, 4.0],
        n=128, nt=None, device=device, seed=seed,
    )

    results["boundary_shift"] = boundary_or_payoff_shift(
        model, "poisson",
        shift_values=[-1.0, -0.5, 0.0, 0.5, 1.0],
        n=128, device=device, seed=seed,
    )

    results["resolution_shift"] = resolution_shift(
        model, "poisson",
        train_n=128,
        test_ns=[64, 128, 256],
        nt=None, device=device, seed=seed,
    )

    results["perturbation_shift"] = perturbation_shift(
        model, "poisson",
        epsilons=[0.0, 1e-3, 1e-2, 5e-2],
        n=128, nt=None, device=device, seed=seed,
    )

    save(results, f"{RESULTS_DIR}/poisson_seed{seed}.pkl")


# --- identical pattern for other PDEs ---

def run_black_scholes(seed: int):
    set_global_seed(seed)
    device = get_device()

    train_loader = make_dataloader(
        pde_name="black_scholes",
        batch_size=8,
        n_samples=512,
        n=256,
        nt=None,
        device=device,
        seed=seed,
    )

    model = default_fno_1d(3, 1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device, seed=seed)

    results = {}
    results["baseline"] = baseline_eval(model, "black_scholes", 256, None, device, seed)

    results["param_sigma"] = parameter_shift(
        model, "black_scholes",
        param_name="sigma",
        param_values=[0.05, 0.15, 0.3, 0.6, 0.9],
        n=256, nt=None, device=device, seed=seed,
    )

    results["payoff_shift"] = boundary_or_payoff_shift(
        model, "black_scholes",
        shift_values=["call", "put", "digital_call", "smooth_call"],
        n=256, device=device, seed=seed,
    )

    results["resolution_shift"] = resolution_shift(
        model, "black_scholes",
        train_n=256,
        test_ns=[128, 256, 512],
        nt=None, device=device, seed=seed,
    )

    results["perturbation_shift"] = perturbation_shift(
        model, "black_scholes",
        epsilons=[0.0, 1e-3, 1e-2],
        n=256, nt=None, device=device, seed=seed,
    )

    save(results, f"{RESULTS_DIR}/black_scholes_seed{seed}.pkl")

def run_schrodinger(seed: int):
    set_global_seed(seed)
    device = get_device()

    train_loader = make_dataloader(
        pde_name="schrodinger",
        batch_size=4,
        n_samples=256,
        n=256,
        nt=20,
        device=device,
        seed=seed,
    )

    model = default_fno_1d(3, 2).to(device)
    model = train(
        model,
        train_loader,
        n_steps=4000,
        lr=1e-3,
        device=device,
        seed=seed,
    )

    results = {}

    results["baseline"] = baseline_eval(
        model,
        pde_name="schrodinger",
        n=256,
        nt=20,
        device=device,
        seed=seed,
    )

    results["param_kappa"] = parameter_shift(
        model,
        "schrodinger",
        param_name="kappa",
        param_values=[0.3, 0.5, 1.0, 2.0, 4.0],
        n=256,
        nt=20,
        device=device,
        seed=seed,
    )

    results["resolution_shift"] = resolution_shift(
        model,
        "schrodinger",
        train_n=256,
        test_ns=[128, 256, 512],
        nt=20,
        device=device,
        seed=seed,
    )

    results["rollout_shift"] = rollout_horizon_shift(
        model,
        "schrodinger",
        horizons=[5, 10, 20, 40],
        n=256,
        nt=40,
        device=device,
        seed=seed,
    )

    results["perturbation_shift"] = perturbation_shift(
        model,
        "schrodinger",
        epsilons=[0.0, 1e-4, 1e-3],
        n=256,
        nt=20,
        device=device,
        seed=seed,
    )

    save(results, f"{RESULTS_DIR}/schrodinger_seed{seed}.pkl")



def run_navier_stokes(seed: int):
    set_global_seed(seed)
    device = get_device()

    train_loader = make_dataloader(
        pde_name="navier_stokes",
        batch_size=2,
        n_samples=128,
        n=64,
        nt=20,
        device=device,
        seed=seed,
    )

    model = default_fno_2d(3, 1).to(device)
    model = train(
        model,
        train_loader,
        n_steps=5000,
        lr=1e-3,
        device=device,
        seed=seed,
    )

    results = {}

    results["baseline"] = baseline_eval(
        model,
        pde_name="navier_stokes",
        n=64,
        nt=20,
        device=device,
        seed=seed,
    )

    results["param_nu"] = parameter_shift(
        model,
        "navier_stokes",
        param_name="nu",
        param_values=[5e-4, 1e-3, 5e-3, 1e-2, 2e-2],
        n=64,
        nt=20,
        device=device,
        seed=seed,
    )

    results["resolution_shift"] = resolution_shift(
        model,
        "navier_stokes",
        train_n=64,
        test_ns=[32, 64, 96],
        nt=20,
        device=device,
        seed=seed,
    )

    results["rollout_shift"] = rollout_horizon_shift(
        model,
        "navier_stokes",
        horizons=[5, 10, 20],
        n=64,
        nt=40,
        device=device,
        seed=seed,
    )

    results["perturbation_shift"] = perturbation_shift(
        model,
        "navier_stokes",
        epsilons=[0.0, 1e-4, 1e-3],
        n=64,
        nt=20,
        device=device,
        seed=seed,
    )

    save(results, f"{RESULTS_DIR}/navier_stokes_seed{seed}.pkl")

def run_kuramoto_sivashinsky(seed: int):
    set_global_seed(seed)
    device = get_device()

    train_loader = make_dataloader(
        pde_name="kuramoto_sivashinsky",
        batch_size=8,
        n_samples=256,
        n=128,
        nt=20,
        device=device,
        seed=seed,
    )

    model = default_fno_1d(2, 1).to(device)
    model = train(
        model,
        train_loader,
        n_steps=3000,
        lr=1e-3,
        device=device,
        seed=seed,
    )

    results = {}

    results["baseline"] = baseline_eval(
        model,
        pde_name="kuramoto_sivashinsky",
        n=128,
        nt=20,
        device=device,
        seed=seed,
    )

    results["rollout_shift"] = rollout_horizon_shift(
        model,
        "kuramoto_sivashinsky",
        horizons=[5, 10, 20, 40],
        n=128,
        nt=40,
        device=device,
        seed=seed,
    )

    results["perturbation_shift"] = perturbation_shift(
        model,
        "kuramoto_sivashinsky",
        epsilons=[0.0, 1e-4, 1e-3, 1e-2],
        n=128,
        nt=20,
        device=device,
        seed=seed,
    )

    save(results, f"{RESULTS_DIR}/kuramoto_sivashinsky_seed{seed}.pkl")


# ============================================================
# Seed sweep driver
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Deterministic multi-seed sweep for neural operator failure atlas"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of random seeds to sweep (default: 10)",
    )

    args = parser.parse_args()
    iters = int(args.iters)

    SEEDS = list(range(iters))

    print(f"\n=== Running deterministic sweep over {iters} seeds ===")

    for seed in SEEDS:
        print(f"\n===== RUNNING SEED {seed} =====")

        run_poisson(seed)
        run_black_scholes(seed)
        run_schrodinger(seed)
        run_navier_stokes(seed)
        run_kuramoto_sivashinsky(seed)

