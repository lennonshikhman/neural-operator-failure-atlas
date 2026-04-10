from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np
import torch

from data import make_dataloader
from eval import evaluate
from operators import make_model
from stress import (
    boundary_or_payoff_shift,
    parameter_shift,
    perturbation_shift,
    resolution_shift,
    rollout_horizon_shift,
)
from train import get_device, train

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_global_seed(seed: int):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[saved] {path}")


def build_model(model_name: str, dim: int, cin: int, cout: int):
    if model_name == "fno":
        return make_model(
            model_name,
            dim=dim,
            in_channels=cin,
            out_channels=cout,
            width=64,
            depth=4,
            modes1=16 if dim == 1 else 12,
            modes2=None if dim == 1 else 12,
            use_coords=True,
        )
    if model_name == "cno":
        return make_model(
            model_name,
            dim=dim,
            in_channels=cin,
            out_channels=cout,
            width=64,
            depth=5,
            use_coords=True,
        )
    return make_model(
        model_name,
        dim=dim,
        in_channels=cin,
        out_channels=cout,
        width=128,
        depth=2,
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
    return evaluate(model, loader, rollout_steps=None if nt is None else 5)


def run_poisson(seed: int, model_name: str):
    set_global_seed(seed)
    device = get_device()
    train_loader = make_dataloader("poisson", batch_size=8, n_samples=512, n=128, nt=None, device=device, seed=seed)
    model = build_model(model_name, dim=1, cin=3, cout=1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device, seed=seed)

    results = {
        "baseline": baseline_eval(model, "poisson", 128, None, device, seed),
        "param_a_scale": parameter_shift(model, "poisson", "a_scale", [0.1, 0.5, 1.0, 2.0, 4.0], 128, None, device=device, seed=seed),
        "boundary_shift": boundary_or_payoff_shift(model, "poisson", [-1.0, -0.5, 0.0, 0.5, 1.0], 128, device=device, seed=seed),
        "resolution_shift": resolution_shift(model, "poisson", 128, [64, 128, 256], None, device=device, seed=seed),
        "perturbation_shift": perturbation_shift(model, "poisson", [0.0, 1e-3, 1e-2, 5e-2], 128, None, device=device, seed=seed),
    }
    save(results, f"{RESULTS_DIR}/poisson_{model_name}_seed{seed}.pkl")


def run_black_scholes(seed: int, model_name: str):
    set_global_seed(seed)
    device = get_device()
    train_loader = make_dataloader("black_scholes", batch_size=8, n_samples=512, n=256, nt=None, device=device, seed=seed)
    model = build_model(model_name, dim=1, cin=3, cout=1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device, seed=seed)

    results = {
        "baseline": baseline_eval(model, "black_scholes", 256, None, device, seed),
        "param_sigma": parameter_shift(model, "black_scholes", "sigma", [0.05, 0.15, 0.3, 0.6, 0.9], 256, None, device=device, seed=seed),
        "payoff_shift": boundary_or_payoff_shift(model, "black_scholes", ["call", "put", "digital_call", "smooth_call"], 256, device=device, seed=seed),
        "resolution_shift": resolution_shift(model, "black_scholes", 256, [128, 256, 512], None, device=device, seed=seed),
        "perturbation_shift": perturbation_shift(model, "black_scholes", [0.0, 1e-3, 1e-2], 256, None, device=device, seed=seed),
    }
    save(results, f"{RESULTS_DIR}/black_scholes_{model_name}_seed{seed}.pkl")


def run_schrodinger(seed: int, model_name: str):
    set_global_seed(seed)
    device = get_device()
    train_loader = make_dataloader("schrodinger", batch_size=4, n_samples=256, n=256, nt=20, device=device, seed=seed)
    model = build_model(model_name, dim=1, cin=3, cout=2).to(device)
    model = train(model, train_loader, n_steps=4000, lr=1e-3, device=device, seed=seed)

    results = {
        "baseline": baseline_eval(model, "schrodinger", 256, 20, device, seed),
        "param_kappa": parameter_shift(model, "schrodinger", "kappa", [0.3, 0.5, 1.0, 2.0, 4.0], 256, 20, device=device, seed=seed),
        "resolution_shift": resolution_shift(model, "schrodinger", 256, [128, 256, 512], 20, device=device, seed=seed),
        "rollout_shift": rollout_horizon_shift(model, "schrodinger", [5, 10, 20, 40], 256, 40, device=device, seed=seed),
        "perturbation_shift": perturbation_shift(model, "schrodinger", [0.0, 1e-4, 1e-3], 256, 20, device=device, seed=seed),
    }
    save(results, f"{RESULTS_DIR}/schrodinger_{model_name}_seed{seed}.pkl")


def run_navier_stokes(seed: int, model_name: str):
    set_global_seed(seed)
    device = get_device()
    train_loader = make_dataloader("navier_stokes", batch_size=2, n_samples=128, n=64, nt=20, device=device, seed=seed)
    model = build_model(model_name, dim=2, cin=3, cout=1).to(device)
    model = train(model, train_loader, n_steps=5000, lr=1e-3, device=device, seed=seed)

    results = {
        "baseline": baseline_eval(model, "navier_stokes", 64, 20, device, seed),
        "param_nu": parameter_shift(model, "navier_stokes", "nu", [5e-4, 1e-3, 5e-3, 1e-2, 2e-2], 64, 20, device=device, seed=seed),
        "resolution_shift": resolution_shift(model, "navier_stokes", 64, [32, 64, 96], 20, device=device, seed=seed),
        "rollout_shift": rollout_horizon_shift(model, "navier_stokes", [5, 10, 20], 64, 40, device=device, seed=seed),
        "perturbation_shift": perturbation_shift(model, "navier_stokes", [0.0, 1e-4, 1e-3], 64, 20, device=device, seed=seed),
    }
    save(results, f"{RESULTS_DIR}/navier_stokes_{model_name}_seed{seed}.pkl")


def run_kuramoto_sivashinsky(seed: int, model_name: str):
    set_global_seed(seed)
    device = get_device()
    train_loader = make_dataloader("kuramoto_sivashinsky", batch_size=8, n_samples=256, n=128, nt=20, device=device, seed=seed)
    model = build_model(model_name, dim=1, cin=2, cout=1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device, seed=seed)

    results = {
        "baseline": baseline_eval(model, "kuramoto_sivashinsky", 128, 20, device, seed),
        "rollout_shift": rollout_horizon_shift(model, "kuramoto_sivashinsky", [5, 10, 20, 40], 128, 40, device=device, seed=seed),
        "perturbation_shift": perturbation_shift(model, "kuramoto_sivashinsky", [0.0, 1e-4, 1e-3, 1e-2], 128, 20, device=device, seed=seed),
    }
    save(results, f"{RESULTS_DIR}/kuramoto_sivashinsky_{model_name}_seed{seed}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic multi-seed robustness sweep for neural operator architectures")
    parser.add_argument("--iters", type=int, default=10, help="Number of random seeds to sweep")
    parser.add_argument("--model", type=str, default="fno", choices=["fno", "deeponet", "cno"], help="Architecture to evaluate")
    args = parser.parse_args()

    print(f"\n=== Running deterministic sweep over {args.iters} seeds | model={args.model} ===")
    for seed in range(args.iters):
        print(f"\n===== RUNNING SEED {seed} / model={args.model} =====")
        run_poisson(seed, args.model)
        run_black_scholes(seed, args.model)
        run_schrodinger(seed, args.model)
        run_navier_stokes(seed, args.model)
        run_kuramoto_sivashinsky(seed, args.model)
