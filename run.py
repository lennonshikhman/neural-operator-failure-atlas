# run.py
# ============================================================
# SINGLE ENTRY POINT — NEURAL OPERATOR FAILURE ATLAS
#
# This file fully instantiates every experiment promised in:
# "Forcing and Diagnosing Failure Modes of Neural Operators Across Diverse PDE Classes"
#
# PDE families:
#   1) Poisson (elliptic)
#   2) Black–Scholes (parabolic)
#   3) Schrödinger (dispersive)
#   4) Navier–Stokes (advective)
#   5) Kuramoto–Sivashinsky (chaotic)
#
# Experiments:
#   A) Parameter shifts
#   B) Boundary / terminal shifts
#   C) Resolution extrapolation (+ spectral diagnostics)
#   D) Long-horizon rollout
#   E) Perturbation sensitivity
#
# This file freezes the experimental protocol.
# ============================================================

from __future__ import annotations

import os
import pickle
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
# Utilities
# ============================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def baseline_eval(model, pde_name, n, nt, device):
    loader = make_dataloader(
        pde_name=pde_name,
        batch_size=8,
        n_samples=64,
        n=n,
        nt=nt,
        device=device,
        shuffle=False,
        seed=0,
    )
    return evaluate(
        model,
        loader,
        rollout_steps=None if nt is None else 5,
    )


def save(obj, name: str):
    path = os.path.join(RESULTS_DIR, name)
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


# ============================================================
# 1) Poisson — elliptic
# ============================================================

def run_poisson():
    device = get_device()

    train_loader = make_dataloader(
        pde_name="poisson",
        batch_size=8,
        n_samples=512,
        n=128,
        nt=None,
        device=device,
    )

    model = default_fno_1d(3, 1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device)

    results = {}
        
    results = {}

    results["baseline"] = baseline_eval(
        model,
        pde_name="poisson",
        n=128,
        nt=None,
        device=device,
    )

    
    # A: parameter shifts
    results["param_a_scale"] = parameter_shift(
        model, "poisson",
        param_name="a_scale",
        param_values=[0.1, 0.5, 1.0, 2.0, 4.0],
        n=128, nt=None, device=device,
    )

    # B: boundary shifts
    results["boundary_shift"] = boundary_or_payoff_shift(
        model, "poisson",
        shift_values=[-1.0, -0.5, 0.0, 0.5, 1.0],
        n=128, device=device,
    )

    # C: resolution extrapolation
    results["resolution_shift"] = resolution_shift(
        model, "poisson",
        train_n=128,
        test_ns=[64, 128, 256],
        nt=None, device=device,
    )

    # E: perturbations
    results["perturbation_shift"] = perturbation_shift(
        model, "poisson",
        epsilons=[0.0, 1e-3, 1e-2, 5e-2],
        n=128, nt=None, device=device,
    )

    save(results, "poisson.pkl")


# ============================================================
# 2) Black–Scholes — parabolic
# ============================================================

def run_black_scholes():
    device = get_device()

    train_loader = make_dataloader(
        pde_name="black_scholes",
        batch_size=8,
        n_samples=512,
        n=256,
        nt=None,
        device=device,
    )

    model = default_fno_1d(3, 1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device)

    results = {}
    
    results["baseline"] = baseline_eval(
        model,
        pde_name="black_scholes",
        n=256,
        nt=None,
        device=device,
    )

    # A: volatility shift
    results["param_sigma"] = parameter_shift(
        model, "black_scholes",
        param_name="sigma",
        param_values=[0.05, 0.15, 0.3, 0.6, 0.9],
        n=256, nt=None, device=device,
    )

    # B: payoff / terminal structure
    results["payoff_shift"] = boundary_or_payoff_shift(
        model, "black_scholes",
        shift_values=["call", "put", "digital_call", "smooth_call"],
        n=256, device=device,
    )

    # C: resolution
    results["resolution_shift"] = resolution_shift(
        model, "black_scholes",
        train_n=256,
        test_ns=[128, 256, 512],
        nt=None, device=device,
    )

    # E: perturbation
    results["perturbation_shift"] = perturbation_shift(
        model, "black_scholes",
        epsilons=[0.0, 1e-3, 1e-2],
        n=256, nt=None, device=device,
    )

    save(results, "black_scholes.pkl")


# ============================================================
# 3) Schrödinger — dispersive
# ============================================================

def run_schrodinger():
    device = get_device()

    train_loader = make_dataloader(
        pde_name="schrodinger",
        batch_size=4,
        n_samples=256,
        n=256,
        nt=20,
        device=device,
    )

    model = default_fno_1d(3, 2).to(device)
    model = train(model, train_loader, n_steps=4000, lr=1e-3, device=device)

    results = {}
    
    results["baseline"] = baseline_eval(
        model,
        pde_name="schrodinger",
        n=256,
        nt=20,
        device=device,
    )


    # A: nonlinearity shift
    results["param_kappa"] = parameter_shift(
        model, "schrodinger",
        param_name="kappa",
        param_values=[0.3, 0.5, 1.0, 2.0, 4.0],
        n=256, nt=20, device=device,
    )

    # C: resolution
    results["resolution_shift"] = resolution_shift(
        model, "schrodinger",
        train_n=256,
        test_ns=[128, 256, 512],
        nt=20, device=device,
    )

    # D: rollout instability
    results["rollout_shift"] = rollout_horizon_shift(
        model, "schrodinger",
        horizons=[5, 10, 20, 40],
        n=256, nt=40, device=device,
    )

    # E: perturbations
    results["perturbation_shift"] = perturbation_shift(
        model, "schrodinger",
        epsilons=[0.0, 1e-4, 1e-3],
        n=256, nt=20, device=device,
    )

    save(results, "schrodinger.pkl")


# ============================================================
# 4) Navier–Stokes — advective
# ============================================================

def run_navier_stokes():
    device = get_device()

    train_loader = make_dataloader(
        pde_name="navier_stokes",
        batch_size=2,
        n_samples=128,
        n=64,
        nt=20,
        device=device,
    )

    model = default_fno_2d(3, 1).to(device)
    model = train(model, train_loader, n_steps=5000, lr=1e-3, device=device)

    results = {}

    results["baseline"] = baseline_eval(
        model,
        pde_name="navier_stokes",
        n=64,
        nt=20,
        device=device,
    )


    # A: viscosity shift
    results["param_nu"] = parameter_shift(
        model, "navier_stokes",
        param_name="nu",
        param_values=[5e-4, 1e-3, 5e-3, 1e-2, 2e-2],
        n=64, nt=20, device=device,
    )

    # C: resolution
    results["resolution_shift"] = resolution_shift(
        model, "navier_stokes",
        train_n=64,
        test_ns=[32, 64, 96],
        nt=20, device=device,
    )

    # D: rollout
    results["rollout_shift"] = rollout_horizon_shift(
        model, "navier_stokes",
        horizons=[5, 10, 20],
        n=64, nt=40, device=device,
    )

    # E: perturbations
    results["perturbation_shift"] = perturbation_shift(
        model, "navier_stokes",
        epsilons=[0.0, 1e-4, 1e-3],
        n=64, nt=20, device=device,
    )

    save(results, "navier_stokes.pkl")


# ============================================================
# 5) Kuramoto–Sivashinsky — chaotic
# ============================================================

def run_kuramoto_sivashinsky():
    device = get_device()

    train_loader = make_dataloader(
        pde_name="kuramoto_sivashinsky",
        batch_size=8,
        n_samples=256,
        n=128,
        nt=20,
        device=device,
    )

    model = default_fno_1d(2, 1).to(device)
    model = train(model, train_loader, n_steps=3000, lr=1e-3, device=device)

    results = {}
    
    results["baseline"] = baseline_eval(
        model,
        pde_name="kuramoto_sivashinsky",
        n=128,
        nt=20,
        device=device,
    )

    # D: rollout chaos
    results["rollout_shift"] = rollout_horizon_shift(
        model, "kuramoto_sivashinsky",
        horizons=[5, 10, 20, 40],
        n=128, nt=40, device=device,
    )

    # E: perturbation amplification
    results["perturbation_shift"] = perturbation_shift(
        model, "kuramoto_sivashinsky",
        epsilons=[0.0, 1e-4, 1e-3, 1e-2],
        n=128, nt=20, device=device,
    )

    save(results, "kuramoto_sivashinsky.pkl")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    RUN = dict(
        poisson=True,
        black_scholes=True,
        schrodinger=True,
        navier_stokes=True,
        kuramoto_sivashinsky=True,
    )

    if RUN["poisson"]:
        run_poisson()

    if RUN["black_scholes"]:
        run_black_scholes()

    if RUN["schrodinger"]:
        run_schrodinger()

    if RUN["navier_stokes"]:
        run_navier_stokes()

    if RUN["kuramoto_sivashinsky"]:
        run_kuramoto_sivashinsky()
