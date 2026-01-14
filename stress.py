# stress.py
# Failure-mode stress tests for neural operator failure atlas
#
# Proposal coverage:
#   A) Parameter / coefficient shifts (continuous + categorical)
#   B) Boundary / terminal condition shifts (Poisson + Black–Scholes)
#   C) Resolution extrapolation (+ spectral diagnostics returned by eval.py)
#   D) Long-horizon rollout stability (curves + growth summaries)
#   E) Perturbation sensitivity (state-only by default; optional params too)
#
# Design:
#   - All functions take a trained model, generate dataloaders, call eval.evaluate(),
#     and return raw dicts/arrays (no plotting).
#   - Determinism: a `seed` parameter is exposed everywhere so comparisons are fair.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from data import make_dataloader, is_time_dependent
from eval import evaluate

from metrics import error_growth_rate, amplification_factor, perturbation_amplification


# ============================================================
# Helpers
# ============================================================

def _default_device_from_model(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_param_channel_mask(x: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    """
    Heuristic: "parameter channels" are constant across the spatial grid for each sample.
    Works for:
      - Schrodinger: kappa channel constant in x
      - KS: L channel constant in x
      - Navier–Stokes: nu channel constant in (x,y)
      - Black–Scholes: sigma/r channels constant in S
    Returns:
      mask: (B, C) boolean; True means channel is (approximately) constant for that sample.
    """
    if x.ndim == 3:
        # (B, N, C)
        spatial = x[:, :, :]
        # std over N
        std = spatial.std(dim=1)  # (B, C)
        return std <= tol
    if x.ndim == 4:
        # (B, N, N, C)
        spatial = x[:, :, :, :]
        std = spatial.std(dim=(1, 2))  # (B, C)
        return std <= tol
    raise ValueError("Unsupported x rank; expected 3D or 4D input.")


def _apply_noise_state_only(
    x: torch.Tensor,
    eps: float,
    *,
    perturb_params: bool = False,
    param_tol: float = 1e-12,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Add Gaussian noise to x.
    By default only perturbs "state channels", not parameter channels (constant-on-grid).
    """
    if eps == 0.0:
        return x

    noise = torch.randn_like(x, generator=generator) * eps

    if perturb_params:
        return x + noise

    # zero out noise on parameter channels per-sample
    mask = _infer_param_channel_mask(x, tol=param_tol)  # (B,C)
    if x.ndim == 3:
        # broadcast (B,1,C)
        mask_bc = mask[:, None, :].to(noise.dtype)
        noise = noise * (1.0 - mask_bc)
    else:
        # x.ndim == 4 broadcast (B,1,1,C)
        mask_bc = mask[:, None, None, :].to(noise.dtype)
        noise = noise * (1.0 - mask_bc)

    return x + noise


def _extract_scalar_l2(eval_out: Dict[str, Any]) -> float:
    if "static_l2" in eval_out:
        return float(eval_out["static_l2"])
    if "one_step_l2" in eval_out:
        return float(eval_out["one_step_l2"])
    raise KeyError("Expected 'static_l2' or 'one_step_l2' in evaluation output.")


# ============================================================
# C) Resolution shift
# ============================================================

def resolution_shift(
    model,
    pde_name: str,
    train_n: int,
    test_ns: List[int],
    nt: Optional[int],
    *,
    batch_size: int = 4,
    n_samples: int = 32,
    device: Optional[torch.device] = None,
    seed: int = 0,
    spectral_bins: int = 16,
):
    """
    Evaluate performance as spatial resolution changes.

    Returns:
        {
          "train_n": int,
          "resolutions": [int...],
          "errors": [eval_dict...],   # each includes spectral/residual when supported
        }
    """
    if device is None:
        device = _default_device_from_model(model)

    results = {"train_n": train_n, "resolutions": [], "errors": []}

    for n in test_ns:
        loader = make_dataloader(
            pde_name=pde_name,
            batch_size=batch_size,
            n_samples=n_samples,
            n=n,
            nt=nt,
            device=device,
            shuffle=False,
            seed=seed,
        )

        out = evaluate(
            model,
            loader,
            rollout_steps=nt if is_time_dependent(pde_name) else None,
            spectral_bins=spectral_bins,
        )

        results["resolutions"].append(int(n))
        results["errors"].append(out)

    return results


# ============================================================
# A) Parameter / coefficient shifts
# ============================================================

def parameter_shift(
    model,
    pde_name: str,
    param_name: str,
    param_values: List[Any],
    n: int,
    nt: Optional[int],
    *,
    batch_size: int = 4,
    n_samples: int = 32,
    device: Optional[torch.device] = None,
    seed: int = 0,
    spectral_bins: int = 16,
):
    """
    Robustness to parameter shifts. Works for numeric OR categorical values.

    param_name must be accepted by pdes.sample_<pde> via make_dataloader(**kwargs).

    Returns:
        {
          "param_name": str,
          "param_values": [...],
          "errors": [eval_dict...]
        }
    """
    if device is None:
        device = _default_device_from_model(model)

    results = {"param_name": param_name, "param_values": [], "errors": []}

    for val in param_values:
        loader = make_dataloader(
            pde_name=pde_name,
            batch_size=batch_size,
            n_samples=n_samples,
            n=n,
            nt=nt,
            device=device,
            shuffle=False,
            seed=seed,
            **{param_name: val},
        )

        out = evaluate(
            model,
            loader,
            rollout_steps=nt if is_time_dependent(pde_name) else None,
            spectral_bins=spectral_bins,
        )

        results["param_values"].append(val)
        results["errors"].append(out)

    return results


def coefficient_shift_poisson_suite(
    model,
    *,
    n: int = 128,
    device: Optional[torch.device] = None,
    seed: int = 0,
):
    """
    Convenience suite for Poisson coefficient/forcing shifts promised in the proposal.
    Runs shifts in: a_scale, f_scale, a_freq, f_freq, rough.
    """
    if device is None:
        device = _default_device_from_model(model)

    suite = {}
    suite["a_scale"] = parameter_shift(
        model, "poisson", "a_scale",
        param_values=[0.1, 0.5, 1.0, 2.0, 4.0],
        n=n, nt=None, device=device, seed=seed
    )
    suite["f_scale"] = parameter_shift(
        model, "poisson", "f_scale",
        param_values=[0.1, 0.5, 1.0, 2.0, 4.0],
        n=n, nt=None, device=device, seed=seed
    )
    suite["a_freq"] = parameter_shift(
        model, "poisson", "a_freq",
        param_values=[1, 2, 3, 4, 5, 6],
        n=n, nt=None, device=device, seed=seed
    )
    suite["f_freq"] = parameter_shift(
        model, "poisson", "f_freq",
        param_values=[1, 2, 3, 4, 5, 6],
        n=n, nt=None, device=device, seed=seed
    )
    suite["rough"] = parameter_shift(
        model, "poisson", "rough",
        param_values=[False, True],
        n=n, nt=None, device=device, seed=seed
    )
    return suite


# ============================================================
# D) Long-horizon rollout stability
# ============================================================

def rollout_horizon_shift(
    model,
    pde_name: str,
    horizons: List[int],
    n: int,
    nt: int,
    *,
    batch_size: int = 2,
    n_samples: int = 16,
    device: Optional[torch.device] = None,
    seed: int = 0,
    spectral_bins: int = 16,
):
    """
    Evaluate error growth as rollout horizon increases.

    Returns:
      {
        "horizons": [...],
        "evals": [eval_dict...],         # includes rollout_l2 curve + summaries
        "error_curves": [np.ndarray...], # extracted rollout_l2 curves (for convenience)
        "growth_rates": [...],
        "amplifications": [...],
      }
    """
    if not is_time_dependent(pde_name):
        raise ValueError("rollout_horizon_shift applies only to time-dependent PDEs.")

    if device is None:
        device = _default_device_from_model(model)

    loader = make_dataloader(
        pde_name=pde_name,
        batch_size=batch_size,
        n_samples=n_samples,
        n=n,
        nt=nt,
        device=device,
        shuffle=False,
        seed=seed,
    )

    results = {
        "horizons": [],
        "evals": [],
        "error_curves": [],
        "growth_rates": [],
        "amplifications": [],
    }

    for H in horizons:
        out = evaluate(
            model,
            loader,
            rollout_steps=int(H),
            spectral_bins=spectral_bins,
        )

        curve = out.get("rollout_l2", None)
        if curve is None:
            raise RuntimeError("evaluate(..., rollout_steps=H) did not return 'rollout_l2'.")

        curve = np.asarray(curve, dtype=np.float64)
        results["horizons"].append(int(H))
        results["evals"].append(out)
        results["error_curves"].append(curve)
        results["growth_rates"].append(float(out.get("rollout_growth_rate", error_growth_rate(curve))))
        results["amplifications"].append(float(out.get("rollout_amplification", amplification_factor(curve))))

    return results


# ============================================================
# E) Perturbation sensitivity
# ============================================================

def perturbation_shift(
    model,
    pde_name: str,
    epsilons: List[float],
    n: int,
    nt: Optional[int],
    *,
    batch_size: int = 4,
    n_samples: int = 32,
    device: Optional[torch.device] = None,
    seed: int = 0,
    spectral_bins: int = 16,
    perturb_params: bool = False,
    param_tol: float = 1e-12,
):
    """
    Inject perturbations into inputs and measure sensitivity.

    Important (proposal-aligned default):
      - By default, perturb only "state channels" and DO NOT perturb constant parameter channels.

    Returns:
      {
        "epsilons": [...],
        "errors": [eval_dict...],
        "base_error": float,
        "amplification": [...],  # (error(eps)-error(0))/eps ; nan at eps=0
      }
    """
    if device is None:
        device = _default_device_from_model(model)

    base_loader = make_dataloader(
        pde_name=pde_name,
        batch_size=batch_size,
        n_samples=n_samples,
        n=n,
        nt=nt,
        device=device,
        shuffle=False,
        seed=seed,
    )

    # baseline
    base_eval = evaluate(
        model,
        base_loader,
        rollout_steps=nt if is_time_dependent(pde_name) else None,
        spectral_bins=spectral_bins,
    )
    base_err = _extract_scalar_l2(base_eval)

    results = {
        "epsilons": [],
        "errors": [],
        "base_error": base_err,
        "amplification": [],
        "perturb_params": bool(perturb_params),
    }

    # make deterministic noise per eps (optional)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 12345)

    for eps in epsilons:
        # precompute perturbed batches
        perturbed_batches = []
        for x, y in base_loader:
            x = x.to(device)
            y = y.to(device)
            x_pert = _apply_noise_state_only(
                x, float(eps),
                perturb_params=perturb_params,
                param_tol=param_tol,
                generator=gen,
            )
            perturbed_batches.append((x_pert, y))

        class _PerturbedLoader:
            def __init__(self, batches, dataset):
                self._batches = batches
                self.dataset = dataset
            def __iter__(self):
                return iter(self._batches)

        loader = _PerturbedLoader(perturbed_batches, base_loader.dataset)

        out = evaluate(
            model,
            loader,
            rollout_steps=nt if is_time_dependent(pde_name) else None,
            spectral_bins=spectral_bins,
        )

        err = _extract_scalar_l2(out)
        results["epsilons"].append(float(eps))
        results["errors"].append(out)

        if float(eps) == 0.0:
            results["amplification"].append(float("nan"))
        else:
            results["amplification"].append(perturbation_amplification(base_err, err, float(eps)))

    return results


# ============================================================
# B) Boundary / terminal shifts
# ============================================================

def boundary_or_payoff_shift(
    model,
    pde_name: str,
    shift_values: List[Any],
    n: int,
    *,
    batch_size: int = 4,
    n_samples: int = 32,
    device: Optional[torch.device] = None,
    seed: int = 0,
    spectral_bins: int = 16,
    # extended controls (proposal-complete for Black–Scholes BC stress)
    bs_shift_kind: str = "payoff_type",   # "payoff_type" or "bc_type"
    poisson_shift_kind: str = "bc0",      # "bc0" or "bc1"
):
    """
    Boundary / terminal condition stress tests.

    Poisson:
      - shift bc0 or bc1 (Dirichlet endpoints)

    Black–Scholes:
      - shift payoff_type OR bc_type (terminal or boundary regime)
        bc_type is supported by pdes.sample_black_scholes.

    Returns:
      {
        "shift_kind": str,
        "shift_values": [...],
        "errors": [eval_dict...]
      }
    """
    if device is None:
        device = _default_device_from_model(model)

    if is_time_dependent(pde_name):
        raise ValueError("boundary_or_payoff_shift applies only to static PDEs.")

    p = pde_name.lower().strip()
    results = {"shift_kind": None, "shift_values": [], "errors": []}

    for val in shift_values:
        if p == "poisson":
            if poisson_shift_kind not in ("bc0", "bc1"):
                raise ValueError("poisson_shift_kind must be 'bc0' or 'bc1'.")
            kwargs = {poisson_shift_kind: val}
            results["shift_kind"] = poisson_shift_kind

        elif p == "black_scholes":
            if bs_shift_kind not in ("payoff_type", "bc_type"):
                raise ValueError("bs_shift_kind must be 'payoff_type' or 'bc_type'.")
            kwargs = {bs_shift_kind: val}
            results["shift_kind"] = bs_shift_kind

        else:
            raise ValueError(f"No boundary/terminal shift defined for {pde_name}.")

        loader = make_dataloader(
            pde_name=pde_name,
            batch_size=batch_size,
            n_samples=n_samples,
            n=n,
            nt=None,
            device=device,
            shuffle=False,
            seed=seed,
            **kwargs,
        )

        out = evaluate(model, loader, spectral_bins=spectral_bins)

        results["shift_values"].append(val)
        results["errors"].append(out)

    return results


def black_scholes_boundary_suite(
    model,
    *,
    n: int = 256,
    device: Optional[torch.device] = None,
    seed: int = 0,
):
    """
    Convenience suite for Black–Scholes terminal + boundary shifts promised in the proposal:
      - payoff_type family
      - bc_type family
    """
    if device is None:
        device = _default_device_from_model(model)

    suite = {}
    suite["payoff_type"] = boundary_or_payoff_shift(
        model, "black_scholes",
        shift_values=["call", "put", "digital_call", "smooth_call"],
        n=n, device=device, seed=seed, bs_shift_kind="payoff_type"
    )
    suite["bc_type"] = boundary_or_payoff_shift(
        model, "black_scholes",
        shift_values=["standard", "discounted_payoff", "dirichlet_zero", "linear_extrapolation"],
        n=n, device=device, seed=seed, bs_shift_kind="bc_type"
    )
    return suite


# ============================================================
# Generic "structural shift grid" helper (optional but useful)
# ============================================================

def kwarg_grid_shift(
    model,
    pde_name: str,
    n: int,
    nt: Optional[int],
    grid: Dict[str, List[Any]],
    *,
    batch_size: int = 4,
    n_samples: int = 32,
    device: Optional[torch.device] = None,
    seed: int = 0,
    spectral_bins: int = 16,
):
    """
    Evaluate over a cartesian product of kwargs.

    Example:
      grid={"forcing_amp":[0.0,1.0], "forcing_k":[1,4]}

    Returns:
      {
        "grid": grid,
        "cases": [ { "kwargs": {...}, "eval": eval_dict }, ... ]
      }
    """
    if device is None:
        device = _default_device_from_model(model)

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    cases = []
    for combo in _product(values):
        kwargs = dict(zip(keys, combo))
        loader = make_dataloader(
            pde_name=pde_name,
            batch_size=batch_size,
            n_samples=n_samples,
            n=n,
            nt=nt,
            device=device,
            shuffle=False,
            seed=seed,
            **kwargs,
        )
        out = evaluate(
            model,
            loader,
            rollout_steps=nt if is_time_dependent(pde_name) else None,
            spectral_bins=spectral_bins,
        )
        cases.append({"kwargs": kwargs, "eval": out})

    return {"grid": grid, "cases": cases}


def _product(lists: List[List[Any]]):
    # small local cartesian product helper (no itertools dependency)
    if not lists:
        yield ()
        return
    first, rest = lists[0], lists[1:]
    for x in first:
        for xs in _product(rest):
            yield (x,) + xs


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    from train import get_device
    from operators import make_fno

    device = get_device()

    # Tiny Poisson model (untrained) just to test stress function wiring
    loader = make_dataloader(
        pde_name="poisson",
        batch_size=4,
        n_samples=8,
        n=128,
        nt=None,
        device=device,
        shuffle=False,
        seed=0,
    )

    model = make_fno(
        dim=1,
        in_channels=3,
        out_channels=1,
        width=32,
        depth=3,
        modes1=16,
        use_coords=True,
    ).to(device)

    res = resolution_shift(
        model,
        pde_name="poisson",
        train_n=128,
        test_ns=[64, 128, 256],
        nt=None,
        device=device,
        seed=0,
    )
    print("Resolution shift keys:", res.keys())

    pert = perturbation_shift(
        model,
        pde_name="poisson",
        epsilons=[0.0, 1e-3, 1e-2],
        n=128,
        nt=None,
        device=device,
        seed=0,
        perturb_params=False,
    )
    print("Perturbation shift keys:", pert.keys())
