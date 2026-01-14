# eval.py
# Evaluation utilities for neural operator failure atlas
#
# Responsibilities (proposal-aligned):
#   - Relative L2 error (static + one-step)
#   - PDE residual error (where computable from available channels)
#   - Spectral error decomposition (1D + 2D)
#   - Multi-step rollout + rollout error growth curves
#   - Rollout growth-rate / amplification summaries
#
# Notes:
#   - Residuals are implemented for Poisson, KS, Schrödinger (NLS), and Navier–Stokes
#     using normalized time step dt = 1/nt when true dt is unavailable at evaluation time.
#   - Black–Scholes residual is not computed because the dataset provides only V(S, t=0)
#     (no time surface), which is insufficient for a PDE residual without additional data.

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from data import is_time_dependent

from metrics import (
    spectral_error_1d,
    spectral_error_2d,
    error_growth_rate,
    amplification_factor,
)

# ============================================================
# Basic utilities
# ============================================================

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def _relative_l2(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.norm(pred - truth)
    den = torch.norm(truth) + eps
    return (num / den).item()

def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if len(xs) else float("nan")

# ============================================================
# Spectral diagnostics (batch-averaged)
# ============================================================

def _spectral_error_batch_1d(
    pred: torch.Tensor,  # (B, N, C)
    truth: torch.Tensor, # (B, N, C)
    n_bins: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Compute frequency-binned spectral error for 1D outputs, averaged over batch and channels.

    Returns:
        {"freqs": (K,), "errors": (K,)}
    """
    pred_np = _to_np(pred)
    truth_np = _to_np(truth)

    freqs_ref = None
    errs_accum = []

    B, N, C = pred_np.shape
    for b in range(B):
        for c in range(C):
            freqs, errs = spectral_error_1d(pred_np[b, :, c], truth_np[b, :, c], n_bins=n_bins)
            if freqs_ref is None:
                freqs_ref = freqs
            else:
                # allow occasional bin-count mismatch if empty bins got dropped; align by min length
                m = min(len(freqs_ref), len(freqs))
                freqs_ref = freqs_ref[:m]
                errs = errs[:m]
            errs_accum.append(errs[: len(freqs_ref)])

    if freqs_ref is None:
        return {"freqs": np.array([]), "errors": np.array([])}

    errs_mat = np.stack(errs_accum, axis=0)
    return {"freqs": freqs_ref, "errors": np.mean(errs_mat, axis=0)}

def _spectral_error_batch_2d(
    pred: torch.Tensor,  # (B, N, N, C)
    truth: torch.Tensor, # (B, N, N, C)
    n_bins: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Radially averaged spectral error for 2D outputs, averaged over batch and channels.

    Returns:
        {"radii": (K,), "errors": (K,)}
    """
    pred_np = _to_np(pred)
    truth_np = _to_np(truth)

    radii_ref = None
    errs_accum = []

    B, N, _, C = pred_np.shape
    for b in range(B):
        for c in range(C):
            radii, errs = spectral_error_2d(pred_np[b, :, :, c], truth_np[b, :, :, c], n_bins=n_bins)
            if radii_ref is None:
                radii_ref = radii
            else:
                m = min(len(radii_ref), len(radii))
                radii_ref = radii_ref[:m]
                errs = errs[:m]
            errs_accum.append(errs[: len(radii_ref)])

    if radii_ref is None:
        return {"radii": np.array([]), "errors": np.array([])}

    errs_mat = np.stack(errs_accum, axis=0)
    return {"radii": radii_ref, "errors": np.mean(errs_mat, axis=0)}

def spectral_error_decomposition(
    pred: torch.Tensor,
    truth: torch.Tensor,
    n_bins: int = 16,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Dispatch spectral error decomposition based on tensor rank.

    Supported:
        1D outputs: (B, N, C)
        2D outputs: (B, N, N, C)
    """
    if pred.ndim == 3:
        return _spectral_error_batch_1d(pred, truth, n_bins=n_bins)
    if pred.ndim == 4:
        return _spectral_error_batch_2d(pred, truth, n_bins=n_bins)
    return None

# ============================================================
# PDE residuals (batch-averaged)
# ============================================================

def _residual_poisson_1d(
    u: torch.Tensor,  # (B, N, 1)
    x_inp: torch.Tensor,  # (B, N, 3) channels: a, f, bc_mask (endpoints store bc values)
    domain: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-12,
) -> float:
    """
    Residual for 1D variable-coefficient Poisson:
        -d/dx( a(x) u_x ) = f(x)  (Dirichlet BC encoded in bc_mask endpoints)

    Uses centered FD with midpoint a_{i+1/2}.
    Returns relative L2 norm of interior residual vs f interior.
    """
    a = x_inp[..., 0]
    f = x_inp[..., 1]

    B, N = a.shape
    a_mid = 0.5 * (a[:, :-1] + a[:, 1:])  # (B, N-1)

    aL = a_mid[:, :-1]   # (B, N-2) = a_{i-1/2}
    aR = a_mid[:, 1:]    # (B, N-2) = a_{i+1/2}

    u0 = u[..., 0]  # (B, N)
    u_im1 = u0[:, :-2]
    u_i   = u0[:, 1:-1]
    u_ip1 = u0[:, 2:]

    aR_duR = aR * (u_ip1 - u_i)
    aL_duL = aL * (u_i - u_im1)

    a_ux_x = (aR_duR - aL_duL)  # (B, N-2)

    a_dom0, a_dom1 = domain
    h = (a_dom1 - a_dom0) / (N - 1)
    lhs = -(a_ux_x) / (h * h)
    rhs = f[:, 1:-1]

    res = lhs - rhs

    num = torch.norm(res)
    den = torch.norm(rhs) + eps
    return (num / den).item()

def _rfft_k_1d(n: int, L: torch.Tensor, device, dtype) -> torch.Tensor:
    """
    Angular frequencies k for periodic domain length L (can be scalar tensor).
    Returns shape (n,) on device.
    """
    k = 2.0 * np.pi * torch.fft.fftfreq(n, d=(L / n).item(), device=device)
    return k.to(dtype=dtype)

def _spectral_derivs_1d(u: torch.Tensor, k: torch.Tensor):
    """
    Compute u_x, u_xx, u_xxxx for periodic 1D signals.
    u: (B, N)
    k: (N,)
    """
    u_hat = torch.fft.fft(u, dim=-1)
    ik = 1j * k
    u_x = torch.fft.ifft(ik * u_hat, dim=-1).real
    u_xx = torch.fft.ifft((ik**2) * u_hat, dim=-1).real
    u_xxxx = torch.fft.ifft((ik**4) * u_hat, dim=-1).real
    return u_x, u_xx, u_xxxx

def _residual_ks_1d(
    u1: torch.Tensor,  # (B, N, 1)
    x_inp: torch.Tensor,  # (B, N, 2): u0, L
    nt: int,
    eps: float = 1e-12,
) -> float:
    """
    Discrete-time residual proxy for KS:
        u_t + u u_x + u_xx + u_xxxx = 0
    Using:
        (u1 - u0)/dt + u0*u0_x + u0_xx + u0_xxxx

    dt approximated as 1/nt.
    """
    u0 = x_inp[..., 0]          # (B, N)
    L = x_inp[..., 1].mean()    # scalar-ish
    dt = 1.0 / max(int(nt), 1)

    device = u0.device
    k = _rfft_k_1d(u0.shape[1], L, device, u0.dtype)

    u0_x, u0_xx, u0_xxxx = _spectral_derivs_1d(u0, k)

    u1p = u1[..., 0]
    res = (u1p - u0) / dt + (u0 * u0_x) + u0_xx + u0_xxxx

    num = torch.norm(res)
    den = torch.norm((u1p - u0) / dt) + eps
    return (num / den).item()

def _residual_schrodinger_1d(
    u1: torch.Tensor,  # (B, N, 2) predicted next (Re,Im)
    x_inp: torch.Tensor,  # (B, N, 3): u0_re, u0_im, kappa
    nt: int,
    eps: float = 1e-12,
) -> float:
    """
    Discrete-time residual proxy for NLS:
        i u_t + u_xx + κ|u|^2 u = 0

    Use u0 for spatial terms; dt approximated as 1/nt.
    """
    u0 = torch.complex(x_inp[..., 0], x_inp[..., 1])  # (B, N)
    kappa = x_inp[..., 2].mean()                      # scalar-ish
    dt = 1.0 / max(int(nt), 1)

    device = u0.device
    N = u0.shape[1]
    # Assume periodic domain length 1 for residual proxy
    L = torch.tensor(1.0, device=device, dtype=u0.real.dtype)
    k = _rfft_k_1d(N, L, device, u0.real.dtype)

    u0_hat = torch.fft.fft(u0, dim=-1)
    u0_xx = torch.fft.ifft((1j * k) ** 2 * u0_hat, dim=-1)

    u1c = torch.complex(u1[..., 0], u1[..., 1])

    res = 1j * (u1c - u0) / dt + u0_xx + (kappa * (torch.abs(u0) ** 2) * u0)

    num = torch.norm(res)
    den = torch.norm(1j * (u1c - u0) / dt) + eps
    return (num / den).item()

def _rfft_k_2d(n: int, L: float, device, dtype):
    """
    Angular frequencies for 2D periodic box length L in each dimension.
    Returns (kx, ky, k2) each shape (n, n)
    """
    k1 = 2.0 * np.pi * torch.fft.fftfreq(n, d=L / n, device=device).to(dtype=dtype)
    kx, ky = torch.meshgrid(k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky
    k2 = k2.clone()
    k2[0, 0] = 1.0
    return kx, ky, k2

def _residual_navier_stokes_vorticity_2d(
    w1: torch.Tensor,     # (B, N, N, 1)
    x_inp: torch.Tensor,  # (B, N, N, 3): w0, s, nu
    nt: int,
    L: float = 2.0 * np.pi,
    eps: float = 1e-12,
) -> float:
    """
    Discrete-time residual proxy for 2D NS (vorticity form):
        ω_t + u·∇ω = νΔω + s

    Using ω0 in the nonlinear term and Laplacian term.
    dt approximated as 1/nt.

    Returns relative residual norm vs ||(w1-w0)/dt||.
    """
    w0 = x_inp[..., 0]  # (B, N, N)
    s  = x_inp[..., 1]
    nu = x_inp[..., 2].mean()  # scalar-ish
    w1p = w1[..., 0]

    B, N, _ = w0.shape
    dt = 1.0 / max(int(nt), 1)

    device = w0.device
    kx, ky, k2 = _rfft_k_2d(N, L, device, w0.dtype)

    w0_hat = torch.fft.fft2(w0, dim=(-2, -1))
    psi_hat = -w0_hat / k2
    psi_hat[..., 0, 0] = 0.0 + 0.0j

    u_hat_x = 1j * ky * psi_hat
    u_hat_y = -1j * kx * psi_hat
    u_x = torch.fft.ifft2(u_hat_x, dim=(-2, -1)).real
    u_y = torch.fft.ifft2(u_hat_y, dim=(-2, -1)).real

    w0_x = torch.fft.ifft2(1j * kx * w0_hat, dim=(-2, -1)).real
    w0_y = torch.fft.ifft2(1j * ky * w0_hat, dim=(-2, -1)).real

    adv = u_x * w0_x + u_y * w0_y
    lap = torch.fft.ifft2(-k2 * w0_hat, dim=(-2, -1)).real  # Δw0

    res = (w1p - w0) / dt + adv - (nu * lap) - s

    num = torch.norm(res)
    den = torch.norm((w1p - w0) / dt) + eps
    return (num / den).item()

def pde_residual_error(
    pde_name: str,
    pred: torch.Tensor,
    x_inp: torch.Tensor,
    dataloader,
) -> Optional[float]:
    """
    Compute PDE residual error (scalar) when supported.
    """
    p = pde_name.lower()
    nt = getattr(dataloader.dataset, "nt", None)

    if p == "poisson":
        if pred.ndim != 3:
            return None
        return _residual_poisson_1d(pred, x_inp)

    if p == "kuramoto_sivashinsky":
        if nt is None:
            return None
        return _residual_ks_1d(pred, x_inp, nt=nt)

    if p == "schrodinger":
        if nt is None:
            return None
        return _residual_schrodinger_1d(pred, x_inp, nt=nt)

    if p == "navier_stokes":
        if nt is None:
            return None
        return _residual_navier_stokes_vorticity_2d(pred, x_inp, nt=nt)

    # black_scholes: not enough information for residual without time surface
    return None

# ============================================================
# Static PDE evaluation
# ============================================================

@torch.no_grad()
def eval_static(
    model,
    dataloader,
    *,
    spectral_bins: int = 16,
    compute_residual: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate static PDEs.
    """
    model.eval()
    device = next(model.parameters()).device

    l2s = []
    residuals = []
    spec_curves = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        l2s.append(_relative_l2(pred, y))

        if compute_residual:
            r = pde_residual_error(dataloader.dataset.pde_name, pred, x, dataloader)
            residuals.append(r)

        spec = spectral_error_decomposition(pred, y, n_bins=spectral_bins)
        if spec is not None and len(spec.get("errors", [])) > 0:
            spec_curves.append(spec)

    out: Dict[str, Any] = {
        "static_l2": _safe_mean(l2s),
    }

    if any(r is not None for r in residuals):
        out["residual_l2"] = _safe_mean(residuals)

    if len(spec_curves):
        if "freqs" in spec_curves[0]:
            freqs = spec_curves[0]["freqs"]
            errs = np.mean(np.stack([s["errors"][: len(freqs)] for s in spec_curves], axis=0), axis=0)
            out["spectral"] = {"freqs": freqs, "errors": errs}
        else:
            radii = spec_curves[0]["radii"]
            errs = np.mean(np.stack([s["errors"][: len(radii)] for s in spec_curves], axis=0), axis=0)
            out["spectral"] = {"radii": radii, "errors": errs}

    return out

# ============================================================
# Time-dependent: one-step evaluation
# ============================================================

@torch.no_grad()
def eval_one_step(
    model,
    dataloader,
    *,
    spectral_bins: int = 16,
    compute_residual: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate one-step prediction for time-dependent PDEs.
    """
    model.eval()
    device = next(model.parameters()).device

    l2s = []
    residuals = []
    spec_curves = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        y_target = y[:, 1]
        pred = model(x)

        l2s.append(_relative_l2(pred, y_target))

        if compute_residual:
            r = pde_residual_error(dataloader.dataset.pde_name, pred, x, dataloader)
            residuals.append(r)

        spec = spectral_error_decomposition(pred, y_target, n_bins=spectral_bins)
        if spec is not None and len(spec.get("errors", [])) > 0:
            spec_curves.append(spec)

    out: Dict[str, Any] = {
        "one_step_l2": _safe_mean(l2s),
    }

    if any(r is not None for r in residuals):
        out["residual_l2"] = _safe_mean(residuals)

    if len(spec_curves):
        if "freqs" in spec_curves[0]:
            freqs = spec_curves[0]["freqs"]
            errs = np.mean(np.stack([s["errors"][: len(freqs)] for s in spec_curves], axis=0), axis=0)
            out["spectral"] = {"freqs": freqs, "errors": errs}
        else:
            radii = spec_curves[0]["radii"]
            errs = np.mean(np.stack([s["errors"][: len(radii)] for s in spec_curves], axis=0), axis=0)
            out["spectral"] = {"radii": radii, "errors": errs}

    return out

# ============================================================
# Rollout utilities
# ============================================================

def autoregressive_update(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Autoregressive update that preserves parameter channels.
    """
    c_out = y.shape[-1]
    params = x[..., c_out:]
    return torch.cat([y, params], dim=-1)

@torch.no_grad()
def rollout(model, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Autoregressive rollout.
    """
    model.eval()
    device = next(model.parameters()).device

    x = x0.to(device)
    preds = []

    for _ in range(steps + 1):
        y = model(x)
        preds.append(y)
        x = autoregressive_update(x, y)

    return torch.stack(preds, dim=0)

@torch.no_grad()
def rollout_error(model, dataloader, steps: int) -> np.ndarray:
    """
    Compute rollout error growth curve (relative L2 at each step).
    """
    model.eval()
    device = next(model.parameters()).device

    error_accum = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        T = min(steps, y.shape[1] - 1)

        preds = rollout(model, x, T)
        truth = y[:, : T + 1]

        truth = truth.permute(1, 0, *range(2, truth.ndim))

        errs = []
        for t in range(T + 1):
            errs.append(_relative_l2(preds[t], truth[t]))

        error_accum.append(np.array(errs, dtype=np.float64))

    return np.mean(np.stack(error_accum, axis=0), axis=0)

# ============================================================
# Unified evaluation entry point
# ============================================================

def evaluate(
    model,
    dataloader,
    rollout_steps: int | None = None,
    *,
    spectral_bins: int = 16,
    compute_residual: bool = True,
    summarize_rollout: bool = True,
) -> Dict[str, Any]:
    """
    Dispatch evaluation based on PDE type.
    """
    pde_name = dataloader.dataset.pde_name

    if not is_time_dependent(pde_name):
        return eval_static(
            model,
            dataloader,
            spectral_bins=spectral_bins,
            compute_residual=compute_residual,
        )

    results = eval_one_step(
        model,
        dataloader,
        spectral_bins=spectral_bins,
        compute_residual=compute_residual,
    )

    if rollout_steps is not None:
        curve = rollout_error(model, dataloader, rollout_steps)
        results["rollout_l2"] = curve

        if summarize_rollout:
            results["rollout_growth_rate"] = error_growth_rate(curve)
            results["rollout_amplification"] = amplification_factor(curve)

    return results

# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    from data import make_dataloader
    from operators import make_fno
    from train import get_device

    device = get_device()

    # --- Static test (Poisson)
    loader = make_dataloader(
        pde_name="poisson",
        batch_size=4,
        n_samples=16,
        n=128,
        nt=None,
        device=device,
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

    out = evaluate(model, loader, spectral_bins=12)
    print("Static eval (untrained): keys =", list(out.keys()))
    if "spectral" in out:
        print("  spectral bins:", out["spectral"].keys())

    # --- Time-dependent test (KS)
    loader = make_dataloader(
        pde_name="kuramoto_sivashinsky",
        batch_size=2,
        n_samples=8,
        n=128,
        nt=20,
        device=device,
    )

    model = make_fno(
        dim=1,
        in_channels=2,
        out_channels=1,
        width=32,
        depth=3,
        modes1=16,
        use_coords=True,
    ).to(device)

    out = evaluate(model, loader, rollout_steps=10, spectral_bins=12)
    print("KS eval (untrained): keys =", list(out.keys()))
    if "spectral" in out:
        print("  spectral bins:", out["spectral"].keys())
