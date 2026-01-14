# metrics.py
# Metric utilities for neural operator failure atlas
#
# Responsibilities:
#   - relative L2 (array-level)
#   - spectral error diagnostics
#   - rollout growth / amplification metrics
#
# No model calls, no plotting, no data loading.

from __future__ import annotations

import numpy as np
import torch


# ============================================================
# Basic metrics
# ============================================================

def relative_l2_np(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-12) -> float:
    """
    Relative L2 error (NumPy).
    """
    num = np.linalg.norm(pred - truth)
    den = np.linalg.norm(truth) + eps
    return float(num / den)


def relative_l2_torch(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Relative L2 error (Torch).
    """
    num = torch.norm(pred - truth)
    den = torch.norm(truth) + eps
    return (num / den).item()


# ============================================================
# Spectral diagnostics
# ============================================================

def spectral_error_1d(
    pred: np.ndarray,
    truth: np.ndarray,
    n_bins: int = 16,
    eps: float = 1e-12,
):
    """
    Compute frequency-binned relative error for 1D signals.

    Args:
        pred, truth: arrays of shape (N,)
        n_bins: number of frequency bins

    Returns:
        freqs: bin centers
        errors: relative L2 error per bin
    """
    N = pred.shape[0]

    pred_hat = np.fft.rfft(pred)
    truth_hat = np.fft.rfft(truth)

    freqs = np.fft.rfftfreq(N)
    bins = np.linspace(0, freqs.max(), n_bins + 1)

    errors = []
    centers = []

    for i in range(n_bins):
        mask = (freqs >= bins[i]) & (freqs < bins[i + 1])
        if not np.any(mask):
            continue

        num = np.linalg.norm(pred_hat[mask] - truth_hat[mask])
        den = np.linalg.norm(truth_hat[mask]) + eps

        errors.append(num / den)
        centers.append(0.5 * (bins[i] + bins[i + 1]))

    return np.array(centers), np.array(errors)


def spectral_error_2d(
    pred: np.ndarray,
    truth: np.ndarray,
    n_bins: int = 16,
    eps: float = 1e-12,
):
    """
    Radially averaged spectral error for 2D fields.

    Args:
        pred, truth: arrays of shape (N, N)

    Returns:
        radii: radial frequency bins
        errors: relative error per bin
    """
    N = pred.shape[0]

    pred_hat = np.fft.fftshift(np.fft.fft2(pred))
    truth_hat = np.fft.fftshift(np.fft.fft2(truth))

    k = np.fft.fftshift(np.fft.fftfreq(N))
    kx, ky = np.meshgrid(k, k, indexing="ij")
    r = np.sqrt(kx**2 + ky**2)

    r_max = r.max()
    bins = np.linspace(0.0, r_max, n_bins + 1)

    errors = []
    centers = []

    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if not np.any(mask):
            continue

        num = np.linalg.norm(pred_hat[mask] - truth_hat[mask])
        den = np.linalg.norm(truth_hat[mask]) + eps

        errors.append(num / den)
        centers.append(0.5 * (bins[i] + bins[i + 1]))

    return np.array(centers), np.array(errors)


# ============================================================
# Rollout / stability diagnostics
# ============================================================

def rollout_error_curve(pred_traj: np.ndarray, truth_traj: np.ndarray):
    """
    Compute relative L2 error at each rollout step.

    Args:
        pred_traj: (T, ...)
        truth_traj: (T, ...)

    Returns:
        errors: (T,)
    """
    T = pred_traj.shape[0]
    errors = np.zeros(T, dtype=np.float64)

    for t in range(T):
        errors[t] = relative_l2_np(pred_traj[t], truth_traj[t])

    return errors


def error_growth_rate(error_curve: np.ndarray, eps: float = 1e-12):
    """
    Estimate exponential growth rate of rollout error.

    Fits:
        log(error(t)) ~ a * t + b

    Returns:
        a: growth rate
    """
    t = np.arange(len(error_curve))
    y = np.log(error_curve + eps)

    a, b = np.polyfit(t, y, 1)
    return float(a)


def amplification_factor(error_curve: np.ndarray):
    """
    Simple amplification metric: max(error) / min(error).
    """
    return float(np.max(error_curve) / np.min(error_curve))


# ============================================================
# Perturbation sensitivity
# ============================================================

def perturbation_amplification(
    base_error: float,
    perturbed_error: float,
    eps: float,
):
    """
    Compute normalized amplification under input perturbation.

    Returns:
        amplification = (perturbed - base) / eps
    """
    return float((perturbed_error - base_error) / eps)


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    # 1D spectral test
    x = np.linspace(0, 2 * np.pi, 256)
    truth = np.sin(x)
    pred = np.sin(x) + 0.1 * np.sin(8 * x)

    freqs, errs = spectral_error_1d(pred, truth)
    print("1D spectral bins:", freqs.shape, errs.shape)

    # rollout growth test
    curve = np.exp(0.2 * np.arange(20))
    print("Growth rate:", error_growth_rate(curve))
