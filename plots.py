# plots.py
# Visualization utilities for neural operator failure atlas
#
# Responsibilities:
#   - convert evaluation + stress-test outputs into publication-quality plots
#   - support multiple complementary visualizations per PDE
#   - no computation, no model calls
#
# All functions take raw arrays/dicts and produce figures.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def _finalize_plot(savepath=None):
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def plot_spectral_error(evaluation, title=None, savepath=None):
    """
    Plot spectral error curve for a single evaluation output.

    Args:
        evaluation: output dict from eval.py containing "spectral"
        title: optional plot title
        savepath: if provided, save figure to this path instead of showing
    """
    if not _has_spectral(evaluation):
        raise ValueError("No spectral data present.")

    set_plot_style()
    spec = evaluation["spectral"]

    plt.figure()
    if "freqs" in spec:
        plt.plot(spec["freqs"], spec["errors"], marker="o")
        plt.xlabel("Frequency")
    else:
        plt.plot(spec["radii"], spec["errors"], marker="o")
        plt.xlabel("Radial frequency")

    plt.ylabel("Relative spectral error")
    if title:
        plt.title(title)

    _finalize_plot(savepath)

def plot_resolution_vs_spectral_energy(results, title=None, savepath=None):
    set_plot_style()

    resolutions = results["resolutions"]
    low_err, high_err = [], []

    for e in results["errors"]:
        if not _has_spectral(e):
            continue
        spec = e["spectral"]["errors"]
        mid = len(spec) // 2
        low_err.append(np.mean(spec[:mid]))
        high_err.append(np.mean(spec[mid:]))

    plt.figure()
    plt.plot(resolutions[:len(low_err)], low_err, marker="o", label="Low-frequency error")
    plt.plot(resolutions[:len(high_err)], high_err, marker="o", label="High-frequency error")
    plt.xlabel("Spatial resolution")
    plt.ylabel("Relative spectral error")
    plt.legend()
    if title:
        plt.title(title)

    _finalize_plot(savepath)


def plot_rollout_error_curve(error_curve, title=None, savepath=None):
    set_plot_style()
    t = np.arange(len(error_curve))

    plt.figure()
    plt.plot(t, error_curve, marker="o")
    plt.xlabel("Rollout step")
    plt.ylabel("Relative $L^2$ error")
    if title:
        plt.title(title)

    _finalize_plot(savepath)



# ============================================================
# Global plotting style
# ============================================================

def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "figure.figsize": (6.5, 4.5),
    })


# ============================================================
# Basic helpers
# ============================================================

def _extract_scalar_error(e):
    if "static_l2" in e:
        return e["static_l2"]
    if "one_step_l2" in e:
        return e["one_step_l2"]
    raise ValueError("Unknown error structure.")

def _has_spectral(e):
    return "spectral" in e and e["spectral"] is not None


# ============================================================
# Resolution shift
# ============================================================

def plot_resolution_shift(results, title=None):
    set_plot_style()

    x = results["resolutions"]
    y = [_extract_scalar_error(e) for e in results["errors"]]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Spatial resolution")
    plt.ylabel("Relative $L^2$ error")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_resolution_vs_spectral_energy(results, title=None):
    """
    Compare low vs high frequency error as resolution changes.
    """
    set_plot_style()

    resolutions = results["resolutions"]
    low_err = []
    high_err = []

    for e in results["errors"]:
        if not _has_spectral(e):
            continue
        spec = e["spectral"]["errors"]
        mid = len(spec) // 2
        low_err.append(np.mean(spec[:mid]))
        high_err.append(np.mean(spec[mid:]))

    plt.figure()
    plt.plot(resolutions[:len(low_err)], low_err, marker="o", label="Low-frequency error")
    plt.plot(resolutions[:len(high_err)], high_err, marker="o", label="High-frequency error")
    plt.xlabel("Spatial resolution")
    plt.ylabel("Relative spectral error")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Spectral diagnostics
# ============================================================

def plot_spectral_error(evaluation, title=None, savepath=None):
    """
    Plot Fourier-domain distribution of prediction error energy.

    The plotted quantity is the normalized spectral error energy,
    i.e. the fraction of total error energy at each frequency.
    """
    if not _has_spectral(evaluation):
        raise ValueError("No spectral data present.")

    set_plot_style()
    spec = evaluation["spectral"]

    # Treat stored values as per-frequency error magnitudes
    err = np.asarray(spec["errors"], dtype=np.float64)

    # Convert to energy and normalize
    err_energy = err**2
    total = err_energy.sum()
    if total > 0:
        err_energy /= total

    plt.figure()
    if "freqs" in spec:
        plt.plot(spec["freqs"], err_energy, marker="o")
        plt.xlabel("Frequency")
    else:
        plt.plot(spec["radii"], err_energy, marker="o")
        plt.xlabel("Radial frequency")

    plt.ylabel("Fraction of total error energy")
    if title:
        plt.title(title)

    _finalize_plot(savepath)




def plot_multiple_spectral_curves(curves, labels, title=None):
    """
    Compare spectral bias across PDEs or parameter regimes.
    """
    set_plot_style()
    plt.figure()

    for spec, label in zip(curves, labels):
        if "freqs" in spec:
            x = spec["freqs"]
        else:
            x = spec["radii"]
        plt.plot(x, spec["errors"], marker="o", label=label)

    plt.xlabel("Frequency")
    plt.ylabel("Relative spectral error")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Rollout diagnostics
# ============================================================

def plot_rollout_error_curve(error_curve, title=None):
    set_plot_style()
    t = np.arange(len(error_curve))

    plt.figure()
    plt.plot(t, error_curve, marker="o")
    plt.xlabel("Rollout step")
    plt.ylabel("Relative $L^2$ error")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_multiple_rollout_curves(curves, labels, title=None):
    set_plot_style()
    plt.figure()

    for curve, label in zip(curves, labels):
        plt.plot(np.arange(len(curve)), curve, label=label)

    plt.xlabel("Rollout step")
    plt.ylabel("Relative $L^2$ error")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_rollout_growth_rates(rates, labels, title=None):
    """
    Bar plot of exponential rollout growth rates.
    """
    set_plot_style()
    plt.figure()

    plt.bar(labels, rates)
    plt.ylabel("Exponential growth rate")
    plt.xticks(rotation=30)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Perturbation sensitivity
# ============================================================

def plot_perturbation_shift(results, title=None):
    set_plot_style()

    eps = results["epsilons"]
    y = [_extract_scalar_error(e) for e in results["errors"]]

    plt.figure()
    plt.plot(eps, y, marker="o")
    plt.xlabel("Perturbation magnitude $\\varepsilon$")
    plt.ylabel("Relative $L^2$ error")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Parameter shift
# ============================================================

def plot_parameter_shift(results, xlabel="Parameter value", title=None):
    set_plot_style()

    x = results["param_values"]
    y = [_extract_scalar_error(e) for e in results["errors"]]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Relative $L^2$ error")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Boundary / payoff shift
# ============================================================

def plot_boundary_or_payoff_shift(results, title=None):
    set_plot_style()

    labels = results["shift_values"]
    y = [_extract_scalar_error(e) for e in results["errors"]]

    plt.figure()
    plt.plot(range(len(labels)), y, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.xlabel("Boundary / payoff condition")
    plt.ylabel("Relative $L^2$ error")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Cross-PDE comparison plots
# ============================================================

def plot_cross_pde_metric(values, labels, ylabel, title=None):
    """
    Generic comparison across PDEs (e.g. growth rate, amplification).
    """
    set_plot_style()
    plt.figure()

    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    # Minimal synthetic test
    fake_eval = {
        "static_l2": 0.1,
        "spectral": {
            "freqs": np.linspace(0, 1, 10),
            "errors": np.linspace(0.05, 0.5, 10),
        }
    }

    plot_spectral_error(fake_eval, title="Spectral error (example)")
