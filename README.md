# Neural Operator Robustness Evaluation Framework

This repository contains the experimental framework for the TMLR 2026 paper:

**Diagnosing Failure Modes of Fourier Neural Operators Across Diverse PDE Families**  
Lennon Shikhman  
*Transactions on Machine Learning Research*, 2026  
OpenReview: https://openreview.net/forum?id=0S1LWZHQYn

The code is designed to stress-test neural PDE solver architectures under deployment-relevant distribution shifts. Rather than reporting a single aggregate degradation score, the framework evaluates how learned neural operators fail across PDE type, shift type, rollout horizon, resolution, and spectral regime.

## Overview

Neural operators are often evaluated on held-out samples drawn from the same distribution as training data. This repository instead focuses on robustness under controlled distribution shifts across multiple PDE families. The goal is to diagnose when and how neural PDE solvers break, especially when the test setting differs from the training regime in ways that are common in scientific computing deployments.

The framework supports multi-seed experiments, cross-PDE comparison, architecture-level evaluation, rollout diagnostics, and spectral error analysis.

## PDE Families

The experiments cover five PDE families with different mathematical and numerical characteristics:

- **Poisson**: elliptic
- **Black–Scholes**: parabolic
- **Schrödinger**: dispersive
- **Navier–Stokes**: advective / multi-scale
- **Kuramoto–Sivashinsky**: chaotic

These families are used to test whether robustness failures are architecture-specific, PDE-specific, or tied to particular forms of distribution shift.

## Architectures

The framework supports robustness evaluation for the following neural PDE solver architectures:

- Fourier Neural Operator (**FNO**)
- DeepONet
- Convolutional Neural Operator (**CNO**)

The accepted paper focuses on diagnosing failure modes of Fourier Neural Operators, while the repository also includes comparison pipelines for related operator-learning architectures.

## Stress Tests

The evaluation suite includes the following stress tests:

- Parameter / coefficient shifts
- Boundary and terminal-condition shifts
- Resolution extrapolation
- Long-horizon rollout stability
- Input perturbation sensitivity

These tests are intended to expose failures that may be hidden by standard in-distribution test error.

## Metrics

The analysis pipeline reports several diagnostics beyond a single degradation score:

- Baseline-normalized degradation factors
- Absolute baseline L2 error
- Rollout growth rate
- Rollout amplification
- Spectral diagnostics, including high-frequency error fraction
- 95% confidence intervals over random seeds

Together, these metrics are intended to separate mild accuracy degradation from structurally meaningful failure modes.

## Repository Contents

The main repository files are organized as a lightweight Python research codebase:

- `run_many_seeds.py`: runs multi-seed robustness sweeps by model architecture
- `analyze_distributions.py`: aggregates experiment outputs into summary tables and cross-PDE degradation matrices
- `data.py`: data generation and loading utilities
- `pdes.py`: PDE-family definitions
- `operators.py`: neural operator model definitions
- `train.py`: training utilities
- `eval.py`: evaluation utilities
- `metrics.py`: robustness and error metrics
- `stress.py`: distribution-shift and stress-test utilities
- `plots.py` and `plot_distributions.py`: plotting and visualization scripts

## Installation

Clone the repository:

```bash
git clone https://github.com/lennonshikhman/neural-operator-failure-atlas.git
cd neural-operator-failure-atlas
```

Create and activate a Python environment. For example, with `conda`:

```bash
conda create -n neural-operator-robustness python=3.10
conda activate neural-operator-robustness
```

Install the core scientific Python dependencies:

```bash
pip install numpy scipy pandas matplotlib torch tqdm
```

A GPU-enabled PyTorch installation is recommended for larger sweeps. If you need a CUDA-specific PyTorch build, install PyTorch using the command recommended for your system by the official PyTorch installation selector.

## Running Experiments

Run multi-seed sweeps by architecture:

```bash
python run_many_seeds.py --model fno --iters 50
python run_many_seeds.py --model deeponet --iters 50
python run_many_seeds.py --model cno --iters 50
```

The `--model` argument selects the architecture to evaluate. The `--iters` argument controls the number of random seeds in the sweep.

For the main FNO experiments from the paper, run:

```bash
python run_many_seeds.py --model fno --iters 50
```

## Aggregating Results

After running the multi-seed sweeps, aggregate the results by architecture:

```bash
python analyze_distributions.py --model fno
python analyze_distributions.py --model deeponet
python analyze_distributions.py --model cno
```

For the main FNO analysis:

```bash
python analyze_distributions.py --model fno
```

## Outputs

The experiment and analysis scripts write results to the following locations:

- `analysis/*_summary.csv`: per-PDE and per-model summary files
- `analysis/cross_pde_degradation_matrix_<model>.csv`: cross-PDE degradation matrix for each architecture
- `paper_visuals/spectral_<pde>_<model>.pdf`: spectral diagnostic figures

These outputs are used to compare robustness across PDE families, stress tests, and model architectures.

## Reproducing Paper Results

To reproduce the main robustness analysis for the accepted TMLR paper, run the FNO multi-seed sweep and then aggregate the results:

```bash
python run_many_seeds.py --model fno --iters 50
python analyze_distributions.py --model fno
```

To reproduce architecture comparisons, repeat the same workflow for `deeponet` and `cno`:

```bash
python run_many_seeds.py --model deeponet --iters 50
python analyze_distributions.py --model deeponet

python run_many_seeds.py --model cno --iters 50
python analyze_distributions.py --model cno
```

The generated CSV files and PDF figures can then be found in `analysis/` and `paper_visuals/`.

## Citation

If you use this repository or build on the associated paper, please cite:

```bibtex
@article{shikhman2026diagnosing,
  title={Diagnosing Failure Modes of Neural Operators Across Diverse {PDE} Families},
  author={Lennon Shikhman},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=0S1LWZHQYn},
  note={}
}
```

## Contact

For questions about the paper or repository, please contact Lennon Shikhman or open an issue on GitHub.
