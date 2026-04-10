# Neural Operator Robustness Evaluation Framework

This repository contains the experimental framework for stress-testing **neural PDE solver architectures** under deployment-relevant distribution shifts.

The project is now framed as a **robustness evaluation framework** rather than an FNO-only benchmark:
- FNO is supported as the canonical reference system.
- A lightweight DeepONet-style baseline is included for architecture comparison.
- A convolutional neural operator baseline (`cno`) is included for cross-architecture robustness comparisons.

## PDE Families
- Poisson (elliptic)
- Black–Scholes (parabolic)
- Schrödinger (dispersive)
- Navier–Stokes (advective / multi-scale)
- Kuramoto–Sivashinsky (chaotic)

## Stress Tests
- Parameter / coefficient shifts
- Boundary and terminal-condition shifts
- Resolution extrapolation
- Long-horizon rollout stability
- Input perturbation sensitivity

## Metrics (beyond a single degradation score)
The analysis pipeline reports:
- baseline-normalized degradation factors
- absolute baseline L2 error
- rollout growth rate and rollout amplification
- spectral diagnostics, including high-frequency error fraction
- 95% confidence intervals over random seeds

## Reproducibility
Run multi-seed sweeps by architecture:

```bash
python run_many_seeds.py --model fno --iters 10
python run_many_seeds.py --model deeponet --iters 10
python run_many_seeds.py --model cno --iters 10
```

Then aggregate results:

```bash
python analyze_distributions.py --model fno
python analyze_distributions.py --model deeponet
python analyze_distributions.py --model cno
```

Outputs:
- `analysis/*_summary.csv` per PDE + model
- `analysis/cross_pde_degradation_matrix_<model>.csv`
- `paper_visuals/spectral_<pde>_<model>.pdf`
