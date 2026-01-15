# Neural Operator Failure Atlas

This repository contains the experimental code for  
**“Forcing and Diagnosing Failure Modes of Fourier Neural Operators Across Diverse PDE Classes.”**

Rather than benchmarking accuracy, this work systematically induces and diagnoses failure
modes of neural operators under controlled violations of modeling assumptions.

## PDE Families
- Poisson (elliptic)
- Black–Scholes (parabolic)
- Schrödinger (dispersive)
- Navier–Stokes (advective)
- Kuramoto–Sivashinsky (chaotic)

## Stress Tests
- Parameter / coefficient shifts
- Boundary and terminal condition shifts
- Resolution extrapolation
- Long-horizon rollout (operator composition)
- Input perturbation sensitivity

## Key Idea
All failures are measured **relative to an in-distribution baseline**, revealing structural
instabilities that are invisible to standard one-step accuracy metrics.

## Reproducibility
Experiments can be reproduced by running:

```bash
python run_many_seeds.py --iters (# of seeds to test)
```

