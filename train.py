# train.py
# Minimal training loop for neural operator failure atlas
#
# Supports:
#   - static PDEs (Poisson, Black–Scholes)
#   - time-dependent PDEs trained via one-step prediction
#   - automatic CUDA / CPU selection
#
# Explicitly NOT included:
#   - schedulers
#   - early stopping
#   - rollout training
#   - logging frameworks

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from data import is_time_dependent
from operators import make_fno

def set_seed(seed: int):
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

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def one_step_target(y: torch.Tensor) -> torch.Tensor:
    """
    Extract one-step target from time-dependent output.

    y: (B, T, ...)
    returns: (B, ...)
    """
    if y.ndim < 2:
        raise ValueError("Expected time-dependent tensor with time dimension.")
    return y[:, 1]


# ============================================================
# Training
# ============================================================

def train(
    model: nn.Module,
    dataloader,
    n_steps: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    log_every: int = 100,
    seed: int = 0
):
    """
    Train a neural operator with MSE loss.

    Args:
        model: FNO model
        dataloader: torch DataLoader
        n_steps: total gradient steps
        lr: learning rate
        device: torch.device (auto-detect if None)
        log_every: print loss every this many steps
    """
    
    set_seed(seed)
    
    if device is None:
        device = get_device()

    model = model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    step = 0
    data_iter = iter(dataloader)

    while step < n_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward
        pred = model(x)

        # Static vs time-dependent handling
        if is_time_dependent(dataloader.dataset.pde_name):
            # y: (B, T, ...)
            y_target = one_step_target(y)
        else:
            y_target = y

        loss = loss_fn(pred, y_target)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"Step {step:06d} | Loss: {loss.item():.6e}")

        step += 1

    print("Training complete.")
    return model


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    from data import make_dataloader

    device = get_device()

    # Example: Poisson
    loader = make_dataloader(
        pde_name="poisson",
        batch_size=8,
        n_samples=64,
        n=128,
        nt=None,
        device=device,
    )

    model = make_fno(
        dim=1,
        in_channels=3,
        out_channels=1,
        width=64,
        depth=4,
        modes1=16,
        use_coords=True,
    )

    train(
        model=model,
        dataloader=loader,
        n_steps=500,
        lr=1e-3,
        device=device,
        log_every=50,
    )
