# data.py
# Dataset + DataLoader utilities for neural operator failure atlas
#
# Responsibilities:
#   - wrap pdes.sample_pde into torch Datasets
#   - handle static vs time-dependent PDEs
#   - move tensors to device
#
# Explicitly NOT responsible for:
#   - training logic
#   - rollout logic
#   - stress tests
#   - metrics

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pdes import sample_pde


# ============================================================
# Utilities
# ============================================================

TIME_DEPENDENT_PDES = {
    "schrodinger",
    "navier_stokes",
    "kuramoto_sivashinsky",
}


def is_time_dependent(pde_name: str) -> bool:
    return pde_name.lower() in TIME_DEPENDENT_PDES


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)


# ============================================================
# Dataset
# ============================================================

class PDEOperatorDataset(Dataset):
    """
    Torch Dataset for supervised operator learning.

    For static PDEs:
        returns (x, y)
            x: (N, C_in) or (N, N, C_in)
            y: (N, C_out) or (N, N, C_out)

    For time-dependent PDEs:
        returns (x, y)
            x: (N, C_in) or (N, N, C_in)
            y: (T, N, C_out) or (T, N, N, C_out)
    """

    def __init__(
        self,
        pde_name: str,
        n_samples: int,
        n: int,
        nt: Optional[int],
        device: torch.device,
        seed: int = 0,
        **pde_kwargs,
    ):
        self.pde_name = pde_name.lower()
        self.n_samples = n_samples
        self.n = n
        self.nt = nt
        self.device = device
        self.pde_kwargs = pde_kwargs

        self.rng = np.random.default_rng(seed)

        self._data = []
        self._meta = []

        self._build()

    def _build(self):
        for _ in range(self.n_samples):

            kwargs = dict(
                rng=self.rng,
                n=self.n,
                **self.pde_kwargs,
            )

            # Only pass nt for time-dependent PDEs
            if is_time_dependent(self.pde_name):
                kwargs["nt"] = self.nt

            inp, out, meta = sample_pde(self.pde_name, **kwargs)

            x = to_tensor(inp, self.device)
            y = to_tensor(out, self.device)

            self._data.append((x, y))
            self._meta.append(meta)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        return self._data[idx]

    def get_meta(self, idx: int) -> Dict[str, Any]:
        return self._meta[idx]


# ============================================================
# DataLoader factory
# ============================================================

def make_dataloader(
    pde_name: str,
    batch_size: int,
    n_samples: int,
    n: int,
    nt: Optional[int],
    device: torch.device,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    **pde_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for a given PDE.

    Args:
        pde_name: one of pdes.PDE_NAMES
        batch_size: batch size
        n_samples: number of samples to generate
        n: spatial resolution
        nt: time steps (for time-dependent PDEs)
        device: torch.device
        shuffle: shuffle dataset
        seed: RNG seed for reproducibility

    Returns:
        torch.utils.data.DataLoader
    """

    dataset = PDEOperatorDataset(
        pde_name=pde_name,
        n_samples=n_samples,
        n=n,
        nt=nt,
        device=device,
        seed=seed,
        **pde_kwargs,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )

    return loader


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Static PDE
    loader = make_dataloader(
        pde_name="poisson",
        batch_size=4,
        n_samples=8,
        n=128,
        nt=None,
        device=device,
    )
    x, y = next(iter(loader))
    print("Poisson batch:", x.shape, y.shape)

    # Time-dependent PDE
    loader = make_dataloader(
        pde_name="kuramoto_sivashinsky",
        batch_size=2,
        n_samples=4,
        n=128,
        nt=20,
        device=device,
    )
    x, y = next(iter(loader))
    print("KS batch:", x.shape, y.shape)
