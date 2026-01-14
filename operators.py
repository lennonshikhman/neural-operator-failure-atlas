# operators.py
# Minimal Fourier Neural Operator (FNO) for 1D and 2D.
# Supports:
#   1D: inputs (B, N, C_in) -> outputs (B, N, C_out)
#   2D: inputs (B, N, N, C_in) -> outputs (B, N, N, C_out)
#
# Key features for failure-atlas work:
#   - explicit Fourier mode truncation (modes parameter)
#   - optional coordinate channels (positional encoding)
#   - clean forward API, no training logic
#
# Dependencies: torch

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


# ============================================================
# Utilities
# ============================================================

def _assert_shape(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def add_coords_1d(x: Tensor) -> Tensor:
    """
    x: (B, N, C)
    returns: (B, N, C+1) with x-coordinate in [0,1)
    """
    B, N, _ = x.shape
    device = x.device
    coord = torch.linspace(0.0, 1.0, N, device=device, dtype=x.dtype, requires_grad=False)
    coord = coord[None, :, None].expand(B, N, 1)
    return torch.cat([x, coord], dim=-1)


def add_coords_2d(x: Tensor) -> Tensor:
    """
    x: (B, N, N, C)
    returns: (B, N, N, C+2) with (x,y) coords in [0,1)
    """
    B, N, M, _ = x.shape
    _assert_shape(N == M, "FNO2D expects square grids (N == M).")
    device = x.device
    xs = torch.linspace(0.0, 1.0, N, device=device, dtype=x.dtype, requires_grad=False)
    ys = torch.linspace(0.0, 1.0, N, device=device, dtype=x.dtype, requires_grad=False)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X, Y], dim=-1)  # (N, N, 2)
    coords = coords[None, :, :, :].expand(B, N, N, 2)
    return torch.cat([x, coords], dim=-1)


# ============================================================
# Spectral convolution layers
# ============================================================

class SpectralConv1d(nn.Module):
    """
    1D spectral convolution:
      - rFFT along spatial axis
      - multiply low modes by complex weights
      - iFFT back

    Input:  (B, C_in, N)
    Output: (B, C_out, N)
    """
    def __init__(self, c_in: int, c_out: int, modes: int):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.modes = modes

        # Complex weights for low-frequency modes:
        # shape: (C_in, C_out, modes)
        scale = 1.0 / (c_in * c_out)
        self.weight_real = nn.Parameter(scale * torch.randn(c_in, c_out, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(c_in, c_out, modes))

    def forward(self, x: Tensor) -> Tensor:
        B, C_in, N = x.shape
        _assert_shape(C_in == self.c_in, "SpectralConv1d: channel mismatch.")
        # rfft -> (B, C_in, N//2+1) complex
        x_ft = torch.fft.rfft(x, dim=-1)

        # allocate output in Fourier domain
        out_ft = torch.zeros(B, self.c_out, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)

        m = min(self.modes, x_ft.shape[-1])
        w = torch.complex(self.weight_real[:, :, :m], self.weight_imag[:, :, :m])  # (C_in, C_out, m)

        # out_ft[..., :m] = sum_{c_in} x_ft[:, c_in, :m] * w[c_in, c_out, :m]
        out_ft[:, :, :m] = torch.einsum("bci,coi->boi", x_ft[:, :, :m], w)

        # back to physical space
        out = torch.fft.irfft(out_ft, n=N, dim=-1)
        return out


class SpectralConv2d(nn.Module):
    """
    2D spectral convolution:
      - rFFT2 over (H,W)
      - multiply a low-frequency rectangle by complex weights
      - iFFT2 back

    Input:  (B, C_in, H, W) with H=W=N (assumed square for simplicity)
    Output: (B, C_out, H, W)
    """
    def __init__(self, c_in: int, c_out: int, modes: Tuple[int, int]):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.modes1, self.modes2 = modes

        # Two sets of weights: positive and negative frequency blocks (common FNO trick)
        # For rfft2, the second axis is truncated (W//2+1), while first keeps full.
        scale = 1.0 / (c_in * c_out)
        m1, m2 = self.modes1, self.modes2

        self.w1_real = nn.Parameter(scale * torch.randn(c_in, c_out, m1, m2))
        self.w1_imag = nn.Parameter(scale * torch.randn(c_in, c_out, m1, m2))
        self.w2_real = nn.Parameter(scale * torch.randn(c_in, c_out, m1, m2))
        self.w2_imag = nn.Parameter(scale * torch.randn(c_in, c_out, m1, m2))

    def forward(self, x: Tensor) -> Tensor:
        B, C_in, H, W = x.shape
        _assert_shape(C_in == self.c_in, "SpectralConv2d: channel mismatch.")

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C_in, H, W//2+1) complex
        out_ft = torch.zeros(B, self.c_out, H, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.shape[-1])

        w1 = torch.complex(self.w1_real[:, :, :m1, :m2], self.w1_imag[:, :, :m1, :m2])
        w2 = torch.complex(self.w2_real[:, :, :m1, :m2], self.w2_imag[:, :, :m1, :m2])

        # low freq block
        out_ft[:, :, :m1, :m2] = torch.einsum("bcij,coij->boij", x_ft[:, :, :m1, :m2], w1)
        # symmetric high freq block in first dimension (negative frequencies)
        out_ft[:, :, -m1:, :m2] = torch.einsum("bcij,coij->boij", x_ft[:, :, -m1:, :m2], w2)

        out = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))
        return out


# ============================================================
# FNO blocks
# ============================================================

class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int, act: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, width, N)
        y = self.spectral(x) + self.pointwise(x)
        if self.act == "gelu":
            return F.gelu(y)
        if self.act == "relu":
            return F.relu(y)
        return y


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes: Tuple[int, int], act: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, width, H, W)
        y = self.spectral(x) + self.pointwise(x)
        if self.act == "gelu":
            return F.gelu(y)
        if self.act == "relu":
            return F.relu(y)
        return y


# ============================================================
# Full FNO models
# ============================================================

@dataclass
class FNOConfig:
    dim: int                  # 1 or 2
    in_channels: int
    out_channels: int
    width: int = 64
    depth: int = 4
    modes1: int = 16
    modes2: Optional[int] = None   # used for 2D
    use_coords: bool = True
    act: str = "gelu"
    pad_ratio: float = 0.0         # optional padding to reduce wrap-around effects (0 means none)


class FNO1D(nn.Module):
    def __init__(self, cfg: FNOConfig):
        super().__init__()
        _assert_shape(cfg.dim == 1, "FNO1D requires cfg.dim == 1")
        self.cfg = cfg
        self.use_coords = cfg.use_coords
        in_ch = cfg.in_channels + (1 if cfg.use_coords else 0)

        # lifting: (B,N,C)->(B,N,width)
        self.fc0 = nn.Linear(in_ch, cfg.width)

        self.blocks = nn.ModuleList([FNOBlock1d(cfg.width, cfg.modes1, act=cfg.act) for _ in range(cfg.depth)])

        # projection: width -> out_channels
        self.fc1 = nn.Linear(cfg.width, cfg.width)
        self.fc2 = nn.Linear(cfg.width, cfg.out_channels)

        self.pad_ratio = float(cfg.pad_ratio)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, N, C_in)
        returns: (B, N, C_out)
        """
        _assert_shape(x.ndim == 3, "FNO1D expects (B, N, C_in).")
        if self.use_coords:
            x = add_coords_1d(x)

        B, N, _ = x.shape

        x = self.fc0(x)  # (B,N,width)

        # optional padding in spatial domain to reduce periodic artifacts
        if self.pad_ratio > 0.0:
            pad = int(N * self.pad_ratio)
        else:
            pad = 0

        # move to (B,width,N)
        x = x.permute(0, 2, 1).contiguous()

        if pad > 0:
            x = F.pad(x, (0, pad), mode="constant", value=0.0)

        for blk in self.blocks:
            x = blk(x)

        if pad > 0:
            x = x[..., :N]

        # back to (B,N,width)
        x = x.permute(0, 2, 1).contiguous()
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class FNO2D(nn.Module):
    def __init__(self, cfg: FNOConfig):
        super().__init__()
        _assert_shape(cfg.dim == 2, "FNO2D requires cfg.dim == 2")
        _assert_shape(cfg.modes2 is not None, "FNO2D requires cfg.modes2 (modes in second dim).")
        self.cfg = cfg
        self.use_coords = cfg.use_coords
        in_ch = cfg.in_channels + (2 if cfg.use_coords else 0)

        # lifting: (B,H,W,C)->(B,H,W,width)
        self.fc0 = nn.Linear(in_ch, cfg.width)

        self.blocks = nn.ModuleList(
            [FNOBlock2d(cfg.width, (cfg.modes1, cfg.modes2), act=cfg.act) for _ in range(cfg.depth)]
        )

        self.fc1 = nn.Linear(cfg.width, cfg.width)
        self.fc2 = nn.Linear(cfg.width, cfg.out_channels)

        self.pad_ratio = float(cfg.pad_ratio)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, H, W, C_in)
        returns: (B, H, W, C_out)
        """
        _assert_shape(x.ndim == 4, "FNO2D expects (B, H, W, C_in).")
        if self.use_coords:
            x = add_coords_2d(x)

        B, H, W, _ = x.shape
        _assert_shape(H == W, "FNO2D expects square grids (H == W).")

        x = self.fc0(x)  # (B,H,W,width)

        pad = int(H * self.pad_ratio) if self.pad_ratio > 0.0 else 0

        # to (B,width,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()

        if pad > 0:
            x = F.pad(x, (0, pad, 0, pad), mode="constant", value=0.0)

        for blk in self.blocks:
            x = blk(x)

        if pad > 0:
            x = x[..., :H, :W]

        # back to (B,H,W,width)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_fno(
    dim: int,
    in_channels: int,
    out_channels: int,
    width: int = 64,
    depth: int = 4,
    modes1: int = 16,
    modes2: Optional[int] = None,
    use_coords: bool = True,
    act: str = "gelu",
    pad_ratio: float = 0.0,
) -> nn.Module:
    """
    Factory for 1D/2D FNO.
    """
    cfg = FNOConfig(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        width=width,
        depth=depth,
        modes1=modes1,
        modes2=modes2,
        use_coords=use_coords,
        act=act,
        pad_ratio=pad_ratio,
    )
    if dim == 1:
        return FNO1D(cfg)
    if dim == 2:
        return FNO2D(cfg)
    raise ValueError("dim must be 1 or 2")


# ============================================================
# Optional: quick shape smoke tests
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1D test
    x1 = torch.randn(8, 128, 3)  # (B,N,C)
    m1 = make_fno(dim=1, in_channels=3, out_channels=1, width=32, depth=4, modes1=16, use_coords=True)
    y1 = m1(x1)
    print("FNO1D:", y1.shape)  # (8,128,1)

    # 2D test
    x2 = torch.randn(4, 64, 64, 3)  # (B,H,W,C)
    m2 = make_fno(dim=2, in_channels=3, out_channels=1, width=32, depth=4, modes1=12, modes2=12, use_coords=True)
    y2 = m2(x2)
    print("FNO2D:", y2.shape)  # (4,64,64,1)
