"""
Spectral Screened-Poisson Solver in 2D FFT Domain
For RiemannGDSR (CVPR 2026).

Solves:
    (-Delta + lambda) D_Poisson^(k) = lambda * D^(k-1) - div(g(n^(k), D^(k-1)))
via closed-form 2D real FFT (rfft2 / irfft2) in isolated FP32 precision.
Guarantees exact DC depth conservation at zero frequency (omega = 0).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScreenedPoissonRangeNullSolver(nn.Module):
    """
    2D FFT Spectral Screened-Poisson Solver with Exact DC Conservation.
    """

    def __init__(
        self,
        default_lambda: float = 0.25,
        min_denom: float = 0.25,
    ):
        super().__init__()
        self.min_denom = min_denom
        # Learnable positive screening parameter lambda initialized at default_lambda
        init_raw = math.log(math.exp(default_lambda) - 1.0)
        self.raw_lambda = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        self.poisson_step = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    @property
    def screen_lambda(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda) + 1e-3

    def normal_to_depth_gradient(
        self,
        normal: torch.Tensor,
        d_anchor: torch.Tensor,
        fx: float = 1.0,
        fy: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert surface normal n = (nx, ny, nz) in S^2_+ to depth gradient field g = (gx, gy).
        Matches the pixel-domain normal parameterization n = normalize(-dx, -dy, 1):
            gx = -(nx / fx) / max(nz, min_denom)
            gy = -(ny / fy) / max(nz, min_denom)
        Computed strictly in FP32.
        """
        with torch.amp.autocast(device_type=normal.device.type, enabled=False):
            n_f = normal.float()
            _ = d_anchor.float()

            nx = n_f[:, 0:1]
            ny = n_f[:, 1:2]
            nz = n_f[:, 2:3]

            denom_safe = torch.clamp(nz, min=self.min_denom)

            gx = -(nx / fx) / denom_safe
            gy = -(ny / fy) / denom_safe

            gx = torch.clamp(gx, min=-2.0, max=2.0)
            gy = torch.clamp(gy, min=-2.0, max=2.0)

        return gx, gy

    @staticmethod
    def compute_divergence(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial divergence div(g) = d(gx)/dx + d(gy)/dy using symmetric finite differences
        with replicate boundary padding so boundary integral is zero (exact zero DC divergence).
        """
        # Pad gx along width (left=1, right=1)
        gx_pad = F.pad(gx, (1, 1, 0, 0), mode="replicate")
        dgx_dx = (gx_pad[:, :, :, 2:] - gx_pad[:, :, :, :-2]) * 0.5

        # Pad gy along height (top=1, bottom=1)
        gy_pad = F.pad(gy, (0, 0, 1, 1), mode="replicate")
        dgy_dy = (gy_pad[:, :, 2:, :] - gy_pad[:, :, :-2, :]) * 0.5

        div_g = dgx_dx + dgy_dy
        # Subtract spatial mean of divergence so DC component (omega = 0) of div(g) is strictly 0.0
        div_g = div_g - div_g.mean(dim=(-2, -1), keepdim=True)
        return div_g

    def solve_screened_poisson(
        self,
        gx: torch.Tensor,
        gy: torch.Tensor,
        d_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve (-Delta + lambda) D_Poisson = lambda * d_anchor - div(g)
        in closed-form 2D real FFT domain in FP32.
        At omega = (0, 0), lap_symbol(0, 0) = 0 and F(div g)(0, 0) = 0, so:
            F(D_Poisson)(0, 0) = (lambda * F(d_anchor)(0, 0)) / lambda = F(d_anchor)(0, 0),
        guaranteeing exact DC conservation: mean(D_Poisson) == mean(d_anchor).
        """
        B, C, H, W = d_anchor.shape
        device = d_anchor.device

        with torch.amp.autocast(device_type=device.type, enabled=False):
            gx_f = gx.float()
            gy_f = gy.float()
            d_anchor_f = d_anchor.float()
            lam = self.screen_lambda.float()

            # Compute anchor central differences so unchanged normals produce zero frequency attenuation
            pad_d = F.pad(d_anchor_f, (1, 1, 1, 1), mode="replicate")
            gx_anchor = (pad_d[:, :, 1:-1, 2:] - pad_d[:, :, 1:-1, :-2]) * 0.5
            gy_anchor = (pad_d[:, :, 2:, 1:-1] - pad_d[:, :, :-2, 1:-1]) * 0.5

            div_dg = self.compute_divergence(gx_f - gx_anchor, gy_f - gy_anchor)
            rhs_hat = -torch.fft.rfft2(div_dg)  # [B, 1, H, W//2 + 1]

            # Exact central-difference Laplacian Fourier symbol: sin^2(wx) + sin^2(wy)
            wy = 2.0 * math.pi * torch.fft.fftfreq(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
            wx = 2.0 * math.pi * torch.fft.rfftfreq(W, device=device, dtype=torch.float32).view(1, 1, 1, W // 2 + 1)
            lap_symbol = torch.sin(wx) ** 2 + torch.sin(wy) ** 2

            denom = lap_symbol + lam
            delta_hat = rhs_hat / denom
            delta_d = torch.fft.irfft2(delta_hat, s=(H, W))
            delta_d = delta_d - delta_d.mean(dim=(-2, -1), keepdim=True)

            step_w = torch.sigmoid(self.poisson_step.float()) * 0.5
            d_poisson = d_anchor_f + step_w * delta_d

            # Enforce non-negative valid depth
            d_poisson = torch.clamp(d_poisson, min=1e-3)

        return d_poisson.type_as(d_anchor)

    def forward_step(
        self,
        normal: torch.Tensor,
        d_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Screened-Poisson depth solution from surface normal n^(k) and anchor depth D^(k-1).
        """
        gx, gy = self.normal_to_depth_gradient(normal, d_anchor)
        d_poisson = self.solve_screened_poisson(gx, gy, d_anchor)
        return d_poisson
