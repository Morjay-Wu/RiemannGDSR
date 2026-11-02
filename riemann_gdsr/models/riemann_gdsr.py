"""
RiemannGDSR: RGB-Guided Depth Super-Resolution & Completion via
Monocular-Prior Self-Calibration and Spectral Screened-Poisson S^2 Geodesic Unrolling
(CVPR 2026)
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .sensor_calibration import SensorCalibration
from .tangent_transport import (
    TangentSpaceProjection,
    LeviCivitaParallelTransport,
)
from .geodesic_retraction import GeodesicExponentialRetraction
from .screened_poisson_solver import ScreenedPoissonRangeNullSolver


@dataclass
class RiemannGDSRConfig:
    """
    Configuration dataclass for RiemannGDSRNet.
    """
    in_channels_rgb: int = 3
    feature_dim: int = 48
    num_steps: int = 4
    psf_kernel_size: int = 15
    coarse_grid_size: int = 16
    max_geodesic_angle: float = math.pi / 3.0
    screen_lambda: float = 0.25
    use_checkpoint: bool = True


class SEChannelAttention(nn.Module):
    """Lightweight Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ResidualGuidanceBlock(nn.Module):
    """
    Edge-preserving multi-dilation residual block with channel attention for RGB/Depth feature extraction.
    """

    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.conv_local = nn.Conv2d(channels, half, kernel_size=3, padding=1)
        self.conv_context = nn.Conv2d(channels, channels - half, kernel_size=3, padding=2, dilation=2)
        self.act = nn.GELU()
        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca = SEChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.conv_local(x), self.conv_context(x)], dim=1)
        h = self.ca(self.fuse(self.act(h)))
        return x + h


class RGBGuidanceEncoder(nn.Module):
    """
    Lightweight multi-scale RGB structural feature encoder (full + 1/2 + 1/4 context pyramid).
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 48):
        super().__init__()
        # Compute explicit Sobel edge magnitudes + raw RGB for sharp boundary awareness
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.block1 = ResidualGuidanceBlock(feature_dim)
        self.block2 = ResidualGuidanceBlock(feature_dim)
        self.ctx_half = ResidualGuidanceBlock(feature_dim)
        self.ctx_quarter = ResidualGuidanceBlock(feature_dim)
        self.out_proj = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )

    @staticmethod
    def _rgb_gradients(rgb: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        pad_g = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx = (pad_g[:, :, 1:-1, 2:] - pad_g[:, :, 1:-1, :-2]) * 0.5
        gy = (pad_g[:, :, 2:, 1:-1] - pad_g[:, :, :-2, 1:-1]) * 0.5
        return torch.cat([gx, gy], dim=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        grad_xy = self._rgb_gradients(rgb)
        x = self.stem(torch.cat([rgb, grad_xy], dim=1))
        x = self.block1(x)
        x_fine = self.block2(x)

        # Multi-scale context pyramid for large scale factors (x8, x16)
        x_d2 = F.avg_pool2d(x_fine, kernel_size=2, stride=2)
        x_d2 = self.ctx_half(x_d2)
        x_d4 = F.avg_pool2d(x_d2, kernel_size=2, stride=2)
        x_d4 = self.ctx_quarter(x_d4)
        x_ctx = F.interpolate(x_d2, size=(H, W), mode="bilinear", align_corners=False) + F.interpolate(
            x_d4, size=(H, W), mode="bilinear", align_corners=False
        )
        return x_fine + self.out_proj(torch.cat([x_fine, x_ctx], dim=1))


class MultiScaleDetailRefiner(nn.Module):
    """
    Scale-invariant multi-scale RGB-D detail refiner and edge-insulated range-null gate predictor.
    """

    def __init__(self, in_dim: int, c_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, c_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fine_block = ResidualGuidanceBlock(c_dim)
        self.coarse_block = ResidualGuidanceBlock(c_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(c_dim * 2, c_dim, kernel_size=1),
            nn.GELU(),
            ResidualGuidanceBlock(c_dim),
        )
        self.delta_head = nn.Conv2d(c_dim, 1, kernel_size=3, padding=1)
        self.gate_head = nn.Conv2d(c_dim, 1, kernel_size=3, padding=1)
        self.res_scale = nn.Parameter(torch.tensor(0.15))

        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = x.shape[-2:]
        feat = self.stem(x)
        f_fine = self.fine_block(feat)
        f_sub = F.avg_pool2d(f_fine, kernel_size=2, stride=2)
        f_sub = self.coarse_block(f_sub)
        f_up = F.interpolate(f_sub, size=(H, W), mode="bilinear", align_corners=False)
        f_out = self.fuse(torch.cat([f_fine, f_up], dim=1))
        delta = self.res_scale * torch.tanh(self.delta_head(f_out))
        gate = torch.sigmoid(self.gate_head(f_out))
        return delta, gate


class RiemannianUnrollingStep(nn.Module):
    """
    Single Coupled Riemannian Unrolling Step:
    1. Predict raw normal update -> Project onto T_{n^(k-1)} S^2 -> Levi-Civita Parallel Transport ->
       Geodesic Exponential Retraction to obtain n^(k) in S^2_+.
    2. Solve Spectral Screened-Poisson PDE via 2D FFT -> RGB-Guided Detail Residual Coupling ->
       Apply Edge-Insulated Range-Null Space Projection P_Null to obtain D^(k).
    """

    def __init__(self, config: RiemannGDSRConfig):
        super().__init__()
        c_dim = config.feature_dim
        # Input to tangent predictor: RGB features (c_dim) + norm depth (1) + norm range res (1) + normal (3)
        self.tangent_update_net = nn.Sequential(
            nn.Conv2d(c_dim + 5, c_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualGuidanceBlock(c_dim),
            nn.Conv2d(c_dim, 3, kernel_size=3, padding=1),
        )
        # Initialize update head with small weights for smooth initial geodesic trajectory
        nn.init.normal_(self.tangent_update_net[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.tangent_update_net[-1].bias)

        self.tangent_proj = TangentSpaceProjection()
        self.parallel_transport = LeviCivitaParallelTransport(channels=3)
        self.retraction = GeodesicExponentialRetraction(max_angle=config.max_geodesic_angle)
        self.poisson_solver = ScreenedPoissonRangeNullSolver(default_lambda=config.screen_lambda)

        # High-frequency RGB-guided depth detail refinement head prior to Range-Null projection
        # Input: f_rgb (c_dim) + d_poisson_norm (1) + d_prev_norm (1) + res_norm (1) + n_k (3) = c_dim + 6
        self.detail_head = MultiScaleDetailRefiner(in_dim=c_dim + 6, c_dim=c_dim)

    def forward(
        self,
        n_prev: torch.Tensor,
        d_prev: torch.Tensor,
        rgb: torch.Tensor,
        f_rgb: torch.Tensor,
        d_lr: torch.Tensor,
        k_spatial: torch.Tensor,
        scale_factor: int,
        mask_lr: torch.Tensor,
        sensor_calib: SensorCalibration,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = d_prev.shape[-2:]
        # Per-sample scale-invariant normalization frame
        d_min = d_prev.amin(dim=(-2, -1), keepdim=True).detach()
        d_max = d_prev.amax(dim=(-2, -1), keepdim=True).detach()
        d_span = (d_max - d_min).clamp_min(1e-2)
        d_prev_norm = (d_prev - d_min) / d_span

        # Smooth bicubic LR measurement residual feedback
        deg_prev = sensor_calib.apply_A(d_prev, k_spatial, scale_factor, mask_lr)
        res_lr_prev = (d_lr - deg_prev) * mask_lr
        res_up_bic = F.interpolate(res_lr_prev.float(), size=(H, W), mode="bicubic", align_corners=False).type_as(d_prev)
        res_norm = torch.clamp(res_up_bic / d_span, min=-2.0, max=2.0)

        # Step 1: Manifold Tangent Space Update on S^2_+
        inp = torch.cat([f_rgb, d_prev_norm, res_norm, n_prev], dim=1)
        v_raw = self.tangent_update_net(inp)
        v_perp = self.tangent_proj(v_raw, n_prev)
        v_trans = self.parallel_transport(v_perp, n_prev, rgb, d_prev_norm)
        n_k = self.retraction(n_prev, v_trans)

        # Step 2: Spectral Screened-Poisson + Multi-Scale Detail Coupling + Edge-Insulated Range-Null Projection
        d_poisson = self.poisson_solver.forward_step(n_k, d_prev)
        d_poisson_norm = (d_poisson - d_min) / d_span
        detail_inp = torch.cat([f_rgb, d_poisson_norm, d_prev_norm, res_norm, n_k], dim=1)
        delta_norm, range_gate = self.detail_head(detail_inp)
        d_candidate = torch.clamp(d_poisson + delta_norm * d_span, min=1e-3)
        d_k = sensor_calib.project_range_null(
            d_candidate, d_lr, k_spatial, scale_factor, mask_lr, soft_gate=range_gate
        )
        d_k = torch.clamp(d_k, min=1e-3)

        return n_k, d_k


class RiemannGDSRNet(nn.Module):
    """
    Main RiemannGDSR Architecture:
    - RGB Guidance Encoder
    - Monocular-Prior Sensor Self-Calibration (alpha_phi, beta_phi, K_phi, D^(0))
    - Initial Normal Estimation n^(0) in S^2_+
    - K-step Coupled S^2_+ Geodesic Unrolling & Screened-Poisson Range-Null Solver
    """

    def __init__(self, config: RiemannGDSRConfig | None = None):
        super().__init__()
        if config is None:
            config = RiemannGDSRConfig()
        self.config = config

        self.rgb_encoder = RGBGuidanceEncoder(
            in_channels=config.in_channels_rgb,
            feature_dim=config.feature_dim,
        )
        self.sensor_calib = SensorCalibration(
            in_channels=config.feature_dim,
            kernel_size=config.psf_kernel_size,
            coarse_grid_size=config.coarse_grid_size,
        )
        self.unrolling_steps = nn.ModuleList(
            [RiemannianUnrollingStep(config) for _ in range(config.num_steps)]
        )

    @staticmethod
    def _depth_to_normal(depth: torch.Tensor) -> torch.Tensor:
        """
        Compute initial unit surface normal n^(0) in S^2_+ from initial depth D^(0) in FP32.
        """
        with torch.amp.autocast(device_type=depth.device.type, enabled=False):
            depth_f = depth.float()
            pad_d = F.pad(depth_f, (1, 1, 1, 1), mode="replicate")
            dx = (pad_d[:, :, 1:-1, 2:] - pad_d[:, :, 1:-1, :-2]) * 0.5
            dy = (pad_d[:, :, 2:, 1:-1] - pad_d[:, :, :-2, 1:-1]) * 0.5

            # Normal vector (-dx, -dy, 1.0) normalized onto upper hemisphere S^2_+
            ones = torch.ones_like(dx)
            n_raw = torch.cat([-dx, -dy, ones], dim=1)
            n_0 = F.normalize(n_raw, p=2, dim=1, eps=1e-6)
        return n_0

    def forward(
        self,
        rgb: torch.Tensor,
        d_lr: torch.Tensor,
        mask_lr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass of RiemannGDSRNet.
        Args:
            rgb: [B, 3, H, W] high-resolution RGB guidance image.
            d_lr: [B, 1, h, w] low-resolution depth input (scale factor s = H // h in {4, 8, 16}).
            mask_lr: optional [B, 1, h, w] binary mask of valid LR depth pixels.
        Returns:
            dict with keys:
                'pred': [B, 1, H, W] final super-resolved depth D^(K)
                'normal': [B, 3, H, W] final unit surface normal n^(K) in S^2_+
                'intermediate_depths': list of [D^(0), D^(1), ..., D^(K)]
                'normals': list of [n^(0), n^(1), ..., n^(K)]
                'alpha': [B, 1, H, W] calibrated scale map
                'beta': [B, 1, H, W] calibrated shift map
                'k_spatial': [B, 1, ks, ks] estimated PSF kernel
        """
        H, W = rgb.shape[-2:]
        h, w = d_lr.shape[-2:]
        scale_factor = H // h

        if mask_lr is None:
            mask_lr = (d_lr > 0).float()
        else:
            mask_lr = mask_lr.float()

        # 1. Extract RGB guidance features
        f_rgb = self.rgb_encoder(rgb)

        # 2. Sensor Self-Calibration & Warm-Start D^(0)
        d_0, k_spatial, alpha, beta = self.sensor_calib(f_rgb, d_lr, mask_lr)
        n_0 = self._depth_to_normal(d_0)

        intermediate_depths: list[torch.Tensor] = [d_0]
        normals: list[torch.Tensor] = [n_0]

        n_cur = n_0
        d_cur = d_0

        # 3. K-step Coupled S^2_+ Geodesic Unrolling & Screened-Poisson Solver
        for step_module in self.unrolling_steps:
            if self.config.use_checkpoint and self.training and torch.is_grad_enabled():
                n_cur, d_cur = checkpoint(
                    step_module,
                    n_cur,
                    d_cur,
                    rgb,
                    f_rgb,
                    d_lr,
                    k_spatial,
                    scale_factor,
                    mask_lr,
                    self.sensor_calib,
                    use_reentrant=False,
                )
            else:
                n_cur, d_cur = step_module(
                    n_cur,
                    d_cur,
                    rgb,
                    f_rgb,
                    d_lr,
                    k_spatial,
                    scale_factor,
                    mask_lr,
                    self.sensor_calib,
                )
            normals.append(n_cur)
            intermediate_depths.append(d_cur)

        return {
            "pred": d_cur,
            "normal": n_cur,
            "intermediate_depths": intermediate_depths,
            "normals": normals,
            "alpha": alpha,
            "beta": beta,
            "k_spatial": k_spatial,
        }
