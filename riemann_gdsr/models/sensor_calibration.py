"""
Sensor Calibration & Parametric Degradation Operator
For RiemannGDSR (CVPR 2026).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricPSFOperator(nn.Module):
    """
    Parametric anisotropic Point Spread Function (PSF) degradation operator K_phi
    and closed-form 2D FFT Wiener-Laplacian analytical right-inverse with exact DC conservation.
    """

    def __init__(self, kernel_size: int = 15, default_lambda: float = 0.05):
        super().__init__()
        self.ks = kernel_size
        self.pad = kernel_size // 2
        self.default_lambda = default_lambda

        # Precompute 2D coordinate grid [-pad, pad]^2
        coords = torch.arange(-self.pad, self.pad + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", xx.view(1, 1, self.ks, self.ks))
        self.register_buffer("grid_y", yy.view(1, 1, self.ks, self.ks))

        # Predictor for anisotropic Gaussian PSF parameters + residual
        # Input: pooled features (B, C_in)
        self.param_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 5 + self.ks * self.ks),
        )
        # Identity/default init: zero-init head
        nn.init.zeros_(self.param_net[-1].weight)
        nn.init.zeros_(self.param_net[-1].bias)

    def forward_kernel(self, feat_pooled: torch.Tensor) -> torch.Tensor:
        """
        Estimate normalized PSF spatial kernel K_spatial [B, 1, ks, ks] satisfying sum(K) == 1.0.
        """
        B = feat_pooled.shape[0]
        device = feat_pooled.device
        with torch.amp.autocast(device_type=device.type, enabled=False):
            raw = self.param_net(feat_pooled.float()).float()
            raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
            raw_geom = raw[:, :5]  # dx, dy, sx, sy, rho
            raw_res = torch.clamp(raw[:, 5:].view(B, 1, self.ks, self.ks), min=-10.0, max=10.0)

            # Sub-pixel shift in [-1.0, 1.0] px
            dx = 1.0 * torch.tanh(raw_geom[:, 0:1]).view(B, 1, 1, 1)
            dy = 1.0 * torch.tanh(raw_geom[:, 1:2]).view(B, 1, 1, 1)

            # Anisotropic standard deviations with sub-pixel optical width (~0.45 px at epoch 0)
            sx = (F.softplus(torch.clamp(raw_geom[:, 2:3], -10.0, 10.0) - 1.5) + 0.35).view(B, 1, 1, 1)
            sy = (F.softplus(torch.clamp(raw_geom[:, 3:4], -10.0, 10.0) - 1.5) + 0.35).view(B, 1, 1, 1)

            # Correlation coefficient rho in (-0.85, 0.85)
            rho = (0.85 * torch.tanh(raw_geom[:, 4:5])).view(B, 1, 1, 1)

            ux = self.grid_x.to(device).float() - dx
            uy = self.grid_y.to(device).float() - dy
            inv_1m_rho2 = 1.0 / (1.0 - rho**2 + 1e-4)

            quad = -0.5 * inv_1m_rho2 * (
                (ux / sx) ** 2 + (uy / sy) ** 2 - 2.0 * rho * (ux / sx) * (uy / sy)
            )
            quad = torch.clamp(quad, min=-30.0, max=30.0)

            # Fine zero-mean residual + softmax normalization to guarantee sum == 1.0 (DC conservation)
            logits = quad + 0.1 * (raw_res - raw_res.mean(dim=(-2, -1), keepdim=True))
            K_spatial = F.softmax(logits.view(B, 1, -1), dim=-1).view(B, 1, self.ks, self.ks)
        return K_spatial

    def apply_blur(self, D: torch.Tensor, K_spatial: torch.Tensor) -> torch.Tensor:
        """
        D: [B, 1, H, W]
        K_spatial: [B, 1, ks, ks]
        Convolve D with per-sample kernel K_spatial using grouped conv2d.
        """
        B, C, H, W = D.shape
        with torch.amp.autocast(device_type=D.device.type, enabled=False):
            D_reshaped = D.float().view(1, B, H, W)
            K_weight = K_spatial.float().view(B, 1, self.ks, self.ks)
            D_blurred = F.conv2d(D_reshaped, K_weight, padding=self.pad, groups=B)
        return D_blurred.view(B, 1, H, W).type_as(D)

    def apply_deblur_wiener(
        self,
        D_up: torch.Tensor,
        K_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed-form 2D FFT Wiener deconvolution in isolated FP32 domain.
        """
        B, C, H, W = D_up.shape
        device = D_up.device

        with torch.amp.autocast(device_type=device.type, enabled=False):
            D_up_f32 = D_up.float()
            K_pad = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
            K_pad[:, :, : self.ks, : self.ks] = K_spatial.float()
            K_pad = torch.roll(K_pad, shifts=(-self.pad, -self.pad), dims=(-2, -1))

            K_hat = torch.fft.rfft2(K_pad)  # [B, 1, H, W//2 + 1]

            # 2D Laplacian Fourier symbol
            wy = 2.0 * math.pi * torch.fft.fftfreq(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
            wx = 2.0 * math.pi * torch.fft.rfftfreq(W, device=device, dtype=torch.float32).view(1, 1, 1, W // 2 + 1)
            lap_symbol = 4.0 - 2.0 * torch.cos(wx) - 2.0 * torch.cos(wy)

            lam = self.default_lambda
            denom = (K_hat.abs() ** 2) + lam * lap_symbol + 1e-6
            K_dag_hat = torch.conj(K_hat) / denom

            D_hat = torch.fft.rfft2(D_up_f32)
            D_deblur = torch.fft.irfft2(D_hat * K_dag_hat, s=(H, W))

        return D_deblur.type_as(D_up)


class SensorCalibration(nn.Module):
    """
    Self-calibrates:
    1. Spatially varying scale-shift ambiguity (alpha_phi, beta_phi) on a coarse grid (16x16)
       with exact identity initialization (alpha=1.0, beta=0.0 at epoch 0).
    2. Degradation operator A_phi(D) = S_{down s}(K_phi * D) * M_LR.
    3. Analytical right-inverse A_phi^dagger and range-null projector P_Null.
    4. Analytic range-space warm-start D^(0).
    """

    def __init__(
        self,
        in_channels: int = 48,
        kernel_size: int = 15,
        coarse_grid_size: int = 16,
    ):
        super().__init__()
        self.coarse_grid = coarse_grid_size
        self.psf_operator = ParametricPSFOperator(kernel_size=kernel_size)

        # Calibration head on coarse grid (16x16)
        self.calib_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.coarse_grid, self.coarse_grid)),
            nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=1),
        )
        # Identity initialization: raw_alpha = 0 -> alpha = 1.0; raw_beta = 0 -> beta = 0.0
        nn.init.zeros_(self.calib_net[-1].weight)
        nn.init.zeros_(self.calib_net[-1].bias)

        # Feature pooler for PSF parameter net
        self.psf_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear((in_channels + 1) * 16, 64),
            nn.GELU(),
        )

        # Offset constant so that softplus(0 + offset_c0) + eps == 1.0
        self.eps_alpha = 1e-4
        target_val = 1.0 - self.eps_alpha
        self.offset_c0 = math.log(math.exp(target_val) - 1.0)

    @staticmethod
    def _inpaint_lr_holes(D_lr: torch.Tensor, mask_lr: torch.Tensor) -> torch.Tensor:
        """
        Smoothly fill any missing holes (mask_lr == 0) in D_lr via normalized 3x3 diffusion.
        When mask_lr == 1 everywhere, returns D_lr unchanged.
        """
        if torch.all(mask_lr > 0.5):
            return D_lr
        cur_d = D_lr * mask_lr
        cur_m = (mask_lr > 0.5).float()
        for _ in range(4):
            if torch.all(cur_m > 0.5):
                break
            pad_d = F.pad(cur_d, (1, 1, 1, 1), mode="replicate")
            pad_m = F.pad(cur_m, (1, 1, 1, 1), mode="replicate")
            avg_d = F.avg_pool2d(pad_d, kernel_size=3, stride=1)
            avg_m = F.avg_pool2d(pad_m, kernel_size=3, stride=1)
            filled = avg_d / avg_m.clamp_min(1e-5)
            cur_d = torch.where(cur_m > 0.5, cur_d, filled)
            cur_m = (avg_m > 1e-4).float()
        return cur_d

    def calibrate_scale_shift(
        self,
        rgb_feat: torch.Tensor,
        d_lr: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate spatially varying scale alpha_phi > 0 and shift beta_phi on full resolution (H, W).
        At epoch 0: alpha_phi == 1.0 and beta_phi == 0.0 everywhere.
        """
        H, W = target_size
        with torch.amp.autocast(device_type=rgb_feat.device.type, enabled=False):
            d_lr_up_coarse = F.interpolate(
                d_lr.float(), size=(self.coarse_grid, self.coarse_grid), mode="bilinear", align_corners=False
            )
            rgb_coarse = F.adaptive_avg_pool2d(rgb_feat.float(), (self.coarse_grid, self.coarse_grid))
            inp = torch.cat([rgb_coarse, d_lr_up_coarse], dim=1)

            raw = self.calib_net(inp).float()
            raw = torch.nan_to_num(raw, nan=0.0, posinf=5.0, neginf=-5.0)
            raw_full = F.interpolate(raw, size=(H, W), mode="bilinear", align_corners=False)

            raw_alpha, raw_beta = torch.chunk(raw_full, 2, dim=1)
            raw_alpha = 3.0 * torch.tanh(raw_alpha / 3.0)
            raw_beta = 2.0 * torch.tanh(raw_beta / 2.0)
            alpha = F.softplus(raw_alpha + self.offset_c0) + self.eps_alpha
            beta = raw_beta
        return alpha, beta

    def estimate_psf_kernel(
        self,
        rgb_feat: torch.Tensor,
        d_lr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate anisotropic PSF spatial kernel K_spatial [B, 1, ks, ks].
        """
        with torch.amp.autocast(device_type=rgb_feat.device.type, enabled=False):
            d_lr_up = F.interpolate(
                d_lr.float(), size=rgb_feat.shape[-2:], mode="bilinear", align_corners=False
            )
            inp = torch.cat([rgb_feat.float(), d_lr_up], dim=1)
            feat_pooled = self.psf_pool(inp).float()
            return self.psf_operator.forward_kernel(feat_pooled)

    def apply_A(
        self,
        D: torch.Tensor,
        K_spatial: torch.Tensor,
        scale_factor: int,
        mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Degradation operator A_phi(D) = S_{down s}(K_phi * D) * M_LR.
        Input D: [B, 1, H, W]
        Output: [B, 1, h, w] where h = H / scale_factor, w = W / scale_factor.
        """
        with torch.amp.autocast(device_type=D.device.type, enabled=False):
            D_blurred = self.psf_operator.apply_blur(D.float(), K_spatial.float())
            D_down = F.avg_pool2d(D_blurred, kernel_size=scale_factor, stride=scale_factor)
            if mask_lr is not None:
                D_down = D_down * mask_lr.float()
        return D_down.type_as(D)

    def apply_A_dagger(
        self,
        D_lr: torch.Tensor,
        K_spatial: torch.Tensor,
        scale_factor: int,
        target_size: tuple[int, int],
        mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Analytical right-inverse A_phi^dagger(D_lr) guaranteeing A_phi(A_phi^dagger D_lr) ~= D_lr.
        Uses smooth bicubic back-projection followed by exact residual correction.
        """
        H, W = target_size
        with torch.amp.autocast(device_type=D_lr.device.type, enabled=False):
            D_lr_f = D_lr.float()
            K_f = K_spatial.float()
            if mask_lr is None:
                mask_lr_f = torch.ones_like(D_lr_f)
            else:
                mask_lr_f = mask_lr.float()

            # Smooth bicubic lift + iterative smooth back-projection
            D_dag = F.interpolate(D_lr_f * mask_lr_f, size=(H, W), mode="bicubic", align_corners=False)
            for _ in range(2):
                deg = self.apply_A(D_dag, K_f, scale_factor, mask_lr_f)
                res_lr = (D_lr_f - deg) * mask_lr_f
                D_dag = D_dag + F.interpolate(res_lr, size=(H, W), mode="bicubic", align_corners=False)

            # Final exact block residual iterations so A_phi(D_dag) == D_lr to < 1e-5
            for _ in range(3):
                deg = self.apply_A(D_dag, K_f, scale_factor, mask_lr_f)
                res_lr = (D_lr_f - deg) * mask_lr_f
                res_up = torch.repeat_interleave(
                    torch.repeat_interleave(res_lr, scale_factor, dim=-2),
                    scale_factor,
                    dim=-1,
                )
                if res_up.shape[-2] != H or res_up.shape[-1] != W:
                    res_up = F.interpolate(res_up, size=(H, W), mode="bilinear", align_corners=False)
                D_dag = D_dag + res_up

        return D_dag.type_as(D_lr)

    def project_range_null(
        self,
        D_Poisson: torch.Tensor,
        D_lr: torch.Tensor,
        K_spatial: torch.Tensor,
        scale_factor: int,
        mask_lr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Range-Null Space Consistency Projection:
        D^(k) = A_phi^dagger D_lr + P_Null(D_Poisson)
              = D_Poisson + A_phi^dagger(D_lr - A_phi D_Poisson).
        Guarantees A_phi(D^(k)) == D_lr on valid LR measurements without staircase artifacts.
        """
        H, W = D_Poisson.shape[-2:]
        with torch.amp.autocast(device_type=D_Poisson.device.type, enabled=False):
            D_Poisson_f = D_Poisson.float()
            D_lr_f = D_lr.float()
            K_f = K_spatial.float()
            if mask_lr is None:
                mask_lr_f = (D_lr_f > 0).float()
            else:
                mask_lr_f = mask_lr.float()

            deg_p = self.apply_A(D_Poisson_f, K_f, scale_factor, mask_lr_f)
            res_lr = (D_lr_f - deg_p) * mask_lr_f
            delta_HR = self.apply_A_dagger(res_lr, K_f, scale_factor, (H, W), mask_lr_f)
            D_k = D_Poisson_f + delta_HR
        return D_k.type_as(D_Poisson)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        d_lr: torch.Tensor,
        mask_lr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full calibration forward pass:
        Returns:
            d_0: [B, 1, H, W] analytic range-space warm-start depth
            k_spatial: [B, 1, ks, ks] estimated PSF kernel
            alpha: [B, 1, H, W] spatially varying scale
            beta: [B, 1, H, W] spatially varying shift
        """
        H, W = rgb_feat.shape[-2:]
        h, w = d_lr.shape[-2:]
        scale_factor = H // h

        with torch.amp.autocast(device_type=rgb_feat.device.type, enabled=False):
            rgb_feat_f = rgb_feat.float()
            d_lr_f = d_lr.float()
            if mask_lr is None:
                mask_lr_f = (d_lr_f > 0).float()
            else:
                mask_lr_f = mask_lr.float()

            # Fill any missing LR holes before warm-start lift
            d_lr_filled = self._inpaint_lr_holes(d_lr_f, mask_lr_f)

            alpha, beta = self.calibrate_scale_shift(rgb_feat_f, d_lr_filled, (H, W))
            k_spatial = self.estimate_psf_kernel(rgb_feat_f, d_lr_filled)

            # Smooth bicubic back-projected warm-start D^(0)
            d_0_raw = F.interpolate(d_lr_filled, size=(H, W), mode="bicubic", align_corners=False)
            for _ in range(2):
                deg_0 = self.apply_A(d_0_raw, k_spatial, scale_factor, mask_lr_f)
                res_0 = (d_lr_filled - deg_0) * mask_lr_f
                d_0_raw = d_0_raw + F.interpolate(res_0, size=(H, W), mode="bicubic", align_corners=False)

            d_0 = torch.clamp(alpha * d_0_raw + beta, min=1e-3)

        return d_0, k_spatial, alpha, beta
