"""
Multi-Stage Riemannian Guided Depth Super-Resolution Loss
For RiemannGDSR (CVPR 2026).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GDSRLoss(nn.Module):
    """
    Comprehensive GDSR Loss:
    1. Multi-Stage Depth Loss: L1 + Gradient across all unrolling steps with discount gamma.
    2. Manifold Normal Cosine Loss: 1 - <n^(k), n_gt> on S^2_+.
    3. Range-Space Sensor Consistency Loss: ||A_phi D^(K) - D_LR||_1.
    4. Calibration Smoothness Loss: ||grad alpha_phi||_1 + ||grad beta_phi||_1.
    """

    def __init__(
        self,
        gamma: float = 0.85,
        lambda_norm: float = 0.2,
        lambda_range: float = 1.0,
        lambda_calib: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.lambda_norm = lambda_norm
        self.lambda_range = lambda_range
        self.lambda_calib = lambda_calib

    @staticmethod
    def compute_gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates L1 gradient difference loss in FP32.
        """
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_x = torch.sum(torch.abs(pred_dx - gt_dx) * mask_x) / (torch.sum(mask_x) + 1e-5)
        loss_y = torch.sum(torch.abs(pred_dy - gt_dy) * mask_y) / (torch.sum(mask_y) + 1e-5)
        return loss_x + loss_y

    @staticmethod
    def estimate_pseudo_gt_normal(depth: torch.Tensor) -> torch.Tensor:
        """
        Estimates surface normals from ground truth depth map via spatial finite differences in FP32.
        """
        with torch.amp.autocast(device_type=depth.device.type, enabled=False):
            depth_f = depth.float()
            pad_d = F.pad(depth_f, (1, 1, 1, 1), mode="replicate")
            dx = (pad_d[:, :, 1:-1, 2:] - pad_d[:, :, 1:-1, :-2]) * 0.5
            dy = (pad_d[:, :, 2:, 1:-1] - pad_d[:, :, :-2, 1:-1]) * 0.5

            # Normal vector n ~ (-dx, -dy, 1)
            n = torch.cat([-dx, -dy, torch.ones_like(dx)], dim=1)
            return F.normalize(n, p=2, dim=1, eps=1e-6)

    def forward(
        self,
        output_dict: dict,
        d_gt: torch.Tensor,
        d_lr: torch.Tensor,
        mask_gt: torch.Tensor | None = None,
        n_gt: torch.Tensor | None = None,
        sensor_calib: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        output_dict: Output of RiemannGDSRNet containing 'pred', 'intermediate_depths', 'normals', etc.
        d_gt: [B, 1, H, W] ground truth depth map.
        d_lr: [B, 1, h, w] low-resolution depth measurement.
        mask_gt: optional [B, 1, H, W] ground truth valid mask.
        n_gt: optional [B, 3, H, W] ground truth surface normal map.
        sensor_calib: optional SensorCalibration module to compute A_phi.
        """
        device = d_gt.device
        with torch.amp.autocast(device_type=device.type, enabled=False):
            d_gt_f = d_gt.float()
            d_lr_f = d_lr.float()

            if mask_gt is None:
                mask_gt_f = (d_gt_f > 0).float()
            else:
                mask_gt_f = mask_gt.float()

            if n_gt is None:
                n_gt_f = self.estimate_pseudo_gt_normal(d_gt_f)
            else:
                n_gt_f = n_gt.float()

            depths = [d.float() for d in output_dict["intermediate_depths"]]
            normals = [n.float() for n in output_dict["normals"]]
            num_stages = len(depths)  # K + 1

            # 1. Multi-Stage Depth Loss
            loss_depth = torch.tensor(0.0, device=device, dtype=torch.float32)
            for k, d_k in enumerate(depths):
                weight = self.gamma ** (num_stages - 1 - k)
                l1_diff = torch.sum(torch.abs(d_k - d_gt_f) * mask_gt_f) / (torch.sum(mask_gt_f) + 1e-5)
                grad_diff = self.compute_gradient_loss(d_k, d_gt_f, mask_gt_f)
                loss_depth = loss_depth + weight * (l1_diff + 0.5 * grad_diff)

            # 2. Manifold Normal Cosine Loss
            loss_norm = torch.tensor(0.0, device=device, dtype=torch.float32)
            for k, n_k in enumerate(normals[1:], start=1):
                weight = self.gamma ** (num_stages - 1 - k)
                dot_norm = torch.sum(n_k * n_gt_f, dim=1, keepdim=True)
                cosine_err = 1.0 - torch.clamp(dot_norm, -1.0, 1.0)
                stage_norm_loss = torch.sum(cosine_err * mask_gt_f) / (torch.sum(mask_gt_f) + 1e-5)
                loss_norm = loss_norm + weight * stage_norm_loss

            # 3. Range-Space Sensor Consistency Loss
            loss_range = torch.tensor(0.0, device=device, dtype=torch.float32)
            k_spatial = output_dict.get("k_spatial", None)
            d_pred = output_dict["pred"].float()
            H, W = d_gt_f.shape[-2:]
            h, w = d_lr_f.shape[-2:]
            scale_factor = H // h
            mask_lr = (d_lr_f > 0).float()

            if sensor_calib is not None and k_spatial is not None:
                d_deg = sensor_calib.apply_A(d_pred, k_spatial.float(), scale_factor, mask_lr)
                loss_range = torch.sum(torch.abs(d_deg - d_lr_f) * mask_lr) / (torch.sum(mask_lr) + 1e-5)
            else:
                # Fallback to downsampled L1 difference
                d_down = F.avg_pool2d(d_pred, kernel_size=scale_factor, stride=scale_factor)
                loss_range = torch.sum(torch.abs(d_down - d_lr_f) * mask_lr) / (torch.sum(mask_lr) + 1e-5)

            # 4. Calibration Smoothness Loss
            alpha = output_dict.get("alpha", None)
            beta = output_dict.get("beta", None)
            loss_calib = torch.tensor(0.0, device=device, dtype=torch.float32)
            if alpha is not None and beta is not None:
                alpha_f = alpha.float()
                beta_f = beta.float()
                d_alpha_x = torch.abs(alpha_f[:, :, :, 1:] - alpha_f[:, :, :, :-1]).mean()
                d_alpha_y = torch.abs(alpha_f[:, :, 1:, :] - alpha_f[:, :, :-1, :]).mean()
                d_beta_x = torch.abs(beta_f[:, :, :, 1:] - beta_f[:, :, :, :-1]).mean()
                d_beta_y = torch.abs(beta_f[:, :, 1:, :] - beta_f[:, :, :-1, :]).mean()
                loss_calib = d_alpha_x + d_alpha_y + d_beta_x + d_beta_y

            total_loss = (
                loss_depth
                + self.lambda_norm * loss_norm
                + self.lambda_range * loss_range
                + self.lambda_calib * loss_calib
            )

        loss_breakdown = {
            "loss_total": total_loss,
            "loss_depth": loss_depth,
            "loss_norm": loss_norm,
            "loss_range": loss_range,
            "loss_calib": loss_calib,
        }
        return total_loss, loss_breakdown
