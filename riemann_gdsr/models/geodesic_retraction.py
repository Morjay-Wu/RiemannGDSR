"""
Closed-Form Geodesic Exponential Retraction on S^2_+
For RiemannGDSR (CVPR 2026).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicExponentialRetraction(nn.Module):
    """
    Closed-form geodesic exponential retraction of unit surface normals n^(k) in S^2_+:
        Exp_n(v_perp) = cos(theta) * n + sin(theta) / ||v_perp|| * v_perp
    Enforcing bounded angle theta <= pi/3 ensures:
    1. Visible positive hemisphere n_z > 0 is strictly preserved.
    2. Zero singularity or gradient explosion across unrolling steps.
    3. Exact unit-sphere manifold invariant ||n^(k)||_2 == 1.0 +- 1e-5.
    """

    def __init__(self, max_angle: float = math.pi / 3.0, eps: float = 1e-8):
        super().__init__()
        self.theta_max = max_angle
        self.eps = eps

    def forward(
        self,
        n_prev: torch.Tensor,
        v_perp: torch.Tensor,
    ) -> torch.Tensor:
        """
        n_prev: [B, 3, H, W] in S^2_+ (||n_prev||_2 == 1.0, n_z > 0)
        v_perp: [B, 3, H, W] in T_{n_prev} S^2 (<v_perp, n_prev> == 0)

        Returns:
            n_next: [B, 3, H, W] in S^2_+ satisfying ||n_next||_2 == 1.0 +- 1e-6 and n_z > 0.
        """
        # Execute retraction in FP32 for numerical manifold precision
        with torch.amp.autocast(device_type=n_prev.device.type, enabled=False):
            n_prev_f32 = n_prev.float()
            v_perp_f32 = v_perp.float()

            # Orthogonal projection guard: enforce <v_perp, n_prev> == 0
            dot_guard = torch.sum(v_perp_f32 * n_prev_f32, dim=1, keepdim=True)
            v_perp_f32 = v_perp_f32 - dot_guard * n_prev_f32

            # Compute tangent update norm
            theta_raw = torch.sqrt(
                torch.sum(v_perp_f32**2, dim=1, keepdim=True) + 1e-10
            )

            # Smoothly bound angle into [0, theta_max)
            theta = self.theta_max * torch.tanh(theta_raw / self.theta_max)

            # Geodesic exponential map on 2-sphere
            sinc_term = torch.sin(theta) / theta_raw
            n_geo = torch.cos(theta) * n_prev_f32 + sinc_term * v_perp_f32

            # Positive hemisphere line-of-sight safeguard: ensure n_z > 0
            nz = torch.clamp(n_geo[:, 2:3], min=1e-4)
            n_geo = torch.cat([n_geo[:, :2], nz], dim=1)

            # Unit normal retraction
            norm = torch.sqrt(torch.sum(n_geo**2, dim=1, keepdim=True) + self.eps)
            n_next = n_geo / norm

        return n_next.type_as(n_prev)
