"""
Line-of-Sight Insulated Tangent-Space Parallel Transport on S^2
For RiemannGDSR (CVPR 2026).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TangentSpaceProjection(nn.Module):
    """
    Strict orthogonal projection of unconstrained innovation v onto the tangent space
    T_n S^2 = { v in R^3 | <v, n> = 0 }.
    """

    def __init__(self):
        super().__init__()

    def forward(self, v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        v: [B, 3, H, W]
        n: [B, 3, H, W] (unit normal, ||n||_2 = 1.0)
        Returns: v_perp strictly satisfying <v_perp, n> == 0.
        """
        with torch.amp.autocast(device_type=v.device.type, enabled=False):
            v_f = torch.clamp(torch.nan_to_num(v.float(), nan=0.0, posinf=10.0, neginf=-10.0), min=-10.0, max=10.0)
            n_f = n.float()
            dot = torch.sum(v_f * n_f, dim=1, keepdim=True)
            v_perp = v_f - dot * n_f
        return v_perp.type_as(v)


class LineOfSightBoundaryBarrier(nn.Module):
    """
    Computes a Line-of-Sight (LoS) insulated boundary barrier from high-resolution RGB
    structural edges and anchor depth discontinuities, preventing tangent gradient bleeding
    across occlusion silhouettes.
    """

    def __init__(self, feat_dim: int = 48):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Conv2d(3 + 1 + 3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.sigma_i = nn.Parameter(torch.tensor(0.5))
        self.sigma_n = nn.Parameter(torch.tensor(0.5))
        self.weight_barrier = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        rgb: torch.Tensor,
        d_anchor: torch.Tensor,
        n_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes LoS boundary barrier B_LoS in [0, 1] [B, 1, H, W].
        """
        inp = torch.cat([rgb, d_anchor, n_prev], dim=1)
        b_los = self.edge_net(inp)
        return b_los


class LeviCivitaParallelTransport(nn.Module):
    """
    Levi-Civita parallel transport on S^2 along geodesics n(q) -> n(p),
    insulated by Line-of-Sight boundary barrier G(p, q).
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        self.proj = TangentSpaceProjection()
        self.barrier_estimator = LineOfSightBoundaryBarrier()

        # Learnable blending weight for transport aggregation (initialized small so epoch 0 is stable)
        self.logit_w = nn.Parameter(torch.tensor(-2.0))

        # Local refinement MLP
        self.refine_mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )
        nn.init.zeros_(self.refine_mlp[-1].weight)
        nn.init.zeros_(self.refine_mlp[-1].bias)

    def forward(
        self,
        v_perp: torch.Tensor,
        n_prev: torch.Tensor,
        rgb: torch.Tensor,
        d_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        v_perp: [B, 3, H, W] in T_{n_prev} S^2
        n_prev: [B, 3, H, W] unit surface normals
        rgb: [B, 3, H, W] guidance image
        d_anchor: [B, 1, H, W] current anchor depth

        Returns:
            v_transported: [B, 3, H, W] in T_{n_prev} S^2
        """
        B, C, H, W = v_perp.shape
        device = v_perp.device

        with torch.amp.autocast(device_type=device.type, enabled=False):
            v_perp_f = torch.nan_to_num(v_perp.float(), nan=0.0, posinf=10.0, neginf=-10.0)
            n_prev_f = n_prev.float()
            rgb_f = rgb.float()
            d_anchor_f = torch.clamp(torch.nan_to_num(d_anchor.float(), nan=1.0), min=0.05, max=20.0)

            # 1. Compute boundary barrier B_los
            b_los = self.barrier_estimator(rgb_f, d_anchor_f, n_prev_f).float()

            # 2. Extract 3x3 local neighborhood for n_prev, v_perp, rgb, and b_los
            # [B, 3, 9, H, W]
            n_nb = F.unfold(n_prev_f, kernel_size=3, padding=1).view(B, 3, 9, H, W)
            v_nb = F.unfold(v_perp_f, kernel_size=3, padding=1).view(B, 3, 9, H, W)
            rgb_nb = F.unfold(rgb_f, kernel_size=3, padding=1).view(B, 3, 9, H, W)
            b_los_nb = F.unfold(b_los, kernel_size=3, padding=1).view(B, 1, 9, H, W)

            n_center = n_prev_f.unsqueeze(2)  # [B, 3, 1, H, W]
            rgb_center = rgb_f.unsqueeze(2)  # [B, 3, 1, H, W]

            # 3. Compute insulated spatial conductance G(p, q) over 9 neighbors
            # Photometric affinity
            diff_rgb_sq = torch.sum((rgb_center - rgb_nb) ** 2, dim=1, keepdim=True)  # [B, 1, 9, H, W]
            sigma_i_sq = F.softplus(self.barrier_estimator.sigma_i.float()) ** 2 + 1e-3

            # Geometric normal affinity
            dot_n = torch.sum(n_center * n_nb, dim=1, keepdim=True)  # [B, 1, 9, H, W]
            sigma_n_sq = F.softplus(self.barrier_estimator.sigma_n.float()) ** 2 + 1e-3

            # Barrier weight
            w_b = F.softplus(self.barrier_estimator.weight_barrier.float())

            logits = (
                -diff_rgb_sq / sigma_i_sq
                - (1.0 - torch.clamp(dot_n, -1.0, 1.0)) / sigma_n_sq
                - w_b * b_los_nb
            )  # [B, 1, 9, H, W]
            logits = torch.clamp(logits, min=-30.0, max=30.0)

            conductance = F.softmax(logits, dim=2)  # [B, 1, 9, H, W], sums to 1 over dim=2

            # 4. Vectorized Levi-Civita Parallel Transport for each neighbor q in 3x3
            # T_{n(q) -> n(p)}(v(q)) = v(q) - <n(p), v(q)> / (1 + <n(q), n(p)>) * (n(q) + n(p))
            dot_np_vq = torch.sum(n_center * v_nb, dim=1, keepdim=True)  # [B, 1, 9, H, W]
            denom = torch.clamp(1.0 + dot_n, min=1e-2)  # [B, 1, 9, H, W]
            sum_n = n_nb + n_center  # [B, 3, 9, H, W]

            v_trans_nb = v_nb - (dot_np_vq / denom) * sum_n  # [B, 3, 9, H, W]

            # 5. Weighted aggregation across 3x3 patch
            v_agg = torch.sum(conductance * v_trans_nb, dim=2)  # [B, 3, H, W]

            # 6. Blending with local innovation and refinement
            w_trans = torch.sigmoid(self.logit_w.float())
            v_blended = (1.0 - w_trans) * v_perp_f + w_trans * v_agg + self.refine_mlp(v_perp_f).float()

            # 7. Final orthogonal projection to guarantee <v_transported, n_prev> == 0 strictly
            v_out = self.proj(v_blended, n_prev_f)

        return v_out.type_as(v_perp)
