"""Production Depth Super-Resolution & Completion Evaluation Metrics.

Computes standard CVPR RGB-D benchmarks metrics:
- RMSE: Root Mean Squared Error (m / mm)
- MAE: Mean Absolute Error
- AbsRel: Absolute Relative Error (|D - D*| / D*)
- Log10: Mean log10 absolute error (|log10(D) - log10(D*)|)
- delta_1, delta_2, delta_3: Percentage of pixels satisfying max(D/D*, D*/D) < 1.25^p
"""

from __future__ import annotations
from typing import Dict, Union, Optional
import torch
import numpy as np


class GDSRMetrics:
    """Running evaluator for Guided Depth Super-Resolution benchmarks.
    
    Accumulates errors across mini-batches and returns global dataset metrics.
    """

    def __init__(self, eps: float = 1e-7, min_depth: float = 1e-3, max_depth: float = 10.0):
        self.eps = eps
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        """Reset running error accumulators."""
        self.total_sq_error = 0.0
        self.total_abs_error = 0.0
        self.total_rel_error = 0.0
        self.total_log10_error = 0.0
        self.total_delta1 = 0.0
        self.total_delta2 = 0.0
        self.total_delta3 = 0.0
        self.total_pixels = 0
        self.total_samples = 0

    @torch.no_grad()
    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        gt: Union[torch.Tensor, np.ndarray],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """Update metrics with a batch of predicted and ground-truth depth maps.
        
        Args:
            pred: [B, 1, H, W] or [B, H, W] predicted depth.
            gt: [B, 1, H, W] or [B, H, W] ground-truth depth.
            mask: Optional [B, 1, H, W] boolean/float validity mask.
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)

        pred = pred.to(gt.device).float()
        gt = gt.float()

        # Valid mask: gt > min_depth and gt <= max_depth, and optional mask
        valid = (gt > self.min_depth) & (gt <= self.max_depth) & torch.isfinite(gt) & torch.isfinite(pred)
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(gt.device)
            valid = valid & (mask.float() > 0.5)

        if not valid.any():
            return

        pred_val = torch.clamp(pred[valid], min=self.min_depth, max=self.max_depth)
        gt_val = gt[valid]

        diff = pred_val - gt_val
        abs_diff = torch.abs(diff)
        sq_diff = diff ** 2

        rel_diff = abs_diff / (gt_val + self.eps)

        log_pred = torch.log10(pred_val + self.eps)
        log_gt = torch.log10(gt_val + self.eps)
        log10_diff = torch.abs(log_pred - log_gt)

        # Threshold delta metrics
        ratio = torch.max(pred_val / (gt_val + self.eps), gt_val / (pred_val + self.eps))
        delta1 = (ratio < 1.25).float()
        delta2 = (ratio < 1.25 ** 2).float()
        delta3 = (ratio < 1.25 ** 3).float()

        num_pts = pred_val.numel()
        self.total_sq_error += float(sq_diff.sum().item())
        self.total_abs_error += float(abs_diff.sum().item())
        self.total_rel_error += float(rel_diff.sum().item())
        self.total_log10_error += float(log10_diff.sum().item())
        self.total_delta1 += float(delta1.sum().item())
        self.total_delta2 += float(delta2.sum().item())
        self.total_delta3 += float(delta3.sum().item())
        self.total_pixels += num_pts
        self.total_samples += pred.shape[0] if pred.dim() >= 3 else 1

    def compute(self) -> Dict[str, float]:
        """Compute dataset-wide average metrics."""
        n = max(self.total_pixels, 1)
        rmse = float(np.sqrt(self.total_sq_error / n))
        mae = float(self.total_abs_error / n)
        abs_rel = float(self.total_rel_error / n)
        log10 = float(self.total_log10_error / n)
        d1 = float(self.total_delta1 / n)
        d2 = float(self.total_delta2 / n)
        d3 = float(self.total_delta3 / n)

        return {
            "rmse": rmse,
            "mae": mae,
            "abs_rel": abs_rel,
            "log10": log10,
            "delta_1": d1,
            "delta_2": d2,
            "delta_3": d3,
            "total_pixels": int(self.total_pixels),
            "total_samples": int(self.total_samples),
        }

    def summary_str(self) -> str:
        """Return formatted string summary of current metrics."""
        m = self.compute()
        return (
            f"RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | AbsRel: {m['abs_rel']:.4f} | "
            f"delta_1: {m['delta_1']*100:.2f}% | delta_2: {m['delta_2']*100:.2f}% | "
            f"delta_3: {m['delta_3']*100:.2f}%"
        )
