"""
RiemannGDSR: RGB-Guided Depth Super-Resolution & Completion
via Monocular-Prior Self-Calibration and Spectral Screened-Poisson S^2 Geodesic Unrolling.
CVPR 2026.
"""

from __future__ import annotations
from .models.riemann_gdsr import RiemannGDSRNet, RiemannGDSRConfig
from .losses.gdsr_loss import GDSRLoss

__all__ = [
    "RiemannGDSRNet",
    "RiemannGDSRConfig",
    "GDSRLoss",
]
