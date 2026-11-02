"""RGB-Guided Depth Super-Resolution (GDSR) Datasets Module for RiemannGDSR."""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from .base_gdsr import BaseGDSRDataset
from .nyuv2 import NYUv2Dataset
from .middlebury import MiddleburyDataset
from .rgbdd import RGBDDDataset


def get_gdsr_dataset(
    name: str = 'synthetic',
    root_dir: Optional[str] = None,
    split: str = 'train',
    scale: int = 4,
    img_size: Tuple[int, int] = (480, 640),
    synthetic: bool = False,
    depth_scale: float = 1.0,
    synthetic_samples: int = 100,
    **kwargs,
) -> Dataset:
    """Factory function to instantiate GDSR datasets."""
    name_clean = name.lower().replace('-', '_')
    if name_clean in ['nyuv2', 'nyu', 'nyu_v2']:
        return NYUv2Dataset(
            root_dir=root_dir,
            split=split,
            scale=scale,
            img_size=img_size,
            synthetic=synthetic,
            synthetic_samples=synthetic_samples,
            **kwargs,
        )
    elif name_clean in ['middlebury', 'mid']:
        return MiddleburyDataset(
            root_dir=root_dir,
            split=split,
            scale=scale,
            img_size=img_size,
            synthetic=synthetic,
            synthetic_samples=synthetic_samples,
            **kwargs,
        )
    elif name_clean in ['lu', 'lu_depth']:
        return NYUv2Dataset(
            root_dir=root_dir,
            split=split,
            scale=scale,
            img_size=img_size,
            synthetic=synthetic,
            depth_scale=1.0 / 25.5,
            synthetic_samples=synthetic_samples,
            **kwargs,
        )
    elif name_clean in ['rgbdd', 'rgb_d_d', 'rgb_dd']:
        return RGBDDDataset(
            root_dir=root_dir,
            split=split,
            scale=scale,
            img_size=img_size,
            synthetic=synthetic,
            synthetic_samples=synthetic_samples,
            **kwargs,
        )
    elif name_clean in ['synthetic', 'synth', 'demo']:
        return BaseGDSRDataset(
            root_dir=None,
            split=split,
            scale=scale,
            img_size=img_size,
            synthetic=True,
            depth_scale=depth_scale,
            synthetic_samples=synthetic_samples,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown GDSR dataset: {name}. Available: 'nyuv2', 'middlebury', 'lu', 'rgbdd', 'synthetic'")


def build_gdsr_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Build PyTorch DataLoader with safe defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


__all__ = [
    "BaseGDSRDataset",
    "NYUv2Dataset",
    "MiddleburyDataset",
    "RGBDDDataset",
    "get_gdsr_dataset",
    "build_gdsr_dataloader",
]
