"""Base RGB-Guided Depth Super-Resolution (GDSR) Dataset with Synthetic Fallback.

Supports:
- Scale factors x4, x8, x16.
- Arbitrary rectangular resolutions (default 480x640 for NYUv2/RGB-D-D).
- Safe file formats: .png, .jpg, .npy, .npz, .pt (No h5py dependency).
- Sensor degradation simulation: downsampling, optional Gaussian sensor noise & missing depth holes.
- Built-in authentic procedural RGB-D scene generation when data is absent.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import glob
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


def load_rgb_tensor(path: str, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Load an RGB image and return [3, H, W] float in [0, 1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        img = Image.open(path).convert('RGB')
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    elif ext == '.npy':
        arr = np.load(path).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if target_size is not None and (tensor.shape[1], tensor.shape[2]) != target_size:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    elif ext == '.pt':
        tensor = torch.load(path, map_location='cpu').float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if target_size is not None and (tensor.shape[1], tensor.shape[2]) != target_size:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    else:
        raise ValueError(f"Unsupported RGB image format: {ext}")

    return tensor.clamp(0.0, 1.0)


def load_depth_tensor(path: str, target_size: Optional[Tuple[int, int]] = None, depth_scale: float = 1.0) -> torch.Tensor:
    """Load depth map and return [1, H, W] float depth in meters.
    
    depth_scale: Conversion factor to meters (e.g. 1/1000.0 for 16-bit mm PNG).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.png', '.bmp']:
        depth_img = Image.open(path)
        arr = np.array(depth_img, dtype=np.float32) * depth_scale
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if target_size is not None and (tensor.shape[1], tensor.shape[2]) != target_size:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
    elif ext == '.npy':
        arr = np.load(path).astype(np.float32) * depth_scale
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if target_size is not None and (tensor.shape[1], tensor.shape[2]) != target_size:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
    elif ext == '.pt':
        tensor = torch.load(path, map_location='cpu').float() * depth_scale
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if target_size is not None and (tensor.shape[1], tensor.shape[2]) != target_size:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
    else:
        raise ValueError(f"Unsupported depth format: {ext}")

    return tensor


class BaseGDSRDataset(Dataset):
    """Base RGB-Guided Depth Super-Resolution dataset.
    
    Args:
        root_dir: Path to dataset root directory.
        split: 'train', 'val', or 'test'.
        scale: Super-resolution scale factor (4, 8, 16).
        img_size: Rectangular target resolution (H, W), default (480, 640).
        synthetic: Force synthetic mode if True, or fallback to synthetic if root_dir missing.
        depth_scale: Scaling factor to convert raw depth to meters.
        synthetic_samples: Number of virtual samples when operating in synthetic mode.
        noise_std: Optional Gaussian sensor noise standard deviation (default 0.0 for standard clean protocol).
        hole_ratio: Optional random missing hole ratio (default 0.0 for standard clean protocol).
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = 'train',
        scale: int = 4,
        img_size: Tuple[int, int] = (480, 640),
        synthetic: bool = False,
        depth_scale: float = 1.0,
        synthetic_samples: int = 100,
        noise_std: float = 0.0,
        hole_ratio: float = 0.0,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.scale = scale
        self.img_size = img_size  # (H, W)
        self.depth_scale = depth_scale
        self.synthetic_samples = synthetic_samples
        self.noise_std = noise_std
        self.hole_ratio = hole_ratio

        self.samples: List[Dict[str, str]] = []
        if (not synthetic) and (root_dir is not None) and os.path.isdir(root_dir):
            self.samples = self._load_sample_list()

        self.is_synthetic = synthetic or len(self.samples) == 0

    def _load_sample_list(self) -> List[Dict[str, str]]:
        """Subclasses should implement real data indexing.
        Returns list of dicts with 'rgb_path', 'depth_path', optional 'lr_depth_path'.
        """
        return []

    def __len__(self) -> int:
        if self.is_synthetic:
            return self.synthetic_samples
        return len(self.samples)

    def _simulate_degradation(
        self,
        d_gt: torch.Tensor,
        noise_std: Optional[float] = None,
        hole_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate low-resolution sensor measurement from high-resolution ground truth.
        
        Defaults to clean downsampling (noise_std=0.0, hole_ratio=0.0) matching standard
        GDSR benchmarks (NYUv2, Middlebury, Lu).
        
        Args:
            d_gt: High-resolution depth map [1, H, W]
        Returns:
            d_lr: Low-resolution degraded depth [1, H//scale, W//scale]
            mask_lr: Low-resolution sensor validity mask [1, H//scale, W//scale]
        """
        if noise_std is None:
            noise_std = self.noise_std
        if hole_ratio is None:
            hole_ratio = self.hole_ratio

        # 1. Downsampling via average pooling over valid pixels
        mask_gt = (d_gt > 0.001).float()
        d_valid = d_gt * mask_gt

        sum_lr = F.avg_pool2d(d_valid.unsqueeze(0), kernel_size=self.scale, stride=self.scale)
        count_lr = F.avg_pool2d(mask_gt.unsqueeze(0), kernel_size=self.scale, stride=self.scale)

        d_lr = (sum_lr / (count_lr + 1e-6)).squeeze(0)
        mask_lr = (count_lr > 0.01).float().squeeze(0)
        d_lr = d_lr * mask_lr

        # 2. Optional realistic sensor noise (for robustness evaluation)
        if noise_std > 0:
            noise = torch.randn_like(d_lr) * noise_std * (1.0 + 0.1 * (d_lr ** 2))
            d_lr = torch.clamp(d_lr + noise, min=0.05)

        # 3. Optional sensor dropouts (for hole-completion evaluation)
        if hole_ratio > 0:
            dropout_mask = (torch.rand_like(d_lr) > hole_ratio).float()
            mask_lr = mask_lr * dropout_mask
            d_lr = d_lr * mask_lr

        return d_lr, mask_lr

    def _generate_synthetic_rgbd(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Procedurally generate authentic indoor RGB-D scene.
        
        Guarantees:
        - Rectangular (H, W).
        - Multi-planar 3D geometry (floor plane, walls, foreground objects, fine poles).
        - Strict alignment between RGB structural edges and depth jump discontinuities.
        - Realistic low-resolution depth measurement at scale factor x4, x8, x16.
        """
        rng = np.random.RandomState(idx + (0 if self.split == 'train' else 54321))
        H, W = self.img_size

        # Coordinate grid in [-1, 1]
        y_g, x_g = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')

        # 1. Base room geometry: Slanted floor plane and back wall
        depth = np.ones((H, W), dtype=np.float32) * 3.5
        floor_mask = y_g > 0.0
        depth[floor_mask] = 3.5 - 2.0 * y_g[floor_mask]

        left_mask = (x_g < -0.4) & (~floor_mask)
        depth[left_mask] = 3.5 + 1.5 * (x_g[left_mask] + 0.4)

        # 2. RGB image base
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[floor_mask, 0] = 0.55 - 0.15 * y_g[floor_mask]
        rgb[floor_mask, 1] = 0.40 - 0.10 * y_g[floor_mask]
        rgb[floor_mask, 2] = 0.25 - 0.08 * y_g[floor_mask]

        rgb[~floor_mask, 0] = 0.80
        rgb[~floor_mask, 1] = 0.82
        rgb[~floor_mask, 2] = 0.85

        # 3. Add foreground geometric objects (boxes, spheres, fine structures)
        num_objects = rng.randint(2, 5)
        for _ in range(num_objects):
            ox = rng.uniform(-0.6, 0.6)
            oy = rng.uniform(0.0, 0.6)
            ow = rng.uniform(0.15, 0.35)
            oh = rng.uniform(0.15, 0.40)
            obj_depth = rng.uniform(1.2, 2.2)
            obj_color = rng.uniform(0.1, 0.9, size=3).astype(np.float32)

            box_mask = (x_g >= ox - ow/2) & (x_g <= ox + ow/2) & (y_g >= oy - oh/2) & (y_g <= oy + oh/2)
            depth[box_mask] = obj_depth
            rgb[box_mask] = obj_color

        # 4. Add fine structures (thin wire/pole, Middlebury-style)
        px = rng.uniform(-0.5, 0.5)
        pole_mask = (np.abs(x_g - px) < 0.015) & (y_g > -0.7) & (y_g < 0.7)
        depth[pole_mask] = 1.0
        rgb[pole_mask] = np.array([0.9, 0.1, 0.1], dtype=np.float32)

        # Convert to PyTorch tensors
        t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).float().clamp(0.0, 1.0)
        t_dgt = torch.from_numpy(depth).unsqueeze(0).float()
        mask_gt = (t_dgt > 0.01).float()

        # Simulate low-resolution degradation
        t_dlr, mask_lr = self._simulate_degradation(t_dgt)

        return t_rgb, t_dgt, t_dlr, mask_gt, mask_lr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.is_synthetic:
            rgb, d_gt, d_lr, mask_gt, mask_lr = self._generate_synthetic_rgbd(idx)
            meta = {"index": idx, "is_synthetic": True, "scale": self.scale}
        else:
            sample = self.samples[idx]
            rgb = load_rgb_tensor(sample['rgb_path'], self.img_size)
            d_gt = load_depth_tensor(sample['depth_path'], self.img_size, self.depth_scale)
            mask_gt = (d_gt > 0.001).float()

            if 'lr_depth_path' in sample and os.path.isfile(sample['lr_depth_path']):
                lr_size = (self.img_size[0] // self.scale, self.img_size[1] // self.scale)
                d_lr = load_depth_tensor(sample['lr_depth_path'], lr_size, self.depth_scale)
                mask_lr = (d_lr > 0.001).float()
            else:
                d_lr, mask_lr = self._simulate_degradation(d_gt)

            meta = {
                "index": idx,
                "is_synthetic": False,
                "scale": self.scale,
                "rgb_path": sample['rgb_path'],
                "depth_path": sample['depth_path'],
            }

        return {
            "rgb": rgb,
            "d_gt": d_gt,
            "d_lr": d_lr,
            "mask_gt": mask_gt,
            "mask_lr": mask_lr,
            "meta": meta,
        }
