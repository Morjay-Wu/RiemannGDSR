"""RGB-D-D: Real-World Sensor Degradation RGB-D Benchmark Loader.

Reference:
"RGB-D-D: A Dataset for RGB-D Sensor Degradation and Depth Restoration"
Real-world paired low-resolution/degraded depth sensor inputs (ToF/Structured Light)
with high-precision ground truth depth maps.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import glob
from .base_gdsr import BaseGDSRDataset


class RGBDDDataset(BaseGDSRDataset):
    """RGB-D-D real sensor degradation depth super-resolution benchmark loader.
    
    Expected Layout:
        root_dir/
            train/ (or val/ or test/)
                rgb/
                depth_lr/ (or tof/, low_res_depth/)
                depth_gt/ (or gt/, high_res_depth/)
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = 'train',
        scale: int = 4,
        img_size: Tuple[int, int] = (480, 640),
        synthetic: bool = False,
        depth_scale: float = 1.0 / 1000.0,
        synthetic_samples: int = 100,
    ):
        super().__init__(
            root_dir=root_dir,
            split=split,
            scale=scale,
            img_size=img_size,
            synthetic=synthetic,
            depth_scale=depth_scale,
            synthetic_samples=synthetic_samples,
        )

    def _load_sample_list(self) -> List[Dict[str, str]]:
        if not self.root_dir or not os.path.isdir(self.root_dir):
            return []

        samples = []
        search_dirs = [os.path.join(self.root_dir, self.split), self.root_dir]

        for s_dir in search_dirs:
            if not os.path.isdir(s_dir):
                continue

            rgb_dir = None
            depth_gt_dir = None
            depth_lr_dir = None

            for c in ['rgb', 'color', 'image']:
                p = os.path.join(s_dir, c)
                if os.path.isdir(p):
                    rgb_dir = p
                    break

            for c in ['depth_gt', 'gt', 'gt_depth', 'depth_high']:
                p = os.path.join(s_dir, c)
                if os.path.isdir(p):
                    depth_gt_dir = p
                    break

            for c in ['depth_lr', 'lr', 'tof', 'depth_low', 'raw_depth']:
                p = os.path.join(s_dir, c)
                if os.path.isdir(p):
                    depth_lr_dir = p
                    break

            if rgb_dir and depth_gt_dir:
                exts = ('*.png', '*.jpg', '*.npy', '*.pt')
                rgb_files = []
                for ext in exts:
                    rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

                for f_rgb in sorted(rgb_files):
                    base = os.path.basename(f_rgb)
                    name_no_ext = os.path.splitext(base)[0]

                    f_gt = os.path.join(depth_gt_dir, base)
                    if not os.path.isfile(f_gt):
                        matches = glob.glob(os.path.join(depth_gt_dir, f"{name_no_ext}.*"))
                        if matches:
                            f_gt = matches[0]

                    f_lr = None
                    if depth_lr_dir:
                        cand_lr = os.path.join(depth_lr_dir, base)
                        if os.path.isfile(cand_lr):
                            f_lr = cand_lr
                        else:
                            matches_lr = glob.glob(os.path.join(depth_lr_dir, f"{name_no_ext}.*"))
                            if matches_lr:
                                f_lr = matches_lr[0]

                    if os.path.isfile(f_gt):
                        item = {
                            'rgb_path': f_rgb,
                            'depth_path': f_gt,
                        }
                        if f_lr and os.path.isfile(f_lr):
                            item['lr_depth_path'] = f_lr
                        samples.append(item)

        return samples
