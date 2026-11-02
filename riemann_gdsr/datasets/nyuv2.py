"""NYU Depth v2: Indoor RGB-D Benchmark Loader for RiemannGDSR.

Reference:
"Indoor Segmentation and Support Inference from RGBD Images" (ECCV 2012)
Standard RGB-guided depth super-resolution benchmark (480x640 resolution).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import glob
from .base_gdsr import BaseGDSRDataset


class NYUv2Dataset(BaseGDSRDataset):
    """NYUv2 indoor RGB-D depth super-resolution dataset loader.
    
    Expected Layout (one of the following):
    Structure A:
        root_dir/
            train/ (or val/ or test/)
                rgb/
                    0001.png, 0002.png, ...
                depth/
                    0001.png, 0002.png, ...
    Structure B:
        root_dir/
            nyu_depth_v2_labeled.mat or pre-extracted pngs
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = 'train',
        scale: int = 4,
        img_size: Tuple[int, int] = (480, 640),
        synthetic: bool = False,
        depth_scale: float = 1.0 / 1000.0,  # mm to meters if 16-bit PNG
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
            depth_dir = None

            for c in ['rgb', 'image', 'images', 'color']:
                p = os.path.join(s_dir, c)
                if os.path.isdir(p):
                    rgb_dir = p
                    break

            for c in ['depth', 'depth_raw', 'sync_depth', 'gt_depth']:
                p = os.path.join(s_dir, c)
                if os.path.isdir(p):
                    depth_dir = p
                    break

            if rgb_dir and depth_dir:
                exts = ('*.png', '*.jpg', '*.npy', '*.pt')
                rgb_files = []
                for ext in exts:
                    rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

                for f_rgb in sorted(rgb_files):
                    base = os.path.basename(f_rgb)
                    name_no_ext = os.path.splitext(base)[0]

                    # Match depth file
                    f_depth = os.path.join(depth_dir, base)
                    if not os.path.isfile(f_depth):
                        matches = glob.glob(os.path.join(depth_dir, f"{name_no_ext}.*"))
                        if matches:
                            f_depth = matches[0]

                    if os.path.isfile(f_depth):
                        samples.append({
                            'rgb_path': f_rgb,
                            'depth_path': f_depth,
                        })

        return samples
