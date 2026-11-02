"""Middlebury: Fine-Structure RGB-D Super-Resolution Benchmark Loader.

Reference:
"A Database and Evaluation Methodology for Stereo Correspondence"
High-resolution indoor scenes with thin structures, wireframes, and sharp discontinuities.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import glob
from .base_gdsr import BaseGDSRDataset


class MiddleburyDataset(BaseGDSRDataset):
    """Middlebury benchmark loader with intricate geometry and thin depth structures.
    
    Expected Layout:
        root_dir/
            scene_name/ (e.g. Art, Books, Dolls, Laundry, Moebius, Reindeer)
                view1.png (or im0.png, rgb.png)
                disp1.png (or disp0.png, depth.png, depth.npy)
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = 'test',
        scale: int = 4,
        img_size: Tuple[int, int] = (480, 640),
        synthetic: bool = False,
        depth_scale: float = 1.0 / 25.5,
        synthetic_samples: int = 50,
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
        scene_dirs = sorted(glob.glob(os.path.join(self.root_dir, '*')))

        for s_dir in scene_dirs:
            if not os.path.isdir(s_dir):
                continue

            # Look for rgb file
            rgb_candidates = ['view1.png', 'im0.png', 'rgb.png', 'image.png', 'color.png']
            depth_candidates = ['disp1.png', 'disp0.png', 'depth.png', 'disp1.npy', 'depth.npy']

            rgb_path = None
            depth_path = None

            for c in rgb_candidates:
                p = os.path.join(s_dir, c)
                if os.path.isfile(p):
                    rgb_path = p
                    break

            for c in depth_candidates:
                p = os.path.join(s_dir, c)
                if os.path.isfile(p):
                    depth_path = p
                    break

            # If not matched directly, glob inside s_dir
            if not rgb_path:
                matches = glob.glob(os.path.join(s_dir, '*rgb*')) + glob.glob(os.path.join(s_dir, '*im0*'))
                if matches:
                    rgb_path = matches[0]

            if not depth_path:
                matches = glob.glob(os.path.join(s_dir, '*disp*')) + glob.glob(os.path.join(s_dir, '*depth*'))
                if matches:
                    depth_path = matches[0]

            if rgb_path and depth_path:
                samples.append({
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                })

        return samples
