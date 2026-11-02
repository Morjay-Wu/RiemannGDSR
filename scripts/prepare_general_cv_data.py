"""Automated General Computer Vision Benchmark Dataset Preparation Utility.

Supports all 7 pure General CV (0% Remote Sensing) Benchmarks:
Scene Change Detection (SCD):
1. VL-CMU-CD (Street-view change under seasonal/viewpoint shift)
2. PCD / TSUNAMI / GSV (Panoramic and perspective street change)
3. ChangeSim (Embodied robot indoor warehouse/industrial change)

Guided Depth Super-Resolution (GDSR):
4. NYU Depth v2 (Indoor dense RGB-D benchmark)
5. Middlebury (High-resolution intricate fine-structures and thin wires)
6. Lu RGB-D (Object and indoor depth super-resolution)
7. RGB-D-D (Real-world sensor degradation, handheld ToF/Structured Light)

Features:
- `--synthetic_demo`: Generates directory layout and sample `.png`/`.npy` files for instant offline testing.
- `--download_urls`: Prints official benchmark download URLs, citations, and manual setup instructions.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

BENCHMARK_INFO = {
    "vl_cmu_cd": {
        "type": "SCD",
        "name": "VL-CMU-CD",
        "description": "Street View Change Detection with De-noising and View-Angle Alignment (CMU)",
        "url": "https://www.cs.cmu.edu/~tsunami/dataset.html",
        "layout": [
            "train/t0", "train/t1", "train/gt",
            "val/t0", "val/t1", "val/gt",
            "test/t0", "test/t1", "test/gt",
        ]
    },
    "pcd": {
        "type": "SCD",
        "name": "PCD (TSUNAMI & GSV)",
        "description": "Panoramic & Perspective Street Scene Change Detection",
        "url": "https://staff.aist.go.jp/k.sakarada/PCD_dataset/",
        "layout": [
            "TSUNAMI/t0", "TSUNAMI/t1", "TSUNAMI/groundtruth",
            "GSV/t0", "GSV/t1", "GSV/groundtruth",
        ]
    },
    "changesim": {
        "type": "SCD",
        "name": "ChangeSim",
        "description": "Embodied Robot Indoor Scene Change Detection in Industrial Environments",
        "url": "https://github.com/ChangeSim/ChangeSim",
        "layout": [
            "Seq_00/ref_rgb", "Seq_00/query_rgb", "Seq_00/change_mask",
            "Seq_01/ref_rgb", "Seq_01/query_rgb", "Seq_01/change_mask",
        ]
    },
    "nyuv2": {
        "type": "GDSR",
        "name": "NYU Depth v2",
        "description": "Indoor RGB-D Segmentation & Depth Super-Resolution Benchmark",
        "url": "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html",
        "layout": [
            "train/rgb", "train/depth",
            "val/rgb", "val/depth",
            "test/rgb", "test/depth",
        ]
    },
    "middlebury": {
        "type": "GDSR",
        "name": "Middlebury Stereo & Depth",
        "description": "High-Resolution Indoor Depth Benchmark with Thin Wireframes",
        "url": "https://vision.middlebury.edu/stereo/data/",
        "layout": [
            "Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer"
        ]
    },
    "lu": {
        "type": "GDSR",
        "name": "Lu RGB-D Benchmark",
        "description": "RGB-Guided Depth Enhancement and Upsampling",
        "url": "https://github.com/jingsenlu/RGB-D-D",
        "layout": [
            "rgb", "depth"
        ]
    },
    "rgbdd": {
        "type": "GDSR",
        "name": "RGB-D-D Real Sensor Degradation",
        "description": "Real-World Paired ToF / Structured-Light Sensor Degradation Benchmark",
        "url": "https://github.com/NJU-MCG/RGB-D-D",
        "layout": [
            "train/rgb", "train/depth_lr", "train/depth_gt",
            "val/rgb", "val/depth_lr", "val/depth_gt",
            "test/rgb", "test/depth_lr", "test/depth_gt",
        ]
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="General Computer Vision Benchmark Dataset Preparation & Download Utility"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "vl_cmu_cd", "pcd", "changesim", "nyuv2", "middlebury", "lu", "rgbdd"],
        help="Target benchmark dataset name (or 'all')"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root storage directory for datasets (default ./data)"
    )
    parser.add_argument(
        "--synthetic_demo",
        action="store_true",
        default=False,
        help="Generate directory structures and authentic sample data files for immediate offline usage"
    )
    parser.add_argument(
        "--download_urls",
        action="store_true",
        default=False,
        help="Display official download URLs and instructions for acquiring raw benchmark datasets"
    )
    return parser.parse_args()


def generate_scd_samples(target_dir: str, num_samples: int = 4, size: tuple = (480, 640)):
    """Create sample RGB image pairs and binary change masks."""
    H, W = size
    for i in range(num_samples):
        # Base scene
        img1 = (np.random.rand(H, W, 3) * 200 + 40).astype(np.uint8)
        img2 = img1.copy()

        # Add camera shift
        img2 = np.roll(img2, shift=int(np.random.randint(-10, 10)), axis=1)

        # Add illumination shift
        img2 = np.clip(img2.astype(np.float32) * 1.1 + 10, 0, 255).astype(np.uint8)

        # Add change object
        mask = np.zeros((H, W), dtype=np.uint8)
        ch_h = int(H * 0.15)
        ch_w = int(W * 0.15)
        cy = int(H * 0.5)
        cx = int(W * 0.5)
        img2[cy:cy+ch_h, cx:cx+ch_w] = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        mask[cy:cy+ch_h, cx:cx+ch_w] = 255

        fname = f"sample_{i:04d}.png"
        return img1, img2, mask, fname


def generate_gdsr_samples(target_dir: str, num_samples: int = 4, size: tuple = (480, 640)):
    """Create sample RGB and Depth maps."""
    H, W = size
    for i in range(num_samples):
        rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        # Depth in mm (16-bit)
        depth_m = np.linspace(1.0, 4.0, H)[:, None] + np.linspace(0.0, 0.5, W)[None, :]
        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        fname = f"sample_{i:04d}.png"
        return rgb, depth_mm, fname


def prepare_synthetic_demo(target_datasets: list, root_data_dir: str):
    """Generate authentic directory structures and sample files."""
    print("=" * 70)
    print("Generating Synthetic Demo Dataset Structures...")
    print(f"Target Root: {os.path.abspath(root_data_dir)}")
    print("=" * 70)

    for ds_key in target_datasets:
        info = BENCHMARK_INFO[ds_key]
        ds_dir = os.path.join(root_data_dir, ds_key)
        print(f"\n[*] Initializing structure for [{info['name']}] ({info['type']})...")

        for sub in info["layout"]:
            sub_dir = os.path.join(ds_dir, sub)
            os.makedirs(sub_dir, exist_ok=True)

        if info["type"] == "SCD":
            # Populate sample pairs in first split or sequence
            if ds_key == "vl_cmu_cd":
                for split in ["train", "val", "test"]:
                    t0_dir = os.path.join(ds_dir, split, "t0")
                    t1_dir = os.path.join(ds_dir, split, "t1")
                    gt_dir = os.path.join(ds_dir, split, "gt")
                    img1, img2, mask, fname = generate_scd_samples(ds_dir)
                    Image.fromarray(img1).save(os.path.join(t0_dir, fname))
                    Image.fromarray(img2).save(os.path.join(t1_dir, fname))
                    Image.fromarray(mask).save(os.path.join(gt_dir, fname))
            elif ds_key == "pcd":
                for sub in ["TSUNAMI", "GSV"]:
                    t0_dir = os.path.join(ds_dir, sub, "t0")
                    t1_dir = os.path.join(ds_dir, sub, "t1")
                    gt_dir = os.path.join(ds_dir, sub, "groundtruth")
                    img1, img2, mask, fname = generate_scd_samples(ds_dir)
                    Image.fromarray(img1).save(os.path.join(t0_dir, fname))
                    Image.fromarray(img2).save(os.path.join(t1_dir, fname))
                    Image.fromarray(mask).save(os.path.join(gt_dir, fname))
            elif ds_key == "changesim":
                for seq in ["Seq_00", "Seq_01"]:
                    ref_dir = os.path.join(ds_dir, seq, "ref_rgb")
                    qry_dir = os.path.join(ds_dir, seq, "query_rgb")
                    msk_dir = os.path.join(ds_dir, seq, "change_mask")
                    img1, img2, mask, fname = generate_scd_samples(ds_dir)
                    Image.fromarray(img1).save(os.path.join(ref_dir, fname))
                    Image.fromarray(img2).save(os.path.join(qry_dir, fname))
                    Image.fromarray(mask).save(os.path.join(msk_dir, fname))

        elif info["type"] == "GDSR":
            if ds_key == "nyuv2":
                for split in ["train", "val", "test"]:
                    rgb_dir = os.path.join(ds_dir, split, "rgb")
                    dep_dir = os.path.join(ds_dir, split, "depth")
                    rgb, depth_mm, fname = generate_gdsr_samples(ds_dir)
                    Image.fromarray(rgb).save(os.path.join(rgb_dir, fname))
                    Image.fromarray(depth_mm).save(os.path.join(dep_dir, fname))
            elif ds_key == "middlebury":
                for scene in info["layout"]:
                    s_dir = os.path.join(ds_dir, scene)
                    rgb, depth_mm, fname = generate_gdsr_samples(ds_dir)
                    Image.fromarray(rgb).save(os.path.join(s_dir, "view1.png"))
                    Image.fromarray(depth_mm).save(os.path.join(s_dir, "disp1.png"))
            elif ds_key == "lu":
                rgb_dir = os.path.join(ds_dir, "rgb")
                dep_dir = os.path.join(ds_dir, "depth")
                rgb, depth_mm, fname = generate_gdsr_samples(ds_dir)
                Image.fromarray(rgb).save(os.path.join(rgb_dir, fname))
                Image.fromarray(depth_mm).save(os.path.join(dep_dir, fname))
            elif ds_key == "rgbdd":
                for split in ["train", "val", "test"]:
                    rgb_dir = os.path.join(ds_dir, split, "rgb")
                    lr_dir = os.path.join(ds_dir, split, "depth_lr")
                    gt_dir = os.path.join(ds_dir, split, "depth_gt")
                    rgb, depth_mm, fname = generate_gdsr_samples(ds_dir)
                    # Downsample lr depth
                    depth_lr = Image.fromarray(depth_mm).resize((160, 120), Image.NEAREST)
                    Image.fromarray(rgb).save(os.path.join(rgb_dir, fname))
                    depth_lr.save(os.path.join(lr_dir, fname))
                    Image.fromarray(depth_mm).save(os.path.join(gt_dir, fname))

        print(f"    [OK] Generated structure and sample data in: {ds_dir}")

    print("\n" + "=" * 70)
    print("Synthetic Demo Setup Successfully Completed!")
    print("=" * 70)


def display_download_urls(target_datasets: list):
    """Print official download URLs and benchmark descriptions."""
    print("=" * 70)
    print("Official General CV Benchmark Download URLs & References")
    print("=" * 70)
    for ds_key in target_datasets:
        info = BENCHMARK_INFO[ds_key]
        print(f"\nBenchmark: {info['name']} ({info['type']})")
        print(f"  Description: {info['description']}")
        print(f"  Official URL: {info['url']}")
        print(f"  Default Directory: ./data/{ds_key}/")
        print(f"  Subdirectories:")
        for sub in info["layout"][:4]:
            print(f"    - {sub}")
        if len(info["layout"]) > 4:
            print(f"    - ... ({len(info['layout'])} layout dirs total)")
    print("\n" + "=" * 70)


def main():
    args = parse_args()

    target_datasets = list(BENCHMARK_INFO.keys()) if args.dataset == "all" else [args.dataset]

    if args.download_urls or (not args.synthetic_demo):
        display_download_urls(target_datasets)

    if args.synthetic_demo:
        prepare_synthetic_demo(target_datasets, args.data_dir)


if __name__ == "__main__":
    main()
