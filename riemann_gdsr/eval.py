"""Standalone Evaluation Script for RiemannGDSR (CVPR 2026).

Computes quantitative depth super-resolution benchmarks:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- AbsRel (Absolute Relative Error)
- Log10 Error
- Threshold accuracies: delta_1, delta_2, delta_3 (< 1.25, 1.25^2, 1.25^3)
Optional visual comparison composite generation.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# Ensure cvpr root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from riemann_gdsr.models.riemann_gdsr import RiemannGDSRNet, RiemannGDSRConfig
from riemann_gdsr.metrics import GDSRMetrics
from riemann_gdsr.datasets import get_gdsr_dataset, build_gdsr_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="RiemannGDSR Standalone Evaluation Pipeline")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained model weights (.pt checkpoint)")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["nyuv2", "middlebury", "lu", "rgbdd", "synthetic"],
                        help="Evaluation benchmark dataset name")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to dataset root directory")
    parser.add_argument("--scale", type=int, default=4, choices=[4, 8, 16],
                        help="Super-resolution scaling factor (x4, x8, x16)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Evaluation split")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size")
    parser.add_argument("--img_size", type=str, default="480,640",
                        help="Image resolution as 'H,W'")
    parser.add_argument("--num_steps", type=int, default=4,
                        help="Number of Riemannian unrolling steps K")
    parser.add_argument("--feature_dim", type=int, default=48,
                        help="Channel dimension for guidance features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Evaluation device (cuda or cpu)")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Evaluate on synthetic benchmark stream")
    parser.add_argument("--save_vis", type=str, default=None,
                        help="Directory to save depth reconstruction composites")
    parser.add_argument("--num_vis", type=int, default=5,
                        help="Number of qualitative prediction samples to save")
    return parser.parse_args()


def depth_to_colormap(depth: np.ndarray, vmin: float = 0.5, vmax: float = 5.0) -> np.ndarray:
    """Simple turbo/plasma pseudo-colormap for depth visualization without extra dependencies."""
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    # Color gradient from blue (near) to green to yellow to red (far)
    r = np.clip(1.5 - np.abs(norm * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(norm * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(norm * 4.0 - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_gdsr_visualization(
    rgb: torch.Tensor,
    d_lr: torch.Tensor,
    d_gt: torch.Tensor,
    d_pred: torch.Tensor,
    save_path: str,
):
    """Save 4-panel depth composite: [RGB Guidance, Low-Res Input, Ground Truth, RiemannGDSR Reconstructed]."""
    rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    H, W = rgb_np.shape[0], rgb_np.shape[1]

    # Resize d_lr to H, W for visualization
    d_lr_up = torch.nn.functional.interpolate(d_lr.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
    lr_np = d_lr_up.squeeze(0).cpu().numpy()
    gt_np = d_gt.squeeze(0).cpu().numpy()
    pr_np = d_pred.squeeze(0).cpu().numpy()

    vmin = float(min(gt_np[gt_np > 0].min() if (gt_np > 0).any() else 0.5, 0.5))
    vmax = float(max(gt_np.max() if gt_np.max() > 0 else 5.0, 5.0))

    lr_col = depth_to_colormap(lr_np, vmin, vmax)
    gt_col = depth_to_colormap(gt_np, vmin, vmax)
    pr_col = depth_to_colormap(pr_np, vmin, vmax)

    composite = np.concatenate([rgb_np, lr_col, gt_col, pr_col], axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(composite).save(save_path)


def main():
    args = parse_args()

    h_w = [int(x.strip()) for x in args.img_size.split(",")]
    img_size = (h_w[0], h_w[1])
    device = torch.device(args.device)

    print("=" * 70)
    print("RiemannGDSR Standalone Evaluation Pipeline")
    print(f"Device: {device} | Scale: x{args.scale} | Resolution: {img_size}")
    print(f"Dataset: {args.dataset} | Split: {args.split} | Synthetic: {args.synthetic}")
    print("=" * 70)

    # 1. Dataset & DataLoader
    eval_dataset = get_gdsr_dataset(
        name=args.dataset,
        root_dir=args.data_root,
        split=args.split,
        scale=args.scale,
        img_size=img_size,
        synthetic=args.synthetic,
        synthetic_samples=max(args.batch_size * 4, 8),
    )
    eval_loader = build_gdsr_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"Loaded {len(eval_dataset)} evaluation samples.")

    # 2. Model Configuration & Checkpoint Loading
    cfg = RiemannGDSRConfig(
        num_steps=args.num_steps,
        feature_dim=args.feature_dim,
        use_checkpoint=True,
    )
    model = RiemannGDSRNet(config=cfg).to(device)

    if args.weights and os.path.isfile(args.weights):
        print(f"Loading checkpoint from: {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("Model checkpoint loaded successfully.")
    else:
        if args.weights:
            print(f"Warning: Checkpoint file '{args.weights}' not found. Evaluating initialized model.")
        else:
            print("No checkpoint specified. Evaluating initialized model.")

    model.eval()

    # 3. Evaluation Loop
    metrics = GDSRMetrics()
    vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            d_gt = batch["d_gt"].to(device, non_blocking=True)
            d_lr = batch["d_lr"].to(device, non_blocking=True)
            mask_gt = batch["mask_gt"].to(device, non_blocking=True)
            mask_lr = batch["mask_lr"].to(device, non_blocking=True)

            preds = model(rgb, d_lr, mask_lr)
            pred_depth = preds["pred"]

            metrics.update(pred_depth, d_gt, mask=mask_gt)

            if args.save_vis and vis_count < args.num_vis:
                B = rgb.shape[0]
                for b in range(B):
                    if vis_count >= args.num_vis:
                        break
                    save_p = os.path.join(args.save_vis, f"sample_{vis_count:03d}.png")
                    save_gdsr_visualization(rgb[b], d_lr[b], d_gt[b], pred_depth[b], save_p)
                    vis_count += 1

    # 4. Report Quantitative Metrics
    results = metrics.compute()
    print("\n" + "=" * 70)
    print(f"RiemannGDSR Quantitative Evaluation Report (Scale x{args.scale})")
    print("=" * 70)
    print(f"  RMSE:      {results['rmse']:.4f} m ({results['rmse']*100:.2f} cm)")
    print(f"  MAE:       {results['mae']:.4f} m ({results['mae']*100:.2f} cm)")
    print(f"  AbsRel:    {results['abs_rel']:.4f}")
    print(f"  Log10:     {results['log10']:.4f}")
    print(f"  delta_1:   {results['delta_1'] * 100:6.2f}% (Threshold < 1.25)")
    print(f"  delta_2:   {results['delta_2'] * 100:6.2f}% (Threshold < 1.25^2)")
    print(f"  delta_3:   {results['delta_3'] * 100:6.2f}% (Threshold < 1.25^3)")
    print("-" * 70)
    print(f"  Total Valid Pixels: {results['total_pixels']}, Samples: {results['total_samples']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
