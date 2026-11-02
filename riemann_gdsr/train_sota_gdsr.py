"""Training Pipeline for 2023-2025 Top-Conference GDSR SOTA Comparison Models.

Supports training:
- dada (CVPR 2023: Deep Anisotropic Diffusion)
- ssdnet (ICCV 2023: Spherical Space Feature Decomposition)
- sgnet (AAAI 2024: Structure Guided Network via Gradient-Frequency Awareness)
- dornet (CVPR 2025: Degradation Oriented and Regularized Network)
- ducos (ICCV 2025: Duality Constrained Depth Super-Resolution)
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from riemann_gdsr.sota_models import build_sota_gdsr_model
from riemann_gdsr.metrics import GDSRMetrics
from riemann_gdsr.datasets import get_gdsr_dataset, build_gdsr_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2023-2025 GDSR SOTA Baselines")
    parser.add_argument("--sota", type=str, required=True,
                        choices=["dada", "ssdnet", "sgnet", "dornet", "ducos"])
    parser.add_argument("--dataset", type=str, default="nyuv2",
                        choices=["nyuv2", "rgbdd"])
    parser.add_argument("--data_root", type=str, default="/home/WUtongquan/dataset/NYUv2")
    parser.add_argument("--scale", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = get_gdsr_dataset(
        name=args.dataset,
        root_dir=args.data_root,
        split="train",
        scale=args.scale,
        img_size=(480, 640),
        synthetic=False,
    )
    val_ds = get_gdsr_dataset(
        name=args.dataset,
        root_dir=args.data_root,
        split="val",
        scale=args.scale,
        img_size=(480, 640),
        synthetic=False,
    )

    train_loader = build_gdsr_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if train_ds.is_synthetic else args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = build_gdsr_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if val_ds.is_synthetic else args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_sota_gdsr_model(args.sota).to(device)
    if args.resume and os.path.isfile(args.resume):
        print(f"[*] Loading {args.sota} checkpoint from {args.resume}", flush=True)
        sd = torch.load(args.resume, map_location=device)
        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        model.load_state_dict(sd, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("=" * 70, flush=True)
    print(f"Training SOTA Baseline: {args.sota.upper()} | Dataset: {args.dataset} | Scale: x{args.scale}", flush=True)
    print("=" * 70, flush=True)

    best_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_b = 0

        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            d_gt = batch["d_gt"].to(device, non_blocking=True)
            d_lr = batch["d_lr"].to(device, non_blocking=True)
            mask_gt = batch["mask_gt"].to(device, non_blocking=True)
            mask_lr = batch["mask_lr"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                pred = model(rgb, d_lr, mask_lr)
                loss = torch.sum(torch.abs(pred - d_gt) * mask_gt) / (torch.sum(mask_gt) + 1e-5)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()

        model.eval()
        metrics = GDSRMetrics()
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                d_gt = batch["d_gt"].to(device, non_blocking=True)
                d_lr = batch["d_lr"].to(device, non_blocking=True)
                mask_gt = batch["mask_gt"].to(device, non_blocking=True)
                mask_lr = batch["mask_lr"].to(device, non_blocking=True)
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    pred = model(rgb, d_lr, mask_lr)
                metrics.update(pred, d_gt, mask=mask_gt)

        m = metrics.compute()
        dt = time.time() - t0
        print(
            f"[{args.sota.upper()} Epoch {epoch:02d}/{args.epochs:02d}] "
            f"Loss: {total_loss/max(n_b,1):.4f} | Val RMSE: {m['rmse']*100:.2f} cm ({m['rmse']:.4f} m) | "
            f"MAE: {m['mae']*100:.2f} cm | delta_1: {m['delta_1']*100:.2f}% | Time: {dt:.1f}s",
            flush=True,
        )

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "val_metrics": m}
        torch.save(ckpt, os.path.join(args.save_dir, "latest.pt"))
        if math.isfinite(m["rmse"]) and m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
