"""Production Training Pipeline for RiemannGDSR (CVPR 2026).

RGB-Guided Depth Super-Resolution & Completion via Monocular-Prior Self-Calibration
and Spectral Screened-Poisson S^2 Geodesic Unrolling.
Features:
- Self-calibrates spatially varying scale-shift (alpha_phi, beta_phi) and PSF kernel K_phi.
- Strict S^2_+ manifold geodesic unrolling & Screened-Poisson range-null consistency.
- Single-sample peak VRAM containment (< 2.5 GB on 480x640).
- Mixed precision training (AMP FP16/BF16) with GradScaler and finite-loss guard.
- Instant runnable verification via `--synthetic` flag.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import argparse
import math
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure cvpr root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from riemann_gdsr.models.riemann_gdsr import RiemannGDSRNet, RiemannGDSRConfig
from riemann_gdsr.losses.gdsr_loss import GDSRLoss
from riemann_gdsr.metrics import GDSRMetrics
from riemann_gdsr.datasets import get_gdsr_dataset, build_gdsr_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="RiemannGDSR Production Training Pipeline")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["nyuv2", "middlebury", "lu", "rgbdd", "synthetic"],
                        help="Benchmark dataset name")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to dataset root directory")
    parser.add_argument("--scale", type=int, default=4, choices=[4, 8, 16],
                        help="Super-resolution scaling factor (x4, x8, x16)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini-batch size (default 1 for VRAM efficiency < 2.5 GB)")
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--img_size", type=str, default="480,640",
                        help="Image resolution as 'H,W' (e.g. 480,640)")
    parser.add_argument("--num_steps", type=int, default=4,
                        help="Number of Riemannian unrolling steps K")
    parser.add_argument("--feature_dim", type=int, default=48,
                        help="Channel dimension for guidance features")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable PyTorch Automatic Mixed Precision (AMP)")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="Disable AMP")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target device (cuda or cpu)")
    parser.add_argument("--save_dir", type=str, default="runs/riemann_gdsr",
                        help="Directory to save training logs and checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional path to checkpoint to warm-start / resume from")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Force synthetic benchmark data generation for instant runnable verification")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Validation interval in epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_autocast(device_type: str, enabled: bool):
    """Safely create autocast context manager without deprecation warnings."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled and device_type == "cuda")


def create_grad_scaler(device_type: str, enabled: bool):
    """Safely create GradScaler without deprecation warnings."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled and device_type == "cuda")


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
) -> dict:
    model.train()
    total_loss = 0.0
    total_depth = 0.0
    total_norm = 0.0
    total_range = 0.0
    total_calib = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        d_gt = batch["d_gt"].to(device, non_blocking=True)
        d_lr = batch["d_lr"].to(device, non_blocking=True)
        mask_gt = batch["mask_gt"].to(device, non_blocking=True)
        mask_lr = batch["mask_lr"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with create_autocast(device.type, amp_enabled and device.type == 'cuda'):
            preds = model(rgb, d_lr, mask_lr)
            loss, loss_dict = criterion(
                preds,
                d_gt=d_gt,
                d_lr=d_lr,
                mask_gt=mask_gt,
                sensor_calib=model.sensor_calib,
            )

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        if amp_enabled and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        total_loss += loss.item()
        total_depth += loss_dict.get("loss_depth", torch.tensor(0.0)).item()
        total_norm += loss_dict.get("loss_norm", torch.tensor(0.0)).item()
        total_range += loss_dict.get("loss_range", torch.tensor(0.0)).item()
        total_calib += loss_dict.get("loss_calib", torch.tensor(0.0)).item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_depth": total_depth / max(num_batches, 1),
        "loss_norm": total_norm / max(num_batches, 1),
        "loss_range": total_range / max(num_batches, 1),
        "loss_calib": total_calib / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict:
    model.eval()
    metrics = GDSRMetrics()

    for batch in dataloader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        d_gt = batch["d_gt"].to(device, non_blocking=True)
        d_lr = batch["d_lr"].to(device, non_blocking=True)
        mask_gt = batch["mask_gt"].to(device, non_blocking=True)
        mask_lr = batch["mask_lr"].to(device, non_blocking=True)

        with create_autocast(device.type, amp_enabled and device.type == 'cuda'):
            preds = model(rgb, d_lr, mask_lr)
            pred_depth = preds["pred"]

        metrics.update(pred_depth, d_gt, mask=mask_gt)

    return metrics.compute()


def main():
    args = parse_args()
    set_seed(args.seed)

    h_w = [int(x.strip()) for x in args.img_size.split(",")]
    img_size = (h_w[0], h_w[1])
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("=" * 70, flush=True)
    print("RiemannGDSR Production Training Pipeline", flush=True)
    print(f"Device: {device} | AMP: {args.amp} | Scale: x{args.scale} | Resolution: {img_size}", flush=True)
    print(f"Dataset: {args.dataset} | Synthetic Mode: {args.synthetic}", flush=True)
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Workers: {args.num_workers} | LR: {args.lr}", flush=True)
    print("=" * 70, flush=True)

    # 1. Datasets & Dataloaders
    train_dataset = get_gdsr_dataset(
        name=args.dataset,
        root_dir=args.data_root,
        split='train',
        scale=args.scale,
        img_size=img_size,
        synthetic=args.synthetic,
        synthetic_samples=max(args.batch_size * 4, 8),
    )
    val_dataset = get_gdsr_dataset(
        name=args.dataset,
        root_dir=args.data_root,
        split='val',
        scale=args.scale,
        img_size=img_size,
        synthetic=args.synthetic,
        synthetic_samples=max(args.batch_size * 2, 4),
    )

    num_workers = 0 if train_dataset.is_synthetic else args.num_workers
    train_loader = build_gdsr_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = build_gdsr_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

    # 2. Model Configuration & Instantiation
    cfg = RiemannGDSRConfig(
        num_steps=args.num_steps,
        feature_dim=args.feature_dim,
        use_checkpoint=True,
    )
    model = RiemannGDSRNet(config=cfg).to(device)

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading warm-start weights from: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)

    criterion = GDSRLoss(
        gamma=0.85,
        lambda_norm=0.2,
        lambda_range=1.0,
        lambda_calib=0.05,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = create_grad_scaler(device.type, enabled=args.amp and device.type == 'cuda')

    best_rmse = float('inf')

    # Initial Epoch 0 validation check
    init_metrics = evaluate(model, val_loader, device, args.amp)
    print(f"[Epoch 00 (Warm-Start)] Validation -> RMSE: {init_metrics['rmse']:.4f} m ({init_metrics['rmse']*100:.2f} cm) | "
          f"MAE: {init_metrics['mae']:.4f} m | AbsRel: {init_metrics['abs_rel']:.4f} | "
          f"delta_1: {init_metrics['delta_1']*100:.2f}%", flush=True)
    if math.isfinite(init_metrics["rmse"]):
        best_rmse = init_metrics["rmse"]
        torch.save({
            "epoch": 0,
            "model_state": model.state_dict(),
            "val_metrics": init_metrics,
            "args": vars(args),
        }, os.path.join(args.save_dir, "best.pt"))

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=args.amp,
            epoch=epoch,
        )
        scheduler.step()

        # Log peak VRAM
        if device.type == "cuda":
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            vram_str = f" | Peak VRAM: {peak_vram_gb:.3f} GB (< 2.5 GB: {peak_vram_gb < 2.5})"
        else:
            vram_str = ""

        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] "
              f"Loss: {train_stats['loss']:.4f} "
              f"(Depth: {train_stats['loss_depth']:.4f}, Norm: {train_stats['loss_norm']:.4f}, "
              f"Range: {train_stats['loss_range']:.4f}, Calib: {train_stats['loss_calib']:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}{vram_str} | Time: {epoch_time:.2f}s", flush=True)

        # Validation
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device, args.amp)
            print(f"  Validation -> RMSE: {val_metrics['rmse']:.4f} m ({val_metrics['rmse']*100:.2f} cm) | "
                  f"MAE: {val_metrics['mae']:.4f} m ({val_metrics['mae']*100:.2f} cm) | "
                  f"AbsRel: {val_metrics['abs_rel']:.4f} | delta_1: {val_metrics['delta_1']*100:.2f}% | "
                  f"delta_2: {val_metrics['delta_2']*100:.2f}% | delta_3: {val_metrics['delta_3']*100:.2f}%", flush=True)

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, "latest.pt"))

            val_rmse = val_metrics["rmse"]
            if math.isfinite(val_rmse) and val_rmse > 1e-4 and val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(checkpoint, os.path.join(args.save_dir, "best.pt"))
                print(f"  [*] Saved new best model checkpoint (RMSE: {best_rmse:.4f} m / {best_rmse*100:.2f} cm)", flush=True)

    total_time = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"RiemannGDSR Training Complete in {total_time:.2f}s. Best RMSE: {best_rmse:.4f} m ({best_rmse*100:.2f} cm)", flush=True)
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Global Peak VRAM Allocated: {peak_vram:.3f} GB (Strictly < 2.5 GB: {peak_vram < 2.5})", flush=True)
    print(f"Checkpoints saved to {args.save_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
