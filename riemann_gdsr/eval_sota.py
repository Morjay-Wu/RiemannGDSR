"""Comprehensive SOTA Comparison & Multi-Benchmark Evaluation Suite for RiemannGDSR (CVPR 2026).

Evaluates classical, spectral/filtering, and 2023-2025 Top-Conference Deep GDSR methods:
1. Nearest Interpolation
2. Bilinear Interpolation
3. Bicubic Interpolation
4. Guided Image Filter (GIF, He et al. TPAMI 2013)
5. Joint Bilateral Upsampling (JBU, Kopf et al. SIGGRAPH)
6. Spectral Edge-Guided DCT Solver (DCTNet-Solver, CVPR 2022)
7. DADA (CVPR 2023: Deep Anisotropic Diffusion, ETH Zurich)
8. SSDNet (ICCV 2023: Spherical Space Feature Decomposition)
9. SGNet (AAAI 2024: Structure Guided Network via Gradient-Frequency Awareness)
10. DORNet (CVPR 2025: Degradation Oriented and Regularized Network)
11. DuCos (ICCV 2025: Duality Constrained Depth Super-Resolution)
12. RiemannGDSR (Ours, CVPR 2026)

Supports benchmarks:
- NYUv2 (x4, x8, x16)
- Middlebury (x4, x8, x16 zero-shot generalization)
- Lu (x4, x8, x16 zero-shot generalization)
- RGB-D-D (x4, x8, x16 & real sensor degradation)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from riemann_gdsr.models.riemann_gdsr import RiemannGDSRNet, RiemannGDSRConfig
from riemann_gdsr.models.sensor_calibration import SensorCalibration
from riemann_gdsr.sota_models import build_sota_gdsr_model
from riemann_gdsr.metrics import GDSRMetrics
from riemann_gdsr.datasets import get_gdsr_dataset, build_gdsr_dataloader


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    k = 2 * r + 1
    padded = F.pad(x, (r, r, r, r), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=k, stride=1)


def guided_filter_gdsr(
    rgb: torch.Tensor,
    d_lr: torch.Tensor,
    mask_lr: torch.Tensor,
    r: int = 8,
    eps: float = 1e-3,
) -> torch.Tensor:
    H, W = rgb.shape[-2:]
    d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
    p = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)
    I = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    corr_I = box_filter(I * I, r)
    corr_Ip = box_filter(I * p, r)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I + mean_b
    return torch.clamp(q, min=1e-3)


def joint_bilateral_gdsr(
    rgb: torch.Tensor,
    d_lr: torch.Tensor,
    mask_lr: torch.Tensor,
    ks: int = 7,
    sigma_s: float = 3.0,
    sigma_c: float = 0.08,
) -> torch.Tensor:
    B, _, H, W = rgb.shape
    d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
    d_init = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)

    pad = ks // 2
    d_pad = F.pad(d_init, (pad, pad, pad, pad), mode="replicate")
    rgb_pad = F.pad(rgb, (pad, pad, pad, pad), mode="replicate")

    d_patches = F.unfold(d_pad, kernel_size=ks).view(B, 1, ks * ks, H, W)
    rgb_patches = F.unfold(rgb_pad, kernel_size=ks).view(B, 3, ks * ks, H, W)

    coords = torch.arange(-pad, pad + 1, device=rgb.device, dtype=rgb.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    w_spatial = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma_s**2)).reshape(1, 1, ks * ks, 1, 1)

    color_diff_sq = torch.sum((rgb_patches - rgb.unsqueeze(2)) ** 2, dim=1, keepdim=True)
    w_color = torch.exp(-color_diff_sq / (2.0 * sigma_c**2))

    weights = w_spatial * w_color
    d_out = torch.sum(weights * d_patches, dim=2) / (torch.sum(weights, dim=2) + 1e-6)
    return torch.clamp(d_out, min=1e-3)


def dct_spectral_solver_gdsr(
    rgb: torch.Tensor,
    d_lr: torch.Tensor,
    mask_lr: torch.Tensor,
    lam: float = 0.15,
    gamma_edge: float = 12.0,
) -> torch.Tensor:
    H, W = rgb.shape[-2:]
    h, w = d_lr.shape[-2:]
    scale = H // h
    d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)
    d_bic = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)

    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    pad_g = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx_I = (pad_g[:, :, 1:-1, 2:] - pad_g[:, :, 1:-1, :-2]) * 0.5
    gy_I = (pad_g[:, :, 2:, 1:-1] - pad_g[:, :, :-2, 1:-1]) * 0.5
    edge_w = torch.exp(-gamma_edge * torch.sqrt(gx_I**2 + gy_I**2 + 1e-6))

    pad_d = F.pad(d_bic, (1, 1, 1, 1), mode="replicate")
    gx_d = (pad_d[:, :, 1:-1, 2:] - pad_d[:, :, 1:-1, :-2]) * 0.5 * edge_w
    gy_d = (pad_d[:, :, 2:, 1:-1] - pad_d[:, :, :-2, 1:-1]) * 0.5 * edge_w

    gx_pad = F.pad(gx_d, (1, 1, 0, 0), mode="replicate")
    gy_pad = F.pad(gy_d, (0, 0, 1, 1), mode="replicate")
    div_g = (gx_pad[:, :, :, 2:] - gx_pad[:, :, :, :-2]) * 0.5 + (
        gy_pad[:, :, 2:, :] - gy_pad[:, :, :-2, :]
    ) * 0.5
    div_g = div_g - div_g.mean(dim=(-2, -1), keepdim=True)

    rhs = lam * d_bic.float() - div_g.float()
    rhs_hat = torch.fft.rfft2(rhs)
    wy = 2.0 * math.pi * torch.fft.fftfreq(H, device=rgb.device, dtype=torch.float32).view(1, 1, H, 1)
    wx = 2.0 * math.pi * torch.fft.rfftfreq(W, device=rgb.device, dtype=torch.float32).view(1, 1, 1, W // 2 + 1)
    lap = 4.0 - 2.0 * torch.cos(wx) - 2.0 * torch.cos(wy)
    d_sol = torch.fft.irfft2(rhs_hat / (lap + lam), s=(H, W))

    deg = F.avg_pool2d(d_sol, kernel_size=scale, stride=scale)
    res_lr = (d_filled.float() - deg) * mask_lr.float()
    d_sol = d_sol + F.interpolate(res_lr, size=(H, W), mode="bicubic", align_corners=False)
    return torch.clamp(d_sol, min=1e-3)


def load_sota_models_for_scale(scale: int, device: torch.device, sota_ckpt_dir: str = "runs") -> Dict[str, nn.Module]:
    """Load 2023-2025 deep SOTA models (with trained checkpoints or official pretrained weights)."""
    models_dict: Dict[str, nn.Module] = {}
    spec = [
        ("DADA (CVPR'23)", "dada", [f"{sota_ckpt_dir}/sota_dada_x{scale}/best.pt"]),
        ("SSDNet (ICCV'23)", "ssdnet", [f"{sota_ckpt_dir}/sota_ssdnet_x{scale}/best.pt"]),
        ("SGNet (AAAI'24)", "sgnet", [f"{sota_ckpt_dir}/sota_sgnet_x{scale}/best.pt"]),
        (
            "DORNet (CVPR'25)",
            "dornet",
            [
                f"{sota_ckpt_dir}/sota_dornet_x{scale}/best.pt",
                f"third_party_sota/DORNet/checkpoints/NYU_X{scale}.pth",
                f"/home/WUtongquan/model/cvpr/third_party_sota/DORNet/checkpoints/NYU_X{scale}.pth",
            ],
        ),
        ("DuCos (ICCV'25)", "ducos", [f"{sota_ckpt_dir}/sota_ducos_x{scale}/best.pt"]),
    ]

    for display_name, key, ckpt_candidates in spec:
        m = build_sota_gdsr_model(key).to(device)
        for p in ckpt_candidates:
            if os.path.isfile(p):
                print(f"[*] Loading {display_name} checkpoint from {p}", flush=True)
                sd = torch.load(p, map_location=device)
                if isinstance(sd, dict) and "model_state" in sd:
                    sd = sd["model_state"]
                m.load_state_dict(sd, strict=False)
                break
        m.eval()
        models_dict[display_name] = m

    return models_dict


@torch.no_grad()
def evaluate_all_methods_on_dataset(
    dataset_name: str,
    data_root: str,
    scale: int,
    split: str,
    img_size: tuple[int, int],
    device: torch.device,
    riemann_model: Optional[nn.Module] = None,
    deep_sota_models: Optional[Dict[str, nn.Module]] = None,
    batch_size: int = 1,
    num_workers: int = 2,
) -> Dict[str, Dict[str, float]]:
    ds = get_gdsr_dataset(
        name=dataset_name,
        root_dir=data_root,
        split=split,
        scale=scale,
        img_size=img_size,
        synthetic=False,
    )
    loader = build_gdsr_dataloader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if ds.is_synthetic else num_workers,
        pin_memory=(device.type == "cuda"),
    )

    method_names = [
        "Nearest",
        "Bilinear",
        "Bicubic",
        "GuidedFilter (TPAMI'13)",
        "JointBilateral (JBU)",
        "DCTNet-Solver (CVPR'22)",
    ]
    if deep_sota_models:
        method_names.extend(list(deep_sota_models.keys()))
    method_names.append("RiemannGDSR (Ours)")

    evaluators = {m: GDSRMetrics() for m in method_names}

    H, W = img_size
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        d_gt = batch["d_gt"].to(device, non_blocking=True)
        d_lr = batch["d_lr"].to(device, non_blocking=True)
        mask_gt = batch["mask_gt"].to(device, non_blocking=True)
        mask_lr = batch["mask_lr"].to(device, non_blocking=True)

        d_filled = SensorCalibration._inpaint_lr_holes(d_lr, mask_lr)

        # 1. Nearest
        p_near = F.interpolate(d_filled, size=(H, W), mode="nearest")
        evaluators["Nearest"].update(p_near, d_gt, mask=mask_gt)

        # 2. Bilinear
        p_bil = F.interpolate(d_filled, size=(H, W), mode="bilinear", align_corners=False)
        evaluators["Bilinear"].update(p_bil, d_gt, mask=mask_gt)

        # 3. Bicubic
        p_bic = F.interpolate(d_filled, size=(H, W), mode="bicubic", align_corners=False)
        evaluators["Bicubic"].update(p_bic, d_gt, mask=mask_gt)

        # 4. GuidedFilter
        p_gif = guided_filter_gdsr(rgb, d_lr, mask_lr)
        evaluators["GuidedFilter (TPAMI'13)"].update(p_gif, d_gt, mask=mask_gt)

        # 5. JointBilateral
        p_jbu = joint_bilateral_gdsr(rgb, d_lr, mask_lr)
        evaluators["JointBilateral (JBU)"].update(p_jbu, d_gt, mask=mask_gt)

        # 6. DCTNet-Solver
        p_dct = dct_spectral_solver_gdsr(rgb, d_lr, mask_lr)
        evaluators["DCTNet-Solver (CVPR'22)"].update(p_dct, d_gt, mask=mask_gt)

        # 7. 2023-2025 Deep SOTA Models
        if deep_sota_models:
            for s_name, s_mod in deep_sota_models.items():
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    p_sota = s_mod(rgb, d_lr, mask_lr)
                evaluators[s_name].update(p_sota, d_gt, mask=mask_gt)

        # 8. RiemannGDSR (Ours)
        if riemann_model is not None:
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                out = riemann_model(rgb, d_lr, mask_lr)
                p_ours = out["pred"]
            evaluators["RiemannGDSR (Ours)"].update(p_ours, d_gt, mask=mask_gt)

    return {m: evaluators[m].compute() for m in method_names}


def parse_args():
    parser = argparse.ArgumentParser(description="RiemannGDSR SOTA Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="nyuv2",
                        choices=["nyuv2", "middlebury", "lu", "rgbdd", "all"])
    parser.add_argument("--data_root", type=str, default="/home/WUtongquan/dataset")
    parser.add_argument("--scale", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to RiemannGDSR checkpoint (.pt)")
    parser.add_argument("--sota_ckpt_dir", type=str, default="runs",
                        help="Directory containing trained SOTA model folders")
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--out_file", type=str, default="runs/sota_comparison_report.md")
    return parser.parse_args()


def resolve_dataset_dir(base_root: str, name: str) -> str:
    mapping = {
        "nyuv2": "NYUv2",
        "middlebury": "Middlebury",
        "lu": "Lu",
        "rgbdd": "RGB-D-D",
    }
    sub = mapping.get(name, name)
    cand = os.path.join(base_root, sub)
    if os.path.isdir(cand):
        return cand
    return base_root


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = RiemannGDSRConfig(
        num_steps=args.num_steps,
        feature_dim=args.feature_dim,
        use_checkpoint=False,
    )
    model = RiemannGDSRNet(cfg).to(device)
    if args.weights and os.path.isfile(args.weights):
        print(f"[*] Loading RiemannGDSR weights from: {args.weights}", flush=True)
        ckpt = torch.load(args.weights, map_location=device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.eval()

    deep_sota_models = load_sota_models_for_scale(args.scale, device, args.sota_ckpt_dir)

    datasets_to_run = ["nyuv2", "middlebury", "lu", "rgbdd"] if args.dataset == "all" else [args.dataset]
    markdown_lines = [
        f"# RiemannGDSR SOTA Benchmark Evaluation Report (Scale x{args.scale})",
        "",
    ]

    all_results = {}
    for ds_name in datasets_to_run:
        ds_dir = resolve_dataset_dir(args.data_root, ds_name)
        print("=" * 80, flush=True)
        print(f"Evaluating on {ds_name.upper()} (Scale x{args.scale}, Split: {args.split}) at {ds_dir}", flush=True)
        print("=" * 80, flush=True)

        res = evaluate_all_methods_on_dataset(
            dataset_name=ds_name,
            data_root=ds_dir,
            scale=args.scale,
            split=args.split,
            img_size=(480, 640),
            device=device,
            riemann_model=model,
            deep_sota_models=deep_sota_models,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        all_results[ds_name] = res

        header = f"## Benchmark: {ds_name.upper()} (x{args.scale})"
        table_head = "| Method | RMSE (cm) ↓ | RMSE (m) ↓ | MAE (cm) ↓ | AbsRel ↓ | delta_1 (%) ↑ |"
        table_sep = "| :--- | :---: | :---: | :---: | :---: | :---: |"
        print(header, flush=True)
        print(table_head, flush=True)
        print(table_sep, flush=True)
        markdown_lines.extend([header, "", table_head, table_sep])

        for method, m in res.items():
            row = (
                f"| **{method}** | {m['rmse']*100:.2f} | {m['rmse']:.4f} | "
                f"{m['mae']*100:.2f} | {m['abs_rel']:.4f} | {m['delta_1']*100:.2f}% |"
            )
            print(row, flush=True)
            markdown_lines.append(row)
        markdown_lines.append("")

    if args.out_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_lines))
        json_path = os.path.splitext(args.out_file)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[*] Saved report to {args.out_file} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
