"""
Comprehensive Standalone Test Suite for RiemannGDSR.
CVPR 2026.
Runnable via: conda activate pytorch ; python riemann_gdsr/tests/test_riemann_gdsr.py
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from riemann_gdsr.models import (
    RiemannGDSRNet,
    RiemannGDSRConfig,
    SensorCalibration,
    TangentSpaceProjection,
    LeviCivitaParallelTransport,
    GeodesicExponentialRetraction,
    ScreenedPoissonRangeNullSolver,
)
from riemann_gdsr.losses import GDSRLoss


def test_manifold_invariants():
    print("\n[Test 1] Verifying Manifold Invariants on S^2_+ and Tangent Space T_n S^2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 2, 64, 64

    # 1. Test Tangent Space Projection: <v_perp, n> == 0
    proj = TangentSpaceProjection()
    raw_n = torch.randn(B, 3, H, W, device=device)
    raw_n[:, 2] = torch.abs(raw_n[:, 2]) + 0.1  # ensure positive z
    n = F.normalize(raw_n, p=2, dim=1)

    v_random = torch.randn(B, 3, H, W, device=device) * 5.0
    v_perp = proj(v_random, n)

    dot_prod = torch.sum(v_perp * n, dim=1, keepdim=True)
    max_dot = torch.max(torch.abs(dot_prod)).item()
    print(f"  Max absolute dot product <v_perp, n>: {max_dot:.8e}")
    assert max_dot < 1e-5, f"Tangent projection invariant violated! Max dot: {max_dot}"

    # 2. Test Levi-Civita Parallel Transport preserves tangent orthogonality
    transport = LeviCivitaParallelTransport(channels=3).to(device)
    rgb = torch.rand(B, 3, H, W, device=device)
    d_anchor = torch.rand(B, 1, H, W, device=device) + 0.5

    v_trans = transport(v_perp, n, rgb, d_anchor)
    dot_trans = torch.sum(v_trans * n, dim=1, keepdim=True)
    max_dot_trans = torch.max(torch.abs(dot_trans)).item()
    print(f"  Max absolute dot product after parallel transport: {max_dot_trans:.8e}")
    assert max_dot_trans < 1e-5, f"Parallel transport tangent invariant violated! Max dot: {max_dot_trans}"

    # 3. Test Geodesic Exponential Retraction: ||n^(k)||_2 == 1.0 +- 1e-5 and n_z > 0
    retract = GeodesicExponentialRetraction(max_angle=math.pi / 3.0)
    n_next = retract(n, v_trans)

    norms = torch.sqrt(torch.sum(n_next**2, dim=1, keepdim=True))
    norm_dev = torch.max(torch.abs(norms - 1.0)).item()
    min_nz = torch.min(n_next[:, 2]).item()

    print(f"  Max normal norm deviation | ||n||_2 - 1.0 |: {norm_dev:.8e}")
    print(f"  Min normal z-component n_z: {min_nz:.6f}")
    assert norm_dev < 1e-5, f"Unit normal manifold invariant violated! Dev: {norm_dev}"
    assert min_nz > 0.0, f"Line-of-sight hemisphere invariant violated! Min nz: {min_nz}"
    print("  -> Manifold Invariants Passed!")


def test_screened_poisson_solver():
    print("\n[Test 2] Verifying Screened-Poisson 2D FFT Solver Invariants...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 2, 128, 128
    scale = 4
    h, w = H // scale, W // scale

    calib = SensorCalibration(in_channels=16, kernel_size=15).to(device)
    solver = ScreenedPoissonRangeNullSolver(default_lambda=0.1).to(device)

    d_anchor = 1.0 + torch.rand(B, 1, H, W, device=device) * 2.0
    d_lr = F.avg_pool2d(d_anchor, kernel_size=scale, stride=scale)
    mask_lr = torch.ones_like(d_lr)

    dummy_rgb_feat = torch.randn(B, 16, H, W, device=device)
    k_spatial = calib.estimate_psf_kernel(dummy_rgb_feat, d_lr)

    raw_n = torch.randn(B, 3, H, W, device=device)
    raw_n[:, 2] = torch.abs(raw_n[:, 2]) + 0.5
    normal = F.normalize(raw_n, p=2, dim=1)

    # 1. Test Exact DC Conservation of Screened-Poisson PDE: mean(D_poisson) == mean(D_anchor)
    gx, gy = solver.normal_to_depth_gradient(normal, d_anchor)
    d_poisson = solver.solve_screened_poisson(gx, gy, d_anchor)

    mean_anchor = torch.mean(d_anchor, dim=(-2, -1))
    mean_poisson = torch.mean(d_poisson, dim=(-2, -1))
    dc_diff = torch.max(torch.abs(mean_poisson - mean_anchor)).item()
    print(f"  Screened-Poisson DC depth difference: {dc_diff:.8e}")
    assert dc_diff < 1e-4, f"Screened-Poisson DC conservation failed! Diff: {dc_diff}"

    # 2. Test Range Consistency: A_phi(D^(k)) ~= D_LR
    d_k = calib.project_range_null(d_poisson, d_lr, k_spatial, scale, mask_lr)
    deg_d_k = calib.apply_A(d_k, k_spatial, scale, mask_lr)

    range_err = torch.mean(torch.abs(deg_d_k - d_lr)).item()
    max_range_err = torch.max(torch.abs(deg_d_k - d_lr)).item()
    print(f"  Range consistency L1 error: {range_err:.8e}, Max error: {max_range_err:.8e}")
    assert range_err < 1e-3, f"Range consistency failed! Error: {range_err}"
    print("  -> Screened-Poisson Solver Invariants Passed!")


def test_scale_factors():
    print("\n[Test 3] Verifying Multi-Scale Support (x4, x8, x16) on 480x640 Resolution...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = 480, 640
    cfg = RiemannGDSRConfig(num_steps=2, feature_dim=32, use_checkpoint=False)
    net = RiemannGDSRNet(cfg).to(device)
    net.eval()

    rgb = torch.rand(1, 3, H, W, device=device)

    for scale in [4, 8, 16]:
        h, w = H // scale, W // scale
        d_lr = torch.rand(1, 1, h, w, device=device) + 0.5
        with torch.no_grad():
            out = net(rgb, d_lr)
        pred = out["pred"]
        print(f"  Scale x{scale}: input [1, 1, {h}, {w}] -> output {list(pred.shape)}")
        assert pred.shape == (1, 1, H, W), f"Output shape mismatch for x{scale}: {pred.shape}"
        assert torch.all(torch.isfinite(pred)), f"Non-finite output detected for x{scale}!"
    print("  -> Multi-Scale Forward Passes Passed!")


def test_peak_vram():
    print("\n[Test 4] Verifying Single-Sample Peak VRAM (< 2.5 GB on 480x640)...")
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping VRAM check.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    H, W = 480, 640
    scale = 8
    h, w = H // scale, W // scale

    cfg = RiemannGDSRConfig(num_steps=4, feature_dim=48, use_checkpoint=True)
    net = RiemannGDSRNet(cfg).to(device)
    net.train()

    loss_fn = GDSRLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

    rgb = torch.rand(1, 3, H, W, device=device)
    d_gt = torch.rand(1, 1, H, W, device=device) + 0.5
    d_lr = F.avg_pool2d(d_gt, kernel_size=scale, stride=scale)

    # Measure memory during training forward + backward under AMP
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = net(rgb, d_lr)
        loss, _ = loss_fn(out, d_gt, d_lr, sensor_calib=net.sensor_calib)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024**3)
    print(f"  Peak VRAM allocated: {peak_gb:.3f} GB (Threshold: 2.500 GB)")
    assert peak_gb < 2.5, f"Peak VRAM exceeded 2.5 GB! Measured: {peak_gb:.3f} GB"
    print("  -> Peak VRAM Verification Passed!")


def test_zero_init_and_amp_gradients():
    print("\n[Test 5] Verifying Zero-Init Calibration & Finite Gradients under AMP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = 120, 160
    scale = 4
    h, w = H // scale, W // scale

    cfg = RiemannGDSRConfig(num_steps=2, feature_dim=32, use_checkpoint=False)
    net = RiemannGDSRNet(cfg).to(device)

    # 1. Check Identity Initialization of Calibration at Epoch 0
    rgb = torch.rand(1, 3, H, W, device=device)
    d_gt = torch.rand(1, 1, H, W, device=device) + 1.0
    d_lr = F.avg_pool2d(d_gt, kernel_size=scale, stride=scale)

    with torch.no_grad():
        out = net(rgb, d_lr)

    alpha = out["alpha"]
    beta = out["beta"]
    max_alpha_dev = torch.max(torch.abs(alpha - 1.0)).item()
    max_beta_dev = torch.max(torch.abs(beta)).item()

    print(f"  Epoch 0 max |alpha - 1.0|: {max_alpha_dev:.6f}")
    print(f"  Epoch 0 max |beta - 0.0|: {max_beta_dev:.6f}")
    assert max_alpha_dev < 1e-3, f"Alpha identity init violated! Dev: {max_alpha_dev}"
    assert max_beta_dev < 1e-3, f"Beta zero init violated! Dev: {max_beta_dev}"

    # 2. Check Finite Gradients under FP16 AMP
    net.train()
    loss_fn = GDSRLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    optimizer.zero_grad()

    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
        out = net(rgb, d_lr)
        loss, _ = loss_fn(out, d_gt, d_lr, sensor_calib=net.sensor_calib)

    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    loss.backward()

    nan_params = []
    zero_grad_params = []
    for name, p in net.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                zero_grad_params.append(name)
            elif not torch.all(torch.isfinite(p.grad)):
                nan_params.append(name)

    print(f"  Checked {len(list(net.parameters()))} parameter tensors.")
    assert len(nan_params) == 0, f"NaN/Inf gradients in parameters: {nan_params}"
    assert len(zero_grad_params) == 0, f"Missing gradients in parameters: {zero_grad_params}"
    print("  -> Zero-Init & Finite AMP Gradients Passed!")


def test_mini_batch_overfitting():
    print("\n[Test 6] Verifying 5-Step Mini-Batch Overfitting Optimization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    H, W = 128, 128
    scale = 4
    h, w = H // scale, W // scale

    # Synthetic sample with fine high frequency geometric pattern
    rgb = torch.rand(1, 3, H, W, device=device)
    d_gt = torch.ones(1, 1, H, W, device=device) * 2.0
    d_gt[:, :, : H // 2, : W // 2] = 3.0
    d_gt[:, :, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    d_lr = F.avg_pool2d(d_gt, kernel_size=scale, stride=scale)

    cfg = RiemannGDSRConfig(num_steps=2, feature_dim=32, use_checkpoint=False)
    net = RiemannGDSRNet(cfg).to(device)
    net.train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    loss_fn = GDSRLoss().to(device)

    rmse_history = []
    loss_history = []

    def compute_rmse(pred, gt):
        return torch.sqrt(torch.mean((pred - gt) ** 2)).item()

    # Step 0 initial RMSE
    with torch.no_grad():
        init_out = net(rgb, d_lr)
        init_rmse = compute_rmse(init_out["pred"], d_gt)
    rmse_history.append(init_rmse)
    print(f"  Step 0 (Initial) RMSE: {init_rmse:.6f}")

    for step in range(1, 6):
        optimizer.zero_grad()
        out = net(rgb, d_lr)
        loss, _ = loss_fn(out, d_gt, d_lr, sensor_calib=net.sensor_calib)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cur_out = net(rgb, d_lr)
            cur_rmse = compute_rmse(cur_out["pred"], d_gt)
        rmse_history.append(cur_rmse)
        loss_history.append(loss.item())
        print(f"  Step {step} Loss: {loss.item():.6f}, RMSE: {cur_rmse:.6f}")

    # Verify that RMSE monotonically decreases across training steps
    print("  RMSE progression:", [f"{r:.6f}" for r in rmse_history])
    for s in range(1, len(rmse_history)):
        assert rmse_history[s] < rmse_history[s - 1], (
            f"Monotonic RMSE decrease violated at step {s}: {rmse_history[s]} >= {rmse_history[s - 1]}"
        )
    assert rmse_history[-1] < rmse_history[0]
    print("  -> 5-Step Mini-Batch Overfitting Passed (Monotonically Decreasing)!")


def main():
    print("=" * 70)
    print("Starting RiemannGDSR Comprehensive Test Suite")
    print("=" * 70)

    try:
        test_manifold_invariants()
        test_screened_poisson_solver()
        test_scale_factors()
        test_peak_vram()
        test_zero_init_and_amp_gradients()
        test_mini_batch_overfitting()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED SUCCESSFULLY! Exit code 0.")
        print("=" * 70)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[TEST FAILURE] Assertion violated: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[UNEXPECTED ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
