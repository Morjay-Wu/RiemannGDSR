#!/usr/bin/env bash
set -e

cd /home/WUtongquan/model/cvpr
source /home/WUtongquan/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

echo "=== Waiting for current RiemannGDSR main suite (PID 3124158) to finish to avoid VRAM OOM ==="
while kill -0 3124158 2>/dev/null; do
    sleep 60
done

echo "=== Training 2023-2025 Top-Conference SOTA Models (DADA, SSDNet, SGNet, DORNet, DuCos) ==="
for scale in 4 8 16; do
    for sota in dada ssdnet sgnet dornet ducos; do
        resume_arg=""
        if [ "$sota" = "dornet" ] && [ -f "third_party_sota/DORNet/checkpoints/NYU_X${scale}.pth" ]; then
            resume_arg="--resume third_party_sota/DORNet/checkpoints/NYU_X${scale}.pth"
        fi
        echo ">>> Training ${sota} on NYUv2 x${scale} <<<"
        python riemann_gdsr/train_sota_gdsr.py \
            --sota "$sota" \
            --dataset nyuv2 \
            --data_root /home/WUtongquan/dataset/NYUv2 \
            --scale "$scale" \
            --epochs 12 \
            --batch_size 4 \
            --num_workers 4 \
            --lr 3e-4 \
            $resume_arg \
            --save_dir "runs/sota_${sota}_x${scale}"
    done

    if [ "$scale" = "8" ]; then
        ours_ckpt="runs/riemann_gdsr_nyuv2_x8_clean/best.pt"
    else
        ours_ckpt="runs/riemann_gdsr_nyuv2_x${scale}/best.pt"
    fi

    echo ">>> Evaluating all 12 methods (including 2023-2025 SOTA) on Scale x${scale} across 4 datasets <<<"
    python riemann_gdsr/eval_sota.py \
        --dataset all \
        --data_root /home/WUtongquan/dataset \
        --scale "$scale" \
        --weights "$ours_ckpt" \
        --sota_ckpt_dir runs \
        --out_file "runs/sota_reports/sota_full_2025_x${scale}.md"
done

echo "=== ALL 2023-2025 SOTA TRAINING & MULTI-DATASET EVALUATIONS COMPLETE ==="
