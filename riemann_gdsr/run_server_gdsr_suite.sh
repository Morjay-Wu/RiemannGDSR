#!/usr/bin/env bash
set -e

cd /home/WUtongquan/model/cvpr
source /home/WUtongquan/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

mkdir -p runs/riemann_gdsr_nyuv2_x4 runs/riemann_gdsr_nyuv2_x8_clean runs/riemann_gdsr_nyuv2_x16 runs/sota_reports

echo "=== [1/3] Training RiemannGDSR on NYUv2 x4 (30 Epochs, Batch Size 4, Clean Protocol) ==="
python riemann_gdsr/train.py \
    --dataset nyuv2 \
    --data_root /home/WUtongquan/dataset/NYUv2 \
    --scale 4 \
    --epochs 30 \
    --batch_size 4 \
    --num_workers 4 \
    --lr 3e-4 \
    --save_dir runs/riemann_gdsr_nyuv2_x4

echo "=== Evaluating SOTA & RiemannGDSR on x4 across all benchmarks ==="
python riemann_gdsr/eval_sota.py \
    --dataset all \
    --data_root /home/WUtongquan/dataset \
    --scale 4 \
    --weights runs/riemann_gdsr_nyuv2_x4/best.pt \
    --out_file runs/sota_reports/sota_x4.md

echo "=== [2/3] Training RiemannGDSR on NYUv2 x8 (30 Epochs, Batch Size 4, Clean Protocol) ==="
python riemann_gdsr/train.py \
    --dataset nyuv2 \
    --data_root /home/WUtongquan/dataset/NYUv2 \
    --scale 8 \
    --epochs 30 \
    --batch_size 4 \
    --num_workers 4 \
    --lr 3e-4 \
    --resume runs/riemann_gdsr_nyuv2_x4/best.pt \
    --save_dir runs/riemann_gdsr_nyuv2_x8_clean

echo "=== Evaluating SOTA & RiemannGDSR on x8 across all benchmarks ==="
python riemann_gdsr/eval_sota.py \
    --dataset all \
    --data_root /home/WUtongquan/dataset \
    --scale 8 \
    --weights runs/riemann_gdsr_nyuv2_x8_clean/best.pt \
    --out_file runs/sota_reports/sota_x8.md

echo "=== [3/3] Training RiemannGDSR on NYUv2 x16 (30 Epochs, Batch Size 4, Clean Protocol) ==="
python riemann_gdsr/train.py \
    --dataset nyuv2 \
    --data_root /home/WUtongquan/dataset/NYUv2 \
    --scale 16 \
    --epochs 30 \
    --batch_size 4 \
    --num_workers 4 \
    --lr 3e-4 \
    --resume runs/riemann_gdsr_nyuv2_x8_clean/best.pt \
    --save_dir runs/riemann_gdsr_nyuv2_x16

echo "=== Evaluating SOTA & RiemannGDSR on x16 across all benchmarks ==="
python riemann_gdsr/eval_sota.py \
    --dataset all \
    --data_root /home/WUtongquan/dataset \
    --scale 16 \
    --weights runs/riemann_gdsr_nyuv2_x16/best.pt \
    --out_file runs/sota_reports/sota_x16.md

echo "=== ALL RIEMANNGDSR TRAINING & SOTA EVALUATIONS COMPLETED SUCCESSFULLY ==="
