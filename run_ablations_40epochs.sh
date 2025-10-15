#!/bin/bash
# Run all ablation experiments for 40 epochs (4x longer than original 10)

set -e

echo "==================== Running TRM Ablation Studies (40 epochs) ===================="
echo ""

# 1. Baseline: 2-layer, H=3, L=6, with EMA
echo "[1/4] Running Baseline (2-layer, H=3, L=6, EMA=True)..."
python pretrain.py --config-name cfg_ablation_baseline_40ep

echo ""
echo "[2/4] Running No EMA Ablation (2-layer, H=3, L=6, EMA=False)..."
python pretrain.py --config-name cfg_ablation_no_ema_40ep

echo ""
echo "[3/4] Running Less Recursion Ablation (2-layer, H=2, L=2, EMA=True)..."
python pretrain.py --config-name cfg_ablation_less_recursion_40ep

echo ""
echo "[4/4] Running Bigger Brain Ablation (4-layer, H=3, L=3, EMA=True)..."
python pretrain.py --config-name cfg_ablation_bigger_brain_40ep

echo ""
echo "==================== All ablation experiments completed! ===================="
