#!/usr/bin/env python3
"""Run v6 with Nash-Gradient Fusion"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.nash_gradient_fusion import NashGradientFusion, Agent
import numpy as np
import pandas as pd

# 读取v7运行结果
run_dir = PROJECT_ROOT / "outputs/runs/strong_backbone_fusion_20260329_v7_5fold_stable"
preds = pd.read_csv(run_dir / "validation_predictions.csv")

# 提取三路预测
baseline_pred = preds["baseline_prediction"].values
gbdt_full_pred = preds["gbdt_full_prediction"].values
gbdt_target_pred = preds["gbdt_target_prediction"].values
y_true = preds["actual"].values

predictions = {
    "baseline": baseline_pred,
    "gbdt_full": gbdt_full_pred,
    "gbdt_target": gbdt_target_pred
}

agents = [
    Agent("baseline", None, 0.35),
    Agent("gbdt_full", None, 0.40),
    Agent("gbdt_target", None, 0.25)
]

print("=== Nash-Gradient Fusion Optimization ===\n")
optimizer = NashGradientFusion(agents, learning_rate=0.05)
weights = optimizer.optimize(predictions, y_true, max_iters=100)

print(f"Original weights: [0.35, 0.40, 0.25]")
print(f"Nash optimal: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")

# 计算新MAPE
fused = sum(w * predictions[a.name] for w, a in zip(weights, agents))
mape = np.mean(np.abs((y_true - fused) / (y_true + 1e-8))) * 100
print(f"\nNash-fused MAPE: {mape:.4f}")
print(f"Original MAPE: 16.7868")
print(f"Improvement: {16.7868 - mape:.4f}")
