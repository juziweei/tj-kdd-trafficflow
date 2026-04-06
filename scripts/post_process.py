#!/usr/bin/env python3
"""Post-processing: smoothing + physical constraints"""
import pandas as pd
import numpy as np

def smooth_predictions(pred, window=3):
    """移动平均平滑"""
    return pd.Series(pred).rolling(window, center=True, min_periods=1).mean().values

def apply_physical_constraints(pred):
    """物理约束：非负+连续性"""
    pred = np.maximum(pred, 0.0)
    # 限制相邻窗口变化率
    for i in range(1, len(pred)):
        if abs(pred[i] - pred[i-1]) / (pred[i-1] + 1) > 2.0:
            pred[i] = pred[i-1] * 1.5
    return pred

# 测试
df = pd.read_csv("outputs/runs/strong_backbone_fusion_20260329_v9_merged_data/validation_predictions.csv")
pred_raw = df["prediction"].values
pred_smooth = smooth_predictions(pred_raw)
pred_final = apply_physical_constraints(pred_smooth)

y_true = df["actual"].values
mape_raw = np.mean(np.abs((y_true - pred_raw) / (y_true + 1e-8))) * 100
mape_final = np.mean(np.abs((y_true - pred_final) / (y_true + 1e-8))) * 100

print(f"Raw MAPE: {mape_raw:.4f}")
print(f"Post-processed MAPE: {mape_final:.4f}")
print(f"Improvement: {mape_raw - mape_final:.4f}")
