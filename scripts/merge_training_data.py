#!/usr/bin/env python3
"""Merge training data from phase1 and phase2"""
import pandas as pd

# 读取两个训练集
vol1 = pd.read_csv("data/raw/dataset_60/training/volume(table 6)_training.csv")
vol2 = pd.read_csv("data/raw/dataset_60/dataSet_phase2/volume(table 6)_training2.csv")

# 合并
vol_merged = pd.concat([vol1, vol2], ignore_index=True)
vol_merged = vol_merged.sort_values('time').reset_index(drop=True)

# 保存
vol_merged.to_csv("data/processed/volume_training_merged.csv", index=False)
print(f"Merged: {len(vol1)} + {len(vol2)} = {len(vol_merged)} rows")
