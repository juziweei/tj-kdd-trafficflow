#!/usr/bin/env python3
"""Analyze feature importance using SHAP"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# 加载v6最优模型的训练数据
print("Loading data...")
# 这里需要重新构建训练数据
# 简化版：直接分析当前特征的相关性

df = pd.read_csv("outputs/runs/strong_backbone_fusion_20260329_v6_1_0_expert_h456boost/validation_predictions.csv")
print(f"Loaded {len(df)} samples")
