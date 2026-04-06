#!/usr/bin/env python3
"""Wrapper to add enhanced features to existing pipeline"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.enhanced_features import build_enhanced_features
import pandas as pd

def add_enhanced_to_dataframe(df: pd.DataFrame, history, weather_map) -> pd.DataFrame:
    """为现有特征DataFrame添加增强特征"""
    enhanced_rows = []

    for idx, row in df.iterrows():
        key = (int(row['tollgate_id']), int(row['direction']))
        ts = pd.Timestamp(row['time_window'])

        lags = {int(c.split('_')[1]): row[c] for c in df.columns if c.startswith('lag_')}
        weather = {c: row[c] for c in df.columns if c.startswith('weather_')}

        enhanced = build_enhanced_features(key, ts, history, lags, weather)
        enhanced_rows.append(enhanced)

    enhanced_df = pd.DataFrame(enhanced_rows, index=df.index)
    return pd.concat([df, enhanced_df], axis=1)

print("Enhanced feature wrapper ready")
