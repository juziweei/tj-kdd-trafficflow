"""Leakage-safe feature builder for 20-minute traffic volume forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 6, 72, 504)
    rolling_window: int = 6
    enhanced_strict_past_only: bool | None = None
    enhanced_slot_stats: bool | None = None
    enhanced_recent_stats: bool | None = None
    enhanced_rush_stats: bool | None = None
    enhanced_alignment: bool | None = None
    enhanced_dense_slice_gating: bool | None = None
    enhanced_dense_target_slices: tuple[str, ...] | None = None
    enhanced_dense_slice_feature_groups: dict[str, tuple[str, ...] | list[str] | str] | None = None
    enhanced_trend: bool | None = None
    enhanced_volatility: bool | None = None
    enhanced_weather_interactions: bool | None = None


TARGET_ANCHOR_HOURS = (8, 17)
TARGET_HORIZON = 6

CALENDAR_FEATURE_COLUMNS = [
    "is_weekend",
    "is_holiday",
    "is_workday",
    "is_preholiday",
    "is_postholiday",
]

ENHANCED_FEATURE_COLUMNS = [
    "slot_mean",
    "slot_std",
    "slot_count",
    "slot_median",
    "slot_p10",
    "slot_p90",
    "slot_iqr",
    "slot_max",
    "slot_min",
    "slot_range",
    "slot_cv",
    "slot_skew",
    "slot_slope",
    "recent_mean",
    "recent_std",
    "recent_count",
    "recent_median",
    "recent_p10",
    "recent_p90",
    "recent_iqr",
    "recent_max",
    "recent_min",
    "recent_range",
    "recent_cv",
    "recent_skew",
    "recent_slope",
    "rush_mean",
    "rush_std",
    "rush_count",
    "rush_median",
    "rush_p10",
    "rush_p90",
    "rush_iqr",
    "rush_max",
    "rush_min",
    "rush_range",
    "rush_cv",
    "rush_skew",
    "rush_slope",
    "short_trend",
    "long_trend",
    "volatility",
    "cv",
    "lag1_slot_ratio",
    "lag1_slot_diff",
    "lag1_recent_rank",
    "lag1_slot_zscore",
    "lag1_recent_zscore",
    "lag72_slot_ratio",
    "lag72_recent_ratio",
    "lag1_rush_ratio",
    "lag1_rush_diff",
    "temp_morning",
    "rain_rush",
    "wind_temp",
]

ENHANCED_ZERO_FEATURES = {k: 0.0 for k in ENHANCED_FEATURE_COLUMNS}

# Relevant public holidays in the 2016 competition period.
HOLIDAY_DATES_2016 = {
    pd.Timestamp("2016-09-15").date(),
    pd.Timestamp("2016-09-16").date(),
    pd.Timestamp("2016-09-17").date(),
    pd.Timestamp("2016-10-01").date(),
    pd.Timestamp("2016-10-02").date(),
    pd.Timestamp("2016-10-03").date(),
    pd.Timestamp("2016-10-04").date(),
    pd.Timestamp("2016-10-05").date(),
    pd.Timestamp("2016-10-06").date(),
    pd.Timestamp("2016-10-07").date(),
}


def is_target_window(ts: pd.Timestamp) -> bool:
    """Whether timestamp is one of the 12 competition target windows per day."""
    hour = ts.hour
    minute = ts.minute
    return (hour in (8, 17) and minute in (0, 20, 40)) or (hour in (9, 18) and minute in (0, 20, 40))


def target_windows_for_days(days: Iterable[pd.Timestamp], horizon: int = TARGET_HORIZON) -> list[pd.Timestamp]:
    """Generate sorted target timestamps for each day (8:00-9:40 and 17:00-18:40)."""
    out: list[pd.Timestamp] = []
    for day in sorted(pd.Timestamp(d).normalize() for d in days):
        for anchor_hour in TARGET_ANCHOR_HOURS:
            anchor = day + pd.Timedelta(hours=anchor_hour)
            for step in range(horizon):
                out.append(anchor + pd.Timedelta(minutes=20 * step))
    return sorted(out)


def horizon_index(ts: pd.Timestamp) -> int:
    """Horizon index in [1, 6] inside an anchor block."""
    if ts.hour in (8, 17):
        return int(ts.minute / 20) + 1
    return int(ts.minute / 20) + 4


def _slot_of_day(ts: pd.Timestamp) -> int:
    return int((ts.hour * 60 + ts.minute) // 20)


def _series_slot_mean(history: dict[tuple[int, int], pd.Series], key: tuple[int, int], slot: int) -> float | None:
    series = history[key]
    if series.empty:
        return None
    slots = ((series.index.hour * 60 + series.index.minute) // 20).astype(int)
    matched = series[slots == slot]
    if matched.empty:
        return None
    return float(matched.mean())


def calendar_feature_vector(ts: pd.Timestamp) -> dict[str, float]:
    """Calendar features known at prediction time (no leakage)."""
    day = ts.date()
    prev_day = (ts - pd.Timedelta(days=1)).date()
    next_day = (ts + pd.Timedelta(days=1)).date()

    is_weekend = 1.0 if ts.dayofweek >= 5 else 0.0
    is_holiday = 1.0 if day in HOLIDAY_DATES_2016 else 0.0
    is_workday = 1.0 if (is_weekend == 0.0 and is_holiday == 0.0) else 0.0
    is_preholiday = 1.0 if next_day in HOLIDAY_DATES_2016 else 0.0
    is_postholiday = 1.0 if prev_day in HOLIDAY_DATES_2016 else 0.0

    return {
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_workday": is_workday,
        "is_preholiday": is_preholiday,
        "is_postholiday": is_postholiday,
    }


def build_feature_row(
    key: tuple[int, int],
    ts: pd.Timestamp,
    history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    cfg: FeatureConfig,
    default_value: float,
    allow_fallback: bool,
    weather: dict[str, float] | None = None,
    use_enhanced_features: bool = True,
) -> dict[str, float] | None:
    """Build features at timestamp ts using history up to ts-20min only."""
    series = history[key]
    slot = _slot_of_day(ts)

    values: dict[str, float] = {}
    lag_values: list[float] = []

    for lag in cfg.lags:
        lag_ts = ts - pd.Timedelta(minutes=20 * lag)
        val = series.get(lag_ts, np.nan)
        if pd.isna(val):
            if not allow_fallback:
                return None
            slot_mean = _series_slot_mean(history, key, slot)
            val = default_value if slot_mean is None else slot_mean
        lag_values.append(float(val))
        values[f"lag_{lag}"] = float(val)

    prev_vals: list[float] = []
    for step in range(1, cfg.rolling_window + 1):
        prev_ts = ts - pd.Timedelta(minutes=20 * step)
        val = series.get(prev_ts, np.nan)
        if not pd.isna(val):
            prev_vals.append(float(val))
    if prev_vals:
        values["mean_prev_6"] = float(np.mean(prev_vals))
    elif allow_fallback:
        values["mean_prev_6"] = float(np.mean(lag_values))
    else:
        return None

    dow = ts.dayofweek
    values["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    values["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    values["slot_sin"] = float(np.sin(2 * np.pi * slot / 72))
    values["slot_cos"] = float(np.cos(2 * np.pi * slot / 72))

    h = horizon_index(ts)
    values["horizon"] = float(h)
    values["is_morning"] = 1.0 if ts.hour in (8, 9) else 0.0

    for sk in series_keys:
        feature_name = f"series_{sk[0]}_{sk[1]}"
        values[feature_name] = 1.0 if sk == key else 0.0

    # 增强特征
    from src.features.enhanced_features import build_enhanced_features
    lag_dict = {lag: values[f"lag_{lag}"] for lag in cfg.lags}
    weather_dict = weather if weather is not None else {}
    if use_enhanced_features:
        enhanced = build_enhanced_features(
            key,
            ts,
            history,
            lag_dict,
            weather_dict,
            strict_past_only=cfg.enhanced_strict_past_only,
            slot_statistics_enabled=cfg.enhanced_slot_stats,
            recent_statistics_enabled=cfg.enhanced_recent_stats,
            rush_statistics_enabled=cfg.enhanced_rush_stats,
            alignment_features_enabled=cfg.enhanced_alignment,
            dense_slice_gating_enabled=cfg.enhanced_dense_slice_gating,
            dense_target_slices=cfg.enhanced_dense_target_slices,
            dense_slice_feature_groups=cfg.enhanced_dense_slice_feature_groups,
            trend_features_enabled=cfg.enhanced_trend,
            volatility_features_enabled=cfg.enhanced_volatility,
            weather_interactions_enabled=cfg.enhanced_weather_interactions,
        )
    else:
        enhanced = ENHANCED_ZERO_FEATURES
    values.update(enhanced)

    return values


def feature_columns(series_keys: list[tuple[int, int]], cfg: FeatureConfig, include_calendar: bool = False) -> list[str]:
    cols = [f"lag_{lag}" for lag in cfg.lags]
    cols += [
        "mean_prev_6",
        "dow_sin",
        "dow_cos",
        "slot_sin",
        "slot_cos",
        "horizon",
        "is_morning",
    ]
    cols += ENHANCED_FEATURE_COLUMNS
    if include_calendar:
        cols += CALENDAR_FEATURE_COLUMNS
    cols += [f"series_{sk[0]}_{sk[1]}" for sk in series_keys]
    return cols
