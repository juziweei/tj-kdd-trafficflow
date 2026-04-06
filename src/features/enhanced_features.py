"""Enhanced feature engineering from official data only."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

ENHANCED_ZERO_FEATURES = {
    "slot_mean": 0.0,
    "slot_std": 0.0,
    "slot_count": 0.0,
    "slot_median": 0.0,
    "slot_p10": 0.0,
    "slot_p90": 0.0,
    "slot_iqr": 0.0,
    "slot_max": 0.0,
    "slot_min": 0.0,
    "slot_range": 0.0,
    "slot_cv": 0.0,
    "slot_skew": 0.0,
    "slot_slope": 0.0,
    "recent_mean": 0.0,
    "recent_std": 0.0,
    "recent_count": 0.0,
    "recent_median": 0.0,
    "recent_p10": 0.0,
    "recent_p90": 0.0,
    "recent_iqr": 0.0,
    "recent_max": 0.0,
    "recent_min": 0.0,
    "recent_range": 0.0,
    "recent_cv": 0.0,
    "recent_skew": 0.0,
    "recent_slope": 0.0,
    "rush_mean": 0.0,
    "rush_std": 0.0,
    "rush_count": 0.0,
    "rush_median": 0.0,
    "rush_p10": 0.0,
    "rush_p90": 0.0,
    "rush_iqr": 0.0,
    "rush_max": 0.0,
    "rush_min": 0.0,
    "rush_range": 0.0,
    "rush_cv": 0.0,
    "rush_skew": 0.0,
    "rush_slope": 0.0,
    "short_trend": 0.0,
    "long_trend": 0.0,
    "volatility": 0.0,
    "cv": 0.0,
    "lag1_slot_ratio": 0.0,
    "lag1_slot_diff": 0.0,
    "lag1_recent_rank": 0.0,
    "lag1_slot_zscore": 0.0,
    "lag1_recent_zscore": 0.0,
    "lag72_slot_ratio": 0.0,
    "lag72_recent_ratio": 0.0,
    "lag1_rush_ratio": 0.0,
    "lag1_rush_diff": 0.0,
    "temp_morning": 0.0,
    "rain_rush": 0.0,
    "wind_temp": 0.0,
    **{f"w{w}_{stat}": 0.0 for w in [3, 6, 12, 36] for stat in ["max", "min", "mean", "std", "slope", "diff_mean", "ratio", "rank"]},
    "w6_w36_ratio": 0.0,
    "w3_w12_diff": 0.0,
    "w6_w12_slope_diff": 0.0,
}

SLOT_STAT_KEYS = [
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
]

RECENT_STAT_KEYS = [
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
]

RUSH_STAT_KEYS = [
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
]

ALIGNMENT_KEYS = [
    "lag1_slot_ratio",
    "lag1_slot_diff",
    "lag1_recent_rank",
    "lag1_slot_zscore",
    "lag1_recent_zscore",
    "lag72_slot_ratio",
    "lag72_recent_ratio",
    "lag1_rush_ratio",
    "lag1_rush_diff",
]

SLOT_STAT_ZERO = {k: 0.0 for k in SLOT_STAT_KEYS}
RECENT_STAT_ZERO = {k: 0.0 for k in RECENT_STAT_KEYS}
RUSH_STAT_ZERO = {k: 0.0 for k in RUSH_STAT_KEYS}
ALIGNMENT_ZERO = {k: 0.0 for k in ALIGNMENT_KEYS}


def _resolve_env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def slot_of_day(ts: pd.Timestamp) -> int:
    """时段索引 (0-71)"""
    return int((ts.hour * 60 + ts.minute) // 20)


def _horizon_index(ts: pd.Timestamp) -> int:
    if ts.hour in (8, 17):
        return int(ts.minute / 20) + 1
    return int(ts.minute / 20) + 4


def _restrict_recent(series: pd.Series, cutoff_ts: pd.Timestamp, lookback_days: int) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    start_ts = cutoff_ts - pd.Timedelta(days=max(1, int(lookback_days)))
    return series[(series.index >= start_ts) & (series.index <= cutoff_ts)]


def _safe_skew(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 1e-8:
        return 0.0
    centered = arr - float(np.mean(arr))
    m3 = float(np.mean(centered ** 3))
    return float(m3 / (std ** 3 + 1e-8))


def _safe_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(arr))
    num = float(np.sum((x - x_mean) * (arr - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    if den <= 1e-8:
        return 0.0
    return num / den


def compute_slot_statistics(
    series: pd.Series,
    slot: int,
) -> dict[str, float]:
    """计算同时段历史统计"""
    if series is None or series.empty:
        return dict(SLOT_STAT_ZERO)

    slots = ((series.index.hour * 60 + series.index.minute) // 20).astype(int)
    matched = series[slots == slot]

    if matched.empty:
        return dict(SLOT_STAT_ZERO)

    vals = matched.to_numpy(dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    q10 = float(matched.quantile(0.10))
    q90 = float(matched.quantile(0.90))
    q25 = float(matched.quantile(0.25))
    q75 = float(matched.quantile(0.75))
    max_val = float(np.max(vals))
    min_val = float(np.min(vals))

    return {
        "slot_mean": mean_val,
        "slot_std": std_val,
        "slot_count": float(len(matched)),
        "slot_median": float(matched.median()),
        "slot_p10": q10,
        "slot_p90": q90,
        "slot_iqr": q75 - q25,
        "slot_max": max_val,
        "slot_min": min_val,
        "slot_range": max_val - min_val,
        "slot_cv": float(std_val / (mean_val + 1e-8)),
        "slot_skew": _safe_skew(vals),
        "slot_slope": _safe_slope(vals),
    }


def compute_recent_statistics(series: pd.Series) -> dict[str, float]:
    if series is None or series.empty:
        return dict(RECENT_STAT_ZERO)
    vals = series.to_numpy(dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    q10 = float(series.quantile(0.10))
    q90 = float(series.quantile(0.90))
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    max_val = float(np.max(vals))
    min_val = float(np.min(vals))
    return {
        "recent_mean": mean_val,
        "recent_std": std_val,
        "recent_count": float(len(series)),
        "recent_median": float(series.median()),
        "recent_p10": q10,
        "recent_p90": q90,
        "recent_iqr": q75 - q25,
        "recent_max": max_val,
        "recent_min": min_val,
        "recent_range": max_val - min_val,
        "recent_cv": float(std_val / (mean_val + 1e-8)),
        "recent_skew": _safe_skew(vals),
        "recent_slope": _safe_slope(vals),
    }


def compute_rush_statistics(series: pd.Series) -> dict[str, float]:
    if series is None or series.empty:
        return dict(RUSH_STAT_ZERO)

    hours = series.index.hour
    rush_mask = (hours >= 8) & (hours <= 10) | (hours >= 17) & (hours <= 19)
    rush = series[rush_mask]
    if rush.empty:
        return dict(RUSH_STAT_ZERO)
    vals = rush.to_numpy(dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    q10 = float(rush.quantile(0.10))
    q90 = float(rush.quantile(0.90))
    q25 = float(rush.quantile(0.25))
    q75 = float(rush.quantile(0.75))
    max_val = float(np.max(vals))
    min_val = float(np.min(vals))
    return {
        "rush_mean": mean_val,
        "rush_std": std_val,
        "rush_count": float(len(rush)),
        "rush_median": float(rush.median()),
        "rush_p10": q10,
        "rush_p90": q90,
        "rush_iqr": q75 - q25,
        "rush_max": max_val,
        "rush_min": min_val,
        "rush_range": max_val - min_val,
        "rush_cv": float(std_val / (mean_val + 1e-8)),
        "rush_skew": _safe_skew(vals),
        "rush_slope": _safe_slope(vals),
    }


def compute_trend_features(lags: dict[int, float]) -> dict[str, float]:
    """趋势特征"""
    if 1 not in lags or 72 not in lags:
        return {"short_trend": 0.0, "long_trend": 0.0}

    short_trend = (lags[1] - lags.get(72, lags[1])) / 72 if 72 in lags else 0.0
    long_trend = (lags.get(72, 0) - lags.get(504, lags.get(72, 0))) / 432 if 504 in lags else 0.0

    return {"short_trend": short_trend, "long_trend": long_trend}


def compute_volatility_features(history: pd.Series, window: int = 6) -> dict[str, float]:
    """波动性特征"""
    if history.empty or len(history) < window:
        return {"volatility": 0.0, "cv": 0.0}

    recent = history.iloc[-window:]
    mean_val = recent.mean()
    std_val = recent.std()

    return {
        "volatility": float(std_val),
        "cv": float(std_val / (mean_val + 1e-8))
    }


def compute_multi_window_features(series: pd.Series, cutoff_ts: pd.Timestamp, lag1: float) -> dict[str, float]:
    """多窗口统计特征 - KDD第一名方案"""
    if series is None or series.empty:
        return {f"w{w}_{stat}": 0.0 for w in [3, 6, 12, 36]
                for stat in ["max", "min", "mean", "std", "slope", "diff_mean", "ratio", "rank"]}

    features = {}
    windows = [3, 6, 12, 36]  # 1h, 2h, 4h, 12h

    for w in windows:
        start = cutoff_ts - pd.Timedelta(minutes=20*w)
        window_data = series[(series.index > start) & (series.index <= cutoff_ts)]

        if window_data.empty or len(window_data) < 2:
            features.update({f"w{w}_max": 0.0, f"w{w}_min": 0.0, f"w{w}_mean": 0.0,
                           f"w{w}_std": 0.0, f"w{w}_slope": 0.0, f"w{w}_diff_mean": 0.0,
                           f"w{w}_ratio": 0.0, f"w{w}_rank": 0.0})
            continue

        vals = window_data.to_numpy(dtype=float)
        mean_val = float(np.mean(vals))
        features[f"w{w}_max"] = float(np.max(vals))
        features[f"w{w}_min"] = float(np.min(vals))
        features[f"w{w}_mean"] = mean_val
        features[f"w{w}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        features[f"w{w}_slope"] = _safe_slope(vals)
        features[f"w{w}_diff_mean"] = vals[-1] - mean_val if len(vals) > 0 else 0.0
        features[f"w{w}_ratio"] = lag1 / max(abs(mean_val), 1e-6)
        features[f"w{w}_rank"] = float((vals <= lag1).sum()) / float(len(vals))

    return features


def compute_cross_window_features(series: pd.Series, cutoff_ts: pd.Timestamp) -> dict[str, float]:
    """跨窗口交互特征"""
    if series is None or series.empty:
        return {"w6_w36_ratio": 0.0, "w3_w12_diff": 0.0, "w6_w12_slope_diff": 0.0}

    features = {}
    w6_start = cutoff_ts - pd.Timedelta(minutes=120)
    w36_start = cutoff_ts - pd.Timedelta(minutes=720)
    w3_start = cutoff_ts - pd.Timedelta(minutes=60)
    w12_start = cutoff_ts - pd.Timedelta(minutes=240)

    w6_data = series[(series.index > w6_start) & (series.index <= cutoff_ts)]
    w36_data = series[(series.index > w36_start) & (series.index <= cutoff_ts)]
    w3_data = series[(series.index > w3_start) & (series.index <= cutoff_ts)]
    w12_data = series[(series.index > w12_start) & (series.index <= cutoff_ts)]

    w6_mean = w6_data.mean() if not w6_data.empty else 0.0
    w36_mean = w36_data.mean() if not w36_data.empty else 0.0
    w3_mean = w3_data.mean() if not w3_data.empty else 0.0
    w12_mean = w12_data.mean() if not w12_data.empty else 0.0

    features["w6_w36_ratio"] = w6_mean / max(abs(w36_mean), 1e-6)
    features["w3_w12_diff"] = w3_mean - w12_mean

    w6_slope = _safe_slope(w6_data.to_numpy()) if len(w6_data) >= 2 else 0.0
    w12_slope = _safe_slope(w12_data.to_numpy()) if len(w12_data) >= 2 else 0.0
    features["w6_w12_slope_diff"] = w6_slope - w12_slope

    return features


def compute_weather_interactions(
    weather: dict[str, float],
    ts: pd.Timestamp,
    enabled: bool | None = None,
) -> dict[str, float]:
    """天气交互特征"""
    if not _resolve_weather_interactions_enabled(enabled):
        return {"temp_morning": 0.0, "rain_rush": 0.0, "wind_temp": 0.0}

    is_morning = 1.0 if 6 <= ts.hour < 12 else 0.0
    is_rush = 1.0 if ts.hour in [8, 9, 17, 18] else 0.0

    return {
        "temp_morning": weather.get("weather_temperature", 0) * is_morning,
        "rain_rush": weather.get("weather_precipitation", 0) * is_rush,
        "wind_temp": weather.get("weather_wind_speed", 0) * weather.get("weather_temperature", 0)
    }


def compute_alignment_features(
    lags: dict[int, float],
    slot_stats: dict[str, float],
    recent_series: pd.Series,
    recent_stats: dict[str, float],
    rush_stats: dict[str, float],
) -> dict[str, float]:
    lag1 = float(lags.get(1, 0.0))
    lag72 = float(lags.get(72, lag1))
    slot_mean = float(slot_stats.get("slot_mean", 0.0))
    slot_std = float(slot_stats.get("slot_std", 0.0))
    recent_mean = float(recent_stats.get("recent_mean", 0.0))
    recent_std = float(recent_stats.get("recent_std", 0.0))
    rush_mean = float(rush_stats.get("rush_mean", 0.0))
    ratio = lag1 / max(abs(slot_mean), 1e-6)
    diff = lag1 - slot_mean

    if recent_series is None or recent_series.empty:
        rank = 0.0
    else:
        rank = float((recent_series <= lag1).sum()) / float(len(recent_series))

    return {
        "lag1_slot_ratio": ratio,
        "lag1_slot_diff": diff,
        "lag1_recent_rank": rank,
        "lag1_slot_zscore": (lag1 - slot_mean) / (slot_std + 1e-6),
        "lag1_recent_zscore": (lag1 - recent_mean) / (recent_std + 1e-6),
        "lag72_slot_ratio": lag72 / max(abs(slot_mean), 1e-6),
        "lag72_recent_ratio": lag72 / max(abs(recent_mean), 1e-6),
        "lag1_rush_ratio": lag1 / max(abs(rush_mean), 1e-6),
        "lag1_rush_diff": lag1 - rush_mean,
    }


def _resolve_strict_past_only(strict_past_only: bool | None) -> bool:
    if strict_past_only is not None:
        return bool(strict_past_only)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_STRICT_PAST_ONLY", default=True)


def _resolve_weather_interactions_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_WEATHER_INTERACTIONS", default=True)


def _resolve_enhanced_features_enabled() -> bool:
    return _resolve_env_enabled("TRAFFIC_ENHANCED_FEATURES", default=True)


def _resolve_slot_statistics_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_SLOT_STATS", default=True)


def _resolve_trend_features_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_TREND", default=True)


def _resolve_volatility_features_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_VOLATILITY", default=True)


def _resolve_recent_statistics_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_RECENT_STATS", default=False)


def _resolve_rush_statistics_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_RUSH_STATS", default=False)


def _resolve_alignment_features_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_ALIGNMENT", default=False)


def _resolve_dense_slice_gating_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return _resolve_env_enabled("TRAFFIC_ENHANCED_DENSE_SLICE_GATING", default=False)


def _normalize_dense_target_slices(raw: tuple[str, ...] | list[str] | None) -> set[str]:
    if raw is None:
        return set()
    out: set[str] = set()
    for x in raw:
        text = str(x).strip().lower()
        if not text:
            continue
        out.add(text)
    return out


def _normalize_dense_slice_feature_groups(
    raw: dict[str, tuple[str, ...] | list[str] | str] | None,
) -> dict[str, set[str]]:
    if not isinstance(raw, dict):
        return {}

    alias_map = {
        "slot": "slot",
        "slot_stats": "slot",
        "recent": "recent",
        "recent_stats": "recent",
        "rush": "rush",
        "rush_stats": "rush",
        "alignment": "alignment",
        "align": "alignment",
    }
    full = {"slot", "recent", "rush", "alignment"}
    out: dict[str, set[str]] = {}
    for key, groups_raw in raw.items():
        slice_key = str(key).strip().lower()
        if not slice_key:
            continue
        if isinstance(groups_raw, str):
            group_items = [groups_raw]
        elif isinstance(groups_raw, (list, tuple, set)):
            group_items = list(groups_raw)
        else:
            continue

        groups: set[str] = set()
        for x in group_items:
            g = str(x).strip().lower()
            if not g:
                continue
            if g == "all":
                groups = set(full)
                break
            norm = alias_map.get(g)
            if norm is not None:
                groups.add(norm)
        if groups:
            out[slice_key] = groups
    return out


def build_enhanced_features(
    key: tuple[int, int],
    ts: pd.Timestamp,
    history: dict[tuple[int, int], pd.Series],
    lags: dict[int, float],
    weather: dict[str, float],
    strict_past_only: bool | None = None,
    slot_statistics_enabled: bool | None = None,
    recent_statistics_enabled: bool | None = None,
    rush_statistics_enabled: bool | None = None,
    alignment_features_enabled: bool | None = None,
    dense_slice_gating_enabled: bool | None = None,
    dense_target_slices: tuple[str, ...] | list[str] | None = None,
    dense_slice_feature_groups: dict[str, tuple[str, ...] | list[str] | str] | None = None,
    trend_features_enabled: bool | None = None,
    volatility_features_enabled: bool | None = None,
    weather_interactions_enabled: bool | None = None,
) -> dict[str, float]:
    """构建所有增强特征"""
    if not _resolve_enhanced_features_enabled():
        return dict(ENHANCED_ZERO_FEATURES)

    features = {}
    use_strict_past_only = _resolve_strict_past_only(strict_past_only)
    cutoff_ts = pd.Timestamp(ts) - pd.Timedelta(minutes=20)
    lookback_days = 7
    base_series = history.get(key)
    if base_series is None:
        series_past = pd.Series(dtype=float)
    else:
        if use_strict_past_only:
            # Guardrail: strictly use information available before the forecast timestamp.
            series_past = base_series[base_series.index <= cutoff_ts]
        else:
            # Legacy behavior for regression audit only.
            series_past = base_series

    recent_series = _restrict_recent(series_past, cutoff_ts=cutoff_ts, lookback_days=lookback_days)
    dense_gating_on = _resolve_dense_slice_gating_enabled(dense_slice_gating_enabled)
    dense_target_set = _normalize_dense_target_slices(dense_target_slices)
    dense_group_map = _normalize_dense_slice_feature_groups(dense_slice_feature_groups)
    slice_key = f"{int(key[0])}_{int(key[1])}_h{int(_horizon_index(pd.Timestamp(ts)))}".lower()
    dense_slice_active = True
    if dense_gating_on:
        dense_slice_active = slice_key in dense_target_set
    dense_group_allow = {"slot", "recent", "rush", "alignment"} if dense_slice_active else set()
    if dense_slice_active and slice_key in dense_group_map:
        dense_group_allow = dense_group_map[slice_key]

    if _resolve_slot_statistics_enabled(slot_statistics_enabled) and dense_slice_active and ("slot" in dense_group_allow):
        slot = slot_of_day(ts)
        slot_stats = compute_slot_statistics(recent_series, slot)
    else:
        slot_stats = dict(SLOT_STAT_ZERO)
    features.update(slot_stats)
    recent_stats = compute_recent_statistics(recent_series)
    if (
        _resolve_recent_statistics_enabled(recent_statistics_enabled)
        and dense_slice_active
        and ("recent" in dense_group_allow)
    ):
        features.update(recent_stats)
    else:
        features.update(RECENT_STAT_ZERO)

    rush_stats = compute_rush_statistics(recent_series)
    if _resolve_rush_statistics_enabled(rush_statistics_enabled) and dense_slice_active and ("rush" in dense_group_allow):
        features.update(rush_stats)
    else:
        features.update(RUSH_STAT_ZERO)

    if _resolve_trend_features_enabled(trend_features_enabled):
        trend = compute_trend_features(lags)
    else:
        trend = {"short_trend": 0.0, "long_trend": 0.0}
    features.update(trend)

    if _resolve_volatility_features_enabled(volatility_features_enabled):
        vol = compute_volatility_features(recent_series)
    else:
        vol = {"volatility": 0.0, "cv": 0.0}
    features.update(vol)
    if _resolve_alignment_features_enabled(alignment_features_enabled) and dense_slice_active and (
        "alignment" in dense_group_allow
    ):
        features.update(compute_alignment_features(lags, slot_stats, recent_series, recent_stats, rush_stats))
    else:
        features.update(ALIGNMENT_ZERO)

    weather_int = compute_weather_interactions(
        weather,
        ts,
        enabled=weather_interactions_enabled,
    )
    features.update(weather_int)

    # 多窗口特征
    lag1 = float(lags.get(1, 0.0))
    multi_window = compute_multi_window_features(series_past, cutoff_ts, lag1)
    features.update(multi_window)

    # 跨窗口交互
    cross_window = compute_cross_window_features(series_past, cutoff_ts)
    features.update(cross_window)

    return features
