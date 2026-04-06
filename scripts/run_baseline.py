#!/usr/bin/env python3
"""Run leakage-safe baseline training, backtest, and submission generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.volume_io import (
    aggregate_to_20min,
    build_series_history,
    complete_20min_grid,
    load_volume_events,
    merge_histories,
)
from src.data.weather_io import (
    WEATHER_FEATURE_COLUMNS,
    get_weather_feature_vector,
    load_weather_table,
    merge_weather_tables,
    weather_defaults,
)
from src.eval.metrics import build_error_slice_table, summarize_metrics
from src.features.volume_features import (
    FeatureConfig,
    build_feature_row,
    calendar_feature_vector,
    feature_columns,
    horizon_index,
    is_target_window,
    target_windows_for_days,
)
from src.inference.submission import build_submission, validate_submission_schema
from src.models.ridge_linear import RidgeLinearModel


@dataclass
class ForecastBundle:
    mode: str
    use_log_target: bool
    log_pred_clip: float
    feature_names: list[str]
    global_model: RidgeLinearModel
    group_models: dict[tuple[int, int, int], RidgeLinearModel]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline_v1.json",
        help="Path to baseline config JSON",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id override")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_timestamp(all_windows: pd.Series, validation_days: int) -> pd.Timestamp:
    last_day = all_windows.max().normalize()
    return last_day - pd.Timedelta(days=validation_days - 1)


def rolling_folds(
    days: list[pd.Timestamp],
    n_folds: int,
    val_days: int,
    min_train_days: int,
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    n = len(days)
    for i in range(n_folds):
        val_end = n - (n_folds - i - 1) * val_days
        val_start = val_end - val_days
        if val_start <= min_train_days:
            continue
        train_days = days[:val_start]
        val_slice = days[val_start:val_end]
        if len(train_days) < min_train_days or len(val_slice) == 0:
            continue
        folds.append((train_days, val_slice))
    return folds


def default_value_from_history(history: dict[tuple[int, int], pd.Series]) -> float:
    values = pd.concat(list(history.values()), axis=0)
    return float(values.mean())


def parse_series_key(text: str) -> tuple[int, int]:
    left, right = text.split("_", 1)
    return int(left), int(right)


def parse_series_horizon_key(text: str) -> tuple[int, int, int]:
    parts = text.split("_")
    if len(parts) != 3 or not parts[2].startswith("h"):
        raise ValueError(f"Invalid series-horizon key: {text}. Expected format like '1_0_h3'")
    tollgate_id = int(parts[0])
    direction = int(parts[1])
    horizon = int(parts[2][1:])
    if horizon < 1 or horizon > 6:
        raise ValueError(f"Invalid horizon in key {text}: {horizon}")
    return tollgate_id, direction, horizon


def resolve_weather_columns(cfg: dict, use_weather: bool) -> list[str]:
    if not use_weather:
        return []
    requested = cfg.get("feature", {}).get("weather_columns")
    if requested is None:
        return WEATHER_FEATURE_COLUMNS.copy()

    unknown = sorted(set(requested) - set(WEATHER_FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown weather feature columns: {unknown}")
    return list(requested)


def build_training_dataset(
    train_grid: pd.DataFrame,
    history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    cfg: FeatureConfig,
    train_end: pd.Timestamp,
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool = True,
    weather_table: pd.DataFrame | None = None,
    weather_defaults_map: dict[str, float] | None = None,
    weather_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    meta_rows: list[dict[str, int | pd.Timestamp]] = []

    for row in train_grid.itertuples(index=False):
        ts = pd.Timestamp(row.time_window)
        if ts >= train_end:
            continue
        if not is_target_window(ts):
            continue

        weather_values: dict[str, float] = {}
        if weather_table is not None and weather_defaults_map is not None:
            weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
            if weather_columns is not None:
                weather_values = {k: weather_values[k] for k in weather_columns}

        key = (int(row.tollgate_id), int(row.direction))
        feat = build_feature_row(
            key=key,
            ts=ts,
            history=history,
            series_keys=series_keys,
            cfg=cfg,
            default_value=default_value,
            allow_fallback=False,
            weather=weather_values,
            use_enhanced_features=use_enhanced_features,
        )
        if feat is None:
            continue

        if include_calendar:
            feat.update(calendar_feature_vector(ts))

        if weather_values:
            feat.update(weather_values)

        rows.append(feat)
        targets.append(float(row.volume))
        meta_rows.append(
            {
                "tollgate_id": int(row.tollgate_id),
                "direction": int(row.direction),
                "time_window": ts,
                "horizon": int(horizon_index(ts)),
                "day": ts.normalize(),
            }
        )

    x_df = pd.DataFrame(rows)
    y = pd.Series(targets, name="volume")
    meta_df = pd.DataFrame(meta_rows)
    return x_df, y, meta_df


def train_primary_models(
    cfg: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[ForecastBundle, dict[str, float | int | str]]:
    modeling_cfg = cfg.get("modeling", {})
    mode = str(modeling_cfg.get("mode", "single"))
    use_log_target = bool(modeling_cfg.get("use_log_target", False))
    log_pred_clip = float(modeling_cfg.get("log_pred_clip", 8.0))

    global_alpha = float(cfg["model"]["ridge_alpha"])
    group_alpha = float(modeling_cfg.get("group_ridge_alpha", global_alpha))
    min_group_samples = int(modeling_cfg.get("min_group_samples", 30))

    target = np.log1p(y_train) if use_log_target else y_train

    global_model = RidgeLinearModel(feature_names=feature_names, alpha=global_alpha)
    global_model.fit(x_train, target)

    group_models: dict[tuple[int, int, int], RidgeLinearModel] = {}
    if mode == "grouped_series_horizon":
        grouped = meta_df.groupby(["tollgate_id", "direction", "horizon"], sort=True)
        for (tollgate_id, direction, horizon), part in grouped:
            idx = part.index.to_numpy()
            if len(idx) < min_group_samples:
                continue
            x_sub = x_train.iloc[idx].reset_index(drop=True)
            y_sub = target.iloc[idx].reset_index(drop=True)

            model = RidgeLinearModel(feature_names=feature_names, alpha=group_alpha)
            model.fit(x_sub, y_sub)
            group_models[(int(tollgate_id), int(direction), int(horizon))] = model

    bundle = ForecastBundle(
        mode=mode,
        use_log_target=use_log_target,
        log_pred_clip=log_pred_clip,
        feature_names=feature_names,
        global_model=global_model,
        group_models=group_models,
    )

    stats = {
        "mode": mode,
        "use_log_target": int(use_log_target),
        "log_pred_clip": log_pred_clip,
        "group_models": len(group_models),
        "group_min_samples": min_group_samples,
        "global_alpha": global_alpha,
        "group_alpha": group_alpha,
    }
    return bundle, stats


def predict_primary_row(
    bundle: ForecastBundle,
    x_row: pd.DataFrame,
    key: tuple[int, int],
    horizon: int,
) -> float:
    model = bundle.group_models.get((key[0], key[1], horizon), bundle.global_model)
    raw = float(model.predict(x_row)[0])
    if bundle.use_log_target:
        raw = min(raw, bundle.log_pred_clip)
        return max(0.0, float(np.expm1(raw)))
    return max(0.0, raw)


def predict_primary_batch(bundle: ForecastBundle, x_df: pd.DataFrame, meta_df: pd.DataFrame) -> np.ndarray:
    preds: list[float] = []
    for i in range(len(x_df)):
        key = (int(meta_df.iloc[i]["tollgate_id"]), int(meta_df.iloc[i]["direction"]))
        h = int(meta_df.iloc[i]["horizon"])
        x_row = x_df.iloc[[i]]
        preds.append(predict_primary_row(bundle, x_row, key, h))
    return np.asarray(preds, dtype=float)


def train_residual_models(
    cfg: dict,
    primary_bundle: ForecastBundle,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[dict[tuple[int, int], RidgeLinearModel], dict[str, dict[str, float | int]]]:
    residual_cfg = cfg.get("residual", {})
    use_residual = bool(residual_cfg.get("use_residual", False))
    if not use_residual:
        return {}, {}

    target_series = residual_cfg.get("target_series", [])
    if not target_series:
        return {}, {}

    min_samples = int(residual_cfg.get("min_samples", 80))
    alpha = float(residual_cfg.get("ridge_alpha", 2.0))

    base_pred = predict_primary_batch(primary_bundle, x_train, meta_df)
    residual_values = y_train.to_numpy(dtype=float) - base_pred

    models: dict[tuple[int, int], RidgeLinearModel] = {}
    stats: dict[str, dict[str, float | int]] = {}

    for series_text in target_series:
        series_key = parse_series_key(series_text)
        mask = (meta_df["tollgate_id"] == series_key[0]) & (meta_df["direction"] == series_key[1])
        sample_count = int(mask.sum())
        if sample_count < min_samples:
            stats[series_text] = {
                "samples": sample_count,
                "trained": 0,
                "reason": f"insufficient_samples_lt_{min_samples}",
            }
            continue

        x_sub = x_train.loc[mask].reset_index(drop=True)
        y_sub = pd.Series(residual_values[mask.to_numpy()], name="residual")

        residual_model = RidgeLinearModel(feature_names=feature_names, alpha=alpha)
        residual_model.fit(x_sub, y_sub)
        models[series_key] = residual_model

        stats[series_text] = {
            "samples": sample_count,
            "trained": 1,
            "residual_mean": float(y_sub.mean()),
            "residual_std": float(y_sub.std(ddof=0)),
        }

    return models, stats


def train_horizon_bias_map(
    cfg: dict,
    primary_bundle: ForecastBundle,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_df: pd.DataFrame,
    residual_models: dict[tuple[int, int], RidgeLinearModel] | None = None,
    residual_clip_abs: float | None = None,
) -> tuple[dict[tuple[int, int, int], float], dict[str, dict[str, float | int]], float | None]:
    bias_cfg = cfg.get("bias_correction", {})
    use_horizon_bias = bool(bias_cfg.get("use_horizon_bias", False))
    if not use_horizon_bias:
        return {}, {}, None

    target_series = bias_cfg.get("target_series", [])
    if not target_series:
        return {}, {}, None

    def parse_horizon_scale(raw: dict | None) -> dict[int, float]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError("horizon_scale must be a dict keyed by horizon string/int")
        parsed: dict[int, float] = {}
        for key, value in raw.items():
            h = int(key)
            if h < 1 or h > 6:
                raise ValueError(f"Invalid horizon in horizon_scale: {h}")
            parsed[h] = float(value)
        return parsed

    default_min_samples = int(bias_cfg.get("min_samples_per_horizon", 12))
    default_clip_abs = float(bias_cfg.get("clip_abs", 8.0))
    default_horizon_scale = parse_horizon_scale(bias_cfg.get("horizon_scale", {}))

    series_params_cfg = bias_cfg.get("series_params", {})
    if not isinstance(series_params_cfg, dict):
        raise ValueError("bias_correction.series_params must be a dict keyed by 'tollgate_direction'")

    base_pred = predict_primary_batch(primary_bundle, x_train, meta_df)
    corrected = base_pred.copy()

    residual_models = residual_models or {}
    if residual_models:
        for i in range(len(x_train)):
            key = (int(meta_df.iloc[i]["tollgate_id"]), int(meta_df.iloc[i]["direction"]))
            if key not in residual_models:
                continue
            x_row = x_train.iloc[[i]]
            corr = float(residual_models[key].predict(x_row)[0])
            if residual_clip_abs is not None:
                corr = float(np.clip(corr, -residual_clip_abs, residual_clip_abs))
            corrected[i] += corr

    residual = y_train.to_numpy(dtype=float) - corrected
    frame = meta_df.copy()
    frame["residual"] = residual

    bias_map: dict[tuple[int, int, int], float] = {}
    stats: dict[str, dict[str, float | int]] = {}

    for series_text in target_series:
        series_key = parse_series_key(series_text)
        series_param = series_params_cfg.get(series_text, {})
        if not isinstance(series_param, dict):
            raise ValueError(f"bias_correction.series_params['{series_text}'] must be an object")

        min_samples = int(series_param.get("min_samples_per_horizon", default_min_samples))
        clip_abs = float(series_param.get("clip_abs", default_clip_abs))
        horizon_scale = default_horizon_scale.copy()
        horizon_scale.update(parse_horizon_scale(series_param.get("horizon_scale", {})))

        part = frame[
            (frame["tollgate_id"] == series_key[0])
            & (frame["direction"] == series_key[1])
        ]

        substats: dict[str, float | int] = {
            "samples": int(len(part)),
            "trained": 1,
            "min_samples_per_horizon": min_samples,
            "clip_abs": clip_abs,
        }
        for h in range(1, 7):
            hp = part[part["horizon"] == h]
            if len(hp) < min_samples:
                bias = 0.0
            else:
                bias = float(hp["residual"].mean())
                bias = float(np.clip(bias, -clip_abs, clip_abs))
            scale = float(horizon_scale.get(h, 1.0))
            bias = float(np.clip(bias * scale, -clip_abs, clip_abs))
            bias_map[(series_key[0], series_key[1], h)] = bias
            substats[f"h{h}_samples"] = int(len(hp))
            substats[f"h{h}_scale"] = scale
            substats[f"h{h}_bias"] = bias

        stats[series_text] = substats

    # Bias values are already clipped during training; no extra global clipping needed in inference.
    return bias_map, stats, None


def train_conditional_residual_models(
    cfg: dict,
    primary_bundle: ForecastBundle,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_df: pd.DataFrame,
    feature_names: list[str],
    residual_models: dict[tuple[int, int], RidgeLinearModel] | None = None,
    residual_clip_abs: float | None = None,
    horizon_bias_map: dict[tuple[int, int, int], float] | None = None,
    horizon_bias_clip_abs: float | None = None,
) -> tuple[
    dict[tuple[int, int, int], RidgeLinearModel],
    dict[str, dict[str, float | int]],
    dict[tuple[int, int, int], float],
    dict[tuple[int, int, int], dict[str, object]],
]:
    hslice_cfg = cfg.get("residual_hslice", {})
    use_hslice = bool(hslice_cfg.get("use_conditional_residual", False))
    if not use_hslice:
        return {}, {}, {}, {}

    target_groups = hslice_cfg.get("target_groups", [])
    if not target_groups:
        return {}, {}, {}, {}

    default_alpha = float(hslice_cfg.get("ridge_alpha", 50.0))
    default_min_samples = int(hslice_cfg.get("min_samples", 80))
    default_clip_abs = float(hslice_cfg.get("clip_abs", 8.0))
    default_use_gate = bool(hslice_cfg.get("use_confidence_gate", False))
    default_gate_quantile = float(hslice_cfg.get("gate_quantile", 0.8))
    default_gate_max_z_raw = hslice_cfg.get("gate_max_z", None)
    default_gate_max_z = float(default_gate_max_z_raw) if default_gate_max_z_raw is not None else None
    default_use_gain_gate = bool(hslice_cfg.get("use_gain_gate", False))
    default_gain_min_mean = float(hslice_cfg.get("gain_min_mean", 0.0))
    default_gain_min_count = int(hslice_cfg.get("gain_min_count", 6))
    default_gain_quantiles = hslice_cfg.get(
        "gain_quantiles",
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    if not isinstance(default_gain_quantiles, list) or not default_gain_quantiles:
        raise ValueError("residual_hslice.gain_quantiles must be a non-empty list")
    default_gain_quantiles = [float(q) for q in default_gain_quantiles]
    group_params_cfg = hslice_cfg.get("group_params", {})
    if not isinstance(group_params_cfg, dict):
        raise ValueError("residual_hslice.group_params must be a dict keyed by 'tollgate_direction_hX'")

    base_pred = predict_primary_batch(primary_bundle, x_train, meta_df)
    corrected = base_pred.copy()

    residual_models = residual_models or {}
    horizon_bias_map = horizon_bias_map or {}

    for i in range(len(x_train)):
        key = (int(meta_df.iloc[i]["tollgate_id"]), int(meta_df.iloc[i]["direction"]))
        h = int(meta_df.iloc[i]["horizon"])
        x_row = x_train.iloc[[i]]

        if key in residual_models:
            corr = float(residual_models[key].predict(x_row)[0])
            if residual_clip_abs is not None:
                corr = float(np.clip(corr, -residual_clip_abs, residual_clip_abs))
            corrected[i] += corr

        bias = float(horizon_bias_map.get((key[0], key[1], h), 0.0))
        if horizon_bias_clip_abs is not None:
            bias = float(np.clip(bias, -horizon_bias_clip_abs, horizon_bias_clip_abs))
        corrected[i] += bias

    residual_values = y_train.to_numpy(dtype=float) - corrected

    models: dict[tuple[int, int, int], RidgeLinearModel] = {}
    stats: dict[str, dict[str, float | int]] = {}
    clip_map: dict[tuple[int, int, int], float] = {}
    gate_meta: dict[tuple[int, int, int], dict[str, object]] = {}

    for group_text in target_groups:
        group_key = parse_series_horizon_key(group_text)
        group_param = group_params_cfg.get(group_text, {})
        if not isinstance(group_param, dict):
            raise ValueError(f"residual_hslice.group_params['{group_text}'] must be an object")

        alpha = float(group_param.get("ridge_alpha", default_alpha))
        min_samples = int(group_param.get("min_samples", default_min_samples))
        clip_abs = float(group_param.get("clip_abs", default_clip_abs))
        use_gate = bool(group_param.get("use_confidence_gate", default_use_gate))
        gate_quantile = float(group_param.get("gate_quantile", default_gate_quantile))
        gate_max_z_raw = group_param.get("gate_max_z", default_gate_max_z)
        gate_max_z = float(gate_max_z_raw) if gate_max_z_raw is not None else None
        use_gain_gate = bool(group_param.get("use_gain_gate", default_use_gain_gate))
        gain_min_mean = float(group_param.get("gain_min_mean", default_gain_min_mean))
        gain_min_count = int(group_param.get("gain_min_count", default_gain_min_count))
        gain_quantiles_raw = group_param.get("gain_quantiles", default_gain_quantiles)
        if not isinstance(gain_quantiles_raw, list) or not gain_quantiles_raw:
            raise ValueError(f"residual_hslice gain_quantiles for {group_text} must be non-empty list")
        gain_quantiles = [float(q) for q in gain_quantiles_raw]

        mask = (
            (meta_df["tollgate_id"] == group_key[0])
            & (meta_df["direction"] == group_key[1])
            & (meta_df["horizon"] == group_key[2])
        )
        sample_count = int(mask.sum())
        if sample_count < min_samples:
            stats[group_text] = {
                "samples": sample_count,
                "trained": 0,
                "alpha": alpha,
                "min_samples": min_samples,
                "clip_abs": clip_abs,
                "gate_enabled": int(use_gate),
                "reason": f"insufficient_samples_lt_{min_samples}",
            }
            continue

        x_sub = x_train.loc[mask].reset_index(drop=True)
        y_sub = pd.Series(residual_values[mask.to_numpy()], name="residual_hslice")
        x_sub_np = x_sub[feature_names].to_numpy(dtype=float)

        model = RidgeLinearModel(feature_names=feature_names, alpha=alpha)
        model.fit(x_sub, y_sub)
        models[group_key] = model
        clip_map[group_key] = clip_abs

        pred_sub = model.predict(x_sub).astype(float)
        pred_sub = np.clip(pred_sub, -clip_abs, clip_abs)
        gain_arr = np.abs(y_sub.to_numpy(dtype=float)) - np.abs(y_sub.to_numpy(dtype=float) - pred_sub)
        pred_abs = np.abs(pred_sub)

        center = np.mean(x_sub_np, axis=0)
        scale = np.std(x_sub_np, axis=0, ddof=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        dists = np.sqrt(np.mean(((x_sub_np - center) / scale) ** 2, axis=1))
        dist_q = float(np.quantile(dists, gate_quantile))
        threshold = dist_q if gate_max_z is None else min(dist_q, gate_max_z)
        if not use_gate:
            threshold = float("inf")

        gain_threshold = 0.0
        gain_best_quantile = 0.0
        gain_best_mean = float(np.mean(gain_arr))
        if use_gain_gate:
            best: tuple[float, float, float] | None = None  # (mean_gain, quantile, threshold)
            for q in gain_quantiles:
                q_clamped = min(1.0, max(0.0, q))
                th = float(np.quantile(pred_abs, q_clamped))
                idx = pred_abs >= th
                cnt = int(np.sum(idx))
                if cnt < gain_min_count:
                    continue
                mean_gain = float(np.mean(gain_arr[idx]))
                if best is None or mean_gain > best[0]:
                    best = (mean_gain, q_clamped, th)
            if best is None or best[0] <= gain_min_mean:
                gain_threshold = float("inf")
                gain_best_mean = best[0] if best is not None else float("-inf")
                gain_best_quantile = best[1] if best is not None else 1.0
            else:
                gain_best_mean, gain_best_quantile, gain_threshold = best

        gate_meta[group_key] = {
            "center": center,
            "scale": scale,
            "threshold": float(threshold),
            "gain_abs_threshold": float(gain_threshold),
        }

        stats[group_text] = {
            "samples": sample_count,
            "trained": 1,
            "alpha": alpha,
            "min_samples": min_samples,
            "clip_abs": clip_abs,
            "gate_enabled": int(use_gate),
            "gate_quantile": gate_quantile,
            "gate_max_z": gate_max_z if gate_max_z is not None else -1.0,
            "gate_threshold": float(threshold),
            "gate_train_dist_mean": float(np.mean(dists)),
            "gate_train_dist_p90": float(np.quantile(dists, 0.9)),
            "gain_gate_enabled": int(use_gain_gate),
            "gain_min_mean": gain_min_mean,
            "gain_min_count": gain_min_count,
            "gain_best_quantile": gain_best_quantile,
            "gain_best_mean": gain_best_mean,
            "gain_abs_threshold": float(gain_threshold),
            "residual_mean": float(y_sub.mean()),
            "residual_std": float(y_sub.std(ddof=0)),
        }

    return models, stats, clip_map, gate_meta


def run_recursive_forecast(
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    primary_bundle: ForecastBundle,
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool = True,
    weather_table: pd.DataFrame | None = None,
    weather_defaults_map: dict[str, float] | None = None,
    weather_columns: list[str] | None = None,
    residual_models: dict[tuple[int, int], RidgeLinearModel] | None = None,
    residual_clip_abs: float | None = None,
    horizon_bias_map: dict[tuple[int, int, int], float] | None = None,
    horizon_bias_clip_abs: float | None = None,
    conditional_residual_models: dict[tuple[int, int, int], RidgeLinearModel] | None = None,
    conditional_residual_clip_map: dict[tuple[int, int, int], float] | None = None,
    conditional_residual_gate_meta: dict[tuple[int, int, int], dict[str, object]] | None = None,
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, float | int | pd.Timestamp]] = []
    residual_models = residual_models or {}
    horizon_bias_map = horizon_bias_map or {}
    conditional_residual_models = conditional_residual_models or {}
    conditional_residual_clip_map = conditional_residual_clip_map or {}
    conditional_residual_gate_meta = conditional_residual_gate_meta or {}

    for ts in schedule:
        for key in series_keys:
            weather_values: dict[str, float] = {}
            if weather_table is not None and weather_defaults_map is not None:
                weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
                if weather_columns is not None:
                    weather_values = {k: weather_values[k] for k in weather_columns}

            feat = build_feature_row(
                key=key,
                ts=ts,
                history=history,
                series_keys=series_keys,
                cfg=feature_cfg,
                default_value=default_value,
                allow_fallback=True,
                weather=weather_values,
                use_enhanced_features=use_enhanced_features,
            )
            if feat is None:
                continue

            if include_calendar:
                feat.update(calendar_feature_vector(ts))

            if weather_values:
                feat.update(weather_values)

            horizon = int(horizon_index(ts))
            x = pd.DataFrame([feat], columns=primary_bundle.feature_names)
            base_pred = predict_primary_row(primary_bundle, x, key, horizon)

            residual_correction = 0.0
            if key in residual_models:
                residual_correction = float(residual_models[key].predict(x)[0])
                if residual_clip_abs is not None:
                    residual_correction = float(np.clip(residual_correction, -residual_clip_abs, residual_clip_abs))

            horizon_bias_correction = float(horizon_bias_map.get((key[0], key[1], horizon), 0.0))
            if horizon_bias_clip_abs is not None:
                horizon_bias_correction = float(np.clip(horizon_bias_correction, -horizon_bias_clip_abs, horizon_bias_clip_abs))

            conditional_residual_correction = 0.0
            conditional_gate_applied = 0
            conditional_conf_gate_applied = 0
            conditional_gain_gate_applied = 0
            conditional_gate_distance = np.nan
            conditional_gate_threshold = np.nan
            conditional_gain_threshold = np.nan
            group_key = (key[0], key[1], horizon)
            if group_key in conditional_residual_models:
                can_apply = True
                gate = conditional_residual_gate_meta.get(group_key)
                if gate is not None:
                    center = gate["center"]  # type: ignore[assignment]
                    scale = gate["scale"]  # type: ignore[assignment]
                    threshold = float(gate["threshold"])
                    gain_threshold = float(gate.get("gain_abs_threshold", 0.0))
                    conditional_gate_threshold = threshold
                    conditional_gain_threshold = gain_threshold
                    if np.isfinite(threshold):
                        x_np = x.iloc[0].to_numpy(dtype=float)
                        conditional_gate_distance = float(
                            np.sqrt(np.mean(((x_np - center) / scale) ** 2))
                        )
                        can_apply = conditional_gate_distance <= threshold
                    if can_apply:
                        conditional_conf_gate_applied = 1

                if can_apply:
                    pred_corr = float(conditional_residual_models[group_key].predict(x)[0])
                    clip_abs = conditional_residual_clip_map.get(group_key)
                    if clip_abs is not None:
                        pred_corr = float(np.clip(pred_corr, -clip_abs, clip_abs))

                    if np.isfinite(conditional_gain_threshold):
                        can_apply = abs(pred_corr) >= conditional_gain_threshold
                    if can_apply:
                        conditional_gain_gate_applied = 1
                        conditional_residual_correction = pred_corr
                    conditional_gate_applied = 1

            pred = max(
                0.0,
                base_pred + residual_correction + horizon_bias_correction + conditional_residual_correction,
            )

            history[key].loc[ts] = pred
            history[key] = history[key].sort_index()

            rec: dict[str, float | int | pd.Timestamp] = {
                "tollgate_id": int(key[0]),
                "direction": int(key[1]),
                "time_window": ts,
                "horizon": horizon,
                "base_prediction": base_pred,
                "residual_correction": residual_correction,
                "horizon_bias_correction": horizon_bias_correction,
                "conditional_residual_correction": conditional_residual_correction,
                "conditional_gate_applied": conditional_gate_applied,
                "conditional_conf_gate_applied": conditional_conf_gate_applied,
                "conditional_gain_gate_applied": conditional_gain_gate_applied,
                "conditional_gate_distance": conditional_gate_distance,
                "conditional_gate_threshold": conditional_gate_threshold,
                "conditional_gain_threshold": conditional_gain_threshold,
                "prediction": pred,
            }
            if actual_map is not None:
                rec["actual"] = float(actual_map[(key, ts)])
            records.append(rec)

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_id = args.run_id or cfg["run_id"]
    run_dir = PROJECT_ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_cfg = FeatureConfig(
        lags=tuple(cfg["feature"]["lags"]),
        rolling_window=int(cfg["feature"]["rolling_window"]),
    )

    use_weather = bool(cfg.get("feature", {}).get("use_weather", False))
    use_calendar = bool(cfg.get("feature", {}).get("use_calendar", False))
    selected_weather_columns = resolve_weather_columns(cfg, use_weather)

    residual_cfg = cfg.get("residual", {})
    use_residual = bool(residual_cfg.get("use_residual", False))
    residual_clip_abs = float(residual_cfg.get("clip_abs", 40.0)) if use_residual else None

    train_weather: pd.DataFrame | None = None
    inference_weather: pd.DataFrame | None = None
    weather_default_map: dict[str, float] | None = None

    if use_weather:
        train_weather = load_weather_table(PROJECT_ROOT / cfg["paths"]["train_weather_csv"])
        test_weather = load_weather_table(PROJECT_ROOT / cfg["paths"]["test_weather_csv"])
        inference_weather = merge_weather_tables(train_weather, test_weather)
        weather_default_map = weather_defaults(train_weather)

    train_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["train_volume_csv"])
    train_agg = aggregate_to_20min(train_events)
    train_grid = complete_20min_grid(train_agg)
    train_history = build_series_history(train_grid)
    series_keys = sorted(train_history.keys())

    split_ts = split_timestamp(train_grid["time_window"], int(cfg["validation"]["days"]))
    default_value = default_value_from_history(train_history)

    x_train, y_train, meta_train = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=selected_weather_columns,
    )
    if x_train.empty:
        raise RuntimeError("No training samples were generated")

    feature_names = feature_columns(series_keys, feature_cfg, include_calendar=use_calendar) + selected_weather_columns
    horizon_windows = int(cfg["target"]["horizon_windows"])

    rolling_cfg = cfg.get("rolling_validation", {})
    use_rolling_validation = bool(rolling_cfg.get("use", False))
    rolling_results: list[dict[str, float | int]] = []

    primary_bundle, primary_stats = train_primary_models(
        cfg=cfg,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        feature_names=feature_names,
    )

    residual_models, residual_stats = train_residual_models(
        cfg=cfg,
        primary_bundle=primary_bundle,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        feature_names=feature_names,
    )

    horizon_bias_map, horizon_bias_stats, horizon_bias_clip_abs = train_horizon_bias_map(
        cfg=cfg,
        primary_bundle=primary_bundle,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        residual_models=residual_models,
        residual_clip_abs=residual_clip_abs,
    )

    (
        conditional_residual_models,
        conditional_residual_stats,
        conditional_residual_clip_map,
        conditional_residual_gate_meta,
    ) = (
        train_conditional_residual_models(
            cfg=cfg,
            primary_bundle=primary_bundle,
            x_train=x_train,
            y_train=y_train,
            meta_df=meta_train,
            feature_names=feature_names,
            residual_models=residual_models,
            residual_clip_abs=residual_clip_abs,
            horizon_bias_map=horizon_bias_map,
            horizon_bias_clip_abs=horizon_bias_clip_abs,
        )
    )

    if use_rolling_validation:
        pre_days = sorted(pd.to_datetime(meta_train["day"].drop_duplicates().tolist()))
        folds = rolling_folds(
            days=pre_days,
            n_folds=int(rolling_cfg.get("n_folds", 3)),
            val_days=int(rolling_cfg.get("val_days", 2)),
            min_train_days=int(rolling_cfg.get("min_train_days", 10)),
        )
        for idx, (fold_train_days, fold_val_days) in enumerate(folds, start=1):
            fold_mask = meta_train["day"].isin(fold_train_days)
            x_fold_train = x_train.loc[fold_mask].reset_index(drop=True)
            y_fold_train = y_train.loc[fold_mask].reset_index(drop=True)
            meta_fold_train = meta_train.loc[fold_mask].reset_index(drop=True)
            if x_fold_train.empty:
                continue

            fold_primary_bundle, _ = train_primary_models(
                cfg=cfg,
                x_train=x_fold_train,
                y_train=y_fold_train,
                meta_df=meta_fold_train,
                feature_names=feature_names,
            )
            fold_residual_models, _ = train_residual_models(
                cfg=cfg,
                primary_bundle=fold_primary_bundle,
                x_train=x_fold_train,
                y_train=y_fold_train,
                meta_df=meta_fold_train,
                feature_names=feature_names,
            )
            fold_bias_map, _, fold_bias_clip_abs = train_horizon_bias_map(
                cfg=cfg,
                primary_bundle=fold_primary_bundle,
                x_train=x_fold_train,
                y_train=y_fold_train,
                meta_df=meta_fold_train,
                residual_models=fold_residual_models,
                residual_clip_abs=residual_clip_abs,
            )
            (
                fold_conditional_models,
                _,
                fold_conditional_clip_map,
                fold_conditional_gate_meta,
            ) = train_conditional_residual_models(
                cfg=cfg,
                primary_bundle=fold_primary_bundle,
                x_train=x_fold_train,
                y_train=y_fold_train,
                meta_df=meta_fold_train,
                feature_names=feature_names,
                residual_models=fold_residual_models,
                residual_clip_abs=residual_clip_abs,
                horizon_bias_map=fold_bias_map,
                horizon_bias_clip_abs=fold_bias_clip_abs,
            )

            fold_schedule = target_windows_for_days(fold_val_days, horizon=horizon_windows)
            fold_history = {k: s.copy() for k, s in train_history.items()}
            fold_actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] = {}
            for key in series_keys:
                for ts in fold_schedule:
                    fold_actual_map[(key, ts)] = float(fold_history[key].get(ts, np.nan))
                    fold_history[key].loc[ts] = np.nan

            fold_pred = run_recursive_forecast(
                history=fold_history,
                schedule=fold_schedule,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                primary_bundle=fold_primary_bundle,
                default_value=default_value,
                include_calendar=use_calendar,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=selected_weather_columns,
                residual_models=fold_residual_models,
                residual_clip_abs=residual_clip_abs,
                horizon_bias_map=fold_bias_map,
                horizon_bias_clip_abs=fold_bias_clip_abs,
                conditional_residual_models=fold_conditional_models,
                conditional_residual_clip_map=fold_conditional_clip_map,
                conditional_residual_gate_meta=fold_conditional_gate_meta,
                actual_map=fold_actual_map,
            )
            fold_pred = fold_pred.dropna(subset=["actual"]).reset_index(drop=True)
            if fold_pred.empty:
                continue
            fold_metrics = summarize_metrics(fold_pred)
            rolling_results.append(
                {
                    "fold": idx,
                    "train_days": int(len(fold_train_days)),
                    "val_days": int(len(fold_val_days)),
                    "overall_mape": float(fold_metrics["overall_mape"]),
                }
            )

    valid_days = sorted(
        pd.to_datetime(
            train_grid.loc[train_grid["time_window"] >= split_ts, "time_window"].dt.normalize().unique()
        )
    )
    valid_schedule = target_windows_for_days(valid_days, horizon=horizon_windows)

    validation_history = {k: s.copy() for k, s in train_history.items()}
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] = {}
    for key in series_keys:
        for ts in valid_schedule:
            actual_map[(key, ts)] = float(validation_history[key].get(ts, np.nan))
            validation_history[key].loc[ts] = np.nan

    val_pred = run_recursive_forecast(
        history=validation_history,
        schedule=valid_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        primary_bundle=primary_bundle,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=selected_weather_columns,
        residual_models=residual_models,
        residual_clip_abs=residual_clip_abs,
        horizon_bias_map=horizon_bias_map,
        horizon_bias_clip_abs=horizon_bias_clip_abs,
        conditional_residual_models=conditional_residual_models,
        conditional_residual_clip_map=conditional_residual_clip_map,
        conditional_residual_gate_meta=conditional_residual_gate_meta,
        actual_map=actual_map,
    )
    val_pred = val_pred.dropna(subset=["actual"]).reset_index(drop=True)
    gate_eligible_rows = 0
    gate_applied_rows = 0
    gate_applied_rate = None
    conf_gate_applied_rows = 0
    gain_gate_applied_rows = 0
    if "conditional_gate_threshold" in val_pred.columns and "conditional_gate_applied" in val_pred.columns:
        gate_eligible_rows = int(val_pred["conditional_gate_threshold"].notna().sum())
        gate_applied_rows = int(val_pred["conditional_gate_applied"].sum())
        if gate_eligible_rows > 0:
            gate_applied_rate = gate_applied_rows / gate_eligible_rows
    if "conditional_conf_gate_applied" in val_pred.columns:
        conf_gate_applied_rows = int(val_pred["conditional_conf_gate_applied"].sum())
    if "conditional_gain_gate_applied" in val_pred.columns:
        gain_gate_applied_rows = int(val_pred["conditional_gain_gate_applied"].sum())
    metrics = summarize_metrics(val_pred)
    slice_df = build_error_slice_table(val_pred)
    slice_path = run_dir / "validation_error_slices.csv"
    slice_df.to_csv(slice_path, index=False)

    test_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["test_volume_csv"])
    test_agg = aggregate_to_20min(test_events)
    test_grid = complete_20min_grid(test_agg)
    test_history = build_series_history(test_grid)
    merged_history = merge_histories(train_history, test_history)

    test_days = sorted(pd.to_datetime(test_grid["time_window"].dt.normalize().unique()))
    test_schedule = target_windows_for_days(test_days, horizon=horizon_windows)

    test_pred = run_recursive_forecast(
        history=merged_history,
        schedule=test_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        primary_bundle=primary_bundle,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=inference_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=selected_weather_columns,
        residual_models=residual_models,
        residual_clip_abs=residual_clip_abs,
        horizon_bias_map=horizon_bias_map,
        horizon_bias_clip_abs=horizon_bias_clip_abs,
        conditional_residual_models=conditional_residual_models,
        conditional_residual_clip_map=conditional_residual_clip_map,
        conditional_residual_gate_meta=conditional_residual_gate_meta,
    )

    submission = build_submission(test_pred)
    validate_submission_schema(submission)

    rolling_avg_mape = (
        float(np.mean([item["overall_mape"] for item in rolling_results]))
        if rolling_results
        else None
    )

    run_meta = {
        "run_id": run_id,
        "use_weather": use_weather,
        "use_calendar": use_calendar,
        "weather_columns": selected_weather_columns,
        "modeling": primary_stats,
        "use_residual": use_residual,
        "residual_target_series": residual_cfg.get("target_series", []),
        "residual_trained_series": [f"{k[0]}_{k[1]}" for k in sorted(residual_models.keys())],
        "residual_stats": residual_stats,
        "use_horizon_bias": int(len(horizon_bias_map) > 0),
        "horizon_bias_target_series": cfg.get("bias_correction", {}).get("target_series", []),
        "horizon_bias_series_params": cfg.get("bias_correction", {}).get("series_params", {}),
        "horizon_bias_stats": horizon_bias_stats,
        "horizon_bias_horizon_scale": cfg.get("bias_correction", {}).get("horizon_scale", {}),
        "use_conditional_residual": int(len(conditional_residual_models) > 0),
        "conditional_residual_target_groups": cfg.get("residual_hslice", {}).get("target_groups", []),
        "conditional_residual_gate": {
            "use_confidence_gate": bool(cfg.get("residual_hslice", {}).get("use_confidence_gate", False)),
            "gate_quantile": cfg.get("residual_hslice", {}).get("gate_quantile"),
            "gate_max_z": cfg.get("residual_hslice", {}).get("gate_max_z"),
            "use_gain_gate": bool(cfg.get("residual_hslice", {}).get("use_gain_gate", False)),
            "gain_min_mean": cfg.get("residual_hslice", {}).get("gain_min_mean"),
            "gain_min_count": cfg.get("residual_hslice", {}).get("gain_min_count"),
        },
        "conditional_residual_trained_groups": [
            f"{k[0]}_{k[1]}_h{k[2]}" for k in sorted(conditional_residual_models.keys())
        ],
        "conditional_residual_stats": conditional_residual_stats,
        "conditional_residual_gate_eligible_rows": gate_eligible_rows,
        "conditional_residual_gate_applied_rows": gate_applied_rows,
        "conditional_residual_gate_applied_rate": gate_applied_rate,
        "conditional_conf_gate_applied_rows": conf_gate_applied_rows,
        "conditional_gain_gate_applied_rows": gain_gate_applied_rows,
        "rolling_validation": {
            "enabled": use_rolling_validation,
            "folds": rolling_results,
            "avg_mape": rolling_avg_mape,
        },
        "split_timestamp": split_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": int(len(x_train)),
        "validation_rows": int(len(val_pred)),
        "validation_slice_rows": int(len(slice_df)),
        "submission_rows": int(len(submission)),
        "artifacts": {
            "validation_predictions_csv": str((run_dir / "validation_predictions.csv").relative_to(PROJECT_ROOT)),
            "validation_error_slices_csv": str(slice_path.relative_to(PROJECT_ROOT)),
        },
        "metrics": metrics,
    }

    (run_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    val_pred.to_csv(run_dir / "validation_predictions.csv", index=False)

    sub_path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{run_id}.csv"
    submission.to_csv(sub_path, index=False)

    print(json.dumps(run_meta, ensure_ascii=False, indent=2))
    print(f"submission_path={sub_path}")


if __name__ == "__main__":
    main()
