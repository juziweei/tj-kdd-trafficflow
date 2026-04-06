#!/usr/bin/env python3
"""Run stronger backbone v5: v4 trunk + post-fusion residual head."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_baseline as baseline_mod  # noqa: E402

from src.data.volume_io import (  # noqa: E402
    aggregate_to_20min,
    build_series_history,
    complete_20min_grid,
    load_volume_events,
    merge_histories,
)
from src.data.weather_io import (  # noqa: E402
    WEATHER_FEATURE_COLUMNS,
    get_weather_feature_vector,
    load_weather_table,
    merge_weather_tables,
    weather_defaults,
)
from src.eval.metrics import build_error_slice_table, summarize_metrics  # noqa: E402
from src.features.volume_features import (  # noqa: E402
    FeatureConfig,
    build_feature_row,
    calendar_feature_vector,
    feature_columns,
    horizon_index,
    is_target_window,
    target_windows_for_days,
)
from src.fusion.adaptive_weight import (  # noqa: E402
    AdaptiveFusionWeights,
    fit_adaptive_fusion_weights,
)
from src.inference.submission import build_submission, validate_submission_schema  # noqa: E402


@dataclass
class BaselineBranchBundle:
    primary_bundle: baseline_mod.ForecastBundle
    residual_models: dict[tuple[int, int], baseline_mod.RidgeLinearModel]
    residual_clip_abs: float | None
    horizon_bias_map: dict[tuple[int, int, int], float]
    horizon_bias_clip_abs: float | None
    conditional_residual_models: dict[tuple[int, int, int], baseline_mod.RidgeLinearModel]
    conditional_residual_clip_map: dict[tuple[int, int, int], float]
    conditional_residual_gate_meta: dict[tuple[int, int, int], dict[str, object]]
    stats: dict[str, object]


@dataclass
class GBDTBundle:
    feature_names: list[str]
    use_log_target: bool
    log_pred_clip: float
    global_model: XGBRegressor
    anchor_models: dict[int, XGBRegressor]


@dataclass
class FusionBundle:
    global_weights: AdaptiveFusionWeights
    anchor_weights: dict[int, AdaptiveFusionWeights]

    def resolve(self, key: tuple[int, int], horizon: int, ts: pd.Timestamp) -> tuple[float, float]:
        anchor = anchor_bucket(ts)
        weight_obj = self.anchor_weights.get(anchor, self.global_weights)
        return weight_obj.resolve(key, horizon)


@dataclass
class PostFusionResidualBundle:
    enabled: bool
    feature_names: list[str]
    models: dict[tuple[int, int], baseline_mod.RidgeLinearModel]
    clip_map: dict[tuple[int, int], float]
    horizon_allowlist: dict[tuple[int, int], set[int]]
    gate_meta: dict[tuple[int, int], dict[str, object]]
    stats: dict[str, object]


def anchor_bucket(ts: pd.Timestamp) -> int:
    hour = pd.Timestamp(ts).hour
    return 0 if hour < 12 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strong backbone v5")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "strong_backbone_v5_main.json",
        help="Path to config JSON",
    )
    parser.add_argument("--run-id", type=str, default=None)
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
    target_only: bool = True,
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
        target_flag = is_target_window(ts)
        if target_only and not target_flag:
            continue

        key = (int(row.tollgate_id), int(row.direction))
        feat = build_feature_row(
            key=key,
            ts=ts,
            history=history,
            series_keys=series_keys,
            cfg=cfg,
            default_value=default_value,
            allow_fallback=False,
        )
        if feat is None:
            continue

        if include_calendar:
            feat.update(calendar_feature_vector(ts))

        if weather_table is not None and weather_defaults_map is not None:
            weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
            if weather_columns is not None:
                weather_values = {k: weather_values[k] for k in weather_columns}
            feat.update(weather_values)

        rows.append(feat)
        targets.append(float(row.volume))
        meta_rows.append(
            {
                "tollgate_id": int(row.tollgate_id),
                "direction": int(row.direction),
                "time_window": ts,
                "horizon": int(horizon_index(ts)),
                "anchor": int(anchor_bucket(ts)),
                "day": ts.normalize(),
                "is_target": int(target_flag),
            }
        )

    x_df = pd.DataFrame(rows)
    y = pd.Series(targets, name="volume")
    meta_df = pd.DataFrame(meta_rows)
    return x_df, y, meta_df


def make_xgb(model_cfg: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 500)),
        max_depth=int(model_cfg.get("max_depth", 4)),
        learning_rate=float(model_cfg.get("learning_rate", 0.03)),
        subsample=float(model_cfg.get("subsample", 0.85)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.85)),
        reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(model_cfg.get("reg_lambda", 4.0)),
        min_child_weight=float(model_cfg.get("min_child_weight", 3.0)),
        gamma=float(model_cfg.get("gamma", 0.2)),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=int(model_cfg.get("seed", 42)),
        n_jobs=int(model_cfg.get("n_jobs", 6)),
        verbosity=0,
    )


def train_gbdt_bundle(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    feature_names: list[str],
    model_cfg: dict,
    training_cfg: dict | None = None,
) -> GBDTBundle:
    use_log_target = bool(model_cfg.get("use_log_target", True))
    log_pred_clip = float(model_cfg.get("log_pred_clip", 6.0))
    training_cfg = training_cfg or {}
    use_anchor_models = bool(training_cfg.get("use_anchor_models", True))
    min_anchor_samples = int(training_cfg.get("min_anchor_samples", 120))
    max_sample_weight = float(training_cfg.get("max_sample_weight", 100.0))

    y_raw = y_train.to_numpy(dtype=float)
    target = np.log1p(y_raw) if use_log_target else y_raw
    sample_weight = 1.0 / np.maximum(y_raw, 1.0)
    target_window_weight = float(training_cfg.get("target_window_weight", 1.0))
    off_target_weight = float(training_cfg.get("off_target_weight", 0.3))
    min_sample_weight = float(training_cfg.get("min_sample_weight", 0.05))
    target_mask = np.ones_like(sample_weight, dtype=bool)
    if "is_target" in meta_train.columns:
        target_mask = meta_train["is_target"].to_numpy(dtype=int) > 0
        time_weight = np.where(target_mask, target_window_weight, off_target_weight)
        sample_weight = sample_weight * time_weight

    # Tail-aware reweighting: emphasize extreme-volume windows while keeping leakage-safe chronology.
    tail_q_raw = training_cfg.get("tail_weight_quantile")
    tail_factor = float(training_cfg.get("tail_weight_factor", 1.0))
    tail_target_only = bool(training_cfg.get("tail_target_only", True))
    tail_anchor_factors_raw = training_cfg.get("tail_anchor_factors", {})
    if (
        tail_q_raw is not None
        and tail_factor > 1.0
        and np.isfinite(float(tail_q_raw))
        and len(y_raw) > 0
    ):
        q = float(np.clip(float(tail_q_raw), 0.5, 0.99))
        pool = y_raw[target_mask] if tail_target_only else y_raw
        if len(pool) > 0:
            threshold = float(np.quantile(pool, q))
            tail_mask = y_raw >= threshold
            if tail_target_only:
                tail_mask = tail_mask & target_mask

            tail_scale = np.ones_like(sample_weight)
            tail_scale[tail_mask] *= tail_factor

            if isinstance(tail_anchor_factors_raw, dict) and "anchor" in meta_train.columns:
                anchor_arr = meta_train["anchor"].to_numpy(dtype=int)
                for anchor_key, factor_val in tail_anchor_factors_raw.items():
                    try:
                        anchor = int(anchor_key)
                        factor = float(factor_val)
                    except (TypeError, ValueError):
                        continue
                    if factor <= 0.0:
                        continue
                    tail_scale[tail_mask & (anchor_arr == anchor)] *= factor

            sample_weight = sample_weight * tail_scale

    sample_weight = np.clip(sample_weight, a_min=min_sample_weight, a_max=None)
    sample_weight = np.clip(sample_weight, a_min=None, a_max=max_sample_weight)

    global_model = make_xgb(model_cfg)
    global_model.fit(x_train[feature_names], target, sample_weight=sample_weight)

    anchor_models: dict[int, XGBRegressor] = {}
    if use_anchor_models and "anchor" in meta_train.columns:
        for anchor in (0, 1):
            mask = meta_train["anchor"] == anchor
            n = int(mask.sum())
            if n < min_anchor_samples:
                continue
            x_sub = x_train.loc[mask, feature_names]
            y_sub = target[mask.to_numpy()]
            w_sub = sample_weight[mask.to_numpy()]
            model = make_xgb(model_cfg)
            model.fit(x_sub, y_sub, sample_weight=w_sub)
            anchor_models[int(anchor)] = model

    return GBDTBundle(
        feature_names=feature_names,
        use_log_target=use_log_target,
        log_pred_clip=log_pred_clip,
        global_model=global_model,
        anchor_models=anchor_models,
    )


def predict_gbdt(bundle: GBDTBundle, x_row: pd.DataFrame, ts: pd.Timestamp) -> float:
    anchor = anchor_bucket(ts)
    model = bundle.anchor_models.get(anchor, bundle.global_model)
    raw = float(model.predict(x_row[bundle.feature_names])[0])
    if bundle.use_log_target:
        raw = min(raw, bundle.log_pred_clip)
        return max(0.0, float(np.expm1(raw)))
    return max(0.0, raw)


def run_gbdt_recursive_forecast(
    bundle: GBDTBundle,
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    default_value: float,
    include_calendar: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, float | int | pd.Timestamp]] = []

    for ts in schedule:
        for key in series_keys:
            feat = build_feature_row(
                key=key,
                ts=ts,
                history=history,
                series_keys=series_keys,
                cfg=feature_cfg,
                default_value=default_value,
                allow_fallback=True,
            )
            if feat is None:
                continue

            if include_calendar:
                feat.update(calendar_feature_vector(ts))

            if weather_table is not None and weather_defaults_map is not None:
                weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
                weather_values = {k: weather_values[k] for k in weather_columns}
                feat.update(weather_values)

            horizon = int(horizon_index(ts))
            x = pd.DataFrame([feat], columns=bundle.feature_names)
            pred = predict_gbdt(bundle, x, ts)

            history[key].loc[ts] = pred
            history[key] = history[key].sort_index()

            rec: dict[str, float | int | pd.Timestamp] = {
                "tollgate_id": int(key[0]),
                "direction": int(key[1]),
                "time_window": ts,
                "horizon": horizon,
                "gbdt_prediction": pred,
            }
            if actual_map is not None:
                rec["actual"] = float(actual_map[(key, ts)])
            records.append(rec)

    return pd.DataFrame(records)


def select_by_days(
    x_df: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    days: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    mask = meta_df["day"].isin(days)
    return (
        x_df.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        meta_df.loc[mask].reset_index(drop=True),
    )


def prepare_history_for_schedule(
    history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    schedule: list[pd.Timestamp],
) -> tuple[dict[tuple[int, int], pd.Series], dict[tuple[tuple[int, int], pd.Timestamp], float]]:
    out = {k: s.copy() for k, s in history.items()}
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] = {}
    for key in series_keys:
        for ts in schedule:
            actual_map[(key, ts)] = float(out[key].get(ts, np.nan))
            out[key].loc[ts] = np.nan
    return out, actual_map


def default_fusion_weights(cfg: dict) -> FusionBundle:
    fusion_cfg = cfg.get("fusion", {})
    default_w = float(fusion_cfg.get("default_gbdt_weight", 0.2))
    global_weights = AdaptiveFusionWeights(
        global_gbdt_weight=default_w,
        series_gbdt_weight={},
        slice_gbdt_weight={},
    )
    return FusionBundle(global_weights=global_weights, anchor_weights={})


def fit_anchor_fusion_weights(
    fit_frame: pd.DataFrame,
    fusion_cfg: dict,
) -> tuple[FusionBundle, dict[str, object]]:
    global_fit, global_stats = fit_adaptive_fusion_weights(fit_frame, fusion_cfg)
    min_anchor_rows = int(fusion_cfg.get("min_anchor_rows", 60))
    anchor_map: dict[int, AdaptiveFusionWeights] = {}
    anchor_stats: dict[str, object] = {}
    for anchor in (0, 1):
        part = fit_frame[fit_frame["anchor"] == anchor]
        if len(part) < min_anchor_rows:
            continue
        fit_obj, stats = fit_adaptive_fusion_weights(part, fusion_cfg)
        anchor_map[anchor] = fit_obj
        anchor_stats[str(anchor)] = stats

    bundle = FusionBundle(global_weights=global_fit, anchor_weights=anchor_map)
    stats = {
        "global": global_stats,
        "anchor": anchor_stats,
        "anchor_weight_count": int(len(anchor_map)),
    }
    return bundle, stats


def train_baseline_branch(
    cfg: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    feature_names: list[str],
) -> BaselineBranchBundle:
    residual_cfg = cfg.get("residual", {})
    use_residual = bool(residual_cfg.get("use_residual", False))
    residual_clip_abs = float(residual_cfg.get("clip_abs", 40.0)) if use_residual else None

    primary_bundle, primary_stats = baseline_mod.train_primary_models(
        cfg=cfg,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        feature_names=feature_names,
    )

    residual_models, residual_stats = baseline_mod.train_residual_models(
        cfg=cfg,
        primary_bundle=primary_bundle,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        feature_names=feature_names,
    )

    horizon_bias_map, horizon_bias_stats, horizon_bias_clip_abs = baseline_mod.train_horizon_bias_map(
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
    ) = baseline_mod.train_conditional_residual_models(
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

    stats: dict[str, object] = {
        "modeling": primary_stats,
        "use_residual": int(use_residual),
        "residual_stats": residual_stats,
        "horizon_bias_stats": horizon_bias_stats,
        "conditional_residual_stats": conditional_residual_stats,
    }

    return BaselineBranchBundle(
        primary_bundle=primary_bundle,
        residual_models=residual_models,
        residual_clip_abs=residual_clip_abs,
        horizon_bias_map=horizon_bias_map,
        horizon_bias_clip_abs=horizon_bias_clip_abs,
        conditional_residual_models=conditional_residual_models,
        conditional_residual_clip_map=conditional_residual_clip_map,
        conditional_residual_gate_meta=conditional_residual_gate_meta,
        stats=stats,
    )


def run_baseline_branch_forecast(
    bundle: BaselineBranchBundle,
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    default_value: float,
    include_calendar: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    pred = baseline_mod.run_recursive_forecast(
        history=history,
        schedule=schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        primary_bundle=bundle.primary_bundle,
        default_value=default_value,
        include_calendar=include_calendar,
        weather_table=weather_table,
        weather_defaults_map=weather_defaults_map,
        weather_columns=weather_columns,
        residual_models=bundle.residual_models,
        residual_clip_abs=bundle.residual_clip_abs,
        horizon_bias_map=bundle.horizon_bias_map,
        horizon_bias_clip_abs=bundle.horizon_bias_clip_abs,
        conditional_residual_models=bundle.conditional_residual_models,
        conditional_residual_clip_map=bundle.conditional_residual_clip_map,
        conditional_residual_gate_meta=bundle.conditional_residual_gate_meta,
        actual_map=actual_map,
    )
    return pred.rename(columns={"prediction": "baseline_prediction"})


def fuse_predictions(
    baseline_pred: pd.DataFrame,
    gbdt_pred: pd.DataFrame,
    fusion_weights: FusionBundle,
) -> pd.DataFrame:
    keys = ["tollgate_id", "direction", "time_window", "horizon"]
    cols = keys + ["gbdt_prediction"] + (["actual"] if "actual" in gbdt_pred.columns else [])
    merged = baseline_pred.merge(gbdt_pred[cols], on=keys, how="inner", suffixes=("", "_g"))

    if "actual" not in merged.columns and "actual_g" in merged.columns:
        merged = merged.rename(columns={"actual_g": "actual"})

    linear_weight: list[float] = []
    gbdt_weight: list[float] = []
    fused: list[float] = []
    for row in merged.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        ts = pd.Timestamp(row.time_window)
        w_linear, w_gbdt = fusion_weights.resolve(key, h, ts)
        pred = max(0.0, w_linear * float(row.baseline_prediction) + w_gbdt * float(row.gbdt_prediction))
        linear_weight.append(w_linear)
        gbdt_weight.append(w_gbdt)
        fused.append(pred)

    merged["linear_weight"] = linear_weight
    merged["gbdt_weight"] = gbdt_weight
    merged["prediction"] = fused
    merged["linear_prediction"] = merged["baseline_prediction"]
    return merged


POST_FUSION_FEATURE_NAMES = [
    "fused_prediction",
    "linear_prediction",
    "gbdt_prediction",
    "pred_gap",
    "abs_pred_gap",
    "gap_ratio",
    "horizon",
    "anchor",
]


def fuse_pair_frame(pair_df: pd.DataFrame, fusion_weights: FusionBundle) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()

    keys = ["tollgate_id", "direction", "time_window", "horizon"]
    base_cols = keys + ["linear_prediction"] + (["actual"] if "actual" in pair_df.columns else [])
    gbdt_cols = keys + ["gbdt_prediction"] + (["actual"] if "actual" in pair_df.columns else [])
    baseline_pred = pair_df[base_cols].rename(columns={"linear_prediction": "baseline_prediction"})
    gbdt_pred = pair_df[gbdt_cols].copy()
    return fuse_predictions(baseline_pred=baseline_pred, gbdt_pred=gbdt_pred, fusion_weights=fusion_weights)


def build_post_fusion_feature_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    linear = pred_df["linear_prediction"].astype(float)
    gbdt = pred_df["gbdt_prediction"].astype(float)
    fused = pred_df["prediction"].astype(float)
    gap = gbdt - linear
    denom = np.maximum(fused.to_numpy(dtype=float), 1.0)
    anchor = pd.to_datetime(pred_df["time_window"]).apply(anchor_bucket).astype(float)

    frame = pd.DataFrame(
        {
            "fused_prediction": fused.to_numpy(dtype=float),
            "linear_prediction": linear.to_numpy(dtype=float),
            "gbdt_prediction": gbdt.to_numpy(dtype=float),
            "pred_gap": gap.to_numpy(dtype=float),
            "abs_pred_gap": np.abs(gap.to_numpy(dtype=float)),
            "gap_ratio": gap.to_numpy(dtype=float) / denom,
            "horizon": pred_df["horizon"].to_numpy(dtype=float),
            "anchor": anchor.to_numpy(dtype=float),
        },
        index=pred_df.index,
    )
    return frame


def _disabled_post_fusion_bundle(reason: str) -> PostFusionResidualBundle:
    return PostFusionResidualBundle(
        enabled=False,
        feature_names=POST_FUSION_FEATURE_NAMES.copy(),
        models={},
        clip_map={},
        horizon_allowlist={},
        gate_meta={},
        stats={"enabled": 0, "reason": reason},
    )


def _parse_horizon_allowlist(raw: object | None) -> set[int]:
    if raw is None:
        return {1, 2, 3, 4, 5, 6}
    if not isinstance(raw, list):
        raise ValueError("post_fusion_residual.apply_horizons must be a list of horizons")
    out: set[int] = set()
    for item in raw:
        h = int(item)
        if h < 1 or h > 6:
            raise ValueError(f"Invalid horizon in post_fusion_residual.apply_horizons: {h}")
        out.add(h)
    return out if out else {1, 2, 3, 4, 5, 6}


def train_post_fusion_residual_bundle(
    cfg: dict,
    oof_pair_frame: pd.DataFrame | None,
    fusion_weights: FusionBundle,
) -> PostFusionResidualBundle:
    post_cfg = cfg.get("post_fusion_residual", {})
    use_post = bool(post_cfg.get("use", False))
    if not use_post:
        return _disabled_post_fusion_bundle("disabled_by_config")
    if oof_pair_frame is None or oof_pair_frame.empty:
        return _disabled_post_fusion_bundle("missing_oof_frame")

    target_series = post_cfg.get("target_series", [])
    if not target_series:
        return _disabled_post_fusion_bundle("empty_target_series")

    default_alpha = float(post_cfg.get("ridge_alpha", 120.0))
    default_min_samples = int(post_cfg.get("min_samples", 30))
    default_clip_abs = float(post_cfg.get("clip_abs", 6.0))
    default_use_gate = bool(post_cfg.get("use_confidence_gate", True))
    default_gate_quantile = float(post_cfg.get("gate_quantile", 0.7))
    default_gate_max_z_raw = post_cfg.get("gate_max_z")
    default_gate_max_z = float(default_gate_max_z_raw) if default_gate_max_z_raw is not None else None
    default_use_gain_gate = bool(post_cfg.get("use_gain_gate", True))
    default_gain_min_mean = float(post_cfg.get("gain_min_mean", 0.0))
    default_gain_min_count = int(post_cfg.get("gain_min_count", 8))
    default_gain_quantiles = post_cfg.get(
        "gain_quantiles",
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    if not isinstance(default_gain_quantiles, list) or not default_gain_quantiles:
        raise ValueError("post_fusion_residual.gain_quantiles must be a non-empty list")
    default_gain_quantiles = [float(q) for q in default_gain_quantiles]

    apply_horizon_cfg = post_cfg.get("apply_horizons", {})
    if not isinstance(apply_horizon_cfg, dict):
        raise ValueError("post_fusion_residual.apply_horizons must be dict keyed by series")
    series_params_cfg = post_cfg.get("series_params", {})
    if not isinstance(series_params_cfg, dict):
        raise ValueError("post_fusion_residual.series_params must be dict keyed by series")

    fused_oof = fuse_pair_frame(oof_pair_frame, fusion_weights)
    if fused_oof.empty:
        return _disabled_post_fusion_bundle("empty_fused_oof")

    feat_df = build_post_fusion_feature_frame(fused_oof)
    residual = fused_oof["actual"].to_numpy(dtype=float) - fused_oof["prediction"].to_numpy(dtype=float)

    models: dict[tuple[int, int], baseline_mod.RidgeLinearModel] = {}
    clip_map: dict[tuple[int, int], float] = {}
    horizon_allowlist: dict[tuple[int, int], set[int]] = {}
    gate_meta: dict[tuple[int, int], dict[str, object]] = {}
    stats: dict[str, dict[str, float | int | object]] = {}

    for series_text in target_series:
        key = baseline_mod.parse_series_key(series_text)
        params = series_params_cfg.get(series_text, {})
        if not isinstance(params, dict):
            raise ValueError(f"post_fusion_residual.series_params['{series_text}'] must be object")

        alpha = float(params.get("ridge_alpha", default_alpha))
        min_samples = int(params.get("min_samples", default_min_samples))
        clip_abs = float(params.get("clip_abs", default_clip_abs))
        use_gate = bool(params.get("use_confidence_gate", default_use_gate))
        gate_quantile = float(params.get("gate_quantile", default_gate_quantile))
        gate_max_z_raw = params.get("gate_max_z", default_gate_max_z)
        gate_max_z = float(gate_max_z_raw) if gate_max_z_raw is not None else None
        use_gain_gate = bool(params.get("use_gain_gate", default_use_gain_gate))
        gain_min_mean = float(params.get("gain_min_mean", default_gain_min_mean))
        gain_min_count = int(params.get("gain_min_count", default_gain_min_count))
        gain_quantiles_raw = params.get("gain_quantiles", default_gain_quantiles)
        if not isinstance(gain_quantiles_raw, list) or not gain_quantiles_raw:
            raise ValueError(f"post_fusion_residual.gain_quantiles for {series_text} must be non-empty list")
        gain_quantiles = [float(q) for q in gain_quantiles_raw]

        allow_raw = params.get("apply_horizons", apply_horizon_cfg.get(series_text))
        allow_horizons = _parse_horizon_allowlist(allow_raw)

        mask = (
            (fused_oof["tollgate_id"] == key[0])
            & (fused_oof["direction"] == key[1])
            & (fused_oof["horizon"].isin(list(allow_horizons)))
        )
        sample_count = int(mask.sum())
        if sample_count < min_samples:
            stats[series_text] = {
                "samples": sample_count,
                "trained": 0,
                "min_samples": min_samples,
                "apply_horizons": sorted(allow_horizons),
                "reason": f"insufficient_samples_lt_{min_samples}",
            }
            continue

        x_sub = feat_df.loc[mask].reset_index(drop=True)
        y_sub = pd.Series(residual[mask.to_numpy()], name="post_fusion_residual")

        model = baseline_mod.RidgeLinearModel(feature_names=POST_FUSION_FEATURE_NAMES, alpha=alpha)
        model.fit(x_sub, y_sub)
        models[key] = model
        clip_map[key] = clip_abs
        horizon_allowlist[key] = allow_horizons

        pred_sub = model.predict(x_sub).astype(float)
        pred_sub = np.clip(pred_sub, -clip_abs, clip_abs)
        gain_arr = np.abs(y_sub.to_numpy(dtype=float)) - np.abs(y_sub.to_numpy(dtype=float) - pred_sub)
        pred_abs = np.abs(pred_sub)

        x_np = x_sub[POST_FUSION_FEATURE_NAMES].to_numpy(dtype=float)
        center = np.mean(x_np, axis=0)
        scale = np.std(x_np, axis=0, ddof=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        dists = np.sqrt(np.mean(((x_np - center) / scale) ** 2, axis=1))
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

        gate_meta[key] = {
            "center": center,
            "scale": scale,
            "threshold": float(threshold),
            "gain_abs_threshold": float(gain_threshold),
        }

        stats[series_text] = {
            "samples": sample_count,
            "trained": 1,
            "alpha": alpha,
            "clip_abs": clip_abs,
            "apply_horizons": sorted(allow_horizons),
            "gate_enabled": int(use_gate),
            "gate_quantile": gate_quantile,
            "gate_max_z": gate_max_z if gate_max_z is not None else -1.0,
            "gate_threshold": float(threshold),
            "gain_gate_enabled": int(use_gain_gate),
            "gain_best_quantile": gain_best_quantile,
            "gain_best_mean": gain_best_mean,
            "gain_abs_threshold": float(gain_threshold),
            "residual_mean": float(y_sub.mean()),
            "residual_std": float(y_sub.std(ddof=0)),
        }

    return PostFusionResidualBundle(
        enabled=True,
        feature_names=POST_FUSION_FEATURE_NAMES.copy(),
        models=models,
        clip_map=clip_map,
        horizon_allowlist=horizon_allowlist,
        gate_meta=gate_meta,
        stats={
            "enabled": 1,
            "oof_rows": int(len(fused_oof)),
            "trained_series_count": int(len(models)),
            "series_stats": stats,
        },
    )


def apply_post_fusion_residual_adjustment(
    pred_df: pd.DataFrame,
    post_bundle: PostFusionResidualBundle,
) -> pd.DataFrame:
    out = pred_df.copy()
    if out.empty:
        return out

    correction: list[float] = []
    gate_applied: list[int] = []
    conf_gate_applied: list[int] = []
    gain_gate_applied: list[int] = []
    gate_distance: list[float] = []
    gate_threshold: list[float] = []
    gain_threshold: list[float] = []
    prediction_before_post = out["prediction"].astype(float).tolist()

    for row in out.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        corr = 0.0
        applied = 0
        conf_ok = 0
        gain_ok = 0
        dist_val = float("nan")
        dist_th = float("nan")
        gain_th = float("nan")

        if post_bundle.enabled and key in post_bundle.models and h in post_bundle.horizon_allowlist.get(key, set()):
            applied = 1
            feat_map = {
                "fused_prediction": float(row.prediction),
                "linear_prediction": float(row.linear_prediction),
                "gbdt_prediction": float(row.gbdt_prediction),
                "pred_gap": float(row.gbdt_prediction) - float(row.linear_prediction),
                "abs_pred_gap": abs(float(row.gbdt_prediction) - float(row.linear_prediction)),
                "gap_ratio": (float(row.gbdt_prediction) - float(row.linear_prediction))
                / max(float(row.prediction), 1.0),
                "horizon": float(h),
                "anchor": float(anchor_bucket(pd.Timestamp(row.time_window))),
            }
            feat_df = pd.DataFrame([feat_map], columns=post_bundle.feature_names)
            can_apply = True
            gate = post_bundle.gate_meta.get(key)
            if gate is not None:
                center = gate["center"]  # type: ignore[assignment]
                scale = gate["scale"]  # type: ignore[assignment]
                dist_th = float(gate["threshold"])
                gain_th = float(gate.get("gain_abs_threshold", 0.0))
                if np.isfinite(dist_th):
                    x_np = feat_df.iloc[0].to_numpy(dtype=float)
                    dist_val = float(np.sqrt(np.mean(((x_np - center) / scale) ** 2)))
                    can_apply = dist_val <= dist_th
                if can_apply:
                    conf_ok = 1

            if can_apply:
                pred_corr = float(post_bundle.models[key].predict(feat_df)[0])
                clip_abs = post_bundle.clip_map.get(key)
                if clip_abs is not None:
                    pred_corr = float(np.clip(pred_corr, -clip_abs, clip_abs))

                if np.isfinite(gain_th):
                    can_apply = abs(pred_corr) >= gain_th
                if can_apply:
                    gain_ok = 1
                    corr = pred_corr

        correction.append(corr)
        gate_applied.append(applied)
        conf_gate_applied.append(conf_ok)
        gain_gate_applied.append(gain_ok)
        gate_distance.append(dist_val)
        gate_threshold.append(dist_th)
        gain_threshold.append(gain_th)

    out["prediction_before_post_fusion"] = prediction_before_post
    out["post_fusion_residual_correction"] = correction
    out["post_fusion_gate_applied"] = gate_applied
    out["post_fusion_conf_gate_applied"] = conf_gate_applied
    out["post_fusion_gain_gate_applied"] = gain_gate_applied
    out["post_fusion_gate_distance"] = gate_distance
    out["post_fusion_gate_threshold"] = gate_threshold
    out["post_fusion_gain_threshold"] = gain_threshold
    out["prediction"] = np.clip(out["prediction_before_post_fusion"] + out["post_fusion_residual_correction"], 0.0, None)
    return out


def train_with_adaptive_fusion(
    cfg: dict,
    x_all: pd.DataFrame,
    y_all: pd.Series,
    meta_all: pd.DataFrame,
    x_gbdt_all: pd.DataFrame,
    y_gbdt_all: pd.Series,
    meta_gbdt_all: pd.DataFrame,
    train_history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    feature_names: list[str],
    default_value: float,
    include_calendar: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    train_days: list[pd.Timestamp],
    horizon_windows: int,
) -> tuple[
    BaselineBranchBundle,
    GBDTBundle,
    FusionBundle,
    dict[str, object],
    pd.DataFrame | None,
]:
    # return: baseline branch, gbdt branch, fusion bundle, adaptation stats
    fusion_cfg = cfg.get("fusion", {})
    gbdt_cfg = cfg.get("gbdt_model", {})
    gbdt_training_cfg = cfg.get("gbdt_training", {})
    weight_learning = str(fusion_cfg.get("weight_learning", "tail_window"))
    adapt_days = int(fusion_cfg.get("adapt_days", 4))
    min_core_days = int(fusion_cfg.get("min_core_days", 8))

    if len(train_days) == 0:
        raise RuntimeError("No train days provided")

    use_adapt = False
    adaptation_stats: dict[str, object] = {
        "use_adaptation": 0,
        "weight_learning": weight_learning,
        "adapt_days": 0,
        "core_days": int(len(train_days)),
        "reason": "adaptation_disabled",
    }
    oof_pair_frame: pd.DataFrame | None = None

    def collect_pair_predictions(
        fit_days: list[pd.Timestamp],
        eval_days: list[pd.Timestamp],
    ) -> pd.DataFrame:
        x_fit, y_fit, meta_fit = select_by_days(x_all, y_all, meta_all, fit_days)
        if x_fit.empty:
            return pd.DataFrame()

        fit_baseline = train_baseline_branch(
            cfg=cfg,
            x_train=x_fit,
            y_train=y_fit,
            meta_train=meta_fit,
            feature_names=feature_names,
        )
        x_gfit, y_gfit, meta_gfit = select_by_days(x_gbdt_all, y_gbdt_all, meta_gbdt_all, fit_days)
        if x_gfit.empty:
            return pd.DataFrame()
        fit_gbdt = train_gbdt_bundle(
            x_train=x_gfit,
            y_train=y_gfit,
            meta_train=meta_gfit,
            feature_names=feature_names,
            model_cfg=gbdt_cfg,
            training_cfg=gbdt_training_cfg,
        )

        eval_schedule = target_windows_for_days(eval_days, horizon=horizon_windows)
        base_history, actual_map = prepare_history_for_schedule(train_history, series_keys, eval_schedule)
        gbdt_history, _ = prepare_history_for_schedule(train_history, series_keys, eval_schedule)

        eval_baseline = run_baseline_branch_forecast(
            bundle=fit_baseline,
            history=base_history,
            schedule=eval_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=include_calendar,
            weather_table=weather_table,
            weather_defaults_map=weather_defaults_map,
            weather_columns=weather_columns,
            actual_map=actual_map,
        )
        eval_gbdt = run_gbdt_recursive_forecast(
            bundle=fit_gbdt,
            history=gbdt_history,
            schedule=eval_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=include_calendar,
            weather_table=weather_table,
            weather_defaults_map=weather_defaults_map,
            weather_columns=weather_columns,
            actual_map=actual_map,
        )
        merged = fuse_predictions(
            baseline_pred=eval_baseline,
            gbdt_pred=eval_gbdt,
            fusion_weights=default_fusion_weights(cfg),
        )
        merged = merged.dropna(subset=["actual"]).reset_index(drop=True)
        return merged

    if weight_learning == "rolling_oof":
        oof_n_folds = int(fusion_cfg.get("oof_n_folds", 3))
        oof_val_days = int(fusion_cfg.get("oof_val_days", 2))
        oof_min_train_days = int(fusion_cfg.get("oof_min_train_days", max(min_core_days, 8)))
        folds = rolling_folds(
            days=train_days,
            n_folds=oof_n_folds,
            val_days=oof_val_days,
            min_train_days=oof_min_train_days,
        )
        oof_parts: list[pd.DataFrame] = []
        for fit_days, eval_days in folds:
            pair = collect_pair_predictions(fit_days, eval_days)
            if pair.empty:
                continue
            oof_parts.append(pair)

        if oof_parts:
            oof_merged = pd.concat(oof_parts, axis=0, ignore_index=True)
            oof_pair_frame = oof_merged[
                [
                    "tollgate_id",
                    "direction",
                    "horizon",
                    "time_window",
                    "actual",
                    "linear_prediction",
                    "gbdt_prediction",
                ]
            ].copy()
            fit_input = oof_pair_frame.copy()
            fit_input["anchor"] = pd.to_datetime(fit_input["time_window"]).apply(anchor_bucket).astype(int)
            fit_input = fit_input.drop(columns=["time_window"])
            fusion_weights, fit_stats = fit_anchor_fusion_weights(fit_input, fusion_cfg)
            use_adapt = True
            adaptation_stats = {
                "use_adaptation": 1,
                "weight_learning": weight_learning,
                "folds": int(len(folds)),
                "oof_rows": int(len(oof_merged)),
                "fit": fit_stats,
            }
    else:
        if adapt_days > 0 and len(train_days) > (min_core_days + 1):
            true_adapt_days = min(adapt_days, len(train_days) - min_core_days)
            if true_adapt_days > 0:
                core_days = train_days[:-true_adapt_days]
                adapt_day_list = train_days[-true_adapt_days:]
                adapt_merged = collect_pair_predictions(core_days, adapt_day_list)
                if not adapt_merged.empty:
                    fit_input = adapt_merged[
                        [
                            "tollgate_id",
                            "direction",
                            "horizon",
                            "time_window",
                            "actual",
                            "linear_prediction",
                            "gbdt_prediction",
                        ]
                    ].copy()
                    fit_input["anchor"] = pd.to_datetime(fit_input["time_window"]).apply(anchor_bucket).astype(int)
                    fit_input = fit_input.drop(columns=["time_window"])
                    fusion_weights, fit_stats = fit_anchor_fusion_weights(fit_input, fusion_cfg)
                    use_adapt = True
                    adaptation_stats = {
                        "use_adaptation": 1,
                        "weight_learning": weight_learning,
                        "adapt_days": int(true_adapt_days),
                        "core_days": int(len(core_days)),
                        "adapt_rows": int(len(adapt_merged)),
                        "fit": fit_stats,
                    }

    if not use_adapt:
        fusion_weights = default_fusion_weights(cfg)

    x_train, y_train, meta_train = select_by_days(x_all, y_all, meta_all, train_days)
    if x_train.empty:
        raise RuntimeError("Training subset is empty")

    final_baseline = train_baseline_branch(
        cfg=cfg,
        x_train=x_train,
        y_train=y_train,
        meta_train=meta_train,
        feature_names=feature_names,
    )
    x_gtrain, y_gtrain, meta_gtrain = select_by_days(x_gbdt_all, y_gbdt_all, meta_gbdt_all, train_days)
    if x_gtrain.empty:
        raise RuntimeError("GBDT training subset is empty")
    final_gbdt = train_gbdt_bundle(
        x_train=x_gtrain,
        y_train=y_gtrain,
        meta_train=meta_gtrain,
        feature_names=feature_names,
        model_cfg=gbdt_cfg,
        training_cfg=gbdt_training_cfg,
    )

    adaptation_stats["final_train_rows"] = int(len(x_train))
    adaptation_stats["final_gbdt_train_rows"] = int(len(x_gtrain))
    return final_baseline, final_gbdt, fusion_weights, adaptation_stats, oof_pair_frame


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
    weather_columns = resolve_weather_columns(cfg, use_weather)

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

    gbdt_training_cfg = cfg.get("gbdt_training", {})
    gbdt_target_only = bool(gbdt_training_cfg.get("target_only", False))

    x_all, y_all, meta_all = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        include_calendar=use_calendar,
        target_only=True,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )
    if x_all.empty:
        raise RuntimeError("No training samples for pre-holdout period")
    x_gbdt_all, y_gbdt_all, meta_gbdt_all = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        include_calendar=use_calendar,
        target_only=gbdt_target_only,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )
    if x_gbdt_all.empty:
        raise RuntimeError("No GBDT training samples for pre-holdout period")

    feature_names = feature_columns(series_keys, feature_cfg, include_calendar=use_calendar) + weather_columns
    horizon_windows = int(cfg["target"]["horizon_windows"])

    pre_days = sorted(pd.to_datetime(meta_all["day"].drop_duplicates().tolist()))

    rolling_cfg = cfg.get("rolling_validation", {})
    use_rolling = bool(rolling_cfg.get("use", False))
    rolling_results: list[dict[str, float | int]] = []

    if use_rolling:
        folds = rolling_folds(
            days=pre_days,
            n_folds=int(rolling_cfg.get("n_folds", 3)),
            val_days=int(rolling_cfg.get("val_days", 2)),
            min_train_days=int(rolling_cfg.get("min_train_days", 10)),
        )

        for idx, (fold_train_days, fold_val_days) in enumerate(folds, start=1):
            base_bundle, gbdt_bundle, fusion_weights, fold_adapt, fold_oof_pair = train_with_adaptive_fusion(
                cfg=cfg,
                x_all=x_all,
                y_all=y_all,
                meta_all=meta_all,
                x_gbdt_all=x_gbdt_all,
                y_gbdt_all=y_gbdt_all,
                meta_gbdt_all=meta_gbdt_all,
                train_history=train_history,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                feature_names=feature_names,
                default_value=default_value,
                include_calendar=use_calendar,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
                train_days=fold_train_days,
                horizon_windows=horizon_windows,
            )
            fold_post_bundle = train_post_fusion_residual_bundle(
                cfg=cfg,
                oof_pair_frame=fold_oof_pair,
                fusion_weights=fusion_weights,
            )

            fold_schedule = target_windows_for_days(fold_val_days, horizon=horizon_windows)
            fold_base_history, fold_actual = prepare_history_for_schedule(train_history, series_keys, fold_schedule)
            fold_gbdt_history, _ = prepare_history_for_schedule(train_history, series_keys, fold_schedule)

            fold_base_pred = run_baseline_branch_forecast(
                bundle=base_bundle,
                history=fold_base_history,
                schedule=fold_schedule,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                default_value=default_value,
                include_calendar=use_calendar,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
                actual_map=fold_actual,
            )
            fold_gbdt_pred = run_gbdt_recursive_forecast(
                bundle=gbdt_bundle,
                history=fold_gbdt_history,
                schedule=fold_schedule,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                default_value=default_value,
                include_calendar=use_calendar,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
                actual_map=fold_actual,
            )
            fold_fused = fuse_predictions(fold_base_pred, fold_gbdt_pred, fusion_weights)
            fold_fused = apply_post_fusion_residual_adjustment(fold_fused, fold_post_bundle)
            fold_fused = fold_fused.dropna(subset=["actual"]).reset_index(drop=True)
            if fold_fused.empty:
                continue
            fold_metrics = summarize_metrics(fold_fused)
            rolling_results.append(
                {
                    "fold": idx,
                    "train_days": int(len(fold_train_days)),
                    "val_days": int(len(fold_val_days)),
                    "overall_mape": float(fold_metrics["overall_mape"]),
                    "global_gbdt_weight": float(fusion_weights.global_weights.global_gbdt_weight),
                    "anchor_weight_count": int(len(fusion_weights.anchor_weights)),
                    "use_adaptation": int(fold_adapt.get("use_adaptation", 0)),
                    "post_fusion_enabled": int(fold_post_bundle.enabled),
                    "post_fusion_trained_series": int(len(fold_post_bundle.models)),
                }
            )

    final_base, final_gbdt, final_fusion, final_adapt_stats, final_oof_pair = train_with_adaptive_fusion(
        cfg=cfg,
        x_all=x_all,
        y_all=y_all,
        meta_all=meta_all,
        x_gbdt_all=x_gbdt_all,
        y_gbdt_all=y_gbdt_all,
        meta_gbdt_all=meta_gbdt_all,
        train_history=train_history,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        feature_names=feature_names,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        train_days=pre_days,
        horizon_windows=horizon_windows,
    )
    final_post_bundle = train_post_fusion_residual_bundle(
        cfg=cfg,
        oof_pair_frame=final_oof_pair,
        fusion_weights=final_fusion,
    )

    holdout_days = sorted(
        pd.to_datetime(train_grid.loc[train_grid["time_window"] >= split_ts, "time_window"].dt.normalize().unique())
    )
    holdout_schedule = target_windows_for_days(holdout_days, horizon=horizon_windows)

    hold_base_history, hold_actual = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)
    hold_gbdt_history, _ = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)

    hold_base_pred = run_baseline_branch_forecast(
        bundle=final_base,
        history=hold_base_history,
        schedule=holdout_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=hold_actual,
    )
    hold_gbdt_pred = run_gbdt_recursive_forecast(
        bundle=final_gbdt,
        history=hold_gbdt_history,
        schedule=holdout_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=hold_actual,
    )
    holdout_pred = fuse_predictions(hold_base_pred, hold_gbdt_pred, final_fusion)
    holdout_pred = apply_post_fusion_residual_adjustment(holdout_pred, final_post_bundle)
    holdout_pred = holdout_pred.dropna(subset=["actual"]).reset_index(drop=True)

    metrics = summarize_metrics(holdout_pred)
    slice_df = build_error_slice_table(holdout_pred)
    slice_path = run_dir / "validation_error_slices.csv"
    slice_df.to_csv(slice_path, index=False)

    test_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["test_volume_csv"])
    test_agg = aggregate_to_20min(test_events)
    test_history = build_series_history(test_agg)
    merged_history = merge_histories(train_history, test_history)

    test_days = sorted(pd.to_datetime(test_agg["time_window"].dt.normalize().unique()))
    test_schedule = target_windows_for_days(test_days, horizon=horizon_windows)

    test_base_history = {k: s.copy() for k, s in merged_history.items()}
    test_gbdt_history = {k: s.copy() for k, s in merged_history.items()}

    test_base_pred = run_baseline_branch_forecast(
        bundle=final_base,
        history=test_base_history,
        schedule=test_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=inference_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=None,
    )
    test_gbdt_pred = run_gbdt_recursive_forecast(
        bundle=final_gbdt,
        history=test_gbdt_history,
        schedule=test_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=inference_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=None,
    )
    test_pred = fuse_predictions(test_base_pred, test_gbdt_pred, final_fusion)
    test_pred = apply_post_fusion_residual_adjustment(test_pred, final_post_bundle)

    submission = build_submission(test_pred)
    validate_submission_schema(submission)

    rolling_avg = float(np.mean([x["overall_mape"] for x in rolling_results])) if rolling_results else None

    run_meta = {
        "run_id": run_id,
        "split_timestamp": split_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": int(len(x_all)),
        "gbdt_train_samples": int(len(x_gbdt_all)),
        "validation_rows": int(len(holdout_pred)),
        "validation_slice_rows": int(len(slice_df)),
        "submission_rows": int(len(submission)),
        "use_weather": use_weather,
        "use_calendar": use_calendar,
        "weather_columns": weather_columns,
        "baseline_branch": final_base.stats,
        "gbdt_model": cfg.get("gbdt_model", {}),
        "gbdt_training": gbdt_training_cfg,
        "fusion": {
            "global_gbdt_weight": float(final_fusion.global_weights.global_gbdt_weight),
            "series_weight_count": int(len(final_fusion.global_weights.series_gbdt_weight)),
            "slice_weight_count": int(len(final_fusion.global_weights.slice_gbdt_weight)),
            "anchor_weight_count": int(len(final_fusion.anchor_weights)),
            "adaptation": final_adapt_stats,
        },
        "post_fusion_residual": final_post_bundle.stats,
        "rolling_validation": {
            "enabled": use_rolling,
            "folds": rolling_results,
            "avg_mape": rolling_avg,
        },
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
    holdout_pred.to_csv(run_dir / "validation_predictions.csv", index=False)

    sub_path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{run_id}.csv"
    submission.to_csv(sub_path, index=False)

    print(json.dumps(run_meta, ensure_ascii=False, indent=2))
    print(f"submission_path={sub_path}")


if __name__ == "__main__":
    main()
