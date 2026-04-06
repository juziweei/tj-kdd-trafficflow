#!/usr/bin/env python3
"""Run stronger dual-trunk backbone with time-safe adaptive fusion."""

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
from src.models.ridge_linear import RidgeLinearModel  # noqa: E402


@dataclass
class LinearBundle:
    feature_names: list[str]
    use_log_target: bool
    log_pred_clip: float
    model: RidgeLinearModel


@dataclass
class GBDTBundle:
    feature_names: list[str]
    use_log_target: bool
    log_pred_clip: float
    model: XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strong dual-trunk backbone")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "strong_backbone_v1_fusion.json",
        help="Path to config JSON",
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
                "day": ts.normalize(),
            }
        )

    x_df = pd.DataFrame(rows)
    y = pd.Series(targets, name="volume")
    meta_df = pd.DataFrame(meta_rows)
    return x_df, y, meta_df


def make_xgb(model_cfg: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 350)),
        max_depth=int(model_cfg.get("max_depth", 5)),
        learning_rate=float(model_cfg.get("learning_rate", 0.03)),
        subsample=float(model_cfg.get("subsample", 0.9)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.9)),
        reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(model_cfg.get("reg_lambda", 2.0)),
        min_child_weight=float(model_cfg.get("min_child_weight", 1.5)),
        gamma=float(model_cfg.get("gamma", 0.0)),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=int(model_cfg.get("seed", 42)),
        n_jobs=int(model_cfg.get("n_jobs", 6)),
        verbosity=0,
    )


def train_linear_bundle(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    model_cfg: dict,
) -> LinearBundle:
    use_log_target = bool(model_cfg.get("use_log_target", False))
    log_pred_clip = float(model_cfg.get("log_pred_clip", 8.0))
    alpha = float(model_cfg.get("ridge_alpha", 5.0))

    target = np.log1p(y_train) if use_log_target else y_train
    model = RidgeLinearModel(feature_names=feature_names, alpha=alpha)
    model.fit(x_train, target)

    return LinearBundle(
        feature_names=feature_names,
        use_log_target=use_log_target,
        log_pred_clip=log_pred_clip,
        model=model,
    )


def train_gbdt_bundle(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    model_cfg: dict,
) -> GBDTBundle:
    use_log_target = bool(model_cfg.get("use_log_target", True))
    log_pred_clip = float(model_cfg.get("log_pred_clip", 6.0))

    target = np.log1p(y_train.to_numpy(dtype=float)) if use_log_target else y_train.to_numpy(dtype=float)
    sample_weight = 1.0 / np.maximum(y_train.to_numpy(dtype=float), 1.0)

    model = make_xgb(model_cfg)
    model.fit(x_train[feature_names], target, sample_weight=sample_weight)

    return GBDTBundle(
        feature_names=feature_names,
        use_log_target=use_log_target,
        log_pred_clip=log_pred_clip,
        model=model,
    )


def predict_linear(bundle: LinearBundle, x_row: pd.DataFrame) -> float:
    raw = float(bundle.model.predict(x_row[bundle.feature_names])[0])
    if bundle.use_log_target:
        raw = min(raw, bundle.log_pred_clip)
        return max(0.0, float(np.expm1(raw)))
    return max(0.0, raw)


def predict_gbdt(bundle: GBDTBundle, x_row: pd.DataFrame) -> float:
    raw = float(bundle.model.predict(x_row[bundle.feature_names])[0])
    if bundle.use_log_target:
        raw = min(raw, bundle.log_pred_clip)
        return max(0.0, float(np.expm1(raw)))
    return max(0.0, raw)


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


def run_dual_recursive_forecast(
    linear_bundle: LinearBundle,
    gbdt_bundle: GBDTBundle,
    fusion_weights: AdaptiveFusionWeights,
    linear_history: dict[tuple[int, int], pd.Series],
    gbdt_history: dict[tuple[int, int], pd.Series],
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
            linear_feat = build_feature_row(
                key=key,
                ts=ts,
                history=linear_history,
                series_keys=series_keys,
                cfg=feature_cfg,
                default_value=default_value,
                allow_fallback=True,
            )
            gbdt_feat = build_feature_row(
                key=key,
                ts=ts,
                history=gbdt_history,
                series_keys=series_keys,
                cfg=feature_cfg,
                default_value=default_value,
                allow_fallback=True,
            )
            if linear_feat is None or gbdt_feat is None:
                continue

            if include_calendar:
                cal = calendar_feature_vector(ts)
                linear_feat.update(cal)
                gbdt_feat.update(cal)

            if weather_table is not None and weather_defaults_map is not None:
                weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
                weather_values = {k: weather_values[k] for k in weather_columns}
                linear_feat.update(weather_values)
                gbdt_feat.update(weather_values)

            horizon = int(horizon_index(ts))
            x_linear = pd.DataFrame([linear_feat], columns=linear_bundle.feature_names)
            x_gbdt = pd.DataFrame([gbdt_feat], columns=gbdt_bundle.feature_names)

            linear_pred = predict_linear(linear_bundle, x_linear)
            gbdt_pred = predict_gbdt(gbdt_bundle, x_gbdt)
            w_linear, w_gbdt = fusion_weights.resolve(key, horizon)
            fused_pred = max(0.0, w_linear * linear_pred + w_gbdt * gbdt_pred)

            linear_history[key].loc[ts] = linear_pred
            linear_history[key] = linear_history[key].sort_index()
            gbdt_history[key].loc[ts] = gbdt_pred
            gbdt_history[key] = gbdt_history[key].sort_index()

            rec: dict[str, float | int | pd.Timestamp] = {
                "tollgate_id": int(key[0]),
                "direction": int(key[1]),
                "time_window": ts,
                "horizon": horizon,
                "linear_prediction": linear_pred,
                "gbdt_prediction": gbdt_pred,
                "linear_weight": w_linear,
                "gbdt_weight": w_gbdt,
                "prediction": fused_pred,
            }
            if actual_map is not None:
                rec["actual"] = float(actual_map[(key, ts)])
            records.append(rec)

    return pd.DataFrame(records)


def default_fusion_weights(cfg: dict) -> AdaptiveFusionWeights:
    fusion_cfg = cfg.get("fusion", {})
    default_w = float(fusion_cfg.get("default_gbdt_weight", 0.35))
    return AdaptiveFusionWeights(
        global_gbdt_weight=default_w,
        series_gbdt_weight={},
        slice_gbdt_weight={},
    )


def train_with_adaptive_fusion(
    cfg: dict,
    x_all: pd.DataFrame,
    y_all: pd.Series,
    meta_all: pd.DataFrame,
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
) -> tuple[LinearBundle, GBDTBundle, AdaptiveFusionWeights, dict[str, object]]:
    fusion_cfg = cfg.get("fusion", {})
    adapt_days = int(fusion_cfg.get("adapt_days", 4))
    min_core_days = int(fusion_cfg.get("min_core_days", 8))

    if len(train_days) == 0:
        raise RuntimeError("No train days provided")

    use_adapt = adapt_days > 0 and len(train_days) > (min_core_days + 1)
    adaptation_stats: dict[str, object] = {
        "use_adaptation": int(use_adapt),
        "adapt_days": 0,
        "core_days": int(len(train_days)),
        "reason": "adaptation_disabled",
    }

    linear_model_cfg = cfg.get("linear_model", {})
    gbdt_model_cfg = cfg.get("gbdt_model", {})

    if use_adapt:
        true_adapt_days = min(adapt_days, len(train_days) - min_core_days)
        if true_adapt_days <= 0:
            use_adapt = False
        else:
            core_days = train_days[:-true_adapt_days]
            adapt_day_list = train_days[-true_adapt_days:]

            x_core, y_core, _ = select_by_days(x_all, y_all, meta_all, core_days)
            if x_core.empty:
                use_adapt = False
            else:
                core_linear = train_linear_bundle(
                    x_train=x_core,
                    y_train=y_core,
                    feature_names=feature_names,
                    model_cfg=linear_model_cfg,
                )
                core_gbdt = train_gbdt_bundle(
                    x_train=x_core,
                    y_train=y_core,
                    feature_names=feature_names,
                    model_cfg=gbdt_model_cfg,
                )

                adapt_schedule = target_windows_for_days(adapt_day_list, horizon=horizon_windows)
                linear_history, actual_map = prepare_history_for_schedule(train_history, series_keys, adapt_schedule)
                gbdt_history, _ = prepare_history_for_schedule(train_history, series_keys, adapt_schedule)

                adapt_pred = run_dual_recursive_forecast(
                    linear_bundle=core_linear,
                    gbdt_bundle=core_gbdt,
                    fusion_weights=default_fusion_weights(cfg),
                    linear_history=linear_history,
                    gbdt_history=gbdt_history,
                    schedule=adapt_schedule,
                    series_keys=series_keys,
                    feature_cfg=feature_cfg,
                    default_value=default_value,
                    include_calendar=include_calendar,
                    weather_table=weather_table,
                    weather_defaults_map=weather_defaults_map,
                    weather_columns=weather_columns,
                    actual_map=actual_map,
                )
                adapt_pred = adapt_pred.dropna(subset=["actual"]).reset_index(drop=True)
                if adapt_pred.empty:
                    use_adapt = False
                else:
                    fusion_weights, fit_stats = fit_adaptive_fusion_weights(adapt_pred, fusion_cfg)
                    adaptation_stats = {
                        "use_adaptation": 1,
                        "adapt_days": int(true_adapt_days),
                        "core_days": int(len(core_days)),
                        "core_train_rows": int(len(x_core)),
                        "adapt_rows": int(len(adapt_pred)),
                        "fit": fit_stats,
                    }

    if not use_adapt:
        fusion_weights = default_fusion_weights(cfg)

    x_train, y_train, _ = select_by_days(x_all, y_all, meta_all, train_days)
    if x_train.empty:
        raise RuntimeError("Training subset is empty for final model fit")

    final_linear = train_linear_bundle(
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        model_cfg=linear_model_cfg,
    )
    final_gbdt = train_gbdt_bundle(
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        model_cfg=gbdt_model_cfg,
    )

    adaptation_stats["final_train_rows"] = int(len(x_train))
    return final_linear, final_gbdt, fusion_weights, adaptation_stats


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

    x_all, y_all, meta_all = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )
    if x_all.empty:
        raise RuntimeError("No training samples for pre-holdout period")

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
            linear_bundle, gbdt_bundle, fusion_weights, fold_adapt_stats = train_with_adaptive_fusion(
                cfg=cfg,
                x_all=x_all,
                y_all=y_all,
                meta_all=meta_all,
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

            fold_schedule = target_windows_for_days(fold_val_days, horizon=horizon_windows)
            fold_linear_history, fold_actual_map = prepare_history_for_schedule(train_history, series_keys, fold_schedule)
            fold_gbdt_history, _ = prepare_history_for_schedule(train_history, series_keys, fold_schedule)

            fold_pred = run_dual_recursive_forecast(
                linear_bundle=linear_bundle,
                gbdt_bundle=gbdt_bundle,
                fusion_weights=fusion_weights,
                linear_history=fold_linear_history,
                gbdt_history=fold_gbdt_history,
                schedule=fold_schedule,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                default_value=default_value,
                include_calendar=use_calendar,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
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
                    "global_gbdt_weight": float(fusion_weights.global_gbdt_weight),
                    "use_adaptation": int(fold_adapt_stats.get("use_adaptation", 0)),
                }
            )

    final_linear, final_gbdt, final_fusion, final_adapt_stats = train_with_adaptive_fusion(
        cfg=cfg,
        x_all=x_all,
        y_all=y_all,
        meta_all=meta_all,
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

    holdout_days = sorted(
        pd.to_datetime(train_grid.loc[train_grid["time_window"] >= split_ts, "time_window"].dt.normalize().unique())
    )
    holdout_schedule = target_windows_for_days(holdout_days, horizon=horizon_windows)

    holdout_linear_history, holdout_actual_map = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)
    holdout_gbdt_history, _ = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)

    holdout_pred = run_dual_recursive_forecast(
        linear_bundle=final_linear,
        gbdt_bundle=final_gbdt,
        fusion_weights=final_fusion,
        linear_history=holdout_linear_history,
        gbdt_history=holdout_gbdt_history,
        schedule=holdout_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=holdout_actual_map,
    )
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

    test_linear_history = {k: s.copy() for k, s in merged_history.items()}
    test_gbdt_history = {k: s.copy() for k, s in merged_history.items()}

    test_pred = run_dual_recursive_forecast(
        linear_bundle=final_linear,
        gbdt_bundle=final_gbdt,
        fusion_weights=final_fusion,
        linear_history=test_linear_history,
        gbdt_history=test_gbdt_history,
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

    submission = build_submission(test_pred)
    validate_submission_schema(submission)

    rolling_avg = float(np.mean([x["overall_mape"] for x in rolling_results])) if rolling_results else None

    linear_cfg = cfg.get("linear_model", {})
    gbdt_cfg = cfg.get("gbdt_model", {})

    run_meta = {
        "run_id": run_id,
        "split_timestamp": split_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": int(len(x_all)),
        "validation_rows": int(len(holdout_pred)),
        "validation_slice_rows": int(len(slice_df)),
        "submission_rows": int(len(submission)),
        "use_weather": use_weather,
        "use_calendar": use_calendar,
        "weather_columns": weather_columns,
        "linear_model": {
            "ridge_alpha": float(linear_cfg.get("ridge_alpha", 5.0)),
            "use_log_target": int(bool(linear_cfg.get("use_log_target", False))),
            "log_pred_clip": float(linear_cfg.get("log_pred_clip", 8.0)),
        },
        "gbdt_model": {
            "n_estimators": int(gbdt_cfg.get("n_estimators", 350)),
            "max_depth": int(gbdt_cfg.get("max_depth", 5)),
            "learning_rate": float(gbdt_cfg.get("learning_rate", 0.03)),
            "subsample": float(gbdt_cfg.get("subsample", 0.9)),
            "colsample_bytree": float(gbdt_cfg.get("colsample_bytree", 0.9)),
            "reg_alpha": float(gbdt_cfg.get("reg_alpha", 0.0)),
            "reg_lambda": float(gbdt_cfg.get("reg_lambda", 2.0)),
            "min_child_weight": float(gbdt_cfg.get("min_child_weight", 1.5)),
            "gamma": float(gbdt_cfg.get("gamma", 0.0)),
            "use_log_target": int(bool(gbdt_cfg.get("use_log_target", True))),
            "log_pred_clip": float(gbdt_cfg.get("log_pred_clip", 6.0)),
        },
        "fusion": {
            "global_gbdt_weight": float(final_fusion.global_gbdt_weight),
            "series_weight_count": int(len(final_fusion.series_gbdt_weight)),
            "slice_weight_count": int(len(final_fusion.slice_gbdt_weight)),
            "adaptation": final_adapt_stats,
        },
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
