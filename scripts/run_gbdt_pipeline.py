#!/usr/bin/env python3
"""Competition-style GBDT pipeline with rolling time validation."""

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
from src.eval.metrics import summarize_metrics
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


@dataclass
class GBDTBundle:
    feature_names: list[str]
    use_log_target: bool
    log_pred_clip: float
    global_model: XGBRegressor
    group_models: dict[tuple[int, int, int], XGBRegressor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GBDT pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gbdt_v1.json",
        help="Path to GBDT config JSON",
    )
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_timestamp(all_windows: pd.Series, validation_days: int) -> pd.Timestamp:
    last_day = all_windows.max().normalize()
    return last_day - pd.Timedelta(days=validation_days - 1)


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
        raise ValueError(f"Unknown weather columns: {unknown}")
    return list(requested)


def build_training_dataset(
    train_grid: pd.DataFrame,
    history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    train_end: pd.Timestamp,
    default_value: float,
    use_calendar: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    metas: list[dict[str, int | pd.Timestamp]] = []

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
            cfg=feature_cfg,
            default_value=default_value,
            allow_fallback=False,
        )
        if feat is None:
            continue

        if use_calendar:
            feat.update(calendar_feature_vector(ts))

        if weather_table is not None and weather_defaults_map is not None:
            w = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
            feat.update({k: w[k] for k in weather_columns})

        rows.append(feat)
        targets.append(float(row.volume))
        metas.append(
            {
                "tollgate_id": key[0],
                "direction": key[1],
                "horizon": int(horizon_index(ts)),
                "time_window": ts,
                "day": ts.normalize(),
            }
        )

    return pd.DataFrame(rows), pd.Series(targets, name="volume"), pd.DataFrame(metas)


def make_xgb(model_cfg: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 200)),
        max_depth=int(model_cfg.get("max_depth", 6)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        subsample=float(model_cfg.get("subsample", 0.9)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.9)),
        reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(model_cfg.get("reg_lambda", 2.0)),
        min_child_weight=float(model_cfg.get("min_child_weight", 1.0)),
        gamma=float(model_cfg.get("gamma", 0.0)),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=int(model_cfg.get("seed", 42)),
        n_jobs=int(model_cfg.get("n_jobs", 4)),
        verbosity=0,
    )


def build_sample_weight(y: np.ndarray, model_cfg: dict) -> tuple[np.ndarray, str]:
    mode = str(model_cfg.get("sample_weight_mode", "inverse")).strip().lower()
    denom_floor = max(float(model_cfg.get("sample_weight_denom_floor", 1.0)), 1e-6)
    max_weight = float(model_cfg.get("sample_weight_max", 100.0))
    normalize = bool(model_cfg.get("sample_weight_normalize_mean", True))

    if mode == "uniform":
        w = np.ones_like(y, dtype=float)
    elif mode == "inverse_sqrt":
        w = 1.0 / np.sqrt(np.maximum(y, denom_floor))
    elif mode == "inverse":
        w = 1.0 / np.maximum(y, denom_floor)
    else:
        raise ValueError(f"Unknown sample_weight_mode={mode}, expected one of [uniform,inverse,inverse_sqrt]")

    w = np.clip(w, 1e-6, max_weight)
    if normalize:
        mean_w = float(np.mean(w))
        if mean_w > 0:
            w = w / mean_w
    return w.astype(float), mode


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    feature_names: list[str],
    model_cfg: dict,
) -> tuple[GBDTBundle, dict[str, float | int]]:
    use_log_target = bool(model_cfg.get("use_log_target", True))
    log_pred_clip = float(model_cfg.get("log_pred_clip", 6.0))
    group_min_samples = int(model_cfg.get("group_min_samples", 24))

    y_raw = y_train.to_numpy(dtype=float)
    y_fit = np.log1p(y_raw) if use_log_target else y_raw
    weight, weight_mode = build_sample_weight(y_raw, model_cfg)

    global_model = make_xgb(model_cfg)
    global_model.fit(x_train[feature_names], y_fit, sample_weight=weight)

    group_models: dict[tuple[int, int, int], XGBRegressor] = {}
    grouped = meta_train.groupby(["tollgate_id", "direction", "horizon"], sort=True)
    for (tollgate_id, direction, horizon), part in grouped:
        idx = part.index.to_numpy()
        if len(idx) < group_min_samples:
            continue
        x_sub = x_train.iloc[idx][feature_names]
        y_sub = y_fit[idx]
        w_sub = weight[idx]

        model = make_xgb(model_cfg)
        model.fit(x_sub, y_sub, sample_weight=w_sub)
        group_models[(int(tollgate_id), int(direction), int(horizon))] = model

    bundle = GBDTBundle(
        feature_names=feature_names,
        use_log_target=use_log_target,
        log_pred_clip=log_pred_clip,
        global_model=global_model,
        group_models=group_models,
    )
    stats = {
        "use_log_target": int(use_log_target),
        "group_models": int(len(group_models)),
        "group_min_samples": int(group_min_samples),
        "sample_weight_mode": weight_mode,
        "sample_weight_min": float(np.min(weight)),
        "sample_weight_max": float(np.max(weight)),
    }
    return bundle, stats


def predict_row(bundle: GBDTBundle, x_row: pd.DataFrame, key: tuple[int, int], horizon: int) -> float:
    model = bundle.group_models.get((key[0], key[1], horizon), bundle.global_model)
    raw = float(model.predict(x_row[bundle.feature_names])[0])
    if bundle.use_log_target:
        raw = min(raw, bundle.log_pred_clip)
        return max(0.0, float(np.expm1(raw)))
    return max(0.0, raw)


def recursive_predict(
    bundle: GBDTBundle,
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    default_value: float,
    use_calendar: bool,
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

            if use_calendar:
                feat.update(calendar_feature_vector(ts))

            if weather_table is not None and weather_defaults_map is not None:
                w = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
                feat.update({k: w[k] for k in weather_columns})

            x = pd.DataFrame([feat])
            h = int(horizon_index(ts))
            pred = predict_row(bundle, x, key, h)

            history[key].loc[ts] = pred
            history[key] = history[key].sort_index()

            rec: dict[str, float | int | pd.Timestamp] = {
                "tollgate_id": key[0],
                "direction": key[1],
                "time_window": ts,
                "horizon": h,
                "prediction": pred,
            }
            if actual_map is not None:
                rec["actual"] = float(actual_map[(key, ts)])
            records.append(rec)

    return pd.DataFrame(records)


def rolling_folds(days: list[pd.Timestamp], n_folds: int, val_days: int, min_train_days: int) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
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

    train_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["train_volume_csv"])
    train_agg = aggregate_to_20min(train_events)
    train_grid = complete_20min_grid(train_agg)
    train_history = build_series_history(train_grid)
    series_keys = sorted(train_history.keys())

    train_weather = None
    inference_weather = None
    weather_default_map = None
    if use_weather:
        train_weather = load_weather_table(PROJECT_ROOT / cfg["paths"]["train_weather_csv"])
        test_weather = load_weather_table(PROJECT_ROOT / cfg["paths"]["test_weather_csv"])
        inference_weather = merge_weather_tables(train_weather, test_weather)
        weather_default_map = weather_defaults(train_weather)

    split_ts = split_timestamp(train_grid["time_window"], int(cfg["validation"]["days"]))
    default_value = default_value_from_history(train_history)

    x_all, y_all, meta_all = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        use_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )
    if x_all.empty:
        raise RuntimeError("No training samples for pre-holdout period")

    feature_names = feature_columns(series_keys, feature_cfg, include_calendar=use_calendar) + weather_columns

    pre_days = sorted(meta_all["day"].drop_duplicates().tolist())
    roll_cfg = cfg.get("rolling_validation", {})
    folds = rolling_folds(
        days=pre_days,
        n_folds=int(roll_cfg.get("n_folds", 3)),
        val_days=int(roll_cfg.get("val_days", 2)),
        min_train_days=int(roll_cfg.get("min_train_days", 10)),
    )

    rolling_results: list[dict[str, float | int]] = []
    for idx, (train_days, val_days) in enumerate(folds, start=1):
        train_mask = meta_all["day"].isin(train_days)
        val_schedule = target_windows_for_days(val_days, horizon=int(cfg["target"]["horizon_windows"]))

        bundle, _ = train_models(
            x_train=x_all.loc[train_mask].reset_index(drop=True),
            y_train=y_all.loc[train_mask].reset_index(drop=True),
            meta_train=meta_all.loc[train_mask].reset_index(drop=True),
            feature_names=feature_names,
            model_cfg=cfg["model"],
        )

        roll_history = {k: s.copy() for k, s in train_history.items()}
        actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] = {}
        for key in series_keys:
            for ts in val_schedule:
                actual_map[(key, ts)] = float(roll_history[key].get(ts, np.nan))
                roll_history[key].loc[ts] = np.nan

        pred = recursive_predict(
            bundle=bundle,
            history=roll_history,
            schedule=val_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            use_calendar=use_calendar,
            weather_table=train_weather,
            weather_defaults_map=weather_default_map,
            weather_columns=weather_columns,
            actual_map=actual_map,
        )
        pred = pred.dropna(subset=["actual"]).reset_index(drop=True)
        m = summarize_metrics(pred)
        rolling_results.append(
            {
                "fold": idx,
                "train_days": len(train_days),
                "val_days": len(val_days),
                "overall_mape": float(m["overall_mape"]),
            }
        )

    final_bundle, model_stats = train_models(
        x_train=x_all,
        y_train=y_all,
        meta_train=meta_all,
        feature_names=feature_names,
        model_cfg=cfg["model"],
    )

    holdout_days = sorted(
        pd.to_datetime(train_grid.loc[train_grid["time_window"] >= split_ts, "time_window"].dt.normalize().unique())
    )
    holdout_schedule = target_windows_for_days(holdout_days, horizon=int(cfg["target"]["horizon_windows"]))

    holdout_history = {k: s.copy() for k, s in train_history.items()}
    holdout_actual: dict[tuple[tuple[int, int], pd.Timestamp], float] = {}
    for key in series_keys:
        for ts in holdout_schedule:
            holdout_actual[(key, ts)] = float(holdout_history[key].get(ts, np.nan))
            holdout_history[key].loc[ts] = np.nan

    holdout_pred = recursive_predict(
        bundle=final_bundle,
        history=holdout_history,
        schedule=holdout_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        use_calendar=use_calendar,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=holdout_actual,
    )
    holdout_pred = holdout_pred.dropna(subset=["actual"]).reset_index(drop=True)
    holdout_metrics = summarize_metrics(holdout_pred)

    test_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["test_volume_csv"])
    test_agg = aggregate_to_20min(test_events)
    test_history = build_series_history(test_agg)
    merged_history = merge_histories(train_history, test_history)

    test_days = sorted(pd.to_datetime(test_agg["time_window"].dt.normalize().unique()))
    test_schedule = target_windows_for_days(test_days, horizon=int(cfg["target"]["horizon_windows"]))

    test_pred = recursive_predict(
        bundle=final_bundle,
        history=merged_history,
        schedule=test_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        use_calendar=use_calendar,
        weather_table=inference_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )

    submission = build_submission(test_pred)
    validate_submission_schema(submission)

    rolling_avg = float(np.mean([x["overall_mape"] for x in rolling_results])) if rolling_results else float("nan")
    run_meta = {
        "run_id": run_id,
        "split_timestamp": split_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": int(len(x_all)),
        "validation_rows": int(len(holdout_pred)),
        "submission_rows": int(len(submission)),
        "modeling": model_stats,
        "rolling_validation": {
            "folds": rolling_results,
            "avg_mape": rolling_avg,
        },
        "metrics": holdout_metrics,
    }

    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout_pred.to_csv(run_dir / "validation_predictions.csv", index=False)

    sub_path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{run_id}.csv"
    submission.to_csv(sub_path, index=False)

    print(json.dumps(run_meta, ensure_ascii=False, indent=2))
    print(f"submission_path={sub_path}")


if __name__ == "__main__":
    main()
