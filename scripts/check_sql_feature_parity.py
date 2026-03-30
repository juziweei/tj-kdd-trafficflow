#!/usr/bin/env python3
"""Compare SQL feature snapshot against pandas pipeline features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.volume_io import aggregate_to_20min, complete_20min_grid, load_volume_events  # noqa: E402
from src.data.weather_io import (  # noqa: E402
    WEATHER_FEATURE_COLUMNS,
    get_weather_feature_vector,
    load_weather_table,
    merge_weather_tables,
    weather_defaults,
)
from src.eval.metrics import mape  # noqa: E402
from src.features.volume_features import is_target_window  # noqa: E402

DEFAULT_TRAIN_VOLUME = PROJECT_ROOT / "data/raw/dataset_60/training/volume(table 6)_training.csv"
DEFAULT_TRAIN_WEATHER = PROJECT_ROOT / "data/raw/dataset_60/training/weather (table 7)_training.csv"
DEFAULT_TEST_WEATHER = PROJECT_ROOT / "data/raw/dataset_60/testing_phase1/weather (table 7)_test1.csv"

KEY_COLUMNS = ["tollgate_id", "direction", "time_window"]
COMPARE_COLUMNS = [
    "volume",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_72",
    "lag_504",
    "mean_prev_6",
    "dow",
    "hour",
    "slot_index",
    "weather_pressure",
    "weather_sea_pressure",
    "weather_wind_speed",
    "weather_temperature",
    "weather_rel_humidity",
    "weather_precipitation",
    "weather_wind_dir_sin",
    "weather_wind_dir_cos",
]
LAG_STEPS = [1, 2, 3, 6, 72, 504]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql-snapshot-csv", type=Path, required=True)
    parser.add_argument("--train-volume-csv", type=Path, default=DEFAULT_TRAIN_VOLUME)
    parser.add_argument("--train-weather-csv", type=Path, default=DEFAULT_TRAIN_WEATHER)
    parser.add_argument("--test-weather-csv", type=Path, default=DEFAULT_TEST_WEATHER)
    parser.add_argument("--include-test-weather", action="store_true")
    parser.add_argument("--split-timestamp", default="2016-10-11 00:00:00")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _build_pandas_reference(args: argparse.Namespace) -> pd.DataFrame:
    events = load_volume_events(args.train_volume_csv)
    agg = aggregate_to_20min(events)
    grid = complete_20min_grid(agg).sort_values(KEY_COLUMNS).reset_index(drop=True)

    by_series = grid.groupby(["tollgate_id", "direction"], sort=False)
    for lag in LAG_STEPS:
        grid[f"lag_{lag}"] = by_series["volume"].shift(lag)
    grid["mean_prev_6"] = by_series["volume"].transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).mean())

    ts = pd.to_datetime(grid["time_window"])
    grid["dow"] = ts.dt.dayofweek
    grid["hour"] = ts.dt.hour
    grid["slot_index"] = ((ts.dt.hour * 60 + ts.dt.minute) // 20).astype(int)

    train_weather = load_weather_table(args.train_weather_csv)
    weather_table = train_weather
    if args.include_test_weather:
        _require_file(args.test_weather_csv, "test weather csv")
        weather_table = merge_weather_tables(train_weather, load_weather_table(args.test_weather_csv))
    defaults = weather_defaults(train_weather)

    weather_rows = [
        get_weather_feature_vector(weather_table, pd.Timestamp(t), defaults)
        for t in pd.to_datetime(grid["time_window"])
    ]
    weather_df = pd.DataFrame(weather_rows, columns=WEATHER_FEATURE_COLUMNS)
    out = pd.concat([grid.reset_index(drop=True), weather_df.reset_index(drop=True)], axis=1)
    out["time_window"] = pd.to_datetime(out["time_window"])
    return out


def _compute_naive_time_validation(frame: pd.DataFrame, split_timestamp: pd.Timestamp) -> dict[str, float | int]:
    ts = pd.to_datetime(frame["time_window"])
    valid_mask = ts >= split_timestamp
    train_mask = ts < split_timestamp
    lag_ready = frame["lag_1"].notna()
    target_mask = ts.map(is_target_window)

    def _safe_mape(mask: pd.Series) -> float:
        local = mask & lag_ready
        if int(local.sum()) == 0:
            return float("nan")
        return mape(frame.loc[local, "volume"], frame.loc[local, "lag_1"])

    return {
        "train_rows": int((train_mask & lag_ready).sum()),
        "valid_rows": int((valid_mask & lag_ready).sum()),
        "train_target_rows": int((train_mask & lag_ready & target_mask).sum()),
        "valid_target_rows": int((valid_mask & lag_ready & target_mask).sum()),
        "train_mape_lag1_all": _safe_mape(train_mask),
        "valid_mape_lag1_all": _safe_mape(valid_mask),
        "train_mape_lag1_target_window": _safe_mape(train_mask & target_mask),
        "valid_mape_lag1_target_window": _safe_mape(valid_mask & target_mask),
    }


def _column_diff_summary(merged: pd.DataFrame) -> dict[str, dict[str, float]]:
    both = merged[merged["_merge"] == "both"].copy()
    out: dict[str, dict[str, float]] = {}
    for col in COMPARE_COLUMNS:
        sql_col = f"{col}_sql"
        pd_col = f"{col}_pd"
        if sql_col not in both.columns or pd_col not in both.columns:
            continue
        diff = (both[sql_col] - both[pd_col]).abs()
        finite = diff.replace([np.inf, -np.inf], np.nan)
        out[col] = {
            "mean_abs_diff": float(np.nanmean(finite.to_numpy(dtype=float))),
            "max_abs_diff": float(np.nanmax(finite.to_numpy(dtype=float))),
            "p99_abs_diff": float(np.nanquantile(finite.to_numpy(dtype=float), 0.99)),
            "exact_ratio": float(np.nanmean((finite.fillna(0.0) <= 1e-12).to_numpy(dtype=float))),
            "nonnull_pairs": int((both[sql_col].notna() & both[pd_col].notna()).sum()),
        }
    return out


def _resolve_output_json(args: argparse.Namespace) -> Path:
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        return args.output_json
    stem = args.sql_snapshot_csv.stem.replace("sql_feature_snapshot_", "")
    output = PROJECT_ROOT / "outputs/sql_features" / f"sql_feature_parity_{stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main() -> None:
    args = parse_args()
    _require_file(args.sql_snapshot_csv, "sql snapshot csv")
    _require_file(args.train_volume_csv, "train volume csv")
    _require_file(args.train_weather_csv, "train weather csv")

    split_timestamp = pd.Timestamp(args.split_timestamp)

    sql_df = pd.read_csv(args.sql_snapshot_csv, parse_dates=["time_window", "weather_anchor"])
    pd_df = _build_pandas_reference(args)

    merged = sql_df.merge(pd_df, on=KEY_COLUMNS, how="outer", suffixes=("_sql", "_pd"), indicator=True)
    coverage = merged["_merge"].value_counts().to_dict()
    both_rows = int(coverage.get("both", 0))
    left_only_rows = int(coverage.get("left_only", 0))
    right_only_rows = int(coverage.get("right_only", 0))

    col_diffs = _column_diff_summary(merged)
    max_diff_overall = max((v["max_abs_diff"] for v in col_diffs.values()), default=float("nan"))

    sql_eval = _compute_naive_time_validation(sql_df, split_timestamp=split_timestamp)
    pd_eval = _compute_naive_time_validation(pd_df, split_timestamp=split_timestamp)

    strict_parity_pass = left_only_rows == 0 and right_only_rows == 0 and float(max_diff_overall) <= 1e-9

    report = {
        "split_timestamp": split_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "input_paths": {
            "sql_snapshot_csv": str(args.sql_snapshot_csv),
            "train_volume_csv": str(args.train_volume_csv),
            "train_weather_csv": str(args.train_weather_csv),
            "test_weather_csv": str(args.test_weather_csv) if args.include_test_weather else None,
        },
        "coverage": {
            "both_rows": both_rows,
            "sql_only_rows": left_only_rows,
            "pandas_only_rows": right_only_rows,
        },
        "diff_summary": col_diffs,
        "max_abs_diff_overall": float(max_diff_overall),
        "time_based_validation": {
            "sql_naive_lag1": sql_eval,
            "pandas_naive_lag1": pd_eval,
            "valid_target_mape_gap_abs": float(
                abs(sql_eval["valid_mape_lag1_target_window"] - pd_eval["valid_mape_lag1_target_window"])
            ),
        },
        "strict_parity_pass": strict_parity_pass,
    }

    output_json = _resolve_output_json(args)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report_json={output_json}")
    print(f"strict_parity_pass={strict_parity_pass}")
    print(
        "coverage="
        f"both:{both_rows}, sql_only:{left_only_rows}, pandas_only:{right_only_rows}"
    )
    print(f"max_abs_diff_overall={report['max_abs_diff_overall']:.12f}")
    print(
        "valid_target_mape_gap_abs="
        f"{report['time_based_validation']['valid_target_mape_gap_abs']:.12f}"
    )


if __name__ == "__main__":
    main()
