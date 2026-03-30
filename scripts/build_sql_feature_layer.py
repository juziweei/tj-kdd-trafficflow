#!/usr/bin/env python3
"""Build a leakage-safe SQL feature snapshot for traffic volume forecasting."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_VOLUME = PROJECT_ROOT / "data/raw/dataset_60/training/volume(table 6)_training.csv"
DEFAULT_TRAIN_WEATHER = PROJECT_ROOT / "data/raw/dataset_60/training/weather (table 7)_training.csv"
DEFAULT_TEST_WEATHER = PROJECT_ROOT / "data/raw/dataset_60/testing_phase1/weather (table 7)_test1.csv"

WEATHER_COLUMNS = [
    "weather_pressure",
    "weather_sea_pressure",
    "weather_wind_speed",
    "weather_temperature",
    "weather_rel_humidity",
    "weather_precipitation",
    "weather_wind_dir_sin",
    "weather_wind_dir_cos",
]

KEY_COLUMNS = ["tollgate_id", "direction", "time_window"]
LAG_COLUMNS = [1, 2, 3, 6, 72, 504]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="", help="Run id used in output artifact names.")
    parser.add_argument("--train-volume-csv", type=Path, default=DEFAULT_TRAIN_VOLUME)
    parser.add_argument("--train-weather-csv", type=Path, default=DEFAULT_TRAIN_WEATHER)
    parser.add_argument("--test-weather-csv", type=Path, default=DEFAULT_TEST_WEATHER)
    parser.add_argument("--include-test-weather", action="store_true")
    parser.add_argument("--split-timestamp", default="2016-10-11 00:00:00")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/sql_features")
    return parser.parse_args()


def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _normalize_volume_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, usecols=["time", "tollgate_id", "direction"])
    events["time"] = pd.to_datetime(events["time"])
    events["tollgate_id"] = events["tollgate_id"].astype(int)
    events["direction"] = events["direction"].astype(int)
    events = events.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    events["time"] = events["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return events


def _normalize_weather_hourly(train_weather_csv: Path, test_weather_csv: Path | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = [pd.read_csv(train_weather_csv)]
    if test_weather_csv is not None and test_weather_csv.exists():
        frames.append(pd.read_csv(test_weather_csv))
    raw = pd.concat(frames, ignore_index=True)

    weather_time = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["hour"].astype(int), unit="h")
    work = pd.DataFrame({"weather_time": weather_time})
    work["weather_pressure"] = pd.to_numeric(raw["pressure"], errors="coerce")
    work["weather_sea_pressure"] = pd.to_numeric(raw["sea_pressure"], errors="coerce")
    work["weather_wind_speed"] = pd.to_numeric(raw["wind_speed"], errors="coerce")
    work["weather_temperature"] = pd.to_numeric(raw["temperature"], errors="coerce")
    work["weather_rel_humidity"] = pd.to_numeric(raw["rel_humidity"], errors="coerce")
    work["weather_precipitation"] = pd.to_numeric(raw["precipitation"], errors="coerce")

    wind_direction = pd.to_numeric(raw["wind_direction"], errors="coerce")
    wind_direction = wind_direction.where((wind_direction >= 0.0) & (wind_direction <= 360.0), np.nan)
    radians = np.deg2rad(wind_direction)
    work["weather_wind_dir_sin"] = np.sin(radians)
    work["weather_wind_dir_cos"] = np.cos(radians)

    out = work.sort_values("weather_time").drop_duplicates(subset=["weather_time"], keep="last")
    out = out.set_index("weather_time")
    full_hourly = pd.date_range(out.index.min(), out.index.max(), freq="1h")
    out = out.reindex(full_hourly).ffill().bfill()
    out = out.fillna(out.mean(numeric_only=True))
    out.index.name = "weather_time"
    out = out.reset_index()
    out["weather_time"] = out["weather_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def _build_sql_snapshot(conn: sqlite3.Connection) -> pd.DataFrame:
    conn.executescript(
        """
        DROP TABLE IF EXISTS volume_agg;
        CREATE TABLE volume_agg AS
        SELECT
            CAST(tollgate_id AS INTEGER) AS tollgate_id,
            CAST(direction AS INTEGER) AS direction,
            datetime((CAST(strftime('%s', time) AS INTEGER) / 1200) * 1200, 'unixepoch') AS time_window,
            COUNT(*) * 1.0 AS volume
        FROM volume_events
        GROUP BY 1, 2, 3;

        DROP TABLE IF EXISTS series_keys;
        CREATE TABLE series_keys AS
        SELECT DISTINCT tollgate_id, direction FROM volume_agg;

        DROP TABLE IF EXISTS time_grid;
        CREATE TABLE time_grid AS
        WITH RECURSIVE seq(time_window) AS (
            SELECT (SELECT MIN(time_window) FROM volume_agg)
            UNION ALL
            SELECT datetime(time_window, '+20 minutes')
            FROM seq
            WHERE time_window < (SELECT MAX(time_window) FROM volume_agg)
        )
        SELECT time_window FROM seq;

        DROP TABLE IF EXISTS volume_grid;
        CREATE TABLE volume_grid AS
        SELECT
            s.tollgate_id,
            s.direction,
            g.time_window,
            COALESCE(a.volume, 0.0) AS volume
        FROM series_keys AS s
        CROSS JOIN time_grid AS g
        LEFT JOIN volume_agg AS a
            ON a.tollgate_id = s.tollgate_id
           AND a.direction = s.direction
           AND a.time_window = g.time_window;

        DROP TABLE IF EXISTS volume_features;
        CREATE TABLE volume_features AS
        SELECT
            tollgate_id,
            direction,
            time_window,
            volume,
            LAG(volume, 1) OVER w AS lag_1,
            LAG(volume, 2) OVER w AS lag_2,
            LAG(volume, 3) OVER w AS lag_3,
            LAG(volume, 6) OVER w AS lag_6,
            LAG(volume, 72) OVER w AS lag_72,
            LAG(volume, 504) OVER w AS lag_504,
            AVG(volume) OVER (
                PARTITION BY tollgate_id, direction
                ORDER BY time_window
                ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
            ) AS mean_prev_6,
            ((CAST(strftime('%w', time_window) AS INTEGER) + 6) % 7) AS dow,
            CAST(strftime('%H', time_window) AS INTEGER) AS hour,
            ((CAST(strftime('%H', time_window) AS INTEGER) * 60 + CAST(strftime('%M', time_window) AS INTEGER)) / 20) AS slot_index,
            datetime((CAST(strftime('%s', time_window) AS INTEGER) / 3600) * 3600 - 3600, 'unixepoch') AS weather_anchor
        FROM volume_grid
        WINDOW w AS (PARTITION BY tollgate_id, direction ORDER BY time_window);

        DROP TABLE IF EXISTS feature_snapshot;
        CREATE TABLE feature_snapshot AS
        SELECT
            f.tollgate_id,
            f.direction,
            f.time_window,
            f.volume,
            f.lag_1,
            f.lag_2,
            f.lag_3,
            f.lag_6,
            f.lag_72,
            f.lag_504,
            f.mean_prev_6,
            f.dow,
            f.hour,
            f.slot_index,
            f.weather_anchor,
            COALESCE(w.weather_pressure, wm.weather_pressure_mean) AS weather_pressure,
            COALESCE(w.weather_sea_pressure, wm.weather_sea_pressure_mean) AS weather_sea_pressure,
            COALESCE(w.weather_wind_speed, wm.weather_wind_speed_mean) AS weather_wind_speed,
            COALESCE(w.weather_temperature, wm.weather_temperature_mean) AS weather_temperature,
            COALESCE(w.weather_rel_humidity, wm.weather_rel_humidity_mean) AS weather_rel_humidity,
            COALESCE(w.weather_precipitation, wm.weather_precipitation_mean) AS weather_precipitation,
            COALESCE(w.weather_wind_dir_sin, wm.weather_wind_dir_sin_mean) AS weather_wind_dir_sin,
            COALESCE(w.weather_wind_dir_cos, wm.weather_wind_dir_cos_mean) AS weather_wind_dir_cos
        FROM volume_features AS f
        LEFT JOIN weather_hourly AS w
            ON w.weather_time = f.weather_anchor
        CROSS JOIN (
            SELECT
                AVG(weather_pressure) AS weather_pressure_mean,
                AVG(weather_sea_pressure) AS weather_sea_pressure_mean,
                AVG(weather_wind_speed) AS weather_wind_speed_mean,
                AVG(weather_temperature) AS weather_temperature_mean,
                AVG(weather_rel_humidity) AS weather_rel_humidity_mean,
                AVG(weather_precipitation) AS weather_precipitation_mean,
                AVG(weather_wind_dir_sin) AS weather_wind_dir_sin_mean,
                AVG(weather_wind_dir_cos) AS weather_wind_dir_cos_mean
            FROM weather_hourly
        ) AS wm;
        """
    )
    snapshot = pd.read_sql_query(
        """
        SELECT
            tollgate_id,
            direction,
            time_window,
            volume,
            lag_1, lag_2, lag_3, lag_6, lag_72, lag_504,
            mean_prev_6,
            dow, hour, slot_index,
            weather_anchor,
            weather_pressure,
            weather_sea_pressure,
            weather_wind_speed,
            weather_temperature,
            weather_rel_humidity,
            weather_precipitation,
            weather_wind_dir_sin,
            weather_wind_dir_cos
        FROM feature_snapshot
        ORDER BY tollgate_id, direction, time_window
        """,
        conn,
        parse_dates=["time_window", "weather_anchor"],
    )
    return snapshot


def _build_quality_report(df: pd.DataFrame, split_timestamp: pd.Timestamp, run_id: str, args: argparse.Namespace) -> dict:
    key_count = int(df[KEY_COLUMNS].drop_duplicates().shape[0])
    row_count = int(len(df))
    duplicate_key_count = int(row_count - key_count)
    split_mask = df["time_window"] >= split_timestamp
    series_counts = df.groupby(["tollgate_id", "direction"], sort=True).size()

    report = {
        "run_id": run_id,
        "split_timestamp": split_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "input_paths": {
            "train_volume_csv": str(args.train_volume_csv),
            "train_weather_csv": str(args.train_weather_csv),
            "test_weather_csv": str(args.test_weather_csv) if args.include_test_weather else None,
        },
        "rows": {
            "row_count": row_count,
            "unique_keys": key_count,
            "duplicate_key_count": duplicate_key_count,
            "series_count": int(series_counts.shape[0]),
            "series_row_count_min": int(series_counts.min()),
            "series_row_count_max": int(series_counts.max()),
            "time_start": df["time_window"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_end": df["time_window"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "train_rows": int((~split_mask).sum()),
            "valid_rows": int(split_mask.sum()),
        },
        "missing_ratio": {col: float(df[col].isna().mean()) for col in df.columns},
    }
    return report


def main() -> None:
    args = parse_args()
    _require_file(args.train_volume_csv, "train volume csv")
    _require_file(args.train_weather_csv, "train weather csv")
    if args.include_test_weather:
        _require_file(args.test_weather_csv, "test weather csv")

    run_id = args.run_id.strip() or f"sql_feature_layer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    split_timestamp = pd.Timestamp(args.split_timestamp)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = output_dir / f"{run_id}.sqlite"
    snapshot_csv = output_dir / f"sql_feature_snapshot_{run_id}.csv"
    quality_json = output_dir / f"sql_feature_quality_{run_id}.json"

    if sqlite_path.exists():
        sqlite_path.unlink()

    volume_events = _normalize_volume_events(args.train_volume_csv)
    test_weather = args.test_weather_csv if args.include_test_weather else None
    weather_hourly = _normalize_weather_hourly(args.train_weather_csv, test_weather)

    with sqlite3.connect(sqlite_path) as conn:
        volume_events.to_sql("volume_events", conn, index=False)
        weather_hourly.to_sql("weather_hourly", conn, index=False)
        snapshot = _build_sql_snapshot(conn)

    snapshot.to_csv(snapshot_csv, index=False)
    quality = _build_quality_report(snapshot, split_timestamp=split_timestamp, run_id=run_id, args=args)
    quality_json.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"snapshot_csv={snapshot_csv}")
    print(f"quality_json={quality_json}")
    print(f"sqlite_db={sqlite_path}")
    print(f"rows={len(snapshot)}")
    print(f"time_range={snapshot['time_window'].min()} -> {snapshot['time_window'].max()}")


if __name__ == "__main__":
    main()
