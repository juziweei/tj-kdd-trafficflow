"""Weather data utilities for leakage-safe forecasting features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from os import PathLike

RAW_WEATHER_COLS = [
    "pressure",
    "sea_pressure",
    "wind_direction",
    "wind_speed",
    "temperature",
    "rel_humidity",
    "precipitation",
]

WEATHER_FEATURE_COLUMNS = [
    "weather_pressure",
    "weather_sea_pressure",
    "weather_wind_speed",
    "weather_temperature",
    "weather_rel_humidity",
    "weather_precipitation",
    "weather_wind_dir_sin",
    "weather_wind_dir_cos",
]


def _to_hourly_table(df: pd.DataFrame) -> pd.DataFrame:
    weather_time = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")

    work = pd.DataFrame({"weather_time": weather_time})
    for col in RAW_WEATHER_COLS:
        work[col] = pd.to_numeric(df[col], errors="coerce")

    # Weather source uses sentinel values for missing wind direction.
    work.loc[(work["wind_direction"] < 0) | (work["wind_direction"] > 360), "wind_direction"] = np.nan

    radians = np.deg2rad(work["wind_direction"])
    work["weather_wind_dir_sin"] = np.sin(radians)
    work["weather_wind_dir_cos"] = np.cos(radians)

    out = pd.DataFrame(
        {
            "weather_pressure": work["pressure"].to_numpy(),
            "weather_sea_pressure": work["sea_pressure"].to_numpy(),
            "weather_wind_speed": work["wind_speed"].to_numpy(),
            "weather_temperature": work["temperature"].to_numpy(),
            "weather_rel_humidity": work["rel_humidity"].to_numpy(),
            "weather_precipitation": work["precipitation"].to_numpy(),
            "weather_wind_dir_sin": work["weather_wind_dir_sin"].to_numpy(),
            "weather_wind_dir_cos": work["weather_wind_dir_cos"].to_numpy(),
        },
        index=weather_time,
    )
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    full_hourly_index = pd.date_range(out.index.min(), out.index.max(), freq="1h")
    out = out.reindex(full_hourly_index).ffill().bfill()
    out = out.fillna(out.mean(numeric_only=True))
    return out


def load_weather_table(path: str | PathLike[str]) -> pd.DataFrame:
    """Load weather CSV and return an hourly forward-filled table."""
    raw = pd.read_csv(path)
    return _to_hourly_table(raw)


def merge_weather_tables(*tables: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly weather tables by timestamp and fill gaps."""
    merged = pd.concat(tables, axis=0).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    hourly_index = pd.date_range(merged.index.min(), merged.index.max(), freq="1h")
    merged = merged.reindex(hourly_index).ffill().bfill()
    merged = merged.fillna(merged.mean(numeric_only=True))
    return merged


def weather_defaults(table: pd.DataFrame) -> dict[str, float]:
    means = table.mean(numeric_only=True)
    return {col: float(means[col]) for col in WEATHER_FEATURE_COLUMNS}


def get_weather_feature_vector(
    table: pd.DataFrame,
    ts: pd.Timestamp,
    defaults: dict[str, float],
) -> dict[str, float]:
    """Get leakage-safe weather features using previous full hour (ts.floor('h') - 1h)."""
    anchor = ts.floor("1h") - pd.Timedelta(hours=1)

    if anchor <= table.index.min():
        row = table.iloc[0]
    elif anchor >= table.index.max():
        row = table.iloc[-1]
    elif anchor in table.index:
        row = table.loc[anchor]
    else:
        row = table.asof(anchor)

    out: dict[str, float] = {}
    for col in WEATHER_FEATURE_COLUMNS:
        val = row.get(col, np.nan)
        if pd.isna(val):
            val = defaults[col]
        out[col] = float(val)
    return out
