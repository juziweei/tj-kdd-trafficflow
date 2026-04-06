"""Data loading and aggregation utilities for tollgate volume forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY_COLS = ["tollgate_id", "direction", "time_window"]


def load_volume_events(path: Path) -> pd.DataFrame:
    """Load raw vehicle-pass records from a CSV file."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["tollgate_id"] = df["tollgate_id"].astype(int)
    df["direction"] = df["direction"].astype(int)
    return df


def aggregate_to_20min(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level records into 20-minute volume windows."""
    work = events.copy()
    work["time_window"] = work["time"].dt.floor("20min")
    agg = (
        work.groupby(KEY_COLS, as_index=False)
        .size()
        .rename(columns={"size": "volume"})
        .sort_values(KEY_COLS)
        .reset_index(drop=True)
    )
    agg["volume"] = agg["volume"].astype(float)
    return agg


def complete_20min_grid(
    agg: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fill missing 20-minute windows with zero volume for each series."""
    if agg.empty:
        return agg.copy()

    start_ts = pd.Timestamp(start) if start is not None else agg["time_window"].min()
    end_ts = pd.Timestamp(end) if end is not None else agg["time_window"].max()
    full_idx = pd.date_range(start_ts, end_ts, freq="20min")

    parts: list[pd.DataFrame] = []
    for (tollgate_id, direction), part in agg.groupby(["tollgate_id", "direction"], sort=True):
        base = pd.DataFrame(
            {
                "tollgate_id": int(tollgate_id),
                "direction": int(direction),
                "time_window": full_idx,
            }
        )
        merged = base.merge(part, on=KEY_COLS, how="left")
        merged["volume"] = merged["volume"].fillna(0.0).astype(float)
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(KEY_COLS).reset_index(drop=True)


def build_series_history(agg: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    """Build per-series time-indexed history from aggregated data."""
    history: dict[tuple[int, int], pd.Series] = {}
    for (tollgate_id, direction), part in agg.groupby(["tollgate_id", "direction"], sort=True):
        series = part.sort_values("time_window").set_index("time_window")["volume"]
        history[(int(tollgate_id), int(direction))] = series.astype(float).copy()
    return history


def merge_histories(*history_maps: dict[tuple[int, int], pd.Series]) -> dict[tuple[int, int], pd.Series]:
    """Merge multiple per-series histories in chronological order."""
    merged: dict[tuple[int, int], pd.Series] = {}
    for mapping in history_maps:
        for key, series in mapping.items():
            if key not in merged:
                merged[key] = series.copy()
            else:
                merged[key] = pd.concat([merged[key], series]).sort_index()
                merged[key] = merged[key][~merged[key].index.duplicated(keep="last")]
    return merged
