"""Evaluation metrics for traffic volume forecasting."""

from __future__ import annotations

import pandas as pd


def mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1.0) -> float:
    denom = y_true.abs().clip(lower=eps)
    return float(((y_true - y_pred).abs() / denom).mean() * 100.0)


def build_error_slice_table(pred_df: pd.DataFrame, eps: float = 1.0) -> pd.DataFrame:
    frame = pred_df.copy()
    denom = frame["actual"].abs().clip(lower=eps)
    frame["ape"] = (frame["actual"] - frame["prediction"]).abs() / denom
    frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()

    grouped = frame.groupby(["tollgate_id", "direction", "horizon"], sort=True)
    out = grouped.agg(
        rows=("ape", "size"),
        mape=("ape", "mean"),
        mae=("abs_error", "mean"),
        median_ape=("ape", "median"),
        p90_ape=("ape", lambda s: s.quantile(0.9)),
    ).reset_index()

    for col in ("mape", "median_ape", "p90_ape"):
        out[col] = out[col] * 100.0
    return out


def summarize_metrics(pred_df: pd.DataFrame) -> dict[str, float | dict[str, float]]:
    overall = mape(pred_df["actual"], pred_df["prediction"])

    by_horizon: dict[str, float] = {}
    for horizon, part in pred_df.groupby("horizon"):
        by_horizon[str(int(horizon))] = mape(part["actual"], part["prediction"])

    by_series: dict[str, float] = {}
    for (tollgate_id, direction), part in pred_df.groupby(["tollgate_id", "direction"]):
        key = f"{int(tollgate_id)}_{int(direction)}"
        by_series[key] = mape(part["actual"], part["prediction"])

    by_series_horizon: dict[str, float] = {}
    for (tollgate_id, direction, horizon), part in pred_df.groupby(
        ["tollgate_id", "direction", "horizon"]
    ):
        key = f"{int(tollgate_id)}_{int(direction)}_h{int(horizon)}"
        by_series_horizon[key] = mape(part["actual"], part["prediction"])

    return {
        "overall_mape": overall,
        "horizon_mape": by_horizon,
        "series_mape": by_series,
        "series_horizon_mape": by_series_horizon,
    }
