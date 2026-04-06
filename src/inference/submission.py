"""Submission formatting and schema validation."""

from __future__ import annotations

import pandas as pd


def format_time_window(ts: pd.Timestamp) -> str:
    end = ts + pd.Timedelta(minutes=20)
    return f"[{ts.strftime('%Y-%m-%d %H:%M:%S')},{end.strftime('%Y-%m-%d %H:%M:%S')})"


def build_submission(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["time_window"] = out["time_window"].map(format_time_window)
    out = out[["tollgate_id", "direction", "time_window", "prediction"]].rename(
        columns={"prediction": "volume"}
    )
    out["tollgate_id"] = out["tollgate_id"].astype(int)
    out["direction"] = out["direction"].astype(int)
    return out


def validate_submission_schema(submission: pd.DataFrame) -> None:
    expected_cols = ["tollgate_id", "direction", "time_window", "volume"]
    if list(submission.columns) != expected_cols:
        raise ValueError(f"Invalid submission columns: {submission.columns.tolist()}")
    if submission.duplicated(subset=["tollgate_id", "direction", "time_window"]).any():
        raise ValueError("Submission contains duplicate keys")
    if (submission["volume"] < 0).any():
        raise ValueError("Submission contains negative volume")
