#!/usr/bin/env python3
"""Fuse ensemble members on validation and submission outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.submission import validate_submission_schema  # noqa: E402

VALID_KEYS = ["tollgate_id", "direction", "time_window", "horizon"]
SUBMISSION_KEYS = ["tollgate_id", "direction", "time_window"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse multiple run outputs by simple averaging")
    parser.add_argument("--run-ids", type=str, default=None, help="Comma-separated run IDs")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Used when --run-ids is omitted")
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="leakfree_ensemble_seed",
        help="Used when --run-ids is omitted: run_id=<run_prefix>_<seed>",
    )
    parser.add_argument(
        "--ensemble-run-id",
        type=str,
        default="leakfree_ensemble_fused_20260329",
        help="Output run_id for fused artifacts",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Keep top-k members after ranking (0 means keep all)")
    parser.add_argument(
        "--rank-by",
        type=str,
        choices=["holdout", "rolling", "hybrid"],
        default="hybrid",
        help="Ranking score source for top-k and auto-weighting",
    )
    parser.add_argument(
        "--rolling-lambda",
        type=float,
        default=0.25,
        help="hybrid score = holdout + rolling_lambda * rolling_avg",
    )
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["equal", "inverse", "softmax", "manual"],
        default="equal",
        help="Weighting strategy among selected members",
    )
    parser.add_argument(
        "--manual-weights",
        type=str,
        default=None,
        help="Comma-separated manual weights; required when --weighting=manual",
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=0.05,
        help="Temperature for softmax weighting over negative score; lower means sharper",
    )
    parser.add_argument("--mape-eps", type=float, default=1.0, help="Numerical stabilizer in MAPE")
    return parser.parse_args()


def parse_csv_tokens(raw: str) -> list[str]:
    out = [token.strip() for token in raw.split(",") if token.strip()]
    if not out:
        raise ValueError("Empty comma-separated input")
    return out


def parse_run_ids(args: argparse.Namespace) -> list[str]:
    if args.run_ids:
        return parse_csv_tokens(args.run_ids)
    seeds = [int(token) for token in parse_csv_tokens(args.seeds)]
    return [f"{args.run_prefix}_{seed}" for seed in seeds]


def parse_float_tokens(raw: str) -> list[float]:
    return [float(token) for token in parse_csv_tokens(raw)]


def load_run_scores(run_id: str) -> dict[str, float | None]:
    path = PROJECT_ROOT / "outputs" / "runs" / run_id / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run metrics for run_id={run_id}: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" in obj and isinstance(obj["metrics"], dict):
        holdout = obj["metrics"].get("overall_mape")
        rolling = ((obj.get("rolling_validation") or {}).get("avg_mape"))
    else:
        holdout = obj.get("overall_mape")
        rolling = obj.get("rolling_avg_mape")
    holdout_v = float(holdout) if holdout is not None else None
    rolling_v = float(rolling) if rolling is not None else None
    return {"holdout": holdout_v, "rolling": rolling_v}


def rank_value(
    score: dict[str, float | None],
    rank_by: str,
    rolling_lambda: float,
) -> float:
    holdout = score.get("holdout")
    rolling = score.get("rolling")
    if rank_by == "holdout":
        if holdout is None:
            raise ValueError("Missing holdout score for rank_by=holdout")
        return float(holdout)
    if rank_by == "rolling":
        if rolling is None:
            raise ValueError("Missing rolling score for rank_by=rolling")
        return float(rolling)
    if holdout is None:
        raise ValueError("Missing holdout score for rank_by=hybrid")
    if rolling is None:
        return float(holdout)
    return float(holdout + rolling_lambda * rolling)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weight sum must be positive")
    return weights / total


def build_weights(
    args: argparse.Namespace,
    selected_scores: list[dict[str, float | None]],
    selected_rank_values: list[float],
) -> np.ndarray:
    n = len(selected_scores)
    if n == 0:
        raise ValueError("No selected runs")
    if args.weighting == "equal":
        return np.full(n, 1.0 / n, dtype=float)
    if args.weighting == "manual":
        if not args.manual_weights:
            raise ValueError("--manual-weights is required when --weighting=manual")
        vals = np.array(parse_float_tokens(args.manual_weights), dtype=float)
        if len(vals) != n:
            raise ValueError(f"manual weights count ({len(vals)}) != selected runs ({n})")
        if (vals < 0).any():
            raise ValueError("manual weights must be non-negative")
        return normalize_weights(vals)
    values = np.array(selected_rank_values, dtype=float)
    if args.weighting == "inverse":
        inv = 1.0 / np.maximum(values, args.mape_eps)
        return normalize_weights(inv)
    if args.weighting == "softmax":
        if args.softmax_temp <= 0:
            raise ValueError("--softmax-temp must be positive")
        shifted = values - values.min()
        logits = -shifted / float(args.softmax_temp)
        logits -= logits.max()
        probs = np.exp(logits)
        return normalize_weights(probs)
    raise ValueError(f"Unknown weighting: {args.weighting}")


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def load_validation_for_run(run_id: str, idx: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "runs" / run_id / "validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation predictions for run_id={run_id}: {path}")
    df = pd.read_csv(path)
    missing = [col for col in VALID_KEYS + ["actual", "prediction"] if col not in df.columns]
    if missing:
        raise ValueError(f"Run {run_id} missing validation columns: {missing}")
    one = df[VALID_KEYS + ["actual", "prediction"]].copy()
    if one.duplicated(VALID_KEYS).any():
        dup_cnt = int(one.duplicated(VALID_KEYS).sum())
        raise ValueError(f"Run {run_id} has {dup_cnt} duplicate validation keys")
    one = one.rename(
        columns={
            "actual": f"actual_{idx}",
            "prediction": f"prediction_{idx}",
        }
    )
    return one


def load_submission_for_run(run_id: str, idx: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing submission for run_id={run_id}: {path}")
    df = pd.read_csv(path)
    missing = [col for col in SUBMISSION_KEYS + ["volume"] if col not in df.columns]
    if missing:
        raise ValueError(f"Run {run_id} missing submission columns: {missing}")
    one = df[SUBMISSION_KEYS + ["volume"]].copy()
    if one.duplicated(SUBMISSION_KEYS).any():
        dup_cnt = int(one.duplicated(SUBMISSION_KEYS).sum())
        raise ValueError(f"Run {run_id} has {dup_cnt} duplicate submission keys")
    one = one.rename(columns={"volume": f"volume_{idx}"})
    return one


def align_and_check_actual(merged: pd.DataFrame, actual_cols: Sequence[str]) -> np.ndarray:
    base = merged[actual_cols[0]].to_numpy()
    for col in actual_cols[1:]:
        cur = merged[col].to_numpy()
        if not np.allclose(base, cur, equal_nan=True):
            max_diff = float(np.nanmax(np.abs(base - cur)))
            raise ValueError(f"Actual mismatch across runs for column {col}, max_diff={max_diff:.6f}")
    return base


def build_error_slices(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    tmp = df.copy()
    tmp["series_key"] = tmp["tollgate_id"].astype(str) + "_" + tmp["direction"].astype(str)
    records: list[dict[str, object]] = []

    for key, group in tmp.groupby("series_key"):
        records.append(
            {
                "slice_type": "series",
                "slice_key": key,
                "rows": int(len(group)),
                "mape": mape(group["actual"].to_numpy(), group["prediction"].to_numpy(), eps),
            }
        )
    for horizon, group in tmp.groupby("horizon"):
        records.append(
            {
                "slice_type": "horizon",
                "slice_key": f"h{int(horizon)}",
                "rows": int(len(group)),
                "mape": mape(group["actual"].to_numpy(), group["prediction"].to_numpy(), eps),
            }
        )
    for (key, horizon), group in tmp.groupby(["series_key", "horizon"]):
        records.append(
            {
                "slice_type": "series_horizon",
                "slice_key": f"{key}_h{int(horizon)}",
                "rows": int(len(group)),
                "mape": mape(group["actual"].to_numpy(), group["prediction"].to_numpy(), eps),
            }
        )
    out = pd.DataFrame.from_records(records)
    out = out.sort_values(["slice_type", "slice_key"]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    run_ids = parse_run_ids(args)

    score_items: list[tuple[str, dict[str, float | None], float]] = []
    for run_id in run_ids:
        scores = load_run_scores(run_id)
        rv = rank_value(scores, args.rank_by, args.rolling_lambda)
        score_items.append((run_id, scores, rv))
    score_items.sort(key=lambda x: x[2])

    if args.top_k > 0:
        score_items = score_items[: args.top_k]
    selected_run_ids = [x[0] for x in score_items]
    selected_scores = [x[1] for x in score_items]
    selected_rank_values = [x[2] for x in score_items]
    member_weights = build_weights(args, selected_scores, selected_rank_values)

    val_merged: pd.DataFrame | None = None
    val_pred_cols: list[str] = []
    val_actual_cols: list[str] = []

    for idx, run_id in enumerate(selected_run_ids):
        cur = load_validation_for_run(run_id, idx)
        val_pred_cols.append(f"prediction_{idx}")
        val_actual_cols.append(f"actual_{idx}")
        if val_merged is None:
            val_merged = cur
        else:
            val_merged = val_merged.merge(cur, on=VALID_KEYS, how="inner")

    if val_merged is None:
        raise RuntimeError("No runs provided")
    if len(val_merged) == 0:
        raise RuntimeError("Validation merge produced zero rows")

    if val_merged[val_pred_cols + val_actual_cols].isnull().any().any():
        raise RuntimeError("Validation merge contains nulls after alignment")

    actual = align_and_check_actual(val_merged, val_actual_cols)
    pred_matrix = val_merged[val_pred_cols].to_numpy(dtype=float)
    fused_pred = pred_matrix @ member_weights
    overall_mape = mape(actual, fused_pred, args.mape_eps)

    fused_val = val_merged[VALID_KEYS].copy()
    fused_val["actual"] = actual
    fused_val["prediction"] = fused_pred
    fused_val = fused_val.sort_values(VALID_KEYS).reset_index(drop=True)
    slice_df = build_error_slices(fused_val, args.mape_eps)

    sub_merged: pd.DataFrame | None = None
    sub_cols: list[str] = []
    for idx, run_id in enumerate(selected_run_ids):
        cur = load_submission_for_run(run_id, idx)
        col = f"volume_{idx}"
        sub_cols.append(col)
        if sub_merged is None:
            sub_merged = cur
        else:
            sub_merged = sub_merged.merge(cur, on=SUBMISSION_KEYS, how="inner")

    if sub_merged is None or len(sub_merged) == 0:
        raise RuntimeError("Submission merge failed")
    if sub_merged[sub_cols].isnull().any().any():
        raise RuntimeError("Submission merge contains nulls after alignment")

    fused_submission = sub_merged[SUBMISSION_KEYS].copy()
    sub_matrix = sub_merged[sub_cols].to_numpy(dtype=float)
    fused_submission["volume"] = sub_matrix @ member_weights
    fused_submission = fused_submission.sort_values(SUBMISSION_KEYS).reset_index(drop=True)
    validate_submission_schema(fused_submission)

    run_dir = PROJECT_ROOT / "outputs" / "runs" / args.ensemble_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = PROJECT_ROOT / "outputs" / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    val_path = run_dir / "validation_predictions.csv"
    slice_path = run_dir / "validation_error_slices.csv"
    metrics_path = run_dir / "metrics.json"
    submission_path = submissions_dir / f"submission_{args.ensemble_run_id}.csv"

    fused_val.to_csv(val_path, index=False)
    slice_df.to_csv(slice_path, index=False)
    fused_submission.to_csv(submission_path, index=False)

    horizon_summary = (
        slice_df[slice_df["slice_type"] == "horizon"][["slice_key", "mape"]]
        .set_index("slice_key")["mape"]
        .to_dict()
    )
    series_summary = (
        slice_df[slice_df["slice_type"] == "series"][["slice_key", "mape"]]
        .set_index("slice_key")["mape"]
        .to_dict()
    )

    member_meta = []
    for i, run_id in enumerate(selected_run_ids):
        score = selected_scores[i]
        member_meta.append(
            {
                "run_id": run_id,
                "weight": float(member_weights[i]),
                "holdout_mape": score.get("holdout"),
                "rolling_avg_mape": score.get("rolling"),
                "rank_value": float(selected_rank_values[i]),
            }
        )

    metrics = {
        "run_id": args.ensemble_run_id,
        "member_run_ids": selected_run_ids,
        "overall_mape": overall_mape,
        "validation_rows": int(len(fused_val)),
        "submission_rows": int(len(fused_submission)),
        "selection": {
            "requested_run_ids": run_ids,
            "selected_run_ids": selected_run_ids,
            "top_k": int(args.top_k),
            "rank_by": args.rank_by,
            "rolling_lambda": float(args.rolling_lambda),
            "weighting": args.weighting,
            "softmax_temp": float(args.softmax_temp),
        },
        "member_weights": member_meta,
        "horizon_mape": horizon_summary,
        "series_mape": series_summary,
        "artifacts": {
            "validation_predictions_csv": str(val_path.relative_to(PROJECT_ROOT)),
            "validation_error_slices_csv": str(slice_path.relative_to(PROJECT_ROOT)),
            "submission_csv": str(submission_path.relative_to(PROJECT_ROOT)),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"submission_path={submission_path}")


if __name__ == "__main__":
    main()
