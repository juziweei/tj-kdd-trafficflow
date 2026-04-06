#!/usr/bin/env python3
"""Fuse existing run outputs by selecting best run for configurable route slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.submission import validate_submission_schema  # noqa: E402

VALID_KEYS = ["tollgate_id", "direction", "time_window", "horizon"]
SUB_KEYS = ["tollgate_id", "direction", "time_window"]
ALLOWED_ROUTE_KEYS = {"series_key", "horizon", "dow", "hour", "minute", "clock", "is_peak"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice-select ensemble by configurable route keys")
    parser.add_argument("--run-ids", type=str, required=True, help="Comma-separated run IDs")
    parser.add_argument("--ensemble-run-id", type=str, required=True, help="Output run_id")
    parser.add_argument(
        "--route-keys",
        type=str,
        default="series_key,horizon",
        help="Comma-separated route keys from {series_key,horizon,dow,hour,minute,clock,is_peak}",
    )
    parser.add_argument(
        "--anchor-run",
        type=str,
        default=None,
        help="Fallback run for tiny gains; defaults to best overall among inputs",
    )
    parser.add_argument(
        "--min-gain-vs-anchor",
        type=float,
        default=0.0,
        help="Require slice MAPE gain >= this value to switch away from anchor run",
    )
    parser.add_argument(
        "--mape-eps",
        type=float,
        default=1.0,
        help="Denominator stabilizer used in MAPE",
    )
    parser.add_argument(
        "--strict-split-match",
        action="store_true",
        help="Fail if input runs have different split_timestamp",
    )
    return parser.parse_args()


def parse_run_ids(raw: str) -> list[str]:
    out = [token.strip() for token in raw.split(",") if token.strip()]
    if not out:
        raise ValueError("No run ids provided")
    return out


def parse_route_keys(raw: str) -> list[str]:
    out = [token.strip() for token in raw.split(",") if token.strip()]
    if not out:
        raise ValueError("No route keys provided")
    unknown = [k for k in out if k not in ALLOWED_ROUTE_KEYS]
    if unknown:
        raise ValueError(f"Unknown route keys: {unknown}; allowed={sorted(ALLOWED_ROUTE_KEYS)}")
    return out


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def load_metrics(run_id: str) -> dict:
    path = PROJECT_ROOT / "outputs" / "runs" / run_id / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics for run_id={run_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_validation(run_id: str, idx: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "runs" / run_id / "validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation predictions for run_id={run_id}: {path}")
    df = pd.read_csv(path)
    missing = [col for col in VALID_KEYS + ["actual", "prediction"] if col not in df.columns]
    if missing:
        raise ValueError(f"Run {run_id} missing validation columns: {missing}")
    out = df[VALID_KEYS + ["actual", "prediction"]].copy()
    if out.duplicated(VALID_KEYS).any():
        dup_cnt = int(out.duplicated(VALID_KEYS).sum())
        raise ValueError(f"Run {run_id} has duplicate validation keys: {dup_cnt}")
    out = out.rename(
        columns={
            "actual": f"actual_{idx}",
            "prediction": f"prediction_{idx}",
        }
    )
    return out


def load_submission(run_id: str, idx: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing submission for run_id={run_id}: {path}")
    df = pd.read_csv(path)
    missing = [col for col in SUB_KEYS + ["volume"] if col not in df.columns]
    if missing:
        raise ValueError(f"Run {run_id} missing submission columns: {missing}")
    out = df[SUB_KEYS + ["volume"]].copy()
    if out.duplicated(SUB_KEYS).any():
        dup_cnt = int(out.duplicated(SUB_KEYS).sum())
        raise ValueError(f"Run {run_id} has duplicate submission keys: {dup_cnt}")
    out = out.rename(columns={"volume": f"volume_{idx}"})
    return out


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
    for h, group in tmp.groupby("horizon"):
        records.append(
            {
                "slice_type": "horizon",
                "slice_key": f"h{int(h)}",
                "rows": int(len(group)),
                "mape": mape(group["actual"].to_numpy(), group["prediction"].to_numpy(), eps),
            }
        )
    for (key, h), group in tmp.groupby(["series_key", "horizon"]):
        records.append(
            {
                "slice_type": "series_horizon",
                "slice_key": f"{key}_h{int(h)}",
                "rows": int(len(group)),
                "mape": mape(group["actual"].to_numpy(), group["prediction"].to_numpy(), eps),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["slice_type", "slice_key"]).reset_index(drop=True)


def parse_left_timestamp(raw_series: pd.Series) -> pd.Series:
    raw = raw_series.astype(str)
    left = raw.str.extract(r"\[([^,]+),", expand=False)
    candidate = left.where(left.notna(), raw)
    ts = pd.to_datetime(candidate, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return ts


def enrich_route_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["series_key"] = out["tollgate_id"].astype(str) + "_" + out["direction"].astype(str)
    if "horizon" not in out.columns:
        ts = parse_left_timestamp(out["time_window"])
        if ts.isnull().any():
            bad_cnt = int(ts.isnull().sum())
            raise ValueError(f"Unable to parse time_window for horizon inference: bad_rows={bad_cnt}")
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute
        out["horizon"] = ((minute_of_day - 8 * 60) // 20 + 1).astype(int)
    else:
        out["horizon"] = out["horizon"].astype(int)
    ts = parse_left_timestamp(out["time_window"])
    if ts.isnull().any():
        bad_cnt = int(ts.isnull().sum())
        raise ValueError(f"Unable to parse route timestamp features: bad_rows={bad_cnt}")
    out["dow"] = ts.dt.dayofweek.astype(int)
    out["hour"] = ts.dt.hour.astype(int)
    out["minute"] = ts.dt.minute.astype(int)
    out["clock"] = ts.dt.strftime("%H:%M:%S")
    out["is_peak"] = out["hour"].isin([8, 17]).astype(int)
    return out


def normalize_key_value(val: object) -> str:
    if pd.isna(val):
        return "__nan__"
    return str(val)


def make_group_key(row: pd.Series, route_keys: list[str]) -> tuple[str, ...]:
    return tuple(normalize_key_value(row[k]) for k in route_keys)


def main() -> None:
    args = parse_args()
    run_ids = parse_run_ids(args.run_ids)
    route_keys = parse_route_keys(args.route_keys)
    run_to_idx = {rid: i for i, rid in enumerate(run_ids)}

    metrics_by_run = {rid: load_metrics(rid) for rid in run_ids}
    if args.strict_split_match:
        split_vals = {rid: metrics_by_run[rid].get("split_timestamp") for rid in run_ids}
        uniq = {v for v in split_vals.values()}
        if len(uniq) != 1:
            raise ValueError(f"Split mismatch across runs: {split_vals}")

    if args.anchor_run is not None:
        if args.anchor_run not in run_to_idx:
            raise ValueError(f"anchor_run {args.anchor_run} not in run_ids")
        anchor_run = args.anchor_run
    else:
        candidates: list[tuple[float, str]] = []
        for rid in run_ids:
            overall = metrics_by_run[rid].get("metrics", {}).get("overall_mape")
            if isinstance(overall, (int, float)):
                candidates.append((float(overall), rid))
        if not candidates:
            raise ValueError("Unable to infer anchor_run: no overall_mape found")
        candidates.sort(key=lambda x: x[0])
        anchor_run = candidates[0][1]
    anchor_idx = run_to_idx[anchor_run]

    merged_val: pd.DataFrame | None = None
    for rid, idx in run_to_idx.items():
        cur = load_validation(rid, idx)
        if merged_val is None:
            merged_val = cur
        else:
            merged_val = merged_val.merge(cur, on=VALID_KEYS, how="inner")
    if merged_val is None or len(merged_val) == 0:
        raise RuntimeError("Validation merge is empty")

    actual_cols = [f"actual_{run_to_idx[rid]}" for rid in run_ids]
    pred_cols = [f"prediction_{run_to_idx[rid]}" for rid in run_ids]
    base_actual = merged_val[actual_cols[0]].to_numpy()
    for col in actual_cols[1:]:
        cur_actual = merged_val[col].to_numpy()
        if not np.allclose(base_actual, cur_actual, equal_nan=True):
            max_diff = float(np.nanmax(np.abs(base_actual - cur_actual)))
            raise ValueError(f"Actual mismatch across runs: {col}, max_diff={max_diff:.6f}")
    merged_val["actual"] = base_actual
    merged_val = enrich_route_features(merged_val)

    selected_map: dict[tuple[str, ...], str] = {}
    selected_map_series_h: dict[tuple[str, int], str] = {}
    selection_rows: list[dict[str, object]] = []
    min_gain = float(args.min_gain_vs_anchor)

    for group_values, group in merged_val.groupby(route_keys, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        key_tuple = tuple(normalize_key_value(v) for v in group_values)
        y = group["actual"].to_numpy(dtype=float)
        anchor_pred = group[f"prediction_{anchor_idx}"].to_numpy(dtype=float)
        anchor_mape = mape(y, anchor_pred, eps=float(args.mape_eps))
        best_run = anchor_run
        best_mape = anchor_mape
        for rid in run_ids:
            idx = run_to_idx[rid]
            pred = group[f"prediction_{idx}"].to_numpy(dtype=float)
            cur_mape = mape(y, pred, eps=float(args.mape_eps))
            if cur_mape < best_mape:
                best_mape = cur_mape
                best_run = rid
        gain = anchor_mape - best_mape
        if best_run != anchor_run and gain < min_gain:
            best_run = anchor_run
            best_mape = anchor_mape
            gain = 0.0
        selected_map[key_tuple] = best_run
        if "series_key" in group.columns and "horizon" in group.columns:
            s = str(group["series_key"].iloc[0])
            h = int(group["horizon"].iloc[0])
            prev = selected_map_series_h.get((s, h))
            if prev is None:
                selected_map_series_h[(s, h)] = best_run
        row_obj: dict[str, object] = {
            "selected_run": best_run,
            "slice_rows": int(len(group)),
            "selected_mape": float(best_mape),
            "anchor_run": anchor_run,
            "anchor_mape": float(anchor_mape),
            "gain_vs_anchor": float(gain),
        }
        for k, v in zip(route_keys, key_tuple):
            row_obj[k] = v
        selection_rows.append(row_obj)

    chosen_pred = np.zeros(len(merged_val), dtype=float)
    chosen_run_col: list[str] = []
    for i, row in merged_val.iterrows():
        key_tuple = make_group_key(row, route_keys)
        rid = selected_map.get(key_tuple)
        if rid is None:
            s = str(row["series_key"])
            h = int(row["horizon"])
            rid = selected_map_series_h.get((s, h), anchor_run)
        idx = run_to_idx[rid]
        chosen_pred[i] = float(row[f"prediction_{idx}"])
        chosen_run_col.append(rid)
    merged_val["prediction"] = chosen_pred
    merged_val["selected_run"] = chosen_run_col

    overall = mape(merged_val["actual"].to_numpy(dtype=float), merged_val["prediction"].to_numpy(dtype=float), eps=float(args.mape_eps))
    val_out = merged_val[VALID_KEYS + ["actual", "prediction", "selected_run"]].copy()
    slice_df = build_error_slices(val_out, eps=float(args.mape_eps))
    selection_sort_cols = [k for k in route_keys if k in ALLOWED_ROUTE_KEYS]
    selection_df = pd.DataFrame.from_records(selection_rows)
    if len(selection_df) > 0:
        selection_df = selection_df.sort_values(selection_sort_cols).reset_index(drop=True)

    merged_sub: pd.DataFrame | None = None
    for rid, idx in run_to_idx.items():
        cur = load_submission(rid, idx)
        if merged_sub is None:
            merged_sub = cur
        else:
            merged_sub = merged_sub.merge(cur, on=SUB_KEYS, how="inner")
    if merged_sub is None or len(merged_sub) == 0:
        raise RuntimeError("Submission merge is empty")
    volume_cols = [f"volume_{run_to_idx[rid]}" for rid in run_ids]
    if merged_sub[volume_cols].isnull().any().any():
        raise RuntimeError("Submission merge contains nulls")

    merged_sub = enrich_route_features(merged_sub)
    clock = merged_sub["clock"]
    clock_to_horizon = {
        "08:00:00": 1,
        "08:20:00": 2,
        "08:40:00": 3,
        "09:00:00": 4,
        "09:20:00": 5,
        "09:40:00": 6,
        "17:00:00": 1,
        "17:20:00": 2,
        "17:40:00": 3,
        "18:00:00": 4,
        "18:20:00": 5,
        "18:40:00": 6,
    }
    if (~clock.isin(list(clock_to_horizon.keys()))).any():
        bad = sorted(set(clock[~clock.isin(list(clock_to_horizon.keys()))].tolist()))
        raise ValueError(f"Unknown target window clocks in submission: {bad}")
    merged_sub["horizon"] = clock.map(clock_to_horizon).astype(int)

    fused_sub = merged_sub[SUB_KEYS].copy()
    out_volume = np.zeros(len(merged_sub), dtype=float)
    for i, row in merged_sub.iterrows():
        key_tuple = make_group_key(row, route_keys)
        rid = selected_map.get(key_tuple)
        if rid is None:
            rid = selected_map_series_h.get((str(row["series_key"]), int(row["horizon"])), anchor_run)
        idx = run_to_idx[rid]
        out_volume[i] = float(row[f"volume_{idx}"])
    fused_sub["volume"] = np.clip(out_volume, 0.0, None)
    validate_submission_schema(fused_sub)

    out_run_dir = PROJECT_ROOT / "outputs" / "runs" / args.ensemble_run_id
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_sub_path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{args.ensemble_run_id}.csv"
    out_sub_path.parent.mkdir(parents=True, exist_ok=True)

    val_out.to_csv(out_run_dir / "validation_predictions.csv", index=False)
    slice_df.to_csv(out_run_dir / "validation_error_slices.csv", index=False)
    selection_df.to_csv(out_run_dir / "selection_map.csv", index=False)
    fused_sub.to_csv(out_sub_path, index=False)

    split_values = sorted({metrics_by_run[rid].get("split_timestamp") for rid in run_ids})
    split_timestamp = split_values[0] if len(split_values) == 1 else None

    metrics_obj = {
        "run_id": args.ensemble_run_id,
        "method": "route_slice_select",
        "route_keys": route_keys,
        "anchor_run": anchor_run,
        "min_gain_vs_anchor": float(min_gain),
        "split_timestamp": split_timestamp,
        "source_runs": run_ids,
        "metrics": {
            "overall_mape": float(overall),
        },
        "artifacts": {
            "validation_predictions_csv": str(out_run_dir / "validation_predictions.csv"),
            "validation_error_slices_csv": str(out_run_dir / "validation_error_slices.csv"),
            "selection_map_csv": str(out_run_dir / "selection_map.csv"),
            "submission_csv": str(out_sub_path),
        },
    }
    (out_run_dir / "metrics.json").write_text(json.dumps(metrics_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics_obj, ensure_ascii=False, indent=2))
    print(f"submission_path={out_sub_path}")


if __name__ == "__main__":
    main()
