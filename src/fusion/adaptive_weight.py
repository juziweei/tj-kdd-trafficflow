"""Time-safe adaptive fusion weights for multi-trunk forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AdaptiveFusionWeights:
    """Hierarchical fusion weights with global -> series -> series+horizon fallback."""

    global_gbdt_weight: float
    series_gbdt_weight: dict[tuple[int, int], float]
    slice_gbdt_weight: dict[tuple[int, int, int], float]

    def resolve(self, key: tuple[int, int], horizon: int) -> tuple[float, float]:
        gbdt_weight = self.slice_gbdt_weight.get(
            (key[0], key[1], horizon),
            self.series_gbdt_weight.get((key[0], key[1]), self.global_gbdt_weight),
        )
        gbdt_weight = float(np.clip(gbdt_weight, 0.0, 1.0))
        return 1.0 - gbdt_weight, gbdt_weight


def _mape(y_true: pd.Series, y_pred: pd.Series, eps: float) -> float:
    denom = y_true.abs().clip(lower=eps)
    return float(((y_true - y_pred).abs() / denom).mean())


def _to_weight(
    err_linear: float,
    err_gbdt: float,
    power: float,
    min_weight: float,
    max_weight: float,
    default_weight: float,
) -> float:
    if not np.isfinite(err_linear) or not np.isfinite(err_gbdt):
        return default_weight
    score_linear = 1.0 / max(err_linear, 1e-9) ** power
    score_gbdt = 1.0 / max(err_gbdt, 1e-9) ** power
    denom = score_linear + score_gbdt
    if denom <= 0.0 or not np.isfinite(denom):
        return default_weight
    raw = score_gbdt / denom
    return float(np.clip(raw, min_weight, max_weight))


def _blend(parent_weight: float, child_weight: float, sample_count: int, shrink: float) -> float:
    ratio = sample_count / (sample_count + max(shrink, 1e-9))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float((1.0 - ratio) * parent_weight + ratio * child_weight)


def fit_adaptive_fusion_weights(
    eval_df: pd.DataFrame,
    cfg: dict,
) -> tuple[AdaptiveFusionWeights, dict[str, object]]:
    """Fit hierarchical fusion weights from a time-ordered adaptation window."""

    if eval_df.empty:
        default_weight = float(cfg.get("default_gbdt_weight", 0.35))
        bundle = AdaptiveFusionWeights(
            global_gbdt_weight=default_weight,
            series_gbdt_weight={},
            slice_gbdt_weight={},
        )
        return bundle, {"reason": "empty_adaptation_frame"}

    eps = float(cfg.get("mape_eps", 1.0))
    power = float(cfg.get("error_power", 1.0))
    min_weight = float(cfg.get("min_model_weight", 0.05))
    max_weight = float(cfg.get("max_model_weight", 0.95))
    default_weight = float(cfg.get("default_gbdt_weight", 0.35))

    min_series_samples = int(cfg.get("min_series_samples", 10))
    min_slice_samples = int(cfg.get("min_slice_samples", 6))
    series_shrink = float(cfg.get("series_shrink", 24.0))
    slice_shrink = float(cfg.get("slice_shrink", 12.0))

    frame = eval_df.copy()
    required = {
        "actual",
        "linear_prediction",
        "gbdt_prediction",
        "tollgate_id",
        "direction",
        "horizon",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing adaptation columns for fusion: {missing}")

    global_err_linear = _mape(frame["actual"], frame["linear_prediction"], eps=eps)
    global_err_gbdt = _mape(frame["actual"], frame["gbdt_prediction"], eps=eps)
    global_weight = _to_weight(
        err_linear=global_err_linear,
        err_gbdt=global_err_gbdt,
        power=power,
        min_weight=min_weight,
        max_weight=max_weight,
        default_weight=default_weight,
    )

    series_map: dict[tuple[int, int], float] = {}
    series_stats: dict[str, dict[str, float | int]] = {}
    grouped_series = frame.groupby(["tollgate_id", "direction"], sort=True)
    for (tollgate_id, direction), part in grouped_series:
        n = int(len(part))
        key = (int(tollgate_id), int(direction))
        if n < min_series_samples:
            continue
        err_linear = _mape(part["actual"], part["linear_prediction"], eps=eps)
        err_gbdt = _mape(part["actual"], part["gbdt_prediction"], eps=eps)
        child_weight = _to_weight(
            err_linear=err_linear,
            err_gbdt=err_gbdt,
            power=power,
            min_weight=min_weight,
            max_weight=max_weight,
            default_weight=global_weight,
        )
        final_weight = _blend(global_weight, child_weight, sample_count=n, shrink=series_shrink)
        series_map[key] = final_weight
        series_stats[f"{key[0]}_{key[1]}"] = {
            "samples": n,
            "linear_mape": err_linear * 100.0,
            "gbdt_mape": err_gbdt * 100.0,
            "gbdt_weight": final_weight,
        }

    slice_map: dict[tuple[int, int, int], float] = {}
    slice_stats: dict[str, dict[str, float | int]] = {}
    grouped_slice = frame.groupby(["tollgate_id", "direction", "horizon"], sort=True)
    for (tollgate_id, direction, horizon), part in grouped_slice:
        n = int(len(part))
        key = (int(tollgate_id), int(direction), int(horizon))
        if n < min_slice_samples:
            continue
        err_linear = _mape(part["actual"], part["linear_prediction"], eps=eps)
        err_gbdt = _mape(part["actual"], part["gbdt_prediction"], eps=eps)
        parent = series_map.get((key[0], key[1]), global_weight)
        child_weight = _to_weight(
            err_linear=err_linear,
            err_gbdt=err_gbdt,
            power=power,
            min_weight=min_weight,
            max_weight=max_weight,
            default_weight=parent,
        )
        final_weight = _blend(parent, child_weight, sample_count=n, shrink=slice_shrink)
        slice_map[key] = final_weight
        slice_stats[f"{key[0]}_{key[1]}_h{key[2]}"] = {
            "samples": n,
            "linear_mape": err_linear * 100.0,
            "gbdt_mape": err_gbdt * 100.0,
            "gbdt_weight": final_weight,
        }

    bundle = AdaptiveFusionWeights(
        global_gbdt_weight=global_weight,
        series_gbdt_weight=series_map,
        slice_gbdt_weight=slice_map,
    )
    stats = {
        "adapt_rows": int(len(frame)),
        "global_linear_mape": global_err_linear * 100.0,
        "global_gbdt_mape": global_err_gbdt * 100.0,
        "global_gbdt_weight": global_weight,
        "series_weights": series_stats,
        "slice_weights": slice_stats,
    }
    return bundle, stats
