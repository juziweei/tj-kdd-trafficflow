#!/usr/bin/env python3
"""Run stronger backbone v6: dual GBDT experts + tri-fusion + post head."""

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

SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_baseline as baseline_mod  # noqa: E402

from src.data.volume_io import (  # noqa: E402
    aggregate_to_20min,
    build_series_history,
    complete_20min_grid,
    load_volume_events,
    merge_histories,
)
from src.data.weather_io import (  # noqa: E402
    WEATHER_FEATURE_COLUMNS,
    get_weather_feature_vector,
    load_weather_table,
    merge_weather_tables,
    weather_defaults,
)
from src.eval.metrics import build_error_slice_table, summarize_metrics  # noqa: E402
from src.features.volume_features import (  # noqa: E402
    FeatureConfig,
    build_feature_row,
    calendar_feature_vector,
    feature_columns,
    horizon_index,
    is_target_window,
    target_windows_for_days,
)
from src.inference.submission import build_submission, validate_submission_schema  # noqa: E402
from src.models.tft_model import (  # noqa: E402
    TORCH_AVAILABLE,
    TFTConfig,
    TemporalFusionTransformer,
)


@dataclass
class BaselineBranchBundle:
    primary_bundle: baseline_mod.ForecastBundle
    residual_models: dict[tuple[int, int], baseline_mod.RidgeLinearModel]
    residual_clip_abs: float | None
    horizon_bias_map: dict[tuple[int, int, int], float]
    horizon_bias_clip_abs: float | None
    conditional_residual_models: dict[tuple[int, int, int], baseline_mod.RidgeLinearModel]
    conditional_residual_clip_map: dict[tuple[int, int, int], float]
    conditional_residual_gate_meta: dict[tuple[int, int, int], dict[str, object]]
    stats: dict[str, object]


@dataclass
class GBDTBundle:
    feature_names: list[str]
    use_log_target: bool
    log_pred_clip: float
    global_model: XGBRegressor
    anchor_models: dict[int, XGBRegressor]


@dataclass
class AdaptiveTriFusionWeights:
    global_weights: tuple[float, float, float]
    series_weights: dict[tuple[int, int], tuple[float, float, float]]
    slice_weights: dict[tuple[int, int, int], tuple[float, float, float]]

    def resolve(self, key: tuple[int, int], horizon: int) -> tuple[float, float, float]:
        return self.slice_weights.get(
            (key[0], key[1], horizon),
            self.series_weights.get((key[0], key[1]), self.global_weights),
        )


@dataclass
class RegimeRouterBundle:
    enabled: bool
    up_threshold: float
    down_threshold: float
    conflict_threshold: float
    conflict_threshold_h6: float | None
    blend_strength: float
    blend_strength_h6: float
    weights_by_regime: dict[str, tuple[float, float, float]]
    h6_weights_by_regime: dict[str, tuple[float, float, float]]
    stats: dict[str, object]

    def classify(
        self,
        horizon: int,
        linear_prediction: float,
        gbdt_full_prediction: float,
        gbdt_target_prediction: float,
    ) -> str:
        denom = max(abs(float(linear_prediction)), 1.0)
        delta_full = (float(gbdt_full_prediction) - float(linear_prediction)) / denom
        delta_target = (float(gbdt_target_prediction) - float(linear_prediction)) / denom
        divergence = abs(float(gbdt_full_prediction) - float(gbdt_target_prediction)) / denom

        conflict_thr = self.conflict_threshold
        if int(horizon) == 6 and self.conflict_threshold_h6 is not None:
            conflict_thr = float(self.conflict_threshold_h6)
        if divergence >= float(conflict_thr):
            return "conflict"

        mean_delta = 0.5 * (delta_full + delta_target)
        if mean_delta >= float(self.up_threshold):
            return "up"
        if mean_delta <= -float(self.down_threshold):
            return "down"
        return "stable"

    def resolve(
        self,
        base_weights: tuple[float, float, float],
        horizon: int,
        linear_prediction: float,
        gbdt_full_prediction: float,
        gbdt_target_prediction: float,
    ) -> tuple[float, float, float]:
        if not self.enabled:
            return base_weights

        regime = self.classify(
            horizon=horizon,
            linear_prediction=linear_prediction,
            gbdt_full_prediction=gbdt_full_prediction,
            gbdt_target_prediction=gbdt_target_prediction,
        )

        child = None
        blend = float(self.blend_strength)
        if int(horizon) == 6 and regime in self.h6_weights_by_regime:
            child = self.h6_weights_by_regime[regime]
            blend = float(self.blend_strength_h6)
        elif regime in self.weights_by_regime:
            child = self.weights_by_regime[regime]

        if child is None:
            return base_weights

        base = np.array(base_weights, dtype=float)
        routed = np.array(child, dtype=float)
        mixed = (1.0 - blend) * base + blend * routed
        return _normalize_triplet(mixed, fallback=base_weights)


@dataclass
class MemoryBucket:
    features: np.ndarray
    residuals: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    distance_gate: float


@dataclass
class MemoryRetrievalBundle:
    enabled: bool
    target_series: set[tuple[int, int]] | None
    apply_horizons: set[int]
    use_anchor: bool
    top_k: int
    distance_power: float
    blend_weight: float
    max_abs_delta: float
    distance_gate_scale: float
    primary_buckets: dict[tuple[int, int, int, int], MemoryBucket]
    series_buckets: dict[tuple[int, int, int], MemoryBucket]
    global_buckets: dict[tuple[int, int], MemoryBucket]
    stats: dict[str, object]

    def _is_enabled_for(self, key: tuple[int, int], horizon: int) -> bool:
        if not self.enabled:
            return False
        if self.target_series is not None and key not in self.target_series:
            return False
        return int(horizon) in self.apply_horizons

    def _query_bucket(self, bucket: MemoryBucket, query: np.ndarray) -> tuple[float, int, float]:
        if bucket.features.size == 0 or bucket.residuals.size == 0:
            return 0.0, 0, float("nan")

        center_dist = float(np.sqrt(np.mean(((query - bucket.center) / bucket.scale) ** 2)))
        if np.isfinite(bucket.distance_gate):
            if center_dist > float(bucket.distance_gate) * float(self.distance_gate_scale):
                return 0.0, 0, center_dist

        dists = np.sqrt(np.mean(((bucket.features - query[None, :]) / bucket.scale[None, :]) ** 2, axis=1))
        if len(dists) == 0:
            return 0.0, 0, center_dist
        k = max(1, min(int(self.top_k), int(len(dists))))
        top_idx = np.argpartition(dists, kth=k - 1)[:k]
        top_dist = dists[top_idx]
        top_res = bucket.residuals[top_idx]
        power = max(float(self.distance_power), 1e-6)
        weights = 1.0 / np.maximum(top_dist, 1e-6) ** power
        w_sum = float(np.sum(weights))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            return 0.0, 0, float(np.min(top_dist)) if len(top_dist) > 0 else center_dist
        delta = float(np.sum(top_res * weights) / w_sum)
        return delta, int(k), float(np.min(top_dist))

    def retrieve_delta(
        self,
        key: tuple[int, int],
        horizon: int,
        anchor: int,
        linear_prediction: float,
        gbdt_full_prediction: float,
        gbdt_target_prediction: float,
        base_prediction: float,
    ) -> tuple[float, int, int, int, float]:
        if not self._is_enabled_for(key=key, horizon=horizon):
            return 0.0, 0, 0, 0, float("nan")

        query = _build_memory_feature_vector(
            linear_prediction=linear_prediction,
            gbdt_full_prediction=gbdt_full_prediction,
            gbdt_target_prediction=gbdt_target_prediction,
            fused_prediction=base_prediction,
        )
        anchor_key = int(anchor) if self.use_anchor else -1

        primary = self.primary_buckets.get((int(key[0]), int(key[1]), int(horizon), anchor_key))
        if primary is not None:
            delta_raw, n_neighbors, min_dist = self._query_bucket(primary, query)
            if n_neighbors > 0:
                delta = float(delta_raw) * float(self.blend_weight)
                delta = float(np.clip(delta, -float(self.max_abs_delta), float(self.max_abs_delta)))
                return delta, 1, 1, n_neighbors, min_dist

        series_bucket = self.series_buckets.get((int(key[0]), int(key[1]), anchor_key))
        if series_bucket is not None:
            delta_raw, n_neighbors, min_dist = self._query_bucket(series_bucket, query)
            if n_neighbors > 0:
                delta = float(delta_raw) * float(self.blend_weight)
                delta = float(np.clip(delta, -float(self.max_abs_delta), float(self.max_abs_delta)))
                return delta, 1, 2, n_neighbors, min_dist

        global_bucket = self.global_buckets.get((int(horizon), anchor_key))
        if global_bucket is not None:
            delta_raw, n_neighbors, min_dist = self._query_bucket(global_bucket, query)
            if n_neighbors > 0:
                delta = float(delta_raw) * float(self.blend_weight)
                delta = float(np.clip(delta, -float(self.max_abs_delta), float(self.max_abs_delta)))
                return delta, 1, 3, n_neighbors, min_dist

        return 0.0, 0, 0, 0, float("nan")


@dataclass
class RiskConstraintBundle:
    enabled: bool
    target_series: set[tuple[int, int]] | None
    apply_horizons: set[int]
    score_weights: tuple[float, float, float]
    threshold: float
    threshold_h6: float | None
    shrink: float
    shrink_h6: float
    safe_branch_global: int
    safe_branch_by_slice: dict[tuple[int, int, int], int]
    stats: dict[str, object]

    def _is_enabled_for(self, key: tuple[int, int], horizon: int) -> bool:
        if not self.enabled:
            return False
        if self.target_series is not None and key not in self.target_series:
            return False
        return int(horizon) in self.apply_horizons

    def _resolve_safe_branch(self, key: tuple[int, int], horizon: int) -> int:
        return int(self.safe_branch_by_slice.get((int(key[0]), int(key[1]), int(horizon)), self.safe_branch_global))

    def apply(
        self,
        key: tuple[int, int],
        horizon: int,
        fused_prediction: float,
        linear_prediction: float,
        gbdt_full_prediction: float,
        gbdt_target_prediction: float,
    ) -> tuple[float, int, float, float, float, int]:
        score = _compute_risk_score(
            linear_prediction=linear_prediction,
            gbdt_full_prediction=gbdt_full_prediction,
            gbdt_target_prediction=gbdt_target_prediction,
            score_weights=self.score_weights,
        )
        if not self._is_enabled_for(key=key, horizon=horizon):
            return float(fused_prediction), 0, score, float("nan"), 0.0, -1

        threshold = float(self.threshold)
        shrink = float(self.shrink)
        if int(horizon) == 6 and self.threshold_h6 is not None:
            threshold = float(self.threshold_h6)
            shrink = float(self.shrink_h6)
        if score < threshold:
            return float(fused_prediction), 0, score, threshold, shrink, -1

        safe_branch = self._resolve_safe_branch(key=key, horizon=horizon)
        candidates = [float(linear_prediction), float(gbdt_full_prediction), float(gbdt_target_prediction)]
        safe_pred = candidates[int(np.clip(safe_branch, 0, 2))]
        out = max(0.0, (1.0 - float(shrink)) * float(fused_prediction) + float(shrink) * float(safe_pred))
        return out, 1, score, threshold, shrink, int(safe_branch)


@dataclass
class FusionBundle:
    global_weights: AdaptiveTriFusionWeights
    anchor_weights: dict[int, AdaptiveTriFusionWeights]
    regime_router: RegimeRouterBundle | None = None
    memory_retrieval: MemoryRetrievalBundle | None = None
    risk_constraint: RiskConstraintBundle | None = None

    def resolve_with_anchor(
        self,
        key: tuple[int, int],
        horizon: int,
        anchor: int,
        linear_prediction: float | None = None,
        gbdt_full_prediction: float | None = None,
        gbdt_target_prediction: float | None = None,
    ) -> tuple[float, float, float]:
        anchor_idx = int(anchor)
        weight_obj = self.anchor_weights.get(anchor_idx, self.global_weights)
        base_weights = weight_obj.resolve(key, horizon)
        if (
            self.regime_router is None
            or not self.regime_router.enabled
            or linear_prediction is None
            or gbdt_full_prediction is None
            or gbdt_target_prediction is None
        ):
            return base_weights

        return self.regime_router.resolve(
            base_weights=base_weights,
            horizon=horizon,
            linear_prediction=float(linear_prediction),
            gbdt_full_prediction=float(gbdt_full_prediction),
            gbdt_target_prediction=float(gbdt_target_prediction),
        )

    def resolve(
        self,
        key: tuple[int, int],
        horizon: int,
        ts: pd.Timestamp,
        linear_prediction: float | None = None,
        gbdt_full_prediction: float | None = None,
        gbdt_target_prediction: float | None = None,
    ) -> tuple[float, float, float]:
        return self.resolve_with_anchor(
            key=key,
            horizon=horizon,
            anchor=anchor_bucket(ts),
            linear_prediction=linear_prediction,
            gbdt_full_prediction=gbdt_full_prediction,
            gbdt_target_prediction=gbdt_target_prediction,
        )


@dataclass
class PostFusionResidualBundle:
    enabled: bool
    feature_names: list[str]
    models: dict[tuple[int, int], baseline_mod.RidgeLinearModel]
    clip_map: dict[tuple[int, int], float]
    horizon_allowlist: dict[tuple[int, int], set[int]]
    gate_meta: dict[tuple[int, int], dict[str, object]]
    stats: dict[str, object]


@dataclass
class TFTBranchBundle:
    enabled: bool
    feature_names: list[str]
    model: TemporalFusionTransformer | None
    target_series: set[tuple[int, int]] | None
    apply_horizons: set[int]
    blend_weight: float
    horizon_blend_weight: dict[int, float]
    use_weather_interactions: bool
    use_event_features: bool
    stats: dict[str, object]


@dataclass
class SeriesExpertBundle:
    enabled: bool
    series_key: tuple[int, int] | None
    model_bundle: GBDTBundle | None
    horizon_model_bundles: dict[int, GBDTBundle]
    rl_policy_map: dict[tuple[int, int, int], float]
    rl_delta_bins: list[float]
    rl_use_anchor: bool
    gate_delta_abs_threshold: dict[int, float]
    apply_horizons: set[int]
    blend_weight: float
    horizon_blend_weight: dict[int, float]
    stats: dict[str, object]


WEATHER_INTERACTION_FEATURE_COLUMNS = [
    "wx_temperature_sq",
    "wx_wind_speed_sq",
    "wx_temp_x_humidity",
    "wx_wind_x_precip",
    "wx_pressure_gap",
    "wx_precip_sqrt",
]

EVENT_FEATURE_COLUMNS = [
    "evt_is_holiday",
    "evt_is_preholiday",
    "evt_is_postholiday",
    "evt_is_weekend",
    "evt_is_morning_peak",
    "evt_is_evening_peak",
    "evt_is_target_window",
    "evt_is_national_day_period",
    "evt_is_mid_autumn_period",
]


def anchor_bucket(ts: pd.Timestamp) -> int:
    hour = pd.Timestamp(ts).hour
    return 0 if hour < 12 else 1


def _disabled_regime_router_bundle(reason: str) -> RegimeRouterBundle:
    return RegimeRouterBundle(
        enabled=False,
        up_threshold=0.06,
        down_threshold=0.06,
        conflict_threshold=0.12,
        conflict_threshold_h6=0.08,
        blend_strength=0.0,
        blend_strength_h6=0.0,
        weights_by_regime={},
        h6_weights_by_regime={},
        stats={"enabled": 0, "reason": reason},
    )


def _disabled_memory_retrieval_bundle(reason: str) -> MemoryRetrievalBundle:
    return MemoryRetrievalBundle(
        enabled=False,
        target_series=None,
        apply_horizons={1, 2, 3, 4, 5, 6},
        use_anchor=False,
        top_k=1,
        distance_power=1.0,
        blend_weight=0.0,
        max_abs_delta=0.0,
        distance_gate_scale=1.0,
        primary_buckets={},
        series_buckets={},
        global_buckets={},
        stats={"enabled": 0, "reason": reason},
    )


def _disabled_risk_constraint_bundle(reason: str) -> RiskConstraintBundle:
    return RiskConstraintBundle(
        enabled=False,
        target_series=None,
        apply_horizons={1, 2, 3, 4, 5, 6},
        score_weights=(0.55, 0.25, 0.20),
        threshold=float("inf"),
        threshold_h6=None,
        shrink=0.0,
        shrink_h6=0.0,
        safe_branch_global=0,
        safe_branch_by_slice={},
        stats={"enabled": 0, "reason": reason},
    )


def _disabled_series_expert_bundle(reason: str, expert_tag: str = "series_expert") -> SeriesExpertBundle:
    return SeriesExpertBundle(
        enabled=False,
        series_key=None,
        model_bundle=None,
        horizon_model_bundles={},
        rl_policy_map={},
        rl_delta_bins=[],
        rl_use_anchor=False,
        gate_delta_abs_threshold={},
        apply_horizons=set(),
        blend_weight=0.0,
        horizon_blend_weight={},
        stats={"enabled": 0, "reason": reason, "expert_tag": expert_tag},
    )


def _normalize_expert_tag(text: object | None) -> str:
    if text is None:
        return "series_expert"
    raw = str(text).strip().lower()
    if not raw:
        return "series_expert"
    out_chars: list[str] = []
    for ch in raw:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_":
            out_chars.append(ch)
        elif ch in {"-", " ", "."}:
            out_chars.append("_")
    norm = "".join(out_chars).strip("_")
    return norm or "series_expert"


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out: dict = {}
    for k, v in base.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        elif isinstance(v, dict):
            out[k] = _deep_merge_dict({}, v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def resolve_series_expert_pool(cfg: dict) -> list[tuple[str, dict]]:
    base_cfg_raw = cfg.get("series_expert", {})
    if base_cfg_raw is None:
        base_cfg_raw = {}
    if not isinstance(base_cfg_raw, dict):
        raise ValueError("series_expert must be an object")
    base_cfg = dict(base_cfg_raw)

    pool_raw = cfg.get("series_expert_pool")
    if pool_raw is None:
        return [("series_expert", base_cfg)]
    if not isinstance(pool_raw, list):
        raise ValueError("series_expert_pool must be a list")
    if not pool_raw:
        return []

    out: list[tuple[str, dict]] = []
    seen_tags: set[str] = set()
    for idx, item in enumerate(pool_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"series_expert_pool[{idx - 1}] must be an object")
        merged = _deep_merge_dict(base_cfg, item)
        tag_source = merged.get("name") or merged.get("series_key") or f"series_expert_{idx}"
        tag = _normalize_expert_tag(tag_source)
        if tag in seen_tags:
            tag = _normalize_expert_tag(f"{tag}_{idx}")
        seen_tags.add(tag)
        out.append((tag, merged))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strong backbone v6")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "strong_backbone_v6_main.json",
        help="Path to config JSON",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--feature-source",
        type=str,
        choices=("pandas", "sql"),
        default="pandas",
        help="Training feature source: pandas pipeline or prebuilt SQL snapshot.",
    )
    parser.add_argument(
        "--sql-snapshot-csv",
        type=Path,
        default=None,
        help="Path to SQL feature snapshot CSV (required when --feature-source=sql unless configured).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_train_grid_from_sql_snapshot(path: Path) -> pd.DataFrame:
    required = {"tollgate_id", "direction", "time_window", "volume"}
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SQL snapshot missing required columns: {missing}")

    out = frame[["tollgate_id", "direction", "time_window", "volume"]].copy()
    out["time_window"] = pd.to_datetime(out["time_window"])
    out = out.dropna(subset=["time_window"])
    out["tollgate_id"] = out["tollgate_id"].astype(int)
    out["direction"] = out["direction"].astype(int)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0).astype(float)

    duplicate_mask = out.duplicated(subset=["tollgate_id", "direction", "time_window"])
    if bool(duplicate_mask.any()):
        dup_cnt = int(duplicate_mask.sum())
        raise ValueError(f"SQL snapshot has duplicate keys: {dup_cnt}")

    out = out.sort_values(["tollgate_id", "direction", "time_window"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"SQL snapshot is empty: {path}")
    return out


def split_timestamp(all_windows: pd.Series, validation_days: int) -> pd.Timestamp:
    last_day = all_windows.max().normalize()
    return last_day - pd.Timedelta(days=validation_days - 1)


def rolling_folds(
    days: list[pd.Timestamp],
    n_folds: int,
    val_days: int,
    min_train_days: int,
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
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
        raise ValueError(f"Unknown weather feature columns: {unknown}")
    return list(requested)


def parse_series_allowlist(raw: object | None) -> set[tuple[int, int]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("target_series must be a list like ['1_0', '2_0']")
    out: set[tuple[int, int]] = set()
    for item in raw:
        out.add(baseline_mod.parse_series_key(str(item)))
    return out if out else None


def add_external_features_to_row(
    feat: dict[str, float],
    ts: pd.Timestamp,
    use_weather_interactions: bool,
    use_event_features: bool,
) -> None:
    if use_weather_interactions:
        temp = float(feat.get("weather_temperature", 0.0))
        wind = float(feat.get("weather_wind_speed", 0.0))
        humid = float(feat.get("weather_rel_humidity", 0.0))
        precip = max(0.0, float(feat.get("weather_precipitation", 0.0)))
        press = float(feat.get("weather_pressure", 0.0))
        sea_press = float(feat.get("weather_sea_pressure", 0.0))
        feat["wx_temperature_sq"] = temp * temp
        feat["wx_wind_speed_sq"] = wind * wind
        feat["wx_temp_x_humidity"] = temp * humid / 100.0
        feat["wx_wind_x_precip"] = wind * precip
        feat["wx_pressure_gap"] = sea_press - press
        feat["wx_precip_sqrt"] = float(np.sqrt(precip))

    if use_event_features:
        cal = calendar_feature_vector(ts)
        day = pd.Timestamp(ts).date()
        feat["evt_is_holiday"] = float(cal.get("is_holiday", 0.0))
        feat["evt_is_preholiday"] = float(cal.get("is_preholiday", 0.0))
        feat["evt_is_postholiday"] = float(cal.get("is_postholiday", 0.0))
        feat["evt_is_weekend"] = float(cal.get("is_weekend", 0.0))
        feat["evt_is_morning_peak"] = 1.0 if ts.hour in (7, 8, 9, 10) else 0.0
        feat["evt_is_evening_peak"] = 1.0 if ts.hour in (16, 17, 18, 19) else 0.0
        feat["evt_is_target_window"] = 1.0 if is_target_window(ts) else 0.0
        feat["evt_is_national_day_period"] = 1.0 if pd.Timestamp("2016-10-01").date() <= day <= pd.Timestamp("2016-10-07").date() else 0.0
        feat["evt_is_mid_autumn_period"] = 1.0 if pd.Timestamp("2016-09-15").date() <= day <= pd.Timestamp("2016-09-17").date() else 0.0


def build_external_feature_columns(
    use_weather_interactions: bool,
    use_event_features: bool,
) -> list[str]:
    cols: list[str] = []
    if use_weather_interactions:
        cols.extend(WEATHER_INTERACTION_FEATURE_COLUMNS)
    if use_event_features:
        cols.extend(EVENT_FEATURE_COLUMNS)
    return cols


def build_tft_feature_names(
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    include_calendar: bool,
    weather_columns: list[str],
    use_weather_interactions: bool,
    use_event_features: bool,
) -> list[str]:
    cols = feature_columns(series_keys, feature_cfg, include_calendar=include_calendar) + weather_columns
    cols.extend(build_external_feature_columns(use_weather_interactions, use_event_features))
    return cols


def build_training_dataset(
    train_grid: pd.DataFrame,
    history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    cfg: FeatureConfig,
    train_end: pd.Timestamp,
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool = True,
    target_only: bool = True,
    weather_table: pd.DataFrame | None = None,
    weather_defaults_map: dict[str, float] | None = None,
    weather_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    meta_rows: list[dict[str, int | pd.Timestamp]] = []

    for row in train_grid.itertuples(index=False):
        ts = pd.Timestamp(row.time_window)
        if ts >= train_end:
            continue
        target_flag = is_target_window(ts)
        if target_only and not target_flag:
            continue

        weather_values: dict[str, float] = {}
        if weather_table is not None and weather_defaults_map is not None:
            weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
            if weather_columns is not None:
                weather_values = {k: weather_values[k] for k in weather_columns}

        key = (int(row.tollgate_id), int(row.direction))
        feat = build_feature_row(
            key=key,
            ts=ts,
            history=history,
            series_keys=series_keys,
            cfg=cfg,
            default_value=default_value,
            allow_fallback=False,
            weather=weather_values,
            use_enhanced_features=use_enhanced_features,
        )
        if feat is None:
            continue

        if include_calendar:
            feat.update(calendar_feature_vector(ts))

        if weather_values:
            feat.update(weather_values)

        rows.append(feat)
        targets.append(float(row.volume))
        meta_rows.append(
            {
                "tollgate_id": int(row.tollgate_id),
                "direction": int(row.direction),
                "time_window": ts,
                "horizon": int(horizon_index(ts)),
                "anchor": int(anchor_bucket(ts)),
                "day": ts.normalize(),
                "is_target": int(target_flag),
            }
        )

    x_df = pd.DataFrame(rows)
    y = pd.Series(targets, name="volume")
    meta_df = pd.DataFrame(meta_rows)
    return x_df, y, meta_df


def make_xgb(model_cfg: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 500)),
        max_depth=int(model_cfg.get("max_depth", 4)),
        learning_rate=float(model_cfg.get("learning_rate", 0.03)),
        subsample=float(model_cfg.get("subsample", 0.85)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.85)),
        reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(model_cfg.get("reg_lambda", 4.0)),
        min_child_weight=float(model_cfg.get("min_child_weight", 3.0)),
        gamma=float(model_cfg.get("gamma", 0.2)),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=int(model_cfg.get("seed", 42)),
        n_jobs=int(model_cfg.get("n_jobs", 6)),
        verbosity=0,
    )


def train_gbdt_bundle(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    feature_names: list[str],
    model_cfg: dict,
    training_cfg: dict | None = None,
) -> GBDTBundle:
    use_log_target = bool(model_cfg.get("use_log_target", True))
    log_pred_clip = float(model_cfg.get("log_pred_clip", 6.0))
    training_cfg = training_cfg or {}
    use_anchor_models = bool(training_cfg.get("use_anchor_models", True))
    min_anchor_samples = int(training_cfg.get("min_anchor_samples", 120))
    max_sample_weight = float(training_cfg.get("max_sample_weight", 100.0))

    y_raw = y_train.to_numpy(dtype=float)
    target = np.log1p(y_raw) if use_log_target else y_raw
    sample_weight = 1.0 / np.maximum(y_raw, 1.0)
    target_window_weight = float(training_cfg.get("target_window_weight", 1.0))
    off_target_weight = float(training_cfg.get("off_target_weight", 0.3))
    min_sample_weight = float(training_cfg.get("min_sample_weight", 0.05))
    target_mask = np.ones_like(sample_weight, dtype=bool)
    if "is_target" in meta_train.columns:
        target_mask = meta_train["is_target"].to_numpy(dtype=int) > 0
        time_weight = np.where(target_mask, target_window_weight, off_target_weight)
        sample_weight = sample_weight * time_weight

    # Tail-aware reweighting: emphasize extreme-volume windows while keeping leakage-safe chronology.
    tail_q_raw = training_cfg.get("tail_weight_quantile")
    tail_factor = float(training_cfg.get("tail_weight_factor", 1.0))
    tail_target_only = bool(training_cfg.get("tail_target_only", True))
    tail_anchor_factors_raw = training_cfg.get("tail_anchor_factors", {})
    if (
        tail_q_raw is not None
        and tail_factor > 1.0
        and np.isfinite(float(tail_q_raw))
        and len(y_raw) > 0
    ):
        q = float(np.clip(float(tail_q_raw), 0.5, 0.99))
        pool = y_raw[target_mask] if tail_target_only else y_raw
        if len(pool) > 0:
            threshold = float(np.quantile(pool, q))
            tail_mask = y_raw >= threshold
            if tail_target_only:
                tail_mask = tail_mask & target_mask

            tail_scale = np.ones_like(sample_weight)
            tail_scale[tail_mask] *= tail_factor

            if isinstance(tail_anchor_factors_raw, dict) and "anchor" in meta_train.columns:
                anchor_arr = meta_train["anchor"].to_numpy(dtype=int)
                for anchor_key, factor_val in tail_anchor_factors_raw.items():
                    try:
                        anchor = int(anchor_key)
                        factor = float(factor_val)
                    except (TypeError, ValueError):
                        continue
                    if factor <= 0.0:
                        continue
                    tail_scale[tail_mask & (anchor_arr == anchor)] *= factor

            sample_weight = sample_weight * tail_scale

    sample_weight = np.clip(sample_weight, a_min=min_sample_weight, a_max=None)
    sample_weight = np.clip(sample_weight, a_min=None, a_max=max_sample_weight)

    global_model = make_xgb(model_cfg)
    global_model.fit(x_train[feature_names], target, sample_weight=sample_weight)

    anchor_models: dict[int, XGBRegressor] = {}
    if use_anchor_models and "anchor" in meta_train.columns:
        for anchor in (0, 1):
            mask = meta_train["anchor"] == anchor
            n = int(mask.sum())
            if n < min_anchor_samples:
                continue
            x_sub = x_train.loc[mask, feature_names]
            y_sub = target[mask.to_numpy()]
            w_sub = sample_weight[mask.to_numpy()]
            model = make_xgb(model_cfg)
            model.fit(x_sub, y_sub, sample_weight=w_sub)
            anchor_models[int(anchor)] = model

    return GBDTBundle(
        feature_names=feature_names,
        use_log_target=use_log_target,
        log_pred_clip=log_pred_clip,
        global_model=global_model,
        anchor_models=anchor_models,
    )


def predict_gbdt(bundle: GBDTBundle, x_row: pd.DataFrame, ts: pd.Timestamp) -> float:
    anchor = anchor_bucket(ts)
    model = bundle.anchor_models.get(anchor, bundle.global_model)
    raw = float(model.predict(x_row[bundle.feature_names])[0])
    if bundle.use_log_target:
        raw = min(raw, bundle.log_pred_clip)
        return max(0.0, float(np.expm1(raw)))
    return max(0.0, raw)


def run_gbdt_recursive_forecast(
    bundle: GBDTBundle,
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, float | int | pd.Timestamp]] = []

    for ts in schedule:
        for key in series_keys:
            weather_values: dict[str, float] = {}
            if weather_table is not None and weather_defaults_map is not None:
                weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
                weather_values = {k: weather_values[k] for k in weather_columns}

            feat = build_feature_row(
                key=key,
                ts=ts,
                history=history,
                series_keys=series_keys,
                cfg=feature_cfg,
                default_value=default_value,
                allow_fallback=True,
                weather=weather_values,
                use_enhanced_features=use_enhanced_features,
            )
            if feat is None:
                continue

            if include_calendar:
                feat.update(calendar_feature_vector(ts))

            if weather_values:
                feat.update(weather_values)

            horizon = int(horizon_index(ts))
            x = pd.DataFrame([feat], columns=bundle.feature_names)
            pred = predict_gbdt(bundle, x, ts)

            history[key].loc[ts] = pred
            history[key] = history[key].sort_index()

            rec: dict[str, float | int | pd.Timestamp] = {
                "tollgate_id": int(key[0]),
                "direction": int(key[1]),
                "time_window": ts,
                "horizon": horizon,
                "gbdt_prediction": pred,
            }
            if actual_map is not None:
                rec["actual"] = float(actual_map[(key, ts)])
            records.append(rec)

    return pd.DataFrame(records)


def _disabled_tft_bundle(reason: str) -> TFTBranchBundle:
    return TFTBranchBundle(
        enabled=False,
        feature_names=[],
        model=None,
        target_series=None,
        apply_horizons=set(),
        blend_weight=0.0,
        horizon_blend_weight={},
        use_weather_interactions=False,
        use_event_features=False,
        stats={"enabled": 0, "reason": reason},
    )


def train_tft_branch_for_days(
    cfg: dict,
    x_gbdt_full_all: pd.DataFrame,
    y_gbdt_full_all: pd.Series,
    meta_gbdt_full_all: pd.DataFrame,
    x_gbdt_target_all: pd.DataFrame,
    y_gbdt_target_all: pd.Series,
    meta_gbdt_target_all: pd.DataFrame,
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    include_calendar: bool,
    weather_columns: list[str],
    train_days: list[pd.Timestamp],
) -> TFTBranchBundle:
    tft_cfg = cfg.get("tft_branch", {})
    if not isinstance(tft_cfg, dict) or not bool(tft_cfg.get("use", False)):
        return _disabled_tft_bundle("disabled_by_config")
    if not TORCH_AVAILABLE:
        return _disabled_tft_bundle("torch_unavailable")

    source_branch = str(tft_cfg.get("source_branch", "target")).lower()
    if source_branch not in {"full", "target"}:
        raise ValueError("tft_branch.source_branch must be one of: full, target")

    if source_branch == "full":
        x_src_all, y_src_all, meta_src_all = x_gbdt_full_all, y_gbdt_full_all, meta_gbdt_full_all
    else:
        x_src_all, y_src_all, meta_src_all = x_gbdt_target_all, y_gbdt_target_all, meta_gbdt_target_all

    x_src, y_src, meta_src = select_by_days(x_src_all, y_src_all, meta_src_all, train_days)
    if x_src.empty:
        return _disabled_tft_bundle(f"empty_source_days_{source_branch}")

    use_weather_interactions = bool(tft_cfg.get("use_weather_interactions", True))
    use_event_features = bool(tft_cfg.get("use_event_features", True))
    apply_horizons = _parse_horizon_allowlist(tft_cfg.get("apply_horizons"))
    target_series = parse_series_allowlist(tft_cfg.get("target_series"))
    if target_series is not None:
        mask_series = meta_src.apply(
            lambda r: (int(r["tollgate_id"]), int(r["direction"])) in target_series,
            axis=1,
        )
        x_src = x_src.loc[mask_series].reset_index(drop=True)
        y_src = y_src.loc[mask_series].reset_index(drop=True)
        meta_src = meta_src.loc[mask_series].reset_index(drop=True)

    mask_h = meta_src["horizon"].astype(int).isin(apply_horizons)
    x_src = x_src.loc[mask_h].reset_index(drop=True)
    y_src = y_src.loc[mask_h].reset_index(drop=True)
    meta_src = meta_src.loc[mask_h].reset_index(drop=True)

    min_samples = int(tft_cfg.get("min_samples", 180))
    if len(x_src) < min_samples:
        return _disabled_tft_bundle(f"insufficient_samples_lt_{min_samples}")

    blend_weight = float(np.clip(float(tft_cfg.get("blend_weight", 0.08)), 0.0, 0.95))
    hbw_raw = tft_cfg.get("horizon_blend_weight", {})
    if not isinstance(hbw_raw, dict):
        raise ValueError("tft_branch.horizon_blend_weight must be an object keyed by horizon")
    horizon_blend_weight: dict[int, float] = {}
    for h_text, w_val in hbw_raw.items():
        h = int(h_text)
        if h < 1 or h > 6:
            raise ValueError(f"Invalid horizon in tft_branch.horizon_blend_weight: {h}")
        horizon_blend_weight[h] = float(np.clip(float(w_val), 0.0, 0.95))

    tft_feature_names = build_tft_feature_names(
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        include_calendar=include_calendar,
        weather_columns=weather_columns,
        use_weather_interactions=use_weather_interactions,
        use_event_features=use_event_features,
    )
    x_tft = x_src.copy()
    ext_cols = build_external_feature_columns(use_weather_interactions, use_event_features)
    for col in ext_cols:
        if col not in x_tft.columns:
            x_tft[col] = 0.0
    ts_arr = pd.to_datetime(meta_src["time_window"])
    for i, ts in enumerate(ts_arr):
        feat_row = x_tft.iloc[i].to_dict()
        add_external_features_to_row(
            feat=feat_row,
            ts=pd.Timestamp(ts),
            use_weather_interactions=use_weather_interactions,
            use_event_features=use_event_features,
        )
        for col in ext_cols:
            x_tft.iat[i, x_tft.columns.get_loc(col)] = float(feat_row.get(col, 0.0))
    for col in tft_feature_names:
        if col not in x_tft.columns:
            x_tft[col] = 0.0
    x_tft = x_tft[tft_feature_names].fillna(0.0)

    model_cfg = tft_cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("tft_branch.model must be an object")
    tft_model_cfg = TFTConfig(
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_encoder_layers=int(model_cfg.get("num_encoder_layers", 2)),
        num_decoder_layers=int(model_cfg.get("num_decoder_layers", 2)),
        learning_rate=float(model_cfg.get("learning_rate", 0.001)),
        batch_size=int(model_cfg.get("batch_size", 32)),
        epochs=int(model_cfg.get("epochs", 60)),
        patience=int(model_cfg.get("patience", 10)),
    )

    model = TemporalFusionTransformer(
        config=tft_model_cfg,
        input_dim=len(tft_feature_names),
        output_horizon=1,
    )
    model.fit(x_tft.to_numpy(dtype=float), y_src.to_numpy(dtype=float).reshape(-1, 1))

    stats = {
        "enabled": 1,
        "source_branch": source_branch,
        "samples": int(len(x_tft)),
        "min_samples": int(min_samples),
        "feature_count": int(len(tft_feature_names)),
        "apply_horizons": sorted(int(h) for h in apply_horizons),
        "target_series_count": int(len(target_series)) if target_series is not None else 0,
        "blend_weight": float(blend_weight),
        "horizon_blend_weight": {str(h): float(w) for h, w in sorted(horizon_blend_weight.items())},
        "use_weather_interactions": int(use_weather_interactions),
        "use_event_features": int(use_event_features),
        "model": {
            "hidden_size": int(tft_model_cfg.hidden_size),
            "num_heads": int(tft_model_cfg.num_heads),
            "dropout": float(tft_model_cfg.dropout),
            "learning_rate": float(tft_model_cfg.learning_rate),
            "batch_size": int(tft_model_cfg.batch_size),
            "epochs": int(tft_model_cfg.epochs),
            "patience": int(tft_model_cfg.patience),
        },
    }
    return TFTBranchBundle(
        enabled=True,
        feature_names=tft_feature_names,
        model=model,
        target_series=target_series,
        apply_horizons=apply_horizons,
        blend_weight=blend_weight,
        horizon_blend_weight=horizon_blend_weight,
        use_weather_interactions=use_weather_interactions,
        use_event_features=use_event_features,
        stats=stats,
    )


def run_tft_recursive_forecast(
    bundle: TFTBranchBundle,
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    if not bundle.enabled or bundle.model is None:
        return pd.DataFrame(columns=["tollgate_id", "direction", "time_window", "horizon", "tft_prediction"])

    records: list[dict[str, float | int | pd.Timestamp]] = []
    for ts in schedule:
        for key in series_keys:
            weather_values: dict[str, float] = {}
            if weather_table is not None and weather_defaults_map is not None:
                weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
                weather_values = {k: weather_values[k] for k in weather_columns}

            feat = build_feature_row(
                key=key,
                ts=ts,
                history=history,
                series_keys=series_keys,
                cfg=feature_cfg,
                default_value=default_value,
                allow_fallback=True,
                weather=weather_values,
                use_enhanced_features=use_enhanced_features,
            )
            if feat is None:
                continue

            if include_calendar:
                feat.update(calendar_feature_vector(ts))
            if weather_values:
                feat.update(weather_values)
            add_external_features_to_row(
                feat=feat,
                ts=ts,
                use_weather_interactions=bundle.use_weather_interactions,
                use_event_features=bundle.use_event_features,
            )

            x_row = pd.DataFrame([feat], columns=bundle.feature_names).fillna(0.0)
            pred = float(bundle.model.predict(x_row.to_numpy(dtype=float))[0, 0])
            pred = max(0.0, pred)

            history[key].loc[ts] = pred
            history[key] = history[key].sort_index()

            rec: dict[str, float | int | pd.Timestamp] = {
                "tollgate_id": int(key[0]),
                "direction": int(key[1]),
                "time_window": ts,
                "horizon": int(horizon_index(ts)),
                "tft_prediction": pred,
            }
            if actual_map is not None:
                rec["actual"] = float(actual_map[(key, ts)])
            records.append(rec)

    return pd.DataFrame(records)


def apply_tft_branch_adjustment(
    pred_df: pd.DataFrame,
    tft_pred_df: pd.DataFrame,
    bundle: TFTBranchBundle,
) -> pd.DataFrame:
    out = pred_df.copy()
    if out.empty:
        return out
    if "prediction_before_tft" not in out.columns:
        out["prediction_before_tft"] = out["prediction"].astype(float)
    if "tft_applied" not in out.columns:
        out["tft_applied"] = 0
    if "tft_weight" not in out.columns:
        out["tft_weight"] = 0.0
    if "tft_prediction" not in out.columns:
        out["tft_prediction"] = np.nan
    if out.empty or not bundle.enabled or bundle.model is None or tft_pred_df.empty:
        return out

    keys = ["tollgate_id", "direction", "time_window", "horizon"]
    merged = out.merge(tft_pred_df[keys + ["tft_prediction"]], on=keys, how="left", suffixes=("", "_new"))
    if "tft_prediction_new" in merged.columns:
        merged["tft_prediction"] = merged["tft_prediction_new"]
        merged = merged.drop(columns=["tft_prediction_new"])

    applied: list[int] = []
    weights: list[float] = []
    preds: list[float] = []
    for row in merged.itertuples(index=False):
        pred = float(row.prediction)
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        tft_pred = float(row.tft_prediction) if pd.notna(row.tft_prediction) else np.nan
        use_tft = (
            np.isfinite(tft_pred)
            and h in bundle.apply_horizons
            and (bundle.target_series is None or key in bundle.target_series)
        )
        if use_tft:
            w = float(np.clip(bundle.horizon_blend_weight.get(h, bundle.blend_weight), 0.0, 0.95))
            pred = max(0.0, (1.0 - w) * pred + w * tft_pred)
            applied.append(1)
            weights.append(w)
        else:
            applied.append(0)
            weights.append(0.0)
        preds.append(pred)

    merged["tft_applied"] = applied
    merged["tft_weight"] = weights
    merged["prediction"] = preds
    return merged


def select_by_days(
    x_df: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    days: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    mask = meta_df["day"].isin(days)
    return (
        x_df.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        meta_df.loc[mask].reset_index(drop=True),
    )


def prepare_history_for_schedule(
    history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    schedule: list[pd.Timestamp],
) -> tuple[dict[tuple[int, int], pd.Series], dict[tuple[tuple[int, int], pd.Timestamp], float]]:
    out = {k: s.copy() for k, s in history.items()}
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] = {}
    for key in series_keys:
        for ts in schedule:
            actual_map[(key, ts)] = float(out[key].get(ts, np.nan))
            out[key].loc[ts] = np.nan
    return out, actual_map


def default_fusion_weights(cfg: dict) -> FusionBundle:
    fusion_cfg = cfg.get("fusion", {})
    raw = fusion_cfg.get("default_branch_weights")
    if isinstance(raw, dict):
        linear = float(raw.get("linear", 0.8))
        gbdt_full = float(raw.get("gbdt_full", 0.12))
        gbdt_target = float(raw.get("gbdt_target", 0.08))
    elif isinstance(raw, list) and len(raw) == 3:
        linear = float(raw[0])
        gbdt_full = float(raw[1])
        gbdt_target = float(raw[2])
    else:
        default_gbdt = float(fusion_cfg.get("default_gbdt_weight", 0.2))
        full_ratio = float(fusion_cfg.get("default_gbdt_full_ratio", 0.6))
        full_ratio = float(np.clip(full_ratio, 0.0, 1.0))
        linear = 1.0 - default_gbdt
        gbdt_full = default_gbdt * full_ratio
        gbdt_target = default_gbdt * (1.0 - full_ratio)

    vec = np.array([linear, gbdt_full, gbdt_target], dtype=float)
    vec = np.clip(vec, 0.0, None)
    if float(vec.sum()) <= 0.0:
        vec = np.array([1.0, 0.0, 0.0], dtype=float)
    vec = vec / vec.sum()

    global_weights = AdaptiveTriFusionWeights(
        global_weights=(float(vec[0]), float(vec[1]), float(vec[2])),
        series_weights={},
        slice_weights={},
    )
    return FusionBundle(
        global_weights=global_weights,
        anchor_weights={},
        regime_router=_disabled_regime_router_bundle("not_fitted"),
        memory_retrieval=_disabled_memory_retrieval_bundle("not_fitted"),
        risk_constraint=_disabled_risk_constraint_bundle("not_fitted"),
    )


def _mape_series(y_true: pd.Series, y_pred: pd.Series, eps: float) -> float:
    denom = y_true.abs().clip(lower=eps)
    return float(((y_true - y_pred).abs() / denom).mean())


def _normalize_triplet(
    values: np.ndarray,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    arr = np.clip(values.astype(float), 0.0, None)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0.0:
        return fallback
    arr = arr / s
    return float(arr[0]), float(arr[1]), float(arr[2])


def _triplet_from_errors(
    err_linear: float,
    err_gbdt_full: float,
    err_gbdt_target: float,
    power: float,
    err_floor: float,
    min_weight: float,
    max_weight: float,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    errs = np.array([err_linear, err_gbdt_full, err_gbdt_target], dtype=float)
    if np.any(~np.isfinite(errs)):
        return fallback
    scores = 1.0 / np.maximum(errs, err_floor) ** power
    raw = np.clip(scores, 0.0, None)
    if float(raw.sum()) <= 0.0:
        return fallback
    raw = raw / raw.sum()
    clipped = np.clip(raw, min_weight, max_weight)
    return _normalize_triplet(clipped, fallback=fallback)


def _blend_triplet(
    parent: tuple[float, float, float],
    child: tuple[float, float, float],
    sample_count: int,
    shrink: float,
) -> tuple[float, float, float]:
    ratio = sample_count / (sample_count + max(shrink, 1e-9))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    p = np.array(parent, dtype=float)
    c = np.array(child, dtype=float)
    mixed = (1.0 - ratio) * p + ratio * c
    return _normalize_triplet(mixed, fallback=parent)


def _build_memory_feature_vector(
    linear_prediction: float,
    gbdt_full_prediction: float,
    gbdt_target_prediction: float,
    fused_prediction: float,
) -> np.ndarray:
    denom = max(abs(float(linear_prediction)), 1.0)
    delta_full = (float(gbdt_full_prediction) - float(linear_prediction)) / denom
    delta_target = (float(gbdt_target_prediction) - float(linear_prediction)) / denom
    divergence = abs(float(gbdt_full_prediction) - float(gbdt_target_prediction)) / denom
    mean_shift = (0.5 * (float(gbdt_full_prediction) + float(gbdt_target_prediction)) - float(linear_prediction)) / denom
    level = float(np.log1p(max(float(fused_prediction), 0.0)))
    return np.array([delta_full, delta_target, divergence, mean_shift, level], dtype=float)


def _compute_risk_score(
    linear_prediction: float,
    gbdt_full_prediction: float,
    gbdt_target_prediction: float,
    score_weights: tuple[float, float, float],
) -> float:
    denom = max(abs(float(linear_prediction)), 1.0)
    divergence = abs(float(gbdt_full_prediction) - float(gbdt_target_prediction)) / denom
    spread = (
        max(float(linear_prediction), float(gbdt_full_prediction), float(gbdt_target_prediction))
        - min(float(linear_prediction), float(gbdt_full_prediction), float(gbdt_target_prediction))
    ) / denom
    mean_shift = abs(0.5 * (float(gbdt_full_prediction) + float(gbdt_target_prediction)) - float(linear_prediction)) / denom
    w_div, w_spread, w_shift = score_weights
    return (
        float(w_div) * float(divergence)
        + float(w_spread) * float(spread)
        + float(w_shift) * float(mean_shift)
    )


def _select_branch_from_map(
    branch_idx: int,
    linear_prediction: float,
    gbdt_full_prediction: float,
    gbdt_target_prediction: float,
) -> float:
    if int(branch_idx) == 1:
        return float(gbdt_full_prediction)
    if int(branch_idx) == 2:
        return float(gbdt_target_prediction)
    return float(linear_prediction)


def _build_memory_bucket(
    feature_rows: list[np.ndarray],
    residual_rows: list[float],
    min_samples: int,
    distance_gate_quantile: float,
) -> MemoryBucket | None:
    if len(feature_rows) < int(min_samples) or len(residual_rows) < int(min_samples):
        return None
    x = np.vstack(feature_rows).astype(float)
    r = np.array(residual_rows, dtype=float)
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    center_dist = np.sqrt(np.mean(((x - center[None, :]) / scale[None, :]) ** 2, axis=1))
    gate_q = float(np.clip(float(distance_gate_quantile), 0.5, 0.995))
    gate = float(np.quantile(center_dist, gate_q))
    return MemoryBucket(
        features=x,
        residuals=r,
        center=center.astype(float),
        scale=scale.astype(float),
        distance_gate=gate,
    )


def fit_adaptive_trifusion_weights(
    eval_df: pd.DataFrame,
    cfg: dict,
) -> tuple[AdaptiveTriFusionWeights, dict[str, object]]:
    default_bundle = default_fusion_weights({"fusion": cfg}).global_weights
    if eval_df.empty:
        return default_bundle, {"reason": "empty_adaptation_frame"}

    eps = float(cfg.get("mape_eps", 1.0))
    err_floor = float(cfg.get("weight_eps", 1e-9))
    power = float(cfg.get("error_power", 1.0))
    min_weight = float(cfg.get("min_branch_weight", 0.0))
    max_weight = float(cfg.get("max_branch_weight", 1.0))
    min_series_samples = int(cfg.get("min_series_samples", 10))
    min_slice_samples = int(cfg.get("min_slice_samples", 6))
    series_shrink = float(cfg.get("series_shrink", 24.0))
    slice_shrink = float(cfg.get("slice_shrink", 12.0))

    frame = eval_df.copy()
    required = {
        "actual",
        "linear_prediction",
        "gbdt_full_prediction",
        "gbdt_target_prediction",
        "tollgate_id",
        "direction",
        "horizon",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing adaptation columns for tri-fusion: {missing}")

    global_err_linear = _mape_series(frame["actual"], frame["linear_prediction"], eps=eps)
    global_err_full = _mape_series(frame["actual"], frame["gbdt_full_prediction"], eps=eps)
    global_err_target = _mape_series(frame["actual"], frame["gbdt_target_prediction"], eps=eps)
    global_weight = _triplet_from_errors(
        global_err_linear,
        global_err_full,
        global_err_target,
        power=power,
        err_floor=err_floor,
        min_weight=min_weight,
        max_weight=max_weight,
        fallback=default_bundle.global_weights,
    )

    series_map: dict[tuple[int, int], tuple[float, float, float]] = {}
    series_stats: dict[str, dict[str, float | int]] = {}
    grouped_series = frame.groupby(["tollgate_id", "direction"], sort=True)
    for (tollgate_id, direction), part in grouped_series:
        n = int(len(part))
        key = (int(tollgate_id), int(direction))
        if n < min_series_samples:
            continue
        err_linear = _mape_series(part["actual"], part["linear_prediction"], eps=eps)
        err_full = _mape_series(part["actual"], part["gbdt_full_prediction"], eps=eps)
        err_target = _mape_series(part["actual"], part["gbdt_target_prediction"], eps=eps)
        child = _triplet_from_errors(
            err_linear,
            err_full,
            err_target,
            power=power,
            err_floor=err_floor,
            min_weight=min_weight,
            max_weight=max_weight,
            fallback=global_weight,
        )
        final = _blend_triplet(global_weight, child, sample_count=n, shrink=series_shrink)
        series_map[key] = final
        series_stats[f"{key[0]}_{key[1]}"] = {
            "samples": n,
            "linear_mape": err_linear * 100.0,
            "gbdt_full_mape": err_full * 100.0,
            "gbdt_target_mape": err_target * 100.0,
            "linear_weight": final[0],
            "gbdt_full_weight": final[1],
            "gbdt_target_weight": final[2],
        }

    slice_map: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    slice_stats: dict[str, dict[str, float | int]] = {}
    grouped_slice = frame.groupby(["tollgate_id", "direction", "horizon"], sort=True)
    for (tollgate_id, direction, horizon), part in grouped_slice:
        n = int(len(part))
        key = (int(tollgate_id), int(direction), int(horizon))
        if n < min_slice_samples:
            continue
        err_linear = _mape_series(part["actual"], part["linear_prediction"], eps=eps)
        err_full = _mape_series(part["actual"], part["gbdt_full_prediction"], eps=eps)
        err_target = _mape_series(part["actual"], part["gbdt_target_prediction"], eps=eps)
        parent = series_map.get((key[0], key[1]), global_weight)
        child = _triplet_from_errors(
            err_linear,
            err_full,
            err_target,
            power=power,
            err_floor=err_floor,
            min_weight=min_weight,
            max_weight=max_weight,
            fallback=parent,
        )
        final = _blend_triplet(parent, child, sample_count=n, shrink=slice_shrink)
        slice_map[key] = final
        slice_stats[f"{key[0]}_{key[1]}_h{key[2]}"] = {
            "samples": n,
            "linear_mape": err_linear * 100.0,
            "gbdt_full_mape": err_full * 100.0,
            "gbdt_target_mape": err_target * 100.0,
            "linear_weight": final[0],
            "gbdt_full_weight": final[1],
            "gbdt_target_weight": final[2],
        }

    bundle = AdaptiveTriFusionWeights(
        global_weights=global_weight,
        series_weights=series_map,
        slice_weights=slice_map,
    )
    stats = {
        "adapt_rows": int(len(frame)),
        "global_linear_mape": global_err_linear * 100.0,
        "global_gbdt_full_mape": global_err_full * 100.0,
        "global_gbdt_target_mape": global_err_target * 100.0,
        "global_linear_weight": global_weight[0],
        "global_gbdt_full_weight": global_weight[1],
        "global_gbdt_target_weight": global_weight[2],
        "series_weights": series_stats,
        "slice_weights": slice_stats,
    }
    return bundle, stats


def _classify_regime_label(
    horizon: int,
    linear_prediction: float,
    gbdt_full_prediction: float,
    gbdt_target_prediction: float,
    up_threshold: float,
    down_threshold: float,
    conflict_threshold: float,
    conflict_threshold_h6: float | None,
) -> str:
    denom = max(abs(float(linear_prediction)), 1.0)
    delta_full = (float(gbdt_full_prediction) - float(linear_prediction)) / denom
    delta_target = (float(gbdt_target_prediction) - float(linear_prediction)) / denom
    divergence = abs(float(gbdt_full_prediction) - float(gbdt_target_prediction)) / denom

    conflict_thr = float(conflict_threshold)
    if int(horizon) == 6 and conflict_threshold_h6 is not None:
        conflict_thr = float(conflict_threshold_h6)
    if divergence >= conflict_thr:
        return "conflict"

    mean_delta = 0.5 * (delta_full + delta_target)
    if mean_delta >= float(up_threshold):
        return "up"
    if mean_delta <= -float(down_threshold):
        return "down"
    return "stable"


def fit_regime_router_weights(
    fit_frame: pd.DataFrame,
    fusion_cfg: dict,
    fallback_weights: tuple[float, float, float],
) -> RegimeRouterBundle:
    raw_router_cfg = fusion_cfg.get("regime_router", {})
    if raw_router_cfg is None:
        raw_router_cfg = {}
    if not isinstance(raw_router_cfg, dict):
        raise ValueError("fusion.regime_router must be an object")

    if not bool(raw_router_cfg.get("use", False)):
        return _disabled_regime_router_bundle("disabled_by_config")
    if fit_frame.empty:
        return _disabled_regime_router_bundle("empty_fit_frame")

    up_threshold = float(raw_router_cfg.get("up_threshold", 0.06))
    down_threshold = float(raw_router_cfg.get("down_threshold", 0.06))
    conflict_threshold = float(raw_router_cfg.get("conflict_threshold", 0.12))
    conflict_threshold_h6_raw = raw_router_cfg.get("conflict_threshold_h6")
    conflict_threshold_h6 = (
        float(conflict_threshold_h6_raw) if conflict_threshold_h6_raw is not None else None
    )
    min_samples = int(raw_router_cfg.get("min_samples", 40))
    min_h6_samples = int(raw_router_cfg.get("min_h6_samples", max(10, min_samples // 2)))
    shrink = float(raw_router_cfg.get("shrink", 60.0))
    h6_shrink = float(raw_router_cfg.get("h6_shrink", max(20.0, shrink * 0.7)))
    blend_strength = float(np.clip(float(raw_router_cfg.get("blend_strength", 0.55)), 0.0, 1.0))
    blend_strength_h6 = float(
        np.clip(float(raw_router_cfg.get("blend_strength_h6", min(1.0, blend_strength + 0.15))), 0.0, 1.0)
    )

    eps = float(fusion_cfg.get("mape_eps", 1.0))
    err_floor = float(fusion_cfg.get("weight_eps", 1e-9))
    power = float(fusion_cfg.get("error_power", 1.0))
    min_weight = float(fusion_cfg.get("min_branch_weight", 0.0))
    max_weight = float(fusion_cfg.get("max_branch_weight", 1.0))

    frame = fit_frame.copy()
    required = {
        "actual",
        "linear_prediction",
        "gbdt_full_prediction",
        "gbdt_target_prediction",
        "horizon",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing adaptation columns for regime router: {missing}")

    regimes: list[str] = []
    for row in frame.itertuples(index=False):
        regimes.append(
            _classify_regime_label(
                horizon=int(row.horizon),
                linear_prediction=float(row.linear_prediction),
                gbdt_full_prediction=float(row.gbdt_full_prediction),
                gbdt_target_prediction=float(row.gbdt_target_prediction),
                up_threshold=up_threshold,
                down_threshold=down_threshold,
                conflict_threshold=conflict_threshold,
                conflict_threshold_h6=conflict_threshold_h6,
            )
        )
    frame["regime"] = regimes
    frame["horizon"] = frame["horizon"].astype(int)

    regime_counts = frame["regime"].value_counts(dropna=False).to_dict()
    regime_weights: dict[str, tuple[float, float, float]] = {}
    regime_stats: dict[str, dict[str, float | int]] = {}

    for regime in ("stable", "up", "down", "conflict"):
        part = frame[frame["regime"] == regime]
        n = int(len(part))
        if n < min_samples:
            continue
        err_linear = _mape_series(part["actual"], part["linear_prediction"], eps=eps)
        err_full = _mape_series(part["actual"], part["gbdt_full_prediction"], eps=eps)
        err_target = _mape_series(part["actual"], part["gbdt_target_prediction"], eps=eps)
        child = _triplet_from_errors(
            err_linear,
            err_full,
            err_target,
            power=power,
            err_floor=err_floor,
            min_weight=min_weight,
            max_weight=max_weight,
            fallback=fallback_weights,
        )
        final = _blend_triplet(fallback_weights, child, sample_count=n, shrink=shrink)
        regime_weights[regime] = final
        regime_stats[regime] = {
            "samples": n,
            "linear_mape": err_linear * 100.0,
            "gbdt_full_mape": err_full * 100.0,
            "gbdt_target_mape": err_target * 100.0,
            "linear_weight": final[0],
            "gbdt_full_weight": final[1],
            "gbdt_target_weight": final[2],
        }

    h6_weights: dict[str, tuple[float, float, float]] = {}
    h6_stats: dict[str, dict[str, float | int]] = {}
    h6_frame = frame[frame["horizon"] == 6]
    for regime in ("stable", "up", "down", "conflict"):
        part = h6_frame[h6_frame["regime"] == regime]
        n = int(len(part))
        if n < min_h6_samples:
            continue
        err_linear = _mape_series(part["actual"], part["linear_prediction"], eps=eps)
        err_full = _mape_series(part["actual"], part["gbdt_full_prediction"], eps=eps)
        err_target = _mape_series(part["actual"], part["gbdt_target_prediction"], eps=eps)
        parent = regime_weights.get(regime, fallback_weights)
        child = _triplet_from_errors(
            err_linear,
            err_full,
            err_target,
            power=power,
            err_floor=err_floor,
            min_weight=min_weight,
            max_weight=max_weight,
            fallback=parent,
        )
        final = _blend_triplet(parent, child, sample_count=n, shrink=h6_shrink)
        h6_weights[regime] = final
        h6_stats[regime] = {
            "samples": n,
            "linear_mape": err_linear * 100.0,
            "gbdt_full_mape": err_full * 100.0,
            "gbdt_target_mape": err_target * 100.0,
            "linear_weight": final[0],
            "gbdt_full_weight": final[1],
            "gbdt_target_weight": final[2],
        }

    if not regime_weights and not h6_weights:
        disabled = _disabled_regime_router_bundle("insufficient_samples")
        disabled.stats["regime_counts"] = {str(k): int(v) for k, v in regime_counts.items()}
        disabled.stats["min_samples"] = int(min_samples)
        disabled.stats["min_h6_samples"] = int(min_h6_samples)
        return disabled

    stats = {
        "enabled": 1,
        "rows": int(len(frame)),
        "regime_counts": {str(k): int(v) for k, v in regime_counts.items()},
        "regime_weight_count": int(len(regime_weights)),
        "h6_regime_weight_count": int(len(h6_weights)),
        "blend_strength": float(blend_strength),
        "blend_strength_h6": float(blend_strength_h6),
        "regime_weights": regime_stats,
        "h6_regime_weights": h6_stats,
    }
    return RegimeRouterBundle(
        enabled=True,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        conflict_threshold=conflict_threshold,
        conflict_threshold_h6=conflict_threshold_h6,
        blend_strength=blend_strength,
        blend_strength_h6=blend_strength_h6,
        weights_by_regime=regime_weights,
        h6_weights_by_regime=h6_weights,
        stats=stats,
    )


def _predict_from_fusion_frame(
    fit_frame: pd.DataFrame,
    fusion_weights: FusionBundle,
    memory_bundle: MemoryRetrievalBundle | None = None,
) -> pd.DataFrame:
    if fit_frame.empty:
        return fit_frame.copy()

    frame = fit_frame.copy()
    if "anchor" not in frame.columns:
        frame["anchor"] = 0

    preds_before_memory: list[float] = []
    memory_corrections: list[float] = []
    memory_applied: list[int] = []
    preds_final: list[float] = []
    for row in frame.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        anchor = int(row.anchor)
        linear_pred = float(row.linear_prediction)
        full_pred = float(row.gbdt_full_prediction)
        target_pred = float(row.gbdt_target_prediction)
        w_linear, w_full, w_target = fusion_weights.resolve_with_anchor(
            key=key,
            horizon=h,
            anchor=anchor,
            linear_prediction=linear_pred,
            gbdt_full_prediction=full_pred,
            gbdt_target_prediction=target_pred,
        )
        pred_before = max(
            0.0,
            w_linear * linear_pred + w_full * full_pred + w_target * target_pred,
        )
        corr = 0.0
        applied = 0
        pred_after = pred_before
        if memory_bundle is not None and memory_bundle.enabled:
            corr, applied, _bucket_type, _bucket_level, _min_dist = memory_bundle.retrieve_delta(
                key=key,
                horizon=h,
                anchor=anchor,
                linear_prediction=linear_pred,
                gbdt_full_prediction=full_pred,
                gbdt_target_prediction=target_pred,
                base_prediction=pred_before,
            )
            if applied:
                pred_after = max(0.0, float(pred_before) + float(corr))

        preds_before_memory.append(float(pred_before))
        memory_corrections.append(float(corr))
        memory_applied.append(int(applied))
        preds_final.append(float(pred_after))

    frame["prediction_before_memory"] = preds_before_memory
    frame["memory_correction"] = memory_corrections
    frame["memory_applied"] = memory_applied
    frame["prediction"] = preds_final
    return frame


def fit_memory_retrieval_bundle(
    fit_frame: pd.DataFrame,
    fusion_cfg: dict,
    fusion_weights: FusionBundle,
) -> MemoryRetrievalBundle:
    raw_cfg = fusion_cfg.get("memory_retrieval", {})
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("fusion.memory_retrieval must be an object")

    if not bool(raw_cfg.get("use", False)):
        return _disabled_memory_retrieval_bundle("disabled_by_config")
    if fit_frame.empty:
        return _disabled_memory_retrieval_bundle("empty_fit_frame")

    target_series = parse_series_allowlist(raw_cfg.get("target_series"))
    apply_horizons = _parse_horizon_allowlist(raw_cfg.get("apply_horizons"))
    use_anchor = bool(raw_cfg.get("use_anchor", True))

    top_k = max(1, int(raw_cfg.get("top_k", 12)))
    distance_power = float(raw_cfg.get("distance_power", 1.0))
    blend_weight = float(np.clip(float(raw_cfg.get("blend_weight", 0.2)), 0.0, 0.95))
    max_abs_delta = max(0.0, float(raw_cfg.get("max_abs_delta", 10.0)))
    distance_gate_quantile = float(raw_cfg.get("distance_gate_quantile", 0.9))
    distance_gate_scale = max(0.1, float(raw_cfg.get("distance_gate_scale", 1.25)))

    min_primary_samples = max(1, int(raw_cfg.get("min_primary_samples", 12)))
    min_series_samples = max(
        1,
        int(raw_cfg.get("min_series_samples", max(min_primary_samples, 18))),
    )
    min_global_samples = max(
        1,
        int(raw_cfg.get("min_global_samples", max(min_series_samples, 28))),
    )

    required = {
        "actual",
        "linear_prediction",
        "gbdt_full_prediction",
        "gbdt_target_prediction",
        "tollgate_id",
        "direction",
        "horizon",
        "anchor",
    }
    missing = sorted(required - set(fit_frame.columns))
    if missing:
        raise ValueError(f"Missing adaptation columns for memory retrieval: {missing}")

    pred_frame = _predict_from_fusion_frame(
        fit_frame=fit_frame,
        fusion_weights=fusion_weights,
        memory_bundle=None,
    )
    if pred_frame.empty:
        return _disabled_memory_retrieval_bundle("empty_pred_frame")

    primary_feats: dict[tuple[int, int, int, int], list[np.ndarray]] = {}
    primary_res: dict[tuple[int, int, int, int], list[float]] = {}
    series_feats: dict[tuple[int, int, int], list[np.ndarray]] = {}
    series_res: dict[tuple[int, int, int], list[float]] = {}
    global_feats: dict[tuple[int, int], list[np.ndarray]] = {}
    global_res: dict[tuple[int, int], list[float]] = {}

    used_rows = 0
    for row in pred_frame.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        if target_series is not None and key not in target_series:
            continue
        if h not in apply_horizons:
            continue
        actual = float(row.actual)
        pred = float(row.prediction)
        if not np.isfinite(actual) or not np.isfinite(pred):
            continue

        anchor_key = int(row.anchor) if use_anchor else -1
        feat = _build_memory_feature_vector(
            linear_prediction=float(row.linear_prediction),
            gbdt_full_prediction=float(row.gbdt_full_prediction),
            gbdt_target_prediction=float(row.gbdt_target_prediction),
            fused_prediction=pred,
        )
        residual = float(actual - pred)

        p_key = (int(key[0]), int(key[1]), int(h), int(anchor_key))
        s_key = (int(key[0]), int(key[1]), int(anchor_key))
        g_key = (int(h), int(anchor_key))

        primary_feats.setdefault(p_key, []).append(feat)
        primary_res.setdefault(p_key, []).append(residual)
        series_feats.setdefault(s_key, []).append(feat)
        series_res.setdefault(s_key, []).append(residual)
        global_feats.setdefault(g_key, []).append(feat)
        global_res.setdefault(g_key, []).append(residual)
        used_rows += 1

    primary_buckets: dict[tuple[int, int, int, int], MemoryBucket] = {}
    for k, feat_rows in primary_feats.items():
        bucket = _build_memory_bucket(
            feature_rows=feat_rows,
            residual_rows=primary_res[k],
            min_samples=min_primary_samples,
            distance_gate_quantile=distance_gate_quantile,
        )
        if bucket is not None:
            primary_buckets[k] = bucket

    series_buckets: dict[tuple[int, int, int], MemoryBucket] = {}
    for k, feat_rows in series_feats.items():
        bucket = _build_memory_bucket(
            feature_rows=feat_rows,
            residual_rows=series_res[k],
            min_samples=min_series_samples,
            distance_gate_quantile=distance_gate_quantile,
        )
        if bucket is not None:
            series_buckets[k] = bucket

    global_buckets: dict[tuple[int, int], MemoryBucket] = {}
    for k, feat_rows in global_feats.items():
        bucket = _build_memory_bucket(
            feature_rows=feat_rows,
            residual_rows=global_res[k],
            min_samples=min_global_samples,
            distance_gate_quantile=distance_gate_quantile,
        )
        if bucket is not None:
            global_buckets[k] = bucket

    if not primary_buckets and not series_buckets and not global_buckets:
        disabled = _disabled_memory_retrieval_bundle("insufficient_bucket_samples")
        disabled.stats["raw_rows"] = int(len(pred_frame))
        disabled.stats["used_rows"] = int(used_rows)
        disabled.stats["candidate_primary_buckets"] = int(len(primary_feats))
        disabled.stats["candidate_series_buckets"] = int(len(series_feats))
        disabled.stats["candidate_global_buckets"] = int(len(global_feats))
        return disabled

    stats: dict[str, object] = {
        "enabled": 1,
        "raw_rows": int(len(pred_frame)),
        "used_rows": int(used_rows),
        "target_series_count": int(len(target_series)) if target_series is not None else -1,
        "apply_horizons": sorted(int(h) for h in apply_horizons),
        "use_anchor": int(use_anchor),
        "top_k": int(top_k),
        "distance_power": float(distance_power),
        "blend_weight": float(blend_weight),
        "max_abs_delta": float(max_abs_delta),
        "distance_gate_quantile": float(distance_gate_quantile),
        "distance_gate_scale": float(distance_gate_scale),
        "min_primary_samples": int(min_primary_samples),
        "min_series_samples": int(min_series_samples),
        "min_global_samples": int(min_global_samples),
        "primary_bucket_count": int(len(primary_buckets)),
        "series_bucket_count": int(len(series_buckets)),
        "global_bucket_count": int(len(global_buckets)),
    }

    return MemoryRetrievalBundle(
        enabled=True,
        target_series=target_series,
        apply_horizons=apply_horizons,
        use_anchor=use_anchor,
        top_k=top_k,
        distance_power=distance_power,
        blend_weight=blend_weight,
        max_abs_delta=max_abs_delta,
        distance_gate_scale=distance_gate_scale,
        primary_buckets=primary_buckets,
        series_buckets=series_buckets,
        global_buckets=global_buckets,
        stats=stats,
    )


def _search_risk_gate(
    score: np.ndarray,
    pred: np.ndarray,
    safe_pred: np.ndarray,
    actual: np.ndarray,
    denom: np.ndarray,
    quantiles: list[float],
    shrink_candidates: list[float],
    min_samples: int,
) -> dict[str, float | int] | None:
    n = int(len(score))
    if n == 0 or n < int(min_samples):
        return None

    base_err = np.abs(actual - pred) / denom
    best: dict[str, float | int] | None = None
    for q in quantiles:
        q_clamped = float(np.clip(float(q), 0.0, 0.995))
        threshold = float(np.quantile(score, q_clamped))
        mask = score >= threshold
        selected = int(mask.sum())
        if selected < int(min_samples):
            continue
        for shrink_raw in shrink_candidates:
            shrink = float(np.clip(float(shrink_raw), 0.0, 0.95))
            pred_new = pred.copy()
            pred_new[mask] = (1.0 - shrink) * pred_new[mask] + shrink * safe_pred[mask]
            pred_new = np.clip(pred_new, 0.0, None)
            new_err = np.abs(actual - pred_new) / denom
            gain = base_err - new_err
            expected_gain = float(np.sum(gain) / max(len(gain), 1))
            mean_selected_gain = float(np.mean(gain[mask])) if selected > 0 else float("-inf")
            cand = {
                "threshold": float(threshold),
                "shrink": float(shrink),
                "selected": int(selected),
                "selected_ratio": float(selected / max(n, 1)),
                "expected_gain": float(expected_gain),
                "mean_selected_gain": float(mean_selected_gain),
                "quantile": float(q_clamped),
            }
            if best is None:
                best = cand
                continue
            if float(cand["expected_gain"]) > float(best["expected_gain"]) + 1e-12:
                best = cand
                continue
            if (
                abs(float(cand["expected_gain"]) - float(best["expected_gain"])) <= 1e-12
                and float(cand["mean_selected_gain"]) > float(best["mean_selected_gain"]) + 1e-12
            ):
                best = cand
    return best


def fit_risk_constraint_bundle(
    fit_frame: pd.DataFrame,
    fusion_cfg: dict,
    fusion_weights: FusionBundle,
    memory_bundle: MemoryRetrievalBundle | None = None,
) -> RiskConstraintBundle:
    raw_cfg = fusion_cfg.get("risk_constraint", {})
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("fusion.risk_constraint must be an object")

    if not bool(raw_cfg.get("use", False)):
        return _disabled_risk_constraint_bundle("disabled_by_config")
    if fit_frame.empty:
        return _disabled_risk_constraint_bundle("empty_fit_frame")

    required = {
        "actual",
        "linear_prediction",
        "gbdt_full_prediction",
        "gbdt_target_prediction",
        "tollgate_id",
        "direction",
        "horizon",
        "anchor",
    }
    missing = sorted(required - set(fit_frame.columns))
    if missing:
        raise ValueError(f"Missing adaptation columns for risk constraint: {missing}")

    target_series = parse_series_allowlist(raw_cfg.get("target_series"))
    apply_horizons = _parse_horizon_allowlist(raw_cfg.get("apply_horizons"))
    eps = float(fusion_cfg.get("mape_eps", 1.0))

    score_weights_raw = raw_cfg.get("score_weights", {})
    if isinstance(score_weights_raw, dict):
        w_div = float(score_weights_raw.get("divergence", 0.55))
        w_spread = float(score_weights_raw.get("spread", 0.25))
        w_shift = float(score_weights_raw.get("mean_shift", 0.20))
    elif isinstance(score_weights_raw, list) and len(score_weights_raw) == 3:
        w_div = float(score_weights_raw[0])
        w_spread = float(score_weights_raw[1])
        w_shift = float(score_weights_raw[2])
    else:
        w_div, w_spread, w_shift = 0.55, 0.25, 0.20
    w_vec = np.array([w_div, w_spread, w_shift], dtype=float)
    w_vec = np.clip(w_vec, 0.0, None)
    if float(w_vec.sum()) <= 0.0:
        w_vec = np.array([0.55, 0.25, 0.20], dtype=float)
    w_vec = w_vec / w_vec.sum()
    score_weights = (float(w_vec[0]), float(w_vec[1]), float(w_vec[2]))

    pred_frame = _predict_from_fusion_frame(
        fit_frame=fit_frame,
        fusion_weights=fusion_weights,
        memory_bundle=memory_bundle,
    )
    if pred_frame.empty:
        return _disabled_risk_constraint_bundle("empty_pred_frame")

    eligible_mask: list[bool] = []
    for row in pred_frame.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        ok = (target_series is None or key in target_series) and (h in apply_horizons)
        ok = ok and np.isfinite(float(row.actual)) and np.isfinite(float(row.prediction))
        eligible_mask.append(bool(ok))
    eligible = pred_frame.loc[np.array(eligible_mask, dtype=bool)].reset_index(drop=True)
    if eligible.empty:
        return _disabled_risk_constraint_bundle("empty_eligible_rows")

    min_slice_samples = max(1, int(raw_cfg.get("min_slice_samples", 10)))
    safe_branch_by_slice: dict[tuple[int, int, int], int] = {}
    grouped = eligible.groupby(["tollgate_id", "direction", "horizon"], sort=True)
    for (tollgate_id, direction, horizon), part in grouped:
        if len(part) < min_slice_samples:
            continue
        err_linear = _mape_series(part["actual"], part["linear_prediction"], eps=eps)
        err_full = _mape_series(part["actual"], part["gbdt_full_prediction"], eps=eps)
        err_target = _mape_series(part["actual"], part["gbdt_target_prediction"], eps=eps)
        safe_branch = int(np.argmin([err_linear, err_full, err_target]))
        safe_branch_by_slice[(int(tollgate_id), int(direction), int(horizon))] = safe_branch

    global_err_linear = _mape_series(eligible["actual"], eligible["linear_prediction"], eps=eps)
    global_err_full = _mape_series(eligible["actual"], eligible["gbdt_full_prediction"], eps=eps)
    global_err_target = _mape_series(eligible["actual"], eligible["gbdt_target_prediction"], eps=eps)
    safe_branch_global = int(np.argmin([global_err_linear, global_err_full, global_err_target]))

    score_vals: list[float] = []
    safe_pred_vals: list[float] = []
    safe_branch_vals: list[int] = []
    for row in eligible.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        score_vals.append(
            _compute_risk_score(
                linear_prediction=float(row.linear_prediction),
                gbdt_full_prediction=float(row.gbdt_full_prediction),
                gbdt_target_prediction=float(row.gbdt_target_prediction),
                score_weights=score_weights,
            )
        )
        safe_branch = safe_branch_by_slice.get((key[0], key[1], h), safe_branch_global)
        safe_branch_vals.append(int(safe_branch))
        safe_pred_vals.append(
            _select_branch_from_map(
                branch_idx=int(safe_branch),
                linear_prediction=float(row.linear_prediction),
                gbdt_full_prediction=float(row.gbdt_full_prediction),
                gbdt_target_prediction=float(row.gbdt_target_prediction),
            )
        )

    score_arr = np.array(score_vals, dtype=float)
    pred_arr = eligible["prediction"].to_numpy(dtype=float)
    safe_pred_arr = np.array(safe_pred_vals, dtype=float)
    actual_arr = eligible["actual"].to_numpy(dtype=float)
    denom_arr = np.maximum(np.abs(actual_arr), eps)

    quantiles_raw = raw_cfg.get("quantiles", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    shrink_raw = raw_cfg.get("shrink_candidates", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    if not isinstance(quantiles_raw, list) or not quantiles_raw:
        raise ValueError("fusion.risk_constraint.quantiles must be a non-empty list")
    if not isinstance(shrink_raw, list) or not shrink_raw:
        raise ValueError("fusion.risk_constraint.shrink_candidates must be a non-empty list")
    quantiles = [float(q) for q in quantiles_raw]
    shrink_candidates = [float(s) for s in shrink_raw]

    min_samples = max(1, int(raw_cfg.get("min_samples", 20)))
    min_expected_gain = float(raw_cfg.get("min_expected_gain", 0.0))
    min_selected_gain = float(raw_cfg.get("min_selected_gain", 0.0))

    best_global = _search_risk_gate(
        score=score_arr,
        pred=pred_arr,
        safe_pred=safe_pred_arr,
        actual=actual_arr,
        denom=denom_arr,
        quantiles=quantiles,
        shrink_candidates=shrink_candidates,
        min_samples=min_samples,
    )
    if best_global is None:
        disabled = _disabled_risk_constraint_bundle("no_candidate_passed_min_samples")
        disabled.stats["eligible_rows"] = int(len(eligible))
        disabled.stats["min_samples"] = int(min_samples)
        return disabled
    if float(best_global["expected_gain"]) < min_expected_gain or float(best_global["mean_selected_gain"]) < min_selected_gain:
        disabled = _disabled_risk_constraint_bundle("candidate_gain_below_threshold")
        disabled.stats["eligible_rows"] = int(len(eligible))
        disabled.stats["best_candidate"] = best_global
        disabled.stats["min_expected_gain"] = float(min_expected_gain)
        disabled.stats["min_selected_gain"] = float(min_selected_gain)
        return disabled

    threshold_h6: float | None = None
    shrink_h6 = float(best_global["shrink"])
    h6_best: dict[str, float | int] | None = None
    if bool(raw_cfg.get("fit_h6_override", True)):
        h6_mask = eligible["horizon"].astype(int).to_numpy() == 6
        h6_count = int(np.sum(h6_mask))
        min_h6_samples = max(1, int(raw_cfg.get("min_h6_samples", max(6, min_samples // 2))))
        if h6_count >= min_h6_samples:
            h6_quantiles_raw = raw_cfg.get("h6_quantiles", quantiles)
            h6_shrink_raw = raw_cfg.get("h6_shrink_candidates", shrink_candidates)
            if not isinstance(h6_quantiles_raw, list) or not h6_quantiles_raw:
                raise ValueError("fusion.risk_constraint.h6_quantiles must be a non-empty list")
            if not isinstance(h6_shrink_raw, list) or not h6_shrink_raw:
                raise ValueError("fusion.risk_constraint.h6_shrink_candidates must be a non-empty list")
            h6_min_expected_gain = float(raw_cfg.get("h6_min_expected_gain", min_expected_gain))
            h6_min_selected_gain = float(raw_cfg.get("h6_min_selected_gain", min_selected_gain))
            h6_best = _search_risk_gate(
                score=score_arr[h6_mask],
                pred=pred_arr[h6_mask],
                safe_pred=safe_pred_arr[h6_mask],
                actual=actual_arr[h6_mask],
                denom=denom_arr[h6_mask],
                quantiles=[float(q) for q in h6_quantiles_raw],
                shrink_candidates=[float(s) for s in h6_shrink_raw],
                min_samples=min_h6_samples,
            )
            if h6_best is not None:
                if (
                    float(h6_best["expected_gain"]) >= h6_min_expected_gain
                    and float(h6_best["mean_selected_gain"]) >= h6_min_selected_gain
                ):
                    threshold_h6 = float(h6_best["threshold"])
                    shrink_h6 = float(h6_best["shrink"])

    branch_names = ["linear", "gbdt_full", "gbdt_target"]
    safe_branch_stats = {
        f"{k[0]}_{k[1]}_h{k[2]}": branch_names[int(v)]
        for k, v in sorted(safe_branch_by_slice.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
    }
    stats: dict[str, object] = {
        "enabled": 1,
        "eligible_rows": int(len(eligible)),
        "target_series_count": int(len(target_series)) if target_series is not None else -1,
        "apply_horizons": sorted(int(h) for h in apply_horizons),
        "score_weights": {
            "divergence": float(score_weights[0]),
            "spread": float(score_weights[1]),
            "mean_shift": float(score_weights[2]),
        },
        "safe_branch_global": branch_names[int(safe_branch_global)],
        "safe_branch_slice_count": int(len(safe_branch_by_slice)),
        "safe_branch_by_slice": safe_branch_stats,
        "best_global": best_global,
        "threshold": float(best_global["threshold"]),
        "shrink": float(best_global["shrink"]),
        "threshold_h6": float(threshold_h6) if threshold_h6 is not None else None,
        "shrink_h6": float(shrink_h6),
        "h6_best": h6_best,
    }

    return RiskConstraintBundle(
        enabled=True,
        target_series=target_series,
        apply_horizons=apply_horizons,
        score_weights=score_weights,
        threshold=float(best_global["threshold"]),
        threshold_h6=threshold_h6,
        shrink=float(best_global["shrink"]),
        shrink_h6=float(shrink_h6),
        safe_branch_global=int(safe_branch_global),
        safe_branch_by_slice=safe_branch_by_slice,
        stats=stats,
    )


def fit_anchor_fusion_weights(
    fit_frame: pd.DataFrame,
    fusion_cfg: dict,
) -> tuple[FusionBundle, dict[str, object]]:
    global_fit, global_stats = fit_adaptive_trifusion_weights(fit_frame, fusion_cfg)
    min_anchor_rows = int(fusion_cfg.get("min_anchor_rows", 60))
    anchor_map: dict[int, AdaptiveTriFusionWeights] = {}
    anchor_stats: dict[str, object] = {}
    for anchor in (0, 1):
        part = fit_frame[fit_frame["anchor"] == anchor]
        if len(part) < min_anchor_rows:
            continue
        fit_obj, stats = fit_adaptive_trifusion_weights(part, fusion_cfg)
        anchor_map[anchor] = fit_obj
        anchor_stats[str(anchor)] = stats

    regime_router = fit_regime_router_weights(
        fit_frame=fit_frame,
        fusion_cfg=fusion_cfg,
        fallback_weights=global_fit.global_weights,
    )
    base_bundle = FusionBundle(
        global_weights=global_fit,
        anchor_weights=anchor_map,
        regime_router=regime_router,
        memory_retrieval=_disabled_memory_retrieval_bundle("not_fitted"),
        risk_constraint=_disabled_risk_constraint_bundle("not_fitted"),
    )
    memory_bundle = fit_memory_retrieval_bundle(
        fit_frame=fit_frame,
        fusion_cfg=fusion_cfg,
        fusion_weights=base_bundle,
    )
    risk_bundle = fit_risk_constraint_bundle(
        fit_frame=fit_frame,
        fusion_cfg=fusion_cfg,
        fusion_weights=base_bundle,
        memory_bundle=memory_bundle,
    )
    bundle = FusionBundle(
        global_weights=global_fit,
        anchor_weights=anchor_map,
        regime_router=regime_router,
        memory_retrieval=memory_bundle,
        risk_constraint=risk_bundle,
    )
    stats = {
        "global": global_stats,
        "anchor": anchor_stats,
        "anchor_weight_count": int(len(anchor_map)),
        "regime_router": regime_router.stats,
        "memory_retrieval": memory_bundle.stats,
        "risk_constraint": risk_bundle.stats,
    }
    return bundle, stats


def train_baseline_branch(
    cfg: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    feature_names: list[str],
) -> BaselineBranchBundle:
    residual_cfg = cfg.get("residual", {})
    use_residual = bool(residual_cfg.get("use_residual", False))
    residual_clip_abs = float(residual_cfg.get("clip_abs", 40.0)) if use_residual else None

    primary_bundle, primary_stats = baseline_mod.train_primary_models(
        cfg=cfg,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        feature_names=feature_names,
    )

    residual_models, residual_stats = baseline_mod.train_residual_models(
        cfg=cfg,
        primary_bundle=primary_bundle,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        feature_names=feature_names,
    )

    horizon_bias_map, horizon_bias_stats, horizon_bias_clip_abs = baseline_mod.train_horizon_bias_map(
        cfg=cfg,
        primary_bundle=primary_bundle,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        residual_models=residual_models,
        residual_clip_abs=residual_clip_abs,
    )

    (
        conditional_residual_models,
        conditional_residual_stats,
        conditional_residual_clip_map,
        conditional_residual_gate_meta,
    ) = baseline_mod.train_conditional_residual_models(
        cfg=cfg,
        primary_bundle=primary_bundle,
        x_train=x_train,
        y_train=y_train,
        meta_df=meta_train,
        feature_names=feature_names,
        residual_models=residual_models,
        residual_clip_abs=residual_clip_abs,
        horizon_bias_map=horizon_bias_map,
        horizon_bias_clip_abs=horizon_bias_clip_abs,
    )

    stats: dict[str, object] = {
        "modeling": primary_stats,
        "use_residual": int(use_residual),
        "residual_stats": residual_stats,
        "horizon_bias_stats": horizon_bias_stats,
        "conditional_residual_stats": conditional_residual_stats,
    }

    return BaselineBranchBundle(
        primary_bundle=primary_bundle,
        residual_models=residual_models,
        residual_clip_abs=residual_clip_abs,
        horizon_bias_map=horizon_bias_map,
        horizon_bias_clip_abs=horizon_bias_clip_abs,
        conditional_residual_models=conditional_residual_models,
        conditional_residual_clip_map=conditional_residual_clip_map,
        conditional_residual_gate_meta=conditional_residual_gate_meta,
        stats=stats,
    )


def run_baseline_branch_forecast(
    bundle: BaselineBranchBundle,
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    pred = baseline_mod.run_recursive_forecast(
        history=history,
        schedule=schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        primary_bundle=bundle.primary_bundle,
        default_value=default_value,
        include_calendar=include_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=weather_table,
        weather_defaults_map=weather_defaults_map,
        weather_columns=weather_columns,
        residual_models=bundle.residual_models,
        residual_clip_abs=bundle.residual_clip_abs,
        horizon_bias_map=bundle.horizon_bias_map,
        horizon_bias_clip_abs=bundle.horizon_bias_clip_abs,
        conditional_residual_models=bundle.conditional_residual_models,
        conditional_residual_clip_map=bundle.conditional_residual_clip_map,
        conditional_residual_gate_meta=bundle.conditional_residual_gate_meta,
        actual_map=actual_map,
    )
    return pred.rename(columns={"prediction": "baseline_prediction"})


def fuse_predictions(
    baseline_pred: pd.DataFrame,
    gbdt_full_pred: pd.DataFrame,
    gbdt_target_pred: pd.DataFrame,
    fusion_weights: FusionBundle,
) -> pd.DataFrame:
    keys = ["tollgate_id", "direction", "time_window", "horizon"]
    full_cols = keys + ["gbdt_prediction"] + (["actual"] if "actual" in gbdt_full_pred.columns else [])
    target_cols = keys + ["gbdt_prediction"] + (["actual"] if "actual" in gbdt_target_pred.columns else [])
    merged = baseline_pred.merge(
        gbdt_full_pred[full_cols].rename(columns={"gbdt_prediction": "gbdt_full_prediction"}),
        on=keys,
        how="inner",
        suffixes=("", "_full"),
    )
    merged = merged.merge(
        gbdt_target_pred[target_cols].rename(columns={"gbdt_prediction": "gbdt_target_prediction"}),
        on=keys,
        how="inner",
        suffixes=("", "_target"),
    )

    if "actual" not in merged.columns:
        for c in ("actual_full", "actual_target"):
            if c in merged.columns:
                merged = merged.rename(columns={c: "actual"})
                break

    linear_weight: list[float] = []
    gbdt_full_weight: list[float] = []
    gbdt_target_weight: list[float] = []
    memory_correction: list[float] = []
    memory_applied: list[int] = []
    memory_bucket_type: list[int] = []
    memory_neighbors: list[int] = []
    memory_min_distance: list[float] = []
    prediction_before_memory: list[float] = []
    risk_applied: list[int] = []
    risk_score: list[float] = []
    risk_threshold: list[float] = []
    risk_shrink: list[float] = []
    risk_safe_branch: list[int] = []
    prediction_before_risk: list[float] = []
    fused: list[float] = []
    for row in merged.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        ts = pd.Timestamp(row.time_window)
        w_linear, w_full, w_target = fusion_weights.resolve(
            key,
            h,
            ts,
            linear_prediction=float(row.baseline_prediction),
            gbdt_full_prediction=float(row.gbdt_full_prediction),
            gbdt_target_prediction=float(row.gbdt_target_prediction),
        )
        pred_before_memory = max(
            0.0,
            w_linear * float(row.baseline_prediction)
            + w_full * float(row.gbdt_full_prediction)
            + w_target * float(row.gbdt_target_prediction),
        )
        pred_before_risk = pred_before_memory
        mem_corr = 0.0
        mem_apply = 0
        mem_bucket = 0
        mem_neighbor_count = 0
        mem_min_dist = float("nan")
        if fusion_weights.memory_retrieval is not None and fusion_weights.memory_retrieval.enabled:
            mem_corr, mem_apply, mem_bucket, mem_neighbor_count, mem_min_dist = fusion_weights.memory_retrieval.retrieve_delta(
                key=key,
                horizon=h,
                anchor=int(anchor_bucket(ts)),
                linear_prediction=float(row.baseline_prediction),
                gbdt_full_prediction=float(row.gbdt_full_prediction),
                gbdt_target_prediction=float(row.gbdt_target_prediction),
                base_prediction=pred_before_memory,
            )
            if mem_apply:
                pred_before_risk = max(0.0, float(pred_before_memory) + float(mem_corr))

        pred_final = pred_before_risk
        risk_apply = 0
        risk_s = float("nan")
        risk_th = float("nan")
        risk_alpha = 0.0
        safe_branch = -1
        if fusion_weights.risk_constraint is not None and fusion_weights.risk_constraint.enabled:
            pred_final, risk_apply, risk_s, risk_th, risk_alpha, safe_branch = fusion_weights.risk_constraint.apply(
                key=key,
                horizon=h,
                fused_prediction=pred_before_risk,
                linear_prediction=float(row.baseline_prediction),
                gbdt_full_prediction=float(row.gbdt_full_prediction),
                gbdt_target_prediction=float(row.gbdt_target_prediction),
            )

        linear_weight.append(w_linear)
        gbdt_full_weight.append(w_full)
        gbdt_target_weight.append(w_target)
        memory_correction.append(float(mem_corr))
        memory_applied.append(int(mem_apply))
        memory_bucket_type.append(int(mem_bucket))
        memory_neighbors.append(int(mem_neighbor_count))
        memory_min_distance.append(float(mem_min_dist))
        prediction_before_memory.append(float(pred_before_memory))
        risk_applied.append(int(risk_apply))
        risk_score.append(float(risk_s))
        risk_threshold.append(float(risk_th))
        risk_shrink.append(float(risk_alpha))
        risk_safe_branch.append(int(safe_branch))
        prediction_before_risk.append(float(pred_before_risk))
        fused.append(float(pred_final))

    merged["linear_weight"] = linear_weight
    merged["gbdt_full_weight"] = gbdt_full_weight
    merged["gbdt_target_weight"] = gbdt_target_weight
    merged["prediction_before_memory"] = prediction_before_memory
    merged["memory_retrieval_correction"] = memory_correction
    merged["memory_retrieval_applied"] = memory_applied
    merged["memory_retrieval_bucket_type"] = memory_bucket_type
    merged["memory_retrieval_neighbors"] = memory_neighbors
    merged["memory_retrieval_min_distance"] = memory_min_distance
    merged["prediction_before_risk"] = prediction_before_risk
    merged["risk_constraint_applied"] = risk_applied
    merged["risk_constraint_score"] = risk_score
    merged["risk_constraint_threshold"] = risk_threshold
    merged["risk_constraint_shrink"] = risk_shrink
    merged["risk_constraint_safe_branch"] = risk_safe_branch
    merged["prediction"] = fused
    merged["linear_prediction"] = merged["baseline_prediction"]
    expert_sum = merged["gbdt_full_weight"] + merged["gbdt_target_weight"]
    merged["gbdt_prediction"] = np.where(
        expert_sum > 1e-9,
        (
            merged["gbdt_full_prediction"] * merged["gbdt_full_weight"]
            + merged["gbdt_target_prediction"] * merged["gbdt_target_weight"]
        )
        / expert_sum,
        merged["gbdt_full_prediction"],
    )
    return merged


POST_FUSION_FEATURE_NAMES = [
    "fused_prediction",
    "linear_prediction",
    "gbdt_full_prediction",
    "gbdt_target_prediction",
    "gbdt_prediction",
    "pred_gap_full",
    "pred_gap_target",
    "pred_gap",
    "abs_pred_gap_full",
    "abs_pred_gap_target",
    "abs_pred_gap",
    "gap_ratio_full",
    "gap_ratio_target",
    "gap_ratio",
    "horizon",
    "anchor",
]


def fuse_pair_frame(pair_df: pd.DataFrame, fusion_weights: FusionBundle) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()

    keys = ["tollgate_id", "direction", "time_window", "horizon"]
    base_cols = keys + ["linear_prediction"] + (["actual"] if "actual" in pair_df.columns else [])
    gbdt_full_cols = keys + ["gbdt_full_prediction"] + (["actual"] if "actual" in pair_df.columns else [])
    gbdt_target_cols = keys + ["gbdt_target_prediction"] + (["actual"] if "actual" in pair_df.columns else [])
    baseline_pred = pair_df[base_cols].rename(columns={"linear_prediction": "baseline_prediction"})
    gbdt_full_pred = pair_df[gbdt_full_cols].rename(columns={"gbdt_full_prediction": "gbdt_prediction"}).copy()
    gbdt_target_pred = pair_df[gbdt_target_cols].rename(columns={"gbdt_target_prediction": "gbdt_prediction"}).copy()
    return fuse_predictions(
        baseline_pred=baseline_pred,
        gbdt_full_pred=gbdt_full_pred,
        gbdt_target_pred=gbdt_target_pred,
        fusion_weights=fusion_weights,
    )


def build_post_fusion_feature_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    linear = pred_df["linear_prediction"].astype(float)
    gbdt_full = pred_df["gbdt_full_prediction"].astype(float)
    gbdt_target = pred_df["gbdt_target_prediction"].astype(float)
    gbdt = pred_df["gbdt_prediction"].astype(float)
    fused = pred_df["prediction"].astype(float)
    gap_full = gbdt_full - linear
    gap_target = gbdt_target - linear
    gap = gbdt - linear
    denom = np.maximum(fused.to_numpy(dtype=float), 1.0)
    anchor = pd.to_datetime(pred_df["time_window"]).apply(anchor_bucket).astype(float)

    frame = pd.DataFrame(
        {
            "fused_prediction": fused.to_numpy(dtype=float),
            "linear_prediction": linear.to_numpy(dtype=float),
            "gbdt_full_prediction": gbdt_full.to_numpy(dtype=float),
            "gbdt_target_prediction": gbdt_target.to_numpy(dtype=float),
            "gbdt_prediction": gbdt.to_numpy(dtype=float),
            "pred_gap_full": gap_full.to_numpy(dtype=float),
            "pred_gap_target": gap_target.to_numpy(dtype=float),
            "pred_gap": gap.to_numpy(dtype=float),
            "abs_pred_gap_full": np.abs(gap_full.to_numpy(dtype=float)),
            "abs_pred_gap_target": np.abs(gap_target.to_numpy(dtype=float)),
            "abs_pred_gap": np.abs(gap.to_numpy(dtype=float)),
            "gap_ratio_full": gap_full.to_numpy(dtype=float) / denom,
            "gap_ratio_target": gap_target.to_numpy(dtype=float) / denom,
            "gap_ratio": gap.to_numpy(dtype=float) / denom,
            "horizon": pred_df["horizon"].to_numpy(dtype=float),
            "anchor": anchor.to_numpy(dtype=float),
        },
        index=pred_df.index,
    )
    return frame


def _disabled_post_fusion_bundle(reason: str) -> PostFusionResidualBundle:
    return PostFusionResidualBundle(
        enabled=False,
        feature_names=POST_FUSION_FEATURE_NAMES.copy(),
        models={},
        clip_map={},
        horizon_allowlist={},
        gate_meta={},
        stats={"enabled": 0, "reason": reason},
    )


def _parse_horizon_allowlist(raw: object | None) -> set[int]:
    if raw is None:
        return {1, 2, 3, 4, 5, 6}
    if not isinstance(raw, list):
        raise ValueError("post_fusion_residual.apply_horizons must be a list of horizons")
    out: set[int] = set()
    for item in raw:
        h = int(item)
        if h < 1 or h > 6:
            raise ValueError(f"Invalid horizon in post_fusion_residual.apply_horizons: {h}")
        out.add(h)
    return out if out else {1, 2, 3, 4, 5, 6}


def train_series_expert_bundle_for_days(
    cfg: dict,
    x_gbdt_full_all: pd.DataFrame,
    y_gbdt_full_all: pd.Series,
    meta_gbdt_full_all: pd.DataFrame,
    x_gbdt_target_all: pd.DataFrame,
    y_gbdt_target_all: pd.Series,
    meta_gbdt_target_all: pd.DataFrame,
    feature_names: list[str],
    train_days: list[pd.Timestamp],
    series_cfg_override: dict | None = None,
    expert_tag: str = "series_expert",
) -> SeriesExpertBundle:
    series_cfg = series_cfg_override if series_cfg_override is not None else cfg.get("series_expert", {})
    if not isinstance(series_cfg, dict):
        raise ValueError("series_expert must be an object")
    if not bool(series_cfg.get("use", False)):
        return _disabled_series_expert_bundle("disabled_by_config", expert_tag=expert_tag)

    series_text = str(series_cfg.get("series_key", "1_0"))
    key = baseline_mod.parse_series_key(series_text)
    source_branch = str(series_cfg.get("source_branch", "full")).lower()
    if source_branch not in {"full", "target"}:
        raise ValueError("series_expert.source_branch must be one of: full, target")

    if source_branch == "full":
        x_src_all, y_src_all, meta_src_all = x_gbdt_full_all, y_gbdt_full_all, meta_gbdt_full_all
        default_model_cfg = cfg.get("gbdt_model_full", cfg.get("gbdt_model", {}))
        default_train_cfg = cfg.get("gbdt_training_full", cfg.get("gbdt_training", {}))
    else:
        x_src_all, y_src_all, meta_src_all = x_gbdt_target_all, y_gbdt_target_all, meta_gbdt_target_all
        default_model_cfg = cfg.get("gbdt_model_target", cfg.get("gbdt_model", {}))
        default_train_cfg = cfg.get("gbdt_training_target", cfg.get("gbdt_training", {}))

    x_src, y_src, meta_src = select_by_days(x_src_all, y_src_all, meta_src_all, train_days)
    if x_src.empty:
        return _disabled_series_expert_bundle(f"empty_source_days_{source_branch}", expert_tag=expert_tag)

    apply_horizons = _parse_horizon_allowlist(series_cfg.get("apply_horizons"))
    blend_weight = float(np.clip(float(series_cfg.get("blend_weight", 0.3)), 0.0, 0.95))
    horizon_blend_weight_raw = series_cfg.get("horizon_blend_weight", {})
    if not isinstance(horizon_blend_weight_raw, dict):
        raise ValueError("series_expert.horizon_blend_weight must be an object keyed by horizon")
    horizon_blend_weight: dict[int, float] = {}
    for h_text, w_val in horizon_blend_weight_raw.items():
        h = int(h_text)
        if h < 1 or h > 6:
            raise ValueError(f"Invalid horizon in series_expert.horizon_blend_weight: {h}")
        horizon_blend_weight[h] = float(np.clip(float(w_val), 0.0, 0.95))

    series_mask = (meta_src["tollgate_id"] == int(key[0])) & (meta_src["direction"] == int(key[1]))
    sample_count = int(series_mask.sum())
    min_samples = int(series_cfg.get("min_samples", 120))
    if sample_count < min_samples:
        return SeriesExpertBundle(
            enabled=False,
            series_key=key,
            model_bundle=None,
            horizon_model_bundles={},
            rl_policy_map={},
            rl_delta_bins=[],
            rl_use_anchor=False,
            gate_delta_abs_threshold={},
            apply_horizons=apply_horizons,
            blend_weight=blend_weight,
            horizon_blend_weight=horizon_blend_weight,
            stats={
                "enabled": 0,
                "reason": "insufficient_samples",
                "expert_tag": expert_tag,
                "series_key": series_text,
                "source_branch": source_branch,
                "samples": sample_count,
                "min_samples": min_samples,
            },
        )

    x_sub = x_src.loc[series_mask].reset_index(drop=True)
    y_sub = y_src.loc[series_mask].reset_index(drop=True)
    meta_sub = meta_src.loc[series_mask].reset_index(drop=True)

    model_cfg = series_cfg.get("model", default_model_cfg)
    if not isinstance(model_cfg, dict):
        raise ValueError("series_expert.model must be an object")
    training_cfg = series_cfg.get("training", default_train_cfg)
    if not isinstance(training_cfg, dict):
        raise ValueError("series_expert.training must be an object")

    use_horizon_models = bool(series_cfg.get("use_horizon_models", False))
    min_samples_per_horizon = int(
        series_cfg.get(
            "min_samples_per_horizon",
            max(12, min_samples // max(len(apply_horizons), 1)),
        )
    )
    if min_samples_per_horizon < 1:
        min_samples_per_horizon = 1

    horizon_models_cfg_raw = series_cfg.get("horizon_models", {})
    if not isinstance(horizon_models_cfg_raw, dict):
        raise ValueError("series_expert.horizon_models must be an object keyed by horizon")

    train_fallback_model = bool(series_cfg.get("train_fallback_model", True))
    fallback_model_bundle: GBDTBundle | None = None
    if (not use_horizon_models) or train_fallback_model:
        fallback_model_bundle = train_gbdt_bundle(
            x_train=x_sub,
            y_train=y_sub,
            meta_train=meta_sub,
            feature_names=feature_names,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
        )

    horizon_model_bundles: dict[int, GBDTBundle] = {}
    horizon_sample_stats: dict[int, int] = {}
    if use_horizon_models:
        for h in sorted(apply_horizons):
            h_cfg_raw = horizon_models_cfg_raw.get(str(h), {})
            if h_cfg_raw is None:
                h_cfg_raw = {}
            if not isinstance(h_cfg_raw, dict):
                raise ValueError(f"series_expert.horizon_models['{h}'] must be an object")

            h_model_cfg = h_cfg_raw.get("model", model_cfg)
            if not isinstance(h_model_cfg, dict):
                raise ValueError(f"series_expert.horizon_models['{h}'].model must be an object")
            h_training_cfg = h_cfg_raw.get("training", training_cfg)
            if not isinstance(h_training_cfg, dict):
                raise ValueError(f"series_expert.horizon_models['{h}'].training must be an object")

            h_mask = meta_sub["horizon"] == int(h)
            h_samples = int(h_mask.sum())
            horizon_sample_stats[h] = h_samples
            if h_samples < min_samples_per_horizon:
                continue

            x_h = x_sub.loc[h_mask].reset_index(drop=True)
            y_h = y_sub.loc[h_mask].reset_index(drop=True)
            meta_h = meta_sub.loc[h_mask].reset_index(drop=True)
            horizon_model_bundles[h] = train_gbdt_bundle(
                x_train=x_h,
                y_train=y_h,
                meta_train=meta_h,
                feature_names=feature_names,
                model_cfg=h_model_cfg,
                training_cfg=h_training_cfg,
            )

    if fallback_model_bundle is None and not horizon_model_bundles:
        return SeriesExpertBundle(
            enabled=False,
            series_key=key,
            model_bundle=None,
            horizon_model_bundles={},
            rl_policy_map={},
            rl_delta_bins=[],
            rl_use_anchor=False,
            gate_delta_abs_threshold={},
            apply_horizons=apply_horizons,
            blend_weight=blend_weight,
            horizon_blend_weight=horizon_blend_weight,
            stats={
                "enabled": 0,
                "reason": "insufficient_horizon_samples",
                "expert_tag": expert_tag,
                "series_key": series_text,
                "source_branch": source_branch,
                "samples": sample_count,
                "min_samples": min_samples,
                "min_samples_per_horizon": min_samples_per_horizon,
                "horizon_samples": {str(h): int(n) for h, n in sorted(horizon_sample_stats.items())},
            },
        )

    return SeriesExpertBundle(
        enabled=True,
        series_key=key,
        model_bundle=fallback_model_bundle,
        horizon_model_bundles=horizon_model_bundles,
        rl_policy_map={},
        rl_delta_bins=[],
        rl_use_anchor=False,
        gate_delta_abs_threshold={},
        apply_horizons=apply_horizons,
        blend_weight=blend_weight,
        horizon_blend_weight=horizon_blend_weight,
        stats={
            "enabled": 1,
            "expert_tag": expert_tag,
            "series_key": series_text,
            "source_branch": source_branch,
            "samples": sample_count,
            "min_samples": min_samples,
            "apply_horizons": sorted(apply_horizons),
            "blend_weight": blend_weight,
            "horizon_blend_weight": {str(h): float(w) for h, w in sorted(horizon_blend_weight.items())},
            "use_horizon_models": int(use_horizon_models),
            "min_samples_per_horizon": int(min_samples_per_horizon),
            "train_fallback_model": int(train_fallback_model),
            "horizon_samples": {str(h): int(n) for h, n in sorted(horizon_sample_stats.items())},
            "trained_horizon_models": sorted(int(h) for h in horizon_model_bundles.keys()),
        },
    )


def run_series_expert_recursive_forecast(
    bundle: SeriesExpertBundle,
    history: dict[tuple[int, int], pd.Series],
    schedule: list[pd.Timestamp],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    actual_map: dict[tuple[tuple[int, int], pd.Timestamp], float] | None = None,
) -> pd.DataFrame:
    if not bundle.enabled or bundle.series_key is None:
        return pd.DataFrame(columns=["tollgate_id", "direction", "time_window", "horizon", "series_expert_prediction"])
    if bundle.model_bundle is None and not bundle.horizon_model_bundles:
        return pd.DataFrame(columns=["tollgate_id", "direction", "time_window", "horizon", "series_expert_prediction"])
    if bundle.series_key not in history:
        return pd.DataFrame(columns=["tollgate_id", "direction", "time_window", "horizon", "series_expert_prediction"])

    key = bundle.series_key
    records: list[dict[str, float | int | pd.Timestamp]] = []
    for ts in schedule:
        weather_values: dict[str, float] = {}
        if weather_table is not None and weather_defaults_map is not None:
            weather_values = get_weather_feature_vector(weather_table, ts, weather_defaults_map)
            weather_values = {k: weather_values[k] for k in weather_columns}

        feat = build_feature_row(
            key=key,
            ts=ts,
            history=history,
            series_keys=series_keys,
            cfg=feature_cfg,
            default_value=default_value,
            allow_fallback=True,
            weather=weather_values,
            use_enhanced_features=use_enhanced_features,
        )
        if feat is None:
            continue

        if include_calendar:
            feat.update(calendar_feature_vector(ts))

        if weather_values:
            feat.update(weather_values)

        h = int(horizon_index(ts))
        active_bundle = bundle.horizon_model_bundles.get(h, bundle.model_bundle)
        if active_bundle is None:
            continue

        x = pd.DataFrame([feat], columns=active_bundle.feature_names)
        pred = predict_gbdt(active_bundle, x, ts)

        history[key].loc[ts] = pred
        history[key] = history[key].sort_index()

        rec: dict[str, float | int | pd.Timestamp] = {
            "tollgate_id": int(key[0]),
            "direction": int(key[1]),
            "time_window": ts,
            "horizon": h,
            "series_expert_prediction": pred,
        }
        if actual_map is not None and (key, ts) in actual_map:
            rec["actual"] = float(actual_map[(key, ts)])
        records.append(rec)

    return pd.DataFrame(records)


def _series_expert_delta_bucket(delta_ratio: float, bins: list[float]) -> int:
    if not bins:
        return 0
    arr = np.array(sorted(float(x) for x in bins), dtype=float)
    return int(np.digitize([float(delta_ratio)], arr, right=False)[0])


def build_series_expert_rl_experience(
    pred_df: pd.DataFrame,
    expert_pred_df: pd.DataFrame,
    bundle: SeriesExpertBundle,
    base_prediction_col: str = "prediction",
) -> pd.DataFrame:
    if pred_df.empty or expert_pred_df.empty or not bundle.enabled or bundle.series_key is None:
        return pd.DataFrame(
            columns=["horizon", "anchor", "delta_ratio", "base_prediction", "expert_prediction", "actual"]
        )

    keys = ["tollgate_id", "direction", "time_window", "horizon"]
    expert_cols = keys + ["series_expert_prediction"]
    base_df = pred_df.copy()
    if "series_expert_prediction" in base_df.columns:
        base_df = base_df.drop(columns=["series_expert_prediction"])
    merged = base_df.merge(expert_pred_df[expert_cols], on=keys, how="left", suffixes=("", "_expert"))
    if "series_expert_prediction_expert" in merged.columns:
        merged["series_expert_prediction"] = merged["series_expert_prediction_expert"]
        merged = merged.drop(columns=["series_expert_prediction_expert"])
    target_key = bundle.series_key
    if "actual" not in merged.columns:
        return pd.DataFrame(
            columns=["horizon", "anchor", "delta_ratio", "base_prediction", "expert_prediction", "actual"]
        )

    base_col = str(base_prediction_col)
    if base_col not in merged.columns:
        base_col = "prediction"
    if base_col not in merged.columns:
        return pd.DataFrame(
            columns=["horizon", "anchor", "delta_ratio", "base_prediction", "expert_prediction", "actual"]
        )
    filt = (
        (merged["tollgate_id"].astype(int) == int(target_key[0]))
        & (merged["direction"].astype(int) == int(target_key[1]))
        & (merged["horizon"].astype(int).isin(bundle.apply_horizons))
        & merged["actual"].notna()
        & merged["series_expert_prediction"].notna()
        & merged[base_col].notna()
    )
    if not bool(filt.any()):
        return pd.DataFrame(
            columns=["horizon", "anchor", "delta_ratio", "base_prediction", "expert_prediction", "actual"]
        )

    sub = merged.loc[filt].copy()
    base = sub[base_col].astype(float).to_numpy()
    expert = sub["series_expert_prediction"].astype(float).to_numpy()
    denom = np.maximum(np.abs(base), 1.0)
    out = pd.DataFrame(
        {
            "horizon": sub["horizon"].astype(int).to_numpy(),
            "anchor": pd.to_datetime(sub["time_window"]).apply(anchor_bucket).astype(int).to_numpy(),
            "delta_ratio": (expert - base) / denom,
            "base_prediction": base,
            "expert_prediction": expert,
            "actual": sub["actual"].astype(float).to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def train_series_expert_rl_policy(
    series_cfg: dict,
    bundle: SeriesExpertBundle,
    experience_df: pd.DataFrame | None,
) -> SeriesExpertBundle:
    if not isinstance(series_cfg, dict):
        return bundle
    rl_cfg = series_cfg.get("rl", {})
    if not isinstance(rl_cfg, dict) or not bool(rl_cfg.get("use", False)):
        bundle.stats["rl_enabled"] = 0
        bundle.stats["rl_experience_rows"] = int(len(experience_df)) if experience_df is not None else 0
        return bundle
    if not bundle.enabled or bundle.series_key is None:
        bundle.stats["rl_enabled"] = 0
        bundle.stats["rl_reason"] = "expert_disabled"
        bundle.stats["rl_experience_rows"] = int(len(experience_df)) if experience_df is not None else 0
        return bundle
    if experience_df is None or experience_df.empty:
        bundle.stats["rl_enabled"] = 0
        bundle.stats["rl_reason"] = "empty_experience"
        bundle.stats["rl_experience_rows"] = 0
        return bundle

    weight_candidates_raw = rl_cfg.get("weight_candidates", [0.0, 0.15, 0.3, 0.45, 0.6])
    if not isinstance(weight_candidates_raw, list) or not weight_candidates_raw:
        raise ValueError("series_expert.rl.weight_candidates must be a non-empty list")
    weight_candidates = sorted(float(np.clip(float(w), 0.0, 0.95)) for w in weight_candidates_raw)
    delta_bins_raw = rl_cfg.get("delta_bins", [-0.2, -0.05, 0.05, 0.2])
    if not isinstance(delta_bins_raw, list):
        raise ValueError("series_expert.rl.delta_bins must be a list")
    delta_bins = sorted(float(x) for x in delta_bins_raw)
    use_anchor = bool(rl_cfg.get("use_anchor", False))
    min_samples_per_state = int(rl_cfg.get("min_samples_per_state", 4))
    if min_samples_per_state < 1:
        min_samples_per_state = 1

    rewards_sum: dict[tuple[int, int, int, int], float] = {}
    rewards_cnt: dict[tuple[int, int, int, int], int] = {}
    state_cnt: dict[tuple[int, int, int], int] = {}

    for row in experience_df.itertuples(index=False):
        h = int(row.horizon)
        if h not in bundle.apply_horizons:
            continue
        anchor = int(row.anchor) if use_anchor else -1
        delta_bucket = _series_expert_delta_bucket(float(row.delta_ratio), delta_bins)
        state_key = (h, anchor, delta_bucket)
        state_cnt[state_key] = state_cnt.get(state_key, 0) + 1
        base_pred = float(row.base_prediction)
        expert_pred = float(row.expert_prediction)
        actual = float(row.actual)
        denom = max(abs(actual), 1.0)
        for a_idx, w in enumerate(weight_candidates):
            pred = max(0.0, (1.0 - w) * base_pred + w * expert_pred)
            reward = -abs(actual - pred) / denom
            k = (h, anchor, delta_bucket, a_idx)
            rewards_sum[k] = rewards_sum.get(k, 0.0) + reward
            rewards_cnt[k] = rewards_cnt.get(k, 0) + 1

    policy_map: dict[tuple[int, int, int], float] = {}
    for state_key, n in state_cnt.items():
        if n < min_samples_per_state:
            continue
        h, anchor, delta_bucket = state_key
        best_w = bundle.horizon_blend_weight.get(h, bundle.blend_weight)
        best_reward = -1e18
        for a_idx, w in enumerate(weight_candidates):
            k = (h, anchor, delta_bucket, a_idx)
            cnt = rewards_cnt.get(k, 0)
            if cnt <= 0:
                continue
            mean_reward = rewards_sum[k] / cnt
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_w = w
        policy_map[state_key] = float(best_w)

    bundle.rl_policy_map = policy_map
    bundle.rl_delta_bins = delta_bins
    bundle.rl_use_anchor = use_anchor
    bundle.stats["rl_enabled"] = int(bool(policy_map))
    bundle.stats["rl_experience_rows"] = int(len(experience_df))
    bundle.stats["rl_states"] = int(len(policy_map))
    bundle.stats["rl_use_anchor"] = int(use_anchor)
    bundle.stats["rl_min_samples_per_state"] = int(min_samples_per_state)
    bundle.stats["rl_weight_candidates"] = [float(w) for w in weight_candidates]
    bundle.stats["rl_delta_bins"] = [float(x) for x in delta_bins]
    if not policy_map:
        bundle.stats["rl_reason"] = "no_state_passed_min_samples"
    return bundle


def _resolve_series_expert_weight(
    bundle: SeriesExpertBundle,
    horizon: int,
    anchor: int,
    delta_ratio: float,
) -> float:
    w = float(np.clip(bundle.horizon_blend_weight.get(horizon, bundle.blend_weight), 0.0, 0.95))
    if bundle.rl_policy_map:
        anchor_key = int(anchor) if bundle.rl_use_anchor else -1
        delta_bucket = _series_expert_delta_bucket(float(delta_ratio), bundle.rl_delta_bins)
        w = float(bundle.rl_policy_map.get((int(horizon), int(anchor_key), int(delta_bucket)), w))
        w = float(np.clip(w, 0.0, 0.95))
    return w


def train_series_expert_gate_policy(
    series_cfg: dict,
    bundle: SeriesExpertBundle,
    experience_df: pd.DataFrame | None,
) -> SeriesExpertBundle:
    if not isinstance(series_cfg, dict):
        return bundle

    gate_cfg = series_cfg.get("gate", {})
    if not isinstance(gate_cfg, dict) or not bool(gate_cfg.get("use", False)):
        bundle.gate_delta_abs_threshold = {}
        bundle.stats["gate_enabled"] = 0
        bundle.stats["gate_experience_rows"] = int(len(experience_df)) if experience_df is not None else 0
        return bundle

    if not bundle.enabled or bundle.series_key is None:
        bundle.gate_delta_abs_threshold = {}
        bundle.stats["gate_enabled"] = 0
        bundle.stats["gate_reason"] = "expert_disabled"
        bundle.stats["gate_experience_rows"] = int(len(experience_df)) if experience_df is not None else 0
        return bundle

    if experience_df is None or experience_df.empty:
        bundle.gate_delta_abs_threshold = {}
        bundle.stats["gate_enabled"] = 0
        bundle.stats["gate_reason"] = "empty_experience"
        bundle.stats["gate_experience_rows"] = 0
        return bundle

    quantiles_raw = gate_cfg.get("quantiles", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    if not isinstance(quantiles_raw, list) or not quantiles_raw:
        raise ValueError("series_expert.gate.quantiles must be a non-empty list")
    quantiles = sorted(float(np.clip(float(q), 0.0, 1.0)) for q in quantiles_raw)

    min_samples = int(gate_cfg.get("min_samples", 6))
    if min_samples < 1:
        min_samples = 1
    min_mean_gain = float(gate_cfg.get("min_mean_gain", 0.0))
    min_expected_gain = float(gate_cfg.get("min_expected_gain", 0.0))

    threshold_map: dict[int, float] = {}
    horizon_stats: dict[str, dict[str, object]] = {}

    for h in sorted(int(x) for x in bundle.apply_horizons):
        sub = experience_df.loc[experience_df["horizon"].astype(int) == int(h)].copy()
        if sub.empty:
            continue

        base = sub["base_prediction"].astype(float).to_numpy()
        expert = sub["expert_prediction"].astype(float).to_numpy()
        actual = sub["actual"].astype(float).to_numpy()
        anchor = sub["anchor"].astype(int).to_numpy()
        delta_ratio = sub["delta_ratio"].astype(float).to_numpy()
        abs_delta = np.abs(delta_ratio)

        denom = np.maximum(np.abs(actual), 1.0)
        base_ape = np.abs(actual - base) / denom

        applied_ape = np.zeros(len(sub), dtype=float)
        for i in range(len(sub)):
            w = _resolve_series_expert_weight(
                bundle=bundle,
                horizon=int(h),
                anchor=int(anchor[i]),
                delta_ratio=float(delta_ratio[i]),
            )
            pred = max(0.0, (1.0 - w) * float(base[i]) + w * float(expert[i]))
            applied_ape[i] = abs(float(actual[i]) - pred) / max(abs(float(actual[i])), 1.0)
        gain = base_ape - applied_ape

        best_threshold: float | None = None
        best_objective = -1e18
        best_mean_gain = 0.0
        best_count = 0
        n_rows = max(len(sub), 1)
        for q in quantiles:
            threshold = float(np.quantile(abs_delta, q))
            mask = abs_delta >= threshold
            count = int(mask.sum())
            if count < min_samples:
                continue
            mean_gain = float(np.mean(gain[mask]))
            expected_gain = float(np.sum(gain[mask]) / n_rows)
            if expected_gain > best_objective:
                best_objective = expected_gain
                best_threshold = threshold
                best_mean_gain = mean_gain
                best_count = count

        h_stat = {
            "rows": int(len(sub)),
            "min_samples": int(min_samples),
            "selected_rows": int(best_count),
            "selected_ratio": float(best_count / n_rows),
            "best_expected_gain": float(best_objective if best_objective > -1e17 else 0.0),
            "best_mean_gain": float(best_mean_gain),
        }
        if (
            best_threshold is not None
            and best_count >= min_samples
            and best_mean_gain >= min_mean_gain
            and best_objective >= min_expected_gain
        ):
            threshold_map[int(h)] = float(best_threshold)
            h_stat["threshold"] = float(best_threshold)
            h_stat["enabled"] = 1
        else:
            h_stat["enabled"] = 0
            h_stat["reason"] = "no_positive_gate"
        horizon_stats[str(h)] = h_stat

    bundle.gate_delta_abs_threshold = threshold_map
    bundle.stats["gate_enabled"] = int(bool(threshold_map))
    bundle.stats["gate_experience_rows"] = int(len(experience_df))
    bundle.stats["gate_min_samples"] = int(min_samples)
    bundle.stats["gate_min_mean_gain"] = float(min_mean_gain)
    bundle.stats["gate_min_expected_gain"] = float(min_expected_gain)
    bundle.stats["gate_quantiles"] = [float(q) for q in quantiles]
    bundle.stats["gate_thresholds"] = {str(h): float(v) for h, v in sorted(threshold_map.items())}
    bundle.stats["gate_horizon_stats"] = horizon_stats
    if not threshold_map:
        bundle.stats["gate_reason"] = "no_horizon_passed_threshold"
    return bundle


def apply_series_expert_adjustment(
    pred_df: pd.DataFrame,
    expert_pred_df: pd.DataFrame,
    bundle: SeriesExpertBundle,
    expert_tag: str | None = None,
) -> pd.DataFrame:
    out = pred_df.copy()
    if out.empty:
        return out

    if "prediction_before_series_expert" not in out.columns:
        out["prediction_before_series_expert"] = out["prediction"].astype(float)
    if "series_expert_applied" not in out.columns:
        out["series_expert_applied"] = 0
    if "series_expert_weight" not in out.columns:
        out["series_expert_weight"] = 0.0
    if "series_expert_prediction" not in out.columns:
        out["series_expert_prediction"] = np.nan
    if "series_expert_applied_count" not in out.columns:
        out["series_expert_applied_count"] = 0

    tag = _normalize_expert_tag(expert_tag)
    per_applied_col = f"series_expert_applied_{tag}"
    per_weight_col = f"series_expert_weight_{tag}"
    per_pred_col = f"series_expert_prediction_{tag}"
    if per_applied_col not in out.columns:
        out[per_applied_col] = 0
    if per_weight_col not in out.columns:
        out[per_weight_col] = 0.0
    if per_pred_col not in out.columns:
        out[per_pred_col] = np.nan

    if out.empty or not bundle.enabled or bundle.series_key is None or expert_pred_df.empty:
        return out

    keys = ["tollgate_id", "direction", "time_window", "horizon"]
    expert_cols = keys + ["series_expert_prediction"]
    merged = out.merge(expert_pred_df[expert_cols], on=keys, how="left", suffixes=("", "_expert"))
    if "series_expert_prediction_expert" in merged.columns:
        merged["series_expert_prediction"] = merged["series_expert_prediction_expert"]
        merged = merged.drop(columns=["series_expert_prediction_expert"])

    applied_any: list[int] = []
    applied_cnt: list[int] = []
    weights: list[float] = []
    preds: list[float] = []
    per_applied: list[int] = []
    per_weights: list[float] = []
    per_preds: list[float] = []
    target_key = bundle.series_key
    for row in merged.itertuples(index=False):
        pred = float(row.prediction)
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        exp_pred = float(row.series_expert_prediction) if pd.notna(row.series_expert_prediction) else np.nan
        current_any = int(getattr(row, "series_expert_applied", 0))
        current_count = int(getattr(row, "series_expert_applied_count", 0))
        use_exp = (
            key == target_key
            and h in bundle.apply_horizons
            and np.isfinite(exp_pred)
        )
        if use_exp:
            anchor = anchor_bucket(pd.Timestamp(row.time_window))
            delta_ratio = (exp_pred - pred) / max(abs(pred), 1.0)
            gate_threshold = bundle.gate_delta_abs_threshold.get(int(h))
            if gate_threshold is not None and abs(float(delta_ratio)) < float(gate_threshold):
                use_exp = False
        if use_exp:
            w = _resolve_series_expert_weight(
                bundle=bundle,
                horizon=int(h),
                anchor=int(anchor),
                delta_ratio=float(delta_ratio),
            )
            pred = max(0.0, (1.0 - w) * pred + w * exp_pred)
            current_any = 1
            current_count += 1
            per_applied.append(1)
            per_weights.append(w)
            per_preds.append(exp_pred)
            weights.append(w)
        else:
            per_applied.append(0)
            per_weights.append(0.0)
            per_preds.append(np.nan)
            weights.append(0.0)
        applied_any.append(current_any)
        applied_cnt.append(current_count)
        preds.append(pred)

    merged["series_expert_applied"] = applied_any
    merged["series_expert_applied_count"] = applied_cnt
    merged["series_expert_weight"] = weights
    merged["prediction"] = preds
    merged[per_applied_col] = per_applied
    merged[per_weight_col] = per_weights
    merged[per_pred_col] = per_preds
    return merged


def train_post_fusion_residual_bundle(
    cfg: dict,
    oof_pair_frame: pd.DataFrame | None,
    fusion_weights: FusionBundle,
) -> PostFusionResidualBundle:
    post_cfg = cfg.get("post_fusion_residual", {})
    use_post = bool(post_cfg.get("use", False))
    if not use_post:
        return _disabled_post_fusion_bundle("disabled_by_config")
    if oof_pair_frame is None or oof_pair_frame.empty:
        return _disabled_post_fusion_bundle("missing_oof_frame")

    target_series = post_cfg.get("target_series", [])
    if not target_series:
        return _disabled_post_fusion_bundle("empty_target_series")

    default_alpha = float(post_cfg.get("ridge_alpha", 120.0))
    default_min_samples = int(post_cfg.get("min_samples", 30))
    default_clip_abs = float(post_cfg.get("clip_abs", 6.0))
    default_use_gate = bool(post_cfg.get("use_confidence_gate", True))
    default_gate_quantile = float(post_cfg.get("gate_quantile", 0.7))
    default_gate_max_z_raw = post_cfg.get("gate_max_z")
    default_gate_max_z = float(default_gate_max_z_raw) if default_gate_max_z_raw is not None else None
    default_use_gain_gate = bool(post_cfg.get("use_gain_gate", True))
    default_gain_min_mean = float(post_cfg.get("gain_min_mean", 0.0))
    default_gain_min_count = int(post_cfg.get("gain_min_count", 8))
    default_gain_quantiles = post_cfg.get(
        "gain_quantiles",
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    if not isinstance(default_gain_quantiles, list) or not default_gain_quantiles:
        raise ValueError("post_fusion_residual.gain_quantiles must be a non-empty list")
    default_gain_quantiles = [float(q) for q in default_gain_quantiles]

    apply_horizon_cfg = post_cfg.get("apply_horizons", {})
    if not isinstance(apply_horizon_cfg, dict):
        raise ValueError("post_fusion_residual.apply_horizons must be dict keyed by series")
    series_params_cfg = post_cfg.get("series_params", {})
    if not isinstance(series_params_cfg, dict):
        raise ValueError("post_fusion_residual.series_params must be dict keyed by series")

    fused_oof = fuse_pair_frame(oof_pair_frame, fusion_weights)
    if fused_oof.empty:
        return _disabled_post_fusion_bundle("empty_fused_oof")

    feat_df = build_post_fusion_feature_frame(fused_oof)
    residual = fused_oof["actual"].to_numpy(dtype=float) - fused_oof["prediction"].to_numpy(dtype=float)

    models: dict[tuple[int, int], baseline_mod.RidgeLinearModel] = {}
    clip_map: dict[tuple[int, int], float] = {}
    horizon_allowlist: dict[tuple[int, int], set[int]] = {}
    gate_meta: dict[tuple[int, int], dict[str, object]] = {}
    stats: dict[str, dict[str, float | int | object]] = {}

    for series_text in target_series:
        key = baseline_mod.parse_series_key(series_text)
        params = series_params_cfg.get(series_text, {})
        if not isinstance(params, dict):
            raise ValueError(f"post_fusion_residual.series_params['{series_text}'] must be object")

        alpha = float(params.get("ridge_alpha", default_alpha))
        min_samples = int(params.get("min_samples", default_min_samples))
        clip_abs = float(params.get("clip_abs", default_clip_abs))
        use_gate = bool(params.get("use_confidence_gate", default_use_gate))
        gate_quantile = float(params.get("gate_quantile", default_gate_quantile))
        gate_max_z_raw = params.get("gate_max_z", default_gate_max_z)
        gate_max_z = float(gate_max_z_raw) if gate_max_z_raw is not None else None
        use_gain_gate = bool(params.get("use_gain_gate", default_use_gain_gate))
        gain_min_mean = float(params.get("gain_min_mean", default_gain_min_mean))
        gain_min_count = int(params.get("gain_min_count", default_gain_min_count))
        gain_quantiles_raw = params.get("gain_quantiles", default_gain_quantiles)
        if not isinstance(gain_quantiles_raw, list) or not gain_quantiles_raw:
            raise ValueError(f"post_fusion_residual.gain_quantiles for {series_text} must be non-empty list")
        gain_quantiles = [float(q) for q in gain_quantiles_raw]

        allow_raw = params.get("apply_horizons", apply_horizon_cfg.get(series_text))
        allow_horizons = _parse_horizon_allowlist(allow_raw)

        mask = (
            (fused_oof["tollgate_id"] == key[0])
            & (fused_oof["direction"] == key[1])
            & (fused_oof["horizon"].isin(list(allow_horizons)))
        )
        sample_count = int(mask.sum())
        if sample_count < min_samples:
            stats[series_text] = {
                "samples": sample_count,
                "trained": 0,
                "min_samples": min_samples,
                "apply_horizons": sorted(allow_horizons),
                "reason": f"insufficient_samples_lt_{min_samples}",
            }
            continue

        x_sub = feat_df.loc[mask].reset_index(drop=True)
        y_sub = pd.Series(residual[mask.to_numpy()], name="post_fusion_residual")

        model = baseline_mod.RidgeLinearModel(feature_names=POST_FUSION_FEATURE_NAMES, alpha=alpha)
        model.fit(x_sub, y_sub)
        models[key] = model
        clip_map[key] = clip_abs
        horizon_allowlist[key] = allow_horizons

        pred_sub = model.predict(x_sub).astype(float)
        pred_sub = np.clip(pred_sub, -clip_abs, clip_abs)
        gain_arr = np.abs(y_sub.to_numpy(dtype=float)) - np.abs(y_sub.to_numpy(dtype=float) - pred_sub)
        pred_abs = np.abs(pred_sub)

        x_np = x_sub[POST_FUSION_FEATURE_NAMES].to_numpy(dtype=float)
        center = np.mean(x_np, axis=0)
        scale = np.std(x_np, axis=0, ddof=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        dists = np.sqrt(np.mean(((x_np - center) / scale) ** 2, axis=1))
        dist_q = float(np.quantile(dists, gate_quantile))
        threshold = dist_q if gate_max_z is None else min(dist_q, gate_max_z)
        if not use_gate:
            threshold = float("inf")

        gain_threshold = 0.0
        gain_best_quantile = 0.0
        gain_best_mean = float(np.mean(gain_arr))
        if use_gain_gate:
            best: tuple[float, float, float] | None = None  # (mean_gain, quantile, threshold)
            for q in gain_quantiles:
                q_clamped = min(1.0, max(0.0, q))
                th = float(np.quantile(pred_abs, q_clamped))
                idx = pred_abs >= th
                cnt = int(np.sum(idx))
                if cnt < gain_min_count:
                    continue
                mean_gain = float(np.mean(gain_arr[idx]))
                if best is None or mean_gain > best[0]:
                    best = (mean_gain, q_clamped, th)
            if best is None or best[0] <= gain_min_mean:
                gain_threshold = float("inf")
                gain_best_mean = best[0] if best is not None else float("-inf")
                gain_best_quantile = best[1] if best is not None else 1.0
            else:
                gain_best_mean, gain_best_quantile, gain_threshold = best

        gate_meta[key] = {
            "center": center,
            "scale": scale,
            "threshold": float(threshold),
            "gain_abs_threshold": float(gain_threshold),
        }

        stats[series_text] = {
            "samples": sample_count,
            "trained": 1,
            "alpha": alpha,
            "clip_abs": clip_abs,
            "apply_horizons": sorted(allow_horizons),
            "gate_enabled": int(use_gate),
            "gate_quantile": gate_quantile,
            "gate_max_z": gate_max_z if gate_max_z is not None else -1.0,
            "gate_threshold": float(threshold),
            "gain_gate_enabled": int(use_gain_gate),
            "gain_best_quantile": gain_best_quantile,
            "gain_best_mean": gain_best_mean,
            "gain_abs_threshold": float(gain_threshold),
            "residual_mean": float(y_sub.mean()),
            "residual_std": float(y_sub.std(ddof=0)),
        }

    return PostFusionResidualBundle(
        enabled=True,
        feature_names=POST_FUSION_FEATURE_NAMES.copy(),
        models=models,
        clip_map=clip_map,
        horizon_allowlist=horizon_allowlist,
        gate_meta=gate_meta,
        stats={
            "enabled": 1,
            "oof_rows": int(len(fused_oof)),
            "trained_series_count": int(len(models)),
            "series_stats": stats,
        },
    )


def apply_post_fusion_residual_adjustment(
    pred_df: pd.DataFrame,
    post_bundle: PostFusionResidualBundle,
) -> pd.DataFrame:
    out = pred_df.copy()
    if out.empty:
        return out

    correction: list[float] = []
    gate_applied: list[int] = []
    conf_gate_applied: list[int] = []
    gain_gate_applied: list[int] = []
    gate_distance: list[float] = []
    gate_threshold: list[float] = []
    gain_threshold: list[float] = []
    prediction_before_post = out["prediction"].astype(float).tolist()

    for row in out.itertuples(index=False):
        key = (int(row.tollgate_id), int(row.direction))
        h = int(row.horizon)
        corr = 0.0
        applied = 0
        conf_ok = 0
        gain_ok = 0
        dist_val = float("nan")
        dist_th = float("nan")
        gain_th = float("nan")

        if post_bundle.enabled and key in post_bundle.models and h in post_bundle.horizon_allowlist.get(key, set()):
            applied = 1
            g_full = float(row.gbdt_full_prediction)
            g_target = float(row.gbdt_target_prediction)
            g_mix = float(row.gbdt_prediction)
            linear = float(row.linear_prediction)
            fused = float(row.prediction)
            feat_map = {
                "fused_prediction": fused,
                "linear_prediction": linear,
                "gbdt_full_prediction": g_full,
                "gbdt_target_prediction": g_target,
                "gbdt_prediction": g_mix,
                "pred_gap_full": g_full - linear,
                "pred_gap_target": g_target - linear,
                "pred_gap": g_mix - linear,
                "abs_pred_gap_full": abs(g_full - linear),
                "abs_pred_gap_target": abs(g_target - linear),
                "abs_pred_gap": abs(g_mix - linear),
                "gap_ratio_full": (g_full - linear) / max(fused, 1.0),
                "gap_ratio_target": (g_target - linear) / max(fused, 1.0),
                "gap_ratio": (g_mix - linear) / max(fused, 1.0),
                "horizon": float(h),
                "anchor": float(anchor_bucket(pd.Timestamp(row.time_window))),
            }
            feat_df = pd.DataFrame([feat_map], columns=post_bundle.feature_names)
            can_apply = True
            gate = post_bundle.gate_meta.get(key)
            if gate is not None:
                center = gate["center"]  # type: ignore[assignment]
                scale = gate["scale"]  # type: ignore[assignment]
                dist_th = float(gate["threshold"])
                gain_th = float(gate.get("gain_abs_threshold", 0.0))
                if np.isfinite(dist_th):
                    x_np = feat_df.iloc[0].to_numpy(dtype=float)
                    dist_val = float(np.sqrt(np.mean(((x_np - center) / scale) ** 2)))
                    can_apply = dist_val <= dist_th
                if can_apply:
                    conf_ok = 1

            if can_apply:
                pred_corr = float(post_bundle.models[key].predict(feat_df)[0])
                clip_abs = post_bundle.clip_map.get(key)
                if clip_abs is not None:
                    pred_corr = float(np.clip(pred_corr, -clip_abs, clip_abs))

                if np.isfinite(gain_th):
                    can_apply = abs(pred_corr) >= gain_th
                if can_apply:
                    gain_ok = 1
                    corr = pred_corr

        correction.append(corr)
        gate_applied.append(applied)
        conf_gate_applied.append(conf_ok)
        gain_gate_applied.append(gain_ok)
        gate_distance.append(dist_val)
        gate_threshold.append(dist_th)
        gain_threshold.append(gain_th)

    out["prediction_before_post_fusion"] = prediction_before_post
    out["post_fusion_residual_correction"] = correction
    out["post_fusion_gate_applied"] = gate_applied
    out["post_fusion_conf_gate_applied"] = conf_gate_applied
    out["post_fusion_gain_gate_applied"] = gain_gate_applied
    out["post_fusion_gate_distance"] = gate_distance
    out["post_fusion_gate_threshold"] = gate_threshold
    out["post_fusion_gain_threshold"] = gain_threshold
    out["prediction"] = np.clip(out["prediction_before_post_fusion"] + out["post_fusion_residual_correction"], 0.0, None)
    return out


def train_with_adaptive_fusion(
    cfg: dict,
    x_all: pd.DataFrame,
    y_all: pd.Series,
    meta_all: pd.DataFrame,
    x_gbdt_full_all: pd.DataFrame,
    y_gbdt_full_all: pd.Series,
    meta_gbdt_full_all: pd.DataFrame,
    x_gbdt_target_all: pd.DataFrame,
    y_gbdt_target_all: pd.Series,
    meta_gbdt_target_all: pd.DataFrame,
    train_history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    feature_cfg: FeatureConfig,
    feature_names: list[str],
    default_value: float,
    include_calendar: bool,
    use_enhanced_features: bool,
    weather_table: pd.DataFrame | None,
    weather_defaults_map: dict[str, float] | None,
    weather_columns: list[str],
    train_days: list[pd.Timestamp],
    horizon_windows: int,
) -> tuple[
    BaselineBranchBundle,
    GBDTBundle,
    GBDTBundle,
    FusionBundle,
    dict[str, object],
    pd.DataFrame | None,
]:
    # return: baseline branch, full-window gbdt branch, target-window gbdt branch, fusion bundle, adaptation stats
    fusion_cfg = cfg.get("fusion", {})
    gbdt_full_model_cfg = cfg.get("gbdt_model_full", cfg.get("gbdt_model", {}))
    gbdt_target_model_cfg = cfg.get("gbdt_model_target", cfg.get("gbdt_model", {}))
    gbdt_full_training_cfg = cfg.get("gbdt_training_full", cfg.get("gbdt_training", {}))
    gbdt_target_training_cfg = cfg.get("gbdt_training_target")
    if gbdt_target_training_cfg is None:
        gbdt_target_training_cfg = dict(cfg.get("gbdt_training", {}))
        gbdt_target_training_cfg["target_only"] = True

    weight_learning = str(fusion_cfg.get("weight_learning", "tail_window"))
    adapt_days = int(fusion_cfg.get("adapt_days", 4))
    min_core_days = int(fusion_cfg.get("min_core_days", 8))

    if len(train_days) == 0:
        raise RuntimeError("No train days provided")

    use_adapt = False
    adaptation_stats: dict[str, object] = {
        "use_adaptation": 0,
        "weight_learning": weight_learning,
        "adapt_days": 0,
        "core_days": int(len(train_days)),
        "reason": "adaptation_disabled",
    }
    oof_pair_frame: pd.DataFrame | None = None

    def collect_pair_predictions(
        fit_days: list[pd.Timestamp],
        eval_days: list[pd.Timestamp],
    ) -> pd.DataFrame:
        x_fit, y_fit, meta_fit = select_by_days(x_all, y_all, meta_all, fit_days)
        if x_fit.empty:
            return pd.DataFrame()

        fit_baseline = train_baseline_branch(
            cfg=cfg,
            x_train=x_fit,
            y_train=y_fit,
            meta_train=meta_fit,
            feature_names=feature_names,
        )
        x_gfull_fit, y_gfull_fit, meta_gfull_fit = select_by_days(
            x_gbdt_full_all,
            y_gbdt_full_all,
            meta_gbdt_full_all,
            fit_days,
        )
        if x_gfull_fit.empty:
            return pd.DataFrame()
        fit_gbdt_full = train_gbdt_bundle(
            x_train=x_gfull_fit,
            y_train=y_gfull_fit,
            meta_train=meta_gfull_fit,
            feature_names=feature_names,
            model_cfg=gbdt_full_model_cfg,
            training_cfg=gbdt_full_training_cfg,
        )
        x_gtarget_fit, y_gtarget_fit, meta_gtarget_fit = select_by_days(
            x_gbdt_target_all,
            y_gbdt_target_all,
            meta_gbdt_target_all,
            fit_days,
        )
        if x_gtarget_fit.empty:
            return pd.DataFrame()
        fit_gbdt_target = train_gbdt_bundle(
            x_train=x_gtarget_fit,
            y_train=y_gtarget_fit,
            meta_train=meta_gtarget_fit,
            feature_names=feature_names,
            model_cfg=gbdt_target_model_cfg,
            training_cfg=gbdt_target_training_cfg,
        )

        eval_schedule = target_windows_for_days(eval_days, horizon=horizon_windows)
        base_history, actual_map = prepare_history_for_schedule(train_history, series_keys, eval_schedule)
        gbdt_full_history, _ = prepare_history_for_schedule(train_history, series_keys, eval_schedule)
        gbdt_target_history, _ = prepare_history_for_schedule(train_history, series_keys, eval_schedule)

        eval_baseline = run_baseline_branch_forecast(
            bundle=fit_baseline,
            history=base_history,
            schedule=eval_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=include_calendar,
            use_enhanced_features=use_enhanced_features,
            weather_table=weather_table,
            weather_defaults_map=weather_defaults_map,
            weather_columns=weather_columns,
            actual_map=actual_map,
        )
        eval_gbdt_full = run_gbdt_recursive_forecast(
            bundle=fit_gbdt_full,
            history=gbdt_full_history,
            schedule=eval_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=include_calendar,
            use_enhanced_features=use_enhanced_features,
            weather_table=weather_table,
            weather_defaults_map=weather_defaults_map,
            weather_columns=weather_columns,
            actual_map=actual_map,
        )
        eval_gbdt_target = run_gbdt_recursive_forecast(
            bundle=fit_gbdt_target,
            history=gbdt_target_history,
            schedule=eval_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=include_calendar,
            use_enhanced_features=use_enhanced_features,
            weather_table=weather_table,
            weather_defaults_map=weather_defaults_map,
            weather_columns=weather_columns,
            actual_map=actual_map,
        )
        merged = fuse_predictions(
            baseline_pred=eval_baseline,
            gbdt_full_pred=eval_gbdt_full,
            gbdt_target_pred=eval_gbdt_target,
            fusion_weights=default_fusion_weights(cfg),
        )
        merged = merged.dropna(subset=["actual"]).reset_index(drop=True)
        return merged

    if weight_learning == "rolling_oof":
        oof_n_folds = int(fusion_cfg.get("oof_n_folds", 3))
        oof_val_days = int(fusion_cfg.get("oof_val_days", 2))
        oof_min_train_days = int(fusion_cfg.get("oof_min_train_days", max(min_core_days, 8)))
        folds = rolling_folds(
            days=train_days,
            n_folds=oof_n_folds,
            val_days=oof_val_days,
            min_train_days=oof_min_train_days,
        )
        oof_parts: list[pd.DataFrame] = []
        for fit_days, eval_days in folds:
            pair = collect_pair_predictions(fit_days, eval_days)
            if pair.empty:
                continue
            oof_parts.append(pair)

        if oof_parts:
            oof_merged = pd.concat(oof_parts, axis=0, ignore_index=True)
            oof_pair_frame = oof_merged[
                [
                    "tollgate_id",
                    "direction",
                    "horizon",
                    "time_window",
                    "actual",
                    "linear_prediction",
                    "gbdt_full_prediction",
                    "gbdt_target_prediction",
                ]
            ].copy()
            fit_input = oof_pair_frame.copy()
            fit_input["anchor"] = pd.to_datetime(fit_input["time_window"]).apply(anchor_bucket).astype(int)
            fit_input = fit_input.drop(columns=["time_window"])
            fusion_weights, fit_stats = fit_anchor_fusion_weights(fit_input, fusion_cfg)
            use_adapt = True
            adaptation_stats = {
                "use_adaptation": 1,
                "weight_learning": weight_learning,
                "folds": int(len(folds)),
                "oof_rows": int(len(oof_merged)),
                "fit": fit_stats,
            }
    else:
        if adapt_days > 0 and len(train_days) > (min_core_days + 1):
            true_adapt_days = min(adapt_days, len(train_days) - min_core_days)
            if true_adapt_days > 0:
                core_days = train_days[:-true_adapt_days]
                adapt_day_list = train_days[-true_adapt_days:]
                adapt_merged = collect_pair_predictions(core_days, adapt_day_list)
                if not adapt_merged.empty:
                    fit_input = adapt_merged[
                        [
                            "tollgate_id",
                            "direction",
                            "horizon",
                            "time_window",
                            "actual",
                            "linear_prediction",
                            "gbdt_full_prediction",
                            "gbdt_target_prediction",
                        ]
                    ].copy()
                    fit_input["anchor"] = pd.to_datetime(fit_input["time_window"]).apply(anchor_bucket).astype(int)
                    fit_input = fit_input.drop(columns=["time_window"])
                    fusion_weights, fit_stats = fit_anchor_fusion_weights(fit_input, fusion_cfg)
                    use_adapt = True
                    adaptation_stats = {
                        "use_adaptation": 1,
                        "weight_learning": weight_learning,
                        "adapt_days": int(true_adapt_days),
                        "core_days": int(len(core_days)),
                        "adapt_rows": int(len(adapt_merged)),
                        "fit": fit_stats,
                    }

    if not use_adapt:
        fusion_weights = default_fusion_weights(cfg)

    x_train, y_train, meta_train = select_by_days(x_all, y_all, meta_all, train_days)
    if x_train.empty:
        raise RuntimeError("Training subset is empty")

    final_baseline = train_baseline_branch(
        cfg=cfg,
        x_train=x_train,
        y_train=y_train,
        meta_train=meta_train,
        feature_names=feature_names,
    )
    x_gfull_train, y_gfull_train, meta_gfull_train = select_by_days(
        x_gbdt_full_all,
        y_gbdt_full_all,
        meta_gbdt_full_all,
        train_days,
    )
    if x_gfull_train.empty:
        raise RuntimeError("GBDT full training subset is empty")
    final_gbdt_full = train_gbdt_bundle(
        x_train=x_gfull_train,
        y_train=y_gfull_train,
        meta_train=meta_gfull_train,
        feature_names=feature_names,
        model_cfg=gbdt_full_model_cfg,
        training_cfg=gbdt_full_training_cfg,
    )
    x_gtarget_train, y_gtarget_train, meta_gtarget_train = select_by_days(
        x_gbdt_target_all,
        y_gbdt_target_all,
        meta_gbdt_target_all,
        train_days,
    )
    if x_gtarget_train.empty:
        raise RuntimeError("GBDT target training subset is empty")
    final_gbdt_target = train_gbdt_bundle(
        x_train=x_gtarget_train,
        y_train=y_gtarget_train,
        meta_train=meta_gtarget_train,
        feature_names=feature_names,
        model_cfg=gbdt_target_model_cfg,
        training_cfg=gbdt_target_training_cfg,
    )

    adaptation_stats["final_train_rows"] = int(len(x_train))
    adaptation_stats["final_gbdt_full_train_rows"] = int(len(x_gfull_train))
    adaptation_stats["final_gbdt_target_train_rows"] = int(len(x_gtarget_train))
    return final_baseline, final_gbdt_full, final_gbdt_target, fusion_weights, adaptation_stats, oof_pair_frame


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    feature_source = str(args.feature_source).strip().lower()
    if feature_source not in {"pandas", "sql"}:
        raise ValueError(f"Unsupported feature source: {feature_source}")

    run_id = args.run_id or cfg["run_id"]
    run_dir = PROJECT_ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    enhanced_cfg = cfg.get("feature", {}).get("enhanced", {})
    if not isinstance(enhanced_cfg, dict):
        enhanced_cfg = {}

    feature_cfg = FeatureConfig(
        lags=tuple(cfg["feature"]["lags"]),
        rolling_window=int(cfg["feature"]["rolling_window"]),
        enhanced_strict_past_only=enhanced_cfg.get("strict_past_only"),
        enhanced_slot_stats=enhanced_cfg.get("slot_stats"),
        enhanced_trend=enhanced_cfg.get("trend"),
        enhanced_volatility=enhanced_cfg.get("volatility"),
        enhanced_weather_interactions=enhanced_cfg.get("weather_interactions"),
    )

    use_weather = bool(cfg.get("feature", {}).get("use_weather", False))
    use_calendar = bool(cfg.get("feature", {}).get("use_calendar", False))
    use_enhanced_features = bool(cfg.get("feature", {}).get("use_enhanced_features", True))
    weather_columns = resolve_weather_columns(cfg, use_weather)

    train_weather: pd.DataFrame | None = None
    inference_weather: pd.DataFrame | None = None
    weather_default_map: dict[str, float] | None = None

    if use_weather:
        train_weather = load_weather_table(PROJECT_ROOT / cfg["paths"]["train_weather_csv"])
        test_weather = load_weather_table(PROJECT_ROOT / cfg["paths"]["test_weather_csv"])
        inference_weather = merge_weather_tables(train_weather, test_weather)
        weather_default_map = weather_defaults(train_weather)

    sql_snapshot_csv: Path | None = None
    if feature_source == "sql":
        cfg_sql_path = cfg.get("paths", {}).get("train_feature_snapshot_csv")
        if args.sql_snapshot_csv is not None:
            sql_snapshot_csv = args.sql_snapshot_csv
            if not sql_snapshot_csv.is_absolute():
                sql_snapshot_csv = PROJECT_ROOT / sql_snapshot_csv
        elif isinstance(cfg_sql_path, str) and cfg_sql_path.strip():
            sql_snapshot_csv = PROJECT_ROOT / cfg_sql_path
        else:
            raise ValueError(
                "SQL feature source requires --sql-snapshot-csv or paths.train_feature_snapshot_csv in config"
            )
        train_grid = load_train_grid_from_sql_snapshot(sql_snapshot_csv)
    else:
        train_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["train_volume_csv"])
        train_agg = aggregate_to_20min(train_events)
        train_grid = complete_20min_grid(train_agg)
    train_history = build_series_history(train_grid)
    series_keys = sorted(train_history.keys())

    split_ts = split_timestamp(train_grid["time_window"], int(cfg["validation"]["days"]))
    default_value = default_value_from_history(train_history)

    gbdt_training_full_cfg = cfg.get("gbdt_training_full", cfg.get("gbdt_training", {}))
    gbdt_training_target_cfg = cfg.get("gbdt_training_target")
    if gbdt_training_target_cfg is None:
        gbdt_training_target_cfg = dict(cfg.get("gbdt_training", {}))
        gbdt_training_target_cfg["target_only"] = True

    x_all, y_all, meta_all = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        target_only=True,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )
    if x_all.empty:
        raise RuntimeError("No training samples for pre-holdout period")
    x_gbdt_full_all, y_gbdt_full_all, meta_gbdt_full_all = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        target_only=bool(gbdt_training_full_cfg.get("target_only", False)),
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )
    if x_gbdt_full_all.empty:
        raise RuntimeError("No full-window GBDT training samples for pre-holdout period")
    x_gbdt_target_all, y_gbdt_target_all, meta_gbdt_target_all = build_training_dataset(
        train_grid=train_grid,
        history=train_history,
        series_keys=series_keys,
        cfg=feature_cfg,
        train_end=split_ts,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        target_only=bool(gbdt_training_target_cfg.get("target_only", True)),
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
    )
    if x_gbdt_target_all.empty:
        raise RuntimeError("No target-window GBDT training samples for pre-holdout period")

    feature_names = feature_columns(series_keys, feature_cfg, include_calendar=use_calendar) + weather_columns
    horizon_windows = int(cfg["target"]["horizon_windows"])

    pre_days = sorted(pd.to_datetime(meta_all["day"].drop_duplicates().tolist()))

    rolling_cfg = cfg.get("rolling_validation", {})
    use_rolling = bool(rolling_cfg.get("use", False))
    series_expert_pool = resolve_series_expert_pool(cfg)
    tft_branch_cfg = cfg.get("tft_branch", {})
    use_tft_branch = bool(isinstance(tft_branch_cfg, dict) and tft_branch_cfg.get("use", False))
    rolling_results: list[dict[str, float | int]] = []
    series_expert_experience_frames: dict[str, list[pd.DataFrame]] = {
        expert_tag: [] for expert_tag, _ in series_expert_pool
    }

    if use_rolling:
        folds = rolling_folds(
            days=pre_days,
            n_folds=int(rolling_cfg.get("n_folds", 3)),
            val_days=int(rolling_cfg.get("val_days", 2)),
            min_train_days=int(rolling_cfg.get("min_train_days", 10)),
        )

        for idx, (fold_train_days, fold_val_days) in enumerate(folds, start=1):
            (
                base_bundle,
                gbdt_full_bundle,
                gbdt_target_bundle,
                fusion_weights,
                fold_adapt,
                fold_oof_pair,
            ) = train_with_adaptive_fusion(
                cfg=cfg,
                x_all=x_all,
                y_all=y_all,
                meta_all=meta_all,
                x_gbdt_full_all=x_gbdt_full_all,
                y_gbdt_full_all=y_gbdt_full_all,
                meta_gbdt_full_all=meta_gbdt_full_all,
                x_gbdt_target_all=x_gbdt_target_all,
                y_gbdt_target_all=y_gbdt_target_all,
                meta_gbdt_target_all=meta_gbdt_target_all,
                train_history=train_history,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                feature_names=feature_names,
                default_value=default_value,
                include_calendar=use_calendar,
                use_enhanced_features=use_enhanced_features,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
                train_days=fold_train_days,
                horizon_windows=horizon_windows,
            )
            fold_post_bundle = train_post_fusion_residual_bundle(
                cfg=cfg,
                oof_pair_frame=fold_oof_pair,
                fusion_weights=fusion_weights,
            )
            fold_tft_bundle = train_tft_branch_for_days(
                cfg=cfg,
                x_gbdt_full_all=x_gbdt_full_all,
                y_gbdt_full_all=y_gbdt_full_all,
                meta_gbdt_full_all=meta_gbdt_full_all,
                x_gbdt_target_all=x_gbdt_target_all,
                y_gbdt_target_all=y_gbdt_target_all,
                meta_gbdt_target_all=meta_gbdt_target_all,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                include_calendar=use_calendar,
                weather_columns=weather_columns,
                train_days=fold_train_days,
            )
            fold_series_expert_bundles: list[tuple[str, dict, SeriesExpertBundle]] = []
            for expert_tag, expert_cfg in series_expert_pool:
                fold_bundle = train_series_expert_bundle_for_days(
                    cfg=cfg,
                    x_gbdt_full_all=x_gbdt_full_all,
                    y_gbdt_full_all=y_gbdt_full_all,
                    meta_gbdt_full_all=meta_gbdt_full_all,
                    x_gbdt_target_all=x_gbdt_target_all,
                    y_gbdt_target_all=y_gbdt_target_all,
                    meta_gbdt_target_all=meta_gbdt_target_all,
                    feature_names=feature_names,
                    train_days=fold_train_days,
                    series_cfg_override=expert_cfg,
                    expert_tag=expert_tag,
                )
                fold_series_expert_bundles.append((expert_tag, expert_cfg, fold_bundle))

            fold_schedule = target_windows_for_days(fold_val_days, horizon=horizon_windows)
            fold_base_history, fold_actual = prepare_history_for_schedule(train_history, series_keys, fold_schedule)
            fold_gbdt_full_history, _ = prepare_history_for_schedule(train_history, series_keys, fold_schedule)
            fold_gbdt_target_history, _ = prepare_history_for_schedule(train_history, series_keys, fold_schedule)

            fold_base_pred = run_baseline_branch_forecast(
                bundle=base_bundle,
                history=fold_base_history,
                schedule=fold_schedule,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                default_value=default_value,
                include_calendar=use_calendar,
                use_enhanced_features=use_enhanced_features,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
                actual_map=fold_actual,
            )
            fold_gbdt_full_pred = run_gbdt_recursive_forecast(
                bundle=gbdt_full_bundle,
                history=fold_gbdt_full_history,
                schedule=fold_schedule,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                default_value=default_value,
                include_calendar=use_calendar,
                use_enhanced_features=use_enhanced_features,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
                actual_map=fold_actual,
            )
            fold_gbdt_target_pred = run_gbdt_recursive_forecast(
                bundle=gbdt_target_bundle,
                history=fold_gbdt_target_history,
                schedule=fold_schedule,
                series_keys=series_keys,
                feature_cfg=feature_cfg,
                default_value=default_value,
                include_calendar=use_calendar,
                use_enhanced_features=use_enhanced_features,
                weather_table=train_weather,
                weather_defaults_map=weather_default_map,
                weather_columns=weather_columns,
                actual_map=fold_actual,
            )
            fold_fused = fuse_predictions(
                baseline_pred=fold_base_pred,
                gbdt_full_pred=fold_gbdt_full_pred,
                gbdt_target_pred=fold_gbdt_target_pred,
                fusion_weights=fusion_weights,
            )
            fold_fused = apply_post_fusion_residual_adjustment(fold_fused, fold_post_bundle)
            if fold_tft_bundle.enabled:
                fold_tft_history, _ = prepare_history_for_schedule(train_history, series_keys, fold_schedule)
                fold_tft_pred = run_tft_recursive_forecast(
                    bundle=fold_tft_bundle,
                    history=fold_tft_history,
                    schedule=fold_schedule,
                    series_keys=series_keys,
                    feature_cfg=feature_cfg,
                    default_value=default_value,
                    include_calendar=use_calendar,
                    use_enhanced_features=use_enhanced_features,
                    weather_table=train_weather,
                    weather_defaults_map=weather_default_map,
                    weather_columns=weather_columns,
                    actual_map=fold_actual,
                )
                fold_fused = apply_tft_branch_adjustment(
                    pred_df=fold_fused,
                    tft_pred_df=fold_tft_pred,
                    bundle=fold_tft_bundle,
                )
            for expert_tag, _expert_cfg, fold_bundle in fold_series_expert_bundles:
                fold_series_expert_history, _ = prepare_history_for_schedule(
                    train_history, series_keys, fold_schedule
                )
                fold_series_expert_pred = run_series_expert_recursive_forecast(
                    bundle=fold_bundle,
                    history=fold_series_expert_history,
                    schedule=fold_schedule,
                    series_keys=series_keys,
                    feature_cfg=feature_cfg,
                    default_value=default_value,
                    include_calendar=use_calendar,
                    use_enhanced_features=use_enhanced_features,
                    weather_table=train_weather,
                    weather_defaults_map=weather_default_map,
                    weather_columns=weather_columns,
                    actual_map=fold_actual,
                )
                fold_series_expert_exp = build_series_expert_rl_experience(
                    pred_df=fold_fused,
                    expert_pred_df=fold_series_expert_pred,
                    bundle=fold_bundle,
                    base_prediction_col="prediction",
                )
                if not fold_series_expert_exp.empty:
                    series_expert_experience_frames.setdefault(expert_tag, []).append(fold_series_expert_exp)
                fold_fused = apply_series_expert_adjustment(
                    pred_df=fold_fused,
                    expert_pred_df=fold_series_expert_pred,
                    bundle=fold_bundle,
                    expert_tag=expert_tag,
                )
            fold_fused = fold_fused.dropna(subset=["actual"]).reset_index(drop=True)
            if fold_fused.empty:
                continue
            fold_metrics = summarize_metrics(fold_fused)
            fold_enabled_experts = [
                expert_tag
                for expert_tag, _expert_cfg, fold_bundle in fold_series_expert_bundles
                if fold_bundle.enabled and fold_bundle.series_key is not None
            ]
            rolling_results.append(
                {
                    "fold": idx,
                    "train_days": int(len(fold_train_days)),
                    "val_days": int(len(fold_val_days)),
                    "overall_mape": float(fold_metrics["overall_mape"]),
                    "global_linear_weight": float(fusion_weights.global_weights.global_weights[0]),
                    "global_gbdt_full_weight": float(fusion_weights.global_weights.global_weights[1]),
                    "global_gbdt_target_weight": float(fusion_weights.global_weights.global_weights[2]),
                    "anchor_weight_count": int(len(fusion_weights.anchor_weights)),
                    "use_adaptation": int(fold_adapt.get("use_adaptation", 0)),
                    "post_fusion_enabled": int(fold_post_bundle.enabled),
                    "post_fusion_trained_series": int(len(fold_post_bundle.models)),
                    "tft_enabled": int(fold_tft_bundle.enabled),
                    "series_expert_enabled": int(bool(fold_enabled_experts)),
                    "series_expert_enabled_count": int(len(fold_enabled_experts)),
                    "series_expert_total": int(len(fold_series_expert_bundles)),
                    "series_expert_series": ",".join(fold_enabled_experts),
                }
            )

    (
        final_base,
        final_gbdt_full,
        final_gbdt_target,
        final_fusion,
        final_adapt_stats,
        final_oof_pair,
    ) = train_with_adaptive_fusion(
        cfg=cfg,
        x_all=x_all,
        y_all=y_all,
        meta_all=meta_all,
        x_gbdt_full_all=x_gbdt_full_all,
        y_gbdt_full_all=y_gbdt_full_all,
        meta_gbdt_full_all=meta_gbdt_full_all,
        x_gbdt_target_all=x_gbdt_target_all,
        y_gbdt_target_all=y_gbdt_target_all,
        meta_gbdt_target_all=meta_gbdt_target_all,
        train_history=train_history,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        feature_names=feature_names,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        train_days=pre_days,
        horizon_windows=horizon_windows,
    )
    final_post_bundle = train_post_fusion_residual_bundle(
        cfg=cfg,
        oof_pair_frame=final_oof_pair,
        fusion_weights=final_fusion,
    )
    final_tft_bundle = train_tft_branch_for_days(
        cfg=cfg,
        x_gbdt_full_all=x_gbdt_full_all,
        y_gbdt_full_all=y_gbdt_full_all,
        meta_gbdt_full_all=meta_gbdt_full_all,
        x_gbdt_target_all=x_gbdt_target_all,
        y_gbdt_target_all=y_gbdt_target_all,
        meta_gbdt_target_all=meta_gbdt_target_all,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        include_calendar=use_calendar,
        weather_columns=weather_columns,
        train_days=pre_days,
    )
    final_series_expert_bundles: list[tuple[str, SeriesExpertBundle]] = []
    for expert_tag, expert_cfg in series_expert_pool:
        final_series_expert_bundle = train_series_expert_bundle_for_days(
            cfg=cfg,
            x_gbdt_full_all=x_gbdt_full_all,
            y_gbdt_full_all=y_gbdt_full_all,
            meta_gbdt_full_all=meta_gbdt_full_all,
            x_gbdt_target_all=x_gbdt_target_all,
            y_gbdt_target_all=y_gbdt_target_all,
            meta_gbdt_target_all=meta_gbdt_target_all,
            feature_names=feature_names,
            train_days=pre_days,
            series_cfg_override=expert_cfg,
            expert_tag=expert_tag,
        )
        series_expert_exp_frames = series_expert_experience_frames.get(expert_tag, [])
        series_expert_exp_df = (
            pd.concat(series_expert_exp_frames, axis=0, ignore_index=True)
            if series_expert_exp_frames
            else pd.DataFrame(
                columns=["horizon", "anchor", "delta_ratio", "base_prediction", "expert_prediction", "actual"]
            )
        )
        final_series_expert_bundle = train_series_expert_rl_policy(
            series_cfg=expert_cfg,
            bundle=final_series_expert_bundle,
            experience_df=series_expert_exp_df,
        )
        final_series_expert_bundle = train_series_expert_gate_policy(
            series_cfg=expert_cfg,
            bundle=final_series_expert_bundle,
            experience_df=series_expert_exp_df,
        )
        final_series_expert_bundles.append((expert_tag, final_series_expert_bundle))

    holdout_days = sorted(
        pd.to_datetime(train_grid.loc[train_grid["time_window"] >= split_ts, "time_window"].dt.normalize().unique())
    )
    holdout_schedule = target_windows_for_days(holdout_days, horizon=horizon_windows)

    hold_base_history, hold_actual = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)
    hold_gbdt_full_history, _ = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)
    hold_gbdt_target_history, _ = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)

    hold_base_pred = run_baseline_branch_forecast(
        bundle=final_base,
        history=hold_base_history,
        schedule=holdout_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=hold_actual,
    )
    hold_gbdt_full_pred = run_gbdt_recursive_forecast(
        bundle=final_gbdt_full,
        history=hold_gbdt_full_history,
        schedule=holdout_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=hold_actual,
    )
    hold_gbdt_target_pred = run_gbdt_recursive_forecast(
        bundle=final_gbdt_target,
        history=hold_gbdt_target_history,
        schedule=holdout_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=train_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=hold_actual,
    )
    holdout_pred = fuse_predictions(
        baseline_pred=hold_base_pred,
        gbdt_full_pred=hold_gbdt_full_pred,
        gbdt_target_pred=hold_gbdt_target_pred,
        fusion_weights=final_fusion,
    )
    holdout_pred = apply_post_fusion_residual_adjustment(holdout_pred, final_post_bundle)
    if final_tft_bundle.enabled:
        hold_tft_history, _ = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)
        hold_tft_pred = run_tft_recursive_forecast(
            bundle=final_tft_bundle,
            history=hold_tft_history,
            schedule=holdout_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=use_calendar,
            use_enhanced_features=use_enhanced_features,
            weather_table=train_weather,
            weather_defaults_map=weather_default_map,
            weather_columns=weather_columns,
            actual_map=hold_actual,
        )
        holdout_pred = apply_tft_branch_adjustment(
            pred_df=holdout_pred,
            tft_pred_df=hold_tft_pred,
            bundle=final_tft_bundle,
        )
    for expert_tag, final_series_expert_bundle in final_series_expert_bundles:
        hold_series_expert_history, _ = prepare_history_for_schedule(train_history, series_keys, holdout_schedule)
        hold_series_expert_pred = run_series_expert_recursive_forecast(
            bundle=final_series_expert_bundle,
            history=hold_series_expert_history,
            schedule=holdout_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=use_calendar,
            use_enhanced_features=use_enhanced_features,
            weather_table=train_weather,
            weather_defaults_map=weather_default_map,
            weather_columns=weather_columns,
            actual_map=hold_actual,
        )
        holdout_pred = apply_series_expert_adjustment(
            pred_df=holdout_pred,
            expert_pred_df=hold_series_expert_pred,
            bundle=final_series_expert_bundle,
            expert_tag=expert_tag,
        )
    holdout_pred = holdout_pred.dropna(subset=["actual"]).reset_index(drop=True)

    metrics = summarize_metrics(holdout_pred)
    slice_df = build_error_slice_table(holdout_pred)
    slice_path = run_dir / "validation_error_slices.csv"
    slice_df.to_csv(slice_path, index=False)

    test_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["test_volume_csv"])
    test_agg = aggregate_to_20min(test_events)
    test_grid = complete_20min_grid(test_agg)
    test_history = build_series_history(test_grid)
    merged_history = merge_histories(train_history, test_history)

    test_days = sorted(pd.to_datetime(test_grid["time_window"].dt.normalize().unique()))
    test_schedule = target_windows_for_days(test_days, horizon=horizon_windows)

    test_base_history = {k: s.copy() for k, s in merged_history.items()}
    test_gbdt_full_history = {k: s.copy() for k, s in merged_history.items()}
    test_gbdt_target_history = {k: s.copy() for k, s in merged_history.items()}

    test_base_pred = run_baseline_branch_forecast(
        bundle=final_base,
        history=test_base_history,
        schedule=test_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=inference_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=None,
    )
    test_gbdt_full_pred = run_gbdt_recursive_forecast(
        bundle=final_gbdt_full,
        history=test_gbdt_full_history,
        schedule=test_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=inference_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=None,
    )
    test_gbdt_target_pred = run_gbdt_recursive_forecast(
        bundle=final_gbdt_target,
        history=test_gbdt_target_history,
        schedule=test_schedule,
        series_keys=series_keys,
        feature_cfg=feature_cfg,
        default_value=default_value,
        include_calendar=use_calendar,
        use_enhanced_features=use_enhanced_features,
        weather_table=inference_weather,
        weather_defaults_map=weather_default_map,
        weather_columns=weather_columns,
        actual_map=None,
    )
    test_pred = fuse_predictions(
        baseline_pred=test_base_pred,
        gbdt_full_pred=test_gbdt_full_pred,
        gbdt_target_pred=test_gbdt_target_pred,
        fusion_weights=final_fusion,
    )
    test_pred = apply_post_fusion_residual_adjustment(test_pred, final_post_bundle)
    if final_tft_bundle.enabled:
        test_tft_history = {k: s.copy() for k, s in merged_history.items()}
        test_tft_pred = run_tft_recursive_forecast(
            bundle=final_tft_bundle,
            history=test_tft_history,
            schedule=test_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=use_calendar,
            use_enhanced_features=use_enhanced_features,
            weather_table=inference_weather,
            weather_defaults_map=weather_default_map,
            weather_columns=weather_columns,
            actual_map=None,
        )
        test_pred = apply_tft_branch_adjustment(
            pred_df=test_pred,
            tft_pred_df=test_tft_pred,
            bundle=final_tft_bundle,
        )
    for expert_tag, final_series_expert_bundle in final_series_expert_bundles:
        test_series_expert_history = {k: s.copy() for k, s in merged_history.items()}
        test_series_expert_pred = run_series_expert_recursive_forecast(
            bundle=final_series_expert_bundle,
            history=test_series_expert_history,
            schedule=test_schedule,
            series_keys=series_keys,
            feature_cfg=feature_cfg,
            default_value=default_value,
            include_calendar=use_calendar,
            use_enhanced_features=use_enhanced_features,
            weather_table=inference_weather,
            weather_defaults_map=weather_default_map,
            weather_columns=weather_columns,
            actual_map=None,
        )
        test_pred = apply_series_expert_adjustment(
            pred_df=test_pred,
            expert_pred_df=test_series_expert_pred,
            bundle=final_series_expert_bundle,
            expert_tag=expert_tag,
        )

    submission = build_submission(test_pred)
    validate_submission_schema(submission)

    rolling_avg = float(np.mean([x["overall_mape"] for x in rolling_results])) if rolling_results else None
    series_expert_stats = []
    for expert_tag, bundle in final_series_expert_bundles:
        stat = dict(bundle.stats)
        stat["expert_tag"] = expert_tag
        if bundle.series_key is not None:
            stat["series_key"] = f"{bundle.series_key[0]}_{bundle.series_key[1]}"
        series_expert_stats.append(stat)
    if not series_expert_stats:
        series_expert_meta: dict[str, object] = {
            "enabled": 0,
            "expert_count": 0,
            "enabled_count": 0,
            "experts": [],
            "reason": "empty_series_expert_pool",
        }
    elif len(series_expert_stats) == 1:
        series_expert_meta = series_expert_stats[0]
    else:
        series_expert_meta = {
            "enabled": int(any(bool(int(stat.get("enabled", 0))) for stat in series_expert_stats)),
            "expert_count": int(len(series_expert_stats)),
            "enabled_count": int(sum(int(bool(int(stat.get("enabled", 0)))) for stat in series_expert_stats)),
            "experts": series_expert_stats,
        }

    run_meta = {
        "run_id": run_id,
        "split_timestamp": split_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_source": feature_source,
        "sql_snapshot_csv": None,
        "train_samples": int(len(x_all)),
        "gbdt_full_train_samples": int(len(x_gbdt_full_all)),
        "gbdt_target_train_samples": int(len(x_gbdt_target_all)),
        "validation_rows": int(len(holdout_pred)),
        "validation_slice_rows": int(len(slice_df)),
        "submission_rows": int(len(submission)),
        "use_weather": use_weather,
        "use_calendar": use_calendar,
        "weather_columns": weather_columns,
        "baseline_branch": final_base.stats,
        "gbdt_model_full": cfg.get("gbdt_model_full", cfg.get("gbdt_model", {})),
        "gbdt_model_target": cfg.get("gbdt_model_target", cfg.get("gbdt_model", {})),
        "gbdt_training_full": gbdt_training_full_cfg,
        "gbdt_training_target": gbdt_training_target_cfg,
        "fusion": {
            "global_linear_weight": float(final_fusion.global_weights.global_weights[0]),
            "global_gbdt_full_weight": float(final_fusion.global_weights.global_weights[1]),
            "global_gbdt_target_weight": float(final_fusion.global_weights.global_weights[2]),
            "series_weight_count": int(len(final_fusion.global_weights.series_weights)),
            "slice_weight_count": int(len(final_fusion.global_weights.slice_weights)),
            "anchor_weight_count": int(len(final_fusion.anchor_weights)),
            "memory_retrieval_enabled": int(
                final_fusion.memory_retrieval is not None and final_fusion.memory_retrieval.enabled
            ),
            "risk_constraint_enabled": int(
                final_fusion.risk_constraint is not None and final_fusion.risk_constraint.enabled
            ),
            "adaptation": final_adapt_stats,
        },
        "post_fusion_residual": final_post_bundle.stats,
        "tft_branch": final_tft_bundle.stats,
        "series_expert": series_expert_meta,
        "rolling_validation": {
            "enabled": use_rolling,
            "folds": rolling_results,
            "avg_mape": rolling_avg,
        },
        "artifacts": {
            "validation_predictions_csv": str((run_dir / "validation_predictions.csv").relative_to(PROJECT_ROOT)),
            "validation_error_slices_csv": str(slice_path.relative_to(PROJECT_ROOT)),
        },
        "metrics": metrics,
    }
    if sql_snapshot_csv is not None:
        try:
            run_meta["sql_snapshot_csv"] = str(sql_snapshot_csv.relative_to(PROJECT_ROOT))
        except ValueError:
            run_meta["sql_snapshot_csv"] = str(sql_snapshot_csv)

    (run_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout_pred.to_csv(run_dir / "validation_predictions.csv", index=False)

    sub_path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{run_id}.csv"
    submission.to_csv(sub_path, index=False)

    print(json.dumps(run_meta, ensure_ascii=False, indent=2))
    print(f"submission_path={sub_path}")


if __name__ == "__main__":
    main()
