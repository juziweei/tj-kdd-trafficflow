#!/usr/bin/env python3
"""Run governance checks for anti-leakage and reproducible time-based validation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.volume_io import aggregate_to_20min, build_series_history, complete_20min_grid, load_volume_events
from src.data.weather_io import WEATHER_FEATURE_COLUMNS, get_weather_feature_vector, load_weather_table, weather_defaults
from src.features.volume_features import FeatureConfig, build_feature_row, is_target_window
from src.inference.submission import validate_submission_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Governance checks for leakage-safe modeling workflow")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config JSON used by training pipeline",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id override (used for artifact checks and report path)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=24,
        help="Number of (series, timestamp) samples for perturbation checks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed for deterministic checks",
    )
    parser.add_argument(
        "--future-delta",
        type=float,
        default=12345.0,
        help="Magnitude used when perturbing future observations",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for invariance checks",
    )
    parser.add_argument(
        "--skip-artifact-check",
        action="store_true",
        help="Skip checks that require outputs/runs/<run_id> artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional report json path",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    **extra: Any,
) -> None:
    item: dict[str, Any] = {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
    }
    item.update(extra)
    checks.append(item)


def max_abs_diff(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return 0.0
    return max(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def find_random_flags(payload: Any, path: str = "config") -> list[str]:
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            key_l = key_str.lower()
            curr = f"{path}.{key_str}"
            if key_l in {"shuffle", "random_split", "use_random_split"} and bool(value):
                hits.append(curr)
            if key_l.endswith("split") and isinstance(value, str) and "random" in value.lower():
                hits.append(curr)
            hits.extend(find_random_flags(value, curr))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            hits.extend(find_random_flags(value, f"{path}[{idx}]"))
    return hits


def resolve_feature_cfg(feature: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        lags=tuple(int(x) for x in feature.get("lags", [1, 2, 3, 6, 72, 504])),
        rolling_window=int(feature.get("rolling_window", 6)),
        enhanced_strict_past_only=feature.get("enhanced_strict_past_only"),
        enhanced_slot_stats=feature.get("enhanced_slot_stats"),
        enhanced_recent_stats=feature.get("enhanced_recent_stats"),
        enhanced_rush_stats=feature.get("enhanced_rush_stats"),
        enhanced_alignment=feature.get("enhanced_alignment"),
        enhanced_dense_slice_gating=feature.get("enhanced_dense_slice_gating"),
        enhanced_dense_target_slices=tuple(feature.get("enhanced_dense_target_slices", []))
        if feature.get("enhanced_dense_target_slices") is not None
        else None,
        enhanced_dense_slice_feature_groups=feature.get("enhanced_dense_slice_feature_groups"),
        enhanced_trend=feature.get("enhanced_trend"),
        enhanced_volatility=feature.get("enhanced_volatility"),
        enhanced_weather_interactions=feature.get("enhanced_weather_interactions"),
    )


def sample_pairs(
    series_keys: list[tuple[int, int]],
    ts_candidates: list[pd.Timestamp],
    sample_size: int,
    seed: int,
) -> dict[pd.Timestamp, list[tuple[int, int]]]:
    pairs = [(key, ts) for ts in ts_candidates for key in series_keys]
    if not pairs:
        return {}
    rng = random.Random(seed)
    if sample_size >= len(pairs):
        chosen = pairs
    else:
        chosen = rng.sample(pairs, sample_size)
    grouped: dict[pd.Timestamp, list[tuple[int, int]]] = {}
    for key, ts in chosen:
        grouped.setdefault(ts, []).append(key)
    return grouped


def run_feature_perturbation_check(
    history: dict[tuple[int, int], pd.Series],
    series_keys: list[tuple[int, int]],
    ts_candidates: list[pd.Timestamp],
    cfg: FeatureConfig,
    default_value: float,
    use_weather: bool,
    weather_table: pd.DataFrame | None,
    weather_default_map: dict[str, float] | None,
    sample_size: int,
    seed: int,
    future_delta: float,
    tolerance: float,
) -> tuple[bool, str, dict[str, Any]]:
    sampled = sample_pairs(series_keys, ts_candidates, sample_size=sample_size, seed=seed)
    if not sampled:
        return False, "No timestamp samples available for perturbation check", {"sampled_rows": 0}

    fail_examples: list[dict[str, Any]] = []
    checked_rows = 0
    skipped_rows = 0

    for ts, keys in sampled.items():
        perturbed_history = {k: s.copy() for k, s in history.items()}
        for key in series_keys:
            s = perturbed_history[key]
            mask = s.index >= ts
            if mask.any():
                s.loc[mask] = s.loc[mask] + future_delta

        weather_vec: dict[str, float] | None = None
        if use_weather and weather_table is not None and weather_default_map is not None:
            weather_vec = get_weather_feature_vector(weather_table, ts, weather_default_map)

        for key in keys:
            base = build_feature_row(
                key=key,
                ts=ts,
                history=history,
                series_keys=series_keys,
                cfg=cfg,
                default_value=default_value,
                allow_fallback=False,
                weather=weather_vec,
                use_enhanced_features=True,
            )
            perturbed = build_feature_row(
                key=key,
                ts=ts,
                history=perturbed_history,
                series_keys=series_keys,
                cfg=cfg,
                default_value=default_value,
                allow_fallback=False,
                weather=weather_vec,
                use_enhanced_features=True,
            )
            if base is None or perturbed is None:
                skipped_rows += 1
                continue
            checked_rows += 1
            diff = max_abs_diff(base, perturbed)
            if diff > tolerance:
                fail_examples.append(
                    {
                        "tollgate_id": int(key[0]),
                        "direction": int(key[1]),
                        "time_window": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "max_abs_diff": float(diff),
                    }
                )
                if len(fail_examples) >= 8:
                    break
        if len(fail_examples) >= 8:
            break

    passed = len(fail_examples) == 0 and checked_rows > 0
    detail = (
        f"checked={checked_rows}, skipped={skipped_rows}, fails={len(fail_examples)}"
        if checked_rows > 0
        else "No valid rows could be checked (all sampled rows skipped)"
    )
    return passed, detail, {
        "sampled_timestamps": int(len(sampled)),
        "checked_rows": int(checked_rows),
        "skipped_rows": int(skipped_rows),
        "fail_examples": fail_examples,
    }


def run_weather_perturbation_check(
    ts_candidates: list[pd.Timestamp],
    weather_table: pd.DataFrame,
    defaults: dict[str, float],
    sample_size: int,
    seed: int,
    future_delta: float,
    tolerance: float,
) -> tuple[bool, str, dict[str, Any]]:
    if weather_table.empty:
        return False, "Weather table is empty", {"checked_rows": 0}
    if not ts_candidates:
        return False, "No timestamp samples available for weather check", {"checked_rows": 0}

    rng = random.Random(seed + 17)
    if sample_size >= len(ts_candidates):
        chosen_ts = list(ts_candidates)
    else:
        chosen_ts = rng.sample(ts_candidates, sample_size)

    fail_examples: list[dict[str, Any]] = []
    checked_rows = 0
    for ts in chosen_ts:
        base = get_weather_feature_vector(weather_table, ts, defaults)

        mutated = weather_table.copy()
        start = ts.floor("1h")
        mask = mutated.index >= start
        if mask.any():
            mutated.loc[mask, WEATHER_FEATURE_COLUMNS] = (
                mutated.loc[mask, WEATHER_FEATURE_COLUMNS] + future_delta
            )
        perturbed = get_weather_feature_vector(mutated, ts, defaults)
        diff = max_abs_diff(base, perturbed)
        checked_rows += 1
        if diff > tolerance:
            fail_examples.append(
                {
                    "time_window": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "max_abs_diff": float(diff),
                }
            )
            if len(fail_examples) >= 8:
                break

    passed = len(fail_examples) == 0 and checked_rows > 0
    detail = f"checked={checked_rows}, fails={len(fail_examples)}"
    return passed, detail, {
        "checked_rows": int(checked_rows),
        "fail_examples": fail_examples,
    }


def expected_split_timestamp(train_grid: pd.DataFrame, validation_days: int) -> pd.Timestamp:
    last_day = pd.Timestamp(train_grid["time_window"].max()).normalize()
    return last_day - pd.Timedelta(days=validation_days - 1)


def run_artifact_checks(
    cfg: dict[str, Any],
    run_id: str,
    train_grid: pd.DataFrame,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    run_dir = PROJECT_ROOT / "outputs" / "runs" / run_id
    add_check(
        checks,
        "artifacts.run_dir_exists",
        run_dir.exists(),
        f"path={run_dir.relative_to(PROJECT_ROOT) if run_dir.exists() else run_dir}",
    )
    if not run_dir.exists():
        return checks

    metrics_path = run_dir / "metrics.json"
    add_check(
        checks,
        "artifacts.metrics_json_exists",
        metrics_path.exists(),
        f"path={metrics_path.relative_to(PROJECT_ROOT) if metrics_path.exists() else metrics_path}",
    )
    if not metrics_path.exists():
        return checks

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rolling = metrics.get("rolling_validation", {})
    add_check(
        checks,
        "artifacts.rolling_validation_enabled",
        bool(rolling.get("enabled", False)),
        f"enabled={rolling.get('enabled', False)}",
    )
    folds = rolling.get("folds", [])
    add_check(
        checks,
        "artifacts.rolling_validation_has_folds",
        isinstance(folds, list) and len(folds) > 0,
        f"fold_count={len(folds) if isinstance(folds, list) else -1}",
    )

    split_raw = str(metrics.get("split_timestamp", ""))
    expected_split = expected_split_timestamp(train_grid, int(cfg["validation"]["days"])).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    add_check(
        checks,
        "artifacts.split_timestamp_expected",
        split_raw == expected_split,
        f"got={split_raw}, expected={expected_split}",
    )

    overall = (((metrics.get("metrics") or {}).get("overall_mape")))
    overall_ok = isinstance(overall, (int, float)) and math.isfinite(float(overall))
    add_check(
        checks,
        "artifacts.overall_mape_finite",
        overall_ok,
        f"overall_mape={overall}",
    )

    artifacts = metrics.get("artifacts", {})
    val_slice_rel = artifacts.get("validation_error_slices_csv")
    val_pred_rel = artifacts.get("validation_predictions_csv")

    if isinstance(val_slice_rel, str):
        val_slice_path = PROJECT_ROOT / val_slice_rel
        add_check(
            checks,
            "artifacts.validation_error_slices_exists",
            val_slice_path.exists(),
            f"path={val_slice_rel}",
        )
        if val_slice_path.exists():
            slice_df = pd.read_csv(val_slice_path)
            required = {"tollgate_id", "direction", "horizon", "rows", "mape", "mae"}
            ok_cols = required.issubset(set(slice_df.columns))
            add_check(
                checks,
                "artifacts.validation_error_slices_schema",
                ok_cols,
                f"columns={slice_df.columns.tolist()}",
            )
            add_check(
                checks,
                "artifacts.validation_error_slices_nonempty",
                len(slice_df) > 0,
                f"rows={len(slice_df)}",
            )
    else:
        add_check(
            checks,
            "artifacts.validation_error_slices_exists",
            False,
            "metrics.artifacts.validation_error_slices_csv missing",
        )

    if isinstance(val_pred_rel, str):
        val_pred_path = PROJECT_ROOT / val_pred_rel
        add_check(
            checks,
            "artifacts.validation_predictions_exists",
            val_pred_path.exists(),
            f"path={val_pred_rel}",
        )
    else:
        add_check(
            checks,
            "artifacts.validation_predictions_exists",
            False,
            "metrics.artifacts.validation_predictions_csv missing",
        )

    submission_path = PROJECT_ROOT / "outputs" / "submissions" / f"submission_{run_id}.csv"
    add_check(
        checks,
        "artifacts.submission_exists",
        submission_path.exists(),
        f"path={submission_path.relative_to(PROJECT_ROOT) if submission_path.exists() else submission_path}",
    )
    if submission_path.exists():
        try:
            sub = pd.read_csv(submission_path)
            validate_submission_schema(sub)
            add_check(
                checks,
                "artifacts.submission_schema_valid",
                True,
                f"rows={len(sub)}",
            )
        except Exception as exc:  # pragma: no cover
            add_check(
                checks,
                "artifacts.submission_schema_valid",
                False,
                f"error={exc}",
            )

    return checks


def default_value_from_history(history: dict[tuple[int, int], pd.Series]) -> float:
    values = pd.concat(list(history.values()), axis=0)
    return float(values.mean())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    feature_cfg = resolve_feature_cfg(cfg.get("feature", {}))
    run_id = str(args.run_id or cfg.get("run_id", "")).strip()

    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "config.run_id_present",
        len(run_id) > 0,
        f"run_id={run_id if run_id else '<empty>'}",
    )
    add_check(
        checks,
        "config.validation_days_positive",
        int(cfg.get("validation", {}).get("days", 0)) > 0,
        f"validation.days={cfg.get('validation', {}).get('days')}",
    )
    rv = cfg.get("rolling_validation", {})
    add_check(
        checks,
        "config.rolling_validation_enabled",
        bool(rv.get("use", False)),
        f"rolling_validation.use={rv.get('use', False)}",
    )
    random_hits = find_random_flags(cfg)
    add_check(
        checks,
        "config.no_random_split_flags",
        len(random_hits) == 0,
        "none" if not random_hits else "; ".join(random_hits),
    )

    train_events = load_volume_events(PROJECT_ROOT / cfg["paths"]["train_volume_csv"])
    train_agg = aggregate_to_20min(train_events)
    train_grid = complete_20min_grid(train_agg)
    history = build_series_history(train_grid)
    series_keys = sorted(history.keys())
    default_value = default_value_from_history(history)

    max_lag = max(feature_cfg.lags) if feature_cfg.lags else 1
    earliest = pd.Timestamp(train_grid["time_window"].min()) + pd.Timedelta(minutes=20 * max_lag)
    latest = pd.Timestamp(train_grid["time_window"].max()) - pd.Timedelta(minutes=20)
    ts_candidates = [
        pd.Timestamp(ts)
        for ts in sorted(pd.to_datetime(train_grid["time_window"].unique()))
        if earliest <= pd.Timestamp(ts) <= latest and is_target_window(pd.Timestamp(ts))
    ]

    use_weather = bool(cfg.get("feature", {}).get("use_weather", False))
    train_weather: pd.DataFrame | None = None
    weather_default_map: dict[str, float] | None = None
    if use_weather:
        train_weather = load_weather_table(PROJECT_ROOT / cfg["paths"]["train_weather_csv"])
        weather_default_map = weather_defaults(train_weather)

    feature_passed, feature_detail, feature_extra = run_feature_perturbation_check(
        history=history,
        series_keys=series_keys,
        ts_candidates=ts_candidates,
        cfg=feature_cfg,
        default_value=default_value,
        use_weather=use_weather,
        weather_table=train_weather,
        weather_default_map=weather_default_map,
        sample_size=max(1, int(args.sample_size)),
        seed=int(args.seed),
        future_delta=float(args.future_delta),
        tolerance=float(args.tolerance),
    )
    add_check(
        checks,
        "leakage.volume_feature_future_invariance",
        feature_passed,
        feature_detail,
        **feature_extra,
    )

    if use_weather and train_weather is not None and weather_default_map is not None:
        weather_passed, weather_detail, weather_extra = run_weather_perturbation_check(
            ts_candidates=ts_candidates,
            weather_table=train_weather,
            defaults=weather_default_map,
            sample_size=max(1, int(args.sample_size)),
            seed=int(args.seed),
            future_delta=float(args.future_delta),
            tolerance=float(args.tolerance),
        )
        add_check(
            checks,
            "leakage.weather_feature_future_invariance",
            weather_passed,
            weather_detail,
            **weather_extra,
        )
    else:
        add_check(
            checks,
            "leakage.weather_feature_future_invariance",
            True,
            "Skipped (weather features disabled)",
        )

    if not args.skip_artifact_check:
        if not run_id:
            add_check(
                checks,
                "artifacts.skipped_without_run_id",
                False,
                "Artifact checks require --run-id or config.run_id",
            )
        else:
            checks.extend(
                run_artifact_checks(
                    cfg=cfg,
                    run_id=run_id,
                    train_grid=train_grid,
                )
            )

    failed = [x for x in checks if not x["passed"]]
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(args.config),
        "run_id": run_id,
        "summary": {
            "total": len(checks),
            "passed": len(checks) - len(failed),
            "failed": len(failed),
        },
        "checks": checks,
    }

    output_path = args.output
    if output_path is None and run_id:
        output_path = PROJECT_ROOT / "outputs" / "runs" / run_id / "governance_check.json"
    if output_path is not None:
        output_path = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(output_path.relative_to(PROJECT_ROOT))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
