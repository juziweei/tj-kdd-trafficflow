#!/usr/bin/env python3
"""Optuna global search runner for strong_backbone_v6 configs."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna global search for v6 pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "strong_backbone_v6_expert_pool_1_0_2_0.json",
        help="Base config path",
    )
    parser.add_argument("--trials", type=int, default=100, help="Optuna trial count")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout seconds, 0 means no timeout")
    parser.add_argument("--study-name", type=str, default="v6_global_search")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage URL (optional)")
    parser.add_argument("--run-id-prefix", type=str, default="strong_backbone_fusion_20260329_v6_tft_external_optuna")
    parser.add_argument("--n-jobs", type=int, default=1, help="Optuna parallel jobs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-temp-configs", action="store_true")
    parser.add_argument(
        "--objective",
        type=str,
        choices=["holdout", "hybrid"],
        default="holdout",
        help="Optimization target",
    )
    parser.add_argument(
        "--rolling-lambda",
        type=float,
        default=0.25,
        help="hybrid objective = holdout + rolling_lambda * rolling_avg",
    )
    return parser.parse_args()


def normalize_weights(linear: float, full: float, target: float) -> dict[str, float]:
    arr = [max(0.0, float(linear)), max(0.0, float(full)), max(0.0, float(target))]
    s = sum(arr)
    if s <= 1e-12:
        return {"linear": 1.0 / 3.0, "gbdt_full": 1.0 / 3.0, "gbdt_target": 1.0 / 3.0}
    return {
        "linear": arr[0] / s,
        "gbdt_full": arr[1] / s,
        "gbdt_target": arr[2] / s,
    }


def apply_trial_params(cfg: dict, trial: optuna.Trial) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("fusion", {})
    w = normalize_weights(
        trial.suggest_float("w_linear_raw", 0.15, 0.80),
        trial.suggest_float("w_full_raw", 0.15, 0.80),
        trial.suggest_float("w_target_raw", 0.15, 0.80),
    )
    out["fusion"]["default_branch_weights"] = w

    if isinstance(out.get("series_expert_pool"), list):
        for idx, expert in enumerate(out["series_expert_pool"]):
            if not isinstance(expert, dict) or not bool(expert.get("use", False)):
                continue
            name = str(expert.get("name", f"expert_{idx}")).lower()
            if name == "expert_1_0":
                expert["blend_weight"] = float(trial.suggest_float(f"{name}_blend", 0.20, 0.50))
            elif name == "expert_2_0":
                expert["blend_weight"] = float(trial.suggest_float(f"{name}_blend", 0.10, 0.35))
            else:
                expert["blend_weight"] = float(trial.suggest_float(f"{name}_blend", 0.10, 0.50))
            hbw = expert.get("horizon_blend_weight", {})
            if isinstance(hbw, dict):
                for h_text in sorted(hbw.keys()):
                    h = int(h_text)
                    low, high = 0.10, 0.60
                    if name == "expert_1_0":
                        if h in (4, 5):
                            low, high = 0.35, 0.75
                        elif h == 6:
                            low, high = 0.05, 0.30
                    elif name == "expert_2_0":
                        if h in (5, 6):
                            low, high = 0.08, 0.35
                    expert["horizon_blend_weight"][h_text] = float(
                        trial.suggest_float(f"{name}_h{h_text}", low, high)
                    )

    tft_cfg = out.get("tft_branch", {})
    if isinstance(tft_cfg, dict) and bool(tft_cfg.get("use", False)):
        tft_cfg["blend_weight"] = float(trial.suggest_float("tft_blend", 0.02, 0.12))
        tft_cfg["min_samples"] = int(trial.suggest_int("tft_min_samples", 100, 140, step=10))
        tft_scope = str(
            trial.suggest_categorical(
                "tft_target_scope",
                ["both", "1_0_only"],
            )
        )
        if tft_scope == "1_0_only":
            tft_cfg["target_series"] = ["1_0"]
        else:
            tft_cfg["target_series"] = ["1_0", "2_0"]
        tft_cfg["use_weather_interactions"] = True
        tft_cfg["use_event_features"] = True
        hbw = tft_cfg.get("horizon_blend_weight", {})
        if isinstance(hbw, dict):
            for h_text in sorted(hbw.keys()):
                h = int(h_text)
                low = 0.01 if h <= 4 else 0.02
                high = 0.12 if h <= 4 else (0.20 if h == 5 else 0.25)
                tft_cfg["horizon_blend_weight"][h_text] = float(
                    trial.suggest_float(f"tft_h{h_text}", low, high)
                )
        model_cfg = tft_cfg.get("model", {})
        if isinstance(model_cfg, dict):
            model_cfg["hidden_size"] = int(trial.suggest_categorical("tft_hidden_size", [32, 48, 64]))
            model_cfg["dropout"] = float(trial.suggest_float("tft_dropout", 0.08, 0.18))
            model_cfg["learning_rate"] = float(trial.suggest_float("tft_lr", 5e-4, 1.8e-3, log=True))
            model_cfg["epochs"] = int(trial.suggest_categorical("tft_epochs", [30, 40, 50]))
            tft_cfg["model"] = model_cfg
        out["tft_branch"] = tft_cfg

    post_cfg = out.get("post_fusion_residual", {})
    if isinstance(post_cfg, dict) and bool(post_cfg.get("use", False)):
        post_cfg["clip_abs"] = float(trial.suggest_float("post_clip_abs", 4.0, 6.5))
        post_cfg["ridge_alpha"] = float(trial.suggest_float("post_alpha", 100.0, 220.0))
        out["post_fusion_residual"] = post_cfg

    return out


def run_single_trial(
    cfg: dict,
    run_id: str,
    tmp_cfg_dir: Path,
    keep_tmp: bool,
    objective: str,
    rolling_lambda: float,
) -> dict[str, float]:
    cfg_path = tmp_cfg_dir / f"{run_id}.json"
    cfg["run_id"] = run_id
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        "python3",
        str(PROJECT_ROOT / "scripts" / "run_strong_backbone_v6.py"),
        "--config",
        str(cfg_path),
        "--run-id",
        run_id,
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"[trial-fail] run_id={run_id} code={exc.returncode}")
        if exc.stdout:
            print(exc.stdout[-1500:])
        if exc.stderr:
            print(exc.stderr[-1500:])
        return 1e6
    finally:
        if not keep_tmp:
            try:
                cfg_path.unlink(missing_ok=True)
            except Exception:
                pass

    metrics_path = PROJECT_ROOT / "outputs" / "runs" / run_id / "metrics.json"
    if not metrics_path.exists():
        return {"value": 1e6, "holdout_mape": 1e6, "rolling_avg_mape": 1e6}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        holdout = float(metrics["metrics"]["overall_mape"])
        rolling = float((metrics.get("rolling_validation") or {}).get("avg_mape", holdout))
        if objective == "hybrid":
            value = holdout + float(rolling_lambda) * rolling
        else:
            value = holdout
        return {
            "value": float(value),
            "holdout_mape": holdout,
            "rolling_avg_mape": rolling,
        }
    except Exception:
        return {"value": 1e6, "holdout_mape": 1e6, "rolling_avg_mape": 1e6}


def main() -> None:
    args = parse_args()
    base_cfg = json.loads(args.config.read_text(encoding="utf-8"))

    tmp_cfg_dir = PROJECT_ROOT / "outputs" / "optuna_tmp_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir = PROJECT_ROOT / "outputs" / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    storage = args.storage if args.storage else None
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = apply_trial_params(base_cfg, trial)
        run_id = f"{args.run_id_prefix}_t{trial.number:03d}"
        result = run_single_trial(
            cfg=trial_cfg,
            run_id=run_id,
            tmp_cfg_dir=tmp_cfg_dir,
            keep_tmp=bool(args.keep_temp_configs),
            objective=str(args.objective),
            rolling_lambda=float(args.rolling_lambda),
        )
        trial.set_user_attr("run_id", run_id)
        trial.set_user_attr("holdout_mape", float(result["holdout_mape"]))
        trial.set_user_attr("rolling_avg_mape", float(result["rolling_avg_mape"]))
        return float(result["value"])

    timeout = None if int(args.timeout) <= 0 else int(args.timeout)
    study.optimize(
        objective,
        n_trials=int(args.trials),
        timeout=timeout,
        n_jobs=int(args.n_jobs),
        show_progress_bar=True,
    )

    best_cfg = apply_trial_params(base_cfg, trial=study.best_trial)
    best_cfg["run_id"] = f"{args.run_id_prefix}_best"
    best_path = out_dir / f"{args.study_name}_best_config.json"
    summary_path = out_dir / f"{args.study_name}_summary.json"
    best_path.write_text(json.dumps(best_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "study_name": args.study_name,
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "best_params": study.best_params,
        "best_run_id": study.best_trial.user_attrs.get("run_id", ""),
        "best_holdout_mape": float(study.best_trial.user_attrs.get("holdout_mape", 1e6)),
        "best_rolling_avg_mape": float(study.best_trial.user_attrs.get("rolling_avg_mape", 1e6)),
        "objective": str(args.objective),
        "rolling_lambda": float(args.rolling_lambda),
        "best_config_path": str(best_path.relative_to(PROJECT_ROOT)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
