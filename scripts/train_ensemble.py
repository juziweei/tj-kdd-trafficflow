#!/usr/bin/env python3
"""Train leak-free ensemble members by varying random seeds."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ensemble members with different seeds")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/strong_backbone_v6_tft_external.json"),
        help="Base config used to generate seed-specific configs",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_strong_backbone_v6.py"),
        help="Training entrypoint script",
    )
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds")
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="leakfree_ensemble_seed",
        help="Run ID prefix, final run_id is <run_prefix>_<seed>",
    )
    parser.add_argument(
        "--config-prefix",
        type=str,
        default="leakfree_ensemble_seed",
        help="Generated config filename prefix",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory to write generated per-seed configs",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to launch training runs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have metrics and validation predictions",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs without training")
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds parsed from --seeds")
    return seeds


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed_recursive(node: Any, seed: int) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "seed" and isinstance(value, (int, float)):
                node[key] = int(seed)
            else:
                set_seed_recursive(value, seed)
        return
    if isinstance(node, list):
        for item in node:
            set_seed_recursive(item, seed)


def ensure_leak_free(cfg: dict[str, Any]) -> None:
    train_csv = str(cfg.get("paths", {}).get("train_volume_csv", ""))
    if not train_csv:
        raise ValueError("Config missing paths.train_volume_csv")
    if "merged" in train_csv.lower():
        raise ValueError(
            f"Refusing to train ensemble from merged training data: {train_csv}. "
            "Use a leak-free training source."
        )


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    base_config_path = resolve_path(args.base_config)
    runner_path = resolve_path(args.runner)
    config_dir = resolve_path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(base_config_path)
    ensure_leak_free(base_cfg)

    run_ids: list[str] = []
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        run_id = f"{args.run_prefix}_{seed}"
        cfg["run_id"] = run_id
        set_seed_recursive(cfg, seed)

        cfg_path = config_dir / f"{args.config_prefix}_{seed}.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")

        run_ids.append(run_id)
        run_dir = PROJECT_ROOT / "outputs" / "runs" / run_id
        metrics_path = run_dir / "metrics.json"
        pred_path = run_dir / "validation_predictions.csv"

        if args.skip_existing and metrics_path.exists() and pred_path.exists():
            print(f"[skip] seed={seed} run_id={run_id} already exists")
            continue

        cmd = [
            args.python_exe,
            str(runner_path),
            "--config",
            str(cfg_path),
            "--run-id",
            run_id,
        ]
        print(f"[train] seed={seed} run_id={run_id}")
        print(f"[cmd] {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    print("\nEnsemble members:")
    for run_id in run_ids:
        print(f"- {run_id}")
    print(
        "Next: python3 scripts/fuse_ensemble.py "
        f"--run-ids {','.join(run_ids)}"
    )


if __name__ == "__main__":
    main()
