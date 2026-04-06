#!/usr/bin/env python3
"""Run weather feature ablation experiments based on baseline config."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "baseline_v2_weather.json"


def run_once(cfg: dict, run_id: str, tmp_dir: Path) -> float:
    cfg_path = tmp_dir / f"{run_id}.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        "python3",
        str(PROJECT_ROOT / "scripts" / "run_baseline.py"),
        "--config",
        str(cfg_path),
        "--run-id",
        run_id,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    metrics_path = PROJECT_ROOT / "outputs" / "runs" / run_id / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics["metrics"]["overall_mape"])


def main() -> None:
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    weather_cols = cfg["feature"]["weather_columns"] if "weather_columns" in cfg["feature"] else [
        "weather_pressure",
        "weather_sea_pressure",
        "weather_wind_speed",
        "weather_temperature",
        "weather_rel_humidity",
        "weather_precipitation",
        "weather_wind_dir_sin",
        "weather_wind_dir_cos",
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PROJECT_ROOT / "outputs" / "runs" / f"weather_ablation_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_root / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | str]] = []

    cfg_all = json.loads(json.dumps(cfg))
    cfg_all["feature"]["use_weather"] = True
    cfg_all["feature"]["weather_columns"] = weather_cols
    all_run_id = f"ablation_all_{stamp}"
    m_all = run_once(cfg_all, all_run_id, tmp_dir)
    records.append({"scenario": "all_weather", "run_id": all_run_id, "overall_mape": m_all})

    for drop_col in weather_cols:
        kept = [c for c in weather_cols if c != drop_col]
        cfg_drop = json.loads(json.dumps(cfg))
        cfg_drop["feature"]["use_weather"] = True
        cfg_drop["feature"]["weather_columns"] = kept

        run_id = f"ablation_drop_{drop_col}_{stamp}"
        mape_value = run_once(cfg_drop, run_id, tmp_dir)
        records.append(
            {
                "scenario": f"drop:{drop_col}",
                "run_id": run_id,
                "overall_mape": mape_value,
            }
        )

    summary = pd.DataFrame(records)
    summary["delta_vs_all"] = summary["overall_mape"] - m_all
    summary = summary.sort_values("overall_mape").reset_index(drop=True)

    summary_csv = run_root / "ablation_summary.csv"
    summary_json = run_root / "ablation_summary.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
