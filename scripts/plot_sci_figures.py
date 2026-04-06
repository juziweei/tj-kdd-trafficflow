#!/usr/bin/env python3
"""Generate SCI-style figures from run metrics with reproducible filtering logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

# Sanitize external matplotlib shadow paths in this environment.
BAD_MPL_PATH_KEY = "matplotlib-26208-work/lib"
sys.path = [p for p in sys.path if BAD_MPL_PATH_KEY not in str(p)]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-config")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito colorblind-friendly palette.
OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
]


@dataclass
class RunRecord:
    run_id: str
    metrics_path: Path
    split_timestamp: str | None
    overall_mape: float
    horizon_mape: dict[int, float]
    series_horizon_mape: dict[tuple[str, int], float]
    rolling_folds: list[float]
    rolling_avg: float | None
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCI-style result figures")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/sci"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/figure_manifest.md"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/figures/sci/run_summary.csv"))
    return parser.parse_args()


def classify_run(run_id: str) -> str:
    rid = run_id.lower()
    if rid.startswith("baseline_"):
        return "baseline"
    if rid.startswith("gbdt_"):
        return "gbdt"
    if rid.startswith("strong_backbone_"):
        return "strong_backbone"
    if rid.startswith("target12_generalize_") or rid.startswith("target145_"):
        return "generalization"
    if rid.startswith("ablation_") or "ablation" in rid:
        return "ablation"
    if "dense_features" in rid or "enhanced" in rid:
        return "enhanced"
    if "fixedblend" in rid or "ensemble" in rid or "fused" in rid or "fusion" in rid:
        return "fusion"
    return "other"


def parse_horizon_map(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            horizon = int(str(key))
            out[horizon] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def parse_series_horizon_map(raw: Any) -> dict[tuple[str, int], float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[tuple[str, int], float] = {}
    for key, value in raw.items():
        # Expected format like "1_0_h3".
        text = str(key)
        parts = text.split("_")
        if len(parts) != 3 or not parts[2].startswith("h"):
            continue
        series = f"{parts[0]}_{parts[1]}"
        try:
            horizon = int(parts[2][1:])
            out[(series, horizon)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def parse_rolling(metrics_blob: dict[str, Any]) -> tuple[list[float], float | None]:
    rv = metrics_blob.get("rolling_validation")
    if not isinstance(rv, dict) or not bool(rv.get("enabled", False)):
        return [], None
    folds = rv.get("folds", [])
    fold_mapes: list[float] = []
    if isinstance(folds, list):
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            value = fold.get("overall_mape")
            try:
                fold_mapes.append(float(value))
            except (TypeError, ValueError):
                continue
    avg = rv.get("avg_mape")
    avg_float: float | None = None
    try:
        avg_float = float(avg) if avg is not None else None
    except (TypeError, ValueError):
        avg_float = None
    return fold_mapes, avg_float


def load_runs(runs_root: Path) -> tuple[list[RunRecord], Counter[str], int]:
    failures: Counter[str] = Counter()
    records: list[RunRecord] = []

    metrics_files = sorted(runs_root.glob("*/metrics.json"))
    total_candidates = len(metrics_files)

    for metrics_path in metrics_files:
        run_id = metrics_path.parent.name
        try:
            blob = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            failures["json_decode_error"] += 1
            continue

        metrics = blob.get("metrics")
        if not isinstance(metrics, dict):
            failures["missing_metrics_block"] += 1
            continue

        overall = metrics.get("overall_mape")
        try:
            overall_mape = float(overall)
            if not math.isfinite(overall_mape):
                raise ValueError("non-finite")
        except Exception:
            failures["missing_or_invalid_overall_mape"] += 1
            continue

        horizon_mape = parse_horizon_map(metrics.get("horizon_mape"))
        series_horizon_mape = parse_series_horizon_map(metrics.get("series_horizon_mape"))
        rolling_folds, rolling_avg = parse_rolling(blob)

        records.append(
            RunRecord(
                run_id=run_id,
                metrics_path=metrics_path,
                split_timestamp=str(blob.get("split_timestamp")) if blob.get("split_timestamp") is not None else None,
                overall_mape=overall_mape,
                horizon_mape=horizon_mape,
                series_horizon_mape=series_horizon_mape,
                rolling_folds=rolling_folds,
                rolling_avg=rolling_avg,
                category=classify_run(run_id),
            )
        )

    return records, failures, total_candidates


def select_best(records: list[RunRecord], predicate) -> RunRecord | None:
    candidates = [r for r in records if predicate(r)]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x.overall_mape)


def select_representatives(records: list[RunRecord]) -> list[RunRecord]:
    # Exclude highly validation-tailored route/oracle runs for main comparison.
    blocked = ("route_seriesh_dow", "sliceh_oracle")

    def is_blocked(run_id: str) -> bool:
        rid = run_id.lower()
        return any(key in rid for key in blocked)

    usable = [r for r in records if not is_blocked(r.run_id)]

    picks: list[RunRecord] = []
    category_rules = [
        ("Baseline", lambda r: r.category == "baseline"),
        ("GBDT", lambda r: r.category == "gbdt"),
        ("StrongBackbone", lambda r: r.category == "strong_backbone"),
        ("Generalization", lambda r: r.category == "generalization"),
        ("Fusion", lambda r: r.category == "fusion"),
        (
            "Enhanced",
            lambda r: ("dense_features" in r.run_id.lower())
            or ("enhanced" in r.run_id.lower() and "no_enhanced" not in r.run_id.lower() and "enhanced_off" not in r.run_id.lower()),
        ),
    ]

    for _, rule in category_rules:
        best = select_best(usable, rule)
        if best is not None and best.run_id not in {p.run_id for p in picks}:
            picks.append(best)

    # Enhanced fallback if no positive enhanced run available.
    if not any("dense_features" in p.run_id.lower() or "enhanced" in p.run_id.lower() for p in picks):
        fallback = select_best(
            usable,
            lambda r: ("dense_features" in r.run_id.lower()) or ("enhanced" in r.run_id.lower()),
        )
        if fallback is not None and fallback.run_id not in {p.run_id for p in picks}:
            picks.append(fallback)

    return picks


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.9,
            "grid.color": "#DDDDDD",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.5,
        }
    )


def short_label(run_id: str) -> str:
    rid = run_id.lower()
    if rid.startswith("baseline_"):
        return "Baseline"
    if rid.startswith("gbdt_"):
        return "GBDT"
    if rid.startswith("strong_backbone_"):
        return "StrongBackbone"
    if "fixedblend" in rid:
        return "FixedBlend"
    if "ensemble" in rid:
        return "Ensemble"
    if rid.startswith("target12_generalize_"):
        return "Generalization"
    if "dense_features" in rid:
        return "DenseFeatures"
    return run_id[:18]


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure1_overall(records: list[RunRecord], output_dir: Path) -> dict[str, Any]:
    labels = [short_label(r.run_id) for r in records]
    values = [r.overall_mape for r in records]
    errs = []
    for r in records:
        if len(r.rolling_folds) >= 2:
            errs.append(float(np.std(np.array(r.rolling_folds), ddof=0)))
        else:
            errs.append(0.0)

    x = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars = ax.bar(
        x,
        values,
        yerr=errs,
        capsize=3.0,
        color=[OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(records))],
        edgecolor="#333333",
        linewidth=0.7,
        alpha=0.92,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Overall MAPE (%)")
    ax.set_title("Figure 1. Overall MAPE Comparison Across Representative Runs")
    ax.grid(axis="y", alpha=0.7)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.08,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    save_figure(fig, output_dir / "figure1_overall_mape")
    return {
        "figure": "Figure 1",
        "type": "bar_with_error",
        "source_runs": [r.run_id for r in records],
        "notes": "Error bars use rolling-fold std when available; otherwise 0.",
    }


def runs_with_full_horizon(records: list[RunRecord]) -> list[RunRecord]:
    out = []
    for r in records:
        if all(h in r.horizon_mape for h in range(1, 7)):
            out.append(r)
    return out


def plot_figure2_horizon(records: list[RunRecord], output_dir: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(1, 7)
    for i, r in enumerate(records):
        y = [r.horizon_mape[h] for h in x]
        ax.plot(
            x,
            y,
            marker="o",
            color=OKABE_ITO[i % len(OKABE_ITO)],
            label=short_label(r.run_id),
        )

    ax.set_xticks(x)
    ax.set_xlabel("Forecast Horizon (1-6)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Figure 2. Horizon-wise MAPE Curves")
    ax.grid(True, alpha=0.7)
    ax.legend(loc="best", frameon=True)

    save_figure(fig, output_dir / "figure2_horizon_mape")
    return {
        "figure": "Figure 2",
        "type": "line",
        "source_runs": [r.run_id for r in records],
        "notes": "Only runs with complete horizon_mape(1..6) are used.",
    }


def pick_heatmap_run(records: list[RunRecord]) -> RunRecord | None:
    candidates = [r for r in records if len(r.series_horizon_mape) > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x.overall_mape)


def plot_figure3_heatmap(run: RunRecord, output_dir: Path) -> dict[str, Any]:
    series = sorted({s for s, _ in run.series_horizon_mape.keys()})
    horizons = [1, 2, 3, 4, 5, 6]
    mat = np.full((len(series), len(horizons)), np.nan, dtype=float)
    for i, s in enumerate(series):
        for j, h in enumerate(horizons):
            mat[i, j] = run.series_horizon_mape.get((s, h), np.nan)

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_yticks(np.arange(len(series)))
    ax.set_yticklabels(series)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Series (tollgate_direction)")
    ax.set_title(f"Figure 3. Series×Horizon MAPE Heatmap ({run.run_id})")

    for i in range(len(series)):
        for j in range(len(horizons)):
            if math.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAPE (%)")

    save_figure(fig, output_dir / "figure3_series_horizon_heatmap")
    return {
        "figure": "Figure 3",
        "type": "heatmap",
        "source_runs": [run.run_id],
        "notes": "Heatmap is built from metrics.series_horizon_mape.",
    }


def load_ablation_records(records: list[RunRecord]) -> tuple[RunRecord | None, list[RunRecord]]:
    ref = select_best(records, lambda r: r.run_id.startswith("ablation_all_"))
    drops = [
        r
        for r in records
        if r.run_id.startswith("ablation_drop_weather_") and "wind_dir_cos" in r.run_id or r.run_id.startswith("ablation_drop_weather_")
    ]
    return ref, sorted(drops, key=lambda x: x.overall_mape)


def plot_figure4_waterfall(ref: RunRecord, ablations: list[RunRecord], output_dir: Path) -> dict[str, Any]:
    # Independent ablations; we visualize delta trend as a waterfall-like chart.
    names: list[str] = []
    deltas: list[float] = []
    for r in ablations:
        tag = r.run_id.replace("ablation_drop_weather_", "").replace("_20260327_182855", "")
        names.append(tag)
        deltas.append(ref.overall_mape - r.overall_mape)

    order = np.argsort(deltas)[::-1]
    names = [names[i] for i in order]
    deltas = [deltas[i] for i in order]

    cumulative = [0.0]
    for d in deltas:
        cumulative.append(cumulative[-1] + d)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, d in enumerate(deltas):
        y0 = cumulative[i] if d >= 0 else cumulative[i] + d
        ax.bar(
            i,
            abs(d),
            bottom=y0,
            color=OKABE_ITO[2] if d >= 0 else OKABE_ITO[1],
            edgecolor="#333333",
            linewidth=0.7,
            width=0.66,
        )
        ax.text(i, y0 + abs(d) + 0.01, f"{d:+.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Delta MAPE vs reference (%)")
    ax.set_title("Figure 4. Weather-Ablation Delta Waterfall (Independent Deltas)")
    ax.grid(axis="y", alpha=0.7)

    save_figure(fig, output_dir / "figure4_ablation_waterfall")
    return {
        "figure": "Figure 4",
        "type": "waterfall_delta",
        "source_runs": [ref.run_id] + [r.run_id for r in ablations],
        "notes": "Deltas are independent single-drop ablations vs ablation_all reference.",
    }


def pick_stability_runs(records: list[RunRecord], preferred: list[RunRecord]) -> list[RunRecord]:
    pool = [r for r in records if len(r.rolling_folds) >= 2]
    out: list[RunRecord] = []
    seen = set()
    for r in preferred:
        if len(r.rolling_folds) >= 2 and r.run_id not in seen:
            out.append(r)
            seen.add(r.run_id)
    for r in sorted(pool, key=lambda x: x.overall_mape):
        if r.run_id in seen:
            continue
        out.append(r)
        seen.add(r.run_id)
        if len(out) >= 6:
            break
    return out[:6]


def plot_figure5_boxplot(records: list[RunRecord], output_dir: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    data = [r.rolling_folds for r in records]
    labels = [short_label(r.run_id) for r in records]
    box = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for i, b in enumerate(box["boxes"]):
        b.set_facecolor(OKABE_ITO[i % len(OKABE_ITO)])
        b.set_alpha(0.6)
        b.set_edgecolor("#333333")
        b.set_linewidth(0.8)
    ax.set_ylabel("Rolling Fold MAPE (%)")
    ax.set_title("Figure 5. Stability Boxplot Across Rolling Folds")
    ax.grid(axis="y", alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    save_figure(fig, output_dir / "figure5_stability_boxplot")
    return {
        "figure": "Figure 5",
        "type": "boxplot",
        "source_runs": [r.run_id for r in records],
        "notes": "Only runs with >=2 rolling folds are included.",
    }


def write_summary_csv(records: list[RunRecord], path: Path) -> None:
    rows = []
    for r in records:
        rows.append(
            {
                "run_id": r.run_id,
                "category": r.category,
                "overall_mape": r.overall_mape,
                "split_timestamp": r.split_timestamp,
                "horizon_points": len(r.horizon_mape),
                "series_horizon_points": len(r.series_horizon_mape),
                "rolling_folds": len(r.rolling_folds),
                "rolling_avg_mape": r.rolling_avg,
                "metrics_path": str(r.metrics_path),
            }
        )
    pd.DataFrame(rows).sort_values("overall_mape", ascending=True).to_csv(path, index=False)


def write_manifest(
    path: Path,
    total_candidates: int,
    valid_records: list[RunRecord],
    failures: Counter[str],
    reps: list[RunRecord],
    figure_entries: list[dict[str, Any]],
    filtered_notes: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Figure Manifest (SCI Style)")
    lines.append("")
    lines.append("## 1. Scan Summary")
    lines.append("")
    lines.append(f"- Total candidate runs scanned: **{total_candidates}**")
    lines.append(f"- Valid runs with numeric `metrics.overall_mape`: **{len(valid_records)}**")
    if failures:
        lines.append("- Filtered invalid runs:")
        for key, cnt in sorted(failures.items()):
            lines.append(f"  - `{key}`: {cnt}")
    else:
        lines.append("- Filtered invalid runs: none")

    lines.append("")
    lines.append("## 2. Representative Runs")
    lines.append("")
    lines.append("| Label | run_id | category | overall_mape | split_timestamp | rolling_folds |")
    lines.append("|---|---|---:|---:|---|---:|")
    for r in reps:
        lines.append(
            f"| {short_label(r.run_id)} | `{r.run_id}` | {r.category} | {r.overall_mape:.4f} | {r.split_timestamp} | {len(r.rolling_folds)} |"
        )

    lines.append("")
    lines.append("## 3. Figure Data Sources")
    lines.append("")
    for item in figure_entries:
        lines.append(f"### {item['figure']}")
        lines.append(f"- Plot type: `{item['type']}`")
        lines.append("- Source runs:")
        for run_id in item["source_runs"]:
            lines.append(f"  - `{run_id}`")
        lines.append(f"- Notes: {item['notes']}")
        lines.append("")

    lines.append("## 4. Filtering Notes")
    lines.append("")
    if filtered_notes:
        for note in filtered_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    setup_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    records, failures, total_candidates = load_runs(args.runs_root)
    if not records:
        raise RuntimeError("No valid run metrics found; cannot produce figures.")

    write_summary_csv(records, args.summary_csv)

    reps = select_representatives(records)
    filtered_notes: list[str] = []
    if not reps:
        reps = sorted(records, key=lambda x: x.overall_mape)[:5]
        filtered_notes.append("Representative selection fallback: used top-5 runs by overall_mape.")

    fig_entries: list[dict[str, Any]] = []
    fig_entries.append(plot_figure1_overall(reps, args.output_dir))

    lineset = runs_with_full_horizon(reps)
    if len(lineset) < 2:
        supplement = [r for r in sorted(records, key=lambda x: x.overall_mape) if all(h in r.horizon_mape for h in range(1, 7))]
        for r in supplement:
            if r.run_id not in {x.run_id for x in lineset}:
                lineset.append(r)
            if len(lineset) >= 5:
                break
        filtered_notes.append("Figure 2 supplemented with globally best complete-horizon runs.")
    fig_entries.append(plot_figure2_horizon(lineset[:5], args.output_dir))

    heatmap_run = pick_heatmap_run(lineset) or pick_heatmap_run(records)
    if heatmap_run is None:
        filtered_notes.append("Figure 3 skipped: no series_horizon_mape available.")
    else:
        fig_entries.append(plot_figure3_heatmap(heatmap_run, args.output_dir))

    ref, ablations = load_ablation_records(records)
    if ref is not None and ablations:
        fig_entries.append(plot_figure4_waterfall(ref, ablations, args.output_dir))
    else:
        filtered_notes.append("Figure 4 skipped: ablation_all or ablation_drop_weather runs missing.")

    stability_runs = pick_stability_runs(records, reps)
    if len(stability_runs) >= 2:
        fig_entries.append(plot_figure5_boxplot(stability_runs, args.output_dir))
    else:
        filtered_notes.append("Figure 5 skipped: insufficient runs with rolling folds.")

    write_manifest(
        path=args.manifest,
        total_candidates=total_candidates,
        valid_records=records,
        failures=failures,
        reps=reps,
        figure_entries=fig_entries,
        filtered_notes=filtered_notes,
    )

    print(f"[done] Generated {len(fig_entries)} figure groups into: {args.output_dir}")
    print(f"[done] Manifest: {args.manifest}")
    print(f"[done] Summary CSV: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
