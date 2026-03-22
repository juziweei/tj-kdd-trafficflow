# KDD-CUP-2017-TrafficFlow (Course Project)

## Goal
Competition-first implementation for highway tollgate traffic-flow forecasting, with research-grade reproducibility.

## Principles
- Maximize leaderboard-like metric (MAPE) under strict anti-leakage rules.
- Keep every experiment reproducible (config + seed + log + artifact).
- Separate baseline, feature engineering, and model fusion pipelines.

## Project Layout
- `configs/`: experiment configs
- `src/`: core pipeline code
- `scripts/`: runnable entrypoints
- `data/`: local datasets (ignored)
- `outputs/`: submissions, plots, and run artifacts
- `docs/`: research notes and analysis
- `tests/`: unit/integration tests

## Quick Start
1. Put dataset files in `data/raw/`.
2. Build baseline features and train model.
3. Generate submission file to `outputs/submissions/`.

## Standards
- Workflow log: `docs/vibe_coding_protocol.md` (must update every coding session).
- Time-based split only (no random shuffle for validation).
- Any feature must use only historical information at prediction time.
- Each run should record config hash and metric summary.
