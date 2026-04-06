# Execution Plan (2026-03-29 ~ 2026-04-04)

## 1) Objective and Constraints
- Primary objective: minimize `overall_mape`.
- Hard constraints:
  - `rolling_validation.avg_mape` must not worsen materially.
  - keep improving hotspot slice `1_0_h6`.
  - fixed time-based split only; no random split.

## 2) Frozen Baseline (Do Not Change)
- Primary baseline run:
  - `run_id`: `strong_backbone_fusion_20260329_v6_tft_external_router_h6_split_memory_h6safe_c`
  - `overall_mape`: `16.0134`
  - `rolling avg`: `23.5135`
  - `1_0_h6`: `27.1592`
  - metrics: `outputs/runs/strong_backbone_fusion_20260329_v6_tft_external_router_h6_split_memory_h6safe_c/metrics.json`
- Reference fallback run:
  - `run_id`: `strong_backbone_fusion_20260329_v6_tft_external_router_h6_split`
  - `overall_mape`: `16.0365`
  - `rolling avg`: `23.5114`
  - `1_0_h6`: `28.2187`

## 3) Fixed Protocol
- Data and split are frozen:
  - training data path from config (`dataset_60/training`)
  - split timestamp must stay `2016-10-11 00:00:00`
- Repro command template:
  - `python3 scripts/run_strong_backbone_v6.py --config <config.json>`
- Every run must write:
  - `outputs/runs/<run_id>/metrics.json`
  - `outputs/runs/<run_id>/validation_error_slices.csv`

## 4) Decision Gates (Per Run)
- Accept candidate only if all pass:
  - `overall_mape <= baseline_overall - 0.005`
  - `rolling_avg <= baseline_rolling + 0.05`
  - `1_0_h6 <= baseline_1_0_h6 - 0.20`
- Hard reject if any hit:
  - `overall_mape > baseline_overall + 0.03`
  - `rolling_avg > baseline_rolling + 0.10`

## 5) Budget and Rounds
- Total budget: max 9 runs.
- Round 1 (3 runs): narrow micro-tuning on current structure only.
  - tune only:
    - `fusion.memory_retrieval.blend_weight` in `[0.08, 0.12]`
    - `fusion.memory_retrieval.distance_gate_quantile` in `[0.90, 0.95]`
    - `fusion.memory_retrieval.distance_gate_scale` in `[1.00, 1.10]`
  - keep scope fixed: `target_series=["1_0"]`, `apply_horizons=[6]`, risk disabled.
- Round 2 (3 runs): robustness check around best Round 1 config.
  - one tighter, one looser, one center repeat (stability check).
- Round 3 (3 runs, conditional):
  - only if Round 2 still improves and stable.
  - otherwise stop modeling changes and produce root-cause report.

## 6) Stop Conditions
- Immediate stop if two consecutive runs fail hard reject gates.
- Immediate stop if best score does not improve after a full round (3 runs).
- After stop: output a one-page failure report with:
  - rejected hypotheses
  - metric deltas
  - next pivot recommendation

## 7) Daily Deliverables
- One compact table:
  - `run_id | overall | rolling | 1_0_h6 | pass/fail`
- One explicit next decision:
  - continue / rollback / stop-and-pivot.
