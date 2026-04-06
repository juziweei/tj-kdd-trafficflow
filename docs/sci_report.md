# SCI-Style Experimental Report: KDD Traffic Forecasting

## 1. Research Background
This project targets highway tollgate traffic forecasting in the KDD Cup 2017 setting. The objective is to minimize MAPE under strict anti-leakage constraints and time-based validation only.

Task target:
- Predict `volume` for each `(tollgate_id, direction, time_window)` in 20-minute windows.
- Evaluate by `overall/horizon/series` MAPE.

## 2. Problem Definition
Given historical tollgate vehicle events and weather observations, forecast future tollgate flow for fixed peak-time windows.

Core challenges:
- Non-stationary temporal dynamics (weekday/holiday and peak-shift effects).
- Multi-horizon recursive error accumulation.
- Distribution shift between validation slices.

## 3. Data and Anti-Leakage Protocol
Data and feature pipeline:
- Volume ingestion and 20-minute aggregation: `src/data/volume_io.py`
- Weather alignment and hourly filling: `src/data/weather_io.py`
- Leakage-safe feature construction: `src/features/volume_features.py`

Anti-leakage controls:
- Time split only; no random split for model selection.
- Features at timestamp `t` only use historical information `<= t-20min`.
- Weather uses previous full hour anchor (`ts.floor('h') - 1h`) in `get_weather_feature_vector`.

Guardrail script:
- `scripts/check_leakage_guardrails.py`

## 4. Technical Route
Pipeline route used in this repository:
1. `baseline` (ridge + residual/horizon-bias correction)  
   entry: `scripts/run_baseline.py`
2. `gbdt` (XGBoost global+grouped)  
   entry: `scripts/run_gbdt_pipeline.py`
3. `strong backbone v6` (dual GBDT experts + tri-fusion + post head)  
   entry: `scripts/run_strong_backbone_v6.py`
4. Fusion and post-processing variants (ensemble/fixed blend and controlled routing).

## 5. System Architecture
Layer mapping:
- Data layer: `src/data/*`
- Feature layer: `src/features/*`
- Model layer: `src/models/*`
- Fusion layer: `src/fusion/*` + `scripts/run_strong_backbone_v6.py`
- Evaluation layer: `src/eval/metrics.py`
- Delivery layer: `src/inference/submission.py`

## 6. Core Experimental Setup
Automatic scan and plotting were executed by:
- `scripts/plot_sci_figures.py`

Scanned artifacts:
- `outputs/runs/*/metrics.json`
- `outputs/runs/*/validation_error_slices.csv` (when present)

Scan result (from `docs/figure_manifest.md`):
- Candidate runs: `304`
- Valid runs with numeric `metrics.overall_mape`: `301`
- Filtered invalid runs: `3` (`missing_metrics_block`)

Representative runs selected for cross-family comparison:
- `baseline_residual_hbias_20260328_02`: overall `19.7078`
- `gbdt_20260327_01`: overall `21.7822`
- `strong_backbone_fusion_20260329_v9_merged_data`: overall `14.7502`
- `target12_generalize_20260404_45_r3_v9merged_probe`: overall `15.4807`
- `ensemble_seed_42`: overall `14.7502`
- `dense_features_test_v1`: overall `19.1144`

## 7. Result Analysis
### 7.1 Overall comparison (Figure 1)
- Best representative overall MAPE is `14.7502` (`strong_backbone_fusion_20260329_v9_merged_data` and `ensemble_seed_42`).
- Relative to baseline (`19.7078`), the strong backbone family improves by about `4.96` absolute MAPE.

### 7.2 Horizon behavior (Figure 2)
For `strong_backbone_fusion_20260329_v9_merged_data`:
- Horizon MAPE: `[11.5009, 13.2227, 13.7191, 13.8779, 15.6926, 20.4881]`
- Main degradation appears on longer horizon (`h6`), consistent with recursive uncertainty amplification.

For baseline (`baseline_residual_hbias_20260328_02`):
- Horizon MAPE: `[13.5012, 15.1434, 21.1038, 20.2889, 21.6305, 26.5789]`

### 7.3 Series×Horizon slices (Figure 3)
Heatmap source run:
- `strong_backbone_fusion_20260329_v9_merged_data`

Observation:
- Slice variance remains significant across `(series, horizon)`; `h5/h6` slices dominate residual error.

### 7.4 Ablation delta (Figure 4)
Reference:
- `ablation_all_20260327_182855`, overall `20.9245`

Best single weather-column drop:
- `ablation_drop_weather_wind_dir_sin_20260327_182855`, overall `20.7917`, delta `+0.1328` vs reference.

### 7.5 Stability (Figure 5)
Boxplot uses rolling-fold distributions from runs with available folds.
Key note:
- `dense_features_test_v1` shows very high rolling average (`101.4186`), indicating poor cross-slice stability despite moderate single-slice score.

## 8. Risks and Limitations
- Some ultra-low runs (e.g., highly route-tailored/oracle-style entries) are not used as representative family baselines in Figure 1/2 due to overfitting risk.
- A few runs miss complete `horizon_mape` or `series_horizon_mape`; those are excluded from specific figures and documented in manifest.
- Waterfall chart is based on independent single-drop ablations; deltas are not additive.

## 9. Generated Figures and Files
Figures (PNG + PDF, 600 dpi PNG) are in:
- `outputs/figures/sci/figure1_overall_mape.*`
- `outputs/figures/sci/figure2_horizon_mape.*`
- `outputs/figures/sci/figure3_series_horizon_heatmap.*`
- `outputs/figures/sci/figure4_ablation_waterfall.*`
- `outputs/figures/sci/figure5_stability_boxplot.*`

Index and provenance:
- `docs/figure_manifest.md`
- `outputs/figures/sci/run_summary.csv`

## 10. Reproducible Commands
```bash
# Generate all SCI-style figures + manifest + run summary
python3 scripts/plot_sci_figures.py

# Main report support files
ls outputs/figures/sci
sed -n '1,220p' docs/figure_manifest.md
```

