# Figure Manifest (SCI Style)

## 1. Scan Summary

- Total candidate runs scanned: **304**
- Valid runs with numeric `metrics.overall_mape`: **301**
- Filtered invalid runs:
  - `missing_metrics_block`: 3

## 2. Representative Runs

| Label | run_id | category | overall_mape | split_timestamp | rolling_folds |
|---|---|---:|---:|---|---:|
| Baseline | `baseline_residual_hbias_20260328_02` | baseline | 19.7078 | 2016-10-11 00:00:00 | 2 |
| GBDT | `gbdt_20260327_01` | gbdt | 21.7822 | 2016-10-11 00:00:00 | 0 |
| StrongBackbone | `strong_backbone_fusion_20260329_v9_merged_data` | strong_backbone | 14.7502 | 2016-10-18 00:00:00 | 4 |
| Generalization | `target12_generalize_20260404_45_r3_v9merged_probe` | generalization | 15.4807 | 2016-10-22 00:00:00 | 0 |
| Ensemble | `ensemble_seed_42` | fusion | 14.7502 | 2016-10-18 00:00:00 | 4 |
| DenseFeatures | `dense_features_test_v1` | enhanced | 19.1144 | 2016-10-11 00:00:00 | 2 |

## 3. Figure Data Sources

### Figure 1
- Plot type: `bar_with_error`
- Source runs:
  - `baseline_residual_hbias_20260328_02`
  - `gbdt_20260327_01`
  - `strong_backbone_fusion_20260329_v9_merged_data`
  - `target12_generalize_20260404_45_r3_v9merged_probe`
  - `ensemble_seed_42`
  - `dense_features_test_v1`
- Notes: Error bars use rolling-fold std when available; otherwise 0.

### Figure 2
- Plot type: `line`
- Source runs:
  - `baseline_residual_hbias_20260328_02`
  - `gbdt_20260327_01`
  - `strong_backbone_fusion_20260329_v9_merged_data`
  - `target12_generalize_20260404_45_r3_v9merged_probe`
  - `ensemble_seed_42`
- Notes: Only runs with complete horizon_mape(1..6) are used.

### Figure 3
- Plot type: `heatmap`
- Source runs:
  - `strong_backbone_fusion_20260329_v9_merged_data`
- Notes: Heatmap is built from metrics.series_horizon_mape.

### Figure 4
- Plot type: `waterfall_delta`
- Source runs:
  - `ablation_all_20260327_182855`
  - `ablation_drop_weather_wind_dir_sin_20260327_182855`
  - `ablation_drop_weather_wind_speed_20260327_182855`
  - `ablation_drop_weather_wind_dir_cos_20260327_182855`
  - `ablation_drop_weather_precipitation_20260327_182855`
  - `ablation_drop_weather_sea_pressure_20260327_182855`
  - `ablation_drop_weather_pressure_20260327_182855`
  - `ablation_drop_weather_temperature_20260327_182855`
  - `ablation_drop_weather_rel_humidity_20260327_182855`
- Notes: Deltas are independent single-drop ablations vs ablation_all reference.

### Figure 5
- Plot type: `boxplot`
- Source runs:
  - `baseline_residual_hbias_20260328_02`
  - `strong_backbone_fusion_20260329_v9_merged_data`
  - `ensemble_seed_42`
  - `dense_features_test_v1`
  - `auto_accept_20260404_42_r3_combo`
  - `auto_accept_20260404_42_r5_combo_memrollback`
- Notes: Only runs with >=2 rolling folds are included.

## 4. Filtering Notes

- none
