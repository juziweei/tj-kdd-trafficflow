# KDD Traffic Flow Project Design

## 1. Positioning
This project is competition-first (optimize MAPE) with research-grade reproducibility.

Core goals:
1. Build a leakage-safe short-term traffic forecasting pipeline.
2. Optimize MAPE under non-stationary peak-hour patterns.
3. Keep all experiments auditable and reproducible.

## 2. System Architecture
The system is organized into six layers.

### 2.1 Problem Layer
- Task: forecast `volume` for each `(tollgate_id, direction, time_window)`.
- Resolution: 20-minute windows.
- Hard constraint: no future information at prediction time.
- Objective: minimize MAPE.

### 2.2 Data Governance Layer
Responsibilities:
1. Raw data ingestion and schema normalization.
2. Timestamp alignment and 20-minute aggregation.
3. Missing-value and outlier handling with traceable rules.
4. Data audit outputs (missingness, distribution shift, anomalies).

### 2.3 Feature Layer
Feature families:
1. Autoregressive: lag windows, rolling statistics, trend features.
2. Periodic: hour/day-of-week, same-slot previous day/week.
3. Exogenous: weather and traffic composition indicators.
4. Interaction: peak-period and station-direction interaction terms.

Constraint:
- Every feature must declare availability at prediction time.

### 2.4 Modeling and Fusion Layer
Model stack:
1. Local models per tollgate-direction.
2. Global model for cross-series shared patterns.
3. Fusion module for weighted combination.
4. Optional adaptive routing policy (bandit/RL-style) for dynamic weighting.

### 2.5 Evaluation Layer
Required protocol:
1. Rolling/blocked time split only.
2. Report overall MAPE + horizon-wise MAPE + tollgate-direction MAPE.
3. Run ablation studies on major feature groups.
4. Keep error-slice diagnostics for failure analysis.

### 2.6 Delivery Layer
Artifacts:
1. Standardized prediction output and submission file.
2. Run logs, configs, seeds, metrics, plots.
3. Reproducible commands for train/eval/inference.

## 3. Data Assets and Paths
Current project raw data root:
- `data/raw/dataset_60/`

Required folders:
1. `training/`
2. `testing_phase1/`
3. `dataSet_phase2/`

The pipeline must read only from these canonical paths.

## 4. Data Contract
Primary key:
- `(tollgate_id, direction, time_window)`

Target column:
- `volume`

Contract rules:
1. Time window format must be normalized before modeling.
2. Direction encoding must be consistent (`0=entry`, `1=exit`).
3. No duplicate keys after aggregation.
4. All transformations must be deterministic under fixed seed/config.

## 5. Anti-Leakage Policy
Hard rules:
1. No feature can reference timestamps later than prediction timestamp.
2. Validation must preserve chronological order.
3. No random split in model selection.
4. Any feature with ambiguous time availability is rejected.

## 6. Repository Component Responsibilities
- `src/data/`: ingestion, cleaning, aggregation, supervised sample build.
- `src/features/`: feature generators and feature registry.
- `src/models/`: baseline and core model trainers.
- `src/fusion/`: model blending and adaptive weighting.
- `src/eval/`: metrics, backtest, ablation, error slicing.
- `src/inference/`: prediction and submission generation.
- `configs/`: all experiment settings (no hard-coded experiment params).
- `outputs/runs/`: run artifacts and reproducibility records.

## 7. Experiment Governance
Each experiment must have:
1. `run_id`
2. config snapshot
3. random seed
4. validation protocol record
5. metric summary
6. artifact links/paths

No experiment result is considered valid without these fields.

## 8. Done Criteria
A modeling iteration is done only if:
1. Code runs end-to-end.
2. Time-based validation metric is produced.
3. Error slices are generated.
4. Session log is updated in `docs/vibe_coding_protocol.md`.
