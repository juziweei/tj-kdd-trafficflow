# Research-Oriented Plan (Competition First)

## Stage 1: Strong Baseline
- 20-min aggregation
- leakage-safe rolling validation
- GBDT/XGBoost baseline

## Stage 2: Feature Expansion
- multi-scale temporal lags (hour/day/week)
- weather and traffic composition features
- holiday/regime features

## Stage 3: Robustness + Analysis
- per-tollgate-direction error analysis
- horizon-wise MAPE decomposition
- ablation on key feature groups

## Stage 4: Advanced Model Layer
- global-local fusion
- optional transformer branch
- online model-weight adaptation (bandit/RL-style policy)
