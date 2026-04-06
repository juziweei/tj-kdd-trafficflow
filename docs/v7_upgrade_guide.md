# V7 升级指南：三大改进实现 SOTA

## 改进概览

### 1. 5折Rolling验证（稳定性提升）
- 从2折增加到5折，减少时间窗口过拟合
- 配置：`rolling_validation.n_folds: 5`

### 2. TFT专家分支（长期依赖建模）
- 针对高误差序列 1_0 和 2_0
- 自动学习 horizon-specific 注意力权重
- 位置：`src/models/tft_model.py`

### 3. Optuna融合优化（全局搜索）
- 贝叶斯优化替代手工调参
- 100 trials 搜索最优融合权重
- 位置：`src/fusion/optuna_optimizer.py`

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行基础版（仅5折验证）
```bash
python3 scripts/run_strong_backbone_v6.py \
  --config configs/strong_backbone_v7_5fold_stable.json
```

### 运行完整版（含Optuna优化）
```bash
python3 scripts/run_strong_backbone_v7.py \
  --config configs/strong_backbone_v7_5fold_stable.json \
  --optimize-weights \
  --optuna-trials 100
```

## 预期效果

- 当前最优：16.79 MAPE
- 预期目标：15.5 以下
- 关键指标：rolling 标准差 < 5.0

## 注意事项

1. Optuna优化耗时约1-2小时（100 trials）
2. TFT需要PyTorch，建议GPU加速
3. 5折验证会增加训练时间约2.5倍
