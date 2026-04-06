# 技术路线与系统架构（论文/答辩版）

## 1. 研究问题与挑战
任务为 KDD Cup 2017 收费站交通流预测，目标是对 `(tollgate_id, direction, time_window)` 的 `volume` 做 20 分钟粒度预测，评价指标为 MAPE。

关键挑战：
- 非平稳：工作日/节假日、早晚高峰模式差异明显。
- 多步递推误差传播：horizon 增大时误差易累积。
- 切片异质性：不同收费站方向和 horizon 的误差分布显著不同。

## 2. 技术路线图
`baseline -> gbdt -> strong backbone -> fusion -> post-process`

对应实现入口：
1. Baseline: `scripts/run_baseline.py`
2. GBDT: `scripts/run_gbdt_pipeline.py`
3. Strong Backbone: `scripts/run_strong_backbone_v6.py`
4. 融合与后处理: `scripts/fuse_ensemble.py`, `scripts/fuse_ensemble_by_series_horizon.py`, `scripts/post_process.py`

## 3. 系统整体架构
### 3.1 数据层
- `src/data/volume_io.py`：事件加载、20 分钟聚合、时间网格补全、序列历史构建。
- `src/data/weather_io.py`：天气小时级对齐、缺失填补、风向正余弦编码。

### 3.2 特征层
- `src/features/volume_features.py`：lag/rolling/周期/日历/系列标识特征。
- `src/features/enhanced_features.py`：趋势、波动、密集切片特征与天气交互。

### 3.3 模型层
- `src/models/ridge_linear.py`：线性主干。
- `src/models/tft_model.py`：可选 TFT 分支（按配置启用）。
- `scripts/run_gbdt_pipeline.py`：XGBoost 主体（全局+分组）。

### 3.4 融合层
- `scripts/run_strong_backbone_v6.py`：dual GBDT experts + tri-fusion + router + memory + post head。
- `src/fusion/*`：自适应权重、Nash 融合、Optuna 优化组件。

### 3.5 评估与交付层
- `src/eval/metrics.py`：overall/horizon/series/series-horizon 指标。
- `src/inference/submission.py`：提交格式与 schema 校验。

## 4. 关键技术细节
### 4.1 防泄漏机制
- 时间切分验证（严格 chronology）。
- 特征仅使用预测时刻之前信息（`build_feature_row` 的 lag 访问策略）。
- 天气锚点使用前一整点（`get_weather_feature_vector`）。
- 规则检查脚本：`scripts/check_leakage_guardrails.py`。

### 4.2 特征工程
- 时序滞后：`lag_1,2,3,6,72,504`
- 短窗统计：`mean_prev_6` 及增强统计特征
- 周期信息：`dow_sin/cos`, `slot_sin/cos`, `horizon`
- 日历信息：周末、节假日、节前节后
- 天气外生变量：气压、风速、温湿度、降水、风向编码

### 4.3 建模策略
- Baseline：Ridge + 条件残差 + horizon 偏置校正（可配置）
- GBDT：XGBoost，全局模型 + `(tollgate,direction,horizon)` 分组模型
- 强主干：线性分支 + full/target 双 GBDT 专家 + 递推融合

### 4.4 融合与后处理
- Tri-fusion 动态权重（全局/series/series-horizon）
- Regime router（up/down/stable/conflict）针对结构漂移
- Memory retrieval 对历史局部误差进行检索式修正
- Post-fusion residual head 做最终偏差修正

## 5. 模块表（模块-作用-输入输出-风险）
| 模块 | 作用 | 输入 | 输出 | 风险 |
|---|---|---|---|---|
| `src/data/volume_io.py` | 体量数据标准化与聚合 | 原始过车记录 CSV | 20 分钟窗口序列表 | 时间戳解析错误导致错位 |
| `src/data/weather_io.py` | 天气对齐与可用性约束 | 天气 CSV | 小时级天气特征表 | 错误时间锚点引入泄漏 |
| `src/features/volume_features.py` | 构建主特征 | 历史序列+时间戳+天气 | 单样本特征向量 | lag 缺失处理不当 |
| `scripts/run_baseline.py` | 基线训练/验证/递推 | config + 特征 | 指标/预测/提交 | 长 horizon 误差传导 |
| `scripts/run_gbdt_pipeline.py` | GBDT 管线 | config + 特征 | 指标/预测/提交 | 分组样本不足导致不稳定 |
| `scripts/run_strong_backbone_v6.py` | 强主干融合 | 多分支预测 | 融合预测与指标 | 结构复杂度高，调参成本大 |
| `src/eval/metrics.py` | 统一评估 | 真实值+预测值 | 多粒度 MAPE | EPS 与切片口径需一致 |
| `src/inference/submission.py` | 交付规范 | 预测表 | 合规 submission | schema 不一致导致提交失败 |

## 6. 实验设计与可复现
统一要求：
- 每次实验有 `run_id`
- 配置存放于 `configs/*.json`
- 结果落盘于 `outputs/runs/<run_id>/`
- 会话日志记录在 `docs/vibe_coding_protocol.md`

示例命令：
```bash
python3 scripts/run_baseline.py --config configs/baseline_v1.json
python3 scripts/run_gbdt_pipeline.py --config configs/gbdt_v1.json
python3 scripts/run_strong_backbone_v6.py --config configs/strong_backbone_v6_main.json
python3 scripts/check_leakage_guardrails.py --config configs/strong_backbone_v6_main.json --sample-size 24
```

## 7. 结果与误差切片（真实 run）
代表性结果（来自 `outputs/figures/sci/run_summary.csv`）：
- Baseline 最优：`baseline_residual_hbias_20260328_02`，overall `19.7078`
- GBDT 最优：`gbdt_20260327_01`，overall `21.7822`
- StrongBackbone 最优：`strong_backbone_fusion_20260329_v9_merged_data`，overall `14.7502`
- Generalization 代表：`target12_generalize_20260404_45_r3_v9merged_probe`，overall `15.4807`
- Fusion 代表：`ensemble_seed_42`，overall `14.7502`

切片分析建议：
- 优先关注 `h5/h6` 与高误差 series（如 `1_0`）的结构性偏差。
- 结合 `validation_error_slices.csv` 做 series×horizon 误差闭环优化。

## 8. 局限与下一步
局限：
- 部分方案在单切片上分数很低，但泛化风险高（验证集定制路径）。
- 强主干结构复杂，调参与稳定性验证成本较高。

下一步：
1. 固定低自由度策略，扩大时间外推验证窗口。
2. 对 `h6` 增设结构化误差约束或分层校正。
3. 将最优稳态主线与轻量后处理融合，平衡分数与稳健性。

