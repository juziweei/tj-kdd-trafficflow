# KDD 交通流预测项目完整介绍（背景 + 技术路线 + 实现细节）

## 1. 项目背景

本项目面向 KDD Cup 2017 高速收费站流量预测任务。核心目标是在严格无未来信息泄漏的前提下，尽可能降低验证/线上口径 MAPE，并保持完整可复现性（配置、run_id、日志、产物）。

问题形式：
- 预测粒度：20 分钟窗口
- 预测对象：`(tollgate_id, direction, time_window) -> volume`
- 评估指标：MAPE（含总体、horizon、收费站方向切片）

---

## 2. 项目目标与设计原则

- 竞赛优先：先追求可量化降分，再做结构优化
- 严格防泄漏：特征与验证必须符合时间因果顺序
- 可复现优先：每次实验必须有 `run_id + config + metrics + artifacts`
- 工程可维护：配置驱动、脚本化入口、统一输出协议

---

## 3. 技术路线（从基线到强主干）

### 阶段 A：泄漏安全 Baseline
- 脚本：`scripts/run_baseline.py`
- 模型：Ridge 线性主干（支持全局 + 分组）
- 特征：lag、多尺度统计、周期、日历、天气、增强特征
- 输出：验证指标、误差切片、submission 文件

### 阶段 B：GBDT 增强分支
- 脚本：`scripts/run_gbdt_pipeline.py`
- 模型：XGBoost 全局 + `(series,horizon)` 分组模型
- 强化点：样本权重策略（`uniform/inverse/inverse_sqrt`）抑制失稳

### 阶段 C：强主干融合（v6）
- 脚本：`scripts/run_strong_backbone_v6.py`
- 结构：线性分支 + 双 GBDT 专家（full/target）+ 三路自适应融合
- 机制：regime router、memory retrieval、后处理校正、可选 TFT 分支

### 阶段 D：实验治理与稳态推进
- 协议：`docs/vibe_coding_protocol.md`
- 检查：`scripts/check_leakage_guardrails.py`
- 目标：每轮实验都可审计、可回放、可比较

---

## 4. 关键实现细节

### 4.1 数据处理层（`src/data/`）
- `volume_io.py`
  - 原始过车事件聚合到 20 分钟窗口
  - 补全时间网格（缺失窗口置零）
  - 构建每个 `(tollgate_id, direction)` 的时序历史
- `weather_io.py`
  - 天气转小时级表并补全缺失
  - 风向转正余弦（`wind_dir_sin/cos`）
  - 使用“预测时刻前一整点”天气，避免未来泄漏

### 4.2 特征层（`src/features/`）
- `volume_features.py`
  - 目标窗口定义：每天早高峰/晚高峰共 12 个预测点
  - 主特征：`lag_1,2,3,6,72,504` + `mean_prev_6`
  - 周期特征：`dow_sin/cos`, `slot_sin/cos`, `horizon`
  - 日历特征：周末、节假日、前后节假日
  - 增强特征：趋势、波动、rush 统计、天气交互等

### 4.3 模型层（`src/models/` + `scripts/`）
- `ridge_linear.py`：线性基线与分组建模底座
- `tft_model.py`：可选时序深度分支（TFT）
- `run_baseline.py`：递推预测 + 偏置校正 + 条件残差
- `run_gbdt_pipeline.py`：XGBoost 方案与时间口径验证
- `run_strong_backbone_v6.py`：当前复杂主干与多策略融合

### 4.4 评估与提交层
- `src/eval/metrics.py`
  - 输出 `overall_mape / horizon_mape / series_mape / series_horizon_mape`
  - 可生成 `tollgate-direction-horizon` 误差切片
- `src/inference/submission.py`
  - 统一格式化 `time_window`
  - 提交 schema 校验（列、主键唯一、非负值）

---

## 5. 反泄漏与实验治理

### 5.1 反泄漏规则
- 只允许时间滚动/阻塞切分，不允许随机切分选模
- 特征只可使用预测时刻之前信息
- 天气特征只取前一整点可用值
- 提交前强制校验 schema 与关键字段

### 5.2 治理机制
- 每次开发前后必须更新 `docs/vibe_coding_protocol.md`
- 每个实验必须记录 `run_id`
- 建议固定 `config + seed + split_timestamp` 做公平比较

---

## 6. 目录说明（建议对外展示）

- `configs/`：实验配置（参数真源）
- `src/`：核心 pipeline 代码
- `scripts/`：可直接执行的训练/融合/检查入口
- `docs/`：方案、日志、复盘、路线文档
- `data/`：本地数据（默认不上传原始数据）
- `outputs/`：实验产物与提交结果（建议不入库）

---

## 7. 一键复现实验（示例）

```bash
# 1) Baseline
python3 scripts/run_baseline.py --config configs/baseline_v1.json

# 2) GBDT pipeline
python3 scripts/run_gbdt_pipeline.py --config configs/gbdt_v1.json

# 3) Strong backbone v6
python3 scripts/run_strong_backbone_v6.py --config configs/strong_backbone_v6_main.json

# 4) Leakage guardrails
python3 scripts/check_leakage_guardrails.py --config configs/strong_backbone_v6_main.json --sample-size 24
```

---

## 8. Git 上传策略（本仓建议）

目标：上传“代码与文档”，不上传“测试代码与结果产物”。

建议保留：
- `src/`, `scripts/`（去除测试脚本）, `configs/`, `docs/`, `README.md`, `AGENTS.md`

建议忽略：
- `tests/`
- `scripts/test_*.py`, `scripts/test_*.sh`
- `outputs/` 下所有实验结果与提交文件
- `data/raw/` 与 `data/processed/` 中真实数据

---

## 9. 给 AI 助手的高质量提示词模板（可直接复用）

### 9.1 实验迭代提示词
```text
你是本仓库的 ML 工程师，请在严格防泄漏前提下，基于现有主线继续降低 MAPE。
要求：
1) 先阅读 AGENTS.md 与 docs/vibe_coding_protocol.md，并新增 Session（In Progress）。
2) 只做最小必要改动，优先改 configs 和单个脚本。
3) 必须使用时间切分验证，禁止随机切分。
4) 输出 run_id、核心指标（overall/horizon/series）、风险、下一步。
5) 回填同一 Session 为 Done/Blocked。
```

### 9.2 代码审查提示词
```text
请对本次改动做“防泄漏 + 回归风险”审查，按严重级别列出问题。
重点检查：
- 特征是否使用未来时间
- 验证是否仍是时间切分
- submission schema 是否可能被破坏
- 是否缺少 run_id / config 快照 / metrics 产物
```

### 9.3 提交整理提示词
```text
请帮我整理一次可提交的 git 变更，目标是“仅提交代码与文档，不提交测试与结果”。
要求：
1) 调整 .gitignore 并验证匹配是否生效；
2) 给出建议 add/commit 命令；
3) 列出仍被跟踪但应排除的文件（如有）并给出 rm --cached 方案。
```

---

## 10. 后续建议

- 若目标是“比赛冲分”，继续做低自由度稳健迭代，避免验证集过拟合式路由。
- 若目标是“项目展示/求职”，优先补充：
  - 系统架构图
  - 一页实验总表（配置 -> 指标 -> 结论）
  - 失败案例剖析（切片误差 + 原因 + 修复动作）
