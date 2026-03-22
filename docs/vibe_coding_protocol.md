# Vibe Coding 规范与进度日志（强制执行）

> 目标：在“竞赛成绩优先”前提下，保持科研级可复现与可审计。

## 1. 使用规则（必须）
1. 每次开始编码前，先在本文档新增一条 `Session` 记录（状态=In Progress）。
2. 每次结束编码后，必须回填结果（状态=Done/Blocked）与下一步。
3. 未写日志，不允许提交代码。
4. 每个实验必须有唯一 `run_id`，并记录配置、数据切分和核心指标。
5. 禁止时间泄漏：任何特征不得使用预测时刻之后的信息。

## 2. 每次 Vibe Coding 的标准流程
1. 定义本次目标：一句话说明“要提升什么指标/修复什么问题”。
2. 锁定边界：改哪些文件，不改哪些文件。
3. 最小实现：先做可运行版本，再做增强。
4. 本地验证：至少跑一个时间滚动验证切片。
5. 记录结论：写入本文件的 Session 日志。

## 3. 强制检查清单（提交前自检）
- [ ] 本次改动目标明确，且和竞赛指标（MAPE）相关。
- [ ] 数据处理符合因果顺序（无未来信息）。
- [ ] 验证方式是时间切分，不是随机切分。
- [ ] 输出了可复现实验信息（run_id、seed、config）。
- [ ] 关键结果写入本文档（指标、发现、风险）。

## 4. Session 记录模板（复制使用）
```md
### Session YYYY-MM-DD-XX
- Time: 2026-03-22 00:00:00 CST
- Owner: juziweei / Codex
- Goal: （本次目标）
- Scope: （本次改动文件）
- Run ID: （如有）
- Validation: （时间切分方式）
- Result: （核心指标和现象）
- Status: In Progress / Done / Blocked
- Next: （下一步）
```

## 5. 里程碑看板（持续更新）
- M1 数据基线管线（20分钟聚合 + 防泄漏样本）: Done
- M2 baseline 模型与本地 backtest: Pending
- M3 特征工程迭代（lag/周期/天气/构成）: Pending
- M4 模型融合与误差分解: Pending
- M5 最终提交与汇报材料: Pending

## 6. Session 日志（持续追加）
### Session 2026-03-22-01
- Time: 2026-03-22 17:21:29 CST
- Owner: juziweei / Codex
- Goal: 初始化项目并建立竞赛优先 + 科研规范的开发框架。
- Scope: `.gitignore`, `README.md`, `docs/research_plan.md`, `scripts/create_private_repo.sh`, 目录骨架
- Run ID: N/A
- Validation: N/A（基础设施阶段）
- Result: 本地仓库初始化完成；私有仓库创建并推送成功（origin/main）。
- Status: Done
- Next: 开始 M2，落地可运行 baseline（数据聚合 -> 特征 -> 训练 -> 本地MAPE）。

### Session 2026-03-22-02
- Time: 2026-03-22 17:21:29 CST
- Owner: juziweei / Codex
- Goal: 建立“Vibe Coding 强制规范 + 进度持续更新”机制。
- Scope: `docs/vibe_coding_protocol.md`, `README.md`
- Run ID: N/A
- Validation: N/A（流程治理阶段）
- Result: 规范文档创建完成，含模板、检查清单、里程碑和日志区。
- Status: Done
- Next: 从下一次编码开始，每次先追加 Session，再改代码并回填结果。
