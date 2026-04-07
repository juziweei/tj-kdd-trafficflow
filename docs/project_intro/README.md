# 项目介绍资料包（答辩/汇报专用）

这个目录用于集中存放“怎么讲清楚项目”的材料，避免讲稿、复盘、PPT提纲分散在对话记录里。

## 文件说明
- `01_answer_framework_6_questions.md`
  - 六个核心答辩问题（问题价值、设计因果、证据、风险、复现、规划算法）的统一答法。
- `02_non_tech_20min_script.md`
  - 面向非技术听众的 20 分钟口播稿（可直接照读，包含过渡句）。
- `03_ppt_12_pages_talktrack.md`
  - 12 页 PPT 逐页讲解稿（每页标题、核心句、讲解要点、建议图）。
- `04_attempts_and_lessons.md`
  - 关键尝试与失败复盘时间线，适合答辩中的“过程真实性”部分。

## 一分钟使用方法
1. 先看 `01_answer_framework_6_questions.md`，确定答辩主线。
2. 用 `03_ppt_12_pages_talktrack.md` 快速搭出 10-15 页演示稿。
3. 彩排时按 `02_non_tech_20min_script.md` 进行整段讲述。
4. 评委追问“你们具体试了什么”时，引用 `04_attempts_and_lessons.md` 的 run 级证据。

## 证据口径（建议统一）
- 当前 in-split 稳健主线：`overall_mape ≈ 14.95`
- 当前 outer 泛化候选：`overall_mape ≈ 17.60`
- 主线入口：`scripts/run_strong_backbone_v6.py`
- 强制约束：时间切分、`run_id`、submission schema 校验

## 不建议的讲法
- 只报最低分，不讲验证口径。
- 只讲模型名，不讲为什么这样设计。
- 回避失败轮次，导致评委怀疑结果选择性汇报。
