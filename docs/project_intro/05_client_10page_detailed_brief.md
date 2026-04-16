# 甲方汇报详细版（10页，技术方案可审计）

> 适用：甲方汇报、项目阶段评审、商务/技术联合沟通
>
> 原则：只讲可验证结论；每页给“结论 + 证据 + 边界”

---

## 第1页：项目目标与交付范围

### 本页要达成
- 让甲方明确：这个项目解决什么问题、当前交付到什么程度。

### 页面内容
- 任务：KDD Cup 2017 收费站流量预测（20分钟粒度）。
- 指标：MAPE（越低越好）。
- 当前交付：
  - 可运行主线脚本 `scripts/run_strong_backbone_v6.py`
  - 可复验 run 产物、submission 导出与 schema 校验
  - 实验治理日志与技术文档

### 讲稿（可直接说）
本项目当前阶段目标是提供一条可持续优化的流量预测主线，而不是一次性低分结果。我们已经完成主线训练、验证、提交导出和审计链路，具备持续迭代能力。

### 证据
- 运行入口：`scripts/run_strong_backbone_v6.py`
- 审计日志：`docs/vibe_coding_protocol.md`

### 追问预案
- Q: 现在是研究原型还是可交付系统？
- A: 当前是“可复验研发交付态”，具备稳定运行与版本审计，但仍需继续做泛化强化。

---

## 第2页：数据与约束边界

### 本页要达成
- 明确结果成立条件，避免甲方误解“任何口径都同样有效”。

### 页面内容
- 数据粒度：20分钟窗口。
- 强约束：
  1) 时间切分选模（禁止随机切分）
  2) 每次实验必须有 `run_id`
  3) 提交必须通过 schema 校验
  4) 特征不得使用未来信息

### 讲稿
我们把工程约束写成硬规则，目的不是增加流程成本，而是防止“离线分数高、线上不可复验”的情况。所有结论都绑定固定口径和 run 证据。

### 证据
- 规则与流程：`docs/START_HERE.md`
- 防泄漏检查：`scripts/check_leakage_guardrails.py`

### 追问预案
- Q: 为什么不使用随机切分提高稳定性？
- A: 随机切分会破坏时间因果，无法代表未来预测场景。

---

## 第3页：技术路线总览

### 本页要达成
- 让甲方看到“方案是分阶段演进，不是单次堆模型”。

### 页面内容
- 路线：`baseline -> gbdt -> strong backbone -> fusion -> post-process`
- 目标分工：
  - baseline：建立可复验下限
  - gbdt：增强非线性表达
  - strong backbone：综合多分支能力
  - fusion/post：提升稳健性与交付一致性

### 讲稿
技术路线是按问题拆解推进，每个阶段只解决一类核心矛盾，避免结构过快复杂化导致无法归因。

### 证据
- 架构文档：`docs/technical_route_architecture.md`

### 追问预案
- Q: 为什么不用单一深度模型？
- A: 当前数据规模和场景下，混合架构在可解释性和稳定性上更可控。

---

## 第4页：当前主线方案（可运行）

### 本页要达成
- 给出当前“实际在跑”的方案，而不是概念图。

### 页面内容
- 主线配置：`configs/strong_backbone_v6_density_r2_glw_r1_auto42_r8_score_shift.json`
- 主线运行：
```bash
python3 scripts/run_strong_backbone_v6.py \
  --config configs/strong_backbone_v6_density_r2_glw_r1_auto42_r8_score_shift.json \
  --run-id <run_id>
```
- 输出：`metrics.json`、`validation_predictions.csv`、`validation_error_slices.csv`、submission 文件。

### 讲稿
本页只讲当前可重复执行入口。我们汇报的所有结果都从这条入口产生，避免“多入口结果混杂”。

### 证据
- 运行产物示例：`outputs/runs/mainline_train_20260416_61_auto42r8/`

### 追问预案
- Q: 一次运行是否能完整产出交付文件？
- A: 能，训练、验证、切片、submission 会同 run_id 一起落盘。

---

## 第5页：验证口径与比较规则

### 本页要达成
- 防止比较口径错误（这是甲方最容易误解的位置）。

### 页面内容
- 比较规则：只有同 split 的结果可直接横向比较。
- 当前采用：固定时间切分或连续时间小块验证（小数据不以 rolling 为主选模）。
- 强调：`split=2016-10-11` 与 `split=2016-10-18` 结果不能直接并排下最终结论。

### 讲稿
为了避免误判，我们把比较规则写死：同口径才能比较。如果 split 不同，结果仅作参考，不作最终优劣判定。

### 证据
- `mainline_train_20260416_61_auto42r8` split=`2016-10-11`
- `strong_backbone_fusion_20260329_v9_merged_data` split=`2016-10-18`

### 追问预案
- Q: 那为什么历史上有更低分？
- A: 口径不同。我们可以展示，但不会当作同口径结论。

---

## 第6页：横向对比（同口径）

### 本页要达成
- 证明当前主线相对基线有真实价值。

### 页面内容（同 split=2016-10-11）
- `baseline_residual_hbias_20260328_02`: `19.7078`
- `gbdt_20260327_01`: `21.7822`
- `mainline_train_20260416_61_auto42r8`: `14.9426`

### 讲稿
在同一时间切分口径下，当前主线从 baseline 的 19.7078 降到 14.9426，绝对改善 4.7652，约 24.18%。这说明主线具有稳定的实际提升，而不是口径切换导致的假收益。

### 证据
- 绝对改善：`4.7652`
- 相对改善：`24.18%`

### 追问预案
- Q: 为什么 GBDT 比 baseline 差？
- A: 说明该单分支在当前口径不稳定，正是强主干融合存在的价值。

---

## 第7页：竖向对比（同路线迭代）

### 本页要达成
- 展示“微迭代有意义，但边际收益受限”。

### 页面内容（outer split=2016-10-15）
- 单模型主线：`target12_generalize_20260404_45_r1_split1015 = 17.6376`
- 固定融合：`target12_generalize_20260404_53_fixedblend_r1_92_long08 = 17.6014`
- 改善：`0.0362`（约 `0.205%`）

### 讲稿
在外段泛化口径下，我们采用低自由度固定融合，得到可复验的小幅提升。这个量级不大，但方向稳定，符合当前小数据阶段“先稳后快”的策略。

### 证据
- run_id 级对照：`45_r1_split1015` vs `53_fixedblend`

### 追问预案
- Q: 改善这么小，是否值得？
- A: 在泛化阶段小幅稳定提升比激进波动更可交付。

---

## 第8页：局限性与风险

### 本页要达成
- 让甲方预期管理到位，避免过度承诺。

### 页面内容
1. 数据规模有限，验证方差较高。  
2. 长 horizon（特别 h6）仍是误差高点。  
3. 强主干结构复杂，调参与归因成本高。  
4. 高自由度方案有过拟合风险。  
5. 不同 split 结果不可直接混比。

### 讲稿
当前结果是有效的，但边界也明确存在。我们不回避风险，后续优化优先级会围绕“泛化稳定性”和“长 horizon 控制”展开。

### 证据
- 当前 run 的 `h6` 误差高于短 horizon（见 `metrics.horizon_mape`）

### 追问预案
- Q: 是否存在上线后回撤风险？
- A: 有，需要通过低自由度策略和持续外段验证控制。

---

## 第9页：参考论文方法映射（来源清晰）

### 本页要达成
- 说明“借鉴了什么”，避免原创归属不清。

### 页面内容
- 线性基线：Ridge 思路（传统统计学习）。
- GBDT：XGBoost（Chen & Guestrin, 2016）。
- 时序深度分支：Temporal Fusion Transformer（Lim et al., 2019/2021）。
- 超参搜索组件：Optuna（Akiba et al., 2019）。

### 讲稿
我们明确区分“论文已提出方法”和“本项目工程组合创新”。这样做有利于甲方判断方案成熟度和可维护性。

### 证据
- 代码入口：`src/models/tft_model.py`, `scripts/run_gbdt_pipeline.py`, `src/fusion/optuna_optimizer.py`

### 追问预案
- Q: 这些方法是否都是你们原创？
- A: 不是。基础方法来自公开研究，我们的创新在组合结构与治理闭环。

---

## 第10页：我们自研结构与差异化价值

### 本页要达成
- 讲清“你们到底新做了什么”。

### 页面内容
- 自研/自定义工程结构：
  1) Dual GBDT Experts（full/target）
  2) Tri-fusion 权重体系（global + series + slice）
  3) Regime Router（up/down/stable/conflict）
  4) Memory Retrieval 误差检索修正
  5) Post-fusion Residual Head 偏差校正
  6) 实验治理闭环（run_id + guardrails + session log + schema）

### 讲稿
我们的差异化不在单一算法，而在系统结构和可审计迭代能力。这个结构让模型改动可以被量化评估并回滚，是可持续交付的基础。

### 证据
- 主线脚本：`scripts/run_strong_backbone_v6.py`
- 治理脚本：`scripts/check_leakage_guardrails.py`
- 会话协议：`docs/vibe_coding_protocol.md`

### 追问预案
- Q: 这套结构对后续扩展的价值是什么？
- A: 能在同一审计框架下接入新模型，不破坏既有验证与交付链路。

---

## 附录A：本次汇报引用的关键 run 与口径
- `baseline_residual_hbias_20260328_02` (`split=2016-10-11`, `overall=19.7078`)
- `gbdt_20260327_01` (`split=2016-10-11`, `overall=21.7822`)
- `mainline_train_20260416_61_auto42r8` (`split=2016-10-11`, `overall=14.9426`)
- `target12_generalize_20260404_45_r1_split1015` (`split=2016-10-15`, `overall=17.6376`)
- `target12_generalize_20260404_53_fixedblend_r1_92_long08` (`split=2016-10-15`, `overall=17.6014`)

## 附录B：汇报纪律（对外口径统一）
1. 所有结论必须附 run_id。
2. 不同 split 不做直接优劣结论。
3. 必须同步报告局限性，不只报最优结果。
4. 甲方可复验路径：命令、配置、产物路径三项同时给出。
