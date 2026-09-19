# Experiment Register

正式实验使用唯一 experiment ID，每次尝试另有 run ID。NEXT_EXPERIMENT 只指向一个待执行正式实验；草案不计为运行。

| Experiment ID | Status | Scope | Evidence / next action |
| --- | --- | --- | --- |
| LEGACY-P0-P7 | HISTORICAL_BLOCKED | 已有checkpoint/forward/reset/stop审计，本次导入索引 | diagnostics/end_causal_audit/gate_review.md；Gate B FAIL |
| LEGACY-FULL200 | HISTORICAL_REPORTED | 历史full-200，本轮未重跑/重算 | diagnostics/end_causal_audit/analysis_validation_report.md；173/200 success，cost总计145 |
| EXP-RESET-001 | DEFERRED | 隔离离线counter起点对rollover/输出的影响 | 已归档至001A/preflight/previous_NEXT_EXPERIMENT.md；优先核查B0身份 |

LEGACY 是引用标签，不伪造原始注册时间或 ID。

后续使用 DRAFT → READY → RUNNING → COMPLETED/FAILED/BLOCKED/INVALID。先归档已执行设计、manifest、results及解释，再替换NEXT。失败/部分run不删除，不挑有利seed；区分工程成功、科学有效与假设支持度。

新条目保留问题、control/treatment/唯一变量、固定条件、run ID、commit/diff、原始产物路径、有效性结论和决策链接。摘要不代替证据。


## 2026-09-11 — B0 reproduction拆分

| Experiment ID | Status | Reference Run / Repeat Run | Scope / evidence |
| --- | --- | --- | --- |
| EXP-B0-REPRO-001A | BLOCKED_PREFLIGHT | NOT_STARTED / NOT_STARTED | 历史前置尝试：授权各2个episode，实际0；research/runs/EXP-B0-REPRO-001A/execution_20260912/before_RESULT_SUMMARY.md |
| EXP-B0-REPRO-001B | NOT_AUTHORIZED / NOT_READY | 均未启动，须为新run | Full 200-task A/A；本轮明令禁止；research/EXP-B0-REPRO-001B_DESIGN.md |

NEXT仅指向001A前置阻断。001B是待人工评审的后续阶段，不是并行待执行任务。
LEGACY-FULL200仅historical reference，不作为任何正式A/A的Reference Run。
前置核查不是smoke run；未分配伪造的run完成记录、episode指标或退出成功状态。

## 2026-09-12 — B0-CANDIDATE-AUDIT

非episode实验；静态审计完成，状态AWAITING_HUMAN_REVIEW。
独立worktree /nvme2/user/qyy/SafeVLA_baseline_clean；reference 2aa82559d272b5f888e53433e258914057f15bed；完整patch SHA256 f8ff5b1c07a07d5ff717f5f262734dc3d7345ca0d88e5cd5c1607a245e6927db。
research/B0_CANDIDATE_AUDIT.md记录全部适配与证据。0 model executions / 0 episodes。
001A不恢复、不改为READY；001B仍NOT_AUTHORIZED。历史86.5%仅historical reference。


## 2026-09-12 — 人工接受DINO基础设施后001A完成（最新状态）

| Experiment ID | Status | Reference Run / Repeat Run | Scope / evidence |
| --- | --- | --- | --- |
| EXP-B0-REPRO-001A | COMPLETED / PROVENANCE PASS | 001A-20260912-reference / 001A-20260912-repeat；各2 completed，exit 0 | 同一B0、2/2 stable task配对、strict load及产物完整性全部通过；runs/EXP-B0-REPRO-001A/RESULT_SUMMARY.md |
| EXP-B0-DINO-EQ-001 | NOT_EXECUTED / NOT_REQUIRED_BY_REVIEW | 无run | 人工接受本地DINO必要基础设施；不要求在线loader RNG/bitwise比较 |
| EXP-B0-REPRO-001B | NOT_AUTHORIZED / PROTOCOL_NOT_FROZEN | 均未启动 | full-200协议、worker/manifest/资源/容差待独立评审，不能因001A通过自动启动 |

Research Question：两个最小run能否确认同一可追溯executable B0并稳定配对？
Hypothesis：冻结代码/资源/协议后identity一致、配对2/2且必需产物齐全。竞争解释为来源漂移、动态ID误用或记录缺失。
唯一变量为运行实例；Reference/Repeat命名，无policy treatment。
固定条件：seed123、worker1、stochastic=true/greedy=false、test_augmentation=true、shuffle=true、同一minival前2 task、horizon600，保留官方reset/counter。
HEAD=2aa82559d272b5f888e53433e258914057f15bed；完整patch SHA256=f8ff5b1c07a07d5ff717f5f262734dc3d7345ca0d88e5cd5c1607a245e6927db。
Required artifacts和完整command、源/权重哈希、episode结果分别保存在execution_20260912/reference/与repeat/；配对及核查为pairing.json、validation.json。
有效性：001A工程/provenance目标PASS；不提供性能等价/提升或reset机制结论。两run轨迹长度不同，未改变baseline。
停止原因：已完成本轮全部获准episode预算；等待人工评审，无commit/push。历史86.5%不作为Reference Run。
