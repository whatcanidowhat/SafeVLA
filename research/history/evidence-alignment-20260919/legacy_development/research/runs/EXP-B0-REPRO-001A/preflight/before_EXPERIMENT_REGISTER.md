# Experiment Register

正式实验使用唯一 experiment ID，每次尝试另有 run ID。NEXT_EXPERIMENT 只指向一个待执行正式实验；草案不计为运行。

| Experiment ID | Status | Scope | Evidence / next action |
| --- | --- | --- | --- |
| LEGACY-P0-P7 | HISTORICAL_BLOCKED | 已有checkpoint/forward/reset/stop审计，本次导入索引 | diagnostics/end_causal_audit/gate_review.md；Gate B FAIL |
| LEGACY-FULL200 | HISTORICAL_REPORTED | 历史full-200，本轮未重跑/重算 | diagnostics/end_causal_audit/analysis_validation_report.md；173/200 success，cost总计145 |
| EXP-RESET-001 | DRAFT_BLOCKED | 隔离离线counter起点对rollover/输出的影响 | NEXT_EXPERIMENT.md；输入/分支/参数/执行授权待齐备 |

LEGACY 是引用标签，不伪造原始注册时间或 ID。

后续使用 DRAFT → READY → RUNNING → COMPLETED/FAILED/BLOCKED/INVALID。先归档已执行设计、manifest、results及解释，再替换NEXT。失败/部分run不删除，不挑有利seed；区分工程成功、科学有效与假设支持度。

新条目保留问题、control/treatment/唯一变量、固定条件、run ID、commit/diff、原始产物路径、有效性结论和决策链接。摘要不代替证据。
