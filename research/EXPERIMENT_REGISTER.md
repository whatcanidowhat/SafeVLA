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


## SUPERSEDED HISTORICAL STATE — 2026-09-11 B0 reproduction拆分

| Experiment ID | Status | Reference Run / Repeat Run | Scope / evidence |
| --- | --- | --- | --- |
| EXP-B0-REPRO-001A | BLOCKED_PREFLIGHT | NOT_STARTED / NOT_STARTED | 历史前置尝试：授权各2个episode，实际0；research/runs/EXP-B0-REPRO-001A/execution_20260912/before_RESULT_SUMMARY.md |
| EXP-B0-REPRO-001B | NOT_AUTHORIZED / NOT_READY | 均未启动，须为新run | Full 200-task A/A；本轮明令禁止；research/EXP-B0-REPRO-001B_DESIGN.md |

历史当时的NEXT仅指向001A前置阻断；该状态已被后续001A完成记录覆盖。001B是待人工评审的后续阶段，不是并行待执行任务。
LEGACY-FULL200仅historical reference，不作为任何正式A/A的Reference Run。
前置核查不是smoke run；未分配伪造的run完成记录、episode指标或退出成功状态。

## SUPERSEDED HISTORICAL STATE — 2026-09-12 B0-CANDIDATE-AUDIT

非episode实验；静态审计完成，状态AWAITING_HUMAN_REVIEW。
独立worktree /nvme2/user/qyy/SafeVLA_baseline_clean；reference 2aa82559d272b5f888e53433e258914057f15bed；完整patch SHA256 f8ff5b1c07a07d5ff717f5f262734dc3d7345ca0d88e5cd5c1607a245e6927db。
research/B0_CANDIDATE_AUDIT.md记录全部适配与证据。0 model executions / 0 episodes。
历史静态审计阶段曾规定001A不恢复、不改为READY；此限制后经人工批准与完成记录覆盖；001B仍NOT_AUTHORIZED。历史86.5%仅historical reference。


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

## Current control draft — 2026-09-13

| Experiment ID | Status | Actual execution | Next actor |
| --- | --- | --- | --- |
| EXP-B0-REPRO-001A | COMPLETED / provenance PASS / awaiting PI review | Reference2 + Repeat2；禁止重复 | PI |
| EXP-B0-REPRO-001B | NOT_AUTHORIZED / PROTOCOL_NOT_FROZEN | 未执行 | PI |
| EXP-LOOP-HANDOFF-001 | DRAFT / PI_REVIEW / NOT_AUTHORIZED | 未执行；预算0GPU/0episode | PI |

本轮bootstrap是控制面建设，不是EXP-LOOP-HANDOFF-001执行；测试使用EXP-TEST合成fixture。

## Latest execution — EXP-LOOP-HANDOFF-001 / 2026-09-14

Status: Executor NOOP_COMPLETED / AWAITING_PI_REVIEW; PI receipt PENDING. Prior DRAFT/NOT_AUTHORIZED entry is historical and superseded. One print-only command, exit0, GPU0, episodes0. Instruction 0dba6b748f26a782a595edeb53f02083850631cb; claim 16748eab9993b568faf2e78213df29a6ebb31d81; claim_id 20aeabe36980458098074f196ec55343. All four handoff outputs are under research/handoffs/handoff-001-20260913/. No retry, no new experiment, no 001B.


## 2026-09-18 — mainline redirected to small-target phenotype

| Experiment ID | Status | Actual execution | Scientific role |
| --- | --- | --- | --- |
| EXP-B0-REPRO-001B-PREFLIGHT | SUPERSEDED / AUTH_EXPIRED / NOT_CLAIMED | no claim/result on control branch | B0 protocol polish deferred; not current bottleneck |
| EXP-SMALLTARGET-PHENOTYPE-001 | APPROVED DESIGN / execution gate follows LOOP_STATE | none | Test whether reported low-SR categories support a genuine policy-independent target-size association before mechanism Probe/intervention |

EXP-SMALLTARGET-PHENOTYPE-001 is observational, not a treatment. Primary explanatory variable is static 3D target size from scene metadata; post-rollout visible pixels are downstream phenotype only. Planned handoff: research/handoffs/smalltarget-phenotype-001-20260918/.


## 2026-09-20 — EXP-SMALLTARGET-PHENOTYPE-001 metadata gate BLOCKED

Claim 72634cf7c3a954dd99456b2d75c3af539da9e68e bound renewed PI approval 290b426ac65d35ef1f8cb4819e9662c4fae1bb31; claim CI 35499476480 succeeded before execution. Budget used: 0 GPU / 0 episode / 0 model load / 0 simulator launch.

The metadata-first audit mapped 200 tasks and all 368 broad-synset target IDs to scene objects. Candidate asset bounding boxes cover only 42 targets; 326 lack candidates in inspected static sources. Only 31 tasks are candidate-complete, and no candidate has been promoted to validated scene-instance physical size: units/scale/transformation equivalence remains unestablished. Thus validated size count and analyzable n are 0. The bounded search does not establish global nonexistence of another static source.

The pre-registered stop condition is met: return BLOCKED to PI. No distance, visible-pixel or category-name proxy, incomplete-target median, category statistical analysis, tertile, association or regression was used. episode_table.csv preserves 200 historical raw rows with empty size cells; category_sr.csv explicitly records NOT_EXECUTED with blank statistics. H-SIZE remains untested.

Handoff: [RESULT_SUMMARY.md](handoffs/smalltarget-phenotype-001-20260918/RESULT_SUMMARY.md), with metadata coverage, source hashes, raw identity checks and all required outputs. Proposed prerequisite only: a version-bound static geometry and instance-transform source covering all broad targets, including built-in THOR assets. No new experiment is approved; executor STOP after publication. Existing development changes and frozen NEXT_EXPERIMENT files are preserved.


## 2026-09-20 — PI review after metadata BLOCKED

| Experiment ID | Status | Actual execution | Scientific role |
| --- | --- | --- | --- |
| EXP-SMALLTARGET-PHENOTYPE-001 | BLOCKED / PI ACKNOWLEDGED | static-only audit; 0GPU/0episode; validated size n=0 | H-SIZE remains untested; static geometry source insufficient |
| EXP-SIZE-RUNTIME-METADATA-001 | DRAFT / NOT_AUTHORIZED | none | Validate exact-runtime initial-state target AABB coverage/repeatability before any size-performance analysis |

The new cycle is `size-runtime-metadata-001-20260920`. It is a measurement prerequisite, not a policy treatment and not an SR experiment.


## 2026-09-20 — EXP-SIZE-RUNTIME-METADATA-001 approved

| Experiment ID | Status | Execution budget | Scientific role |
| --- | --- | --- | --- |
| EXP-SIZE-RUNTIME-METADATA-001 | APPROVED_FOR_CODEX / awaiting claim | max 1 simulator-graphics GPU, 0 episodes, 0 model loads, <=224 scene initializations | Validate exact-runtime initial-state scene-instance AABB coverage/repeatability; no outcome association |

Approval does not imply H-RUNTIME-AABB is true. If exact historical build identity, exact-ID mapping, primary AABB validity or repeatability fails, return BLOCKED without imputation or downstream analysis.

## 2026-09-21 — Runtime metadata handoff: BLOCKED / next actor PI

EXP-SIZE-RUNTIME-METADATA-001 completed 24 preflight and 20 full scene initializations before a BrokenPipeError in its stdout progress print. No retry or resume was performed. Preflight: 12 tasks loaded twice, 18 exact-equal target comparisons, max absolute/relative dimension difference 0. Partial full pass: 26/368 exact-mapped valid AABBs, 20/200 complete tasks; 342 targets / 180 tasks unattempted, not observed missing geometry. The complete-coverage hypothesis remains unresolved.

One simulator graphics GPU, 44/224 scene initializations, 0 episodes, 0 model/checkpoint loads, 0 Actor/Critic forwards. Geometry extraction read no outcomes and made no size-success association. Creation-state world-axis AABB is not intrinsic volume or demonstrated settled evaluation-state geometry. Development HEAD/diff/status and frozen NEXT_EXPERIMENT files are preserved.

Handoff: [RESULT_SUMMARY.md](handoffs/size-runtime-metadata-001-20260920/RESULT_SUMMARY.md), [coverage_report.md](handoffs/size-runtime-metadata-001-20260920/coverage_report.md), [ARTIFACT_INDEX.json](handoffs/size-runtime-metadata-001-20260920/ARTIFACT_INDEX.json). Original traceback, script, raw snapshots, partial CSVs and independent validation are Git-readable. Required outputs are complete; scientific extraction is incomplete.

The only next action is PI review. A fresh transport-resilient extraction cycle is a proposal only, requiring an explicit design and budget that account for this partial run. The current claim is terminal and must not auto-resume. No further experiment or PI acknowledgement is authorized or fabricated. Executor STOP after handoff publication.


## 2026-09-22 — PI acknowledgement of runtime metadata BLOCKED

| Experiment ID | Status | Actual execution | Evidence status |
| --- | --- | --- | --- |
| EXP-SIZE-RUNTIME-METADATA-001 | BLOCKED / PI ACKNOWLEDGED | 24 preflight + 20 full scene initializations; 0 episodes | 18/18 repeat comparisons exact; 26/26 attempted full targets valid; complete coverage unresolved due BrokenPipe output interruption |

Old claim is closed and cannot resume. Any completion attempt requires a fresh cycle.


## 2026-09-22 — transport-resilient geometry completion draft

| Experiment ID | Status | Planned execution | Scientific role |
| --- | --- | --- | --- |
| EXP-SIZE-RUNTIME-METADATA-002 | DRAFT / NOT_AUTHORIZED | fresh 200 task loads; first 20 are frozen overlap control; max1 simulator GPU / 0 episodes | Complete and validate runtime scene-instance AABB coverage without outcome access |

No old/new row splicing is allowed for the primary dataset. Complete geometry recovery still does not authorize H-SIZE testing.


## 2026-09-22 — mainline redirected to failure-stage localization

| Experiment ID | Status | Reason |
| --- | --- | --- |
| EXP-SIZE-RUNTIME-METADATA-002 | SUPERSEDED BEFORE APPROVAL / NOT EXECUTED | Researcher deprioritized further physical-size measurement |
| EXP-LOW-SR-STAGE-LOCALIZATION-001 | DRAFT / NOT_AUTHORIZED | Formalize task-difficulty proxy and localize low-SR failures to room-arrival / camera-encounter / post-visibility stage |

No model execution is authorized by this staging commit.


## 2026-09-22 — 02-mainline diagnostic redesign

| Experiment ID | Status | Reason |
| --- | --- | --- |
| EXP-LOW-SR-STAGE-LOCALIZATION-001 | SUPERSEDED BEFORE APPROVAL / NOT EXECUTED | Stage taxonomy alone did not test the established 02 representation-vs-use question |
| EXP-SEMANTIC-DECISION-MISMATCH-001 | DRAFT / NOT_AUTHORIZED | Clean-B0 targeted causal-path diagnosis: exploration -> representation -> Actor use/readout -> termination |

Planned max budget: 1 GPU, 36 diagnostic episodes. No online behavior intervention is part of this draft.


## 2026-09-23 — 02 Gate-B order restored

| Experiment ID | Status | Reason |
| --- | --- | --- |
| EXP-SEMANTIC-DECISION-MISMATCH-001 | SUPERSEDED BEFORE APPROVAL / NOT EXECUTED | 02 branch explicitly blocks formal hidden-state/Probe interpretation until reset rollover is causally bounded |
| EXP-RESET-ROLLOVER-CAUSAL-001 | DRAFT / NOT_AUTHORIZED | Real-input offline counterfactual test of official carried reset vs clean reset around max_steps=500 rollover |

Planned budget: max1 GPU, max32 read-only B0 capture episodes; offline replay has no simulator actions.


## 2026-09-23 — premature-end dynamics becomes the next diagnostic

| Experiment ID | Status | Reason |
| --- | --- | --- |
| EXP-RESET-ROLLOVER-CAUSAL-001 | SUPERSEDED BEFORE APPROVAL / DEFERRED | Gate B remains open for future hidden-state/Probe interpretation, but it does not block a zero-rollout audit of already-recorded Actor end probabilities/actions |
| EXP-PREMATURE-END-DYNAMICS-001 | DRAFT / NOT_AUTHORIZED | Characterize all historical sub-horizon failures as early-high-prior, later-rise, low-probability stochastic end, or mixed before attributing cause to SafeRL/representation |

Planned budget: 0 GPU / 0 episode / 0 model forward / 0 simulator. Existing historical videos and aligned tables only.

## 2026-09-23 — EXP-PREMATURE-END-DYNAMICS-001 completed / awaiting PI review

All 16 historical sub-horizon failure videos were recovered reliably and confirm final executed end. Frozen morphology counts: LATE_RISE 13/16, EARLY_HIGH_PRIOR 1/16, OTHER_OR_UNCLEAR 2/16, LOW_PROB_END 0/16. All initial end bars were sub-resolution; all preterminal maxima were below 0.5. The 14 first crossings of 0.5 occurred at the terminal decision itself. The one early-high case is a two-step episode, not evidence of an elevated first-decision prior.

All 16 same-category expert-length matches were recovered (11 unique successes, reused). Ten of those 11 successes also first crossed 0.5 at successful termination. Thirteen prescribed control windows include the success terminal frame; expert-length matching gaps have median 34 and range 2–133. The abrupt terminal jump is therefore not failure-specific and does not establish SafeRL causation. Target scope: STRICT 12 / AMBIGUOUS 4; narrow visibility cannot establish broad-target invisibility in ambiguous cases.

This was historical CPU-only video/table analysis: 0 GPU, 0 episodes, 0 simulator, 0 model/checkpoint loads, 0 Actor/Critic forwards. All labels agree under the secondary pixel method and declared quantization sensitivity checks. Zero-width and saturated bars are approximate, not exact 0/exact 1. Independent validation passed for sources, metrics, frame alignment, matches and unchanged development identity.

Handoff: [RESULT_SUMMARY.md](handoffs/premature-end-dynamics-001-20260923/RESULT_SUMMARY.md), [analysis](handoffs/premature-end-dynamics-001-20260923/premature_end_analysis.md), [artifact index](handoffs/premature-end-dynamics-001-20260923/ARTIFACT_INDEX.json). Claim a5045c5255b67f3e0f1d4f559d2d6078f70f9544 and its green CI preceded execution. Frozen designs and claim/authorization identity are preserved.

Next actor PI. One proposal follows the frozen LATE_RISE branch: review a matched hard-task SafeVLA versus comparable non-safety/base-policy causal comparison, with comparability and confounds explicitly controlled. This proposal is not approved or executed. Gate B remains unresolved for future hidden-state interpretation. No next experiment or PI acknowledgement is self-issued. Executor STOP after publication.


## 2026-09-23 — Premature-end result reviewed; clean-B0 reproduction is next gate

| Experiment ID | Status | Scientific role |
| --- | --- | --- |
| EXP-PREMATURE-END-DYNAMICS-001 | COMPLETED / PI REVIEWED | Historical audit: 16/16 sub-horizon failures final=end; terminal p(end) jump is not failure-specific |
| EXP-CLEAN-B0-PREMATURE-END-REPRO-001 | DRAFT / NOT_AUTHORIZED | Establish clean full-200 B0 invalid-end prevalence before safety-alignment or hidden-state attribution |

Planned budget for the new draft: max 1 GPU / 200 ObjectNav episodes; no treatment arm, no Probe, no extra live oracle query.


## 2026-09-23 — Reproduction branch dropped; terminal legality calibration is next

| Experiment ID | Status | Reason |
| --- | --- | --- |
| EXP-CLEAN-B0-PREMATURE-END-REPRO-001 | SUPERSEDED BEFORE APPROVAL / NOT EXECUTED | Researcher judged rerunning already-observed invalid-end tasks to have insufficient information value for the current mechanism mainline |
| EXP-END-LEGALITY-CALIBRATION-001 | DRAFT / NOT_AUTHORIZED | Test whether Actor terminal confidence separates legal successful end from illegal failed end using existing artifacts |

Budget: 0 GPU / 0 episode.
