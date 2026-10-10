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


## 2026-09-24 — Terminal legality calibration draft superseded before approval

| Experiment ID | Status | Reason |
| --- | --- | --- |
| EXP-END-LEGALITY-CALIBRATION-001 | SUPERSEDED BEFORE APPROVAL / NOT EXECUTED | Researcher judged further p(end)-based legal-vs-illegal calibration to duplicate already completed probability/termination analyses and provide insufficient incremental information |

No successor experiment is authorized by this decision.


## 2026-10-07 — PI Cycle 2 Reset Gate

| Experiment ID | Status | Role |
| --- | --- | --- |
| EXP-END-LEGALITY-CALIBRATION-001 | SUPERSEDED BEFORE APPROVAL / NOT EXECUTED | Probability-only follow-up retained as historical draft |
| EXP-RESET-001A | DRAFT / NOT_AUTHORIZED | Causal Gate-B test: Actor state carry / rollover -> hidden/logit/action difference under identical teacher-forced inputs |

Budget: max1 GPU, max4 live episodes, no benchmark run. Treatment patch is local-only; handoff evidence is published to research-loop.


## 2026-10-07 — EXP-RESET-001A executable approval

| Experiment ID | Status | Budget |
| --- | --- | --- |
| EXP-RESET-001A | APPROVED_FOR_CODEX / awaiting claim | max1 GPU; max4 live episodes; bounded offline replay |

The failed timestamp approval was never claimed. The valid approval is the later green reapproval.


## 2026-10-08 — EXP-RESET-001A executor handoff

Status: AWAITING_PI_REVIEW; classification: R1; next_actor: PI.

Outcome: **R1**. A/A OFF passed; both offline conditions passed repeat stability.
Fixed real input sequence: 600 decisions. CARRY first rollover: local step 200 (zero based).
First logit divergence above declared tolerance: 200.
First argmax divergence: 205.
First common-RNG sampled-action divergence: 200.
Before/boundary/after maxima: `{"pre_rollover": {"n": 200, "max_abs_logit": 3.814697265625e-06, "max_abs_hidden": 2.86102294921875e-06}, "boundary": {"n": 1, "max_abs_logit": 5.018692970275879, "max_abs_hidden": 3.9560751914978027}, "post_rollover": {"n": 399, "max_abs_logit": 12.482795715332031, "max_abs_hidden": 6.683389663696289}}`.

A valid Actor carry package reproducibly changes the policy distribution and at least one decision under the tested identical current inputs. This supports the state-carry-to-Actor-decision pathway; it does not establish SR or Safety Cost impact.

Claim `c4ab4b395817f439a0239514f3469480aa0d8500` was recovered without generating a replacement, and CI 37752169988 passed before execution. Budget used: 1 GPU / 3 live episodes started / 3 completed. Original tracked development changes and frozen NEXT_EXPERIMENT files are preserved.

Handoff: [RESULT_SUMMARY.md](handoffs/reset-001a-20261007/RESULT_SUMMARY.md). Required outputs and CPU validation are included. Executor STOP after publication; no successor approved.


## 2026-10-08 — EXP-RESET-001A PI review

| Experiment ID | Status | Result | Accepted claim |
| --- | --- | --- | --- |
| EXP-RESET-001A | COMPLETED / PI ACKNOWLEDGED | R1 | Valid Actor carry/rollover state can causally alter Actor logits and action preference under identical teacher-forced current inputs |

No SR, Safety Cost, premature-end, or population-level performance claim is accepted from 001A.


## 2026-10-09 — Direct early-end reset test draft

| Experiment ID | Status | Role |
| --- | --- | --- |
| EXP-RESET-EARLYEND-001B | DRAFT / NOT_AUTHORIZED | Paired online test of Actor reset on historical premature failed-end tasks, with rollover-exposure audit and UNEXPOSED negative controls |

No overall benchmark-SR claim is authorized.


## 2026-10-09 — EXP-RESET-EARLYEND-001B approved

| Experiment ID | Status | Budget | Purpose |
| --- | --- | --- | --- |
| EXP-RESET-EARLYEND-001B | APPROVED_FOR_CODEX / awaiting claim | max1 GPU / max40 live episodes | Direct paired online test of whether the proven Actor reset/rollover mechanism changes historical premature failed-end behavior |

No overall SR improvement claim is authorized by this screening cycle.


## 2026-10-09 — EXP-RESET-EARLYEND-001B executor handoff

Status: BLOCKED; decision B4; next actor PI. P0 verified all 16 historical failed
sub-horizon terminal-end cases. Historical worker-local counter_start is UNKNOWN
for all 16: original retained tables, raw W&B journal and logs do not persist the
task-to-worker ordered mapping. Do not infer counters from global completion order.
The approved P0 stop condition applies; P1 and online comparisons were NOT_STARTED.
Budget used: 0 GPU, 0 live episodes, 0 policy loads, 0 simulator starts.

This is missing historical state evidence, not a no-effect result. B1/B2/B3 cannot
be evaluated; accepted 001A R1 remains unchanged. Frozen designs and approval are
preserved. The unique claim fd3a11bbc5bb6ef365f3f1edc68e1e6a9888264c passed control
CI 37872063939 before P0. No replacement claim or automatic retry was generated.

Handoff: [RESULT_SUMMARY.md](handoffs/reset-earlyend-001b-20261009/RESULT_SUMMARY.md).
Required outputs include explicit empty NOT_STARTED paired tables and hashed audit
evidence. PI next action: decide whether an authentic worker-local ordered log can
be recovered; otherwise review a new design under a fresh approval. Executor STOP
after publication; no successor is approved and no PI acknowledgement is authored.


## 2026-10-09 — EXP-RESET-EARLYEND-001B closed

| Experiment ID | Status | Result | Resource use |
| --- | --- | --- | --- |
| EXP-RESET-EARLYEND-001B | PI ACKNOWLEDGED / CLOSED | B4 historical counter provenance unreconstructable | 0 GPU / 0 live episodes |


## 2026-10-09 — EXP-COUNTER-DRIFT-EARLYEND-001C staged

| Experiment ID | Status | Budget | Question |
| --- | --- | --- | --- |
| EXP-COUNTER-DRIFT-EARLYEND-001C | DRAFT / NOT_AUTHORIZED | max1 GPU / 32 live target episodes | Does controlled mid-episode current-context truncation causally change Actor/trajectory/termination behavior on all 16 confirmed historical early-end tasks? |

Primary mechanistic signature: OFF effective current-episode context 50 -> 1 at local step 50; ON continues to 51.


## 2026-10-09 — EXP-COUNTER-DRIFT-EARLYEND-001C approved

| Experiment ID | Status | Budget | Frozen primary signature |
| --- | --- | --- | --- |
| EXP-COUNTER-DRIFT-EARLYEND-001C | APPROVED_FOR_CODEX / awaiting claim | max1 GPU / max32 live target episodes | OFF context 50 -> 1 at local step 50 while ON continues to 51; measure Actor -> action -> trajectory -> termination drift |

Official full-200 SR evaluation is outside this cycle.


## 2026-10-09 — EXP-COUNTER-DRIFT-EARLYEND-001C unused approval revoked

| Experiment ID | Latest status | Execution |
| --- | --- | --- |
| EXP-COUNTER-DRIFT-EARLYEND-001C | PI_REVIEW / REVOKED_UNCLAIMED / NOT_AUTHORIZED | No claim or live run known in checked GitHub/server scope; 0 GPU / 0 episodes in the review record. |

The previously approved max1 GPU / 32 live episodes budget is historical, not executable. Separate 001D P1 staging follows only after revoke-control CI success.


## 2026-10-09 — EXP-ACTOR-RESET-FULL200-001D P1 staged

| Experiment ID | State | Proposed scope |
| --- | --- | --- |
| EXP-ACTOR-RESET-FULL200-001D | PI_REVIEW / NOT_AUTHORIZED / NOT_EXECUTED | New `actor-reset-full200-p1-20261009` engineering validity gate: OFF/OFF A/A 5+5 tasks and isolated offline logger/reset invariants; proposed max1 GPU/10 episodes; P2 400-episode A/B is not authorized. |

Previous 001C unclaimed authorization was revoked in `e93a9cdff70fe537cfa377c2d1f63c7b432d45be` and revoke CI passed. P1 command/worktree/real task manifest remain unfrozen; any claim or run is prohibited pending separate PI approval.


## 2026-10-09 — PI freezes EXP-ACTOR-RESET-FULL200-001D P1 engineering preflight for separate approval

After P0 review and 001C unclaimed revocation, the PI registers concrete P1 command `/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1-20261009/run_p1_preflight.py`, proposed isolated worktree `/nvme2/user/qyy/SafeVLA_p1_001d`, singleworker OFF A/A (5+5) only, 1 GPU/10 episode proposed max, 6000 live step/2400 offline combined forward/8 GPU-hour/50GiB stop ceilings, deterministic identical-input logger gate <=1e-5, exact root Actor reset/non-target critic invariants, and official cost/source/manifest gates. This is state v50 PI_REVIEW / NOT_AUTHORIZED (0 active GPU/episode). The command will be implemented by the executor only after a distinct PI-approved commit, successful claim and claim CI. P2 200+200 is NOT AUTHORIZED.


## 2026-10-09 — PI explicitly approves EXP-ACTOR-RESET-FULL200-001D P1 execution, NOT P2

Following accepted P0 review, frozen P1 staging commit `cc1db805b5e6392a225a554f0251e4a1f8fdb5c2` passed control CI #37953123220. PI authorizes P1 only: two independent one-worker OFF sessions of the exact same five frozen tasks (10 live episode starts max), matched-input offline logger equivalence and isolated root-Actor-only reset/critic invariants. Approved protocol: one GPU max; 6000 live decisions, 2400 offline combined forwards, 8 GPU-hours, 50GiB raw artifacts hard STOP, expiry `2026-10-11T15:41:29.332Z`. Frozen worktree `/nvme2/user/qyy/SafeVLA_p1_001d`, single execution command `/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1-20261009/run_p1_preflight.py`. The CODEX claim protocol (claim latest green authorization head then await green claim CI) is mandatory. No claim or execution is performed by PI; no ON live episodes, no full200 P2, no automatic retries or extra seeds; executor must STOP after evidence publication. Approval state_version=51; 001C remains revoked.


## 2026-10-10 — PI renews unused P1 authorization: safe revocation stage

On user's explicit request to reauthorize P1, checked remote research-loop HEAD `5609f3fbb456c23f8645800ff46b5371cc294d47`, green approval CI #37953638029, status=APPROVED_FOR_CODEX, state_version=51 and null claim/instruction_commit. Prior window had not expired. PI applies supported unclaimed `APPROVED_FOR_CODEX -> PI_REVIEW` transition, clears active approval details, sets active GPU/episode budget 0 and returns both design statuses to DRAFT. This is renewal administration only: same EXP-ACTOR-RESET-FULL200-001D P1 cycle, protocol and future max1 GPU/10 episodes; no P2, claim, GPU or execution. Renewal reapproval is contingent on revoke CI SUCCESS; if validation fails, STOP and leave P1 NOT_AUTHORIZED.


## 2026-10-10 — PI reissues unclaimed 001D P1 authorization without altering the protocol

At the user's explicit request, the previous P1 authorization `5609f3fbb456c23f8645800ff46b5371cc294d47` was revoked unclaimed by `f2f352480e0b908b1ac95677a9aa9570e883e228`; control CI #38013257402 SUCCESS. PI now restores **APPROVED_FOR_CODEX**, state_version=53, the same `actor-reset-full200-p1-20261009` cycle and EXP-ACTOR-RESET-FULL200-001D P1 scope. New approval timestamp `2026-10-10T01:30:05.821Z`; expiration `2026-10-12T01:30:05.821Z` (48-hour renewal). Approved budget remains max1 GPU/max10 live episodes (OFF/OFF 5+5), with 6000 live steps, 2400 offline forwards, 8 GPU-hours and 50GiB limits. The approved command, output paths, worktree, checkpoint/data/metric constraints and P2 prohibition remain identical. The PI does not claim or run this experiment. Executor must wait for this renewed approval CI PASS and valid claim+claim CI before P1 execution. No successor approval.


## 2026-10-10 — EXP-ACTOR-RESET-FULL200-001D P1 executor handoff INVALID

Unique published claim334e80351ff6cb2cfb0dac166fa7227ae055a5bd (CI38016589200 SUCCESS) executed the approved P1 command once. Offline snapshot restore failed after one root Actor step because B0 lazily changed initial cache shape from[0,500,8,64] to[1,500,8,64]; in-place restore could not recover the empty snapshot. This is an executor harness defect, not a policy/reset result. Paired logger controls and reset invariants NOT_REACHED; both online OFF sessions NOT_STARTED. Budget1 GPU /0 live initialization attempts /0 online decisions. No retry, live ON or P2. Baseline/development tracked changes and frozen designs preserved.

Status INVALID, next_actor PI, state_version55. All required evidence: [RESULT_SUMMARY.md](handoffs/actor-reset-full200-p1-20261009/RESULT_SUMMARY.md). P2 blocked. Sole proposed next step is PI review of shape-aware isolated snapshot restoration and lifecycle validation, requiring a fresh cycle/approval before any rerun. Executor STOP after publication; no PI acknowledgement or successor approval authored.


## 2026-10-10 — PI acknowledges terminal INVALID 001D P1 and denies automatic retry

After directly reading GitHub handoff `a137eafbb5931190b68ff042a5b0c9e34dbfd68d` and control CI #38034324471 SUCCESS, PI accepts the executor's terminal `INVALID`: isolated `run_p1_preflight.py:75` `copy_` fails restoring a zero-batch cache `[0,500,8,64]` after original Attention lazy initialization to `[1,500,8,64]` on the single completed offline root Actor forward. 0 online episode starts, 0 original-loader online cross-checks, no completed logger pairs or critic state treatment tests; official SR and Safety Cost are NOT_MEASURED, not zero. Preserved claim and instruction provenance, 14 required artifact files, incomplete gate evidence and source identity. Technical root is a harness cache lifecycle defect, not validated as a model/Actor/metric effect. The original P1 cycle is terminal and may not be reclaimed.

PI acknowledges `INVALID -> PI_REVIEW`, state_version=56 and `reviewed_result_commit=a137eafbb5931190b68ff042a5b0c9e34dbfd68d`, clearing approval/time metadata and returning both design files to DRAFT. Next possible action is a **fresh unused cycle** with an independently reviewed shape-aware cache restore, CPU lifecycle regression gate, then bounded offline engineering validity and, only with fresh PI authorization, original P1 online 5+5 maximum. No P1 rerun or P2 is authorized by this receipt.


## 2026-10-10 — PI stages fresh 001D P1R cache-lifecycle recovery

User approved registering and subsequently authorizing a fresh `EXP-ACTOR-RESET-FULL200-001D-P1R` control cycle with sequential Gate A CPU (zero-batch lazy-cache tensor restore), Gate B offline GPU logger and Actor-only reset/critic invariants, and Gate C OFF/OFF 5+5 online preflight only. Prior claimed P1 `actor-reset-full200-p1-20261009` is terminal INVALID and PI-reviewed in `45b942ec2c3980401b5b06b85977318b9297a5ae` (CI #38042403557 SUCCESS), with 1 offline forward, 0 online episodes and no SR/Cost observation. Original failure involved isolated runner's shape-invariant `copy_` after B0 Llama lazy 0-batch→1-batch cache allocation. This new unused cycle `actor-reset-full200-p1r-20261010` is status `PI_REVIEW / NOT_AUTHORIZED` state_version=57, new clean worktree `/nvme2/user/qyy/SafeVLA_p1r_001d`, dedicated new command and required output directory, 0 active GPU/episode. Proposed max1 GPU / max10 initiated live episodes, 6000 online steps, 2400 offline forwards, 8 GPU-hours and 50 GiB. No claim, no executor activity, no P2 approval. Each gate is fail-closed and conditional on predecessor passing. PI will separately approve only after staging-control CI SUCCESS.


## 2026-10-10 — PI approves fresh EXP-ACTOR-RESET-FULL200-001D-P1R Gate A/B/C execution

User explicitly approved the narrow P1R fresh cycle after PI signed terminal prior P1 INVALID (a137eafbb5931190b68ff042a5b0c9e34dbfd68d), without changing accepted B0. Frozen staging commit `82d595cc179b00e1c1f6cc0437bb9fc9e082af34` passed control CI #38043747983 SUCCESS. This transition changes state_version=57 PI_REVIEW/NOT_AUTHORIZED to **v58 APPROVED_FOR_CODEX** with PI approval timestamp `2026-10-10T10:10:15.465Z`, expiry `2026-10-12T10:10:15.465Z`, max 1 GPU and 10 live Episode starts; no claim yet. Scope is Gate A CPU real cache lifecycle and shape-aware snapshot-only runner repair, then Gate B GPU offline logger/reset invariants, then Gate C OFF/OFF 5+5 single-worker online. Fail any Gate => STOP, no independent retry, no online ON, no full200 P2. Additional hard bounds: <=6000 online steps, <=2400 offline forwards, <=8 GPU-hour/50GiB raw. Only the new isolated P1R worktree `/nvme2/user/qyy/SafeVLA_p1r_001d` and one frozen command `/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1r-20261010/run_p1r_preflight.py` are authorized; original P1 INVALID handoff remains immutable. CODEX must publish unique claim against this approved SHA, verify claim CI green, then execute once and hand evidence to PI; PI is not claiming/running any task.


## 2026-10-10 — EXP-ACTOR-RESET-FULL200-001D-P1R terminal resource BLOCKED

Unique claimd35f645cabf45bfb2ad4c3dffa548cdd7da7c8e5, claim CI38045418732 SUCCESS. Executed once in new P1R snapshot. CPU Gate A PASS:4 genuine three-layer Attention lifecycle cases (empty/allocated,float32/float64), max_abs0 and exact shape/dtype/device/value/counter/RNG/alias invariants;24 CPU Attention calls. Gate B child NOT_STARTED: GPU0 free3954 MiB below inherited frozen >6823.391 MiB capacity guard. Gate C NOT_STARTED;0 GPU/0 episode starts/0 online decisions/0 policy loads. No retries, GPU switch, gate relaxation, ON or P2.112 B0 and149 old P1 files preserved.

Handoff status BLOCKED/next_actor PI/v60. Original supervisor INVALID and prelaunch B RUNNING records preserved separately; handoff explicitly classifies resource stop before GPU launch. See [RESULT_SUMMARY.md](handoffs/actor-reset-full200-p1r-20261010/RESULT_SUMMARY.md), all16 required outputs. CPU fix evidence does not establish GPU logger/reset/critic validity or SR/Cost effects. PI next: review evidence and arrange capacity before any fresh independent cycle approval. Executor STOP after publishing; no automatic continuation or successor authorized.
