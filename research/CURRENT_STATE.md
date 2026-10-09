# Current State

Updated: 2026-10-07（Asia/Shanghai）
Mode: EXP-RESET-EARLYEND-001B APPROVED_FOR_CODEX; next actor CODEX; awaiting valid claim.

## Latest PI decision

The researcher restored the original 02｜SafeVLA Gate-B ordering and supplied the Cycle-2 causal design. The previous probability-only end-legality draft remains superseded and unexecuted.

The unique next DRAFT is `EXP-RESET-001A`: first audit the actual Actor decoder state carrier, then verify an explicit OFF/ON reset switch, then run a teacher-forced rollover stress test on identical Actor inputs. The experiment tests only `state carry -> Actor hidden/logit/action change`; it does not test SR impact.

Two design safeguards were added during PI review:
1. because accepted B0 is stochastic and same-seed runs are not bitwise trajectory-reproducible, A/A OFF equivalence is judged primarily on identical saved Actor inputs plus identical RNG/common random numbers, not on whole live-trajectory equality;
2. `counter=300` may not be fabricated with an incompatible cache. CARRY-300 must be a semantically valid state package, produced by a valid warm-up replay if cache/counter are coupled.

## Preserved facts

- 16/16 historical sub-horizon failures end by executing `end`.
- First-decision `p(end)` is sub-resolution in all 16; preterminal maxima remain below 0.5.
- The terminal `p(end)` jump also appears in successful ends, so it is not failure-specific.
- H-PRIOR and a simple gradual H-SEARCH-TRIGGERED explanation are weakened.
- Historical Probe evidence remains exploratory/provenance-limited.
- H-RESET remains unresolved and must be controlled before formal hidden-state / Probe causal interpretation.

## Current research direction

Gate B is now the only mainline blocker. Do not reopen H2, size measurement, additional p(end)-only audits, broad Probe work, Stop Gate or 200-task performance evaluation before `EXP-RESET-001A` is reviewed.

If R1/R2: decide whether a very small paired behavior test is justified.
If R3: immediately downgrade H-RESET and return to exploration / perception / representation / readout / SafeRL attribution.
If R4: repair the experiment only; do not interpret the hypothesis.

## Control State

Current control state remains PI_REVIEW / NOT_AUTHORIZED. `EXP-END-LEGALITY-CALIBRATION-001` is retained only as an append-only superseded draft record; it must not be claimed or executed.

## 02 research-history migration — 2026-09-18

The substantive legacy research state from Project conversation `02｜SafeVLA研究` and its branches has now been migrated into the shared control plane:

- `research/LEGACY_RESEARCH_STATE.md`: facts / observations / active hypotheses / rejected or corrected interpretations / legacy branch inventory.
- `research/EVIDENCE_REGISTER.md`: evidence IDs, raw/server/Git locations, what each source can and cannot support, and discrepancies that must be reconciled.

Executor must read both files before claiming scientific conclusions from legacy work. Raw server evidence outranks migrated summaries. In particular, do not merge the researcher's later "~50% small-category SR" recollection with the preserved 173/200 category table until run identity is reconciled; do not use historical Probe AUC or modified-policy/PT-Guard runs as B0 small-target evidence.

## Control State

Current cycle: premature-end-dynamics-001-20260923; experiment: EXP-PREMATURE-END-DYNAMICS-001; status: AWAITING_PI_REVIEW; next_actor: PI. Claim a5045c5255b67f3e0f1d4f559d2d6078f70f9544 binds instruction 681cd28355cfa79a6eaec7f663d9b3776da7f37e. Execution is complete; no new experiment may start without PI review and fresh approval. Earlier cycles and proposals below are historical records.


## Control incident — 2026-09-18

The first small-target approval sequence was staged through multiple Git commits. One intermediate PI_REVIEW state used an empty `required_outputs` list, and the eventual APPROVED state lacked several validator-required machine-readable design fields even though the Markdown design contained their scientific content. GitHub Actions correctly rejected the history. No Executor claim, GPU allocation, simulator launch, model load, or episode occurred.

Repair policy:
- preserve the bad commits in append-only history rather than force-push/rewrite;
- record a narrow historical validator exception only for this exact unclaimed small-target staging incident;
- add a tested PI edge for revoking an unclaimed approval;
- return to PI_REVIEW, complete the design fields without changing scientific scope, then re-approve only after CI is green.


## Control repair completion

- Repair-state CI passed before re-authorization.
- Re-authorization commit: `cd411d523bcfa5f335240b266c43bdfe14ada6d6`.
- Its control-protocol job passed all synthetic tests and full history validation.
- Executor must claim the latest green `origin/research-loop` HEAD; it must not bind to the earlier failed approval commit.


## 2026-09-20 — Historical server evidence aligned

User-authorized evidence archival only; no experiment claim or execution and no change to LOOP_STATE/NEXT_EXPERIMENT.
Packet: [history/evidence-alignment-20260919/README.md](history/evidence-alignment-20260919/README.md).
The inventory records 684 paths (680 existing/readable, 4 missing names); 645 existing sources lack byte-identical copies on the four inspected GitHub branches.
Final W&B raw table is 200 rows / 173 successes / sum_cost 145 and matches its recorded SHA256. Stable task-path normalization pairs 200/200 tasks, with 200/200 expert_length == gt_episode_len and room-visitation values present.
Static scene/asset sources exist, but annotated size is not yet validated as transformed per-target simulator bounding-box size. Four recovered Probe worker tensor hashes differ; the original AUC artifact/analysis identity remains unresolved.
Raw tensors, media, large logs and datasets remain server-side with SHA256/size. Existing experimental source copies are referenced rather than duplicated into the control plane. Old development status snapshots inside the archive are historical, not current authority.


## 2026-09-20 — PI review of aligned evidence

Independent GitHub verification:
- archival evidence commit: `06a8265051298b103a76ff26db490d262da3882c`;
- current aligned control HEAD before PI renewal: `fed5d8fe282bff0d3c119033ccb565f1574753d9`;
- GitHub Actions run `35463659449`: completed / success, including synthetic control tests and full history/artifact validation;
- evidence packet: `research/history/evidence-alignment-20260919/`.

What changed scientifically:
- the exact historical 173/200 result and its 200 stable task identities are now directly recoverable from shared/raw evidence;
- `expert_length` and room-visitation fields are available for all 200;
- static scene/asset metadata is available, but the proposed physical-size variable is **not yet validated** as actual transformed per-target simulator bounding-box size;
- four recovered Probe worker tensors have different hashes, so the old AUC artifact identity remains unresolved.

Decision impact:
- no reason to change the unique next experiment;
- `EXP-SMALLTARGET-PHENOTYPE-001` should first validate whether a policy-independent task-level size variable can be constructed from the archived static metadata;
- if that cannot be done without simulator/model execution or an invalid proxy, the experiment must return BLOCKED as pre-registered.


## 2026-09-20 — EXP-SMALLTARGET-PHENOTYPE-001 metadata gate BLOCKED

Claim 72634cf7c3a954dd99456b2d75c3af539da9e68e bound renewed PI approval 290b426ac65d35ef1f8cb4819e9662c4fae1bb31; claim CI 35499476480 succeeded before execution. Budget used: 0 GPU / 0 episode / 0 model load / 0 simulator launch.

The metadata-first audit mapped 200 tasks and all 368 broad-synset target IDs to scene objects. Candidate asset bounding boxes cover only 42 targets; 326 lack candidates in inspected static sources. Only 31 tasks are candidate-complete, and no candidate has been promoted to validated scene-instance physical size: units/scale/transformation equivalence remains unestablished. Thus validated size count and analyzable n are 0. The bounded search does not establish global nonexistence of another static source.

The pre-registered stop condition is met: return BLOCKED to PI. No distance, visible-pixel or category-name proxy, incomplete-target median, category statistical analysis, tertile, association or regression was used. episode_table.csv preserves 200 historical raw rows with empty size cells; category_sr.csv explicitly records NOT_EXECUTED with blank statistics. H-SIZE remains untested.

Handoff: [RESULT_SUMMARY.md](handoffs/smalltarget-phenotype-001-20260918/RESULT_SUMMARY.md), with metadata coverage, source hashes, raw identity checks and all required outputs. Proposed prerequisite only: a version-bound static geometry and instance-transform source covering all broad targets, including built-in THOR assets. No new experiment is approved; executor STOP after publication. Existing development changes and frozen NEXT_EXPERIMENT files are preserved.

## 2026-09-21 — Runtime metadata handoff: BLOCKED / next actor PI

EXP-SIZE-RUNTIME-METADATA-001 completed 24 preflight and 20 full scene initializations before a BrokenPipeError in its stdout progress print. No retry or resume was performed. Preflight: 12 tasks loaded twice, 18 exact-equal target comparisons, max absolute/relative dimension difference 0. Partial full pass: 26/368 exact-mapped valid AABBs, 20/200 complete tasks; 342 targets / 180 tasks unattempted, not observed missing geometry. The complete-coverage hypothesis remains unresolved.

One simulator graphics GPU, 44/224 scene initializations, 0 episodes, 0 model/checkpoint loads, 0 Actor/Critic forwards. Geometry extraction read no outcomes and made no size-success association. Creation-state world-axis AABB is not intrinsic volume or demonstrated settled evaluation-state geometry. Development HEAD/diff/status and frozen NEXT_EXPERIMENT files are preserved.

Handoff: [RESULT_SUMMARY.md](handoffs/size-runtime-metadata-001-20260920/RESULT_SUMMARY.md), [coverage_report.md](handoffs/size-runtime-metadata-001-20260920/coverage_report.md), [ARTIFACT_INDEX.json](handoffs/size-runtime-metadata-001-20260920/ARTIFACT_INDEX.json). Original traceback, script, raw snapshots, partial CSVs and independent validation are Git-readable. Required outputs are complete; scientific extraction is incomplete.

The only next action is PI review. A fresh transport-resilient extraction cycle is a proposal only, requiring an explicit design and budget that account for this partial run. The current claim is terminal and must not auto-resume. No further experiment or PI acknowledgement is authorized or fabricated. Executor STOP after handoff publication.


## 2026-09-22 — PI acknowledgement of EXP-SIZE-RUNTIME-METADATA-001 BLOCKED

PI independently reviewed result commit `ac2c9fe80f113345f5092205b07192a47a0e0358` and the required handoff outputs.

Accepted facts:
- exact historical AI2-THOR build identity was recovered to commit `966bd7758586e05d18f6181f459c0e90ba318bec` with CloudRendering;
- the 12-task double-load preflight passed: 18/18 target comparisons were exact-equal, max absolute and relative geometry difference 0;
- the full extraction attempted only 20/200 tasks before an inherited stdout progress print raised `BrokenPipeError`;
- all 26 targets attempted in the full phase exact-mapped and had valid creation-state AABB; all 20 attempted tasks were complete;
- 342 targets / 180 tasks are unattempted, not observed missing;
- actual budget was 44 scene initializations, one simulator-graphics GPU, 0 ObjectNav episodes, 0 model loads and 0 Actor/Critic forwards;
- no outcome fields were read and no size-success association was run.

Scientific interpretation:
- H-RUNTIME-AABB remains unresolved for complete 368/368 coverage;
- the partial result materially strengthens the plausibility of runtime geometry recovery but does not establish complete coverage;
- the blocking event is an execution-output transport failure, not an observed geometry/mapping failure;
- creation-state world-axis AABB remains a scene-instance extent, not canonical intrinsic volume.

The old claim is closed and cannot resume. Any completion attempt requires a fresh cycle and an explicit new PI approval.


## 2026-09-22 — Mainline redirected from size measurement to mechanism-stage localization

Researcher decision: no further physical-size measurement on the mainline. The project now treats the relevant categories as a qualitative small-object-like / formally low-SR group, without claiming that physical size has been causally proven.

Important PI correction:
- `sub_house_id` is the shuffled dataset sample index, not house difficulty. Earlier interpretations of `sub_house_id<20` as "harder tasks" are withdrawn.
- `expert_length` is the recorded expert trajectory length and can be used only as a long-horizon difficulty proxy; it is not a proven shortest path or pure environment difficulty metric.
- real `house_index` shows a historical low-index failure cluster, but index ordering is not itself a validated difficulty scale.

Exploratory aligned-data facts to be formally reproduced by the next experiment:
- mug+basketball+laptop+bowl: 34/48 successes (70.8%) versus 139/152 (91.4%) in other categories;
- among 14 failures in that group: 9 never entered the target room, 10 never had narrow target visibility in the nav camera, 4 were nav-visible;
- coarse expert_length strata retain a group gap: <=50 92.3% vs 97.8%; 51–100 57.1% vs 90.7%; >100 25.0% vs 58.8%.

These facts motivate exploration/room-arrival as the current leading stage hypothesis, but internal mechanism claims await the formal stage audit and a subsequent controlled diagnostic.


## 2026-09-22 — Experiment redesigned from 02｜SafeVLA research logic

The previous zero-rollout stage-localization draft was too shallow relative to the established 02 research question. The mainline is now restored to the original semantic-decision-mismatch chain:

`target evidence -> fusion/temporal representation -> Actor readout/logits -> end/move/turn behavior`

Key carried-forward facts:
- historical layer-wise AUC increased rather than collapsed, so "Layer 3 forgets target information" is not the leading interpretation;
- old Probe evidence is not formal because of PT-Guard, missing episode IDs and frame-level leakage risk;
- done-gating previously traded premature termination for hesitation/loops without net SR gain;
- SafeVLA exhibits both illegal high-end-prob failure and long low-end-prob horizon failure;
- therefore the next experiment must distinguish exploration-before-evidence, representation weakness, Actor-use/readout mismatch and termination calibration on clean B0.

The new experiment uses targeted historical cases only for diagnostic selection, reruns them under clean B0, recomputes fresh phenotypes, performs episode/house-heldout target-visible probes, and—only offline—tests Actor-head sensitivity to the clean L3 probe direction versus random/shuffled controls.


## 2026-09-23 — PI reread of 02｜SafeVLA changes experiment order

Direct reread of the migrated 02 branch recovered an explicit methodological stop that the previous draft violated: Gate B was FAIL because official reset retains decoder counters/KV cache across episodes, with cumulative rollover at max_steps=500. The historical short controlled test only showed that old cache is masked at a fresh episode start; it did not eliminate later rollover effects.

Therefore the clean Probe/readout experiment cannot yet be the next formal experiment. Hidden-state interpretation must first bound H-RESET on real B0 input traces.

The previous `EXP-SEMANTIC-DECISION-MISMATCH-001` is superseded before approval, not executed. Its representation/use design remains queued and should return only if Gate B2 passes or is otherwise causally bounded.

## 2026-09-23 — EXP-PREMATURE-END-DYNAMICS-001 completed / awaiting PI review

All 16 historical sub-horizon failure videos were recovered reliably and confirm final executed end. Frozen morphology counts: LATE_RISE 13/16, EARLY_HIGH_PRIOR 1/16, OTHER_OR_UNCLEAR 2/16, LOW_PROB_END 0/16. All initial end bars were sub-resolution; all preterminal maxima were below 0.5. The 14 first crossings of 0.5 occurred at the terminal decision itself. The one early-high case is a two-step episode, not evidence of an elevated first-decision prior.

All 16 same-category expert-length matches were recovered (11 unique successes, reused). Ten of those 11 successes also first crossed 0.5 at successful termination. Thirteen prescribed control windows include the success terminal frame; expert-length matching gaps have median 34 and range 2–133. The abrupt terminal jump is therefore not failure-specific and does not establish SafeRL causation. Target scope: STRICT 12 / AMBIGUOUS 4; narrow visibility cannot establish broad-target invisibility in ambiguous cases.

This was historical CPU-only video/table analysis: 0 GPU, 0 episodes, 0 simulator, 0 model/checkpoint loads, 0 Actor/Critic forwards. All labels agree under the secondary pixel method and declared quantization sensitivity checks. Zero-width and saturated bars are approximate, not exact 0/exact 1. Independent validation passed for sources, metrics, frame alignment, matches and unchanged development identity.

Handoff: [RESULT_SUMMARY.md](handoffs/premature-end-dynamics-001-20260923/RESULT_SUMMARY.md), [analysis](handoffs/premature-end-dynamics-001-20260923/premature_end_analysis.md), [artifact index](handoffs/premature-end-dynamics-001-20260923/ARTIFACT_INDEX.json). Claim a5045c5255b67f3e0f1d4f559d2d6078f70f9544 and its green CI preceded execution. Frozen designs and claim/authorization identity are preserved.

Next actor PI. One proposal follows the frozen LATE_RISE branch: review a matched hard-task SafeVLA versus comparable non-safety/base-policy causal comparison, with comparability and confounds explicitly controlled. This proposal is not approved or executed. Gate B remains unresolved for future hidden-state interpretation. No next experiment or PI acknowledgement is self-issued. Executor STOP after publication.


## 2026-10-07 — PI Cycle 2 staged from user-provided Gate-B plan

Current experiment: `EXP-RESET-001A`
Cycle: `reset-001a-20261007`
Status: PI_REVIEW / NOT_AUTHORIZED
Budget: max 1 GPU, max 4 live ObjectNav episodes; offline replay only beyond that.

The treatment patch must not be committed into the B0 execution branch. Per the established research-loop protocol, handoff evidence itself must still be published to `research-loop` for PI review; this is the only intended exception to the user instruction "do not commit/push."


## 2026-10-07 — EXP-RESET-001A final approval

Authoritative approval HEAD before documentation maintenance: `f6f8bcf82b41bb1b8feb9a5b2088527dc3768022`.
Approval CI `37631919214` completed successfully.

Current budget:
- max GPU: 1;
- max live ObjectNav episodes: 4;
- offline replay only within frozen P1/P2 scope.

A prior approval commit `bb2a10792e615153485fa16e00f67040e6478612` failed history validation because its timestamp regressed relative to the staging parent. No claim or experiment execution occurred. The incident is preserved append-only, the unclaimed approval was revoked, and the experiment was re-approved only after the repair CI passed.

Executor must claim the latest green research-loop HEAD and wait for claim CI before execution.


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


## 2026-10-08 — PI review of EXP-RESET-001A

PI independently reviewed commit `2f7864ae2c3a4761932aa4d1fedd6b6d5b3388c6`, control CI 37777926111, required handoff artifacts, the state-carrier audit, OFF A/A evidence, reset patch, rollover result and serialized trace.

Accepted result: **R1** within the frozen scope.

The decisive causal pattern is not generic cross-episode token leakage. Before CARRY rollover, CLEAN and CARRY are numerically equivalent within the declared tolerance. When the carried Actor counter reaches 500 at current-episode local step 200, the official path resets its model-side position to zero while episode-local time remains 200. The decoder therefore stops addressing the current episode's earlier temporal context. Hidden/logit differences jump exactly at that boundary and later change Actor action preference.

Evidence:
- pre-rollover max abs logit difference = 3.8147e-06 (<1e-5 tolerance);
- rollover-boundary max abs logit difference = 5.01869;
- first argmax action divergence = local step 205;
- post-rollover max abs logit difference = 12.4828;
- OFF A/A exact on saved inputs/RNG; within-condition repeated replay exact;
- no SR or Safety Cost claim is accepted.

PI interpretation:
`prior episode state -> rollover timing -> current-episode temporal-context truncation -> Actor distribution/action change` is now an established causal pathway for the tested stress trace.

Not established:
- population prevalence;
- historical full-200 outcome impact;
- low-SR category explanation;
- premature-end causation;
- SR/Safety Cost improvement from reset;
- separate causal attribution of counter versus K/V contents.

Next research question must be behavioral relevance under a small paired online treatment, not another broad Probe or probability-only audit.


## 2026-10-09 — Mainline narrowed to direct early-end test

Researcher requested the most direct follow-up: validate the reset effect on historical premature-end cases themselves.

New unique DRAFT: `EXP-RESET-EARLYEND-001B`.

Key design gate: not every sub-horizon failure is automatically rollover-exposed. The audit must confirm final executed `end` and reconstruct worker-local Actor counter at episode start. Historical early ends occurring before the calculated rollover are negative controls and cannot be explained by H-RESET in that run.

Primary causal comparison is paired OFF vs ON on the exact historical target task, starting from the same valid carry package and common random numbers. OFF retains Actor counter/KV; ON clears the Actor package only. No Probe or benchmark-wide evaluation is part of this screening cycle.


## 2026-10-09 — EXP-RESET-EARLYEND-001B approved

User/PI explicitly approved the direct historical early-end paired reset test after draft CI `37870566672` passed.

Frozen authorization:
- max GPU: 1;
- max live ObjectNav episodes: 40;
- historical exposure audit first;
- only confirmed failed-end cases with reconstructable worker-local counter state may enter the primary paired analysis;
- OFF retains the valid Actor carry package; ON clears only the reviewed Actor counter+K/V reset package;
- no Probe, Stop/Oracle Gate, checkpoint comparison or 200-task benchmark.

Executor must claim the latest green research-loop HEAD and wait for claim CI before execution.
