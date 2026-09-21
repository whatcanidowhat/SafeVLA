# Current State

Updated: 2026-09-21（Asia/Shanghai）
Mode: PI_REVIEW. EXP-SIZE-RUNTIME-METADATA-002 is the unique next DRAFT; NOT_AUTHORIZED.

## Verified Facts

- EXP-B0-REPRO-001A is COMPLETED / PROVENANCE PASS for its narrow goal: two Reference + two Repeat episodes on the same executable B0, stable 2/2 task pairing, and traceable source/resource identity. It does not prove full-200 performance equivalence.
- Executable B0 remains official reference 2aa82559d272b5f888e53433e258914057f15bed plus the manually accepted local-DINO infrastructure adaptation only.
- Historical full-200 raw evidence has now been re-aligned to the archived 2026-08-03 run: final W&B table has 200 rows, 173 successes and sum_cost=145; stable task-path normalization pairs 200/200 tasks, and expert_length == gt_episode_len for all 200. This verifies the identity/integrity of that historical run, but does not convert it into a fresh formal B0 rerun.
- A historical canonical state recorded stop_legal=false while Actor end probability was approximately 0.993365 and greedy action was end. This establishes an illegal/premature-end phenotype in at least one state, not its mechanism.
- Source audit confirms end is a learned Actor action. Actor hidden beliefs are mapped by a LinearActorHead into an action distribution; end is sampled or mode-selected like other actions. Only after end is executed does ObjectNav successful_if_done() check whether a valid target is visible in the navigation camera within maximum_distance=2.
- Therefore end_prob≈0.993365 is not an "end Probe"; it is the Actor probability assigned to the end action. The historical Probe AUC values 0.955/0.982/0.992 are a separate exploratory layer-wise decoder Probe for a close-and-visible / stop-legality-like label.
- Those historical Probe results are hypothesis-generating only: the run contained PT-Guard action rewriting, the saved artifact lacks full episode/task provenance, and step-level splitting risks temporal leakage. They do not prove small-target representation.
- A later topology audit established that a formal Actor Probe must use the root actor_critic.decoder branch; reward-critic and cost-critic hidden states must not be mixed into Actor representation claims.
- The old EXP-B0-REPRO-001B-PREFLIGHT approval expired on 2026-09-14T07:34:44Z and control state showed claim_id=null. It has been revoked rather than executed.

## Researcher Observation — not yet independently audited

The researcher reports several target categories such as cup/mug-like objects, basketball, kettle and apple showing navigation SR around 50% in evaluation summaries.

This is a valid phenomenon candidate. It is not yet evidence that "small physical size" is the cause: category, sample size, house/scene difficulty, expert path length, initial distance and occlusion remain alternatives. Basketball/kettle also demonstrate that low-SR category and "small object" cannot be treated as synonyms.

## Active Hypotheses

- H-SIZE: policy-independent target physical size contributes to lower ObjectNav SR.
- H-CATEGORY: semantic category explains low SR independent of size.
- H-DIFFICULTY: hard houses/tasks, longer expert paths, farther starts or occlusion explain the pattern.
- H-SAMPLE: category-level ~50% SR is unstable because n is small.
- H-REPRESENTATION / H-READOUT / H-EXPLORATION / H-TERMINATION / H-SAFETY remain downstream mechanism hypotheses. They should not be selected before the target-size/category phenotype is made precise.
- H-RESET remains a known confound/risk hypothesis; its causal effect on SR is still unproven and it is not the current mainline.

## Unsupported Interpretations

- "Small targets are already proven to cause low SR."
- "AUC 0.992 proves the model represents small objects."
- "end_prob≈0.993 is a Probe score or a success probability."
- "High end probability proves representation loss, readout mismatch, or safety-induced termination."
- "Safety critic directly gates inference and forces end."
- "Trajectory max visible pixels is target physical size." It is policy-dependent and may be a consequence/mediator of search and approach behavior.

## Highest-Value Uncertainty

The runtime measurement path now has strong partial support but incomplete coverage. The open question is no longer whether the first few scenes expose stable geometry; it is whether the same exact path completes all 368 targets / 200 tasks. The previous run cannot answer this because it ended at task 20 from a stdout BrokenPipe.

The prior 20 full tasks are now a frozen cross-cycle overlap control. A fresh run must reproduce them exactly before proceeding, then complete the remaining tasks without transport-dependent output.

## Unique Next Experiment

EXP-SIZE-RUNTIME-METADATA-002: fresh 200-task, outcome-blind runtime geometry extraction with transport-resilient atomic per-task checkpoints. First 20 tasks must exactly reproduce the frozen prior partial run. No size-success association is part of this experiment.


## 02 research-history migration — 2026-09-18

The substantive legacy research state from Project conversation `02｜SafeVLA研究` and its branches has now been migrated into the shared control plane:

- `research/LEGACY_RESEARCH_STATE.md`: facts / observations / active hypotheses / rejected or corrected interpretations / legacy branch inventory.
- `research/EVIDENCE_REGISTER.md`: evidence IDs, raw/server/Git locations, what each source can and cannot support, and discrepancies that must be reconciled.

Executor must read both files before claiming scientific conclusions from legacy work. Raw server evidence outranks migrated summaries. In particular, do not merge the researcher's later "~50% small-category SR" recollection with the preserved 173/200 category table until run identity is reconciled; do not use historical Probe AUC or modified-policy/PT-Guard runs as B0 small-target evidence.

## Control State

The BLOCKED result `ac2c9fe80f113345f5092205b07192a47a0e0358` has been PI-acknowledged and its claim is closed. Current fresh cycle is `size-runtime-metadata-002-20260922`, experiment `EXP-SIZE-RUNTIME-METADATA-002`, status `PI_REVIEW` / `NOT_AUTHORIZED`, with `instruction_commit=null` and `claim_id=null`. No scene initialization or Executor claim is authorized until explicit user/PI approval and a green approval CI.


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
