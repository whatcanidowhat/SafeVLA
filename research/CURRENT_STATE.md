# Current State

Updated: 2026-09-24（Asia/Shanghai）
Mode: PI_REVIEW. No experiment is currently authorized for execution.

## Latest PI decision

`EXP-END-LEGALITY-CALIBRATION-001` is **SUPERSEDED BEFORE APPROVAL / NOT EXECUTED** by explicit researcher decision.

Reason: the proposed legal-vs-illegal terminal `p(end)` calibration repeats an evidence layer that has already been explored. The completed termination-dynamics work already established the relevant `p(end)` behavior sufficiently for the current mainline; another probability-only audit is not expected to materially change the research state.

## Preserved facts

- 16/16 historical sub-horizon failures end by executing `end`.
- First-decision `p(end)` is sub-resolution in all 16; preterminal maxima remain below 0.5.
- The terminal `p(end)` jump also appears in successful ends, so it is not failure-specific.
- H-PRIOR and a simple gradual H-SEARCH-TRIGGERED explanation are weakened.
- Historical Probe evidence remains exploratory/provenance-limited.
- H-RESET remains unresolved and must be controlled before formal hidden-state / Probe causal interpretation.

## Current research direction

Do not spend another cycle on `p(end)` description or reproduction. The next experiment has **not yet been selected or authorized**. The next design should move to a deeper mechanism layer, while respecting the existing H-RESET gate before formal hidden-state interpretation.

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
