# Next Experiment — Small-target phenotype audit

Experiment ID: EXP-SMALLTARGET-PHENOTYPE-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: smalltarget-phenotype-001-20260918

## Why now

001A has already provided enough provenance for read-only diagnosis. The old 001B preflight authorization expired and was never claimed. The scientific bottleneck is earlier: several categories were observed by the researcher to have roughly 50% SR, but "low-SR category" has not yet been shown to mean "small physical target."

Historical Probe AUC values 0.955/0.982/0.992 are exploratory close-and-visible / stop-legality decoding results, not small-object-size probes, so they cannot answer this question.

## Research question

In the existing historical ObjectNav full-200 evaluation, is lower target physical size associated with lower success after first checking per-category sample size/uncertainty and available pre-policy difficulty covariates?

## Hypotheses

H-SIZE: smaller targets, measured from policy-independent 3D object metadata, have lower SR.

Competing explanations:
- H-CATEGORY: semantic category explains the low SR, not size.
- H-DIFFICULTY: low-SR categories occur in harder houses/tasks, longer expert paths, or farther initial conditions.
- H-SAMPLE: the apparent ~50% SR is driven by small n and wide uncertainty.
- H-BEHAVIOR: low visible-pixel counts are consequences of failed search/approach and cannot be used as an exogenous size definition.

## Study type / unique explanatory variable

Zero-rollout observational audit. No policy treatment.

Primary explanatory variable:
- per task, median 3D bounding-box volume across all valid broad-synset target object IDs in that house;
- also record median maximum side length as a robustness descriptor.

Do not use target distance or trajectory max-visible-pixels as "physical size."

## Control

Continuous size analysis first. For descriptive comparison, define small/medium/large tertiles from the full analyzable set without using success labels; the large-target tertile is the descriptive control.

## Fixed conditions

- 0 GPU, 0 episode, no model load, no AI2-THOR.
- Do not modify B0, runtime source, evaluation semantics, success/end logic, reset/cache behavior, or the dev worktree.
- Use only existing historical full-200 raw results, task specs, and static scene metadata.
- Historical 173/200=86.5% is context; raw episode-level evidence must be reconciled to it before interpretation.
- Post-rollout visible pixels, episode length, final distance, premature end and videos are downstream phenotype variables only.
- Old PT-Guard Probe trajectories must not be used as clean-B0 mechanism evidence.

## Required evidence

Build one row per historical task with:
- stable task key;
- target category/synset;
- success;
- episode length;
- expert_length if available;
- failure termination type if derivable;
- all valid target object IDs;
- primary physical-size summary;
- initial target distance only if derivable from pre-policy/static evidence;
- existing max nav-visible-pixels only as downstream phenotype.

Report:
- every category: n, successes, SR, Wilson 95% CI;
- suspected low-SR categories if present, without restricting the analysis to them;
- continuous association between log physical size and success, with effect estimate and uncertainty;
- if complete data permit: logistic(success ~ log_size + expert_length [+ independently available initial_distance]);
- small/medium/large tertile SR + Wilson CI;
- missing metadata count, analyzable n, duplicate-key check, and reconciliation to 200 tasks / 173 successes.

Do not include category as a fixed effect if the small dataset makes physical size non-identifiable; instead report category-stratified sensitivity and state the limitation.

## Expected result

H-SIZE is strengthened only if the low-SR observation survives sample-size/CI inspection and policy-independent physical size shows a stable association with failure under available difficulty adjustment.

H-SIZE is weakened if category SR is unstable, size has little/reversed association, or the relation disappears after available controls.

## Alternative explanations

Category semantics, training frequency, scene/house layout, path difficulty, starting distance, occlusion and sample-size instability remain live alternatives.

## Stop conditions

Return BLOCKED rather than running anything new if:
- raw full-200 evidence cannot be tied to the historical 173/200 result;
- stable task identity cannot be reconstructed;
- static target-size metadata cannot be obtained without simulator/model execution;
- size would have to be defined using distance or trajectory-derived visible pixels;
- any new episode, model load, GPU inference, AI2-THOR launch, or baseline code change would be required.

## Required derived artifacts

- research/handoffs/smalltarget-phenotype-001-20260918/episode_table.csv
- research/handoffs/smalltarget-phenotype-001-20260918/category_sr.csv
- research/handoffs/smalltarget-phenotype-001-20260918/size_analysis.md

## Required handoff outputs

- research/handoffs/smalltarget-phenotype-001-20260918/RESULT_SUMMARY.md
- research/handoffs/smalltarget-phenotype-001-20260918/RUN_MANIFEST.json
- research/handoffs/smalltarget-phenotype-001-20260918/ARTIFACT_INDEX.json
- research/handoffs/smalltarget-phenotype-001-20260918/REVIEW_NOTES.md

After handoff, STOP. Do not start Probe, video-mechanism, reset, Safe-vs-IL, or a new full-200 run.
