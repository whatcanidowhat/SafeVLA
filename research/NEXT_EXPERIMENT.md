# Next Experiment — Transport-resilient complete runtime geometry extraction

Experiment ID: EXP-SIZE-RUNTIME-METADATA-002
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: size-runtime-metadata-002-20260922

## Why this experiment now

EXP-SIZE-RUNTIME-METADATA-001 established a strong but incomplete measurement path:
- the exact historical AI2-THOR build was recovered;
- 12 tasks loaded twice produced 18/18 exact-equal target-geometry comparisons;
- the first 20 full tasks produced 26/26 exact-mapped valid creation-state AABBs and 20/20 complete task summaries;
- execution stopped only because a progress `print(..., flush=True)` hit `BrokenPipeError` after task 20;
- 342 targets / 180 tasks were never attempted, so complete coverage remains unresolved.

The prior partial dataset is not a complete scientific result, but it is now a useful frozen overlap reference. The next experiment should repeat all 200 tasks under the same geometry semantics while changing only output/checkpoint transport so the measurement can complete without depending on an inherited stdout pipe.

## Research question

Under the exact same historical simulator build, scene inputs, target identities and creation-state geometry definition, can a transport-resilient extraction recover valid scene-instance AABB geometry for all 368 broad targets / 200 tasks, while exactly reproducing the 20 previously completed full tasks?

## Hypothesis

H-COMPLETE-RUNTIME-GEOMETRY:
The previous blocker was transport-only. A fresh full extraction with durable per-task checkpoints and no inherited stdout dependency will:
1. exactly reproduce the prior 20-task full geometry; and
2. provide exact-ID valid creation-state AABB geometry for all 368/368 broad targets and complete task summaries for 200/200 tasks.

## Competing explanation

H-LATE-RUNTIME-GAP:
Although the first 20 tasks succeeded, later tasks contain runtime-ID, scene-loading, geometry-validity or stability failures that prevent complete 368/368 coverage. The prior BrokenPipe merely occurred before those failures could be observed.

## Reference

Frozen result commit:
`ac2c9fe80f113345f5092205b07192a47a0e0358`

Reference facts:
- historical build commit: `966bd7758586e05d18f6181f459c0e90ba318bec`;
- 20 full tasks already measured;
- 26 full target geometries valid;
- original partial `target_geometry.csv`, `task_geometry.csv` and per-task geometry sample JSONs are read-only reference artifacts;
- prior 12-task double-load preflight already passed and is not repeated as a separate 24-load stage.

## Repeat / treatment

This is a measurement repeat/completion, not a policy treatment.

Fresh run:
- deterministically rerun **all 200 historical tasks once** from the beginning;
- the first 20 tasks form a cross-cycle overlap control against the prior partial run;
- after task 20, require exact equality of target IDs and recorded AABB geometry to the frozen prior values before proceeding to tasks 21–200;
- the final primary dataset comes entirely from this fresh cycle. Do not splice old and new task rows.

## Unique variable

The only intended change from EXP-SIZE-RUNTIME-METADATA-001 is **transport/checkpoint implementation**:

Previous:
- progress writes to inherited stdout could terminate the process;
- full-task summaries were primarily materialized at finalization.

New:
- no progress or final status depends on inherited stdout/stderr;
- redirect process diagnostics to local server files or suppress inherited-pipe writes;
- persist each completed task's raw geometry and task summary atomically before moving to the next task;
- maintain a durable machine-readable progress/checkpoint manifest after every task;
- final CSVs are regenerated from durable per-task records.

Simulator build, scene/task bytes, target IDs, CreateHouse initialization, `autoSimulation=False`, no-post-load-physics semantics and geometry extraction logic must otherwise remain fixed.

## Fixed conditions

- Exact historical AI2-THOR build commit `966bd7758586e05d18f6181f459c0e90ba318bec`, CloudRendering.
- Before any scene initialization, re-hash the runtime executable, UnityPlayer, Assembly-CSharp and relevant Python/runtime source files; require identity with the accepted prior runtime manifest unless PI explicitly reviews a mismatch.
- Use the same archived 200 task specs and the same scene/asset inputs as the prior cycle.
- Outcome-blind execution: do not read success/failure, episode length, Safety Cost, visible pixels, category SR tables or any outcome-derived file.
- No SafeVLA checkpoint/model import or load.
- No Actor/Critic forward.
- No navigation policy action.
- No ObjectNav/SafeVLA evaluation episode.
- Exactly one fresh scene initialization per historical task; maximum 200 scene initializations.
- Exact runtime object-ID mapping only; no fuzzy matching.
- Primary descriptor remains initial creation-state world-axis AABB. It is a scene-instance extent, not canonical intrinsic volume.
- Record OBB availability/points as secondary metadata when exposed; do not substitute OBB for a missing primary AABB.
- No size-success/category association in this experiment.
- No modification of B0/runtime source, task specs, success/end logic, reset/cache behavior or historical outcome tables.

## Transport-resilience requirements

- The extractor must not call an inherited stdout/stderr progress `print` whose failure can abort measurement.
- After each task, write a per-task record using write-temp + atomic rename (or an equivalently crash-safe local operation).
- After each task, update a durable progress manifest containing completed task keys, scene-initialization count and last successful task.
- Re-running the extractor after a process failure is **not authorized in this cycle** unless the failure happens before any scene initialization. Any post-initialization failure returns BLOCKED to PI; no implicit resume.
- Final CSVs must be derivable solely from the fresh cycle's durable task records.
- Do not overwrite or mutate prior-cycle handoff artifacts.

## Cross-cycle overlap control

Before task 21:
- compare all target IDs and primary AABB vectors for fresh tasks 1–20 against the frozen prior full-pass values;
- require 20/20 task identity agreement;
- require all previously recorded 26 targets to exist with exact ID match;
- report exact equality and max absolute/relative geometry difference;
- default acceptance: exact equality; if floating representation differs but is within the previously registered numerical tolerance, STOP as BLOCKED for PI interpretation rather than silently accepting a looser criterion.

## Metrics

- runtime/version/hash identity status;
- fresh scene initializations / 200;
- first-20 task overlap agreement / 20;
- first-20 target overlap agreement / 26;
- overlap exact-equal target count;
- overlap max absolute and relative AABB-vector difference;
- exact runtime target-ID mapping n / 368;
- valid primary AABB n / 368;
- complete task geometry n / 200;
- OBB availability n / 368;
- missing/ambiguous/invalid target list if any;
- per-task median AABB volume and median maximum side only when all broad targets in that task are valid;
- durable checkpoint consistency: completed task record count equals progress-manifest count equals final complete-task count.

## Expected result

If H-COMPLETE-RUNTIME-GEOMETRY holds:
- the first 20 tasks exactly reproduce the prior partial run;
- 368/368 targets exact-map to valid creation-state AABBs;
- 200/200 tasks receive complete fresh-cycle geometry summaries;
- no geometry/mapping failure occurs;
- H-SIZE still remains untested until a later, separately approved outcome-association experiment.

## Falsifying result

The hypothesis is falsified as a complete recovery path if:
- accepted runtime/source identity changes;
- any first-20 overlap target is not exactly reproduced;
- any of the 368 target IDs cannot be exactly mapped;
- any primary AABB is missing, non-finite or non-positive;
- any task is incomplete;
- the extraction cannot finish within 200 scene initializations;
- a process/transport failure occurs after scene initialization begins.

Partial data must be preserved but cannot be silently combined with the previous cycle to claim complete coverage.

## Alternative explanations

- a later scene or built-in/custom asset exposes different runtime metadata than the first 20 tasks;
- some archived scene object IDs are not stable under runtime creation;
- creation-state physics/asset hooks produce scene-specific metadata failures in later houses;
- the output-channel crash was only an early interruption and hid a later scientific coverage gap;
- world-axis AABB is reproducible but remains an orientation-dependent scene-instance extent;
- the historical build chain is source/version matched but historical executable byte equality was not contemporaneously archived.

## Stop conditions

Return BLOCKED and STOP if:
- runtime/source hashes or build identity fail the frozen identity checks;
- any outcome-derived file/column would be needed;
- any SafeVLA model/policy component would need to load;
- any policy/environment navigation action would be needed;
- the first-20 overlap is not exact;
- any task requires fuzzy target mapping;
- any primary AABB is invalid/missing;
- any post-start process/transport error occurs;
- completing the dataset would require >200 scene initializations;
- any size-success association, Probe, replay, reset treatment, Safe-vs-IL comparison or B0 evaluation would start.

## Resource budget

- max GPU: 1, simulator graphics/runtime only;
- max ObjectNav/SafeVLA episodes: 0;
- max scene initializations: 200;
- model/checkpoint loads: 0;
- Actor/Critic forwards: 0.

## Required outputs

- `research/handoffs/size-runtime-metadata-002-20260922/RESULT_SUMMARY.md`
- `research/handoffs/size-runtime-metadata-002-20260922/RUN_MANIFEST.json`
- `research/handoffs/size-runtime-metadata-002-20260922/ARTIFACT_INDEX.json`
- `research/handoffs/size-runtime-metadata-002-20260922/REVIEW_NOTES.md`
- `research/handoffs/size-runtime-metadata-002-20260922/runtime_version_manifest.json`
- `research/handoffs/size-runtime-metadata-002-20260922/cross_cycle_overlap.csv`
- `research/handoffs/size-runtime-metadata-002-20260922/target_geometry.csv`
- `research/handoffs/size-runtime-metadata-002-20260922/task_geometry.csv`
- `research/handoffs/size-runtime-metadata-002-20260922/coverage_report.md`
- `research/handoffs/size-runtime-metadata-002-20260922/batch_manifest.json`
- `research/handoffs/size-runtime-metadata-002-20260922/extract_runtime_geometry_v2.py`

After handoff publication, STOP. Complete geometry coverage alone does not authorize a size-performance analysis.
