# Next Experiment — Runtime scene-instance geometry recovery

Experiment ID: EXP-SIZE-RUNTIME-METADATA-001
Status: DRAFT
Authorization: NOT_AUTHORIZED — PI acknowledged BLOCKED handoff; no retry/resume authorized
Cycle ID: size-runtime-metadata-001-20260920

## Why this experiment now

EXP-SMALLTARGET-PHENOTYPE-001 stopped correctly at its metadata gate. Static sources mapped all 368 success-eligible broad target IDs to scene objects, but only 42 targets had candidate asset bounding boxes; zero task-level physical-size values were validated.

The next bottleneck is therefore measurement, not representation or policy mechanism. A runtime simulator metadata path may expose scene-instance bounding boxes after an exact scene load, but that must be verified in the exact historical simulator build before it can be used as the policy-independent size variable.

A PI exploratory audit of the already archived 200-row table also found severe allocation confounding: sub_house_id<20 contains 11/13 mug tasks, 8/9 basketball tasks and 1 vase task; all 8 failures in that 20-task stratum are mug/basketball. Outside that stratum, only 2 mug and 1 basketball tasks remain. Therefore the historical low-SR category observation is not independently identifiable from task allocation in this sample. This is exploratory motivation, not a formal category-effect result.

## Research question

Can the exact historical simulator/runtime environment provide a valid, policy-independent scene-instance 3D bounding box for every success-eligible broad-synset target in the 200-task historical ObjectNav set, before any SafeVLA policy action is executed?

## Hypothesis

H-RUNTIME-AABB:
After loading each archived house with the exact historical simulator build and before policy execution, runtime object metadata exposes a finite, non-zero scene-instance axis-aligned bounding box for all 368 success-eligible target IDs, allowing all 200 tasks to receive a complete task-level size summary.

## Competing explanation

H-RUNTIME-GAP:
The static gap reflects a real runtime metadata/mapping limitation, build mismatch, unsupported custom assets, or unstable scene initialization; runtime metadata will therefore fail to provide complete and reproducible scene-instance geometry.

## Reference

The reference is the BLOCKED handoff from EXP-SMALLTARGET-PHENOTYPE-001 at commit 9910cb2c5645a3549d6e8e474827ee12de1df8bd:
- 200/200 task identity recovered;
- 368 broad target IDs mapped to static scene objects;
- 42/368 candidate static asset boxes;
- 0 validated task-level physical sizes.

## Repeat / treatment

This is a measurement-recovery experiment, not a policy treatment.

Only new measurement path:
- exact-version simulator runtime object metadata immediately after deterministic scene initialization.

Preflight repeatability control:
- deterministically select 12 tasks without reading success labels:
  - first 6 stable task keys with at least one static candidate bbox;
  - first 6 stable task keys with no static candidate bbox;
- load each selected scene twice independently;
- require exact target mapping and stable geometry before full extraction.

If the preflight passes, load the 200 historical tasks once for geometry extraction.

## Unique variable

The only changed factor is **geometry metadata acquisition source**:
static asset/annotation metadata (previously insufficient) -> exact historical runtime scene-instance object metadata.

Task specs, scene files, target IDs and all policy/evaluation behavior remain fixed.

## Fixed conditions

- Use the exact archived 200 task specs and the same scene/asset datasets already hashed in the evidence packet.
- Resolve and record the exact historical simulator package/build/executable identity before scene loading. If it cannot be tied to the historical run, STOP.
- No SafeVLA checkpoint/model load.
- No Actor/Critic forward.
- No navigation policy action and no ObjectNav evaluation episode.
- Scene creation/reset required to materialize metadata is allowed; maximum 224 scene initializations (24 preflight loads + at most 200 full extraction loads).
- Geometry extraction must not load success/failure labels, episode length, Safety Cost, visible pixels or other outcome columns.
- Map targets by exact runtime object identity. No fuzzy category/name matching.
- Primary geometry field is the runtime scene-instance axis-aligned bounding box at initial scene state.
- Do not call AABB volume an intrinsic/canonical object volume: world-axis AABB may depend on the fixed scene pose/orientation.
- Object-oriented bounding box may be recorded as a secondary descriptor when the exact runtime exposes it, but may not replace missing primary AABB values.
- Do not modify B0/runtime source, benchmark task specs, success/end logic, reset/cache behavior or the historical result table.
- No target-size versus success association is permitted in this experiment.

## Control and validity checks

1. Version control: exact simulator package/build and relevant executable/source identity recorded before extraction.
2. Identity control: 368 expected broad target IDs; report exact runtime mapping count and every mismatch.
3. Geometry validity: each primary AABB must have finite positive x/y/z extents and/or eight finite corner points.
4. Repeatability: for the deterministic 12-task preflight, compare two independent initializations. Report exact equality plus max absolute/relative dimension difference.
5. Coverage: report target-level valid AABB n/368 and task-level complete n/200.
6. Static-reference check: for the 42 prior candidate-static-box targets, record whether runtime metadata exists. Do not demand numeric equality unless the scale/transform semantics are explicitly reconstructed and documented.
7. Optional OBB audit: report OBB availability separately; never mix AABB and OBB into one primary size variable.

## Metrics

- exact simulator/runtime identity status;
- preflight target mapping rate;
- preflight AABB validity rate;
- preflight repeated-load max absolute and relative difference;
- full exact-ID mapping n/368;
- valid runtime AABB n/368;
- complete task geometry n/200;
- missing/ambiguous targets by target synset and source type;
- optional OBB availability n/368;
- if and only if a task is complete: median target AABB volume and median maximum side length, computed across all valid broad targets in that task.

## Expected result

If H-RUNTIME-AABB holds, the exact runtime will provide stable AABB metadata for 368/368 broad targets and complete task-level geometry for 200/200 tasks. This would remove the measurement blocker but would **not** itself support H-SIZE; outcome association remains a later experiment.

## Falsifying result

H-RUNTIME-AABB is falsified as a complete recovery path if any success-eligible target cannot be mapped exactly to a valid runtime AABB, if the exact historical runtime build cannot be established, or if repeated scene initialization yields materially unstable geometry.

A partial result is still reported, but no missing target may be imputed and no downstream size-success analysis may start automatically.

## Alternative explanations

- the installed simulator/runtime differs from the historical evaluation build;
- custom Objaverse or built-in THOR assets expose different metadata paths;
- runtime object IDs differ from archived scene IDs;
- physics/scene initialization changes object transforms;
- world-axis AABB varies with object orientation and measures scene-instance extent rather than canonical intrinsic size;
- the needed geometry exists only after an environment mutation that would invalidate the intended initial-state measurement.

## Stop conditions

Return BLOCKED and STOP if:
- exact historical simulator build identity cannot be recovered;
- any SafeVLA model/checkpoint or policy forward would be required;
- target identity requires fuzzy mapping;
- preflight target mapping or AABB validity is incomplete;
- repeat initialization changes target geometry beyond documented numerical tolerance without an explained deterministic cause;
- obtaining the metadata requires navigation/policy actions or baseline/runtime source modification;
- units/extent semantics cannot be established well enough to interpret the runtime field;
- any success/failure association, Probe, replay, reset treatment, Safe-vs-IL comparison or new B0 evaluation would be started.

## Resource budget

- max GPU: 1, simulator graphics/runtime only; model GPU use is forbidden.
- max ObjectNav/SafeVLA episodes: 0.
- max scene initializations: 224.
- model loads: 0.

## Required outputs

- `research/handoffs/size-runtime-metadata-001-20260920/RESULT_SUMMARY.md`
- `research/handoffs/size-runtime-metadata-001-20260920/RUN_MANIFEST.json`
- `research/handoffs/size-runtime-metadata-001-20260920/ARTIFACT_INDEX.json`
- `research/handoffs/size-runtime-metadata-001-20260920/REVIEW_NOTES.md`
- `research/handoffs/size-runtime-metadata-001-20260920/runtime_version_manifest.json`
- `research/handoffs/size-runtime-metadata-001-20260920/preflight_repeatability.csv`
- `research/handoffs/size-runtime-metadata-001-20260920/target_geometry.csv`
- `research/handoffs/size-runtime-metadata-001-20260920/task_geometry.csv`
- `research/handoffs/size-runtime-metadata-001-20260920/coverage_report.md`
- `research/handoffs/size-runtime-metadata-001-20260920/extract_runtime_geometry.py`

After producing the handoff, STOP. PI must review coverage and measurement validity before any size-performance association is authorized.
