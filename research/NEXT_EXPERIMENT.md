# PI Cycle 4 — Controlled mid-episode context truncation on historical early-end tasks

Experiment ID: EXP-COUNTER-DRIFT-EARLYEND-001C
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: counter-drift-earlyend-001c-20261009

## Mechanism under test

EXP-RESET-001A established R1 under identical teacher-forced current inputs:

`prior episode Actor state -> earlier decoder rollover -> current-episode temporal-context truncation -> hidden/logit/action-preference divergence`.

The key defect is not generic reading of old-episode tokens. Episode-local time resets at a new task, while the root Actor model-side counter does not. Old-episode cache is masked before rollover. If the inherited counter reaches the decoder limit (500) during the new episode, model-side position rolls to zero while episode-local time continues. The decoder then stops addressing the current episode's earlier temporal context.

EXP-RESET-EARLYEND-001B was correctly BLOCKED because the historical four-worker task-to-worker order was not persisted. This cycle does not guess historical worker state. It tests the mechanism causally under one controlled valid carry state on the same 16 confirmed historical failed-end tasks.

## Research question

Does cross-episode Actor counter drift, by forcing a premature decoder rollover and truncating already accumulated current-episode temporal context, causally alter Actor outputs, action selection, online trajectory evolution, and termination outcome on the 16 confirmed historical premature failed-end ObjectNav tasks?

## Hypotheses

### H1 — Actor causal effect
Before the controlled OFF rollover, paired OFF/ON target trajectories and Actor outputs remain equivalent within the declared tolerance. At the OFF rollover boundary, effective current-episode context collapses and Actor hidden/logit outputs diverge.

### H2 — trajectory propagation
If the policy distribution change alters the executed action, the environment trajectories subsequently diverge and later policy differences contain both the direct reset effect and trajectory-feedback amplification.

### H3 — termination relevance
At least a subset of the historical early-end tasks changes failed-end status, end timing, or success outcome under the clean Actor reset.

### H0 / competing explanation
The context truncation can perturb representation/distribution without materially changing executed navigation or premature termination. Very-early termination may arise from a separate termination/calibration/exploration mechanism.

## P0 — frozen donor and valid CARRY-450

Use the already captured real Actor inputs from EXP-RESET-001A. The fixed donor is:

- task: ObjectNavType
- sub_house_id: 132
- house: 13352
- goal: find a bowl
- captured episode decisions: 600
- accepted 001A result commit: `2f7864ae2c3a4761932aa4d1fedd6b6d5b3388c6`

Construct exactly one donor state from a clean root Actor by replaying the donor's first 450 sequential captured Actor inputs through the same root Actor forward path.

Do not fabricate `counter=450` with an incompatible cache. The resulting state must have:
- root Actor `time_step_counter=450`;
- each decoder-layer K/V state populated by the 450 valid sequential forwards;
- checkpoint/DINO identity matching the accepted baseline;
- deterministic reconstruction hash/summary recorded.

This one CARRY-450 package is cloned for all 16 target pairs. The target tasks never manufacture their own carry state.

Large tensor snapshots remain server-side and are indexed by path/hash/size/shape/dtype in ARTIFACT_INDEX and donor_state_manifest; do not commit tensor files.

## Controlled rollover geometry

Zero-based target local steps are used.

OFF begins the target episode with Actor counter 450. ON begins with Actor counter 0.

Expected current-episode effective-context progression:

- target local step 0: OFF context=1, ON context=1;
- ...
- target local step 49: OFF context=50, ON context=50;
- target local step 50:
  - OFF sees counter>=500 before the decision path, rolls model-side position to 0, and can address only the newly written current token: effective current-episode context=1;
  - ON continues normally and can address target steps 0..50: effective current-episode context=51.

Therefore the key controlled event is:

`OFF current-episode effective context: 50 -> 1`

at target local step 50.

This is an addressability/context collapse, not a claim that the entire physical cache tensor is zeroed.

## Conditions and unique variable

For every target case, clone the identical CARRY-450 package and initialize the exact same target task/environment.

CONTROL / OFF:
- call the official episode reset path;
- retain root Actor counter/K/V exactly as official behavior does.

TREATMENT / ON:
- call the same official episode reset path;
- additionally apply only the reviewed Actor reset package: root Actor counter=0 and root Actor decoder K/V zeroed;
- reward critic and cost critic state are not modified by the treatment.

Unique intervention:
`root Actor episode-boundary temporal state retained vs cleared`.

Fixed across the pair:
- checkpoint, DINO, source/runtime;
- task specification, scene, initial pose and target;
- success/end semantics and horizon=600;
- action mapping;
- sampling protocol;
- visual augmentation protocol;
- environment seed and initialization;
- critic treatment (unchanged);
- instrumentation semantics.

No Stop Gate, Oracle Gate, Probe, checkpoint comparison, size intervention, or benchmark-wide run is allowed.

## Target set — all 16 confirmed historical failed-end tasks

The target IDs and historical zero-based failed-end steps are frozen:

| sub_house_id | historical end step | role |
| ---: | ---: | --- |
| 55 | 1 | historically pre-rollover reference |
| 0 | 7 | historically pre-rollover reference |
| 68 | 15 | historically pre-rollover reference |
| 6 | 46 | historically pre-rollover reference |
| 3 | 52 | primary context-truncation target |
| 110 | 98 | primary context-truncation target |
| 196 | 112 | primary context-truncation target |
| 175 | 130 | primary context-truncation target |
| 87 | 135 | primary context-truncation target |
| 13 | 178 | primary context-truncation target |
| 117 | 180 | primary context-truncation target |
| 46 | 183 | primary context-truncation target |
| 4 | 186 | primary context-truncation target |
| 125 | 196 | primary context-truncation target |
| 24 | 376 | primary context-truncation target |
| 176 | 449 | primary context-truncation target |

The four historically very-short cases (55, 0, 68, 6) are reference cases, not proof that they will terminate before step 50 in this controlled run. For any case that actually terminates before OFF rollover, OFF and ON must remain paired-equivalent through termination or the case fails validity.

## Pairing and randomness

Run one OFF and one ON target episode for every case: 16 x 2 = 32 live target episodes.

Predeclare pair execution order by the frozen case-list index:
- even index: OFF then ON;
- odd index: ON then OFF.

Before each pair, snapshot all relevant RNG states used by the policy/evaluator/augmentation. Restore the identical RNG snapshot for the second condition. Logging/hook code must consume no RNG and must not add a policy forward.

Common random numbers are interpretable only while both branches have consumed the same stochastic call pattern. Record RNG-state/hash evidence at pair start and first action divergence.

After first executed-action divergence, allow both branches to continue naturally; do not teacher-force them back together.

## P1 — pre-rollover equivalence validity gate

For every pair, the primary causal analysis requires equivalence before OFF rollover.

For target local steps 0..49 while both episodes remain active:
- same target task and initial environment state;
- same observation/visual-feature hash;
- same text feature / previous action / episode-local timestep;
- same executed action;
- same pose;
- root Actor hidden/logit differences remain below declared tolerance except ordinary floating-point roundoff.

Use the accepted 001A numerical tolerance:
- atol = 1e-5
- rtol = 1e-5

If sampled or executed actions diverge before step 50, mark that case `VALIDITY_FAIL_PRE_ROLLOVER` and exclude it from the context-truncation causal outcome count. Preserve it in outputs.

If an episode terminates before step 50, OFF/ON termination behavior must be identical under the paired run for it to serve as pre-rollover negative-control evidence.

## P2 — effective current-episode context as a primary mechanistic variable

At every decision record:
- root Actor counter_before and counter_after;
- decoder start_pos;
- whether rollover occurred;
- cache read prefix length;
- mask-derived current-episode accessible range;
- `effective_current_episode_context_len`.

The runner must derive the effective-context length from the actual decoder address/mask semantics, not assume it from the intended formula alone.

Primary mechanistic validation requires:
- OFF effective context reaches 50 at local step 49;
- OFF effective context collapses to 1 at local step 50;
- ON effective context reaches 51 at local step 50;
- no earlier unexplained context collapse.

If the actual runtime semantics do not produce this pattern, return INVALID/BLOCKED rather than reinterpret the experiment post hoc.

## P3 — capture every Actor decision output without extra forward

For every live decision in both conditions capture from the actual existing Actor forward:

Identity/state:
- case_id, condition, local_step;
- counter_before/counter_after, start_pos, rollover_event;
- effective_current_episode_context_len;
- paired_identical_input flag.

Actor inputs:
- visual/encoder-output hash;
- text-feature hash;
- previous action;
- episode-local timestep;
- decoder-input hash.
Large raw input tensors may remain server-side with hashes.

Actor internal output:
- L1/L2/L3 hidden state vectors (raw server-side);
- per-layer paired hidden max-abs, L2 distance and cosine similarity in analysis.

Policy output:
- complete runtime action-logit vector;
- complete runtime action-probability vector;
- runtime-resolved action-name mapping and end index;
- argmax action;
- sampled action;
- executed action;
- p(end), end rank, end-vs-best-other margin.

Environment:
- pose before and after action (x/y/z/yaw/camera horizon or runtime equivalents);
- collision if already available;
- episode done;
- stop_legal / successful_if_done only if obtainable read-only without altering simulator semantics;
- passive target visibility/room fields only if already present; no extra simulator query.

Instrumentation must not add Actor forwards, alter action selection, mutate model/env state, or consume RNG.

## P4 — trajectory-drift decomposition

For every case compute:

- `t_context`: first effective-context collapse;
- `t_hidden_L1/L2/L3`: first paired hidden difference above tolerance while inputs remain identical;
- `t_logit`: first paired logit difference above tolerance while inputs remain identical;
- `t_argmax`: first argmax-action divergence;
- `t_sample`: first sampled-action divergence;
- `t_exec`: first executed-action divergence;
- `t_pose`: first pose/observation divergence.

Phase A — direct Actor causal window:
before first executed-action divergence, paired observations/inputs remain identical. Actor differences in this window may be attributed to the temporal-state intervention/context truncation subject to validity checks.

Phase B — trajectory-feedback window:
after executed-action divergence, subsequent input differences are consequences of different trajectories. Later policy differences are reported as drift propagation and are not attributed step-by-step solely to reset.

## Primary outcome metrics

Per target pair:
- pre-rollover validity PASS/FAIL;
- OFF/ON rollover step and effective context lengths;
- all divergence times above;
- OFF/ON failed_end;
- OFF/ON end_step;
- OFF/ON success;
- OFF/ON episode length;
- OFF/ON official Safety Cost (descriptive only);
- historical failed-end step for reference;
- rescue / regression / unchanged classification.

Definitions:

`rescue`: OFF executes an unsuccessful end and ON does not, or OFF failure -> ON success.

`regression`: ON has a worse termination/outcome switch relative to OFF; report symmetrically.

Do not count a validity-failed pair as mechanistic support even if its final outcome differs.

## Decision rules

### C1 — context truncation has direct termination relevance
Required:
1. pre-rollover validity passes;
2. OFF context collapses at the predicted rollover while ON context continues;
3. hidden/logit divergence begins at or after the context-collapse boundary under identical inputs;
4. at least one valid case changes termination outcome/timing.

Per-case causal claim is allowed for a valid changed case.

A stronger cross-task claim requires at least two independent valid target tasks with OFF failed-end -> ON no failed-end or success.

Allowed wording:
> Controlled mid-episode context truncation causally changes termination behavior in a subset of tested historical early-end tasks.

Not allowed:
> the original historical failures were definitely caused by this bug.
> the official benchmark SR is improved.

### C2 — online trajectory effect without termination effect
Valid pairs show context -> Actor -> action/trajectory divergence but termination outcomes are essentially unchanged.

Conclusion: the implementation defect changes navigation behavior but is not yet shown to be a major premature-end cause.

### C3 — representation/policy effect only
Hidden/logit distributions diverge but executed actions/trajectory remain materially unchanged.

Conclusion: policy is behaviorally robust to the tested context truncation.

### C4 — no meaningful downstream effect
The verified context collapse yields no meaningful hidden/logit/action/trajectory effect beyond tolerance. This substantially downgrades the mechanism as a practical failure explanation.

### C5 — INVALID/BLOCKED
Return INVALID/BLOCKED and do not interpret outcome causally if:
- donor CARRY-450 cannot be reconstructed validly;
- OFF does not produce the predicted context collapse or ON unexpectedly rolls over;
- target initialization/RNG pairing cannot be matched;
- executed action or pose diverges before step 50 without an explained numerical/RNG validity cause;
- logging adds forward calls or changes policy/env/RNG semantics;
- treatment changes critics or another model/evaluator behavior.

## Output/data policy

Git-readable PI-required outputs are compact summaries and manifests. Full per-decision raw outputs are mandatory server evidence and must be hashed/indexed, but large tensors/traces must not be committed if they exceed control limits.

Mandatory server-side raw evidence includes:
- complete per-step action logits/probabilities for every live Actor decision;
- L1/L2/L3 hidden vectors for every live Actor decision;
- per-step state/counter/context and trajectory records;
- pair-start RNG snapshots or hashes sufficient to verify pairing;
- donor CARRY-450 snapshot and reconstruction evidence.

`trace_index.json` must list each raw server artifact with path, SHA256, bytes, row/shape/dtype metadata and case/condition coverage.

## Required Git-readable outputs

- `research/handoffs/counter-drift-earlyend-001c-20261009/RESULT_SUMMARY.md`
- `research/handoffs/counter-drift-earlyend-001c-20261009/RUN_MANIFEST.json`
- `research/handoffs/counter-drift-earlyend-001c-20261009/ARTIFACT_INDEX.json`
- `research/handoffs/counter-drift-earlyend-001c-20261009/REVIEW_NOTES.md`
- `research/handoffs/counter-drift-earlyend-001c-20261009/donor_state_manifest.json`
- `research/handoffs/counter-drift-earlyend-001c-20261009/target_case_manifest.csv`
- `research/handoffs/counter-drift-earlyend-001c-20261009/paired_episode_results.csv`
- `research/handoffs/counter-drift-earlyend-001c-20261009/divergence_summary.csv`
- `research/handoffs/counter-drift-earlyend-001c-20261009/trace_index.json`
- `research/handoffs/counter-drift-earlyend-001c-20261009/analysis.md`
- `research/handoffs/counter-drift-earlyend-001c-20261009/run_counter_drift_earlyend_001c.py`
- `research/handoffs/counter-drift-earlyend-001c-20261009/validate_outputs.py`

## Resource budget

- max GPU: 1;
- max live target episodes: 32;
- donor construction: offline replay only, not a live simulator episode;
- worker count for target evaluation: 1 unless the frozen runner requires otherwise; do not introduce multi-worker scheduling;
- target horizon: 600;
- no unplanned retries or extra seeds.

If any pair fails for engineering reasons, preserve the failure and stop/return BLOCKED if the frozen 32-episode budget cannot complete the planned pair without an explicitly authorized rerun.

After handoff publication, STOP. Replication across additional seeds or an official full-200 reset benchmark requires a new PI cycle.
