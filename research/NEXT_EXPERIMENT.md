# PI Cycle 2 — Reset state-carrier causal test

Experiment ID: EXP-RESET-001A
Status: APPROVED
Authorization: APPROVED_FOR_CODEX
Cycle ID: reset-001a-20261007

## Why this is now the unique next experiment

The project has finished the historical premature-end probability audit and explicitly rejected further probability-only end-legality work as low incremental value. The unresolved blocker from 02｜SafeVLA研究 is Gate B:

- Gate A is sufficient for the audited checkpoint/runtime/action/logger path.
- Gate C is sufficient for the audited end/success timing path.
- Gate B remains FAIL because model-side Actor/Reward-Critic/Cost-Critic `time_step_counter` and decoder K/V cache survive episode reset.
- A short historical control showed that old cache was masked at a fresh episode start, but did not test later cumulative rollover.
- Actor is the action-producing branch, so hidden-state / Probe / readout claims remain confounded until Actor-side reset carry is causally bounded.

This cycle asks only:

> Does cross-episode model-side state carry actually change Actor decisions?

It does not test SR impact yet.

## Research question

In the official SafeVLA inference implementation, can retained Actor `time_step_counter / decoder K-V cache` cause the **same teacher-forced episode input sequence** to produce different Actor hidden states, logits, or actions solely because the episode starts from a different valid model-side history position?

## Current fact vs open claim

Established:
`cross-episode model-side state retention exists`.

Not established:
`state retention -> Actor policy change -> navigation/SR change`.

This experiment tests only the first arrow:
`state retention -> Actor hidden/logit/action difference`.

## P0 — Complete decoder state-carrier audit

Before implementing any reset treatment, statically audit the root Actor decoder and document:

- where `time_step_counter` is owned and updated;
- K-cache owner, tensor shape and lifetime;
- V-cache owner, tensor shape and lifetime;
- cache write index / `start_pos` semantics;
- attention-mask interaction with cached positions;
- exact rollover condition and code path around `max_steps=500`;
- whether any official reset/cache-clear API exists;
- whether Actor / Reward Critic / Cost Critic each own independent counters/caches or share state;
- whether clearing only `time_step_counter=0` is semantically valid;
- the minimal correct Actor-only episode reset package.

### P0 decision gate

Do **not** assume that "counter=0" alone is a valid reset.

If code shows counter and cache position/content are semantically coupled, the treatment must reset the coupled Actor state together. If a correct Actor-only reset cannot be defined without changing shared state used by other branches, return R4 / BLOCKED.

### P0 outputs

- `decoder_reset_audit.md`
- `decoder_state_map.json`

The state map must contain at least:

`state_carrier | owner_module | lifetime | episode_reset_now | rollover_condition | actor_effect_path | correct_reset_operation`.

## P1 — Single-variable RESET_STATE_FIX treatment design

Implement an explicit switch:

`RESET_STATE_FIX=0`
- reference/control;
- official behavior;
- no Actor state reset beyond the original code path.

`RESET_STATE_FIX=1`
- treatment;
- at episode boundary, reset only the **Actor decision branch** episode-local state identified by P0 as necessary and sufficient.

Do not reset Reward/Cost critic state in 001A unless P0 proves Actor state is physically shared and cannot be isolated. If shared reset is unavoidable, stop as R4 rather than pretending the Actor effect is identified.

Frozen conditions:
- checkpoint;
- local DINO resource;
- success/end semantics;
- horizon;
- reward/cost definitions;
- action mapping;
- stochastic/greedy mode;
- augmentation;
- task inputs;
- environment semantics;
- policy forward count.

No Stop Gate, PT-Guard, steering, reranking, action rewrite, success-oracle gate, Probe, size analysis or benchmark rerun.

### A/A OFF protection

Because accepted B0 is stochastic and historical same-seed repeats are not bitwise trajectory-identical, **live trajectory equality is not by itself a valid A/A criterion**.

The primary A/A test must therefore use the **same captured Actor input tensors + the same RNG state/common random numbers** to compare:
- untouched official reference implementation;
- patched implementation with `RESET_STATE_FIX=0`.

Require exact/declared-tolerance equivalence for:
- Actor hidden output;
- full action logits/probabilities;
- sampled action under the same RNG draw;
- greedy action;
- selected action/history update logic.

Also statically verify that the OFF branch does not alter the environment call path. If an additional live smoke is run, report it only as a smoke; do not require two stochastic trajectories to be bitwise identical.

Any model-level A/A difference -> R4 / STOP.

### P1 outputs

- `reset_fix_design.md`
- `reset_fix.patch`
- `aa_off_equivalence.json`

Treatment code may exist locally for execution, but must not be committed into the B0 execution branch. The patch is the review artifact. Research-loop handoff artifacts are still published normally for PI review.

## P2 — Rollover stress test

The purpose is to expose the rollover mechanism under a fixed input sequence, not to estimate benchmark performance.

### Fixed test sequence

Acquire or recover one real B0 Actor-decoder input sequence long enough to observe at least 220 local decisions after the synthetic episode boundary. Prefer existing raw/captured evidence if it contains exact Actor inputs; otherwise a bounded read-only B0 capture is allowed.

Live budget:
- maximum 4 ObjectNav episodes total;
- only for obtaining a sufficiently long fixed Actor-input sequence and/or one A/A smoke;
- no performance conclusion from these episodes.

If a valid >=220-decision test sequence cannot be obtained within the budget, return BLOCKED.

### Two offline conditions

Condition A — CLEAN-START
- episode starts with Actor counter=0;
- Actor cache reset according to the P0-defined correct operation.

Condition B — CARRY-300
- episode starts from a **valid** Actor state carrier corresponding to 300 prior decisions;
- if cache contents are required for semantic validity, create this state by replaying a fixed 300-step warm-up prefix through the Actor and then applying the official episode-boundary semantics that retain Actor state;
- do not fabricate `counter=300` with an incompatible empty/stale cache.

Then feed the exact same teacher-forced test sequence to A and B:
- identical Actor decoder input tensors;
- identical episode-local `time_step`;
- identical masks;
- identical recorded previous-action inputs;
- identical weights;
- identical augmentation result already embedded in saved Actor inputs.

Once action outputs diverge, **do not feed divergent predicted actions back**. Continue teacher-forcing the frozen recorded inputs so the only variable remains initial model-side state.

### Sampling comparison

Primary action comparison: argmax/greedy action from the logits.

Optional stochastic comparison: use a synchronized common-random-number stream for both conditions so any sampled-action difference is attributable to the changed distribution rather than a different RNG draw.

### Per-step record

Record:
- local_timestep;
- Actor `time_step_counter`;
- cache write/start position;
- full action logits/probabilities;
- runtime-resolved `end` index;
- `policy_end_prob`;
- end rank and end-vs-next margin;
- argmax action;
- common-RNG sampled action if reported;
- max absolute and relative hidden-state difference;
- max absolute and relative logit difference.

Report:
- `first_logit_divergence_step`;
- `first_argmax_action_divergence_step`;
- optional first common-RNG sampled-action divergence;
- relation of divergence to the first rollover event.

## P2 validity requirement

Before comparing A vs B, repeat each offline condition at least twice and require deterministic replay within declared numerical tolerance.

If the same condition is not replay-stable, return R4.

## Primary metric

[
\Delta z_t = \max_i |z^A_{t,i} - z^B_{t,i}|
]

Evaluate separately:
- pre-rollover-relevant region;
- rollover boundary;
- post-rollover region.

## Result classification

### R1 — Strong support for H-RESET

A valid initial-state difference:
`different Actor carry state -> different rollover timing -> same teacher-forced inputs produce reproducible logit divergence -> argmax/common-RNG action divergence`.

Allowed conclusion:
> Model-side episode-state rollover can cause prior episode history to change the Actor decision for an otherwise identical current input sequence.

Not allowed:
> It lowers SR.

Next cycle: small fixed-episode paired behavioral causal test with B0 retained as control.

### R2 — Distribution effect only

Reproducible hidden/logit difference, but no argmax/common-RNG action divergence in the tested sequence.

Conclusion:
> The state carrier changes the policy distribution, but behavioral impact is not yet shown.

H-RESET remains open but lower priority; next step is only a small paired behavior test if effect size justifies it.

### R3 — No meaningful difference

After genuinely crossing the rollover-relevant boundary, A and B remain equivalent within predeclared tolerance.

Conclusion:
> H-RESET is substantially downgraded as a performance-bottleneck explanation for the tested path.

Gate B remains an engineering semantics irregularity, but mainline returns immediately to:
- H-EXPLORATION;
- H-PERCEPTION;
- H-REPRESENTATION;
- H-READOUT;
- H-SAFEQUIT / safety-optimization attribution.

### R4 — Invalid treatment / invalid replay

Examples:
- clearing counter alone corrupts cache semantics;
- Actor-only reset is not isolatable;
- `RESET_STATE_FIX=0` differs from official model-level replay;
- logger adds an extra forward or changes inputs/RNG;
- fixed inputs cannot be paired;
- replay is not deterministic enough for interpretation.

Conclusion: experiment invalid. Fix the experiment; do not interpret H-RESET.

## Explicitly excluded

- formal layer-wise Probe;
- Oracle illegal-end gate;
- Stop Gate;
- 200-task evaluation;
- Safety metric redefinition;
- physical-size method;
- IL/RL/Safe-aligned three-checkpoint comparison;
- online clean-reset treatment used for performance claims.

## Resources

- max GPU: 1;
- max live ObjectNav episodes: 4;
- offline replay/model forwards: allowed only for P1/P2;
- no full benchmark.

## Required outputs

- `research/handoffs/reset-001a-20261007/RESULT_SUMMARY.md`
- `research/handoffs/reset-001a-20261007/RUN_MANIFEST.json`
- `research/handoffs/reset-001a-20261007/ARTIFACT_INDEX.json`
- `research/handoffs/reset-001a-20261007/REVIEW_NOTES.md`
- `research/handoffs/reset-001a-20261007/decoder_reset_audit.md`
- `research/handoffs/reset-001a-20261007/decoder_state_map.json`
- `research/handoffs/reset-001a-20261007/reset_fix_design.md`
- `research/handoffs/reset-001a-20261007/reset_fix.patch`
- `research/handoffs/reset-001a-20261007/aa_off_equivalence.json`
- `research/handoffs/reset-001a-20261007/rollover_stress_result.json`
- `research/handoffs/reset-001a-20261007/rollover_trace.csv`
- `research/handoffs/reset-001a-20261007/run_reset_001a.py`

After handoff publication, STOP. PI will judge only:
1. whether the treatment is truly single-variable;
2. whether model-side state carry actually changes Actor behavior.
