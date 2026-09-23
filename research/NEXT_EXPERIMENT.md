# Next Experiment — Reset / rollover causal gate

Experiment ID: EXP-RESET-ROLLOVER-CAUSAL-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: reset-rollover-causal-001-20260923

## Why this experiment is the correct next step after rereading 02｜SafeVLA研究

The 02 branch contains an explicit gate that the previous draft skipped:

- Gate A passed for checkpoint/runtime/action/logging identity.
- Gate C passed for end/success label timing in the audited scope.
- **Gate B failed** because official `InferenceAgentVIDA.reset()` does not clear Actor/Reward-Critic/Cost-Critic `time_step_counter` or K/V cache.
- A short controlled test showed old cache is masked at a fresh episode start and dirty-old-cache vs zero-old-cache output matched in that short regime.
- However model `max_steps=500` while ObjectNav horizon is 600, and the counter accumulates across episodes. Therefore rollover timing can depend on prior episode lengths and may alter the effective temporal history later in an episode.

02 explicitly stopped formal hidden-state/Probe interpretation until this reset pathway was causally bounded.

Therefore the previously staged `EXP-SEMANTIC-DECISION-MISMATCH-001` is superseded before approval. We first close Gate B with a minimal causal replay; only then may a clean Probe/readout experiment be authorized.

## Research question

Holding the exact Actor-decoder input sequence fixed, does the official cross-episode retention of counter/cache causally change Actor hidden states, logits or action preference when cumulative `time_step_counter` crosses the `max_steps=500` rollover boundary?

## Hypothesis

H-RESET-CAUSAL:
For an identical captured sequence of Actor-decoder inputs and episode masks, the official carried-state reset semantics and a clean-reset counterfactual are identical before the rollover-relevant region but diverge at or after the point where the carried counter reaches 500, changing hidden states and/or Actor logits.

## Null / competing explanation

H-RESET-MASKED:
The official mask/cache logic makes the retained counter/cache behavior behaviorally inert for the tested real B0 input traces; replay outputs remain numerically identical through and beyond the rollover-relevant region. Observed trajectory variability is then better explained by augmentation/GPU stochasticity or other hypotheses.

## Unique variable

Offline replay reset semantics only:

Reference — OFFICIAL-CARRY:
- reproduce official episode-boundary reset behavior: episode time_step restarts but decoder counter/cache are retained exactly as implemented.

Treatment — CLEAN-RESET:
- at each recorded episode boundary, clear the decoder `time_step_counter` and K/V cache before replaying the next episode.

All replay input tensors, masks, model weights, Actor head, episode boundaries and ordering are identical.

This treatment is **offline only**. No treatment action is executed in the environment.

## Phase 1 — read-only B0 capture

Purpose: obtain real Actor-decoder input sequences long enough to cross cumulative 500 decisions.

Run two independent capture traces:
- trace A: seed 123
- trace B: seed 456

For each trace:
- fresh process;
- official clean B0 source/checkpoint and official evaluation semantics;
- workers=1 for deterministic provenance;
- preserve official stochastic action sampling and test augmentation;
- capture consecutive episodes in official task order until cumulative top-level decisions >= 540;
- maximum 16 episodes per trace; if 540 decisions are not reached within 16 episodes, return BLOCKED for that trace rather than changing task selection.

Maximum live budget: 32 ObjectNav episodes total.

Capture only:
- exact task/episode/step boundary;
- pre-Actor-decoder input tensor actually supplied to the root Actor decoder;
- temporal mask / time_step metadata required to replay;
- Actor decoder output, Actor logits and executed action for reference integrity;
- internal counter/cache shape/position metadata;
- RNG fingerprints before/after logger serialization;
- actor-forward count.

No extra policy forward, no action rewrite, no simulator oracle query.

Large tensors may remain server-side and must be registered by path, SHA256 and size in ARTIFACT_INDEX.

## Capture validity gates

For every live decision:
- exactly one top-level Actor decision;
- logger does not change Python/NumPy/Torch CPU/CUDA RNG fingerprints;
- selected == executed == environment-received action;
- no PT-Guard, Done Gate, steering, reranking or legacy treatment active;
- B0 source/checkpoint/resource identity matches the accepted clean baseline;
- no extra decoder/Actor forward is introduced by capture.

Any failure -> BLOCKED before offline interpretation.

## Phase 2 — deterministic offline counterfactual replay

For each captured trace, replay the exact saved Actor-decoder input sequence twice in eval mode:

A. OFFICIAL-CARRY
- replay episode boundaries using the official reset semantics.

B. CLEAN-RESET
- identical replay, except counter + K/V cache are cleared at every episode boundary.

Replay must not use simulator or environment actions.

Before using the replay for conclusions, require the OFFICIAL-CARRY replay to reproduce the captured live Actor decoder output/logits within a predeclared numerical tolerance. If it cannot, return BLOCKED; do not compare treatment.

## Metrics

Per trace:
- cumulative decision index and episode-local step;
- official counter value before/after each decision;
- rollover event index;
- max absolute / relative difference between OFFICIAL-CARRY and CLEAN-RESET Actor hidden state;
- max absolute / relative Actor-logit difference;
- `end` logit/probability difference;
- argmax-action agreement/disagreement;
- first divergence step;
- distance in decisions from the first rollover event to first divergence;
- number and fraction of post-rollover states with action-rank changes.

Replay-integrity:
- OFFICIAL-CARRY replay vs captured live hidden/logits max difference;
- repeated offline replay exactness.

## Decision rule

Gate B2 PASS FOR MECHANISM DIAGNOSTICS:
- OFFICIAL-CARRY faithfully reproduces captured live outputs; and
- CLEAN-RESET vs OFFICIAL-CARRY shows no meaningful hidden/logit/action divergence through at least 40 real captured decisions after the first rollover in both traces.

Interpretation: the retained reset state is still an implementation irregularity, but no causal effect is observed in the tested real traces. Clean hidden-state/Probe diagnostics may proceed with H-RESET retained as a limitation.

Gate B2 CAUSAL EFFECT CONFIRMED:
- replay integrity passes; and
- CLEAN-RESET vs OFFICIAL-CARRY diverges reproducibly at/after rollover in hidden states or Actor logits/actions.

Interpretation: H-RESET is behaviorally active. The next experiment must quantify its effect on actual navigation behavior with B0 retained as control before interpreting Probe/readout failure.

BLOCKED:
- live capture is invasive;
- 540 decisions cannot be collected within the fixed episode budget;
- official replay cannot reproduce captured live Actor outputs;
- reset/counter state cannot be replayed unambiguously.

## Expected result

Based on the earlier short mask-isolation test, the pre-rollover region is expected to match. The critical unknown is whether divergence appears when the carried counter reaches the rollover boundary.

Either outcome is useful:
- no divergence closes the major 02 Gate-B blocker enough to proceed to clean semantic-decision diagnostics;
- reproducible divergence identifies a concrete temporal-state mechanism that must be resolved before Probe claims.

## Alternative explanations

- replay may omit hidden implementation state required for exact live reproduction;
- CUDA numerical nondeterminism may exceed tolerance even with fixed saved inputs;
- one or both real traces may not exercise a harmful rollover configuration;
- retained Reward/Cost critic state may differ while Actor state does not; because critics do not directly gate Actor inference, Actor conclusions must be based on the Actor branch;
- clearing counter and cache together identifies the reset package as causal but does not separately attribute counter versus cache. If causal, a later ablation can separate them.

## Explicitly not part of this experiment

- no small-object size validation;
- no formal layer-wise Probe training;
- no readout steering/intervention;
- no Done Gate / Stop Gate;
- no Safe-vs-IL comparison;
- no benchmark SR claim from the capture episodes;
- no modification of official B0 behavior during live capture.

## Required outputs

- `research/handoffs/reset-rollover-causal-001-20260923/RESULT_SUMMARY.md`
- `research/handoffs/reset-rollover-causal-001-20260923/RUN_MANIFEST.json`
- `research/handoffs/reset-rollover-causal-001-20260923/ARTIFACT_INDEX.json`
- `research/handoffs/reset-rollover-causal-001-20260923/REVIEW_NOTES.md`
- `research/handoffs/reset-rollover-causal-001-20260923/capture_manifest.json`
- `research/handoffs/reset-rollover-causal-001-20260923/replay_metrics.csv`
- `research/handoffs/reset-rollover-causal-001-20260923/gate_b_decision.md`
- `research/handoffs/reset-rollover-causal-001-20260923/capture_actor_inputs.py`
- `research/handoffs/reset-rollover-causal-001-20260923/replay_reset_counterfactual.py`

After handoff publication, STOP. PI selects the next experiment based on Gate B2 outcome.
