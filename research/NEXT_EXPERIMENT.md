# Next Experiment — Semantic decision mismatch localization

Experiment ID: EXP-SEMANTIC-DECISION-MISMATCH-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: semantic-decision-mismatch-001-20260922

## Why this experiment replaces the previous stage-only audit

The 02｜SafeVLA研究 mainline is not merely "which failures are before/after visibility." Its central question is whether low-SR ObjectNav failures come from:

1. target evidence never reaching the policy because exploration/search fails;
2. target evidence entering the network but becoming weak/lost in representation;
3. target/stop-relevant evidence remaining decodable, but the Actor readout failing to use it;
4. termination calibration producing two opposite morphologies: premature `end` and termination hesitation/long loops.

Historical evidence already constrains the design:
- the old three-layer exploratory Probe produced AUC ~0.955 / 0.982 / 0.992 for a close-and-visible/stop-like label, so the old "Layer 3 forgets target information" story is not supported;
- that Probe is not formal evidence because it used PT-Guard-modified trajectories, lacked episode/task IDs and used a leakage-prone step split;
- a historical Done Gate treatment changed failure morphology (less premature done, more hesitation/loops) without improving net SR, so hard stopping rules are not the next experiment;
- at least one cleanly audited failure has illegal `end` with `end_prob≈0.993`, while another recovered 600-step case had very low rendered `p(done)` throughout; a single "model likes end" mechanism cannot explain both;
- source topology shows the Actor distribution is produced from the actor branch; the cost critic does not directly gate/override `end` at inference.

Therefore the next experiment should directly discriminate **exploration vs representation vs Actor-use/readout vs termination calibration** on clean B0 trajectories.

## Research question

For historically low-SR ObjectNav tasks, where does the causal chain first fail under clean B0?

`goal/task -> exploration -> target enters nav camera -> fusion/L1/L2/L3 representation -> Actor logits -> end/move/turn action`

Specifically:
- when target evidence never appears, is the failure an exploration/search problem rather than a visual-representation problem?
- when target evidence appears, is it still decodable in the final Actor representation?
- if it is decodable, does the Actor end readout use that direction consistently, or is there a representation-use/readout mismatch?

## Hypotheses

H-EXPLORE:
A substantial subset of low-SR failures occurs before target evidence enters the nav camera. These episodes should be analyzed through room/search/action dynamics, not through target-representation Probe conclusions.

H-REP:
For post-visibility failures, target/stop-proxy information is weak or degrades from fusion/L1-L3, consistent with a representation bottleneck.

H-USE:
For post-visibility failures, target/stop-proxy information remains strongly decodable at L3, but the Actor end readout is weakly aligned/sensitive to that information or poorly separates stop-proxy-positive from negative states.

H-TERM:
Termination is miscalibrated in both directions: some episodes show high-confidence illegal/premature end before sufficient target evidence, while others show low end propensity / persistent movement-loop behavior after extended search.

## Competing explanation

H-DIFFICULTY:
Long-horizon tasks explain much of the apparent target-specific failure. `expert_length` is therefore used for task selection/matching as a pre-policy long-horizon proxy, but not treated as shortest-path distance or pure environment difficulty.

## Scientific target group

Historical low-SR diagnostic categories:
- mug.n.04
- basketball.n.02
- laptop.n.01
- bowl.n.03

These are called **low-SR target categories** in formal outputs. The researcher's qualitative observation that many are physically small may motivate the project, but this experiment does not measure or claim a physical-size causal effect.

## Diagnostic task selection — fixed before execution

Select six historical failure tasks from the aligned 200-run:

Pre-visibility historical failures:
1. mug — house 2986 / sub_house_id 6 / expert_length 84 / historical nav-visible=0
2. basketball — house 8786 / sub_house_id 15 / expert_length 141 / nav-visible=0
3. laptop — house 13975 / sub_house_id 117 / expert_length 209 / nav-visible=0

Post-visibility historical failures:
4. mug — house 1730 / sub_house_id 3 / expert_length 62 / nav-visible>0
5. bowl — house 13943 / sub_house_id 46 / expert_length 48 / nav-visible>0
6. laptop — house 13458 / sub_house_id 110 / expert_length 115 / nav-visible>0

For each failure task, add:
- one unique same-category historical-success control minimizing absolute expert_length difference;
- one unique cross-category historical-success control minimizing absolute expert_length difference.

If a same-category control has poor expert-length overlap, keep it but flag the gap; do not replace it with outcome-derived tuning. Cross-category control is the difficulty-matched complement.

The selection script must be deterministic and saved as `diagnostic_task_manifest.csv`.

## Execution design

Run the resulting 18 unique diagnostic tasks under clean B0 with exactly two predeclared seeds:
- seed 123
- seed 456

Maximum: 36 ObjectNav episodes.

These are **diagnostic case-control reruns**, not a benchmark SR estimate. Historical success/failure is used only for task selection; all mechanism labels must be recomputed from the fresh diagnostic runs.

## B0 constraints

- Official B0 source/checkpoint/evaluation semantics remain unchanged.
- No PT-Guard, no done gate, no steering, no reranking, no action rewrite.
- Official stochastic/augmentation/action-history behavior is preserved.
- Diagnostic task manifest changes only which episodes are selected; it does not define a new benchmark score.
- One normal policy decision per environment decision. No extra Actor forward.
- Hook only the root actor causal branch `actor_critic.decoder`; do not mix reward-critic or cost-critic hidden states into Actor representation claims.
- Diagnostic hooks/loggers must be read-only and must not mutate returned tensors, action distribution, memory, cache, RNG, observation or environment state.
- Record actor-forward counts and RNG fingerprints around logger serialization. Any evidence of extra forward or state/RNG mutation -> BLOCKED.
- Do not call `successful_if_done()` or `object_is_visible_in_camera()` as an extra pre-decision oracle during live policy execution.

## Passive per-step evidence

Record only values already produced by the normal policy/environment step or passive reads of the current event/state:

Policy:
- actor logits/probabilities for all actions;
- chosen/executed action;
- explicit `end` probability using the runtime-resolved action index, never a hard-coded index;
- fusion token / Actor L1 / L2 / L3 hidden state;
- Actor forward count;
- previous-action/timestep metadata required to reconstruct policy history.

Environment/observation:
- exact task/house/step identity;
- agent pose;
- current nav RGB reference/hash;
- exact target IDs;
- current-event target visibility / target pixels if already present in event metadata without an extra simulator action/query;
- target distance only if already present in current-event metadata/passive object metadata;
- official end success/failure only when `end` is actually executed and the task already evaluates it.

If exact current-frame target visibility cannot be obtained passively, preserve the trajectory and return that label as unavailable; do not add a new live oracle call.

## Fresh phenotype labels

For each fresh rerun:

P0 — PRE-VISIBILITY:
No exact target instance is ever observed in the nav camera during the fresh run.

P1 — POST-VISIBILITY:
Target is observed at least once.

Within P0:
- P0-EARLY-END: fresh run executes `end` before target encounter and official end result is failure.
- P0-HORIZON/SEARCH: no target encounter and no illegal early end; report horizon/search/loop morphology.

Within P1:
- P1-END-ERROR: target was observed and run later executes an unsuccessful `end`.
- P1-HORIZON: target was observed but run reaches horizon without success.
- P1-SUCCESS: successful control/reference trajectory.

Historical labels must not overwrite these fresh-run labels.

## Representation analysis

Primary clean Probe label:
- `target_visible_nav` from exact target-instance passive current-frame evidence.

Secondary stop proxy, only if passive distance is available:
- `stop_proxy = target_visible_nav AND min_target_distance <= 2m`.
- This is a diagnostic proxy, **not** claimed identical to official `successful_if_done()`.

Probe sites:
- fusion representation;
- Actor L1;
- Actor L2;
- Actor L3.

Protocol:
- freeze SafeVLA;
- linear probe only;
- split by episode/house, never by frame;
- report class prevalence, number of positive episodes, ROC-AUC, PR-AUC, balanced accuracy/F1 and bootstrap CI;
- if fewer than 6 positive episodes or fewer than 50 positive states are available, mark the corresponding Probe underpowered; do not manufacture a layer conclusion.

Interpretation:
- weak target-visible decoding already at fusion -> perception/fusion candidate;
- strong fusion but degraded L1->L3 -> temporal-representation candidate;
- strong L3 decoding -> evidence against simple "information absent at L3"; proceed to Actor-use test.

## Actor-use / readout test

This is the only causal manipulation in the experiment and it is **offline only**.

For a clean L3 probe direction `w_probe` learned on training episodes:
1. normalize `w_probe`;
2. for held-out recorded L3 states, construct `h' = h ± alpha * w_probe`;
3. pass `h'` through the frozen Actor linear head offline only;
4. measure change in `end` logit/probability;
5. compare with equal-norm random directions and shuffled-label probe directions.

Also report:
- cosine alignment between normalized `w_probe` and the Actor end-readout direction;
- rank/percentile of probe-direction sensitivity versus random controls.

No environment step is taken from the perturbed representation.

Interpretation:
- high L3 decodability + weak Actor sensitivity/alignment -> supports representation-use/readout mismatch;
- high L3 decodability + strong sensitivity but wrong actions -> redirects toward calibration/history/competing evidence rather than "Actor cannot read the information";
- low L3 decodability -> readout mismatch is not the first explanation.

## Exploration / termination analysis

For P0 and P1 trajectories, report:
- time/steps to first target encounter;
- room-entry/transition sequence if derivable from passive pose + static room polygons;
- action entropy over time;
- fraction of move/turn/end actions;
- end-probability trajectory before first target encounter;
- repeated-turn / short-cycle oscillation metrics;
- long no-progress windows based on pose displacement;
- final 20-step action/logit summary;
- premature-end and horizon/hesitation morphology.

This explicitly tests the 02 research observation that hard done suppression can trade premature end for hesitation/looping instead of fixing the underlying semantic decision problem.

## Task-difficulty control

Use `expert_length` only as a long-horizon matching covariate.

Do not call:
- `sub_house_id` a difficulty variable;
- `episode length`, room visitation or visibility a pre-existing task difficulty;
- house_index ordering a numeric difficulty scale.

Report same-category and cross-category control gaps in expert_length. If overlap is poor, state that the mechanistic comparison remains confounded.

## Metrics

Primary:
- fresh P0 vs P1 phenotype count by selected task/seed;
- premature illegal-end count and horizon/hesitation count;
- first-target-encounter rate;
- fusion/L1/L2/L3 held-out visibility Probe metrics;
- optional stop-proxy Probe metrics;
- Actor end-probability distributions before/after target encounter;
- Actor readout alignment/sensitivity versus random/shuffled controls;
- loop/oscillation metrics;
- same-category and cross-category expert_length control gaps.

Secondary:
- phenotype reproducibility across seeds;
- action entropy and action-rank trajectories;
- any correlation between existing cost-critic output and end probability may be logged only as descriptive evidence; it must not be interpreted as direct gating.

## Expected outcomes and decision rules

Outcome A — exploration/search dominant:
Most fresh failures remain P0, especially with low end probability/horizon or repeated loops.
Next experiment: goal-conditioned exploration / temporal-memory / room-search mechanism.

Outcome B — premature termination before evidence:
P0 failures frequently execute high-confidence illegal `end`.
Next experiment: end calibration/history cause under no-target-evidence states.

Outcome C — representation bottleneck:
P1 failures show weak/declining clean target-visible decoding by L3 relative to controls.
Next experiment: visual/fusion/temporal representation intervention.

Outcome D — representation-use mismatch:
P1 failures show strong L3 decodability but weak Actor sensitivity/alignment to the same information and erroneous end/move decisions.
Next experiment: minimal readout/calibration intervention.

Outcome E — Actor uses the information but decisions remain wrong:
Strong L3 Probe + strong readout sensitivity, but errors persist.
Next experiment: temporal-history/context competition, objective/calibration or training-induced policy bias; do not claim readout failure.

## Alternative explanations

- targeted reruns may not reproduce historical failure because official evaluation is stochastic;
- passive visibility evidence may differ from official success-oracle visibility semantics;
- expert_length may not remove scene/initial-distance difficulty;
- linear Probe can miss nonlinear information or exploit correlated features;
- probe direction may not be unique because representation is redundant;
- offline Actor-head perturbation tests head sensitivity, not full online behavioral causality;
- selected case-control tasks are diagnostic and do not estimate population SR.

## Stop conditions

Return BLOCKED if:
- clean B0 source/checkpoint identity cannot be preserved;
- diagnostic code changes policy action, history, RNG, cache, forward count or environment semantics;
- any PT-Guard/done gate/steering/action rewrite is activated;
- target visibility would require an extra live simulator/oracle query rather than passive evidence;
- selected tasks cannot be stably identified;
- any behavior intervention is attempted inside this experiment.

If Probe sample support is merely insufficient, complete the run and report that component as underpowered rather than silently changing selection or adding episodes.

## Required outputs

- `research/handoffs/semantic-decision-mismatch-001-20260922/RESULT_SUMMARY.md`
- `research/handoffs/semantic-decision-mismatch-001-20260922/RUN_MANIFEST.json`
- `research/handoffs/semantic-decision-mismatch-001-20260922/ARTIFACT_INDEX.json`
- `research/handoffs/semantic-decision-mismatch-001-20260922/REVIEW_NOTES.md`
- `research/handoffs/semantic-decision-mismatch-001-20260922/diagnostic_task_manifest.csv`
- `research/handoffs/semantic-decision-mismatch-001-20260922/episode_summary.csv`
- `research/handoffs/semantic-decision-mismatch-001-20260922/step_trace.parquet`
- `research/handoffs/semantic-decision-mismatch-001-20260922/probe_results.csv`
- `research/handoffs/semantic-decision-mismatch-001-20260922/readout_sensitivity.csv`
- `research/handoffs/semantic-decision-mismatch-001-20260922/failure_path_analysis.md`
- `research/handoffs/semantic-decision-mismatch-001-20260922/diagnostic_runner.py`
- `research/handoffs/semantic-decision-mismatch-001-20260922/offline_probe_readout.py`

After handoff publication, STOP. No automatic intervention experiment.
