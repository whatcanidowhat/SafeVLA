# Next Experiment — Premature termination dynamics audit

Experiment ID: EXP-PREMATURE-END-DYNAMICS-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: premature-end-dynamics-001-20260923

## Why this experiment is now the unique next step

The current GitHub draft `EXP-RESET-ROLLOVER-CAUSAL-001` remains scientifically relevant, but its Gate-B role is narrower than previously treated: it blocks formal hidden-state / Probe / readout interpretation when cross-episode decoder state is uncontrolled. It does **not** block a zero-rollout audit of already-recorded actions and rendered Actor probabilities from the historical full-200 run.

Newly reviewed evidence makes premature termination the highest-value uncertainty:
- the aligned historical full-200 run has 27 failures, of which 16 are sub-horizon (`eps_len<600`);
- one audited historical state has `stop_legal=false`, Actor `p(end)≈0.993365`, greedy `end`, and final failure;
- one opposite 600-step failure (sub120) has video-rendered `p(done)<~0.018` throughout, so termination failures are heterogeneous;
- official ObjectNav RL reward configuration uses `step_penalty=0`, `failed_stop_reward=0`, `reached_horizon_reward=0`, `goal_success_reward=10`; therefore a termination loophole is mechanistically plausible but **not proven**;
- the SafeVLA paper reports cautious / curtailed behavior under extreme failure, but this does not prove that safety alignment causes early termination in the historical B0 run.

The immediate question is therefore not yet “does SafeRL cause early exit?” but first: **what termination dynamics actually precede the historical sub-horizon failures?**

## Research question

Among the aligned historical full-200 ObjectNav failures that terminate before horizon, what Actor-end dynamics precede the executed termination?

Do these failures primarily show:
1. an abnormally high termination prior from the first few decisions;
2. a later rise in end probability after unsuccessful search;
3. a low-probability stochastic end event;
4. or no single dominant pattern?

## Competing hypotheses

### H-PRIOR — abnormal termination prior
A substantial subset of invalid-end failures has high rendered `p(end)` from the first few decisions, before meaningful exploration could occur.

### H-SEARCH-TRIGGERED — search-triggered conservative termination
Rendered `p(end)` starts low and rises later in the episode before the invalid end. This is compatible with, but does not prove, a learned conservative / risk-avoidance policy.

### H-STOCHASTIC — low-probability stochastic termination
Rendered `p(end)` remains low at the executed end, indicating that stochastic action sampling rather than a strong termination preference explains the final action.

### H-MIXED
The 16 historical sub-horizon failures contain multiple mechanisms without a dominant termination morphology.

## Study type

Zero-rollout observational audit of preserved historical artifacts only.

No simulator launch, no SafeVLA model load, no new episode, no extra Actor forward, no hidden-state Probe, no reset treatment.

## Primary population

Start from **all 16 historical failures with `eps_len<600`** in the aligned 173/200 run.

Do not select by category, object size, house index, or desired mechanism.

For every candidate:
1. verify the exact task identity against the aligned 200-row evidence;
2. verify video availability and frame count;
3. recover the final executed action;
4. only call it a **confirmed invalid-end failure** if the final executed action is `end/done`.

If a sub-horizon failure does not end in `end/done`, report it separately as an abnormal-termination case and do not force it into the premature-end mechanism classes.

## Probability recovery

Generalize the already validated video-recovery method used for sub120.

For each reliable frame recover:
- executed action;
- video-rendered quantized `p(end)`;
- optionally the other displayed action probabilities when reliable.

All recovered probabilities must be labeled:
**video-rendered quantized probability approximation**.

A zero-width bar means only that probability is below the rendering resolution (approximately 1/55 ≈ 0.018), not that the true probability equals zero.

## Matched success controls

For each confirmed invalid-end failure, select one historical successful episode:
- same target category when available;
- minimum absolute `expert_length` difference;
- deterministic tie break by stable sample ID.

These are **expert-length-matched descriptive controls**, not fully difficulty-matched controls.

If no same-category success exists, use one cross-category expert-length-nearest success and flag it as fallback.

For each control, inspect the same early time window up to `min(failure_end_step, control_length)`.

## Target-evidence boundary

Carry forward the narrow/broad target-ID audit:
- `STRICT`: narrow target IDs == broad success-eligible target IDs;
- `AMBIGUOUS`: narrow != broad.

Historical `vis_pix_navigation==0` may be used only as a recorded narrow-target visibility field.
For AMBIGUOUS cases, do not claim that every success-eligible target was unseen.

This experiment is about termination dynamics; target evidence is secondary descriptive context.

## Primary per-case metrics

- `end_step`;
- final executed action;
- `p_end_step0`;
- `max_p_end_first5`;
- `max_p_end_before_end`;
- `p_end_at_end`;
- `delta_p_end = p_end_at_end - p_end_step0`;
- first step with `p(end)>=0.5`;
- first step with `p(end)>=0.9`;
- matched-control `p(end)` at the failure end step/window;
- target-scope status;
- historical narrow `vis_pix_navigation` and room-visit fields as descriptive context only.

## Operational morphology labels

Thresholds are deliberately far above the ~0.018 rendering resolution.

- **EARLY_HIGH_PRIOR**:
  `max_p_end_first5 >= 0.5`.

- **LATE_RISE**:
  `max_p_end_first5 < 0.1`, episode length > 5, and `p_end_at_end >= 0.5`.

- **LOW_PROB_END**:
  executed final action is `end` and `p_end_at_end < 0.1`.

- **OTHER_OR_UNCLEAR**:
  reliable trace that does not satisfy the above.

These are observed morphologies, not neural or training-mechanism labels.

## Decision rule

Report counts/proportions and matched-control contrasts.

The next causal experiment is selected as follows:

- EARLY_HIGH_PRIOR is prominent and matched successes do not show the same early end probabilities:
  next investigate task-start goal/fusion/Actor termination bias, with H-RESET explicitly controlled before hidden-state interpretation.

- LATE_RISE is prominent:
  next run a policy comparison on matched hard tasks (SafeVLA vs the closest available non-safety-aligned/base policy, ideally FLaRe if the matching checkpoint is available) to test whether safety alignment contributes to conservative termination.

- LOW_PROB_END is prominent:
  next investigate stochastic sampling / termination calibration rather than representation loss.

- no clear morphology:
  do not force a termination-root-cause story; return to broader stage-localization / exploration analysis.

No result in this experiment can by itself prove that SafeRL, the cost critic, a specific training scene, or deliberate author design caused the behavior.

## Controls / fixed conditions

- historical aligned 2026-08-03 full-200 artifacts only;
- exact stable task identities from the aligned evidence packet;
- no category preselection;
- no policy rerun;
- no simulator;
- no model/checkpoint load;
- no extra forward;
- no hidden-state Probe;
- no replay;
- no reset/counter intervention;
- no action rewriting;
- no causal claim from `cost=0`;
- `expert_length` is only a long-horizon matching proxy;
- `sub_house_id` is the original dataset sample index, not house identity or difficulty.

## Alternative explanations

- rendered probability bars are quantized and cannot resolve probabilities below ~0.018;
- stochastic sampling can execute a non-argmax action;
- instruction wording, house geometry, initial target distance, and task horizon remain confounds;
- narrow/broad target-ID mismatch can contaminate “no target evidence” interpretation;
- `expert_length` is an incomplete difficulty proxy;
- cross-episode reset/counter state remains an open implementation confound for later hidden-state interpretation;
- a termination morphology does not identify whether it came from IL, RL, SafeRL, reward design, data distribution, or another source.

## Stop conditions

Return BLOCKED for population-level interpretation if:
- fewer than 12 of the 16 candidate failure videos yield reliable frame/action/probability recovery;
- action-schema / row-to-action mapping cannot be frozen;
- frame-to-decision alignment is inconsistent and cannot be reconciled;
- final-action recovery is ambiguous;
- the audit would require a new simulator run, model inference, hidden-state Probe, or behavior intervention.

Matched-control conclusions are separately marked unavailable if success-control videos cannot be recovered reliably; this does not invalidate a reliable failure-only morphology audit.

## Required outputs

- `research/handoffs/premature-end-dynamics-001-20260923/RESULT_SUMMARY.md`
- `research/handoffs/premature-end-dynamics-001-20260923/RUN_MANIFEST.json`
- `research/handoffs/premature-end-dynamics-001-20260923/ARTIFACT_INDEX.json`
- `research/handoffs/premature-end-dynamics-001-20260923/REVIEW_NOTES.md`
- `research/handoffs/premature-end-dynamics-001-20260923/premature_end_cases.csv`
- `research/handoffs/premature-end-dynamics-001-20260923/matched_success_controls.csv`
- `research/handoffs/premature-end-dynamics-001-20260923/pdone_trace_summary.csv`
- `research/handoffs/premature-end-dynamics-001-20260923/premature_end_analysis.md`
- `research/handoffs/premature-end-dynamics-001-20260923/analyze_premature_end.py`

After handoff publication, STOP. Do not automatically start a SafeVLA-vs-baseline run or hidden-state Probe.
