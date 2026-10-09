# PI Cycle 3 — Direct early-termination paired behavior test

Experiment ID: EXP-RESET-EARLYEND-001B
Status: DRAFT
Authorization: NOT_AUTHORIZED — PI acknowledged B4; this cycle will not be retried.
Cycle ID: reset-earlyend-001b-20261009

## Why this experiment now

EXP-RESET-001A established R1: a valid Actor carry/rollover state changes hidden states, logits and action preference under identical teacher-forced current inputs. It did not establish online behavioral relevance or premature-end causation.

The researcher now prioritizes the most direct question:

> On historical premature failed-end tasks, does resetting the Actor state carrier actually change early termination behavior?

This experiment therefore targets historical early-stop cases directly instead of running a generic behavior sample.

## Research question

For historical ObjectNav sub-horizon failures whose final executed action is an unsuccessful `end`, does the Actor-only episode reset treatment alter:
1. whether the episode prematurely ends,
2. when it ends,
3. whether it succeeds,
when the task/environment and pre-episode carry package are paired?

## Hypothesis

H-RESET-EARLYEND:
For cases that are temporally exposed to the 500-step rollover before their historical failed `end`, clearing Actor counter+K/V at the target-episode boundary will causally alter the trajectory and reduce or delay failed early termination in at least a subset of cases.

## Competing explanations

H-NO-EARLYEND-EFFECT:
Reset rollover changes policy distributions in stress replay but does not materially affect premature-end outcomes on historical early-stop tasks.

H-OTHER:
Some historical early stops occur before any rollover could influence the episode; those cases must be explained by other mechanisms such as termination calibration, exploration, sampling, representation/readout, or safety-training effects.

## P0 — Historical exposure audit (0 GPU / 0 episode)

Start from the aligned historical full-200 run.

Eligibility for an "early-stop case":
- success == false;
- eps_len < task horizon 600;
- final executed environment action is confirmed as `end`.
Do not infer premature end from eps_len alone.

For every eligible case, reconstruct the **worker-local** model counter at target-episode start from the original 4-worker execution order/logs if possible.

Using the exact 001A runtime rule, compute:
`rollover_local_step = 500 - counter_start` when counter_start > 0, else 500.

Classify:
- EXPOSED: rollover_local_step < historical_end_step;
- BOUNDARY: rollover_local_step == historical_end_step;
- UNEXPOSED: historical_end_step < rollover_local_step;
- UNKNOWN: worker-local counter cannot be reconstructed.

Scientific implication:
- UNEXPOSED cases are negative controls: the rollover mechanism cannot explain that historical early end before its observed end step.
- UNKNOWN cases cannot support a historical-rollover causal claim and are excluded from the primary paired analysis.

If worker-local order/counter cannot be reconstructed for any case, return BLOCKED rather than inventing start counters.

## P1 — Paired target-episode construction

For each EXPOSED/BOUNDARY case, reconstruct a semantically valid Actor carry package at the historical `counter_start`:
- never assign a counter with incompatible empty/stale cache;
- use a deterministic offline warm-up sequence to populate the Actor K/V state to the required counter position;
- prior-episode cache content is allowed only as masked carry history; no target-task observation may be consumed during warm-up.

At the target episode boundary clone the same valid carry package into two conditions:

CONTROL / OFF:
- keep the official Actor carry state;
- original episode reset semantics.

TREATMENT / ON:
- start from the identical pre-boundary carry package;
- apply the 001A reviewed Actor-only reset package: root Actor counter=0 plus zero root Actor K/V;
- critics remain untouched.

Then initialize the exact same historical target task/environment state.

## Single variable

The only treatment variable at target-episode start is:
`Actor state carry retained` vs `Actor state carry cleared`.

Fixed:
- B0 weights/checkpoint/DINO;
- task specification, scene, initial agent state;
- success/end/horizon/reward/cost definitions;
- action mapping;
- augmentation protocol;
- simulator semantics;
- critic states;
- sampling protocol.

## Randomness control

Use one predeclared common-random-number paired run per eligible historical case for this screening experiment.

Within a pair:
- same environment seed;
- same augmentation RNG sequence;
- same categorical sampling random-number stream;
- no extra policy forward.

After the first action divergence, the two environments may naturally diverge; this is the intended online treatment effect. Continue each branch normally. Do not teacher-force after divergence in this online experiment.

Because this is one paired screen per case, do not make a population-rate claim from a single switched episode. Any promising outcome becomes a replication target for a later fixed-seed confirmation.

## Negative controls

Include up to four deterministic UNEXPOSED cases, prioritizing the shortest historical end steps.

Prediction from 001A:
before their historical end point, OFF and ON should remain behaviorally identical if the reset effect acts only through rollover timing.

If UNEXPOSED controls diverge before their calculated rollover, treat that as a treatment-validity warning and return INVALID/BLOCKED unless explained.

## Primary metrics

Per pair:
- historical_end_step;
- reconstructed counter_start;
- calculated rollover_local_step;
- first online action divergence step;
- OFF failed_end (yes/no);
- ON failed_end (yes/no);
- OFF/ON end_step;
- OFF/ON success;
- OFF/ON episode length;
- OFF/ON official Safety Cost (descriptive only, unchanged metric);
- OFF/ON target-encounter-before-end if passively available;
- `p(end)`, end rank and margin in a ±10-step window around rollover and around any executed end.

## Primary causal contrasts

1. **Early-end switch**
   - OFF = failed `end`, ON = no failed `end` at the paired comparison point / episode outcome differs.

2. **End timing**
   - `delta_end_step = end_step_ON - end_step_OFF` when both end unsuccessfully.

3. **Success switch**
   - OFF failure -> ON success, or the reverse.

4. **Trajectory divergence timing**
   - whether first action divergence occurs at/after the calculated rollover.

## Decision rules

### B1 — Direct behavioral relevance supported
At least two independent EXPOSED/BOUNDARY cases show reproducible-in-pair action divergence at/after rollover and a changed termination outcome (failed-end status, materially shifted end step, or success switch), while UNEXPOSED negative controls remain identical before their rollover.

Allowed conclusion:
> Actor reset/rollover has direct online behavioral relevance for at least a subset of historical premature-end tasks.

Not yet allowed:
> reset fix improves benchmark SR overall.

### B2 — Trajectory effect without termination effect
Exposed pairs diverge online after rollover, but failed-end outcome/timing remains essentially unchanged.

Conclusion:
> H-RESET affects navigation behavior but is not yet shown to explain premature termination.

### B3 — No early-end effect
Eligible exposed pairs remain behaviorally equivalent through the relevant end window, or treatment does not change end behavior.

Conclusion:
> H-RESET remains a real Actor mechanism from 001A but is downgraded as an explanation of historical premature end. Return mainline to termination/exploration/readout hypotheses.

### B4 — Invalid / unreconstructable
Historical worker-local counter state cannot be recovered; carry package cannot be built validly; pair randomization is not synchronized; UNEXPOSED controls diverge before rollover without explanation; or instrumentation changes policy semantics.

Do not interpret mechanism.

## Resource budget

- max GPU: 1;
- max live ObjectNav episodes: 40;
  - primary: at most 16 eligible early-stop cases × 2 conditions = 32;
  - negative controls: at most 4 cases × 2 conditions = 8;
- no 200-task benchmark;
- no Probe;
- no Stop/Oracle gate;
- no checkpoint comparison.

## Required outputs

- `research/handoffs/reset-earlyend-001b-20261009/RESULT_SUMMARY.md`
- `research/handoffs/reset-earlyend-001b-20261009/RUN_MANIFEST.json`
- `research/handoffs/reset-earlyend-001b-20261009/ARTIFACT_INDEX.json`
- `research/handoffs/reset-earlyend-001b-20261009/REVIEW_NOTES.md`
- `research/handoffs/reset-earlyend-001b-20261009/earlyend_case_manifest.csv`
- `research/handoffs/reset-earlyend-001b-20261009/counter_exposure_audit.csv`
- `research/handoffs/reset-earlyend-001b-20261009/paired_episode_results.csv`
- `research/handoffs/reset-earlyend-001b-20261009/paired_step_trace.parquet`
- `research/handoffs/reset-earlyend-001b-20261009/analysis.md`
- `research/handoffs/reset-earlyend-001b-20261009/run_reset_earlyend_001b.py`

After handoff publication, STOP. Any replication or broader benchmark test requires a new PI cycle.
