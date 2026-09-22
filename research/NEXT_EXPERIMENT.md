# Next Experiment — Low-SR failure-stage localization

Experiment ID: EXP-LOW-SR-STAGE-LOCALIZATION-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: low-sr-stage-localization-001-20260922

## Why this experiment now

The researcher explicitly deprioritized further physical-size measurement. The research objective is now to determine where the low success rate enters the causal chain as quickly as possible.

This change is accepted with one scientific boundary: the project may use "small-object-like / low-SR categories" as a practical phenomenon label, but physical size is no longer claimed as a proven causal variable. The current formal target group is defined from the historical 200-run performance table, not from a newly measured size threshold.

A PI re-audit also corrected an earlier mistake: `sub_house_id` is the shuffled dataset sample index, not a house-difficulty variable. Earlier statements that "sub_house_id<20 tasks are harder" are invalid. Real house identity is `house_index`.

Preliminary existing-data checks motivate a fast formal localization:
- selected low-SR categories: mug, basketball, laptop, bowl;
- combined historical SR: 34/48 = 70.8% versus 139/152 = 91.4% for other categories;
- among their 14 failures, 10/14 never become visible in the navigation camera and 9/14 never enter the target room;
- coarse expert-length strata still show lower SR for the selected group:
  - <=50: 24/26 = 92.3% vs 90/92 = 97.8%;
  - 51–100: 8/14 = 57.1% vs 39/43 = 90.7%;
  - >100: 2/8 = 25.0% vs 10/17 = 58.8%.
These are PI exploratory checks and must be reproduced by this experiment before being promoted.

## Research question

For the historically low-SR target categories (mug, basketball, laptop, bowl), after accounting descriptively for long-horizon task difficulty using the pre-policy `expert_length` proxy, where do failures first appear in the observable causal chain?

Causal chain for this audit:
task/goal -> exploration/room arrival -> target enters nav camera -> post-visibility approach/termination -> success.

## Hypotheses

H-EXPLORE:
The dominant low-SR failure mass occurs before target visual encounter: the agent often fails to reach the target room or never puts the target into the navigation camera.

H-DOWNSTREAM:
After accounting for task horizon, most low-SR failures still reach the target room / see the target, so the primary bottleneck is downstream perception/representation/approach/termination.

## Competing explanation

H-DIFFICULTY:
The low SR is substantially explained by longer/harder tasks rather than a target-specific mechanism. The selected categories should approach the other-category SR once compared within similar expert-length strata.

## Definition of "harder task" for this audit

Primary pre-policy difficulty proxy:
- `expert_length = task_info["expert_length"] = historical gt_episode_len`.

Important limitation:
- this is the recorded expert trajectory length, not a proven shortest path and not a pure environment-difficulty variable;
- it is used only as a **long-horizon difficulty proxy**.

Do NOT use the following to define task difficulty because they are policy outcomes or invalid identifiers:
- `sub_house_id` (sample index);
- episode length;
- target-room visitation;
- visible pixels;
- whether end was taken;
- success/failure.

Secondary descriptive scene factor:
- real `house_index`, reported only as a scene stratum / cluster indicator, not as a numeric difficulty score.
- Any claim that low house_index itself causes difficulty is prohibited.

## Study type

Zero-rollout observational audit of the existing aligned historical full-200 evidence.

No SafeVLA execution, no simulator, no GPU, no new episode.

## Target groups

Primary low-SR group, fixed before analysis:
- `mug.n.04`
- `basketball.n.02`
- `laptop.n.01`
- `bowl.n.03`

Reference group:
- all other ObjectNav categories in the same aligned 200-task run.

The group is called **low-SR target group** in formal outputs. Do not label the statistical group as "small objects" unless explicitly marked as the researcher's qualitative description.

## Difficulty analysis

1. Reproduce overall n/SR for target and reference groups.
2. Reproduce expert_length distribution for success and failure.
3. Pre-declared descriptive bins:
   - <=50
   - 51–100
   - >100
4. Report n, success, SR and Wilson 95% CI for target/reference groups within each bin.
5. Fit one parsimonious sensitivity model if numerically identifiable:
   `success ~ low_sr_group + log1p(expert_length)`
   Report coefficient/odds ratio + uncertainty; do not use it as causal proof.
6. Report house_index distribution and the previously observed low-house-index cluster, but do not treat house_index ordering as a difficulty scale.

## Failure-stage taxonomy

Apply only to historical failures, using already-aligned narrow target-ID evidence:

S0 — room-arrival failure:
- `has_agent_been_in_room == False`.
- Interpretation: failure occurred before confirmed target-room arrival; downstream visual recognition cannot be the sole explanation for this episode.

S1 — local-search / camera-encounter failure:
- target room was entered, but narrow target `vis_pix_navigation == 0`.
- Interpretation: agent reached the relevant room but never put the target into the nav camera.

S2 — post-visibility failure:
- narrow target `vis_pix_navigation > 0`.
- Split descriptively:
  - S2a: episode length < 600 (termination-like morphology; exact final end action is not assumed unless separately evidenced);
  - S2b: episode length == 600 (horizon morphology).

Do not infer a neural mechanism directly from S0/S1/S2.

## Controls

- Report the same S0/S1/S2 failure-stage distribution for the reference-category failures.
- For each low-SR failure, identify up to 3 nearest successful controls from the same category by absolute expert_length difference.
- Mark controls with poor overlap rather than forcing a match.
- Also provide cross-category expert-length-nearest controls if same-category overlap is absent; keep the two control types separate.

## Metrics

- target-group and reference-group n/SR/Wilson CI;
- expert_length mean/median by group × outcome;
- SR/Wilson CI within the three fixed expert-length bins;
- optional parsimonious logistic sensitivity estimate;
- low-SR failure counts and proportions in S0/S1/S2a/S2b;
- reference failure counts and proportions in S0/S1/S2a/S2b;
- same-category matched-control expert-length gaps;
- count of failures with no reasonable same-category difficulty overlap;
- real house_index distribution by group/outcome.

## Expected result

If H-EXPLORE is strengthened:
- S0+S1 is the majority of low-SR failures, and
- the low-SR group remains meaningfully below the reference group within at least the medium/long expert-length strata.

This would justify the next causal experiment focusing on goal-conditioned exploration / room search / temporal policy behavior before target visibility.

If H-DOWNSTREAM is strengthened:
- most low-SR failures are S2, especially under matched difficulty.
Then the next causal experiment should focus on visual representation, Actor readout and termination after target encounter.

If H-DIFFICULTY is strengthened:
- within expert-length strata, the low-SR gap largely disappears.
Then mechanism work must focus on long-horizon navigation difficulty rather than target-specific perception.

## Stop conditions

Return BLOCKED if:
- the aligned 200-task raw table or narrow failure evidence cannot be reconciled;
- `expert_length` identity cannot be tied to the same historical tasks;
- failure-stage labels would require inventing missing visibility/room fields;
- analysis would require new rollout, simulator, model inference or size measurement;
- any internal Probe or intervention would be started automatically.

## Required outputs

- `research/handoffs/low-sr-stage-localization-001-20260922/RESULT_SUMMARY.md`
- `research/handoffs/low-sr-stage-localization-001-20260922/RUN_MANIFEST.json`
- `research/handoffs/low-sr-stage-localization-001-20260922/ARTIFACT_INDEX.json`
- `research/handoffs/low-sr-stage-localization-001-20260922/REVIEW_NOTES.md`
- `research/handoffs/low-sr-stage-localization-001-20260922/difficulty_strata.csv`
- `research/handoffs/low-sr-stage-localization-001-20260922/failure_stage_table.csv`
- `research/handoffs/low-sr-stage-localization-001-20260922/matched_controls.csv`
- `research/handoffs/low-sr-stage-localization-001-20260922/stage_analysis.md`
- `research/handoffs/low-sr-stage-localization-001-20260922/analyze_low_sr_stage.py`

After handoff, STOP. Internal model diagnostics are a separate experiment selected from the stage-localization result.
