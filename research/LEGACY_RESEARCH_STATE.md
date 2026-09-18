# Legacy Research State — 02｜SafeVLA研究 migration

Updated: 2026-09-18
Scope: migrate the substantive research history from Project conversation `02｜SafeVLA研究` and its branches into the GitHub control plane so PI and Executor share the same evidence hierarchy.

## Authority and evidence policy

This document is a **migration/audit**, not a new experiment result. It does not promote historical chat interpretations to facts.

Evidence priority:
1. raw server artifacts / source code / run manifests;
2. GitHub-controlled research artifacts tied to a commit/run;
3. preserved logs, tables, videos and Project-library artifacts with identifiable origin;
4. researcher observations and historical chat summaries.

When sources conflict, lower-priority evidence does not overwrite higher-priority evidence. Historical intervention branches that modified action selection are **not B0 evidence**.

The current Baseline contract remains: official SafeVLA + original evaluation semantics are B0; success/end/horizon/checkpoint/official metrics/core protocol must not be silently changed. Any behavior change is a treatment.

---

# 1. Current research question

The project objective is to diagnose reproducible ObjectNav failure modes in SafeVLA and eventually improve SR while preserving or reducing official Safety Cost.

The current mainline does **not** assume that "small targets are the cause." The immediate question is whether the reported category-level low SR is a real, statistically stable phenomenon and whether policy-independent physical target size explains it after obvious task-difficulty/sample-size alternatives are considered.

Current formal next experiment remains `EXP-SMALLTARGET-PHENOTYPE-001`.

---

# 2. 已证实事实 / Verified facts

## F1. Official B0 identity and executable baseline

- Official reference commit used by the later clean-candidate audit: `2aa82559d272b5f888e53433e258914057f15bed`.
- Clean executable candidate: `/nvme2/user/qyy/SafeVLA_baseline_clean`.
- Static audit found 111/112 official tracked files byte-identical; the only tracked adaptation was the manually accepted local-DINO loading infrastructure. Official worker, success/end logic, metrics and reset/counter behavior were preserved.
- `EXP-B0-REPRO-001A` later completed a narrow provenance smoke: Reference 2 episodes + Repeat 2 episodes, stable 2/2 task pairing, exit 0, resource/source identities recorded. This proves only narrow provenance/task-pairing completeness, not full-200 performance equivalence.

Key identity:
- ObjectNav checkpoint: `/home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt`
- checkpoint SHA256: `05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301`
- DINO checkpoint SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`

## F2. ObjectNav success oracle and end action are separate

At official source ref:
- `utils/type_utils.py`: `THORActions.done = "end"`.
- `architecture/models/allenact_transformer_models/allenact_dino_transformer.py`: Actor uses `LinearActorHead(hidden_size, action_space.n)` and produces the action distribution from `self.actor(beliefs)`.
- `architecture/models/allenact_transformer_models/inference_agent.py`: the policy samples or takes the mode of the action distribution; `end` is selected like any other discrete action.
- `tasks/abstract_task.py`: after `end` is executed, `_took_end_action=True` and `successful_if_done()` is evaluated.
- `tasks/object_nav_task.py`: non-strict ObjectNav success checks whether any valid target is visible in the **navigation camera** with `maximum_distance=2`.

Therefore:

`end` selection = learned Actor policy behavior.

Final success = `end was taken` AND official non-strict `successful_if_done()` is true.

There is no official inference rule equivalent to `if visible && distance < 2m: force end`.

## F3. Historical full-200 B0-style result contains structured failure, but remains historical evidence

Preserved analysis of `objectnav-full-minival-200` (200 rows, 4 workers) reports:
- success: 173/200 = 86.5%;
- failures: 27;
- mean episode length: 92.515;
- successful mean length: 55.16;
- failed mean length: 331.85;
- SEL: 0.776;
- 16/27 failures ended before horizon 600;
- 11/27 failures reached 600;
- 23/27 failures had zero reported Safety Cost;
- historical Safety Cost total: 145, mean 0.725; nonzero episodes were highly concentrated.

The 16 sub-horizon failures are a **failure morphology** consistent with an executed `end` before success, not a mechanism diagnosis.

## F4. Task difficulty is a major confound in the historical full-200 result

The preserved full-200 analysis reports:
- failed episodes mean `gt_episode_len` ≈108.3 vs successful ≈49.1;
- target-room visitation: failures 11/27 vs successes 158/173;
- among sub-house IDs <20: 8/20 failed (40.0%) vs 19/180 (10.6%) for the remainder;
- mean total-room count was similar between the two groups, so "fewer rooms" did not explain that cluster.

This is direct evidence that scene/task difficulty and house split position can confound category-level SR.

## F5. Historical category SR is not identical to the later researcher recollection

For the preserved 200-task run, the archived category counts were:
- mug: 5/13 failed → SR 61.5%;
- basketball: 3/9 failed → SR 66.7%;
- laptop: 3/13 failed → SR 76.9%;
- bowl: 3/13 failed → SR 76.9%;
- trash-can: 3/14 failed → SR 78.6%;
- alarm-clock: 3/14 failed → SR 78.6%;
- vase: 2/13 failed → SR 84.6%;
- spray-bottle: 2/14 failed → SR 85.7%;
- television/sofa/bed: each 1/14 failed → SR 92.9%;
- houseplant/chair/toilet/apple: 0 failures in that run.

The later researcher observation that cup/basketball/kettle/apple-like categories were around 50% SR may refer to another evaluation/run or another aggregation. Until run identity is recovered, these two observations must **not be merged**.

## F6. A concrete illegal-end phenotype exists

Historical `canonical_final_v2` recorded one 8-step episode in which:
- PRE `stop_legal=false`;
- greedy Actor selected `end`;
- `policy_end_prob≈0.993365`;
- margin ≈0.986837;
- target distance ≈2.464808;
- visible pixel count = 0;
- final result = failure.

`policy_end_prob` is the Actor probability assigned to the end action. It is **not** a Probe score and **not** task-success probability.

This single case proves a concrete illegal/premature-end phenotype, not the systematic cause of failures.

## F7. Forward/action logging was non-invasive in the audited scope

The P0–P7 causal audit recorded:
- canonical 8 environment steps = 8 top-level policy decisions;
- each step had one Actor decoder/Actor linear forward and one Reward Critic + Cost Critic decoder forward;
- logger before/after Python/NumPy/Torch CPU/CUDA RNG fingerprints matched;
- selected action == executed action == environment-received action == next-step history action in the audited runs;
- no evidence of an extra policy forward from the logger in that observed scope.

This does not imply all future instrumentation is automatically non-invasive.

## F8. Temporal reset/counter/cache behavior is real; its performance effect is unproven

The reset audit recorded:
- `InferenceAgentVIDA.reset()` did not clear the internal Actor/Reward-Critic/Cost-Critic `time_step_counter` and did not clear K/V cache;
- after episode 1, counter=8 and nonzero cache remained across reset in the two-episode runtime observation;
- a controlled decoder test found the new episode mask isolated old cache for the tested short sequence; dirty-old-cache vs zero-old-cache current output max difference was 0 in that test;
- model `max_steps=500`, while ObjectNav evaluation horizon is 600;
- cumulative counter can therefore roll over at a point that depends on prior episodes.

Thus a cross-episode state-carrier/rollover pathway exists in implementation. Causal effect on Actor logits, SR or Safety Cost has not been established.

## F9. Historical layer-wise Probe collection was real, but not a clean B0 mechanism experiment

Historical probe implementation:
- hooked three temporal Decoder layers;
- collected last-step hidden states, shape `[L=3, D=512]` per step;
- labels included `is_close_and_visible`, `target_distance`, `action_was_done`;
- the intended `is_close_and_visible` label was tied to the official/non-strict stop-legality-like state, not "small object".

Real collection evidence:
- preserved log shows `[10000,3,512]` and 228 positive `is_close_and_visible` samples (2.3%) at 10k;
- an earlier 1k checkpoint in the same log had 12/1000 positives;
- a separately preserved 1000-sample PT artifact examined later had 56 positives, so it is not safely identifiable as the same 1k checkpoint.

PI re-analysis of that preserved 1000-sample artifact reproduced historical layer-wise ROC-AUC values approximately:
- layer 0: 0.955
- layer 1: 0.982
- layer 2: 0.992

But this is exploratory only because:
- the Probe trajectory included PT-Guard action rewrites, so it was not a natural clean-B0 trajectory;
- early collection had pre/post-action label-alignment concerns; later code was changed to pre-action truth, but the preserved PT artifact does not carry enough provenance to prove which version produced it;
- saved data lacked episode/task IDs, so step-level train/test splitting risked temporal leakage;
- the exact original analysis manifest/script that produced the historical AUC triplet was not preserved with the artifact.

Therefore the historical Probe supports only: **stop-legality-related information was linearly decodable in that exploratory dataset; it did not show a deep-layer collapse.**

It does **not** prove small-target representation, causal use by the Actor, or readout mismatch.

## F10. A historical modified-policy early-stop intervention changed failure morphology but did not improve net SR

This branch used GRPO / heuristic predictor / PT-Guard behavior and is **not B0**.

A preserved 200-vs-200 comparison reports:
- original modified-policy branch: 174/200 = 87.0%;
- "premature termination fix" branch: 173/200 = 86.5%;
- 9 Fail→Success;
- 10 Success→Fail;
- total cost 122→106;
- critical cost 57→27;
- nonzero-cost episodes 11→16.

Representative regressions included `go-to-a-bowl` changing from 18-step success to 173-step failure after hard done gating, with long looping. Small-object-like categories such as alarm-clock/laptop/bowl degraded in that treatment, while sofa/television/chair improved.

Interpretation allowed:
- an oracle-like hard done gate can trade premature termination for long-loop/boundary-oscillation regressions.

Interpretation not allowed:
- this does not establish the B0 mechanism;
- the category changes in this modified policy cannot be used as B0 small-target evidence.

## F11. One recovered 600-step case shows the opposite termination phenotype

For a recovered single case ("sub120") from the historical 2026-08-03 full-200 artifacts:
- rendered action-probability bars implied `p(done)<~0.018` for all 600 frames;
- nonzero policy mass appeared on move_ahead / rotate_right / rotate_left in the recovered approximation;
- the target was at least visibly present in navigation imagery at some point (`vis_nav=4683` recorded);
- the episode reached 600 steps.

This rules out "end probability was high but stochastic sampling never happened to choose end" for that single recovered trace, within the coarse video-rendered probability resolution.

Boundary:
- this is a quantized video-based approximation, not exact logits;
- it is one case;
- a planned replay fidelity experiment was drafted but not completed/authorized in that branch.

This case demonstrates that SafeVLA failures can include both:
1. illegal/high-confidence premature end; and
2. failure to develop stop propensity even during a long episode.

---

# 3. 观察 / Observations not yet promoted to general facts

## O1. "Small-target problem" is a research candidate, not a proven causal statement

The researcher has observed some low-SR categories and believes many are small objects. The preserved historical full-200 run does show elevated failure for mug/basketball/laptop/bowl relative to sofa/TV/bed, but:
- category sample sizes are small;
- mug failures are strongly clustered in the difficult early sub-house subset;
- basketball has only 9 samples;
- apple was perfect in that specific run;
- physical size was not directly measured.

Therefore "small object causes low SR" remains unproven.

## O2. Premature end is frequent in one historical full-200 run

16/27 failures were sub-horizon. This makes termination a high-value phenotype to characterize, but not automatically the root cause.

## O3. Safety Cost is highly skewed and usually absent in failures

Most failures had zero cost, while some successful runs had large cost. This weakens any simple claim that official Safety Cost events directly cause most ObjectNav failures.

## O4. Historical intervention evidence suggests hard oracle-based stop gating can create new loops

The modified-policy branch reduced some early exits but caused nearly equal regressions and long-loop behavior. This is negative design evidence against using a hard success-oracle gate as the final method.

---

# 4. 待验证假设 / Active hypotheses

## H-SIZE
Policy-independent physical target size contributes to lower ObjectNav SR.

Required evidence: static 3D target size + per-task outcome + task-difficulty controls. Current experiment tests this first.

## H-CATEGORY
Some semantic categories are harder independent of physical size because of visual semantics, training frequency, clutter/context, or instruction grounding.

## H-DIFFICULTY
House/task difficulty explains much of the apparent category effect. Historical `gt_episode_len` and target-room visitation differences make this a strong competing explanation.

## H-PERCEPTION
For some failure states, target information is weak or absent already in visual representations.

## H-REPRESENTATION
Target/stop-relevant information may be degraded somewhere between vision/fusion/temporal layers. Historical Probe does not currently support a simple "layer 3 loses stop information" story.

## H-READOUT
Stop-relevant information may exist in the final Actor representation but be poorly used/calibrated by the Actor head.

Required evidence: clean-B0 episode-aware Probe + actor sensitivity/intervention, not AUC alone.

## H-EXPLORATION
Some failures arise from never entering the target room / insufficient search / local loops before any valid stop state is reached.

## H-TERMINATION-CALIBRATION
The Actor may assign inappropriate end probability in some states: too high when illegal in some cases, too low when stopping should be considered in others.

## H-SAFETY-TRAINING
Safety-constrained training may have learned a more conservative Actor policy in some states.

Important: source topology does not show cost critic directly forcing `end` at inference. Any safety effect would have to be established as a learned Actor-policy effect.

## H-RESET
Cross-episode counter/cache rollover may alter the effective temporal history and eventually affect logits/actions. The implementation pathway is real; causal performance impact remains unproven.

---

# 5. 被否定 / 被更正 / 当前不允许的解释

## R1. "ObjectNav success requires target centered or ~1m away"
Superseded. Official non-strict success uses navigation-camera visibility with maximum distance 2m. Earlier video analyses that relied on a center/1m rule must not be reused as evidence.

## R2. "end_prob≈0.993 is an end Probe or 99.3% task-success probability"
False. It is the Actor action probability for `end`.

## R3. "AUC 0.992 proves the model represents small objects"
False. The historical label was close-and-visible / stop-legality-like, not object physical size.

## R4. "Layer 3 loses target/stop information"
Not supported by the preserved exploratory Probe; its layer-wise AUC increased rather than collapsed. A future clean Probe may still find a different representation failure, but the old result cannot support this statement.

## R5. "A high illegal-end probability proves representation loss or readout mismatch"
Unsupported. It proves an Actor output error in that state, not where the causal failure entered the chain.

## R6. "SafeVLA quits because the cost critic directly gets scared and gates end"
Not supported by source topology. Actor distributions come from the actor branch; c-values are returned separately. A safety-training effect remains testable, but not as a direct runtime gate claim.

## R7. "Cache retained across reset means the first step of the next episode necessarily reads old tokens"
Refuted for the tested short controlled decoder case; old cache was masked and current output matched zero-old-cache. Rollover effects later in an episode remain open.

## R8. "Logger caused the audited premature end by adding extra forward/RNG/action changes"
Not supported in the audited canonical/smoke scope. Do not generalize this to all future instrumentation.

## R9. "The historical modified-policy/PT-Guard branch is equivalent to official B0"
False. It changes action selection/environment trajectory and must remain a treatment/legacy branch.

## R10. "The proposed H1-H12 / P1-P4 diagnostic plan was fully executed"
False. The old `model plan.txt` contained proposed visibility/category/location/oracle-action probes and Action Logit Lens hooks. Those were design proposals; the only clearly evidenced historical Probe collection is the three-layer stop-legality-like collector described above.

---

# 6. Historical experiment / branch inventory

| ID / label | Status | What it can support | What it cannot support |
|---|---|---|---|
| LEGACY-FULL200 | HISTORICAL_REPORTED | 173/200 result, failure morphology, category/task-difficulty associations, cost distribution | current reproducibility or causal mechanism |
| LEGACY-GRPO-DIRECTION-A | HISTORICAL_TREATMENT | hard early-stop intervention can rescue some episodes and cause regressions/loops | B0 mechanism or B0 SR |
| LEGACY-PROBE-3L | EXPLORATORY / PROVENANCE-LIMITED | stop-legality-related information decodable in a modified trajectory dataset | small-target representation, causal Actor use, paper-grade layer claim |
| LEGACY-P0-P7-END-AUDIT | HISTORICAL_AUDIT | end probability semantics, logger/action topology, reset/counter facts, one illegal-end example | systematic end mechanism frequency |
| LEGACY-SUB120-TRACE | SINGLE-CASE / APPROX | one horizon failure had near-zero rendered p(done) throughout | exact logits or population-level claim |
| EXP-B0-REPRO-001A | COMPLETED / PROVENANCE PASS | narrow executable-B0 identity/task pairing/artifact completeness | full-200 performance |
| EXP-SMALLTARGET-PHENOTYPE-001 | APPROVED_FOR_CODEX | current zero-rollout phenotype/size audit | mechanism intervention |

---

# 7. What this migration changes for the current experiment

Executor must use this history to avoid rediscovering or misusing old claims.

For `EXP-SMALLTARGET-PHENOTYPE-001`:
- use raw historical 200-task evidence, not the later category recollection as ground truth;
- explicitly reconcile the preserved category table with the researcher's later ~50% observation;
- treat `gt_episode_len`, target-room visitation and house/sub-house effects as important confounds;
- do not define "small" using target distance, trajectory max visible pixels, or object category names;
- do not use historical Probe AUC as small-target evidence;
- do not launch replay, simulator, Probe, video-mechanism analysis or new rollout in this experiment.

---

# 8. Executor reading rule

Before executing the current experiment, read:
1. `research/LOOP_STATE.json`
2. `research/NEXT_EXPERIMENT.md` + `NEXT_EXPERIMENT.json`
3. `research/CURRENT_STATE.md`
4. **this file**
5. `research/EVIDENCE_REGISTER.md`
6. the raw evidence paths available on the approved execution worktree/server.

If a migrated statement conflicts with raw server evidence, raw evidence wins and the conflict must be reported in the handoff.
