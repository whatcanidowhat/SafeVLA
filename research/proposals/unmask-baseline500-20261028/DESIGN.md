# EXP-ACTOR-MEMORY-UNMASK-001 — corrected effective Baseline

Status: DRAFT_FOR_PI_REVIEW. This revision supersedes the erroneous **1000** configuration statements in earlier unmask preparation/design packets. Historical files/results remain preserved; they are not silently relabeled as500 evidence. No new PI authorization or Claim is created here.

## Scientific question and endpoints

Does removing only the root Actor cross-Episode attention mask improve official SR and reduce official Safety Cost? Define ΔSR=SR_MASK_OFF−SR_MASK_ON and ΔCost=mean_cost_MASK_OFF−mean_cost_MASK_ON. Desired joint direction is **ΔSR>0 and ΔCost<0**, with both endpoints, uncertainty, five component totals/distributions and all paired tasks reported. Changes in Actor logits are not the scientific endpoint and are not evidence of SR/Cost benefit.

Hypothesis: preserved useful history improves navigation and safety. Competing explanations: old-goal/scene interference, stochastic online trajectory and augmentation divergence, and the interaction between Episode order/length and the unchanged500-slot rollover. No benefit is assumed. Failure/no improvement or worse cost falsifies the directional prediction; nonsignificance is not equivalence.

## Corrected frozen conditions

| Setting | MASK_ON200 Baseline | MASK_OFF200 treatment |
|---|---|---|
| Root Actor cross-Episode mask | ON | OFF |
| Root Actor boundary reset | OFF | OFF |
| Actor / reward Critic / cost Critic max_steps |500 /500 /500|500 /500 /500|
| Per-layer K/V cache sequence capacity, all branches |500|500|
| Episode horizon |600|600|
| Checkpoint, sensors, official metrics |same|same|
| TaskSpecs |same ordered200|same ordered200|
| Seed / worker |123 /1 persistent worker0|123 /1 persistent worker0|
| Sampling / augmentation |original stochastic sample then mode; test augmentation ON|same|

Effective500 comes from BaseConfigParams.max_steps; DinoV2ViTSTSFMBase passes self.params.max_steps into both model.max_steps and initial_tgt_cache_shape[0]. Model-class constructor default1000 is overridden by the standard evaluator and remains unchanged. Horizon600 permits a single Episode to cross the unchanged Actor rollover at500. Cache is not extended, cleared or resized as a treatment; counter and cache writes/rollover retain original behavior.

Unique variable: root Actor mask ON versus OFF. prev_actions masks, Critic masks, parameters, original decoder/cache code, time_step_counter/max_steps rollover, action map/end semantics, official success/cost aggregation, task data/order and number of live policy forwards remain unchanged. OFF retains causality and exposes only the valid cached prefix; there is no extra live counterfactual forward.

Checkpoint SHA25605b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301. DINO SHA256b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9. Ordered TaskSpec manifest SHA2568ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0. Do not reuse old OFF200 as the new control.

## Minimal corrected validation

CPU fixture and component replay explicitly use500 slots. Regenerate the small existing fixed-input check once, not a larger logits study. Genuine500-step rollover tests compare the original forward with the unchanged-default-mask candidate across steps499/500/501. Read-only runtime assertions after complete checkpoint loading inspect actual root Actor, reward Critic and cost Critic max_steps, decoder max_seq_len, time encoder configuration and every layer's actual K/V sequence capacity. Any mismatch aborts before configuring the root treatment; no repair/reset/resize to make the assertion pass. Asserted metadata goes into the runtime manifest.

CPU tests cover both valid allocated and empty-batch500-slot cache states, no RNG/cache/counter/forward mutation by assertions, rejection without repair of wrong Actor/Critic/cache configuration, baseline500/horizon600 approval fields, and behavior/result sign calculations. These tests are not full GPU/renderer equivalence evidence.

## Online execution and reports

After target environment/assets/rendering and protocol gates pass, execute **MASK_ON200 → MASK_OFF200**, two fresh processes on the same one Slurm V100 allocation/UUID, fixed seed before each, persistent history inside each arm. Total400 initiated Episodes and at most240000 decisions; a total200-start budget is insufficient. No per-Episode toggling/reset or new old-control reuse. Resources: one GPU,64GiB host RAM,12h cumulative execution,90GiB output ceiling and100GiB initial free disk; queue/absolute approval window remain PI-owned. Preserve failed/partial results and stop without automatic retry, switch or unregistered parameter change.

Per arm: official SR numerator/200, Safety Cost sum and mean/200, **danger/corner/blind/fragile/critical**, cost distribution/median/p90/max. Every matched TaskSpec reports Δsuccess, Δcost and all five Δcomponents, plus paired raw behavior summaries. Δ values are OFF−ON. Paired bootstrap10000 replicates, analysis seed20261028; McNemar and uncertainty are descriptive given sequential cross-Episode history dependence, not an independence or causal-identification guarantee.

Failure behavior: failed early END means official failure and terminal action=end before600; successful early END is kept separate. Report failed horizon exhaustion, official percentage/total rooms visited, whether the target room was visited, adjacent repetition and ABA oscillation. Room coverage/target-room visits are exploration proxies, not a proven causal diagnosis of insufficient exploration. Missing official fields remain unavailable. All analysis consumes archived outputs only; no extra policy/environment forward or controller queries.

## Authorization, confounds and stop conditions

The user's execution request persists; no redundant general GPU-consent question is required. Independent PI authority and the existing control protocol still apply: handle the prior terminal cycle, review the exact corrected shared SHA, stage/approve the unique unmask design with500/cache500/horizon600, pass approval CI, create one new Claim, pass that exact Claim CI, and satisfy live source/environment/resource/TLS/asset/renderer gates. Old Claim and old GPU logger waiver do not apply. Executor never approves as PI or edits control status to READY.

Stop before model launch on wrong source/config/cache/TaskSpec/checkpoint/dependency identity, missing asset/renderer review, old/missing/expired authorization or Claim/CI, extra forward, unsafe GPU/disk/deadline, existing output path, or any integrity error. Any online branch mismatch or incomplete arm prevents a full benchmark contrast. Preserve evidence and hand back to PI.

Principal confounds: cross-task goal/scene history, Episode-length-dependent rollover phase, sequential arm/wall-clock effects, augmentation and stochastic trajectory divergence despite the same seed, target rendering/environment adaptation and unverified GPU logger noninterference. All remain visible in the final report. Current scientific conclusion: **SR and Safety Cost effects remain unmeasured until both authorized complete online arms exist.**
