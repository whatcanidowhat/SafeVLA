# PI Cycle — EXP-ACTOR-RESET-FULL200-001D P1 preflight

Experiment ID: EXP-ACTOR-RESET-FULL200-001D
Status: APPROVED
Cycle ID: actor-reset-full200-p1-20261009
Authorization: APPROVED_FOR_CODEX — RESEARCH_EXPERIMENT / max 1 GPU / max 10 live episodes. Expires 2026-10-11T15:41:29.332Z. No execution before green claim CI.

## Decision and scientific purpose

Following user approval, the unclaimed 001C execution authorization was revoked in commit `e93a9cdff70fe537cfa377c2d1f63c7b432d45be` (control CI #37949189376 SUCCESS). This new unused cycle registers **only P1** as a draft. The separate 001D P2 200+200 performance trial remains NOT_AUTHORIZED and out of scope.

Scientific priority for the overarching 001D study: improve SafeVLA ObjectNav Success Rate while maintaining or reducing official Safety Cost. This P1 phase is a **measurement/implementation validity gate**, not a reset-performance experiment; it cannot establish SR or Cost benefit.

## Immutable evidence references

- P0 proposal and archived independent audit: https://github.com/whatcanidowhat/SafeVLA/tree/724a4b1db31c07dec3d69e5dc5f377ce1edd8eaf/research/proposals/actor-reset-full200-001d-20261009/p0-audit
- PR #1: https://github.com/whatcanidowhat/SafeVLA/pull/1 (Draft).
- Accepted 001A R1: https://github.com/whatcanidowhat/SafeVLA/tree/2f7864ae2c3a4761932aa4d1fedd6b6d5b3388c6/research/handoffs/reset-001a-20261007
- Prior 001C: revoked without claim or experiment; preserved in control Git history.

## P1 approved protocol — executable only after green CODEX claim CI

1. **Frozen identity and complete manifest**: isolate accepted official B0 `PKU-Alignment/SafeVLA@2aa82559d272b5f888e53433e258914057f15bed` plus approved local DINO loader, preserve source/weight/import hashes, and use exactly the original complete 200-task minival selection/shuffle semantics with `seed=123`. Export and validate a full 200-task TaskSpec manifest before any online run; then select its original first 5 ordered tasks.
2. **Online OFF/OFF only**: two independent clean one-worker OFF sessions, five tasks each (max **10 total live episodes**, horizon 600). Keep original stochastic sampling, `greedy=false`, `test_augmentation=true`, original success/cost logic and time/history behavior. No ON live run; no full200 experiment or additional seeds.
3. **Passive logging**: instrument the existing forward without extra calls or RNG, read **raw root actor.linear logits** independently from distribution normalized logits/probs, sampled/mode/executed action, root counter/KV and existing worker official cost components; never add controller queries or recompute safety outcomes with different timing.
4. **Offline input-matched checks**: on separately isolated Actor snapshots with identical inputs and RNG, compare passive logger absent/present and independently test root-Actor-only counter/K/V reset; confirm reward/cost critic counter/K/V and unrelated runtime state remain unchanged. Do not inject offline probes into live actor.
5. **Audit gates**: verify initial RNG/model-build/augmentation identity and origins; check full task ordering, A/A discrepancy sources, logger equivalence (tight declared tolerances), forward/sample count, official cost aggregation and output coverage. Stochastic online A/A need not be bitwise identical; unexplained divergence must block P2 interpretation.
6. **Failure policy**: retain partial evidence and STOP on any identity, instrumentation, cost, action or resource mismatch. No hidden retries, task replacement, silent fixes, model/simulator restart or partial-session continuation.

## Approved P1 budget (hard ceilings; no automatic retry or P2)

- Simultaneous GPU: max 1; live episodes: **5+5 = 10**; live decisions: max **6000**.
- Offline replay: max **2400 combined forward calls**; proposed hard limit **8 GPU-hours** and **50 GiB** raw artifacts.
- Frozen isolated execution worktree: `/nvme2/user/qyy/SafeVLA_p1_001d`. Executor may create it from accepted B0 only after approved claim+green claim CI; the original B0 and development worktrees remain untouched.
- Frozen execution entry (to be implemented in isolated worktree under the approved protocol only): `/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1-20261009/run_p1_preflight.py`. May be executed only after CODEX claims this exact approval commit and claim CI succeeds.
- Hard limits are **1 GPU / 10 live episodes / 6000 live decisions / 2400 offline combined forwards / 8 GPU-hours / 50GiB raw**. These are executable ceilings only following approved CODEX claim and successful claim CI.

## Required handoff artifacts for this approved P1

- `research/handoffs/actor-reset-full200-p1-20261009/RESULT_SUMMARY.md`
- `research/handoffs/actor-reset-full200-p1-20261009/RUN_MANIFEST.json`
- `research/handoffs/actor-reset-full200-p1-20261009/ARTIFACT_INDEX.json`
- `research/handoffs/actor-reset-full200-p1-20261009/REVIEW_NOTES.md`
- `research/handoffs/actor-reset-full200-p1-20261009/FROZEN_PROTOCOL.md`
- `research/handoffs/actor-reset-full200-p1-20261009/TASK_MANIFEST.csv`
- `research/handoffs/actor-reset-full200-p1-20261009/AA_VALIDATION.md`
- `research/handoffs/actor-reset-full200-p1-20261009/LOGGER_EQUIVALENCE.md`
- `research/handoffs/actor-reset-full200-p1-20261009/RESET_INVARIANTS.md`
- `research/handoffs/actor-reset-full200-p1-20261009/P1_RESOURCE_PROFILE.json`

## Expected / falsifying outcomes

Expected: two valid five-task OFF sessions plus input-matched deterministic logger equivalence, complete authoritative official-cost/manifest data and offline Actor-only reset invariants. This **only** qualifies P1 for PI review.

Falsifying/blocking: logger changes any forward, RNG, action or context; ON affects reward/cost critic; wrong task order, source hashes, official metrics or missing evidence; budget exceeds limits.

P1 is separately approved by the PI in this control transition; CODEX must claim this exact approved instruction commit and receive green claim CI before executing. The executor must then build the one-shot runner in the isolated worktree and abort before any live episode if source, weights, full manifest, original metrics, logger/critic isolation or numerical equivalence gate fails. P2 requires another distinct cycle and authorization.


## PI-approved P1 protocol addendum (claim and green claim CI required)

P1 **engineering** preflight is frozen for approval, not the P2 performance experiment. Build a separate snapshot in `/nvme2/user/qyy/SafeVLA_p1_001d`, preserving accepted official B0 `2aa82559d272b5f888e53433e258914057f15bed`, DINO infrastructure adaptation, checkpoint/data digests, and original stochastic evaluation. Use exactly two independently initialized **OFF** single-worker, five-task sessions from the original complete 200-task manifest order (`shuffle=true, seed=123`, task horizon 600). Never introduce live ON or additional episode trials.

Launcher initialization policy for this single-worker research protocol: set Python/NumPy/PyTorch CPU+GPU RNG seed=123 at each new process start *before* model construction, record RNG provenance, and preserve original evaluator shuffle/augmentation/sampling code and later state evolution. This is an explicit protocol choice, not a claim of bitwise equivalence to official 8-worker evaluation. Separate processes per OFF session; no per-episode reseeding.

Offline checks use independent model/cache snapshots and bounded saved real Actor inputs. Compare logger disabled/enabled with matched input/state/RNG: raw root `actor.linear` logits vs normalized categorical logits/probs clearly separated; max_abs numerical difference <= **1e-5** for relevant float outputs/state, mode/sample/executed action and RNG/forward/sample counts identical. Independently verify OFF versus ON episode-boundary reset changes root Actor counter and per-layer K/V only; reward/cost critic counter/K/V and other non-target state exact; no extra live forward, sampler RNG draw, or controller query. Capture official worker `metrics.cost` and five components with original pre-step accounting, success with official threshold/boolean semantics. Missing source/import/action/metric identity is STOP, not a silent fallback.

STOP on any source/checkpoint/DINO/TaskSpec order drift, extra forward/sample/query, critic alteration, logger numerical/treatment contamination, failed metric reconciliation, missing output, > 1 concurrent GPU, > 10 live episode initialization attempts, > 6000 live steps, > 2400 offline combined forward calls, > 8 GPU-hours or > 50GiB raw. No automatic retry, new seed, restart/continuation, P2 or additional task. Preserve partial error evidence and handoff for PI review.

Executable command **once and only after** authorization+claim CI:

```
/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1-20261009/run_p1_preflight.py
```

Required additional outputs:
- `research/handoffs/actor-reset-full200-p1-20261009/run_p1_preflight.py`
- `research/handoffs/actor-reset-full200-p1-20261009/validate_p1_outputs.py`
- `research/handoffs/actor-reset-full200-p1-20261009/IMPLEMENTATION_DIFF.md`
- `research/handoffs/actor-reset-full200-p1-20261009/LOGIT_AND_COST_SCHEMA.md`

All implementation paths and runtime code hashes must be archived; any deviation needs a new PI decision before execution. The P2 200+200 A/B budget is not approved.

## P1 explicit PI authorization decision

User direction: P0 review and control CI are complete; permit Codex to execute the *P1 engineering preflight* rather than repeat P0. Approval commit grants **only** the frozen 5+5 OFF A/A online tasks and isolated offline logger/reset invariants. Active authorization: 1 simultaneous GPU and at most 10 initiated live episodes; execution ceiling 6000 decisions, 2400 offline combined forwards, 8 GPU-hours and 50GiB raw artifacts; expires 2026-10-11T15:41:29.332Z. No P2, no live ON, no retries or added seeds. Approval is permission to claim; it is **not** a claim or a launched experiment. Codex must fetch latest research-loop, claim exact green approval HEAD with claim CI and stop after publishing required handoff evidence.
