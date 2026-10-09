# PI Cycle — EXP-ACTOR-RESET-FULL200-001D P1 preflight

Experiment ID: EXP-ACTOR-RESET-FULL200-001D
Status: DRAFT
Cycle ID: actor-reset-full200-p1-20261009
Authorization: NOT_AUTHORIZED — PI_REVIEW only; no CODEX claim/GPU/episode/model execution.

## Decision and scientific purpose

Following user approval, the unclaimed 001C execution authorization was revoked in commit `e93a9cdff70fe537cfa377c2d1f63c7b432d45be` (control CI #37949189376 SUCCESS). This new unused cycle registers **only P1** as a draft. The separate 001D P2 200+200 performance trial remains NOT_AUTHORIZED and out of scope.

Scientific priority for the overarching 001D study: improve SafeVLA ObjectNav Success Rate while maintaining or reducing official Safety Cost. This P1 phase is a **measurement/implementation validity gate**, not a reset-performance experiment; it cannot establish SR or Cost benefit.

## Immutable evidence references

- P0 proposal and archived independent audit: https://github.com/whatcanidowhat/SafeVLA/tree/724a4b1db31c07dec3d69e5dc5f377ce1edd8eaf/research/proposals/actor-reset-full200-001d-20261009/p0-audit
- PR #1: https://github.com/whatcanidowhat/SafeVLA/pull/1 (Draft).
- Accepted 001A R1: https://github.com/whatcanidowhat/SafeVLA/tree/2f7864ae2c3a4761932aa4d1fedd6b6d5b3388c6/research/handoffs/reset-001a-20261007
- Prior 001C: revoked without claim or experiment; preserved in control Git history.

## P1 protocol to be approved later, NOT executable now

1. **Frozen identity and complete manifest**: isolate accepted official B0 `PKU-Alignment/SafeVLA@2aa82559d272b5f888e53433e258914057f15bed` plus approved local DINO loader, preserve source/weight/import hashes, and use exactly the original complete 200-task minival selection/shuffle semantics with `seed=123`. Export and validate a full 200-task TaskSpec manifest before any online run; then select its original first 5 ordered tasks.
2. **Online OFF/OFF only**: two independent clean one-worker OFF sessions, five tasks each (max **10 total live episodes**, horizon 600). Keep original stochastic sampling, `greedy=false`, `test_augmentation=true`, original success/cost logic and time/history behavior. No ON live run; no full200 experiment or additional seeds.
3. **Passive logging**: instrument the existing forward without extra calls or RNG, read **raw root actor.linear logits** independently from distribution normalized logits/probs, sampled/mode/executed action, root counter/KV and existing worker official cost components; never add controller queries or recompute safety outcomes with different timing.
4. **Offline input-matched checks**: on separately isolated Actor snapshots with identical inputs and RNG, compare passive logger absent/present and independently test root-Actor-only counter/K/V reset; confirm reward/cost critic counter/K/V and unrelated runtime state remain unchanged. Do not inject offline probes into live actor.
5. **Audit gates**: verify initial RNG/model-build/augmentation identity and origins; check full task ordering, A/A discrepancy sources, logger equivalence (tight declared tolerances), forward/sample count, official cost aggregation and output coverage. Stochastic online A/A need not be bitwise identical; unexplained divergence must block P2 interpretation.
6. **Failure policy**: retain partial evidence and STOP on any identity, instrumentation, cost, action or resource mismatch. No hidden retries, task replacement, silent fixes, model/simulator restart or partial-session continuation.

## Proposed P1 budget (NOT authorized)

- Simultaneous GPU: max 1; live episodes: **5+5 = 10**; live decisions: max **6000**.
- Offline replay: max **2400 combined forward calls**; proposed hard limit **8 GPU-hours** and **50 GiB** raw artifacts.
- `execution_worktree` stays null until a verified isolated B0 runtime snapshot is selected during further PI approval.
- No experimental command is frozen or executable yet (`NEXT_EXPERIMENT.json.command=[]`).
- All proposed ceilings are DRAFT only. Actual authorization currently grants **0 GPU / 0 episodes**.

## Required handoff artifacts if a later P1 is explicitly approved and run

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

The PI must separately approve P1 and CODEX must claim a subsequently approved instruction commit with green claim CI before executing. P2 requires another distinct cycle and authorization.

