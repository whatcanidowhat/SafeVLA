# PI Cycle — 001D P1R: shape-aware cache recovery and gated engineering preflight

Experiment ID: EXP-ACTOR-RESET-FULL200-001D-P1R
Status: DRAFT
Cycle ID: actor-reset-full200-p1r-20261010
Authorization: NOT_AUTHORIZED — PI_REVIEW, 0 active GPU / 0 episodes (staging only)

## Decision and scope

User explicitly approved PI proposal to register a **fresh P1R** cycle and, after green control CI, authorize Codex for sequential Gate A/B/C execution within the original P1 resource ceilings. This stage commit **does not grant execution**. Prior P1 cycle `actor-reset-full200-p1-20261009` is terminal INVALID and may never be claimed or retried.

Accepted old handoff `a137eafbb5931190b68ff042a5b0c9e34dbfd68d`, PI acknowledgement `45b942ec2c3980401b5b06b85977318b9297a5ae`, control CI #38042403557 SUCCESS. Technical root: original runner's `restore_temporal()` used `.copy_()` from pre-first-forward cache [0,500,8,64] into lazily allocated [1,500,8,64] tensor; original B0 Llama Attention code creates cache on the first single-step forward. Neither SR nor official Safety Cost was measured in P1.

**Allowed correction:** in a new isolated runner only, restore complete K/V cache tensor snapshots by shape/dtype/device/value-preserving clone/attribute rebind; restore root counter coherently. Do not modify model Attention.forward, accepted B0, reward/cost critics, action mapping, simulator, checkpoint, official metric definitions, or the failed P1 execution worktree.

## Gate A — CPU cache lifecycle (must PASS before any GPU)

- Use real frozen B0 Llama Attention/cache code where possible, with real zero-batch initial state.
- Test `[0,500,8,64] -> first single-step forward -> [1,500,8,64] -> restore [0,500,8,64] -> repeat same forward`.
- Compare identical inputs + RNG: output, resulting counter, K/V shapes, dtypes, devices, full values/hashes; ensure initial snapshot not mutated, restore allocated cache state too, all decoder layers and tensor alias concerns covered.
- Failure, inability to load genuine implementation, or any required B0 source change => `INVALID/BLOCKED` and STOP. **No GPU use and 0 live episodes.**

## Gate B — GPU offline P1 engineering tests (only after A PASS)

- Preserve accepted B0 identity: 112 tracked source files; approved local DINO infra; checkpoint SHA256 `05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301`; DINO weight SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- Reuse frozen accepted 001A saved input SHA256 `417e684e96563b043138d328776d693f05dc74bb7c074099dea39c3b5881500e`, 600 sequential Actor inputs; reconstruct decoder inputs exactly; record every new runner SHA/diff.
- Logger absent/present paired replay at identical saved input/state/RNG. Max_abs for raw logits, normalized logits, probs, next K/V <= `1e-5`; discrete sample/mode/action/history/RNG/counter/forward-count exact. No extra forward/query in any live actor.
- OFF/ON reset test **offline only**: root counter and every root K/V reset, critic counter/K/V and all other model/RNG/augmentation state unchanged. Sentinel critic states are engineering fixtures, not real historical behavior. Failing any assertion => STOP and no online phase.

## Gate C — bounded online OFF/OFF A/A (only after B PASS)

- In new clean individual processes, run *OFF* one-worker 5-task prefix twice, max 10 initiated episodes in total; 600 horizon, max 6000 online decisions; no live ON.
- Full accepted original 200-row manifest SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0` from prior run may be reused only after independently regenerating and cross-checking source TaskSpec/order and original loader before launching first episode. First five original source indices: 127,103,16,6,50.
- `shuffle=true`, seed=123 before each process model construction, `greedy=false`, `test_augmentation=true`, preserve all subsequent RNG/augmentation/Agent state and official `success` and five-component `metrics.cost` pre-step aggregation. Independent A/A stochastic divergence is documented; not misinterpreted as reset effect.
- Invalid manifest, altered metrics, added forward/sample/query, unaudited runtime identity or a failed phase => STOP and preserve partial results. No retries or extra seeds.

## Budget, isolated worktree, one-shot handoff

- Active in draft: **0 GPU / 0 episodes**. Proposed hard upper bounds only after independent PI approval: **1 GPU, 10 live episode starts (5+5), 6000 online steps, 2400 offline combined forwards, 8 GPU-hours, 50 GiB raw outputs**. Do not shift unused old P1 budget or re-use old claim.
- New execution worktree: `/nvme2/user/qyy/SafeVLA_p1r_001d` (ensure fresh/empty, separate from `/nvme2/user/qyy/SafeVLA_p1_001d`, baseline and development trees).
- Frozen one-shot entry (must be implemented in the new isolated worktree and vetted against this exact instruction; do not execute before green approval+claim CI):

```bash
/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1r-20261010/run_p1r_preflight.py
```

- Required Git-readable handoff outputs, even when BLOCKED/INVALID, with explicit NOT_STARTED markers:
  - `research/handoffs/actor-reset-full200-p1r-20261010/RESULT_SUMMARY.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/RUN_MANIFEST.json`
  - `research/handoffs/actor-reset-full200-p1r-20261010/ARTIFACT_INDEX.json`
  - `research/handoffs/actor-reset-full200-p1r-20261010/REVIEW_NOTES.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/FROZEN_PROTOCOL.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/TASK_MANIFEST.csv`
  - `research/handoffs/actor-reset-full200-p1r-20261010/AA_VALIDATION.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/LOGGER_EQUIVALENCE.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/RESET_INVARIANTS.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/P1R_RESOURCE_PROFILE.json`
  - `research/handoffs/actor-reset-full200-p1r-20261010/run_p1r_preflight.py`
  - `research/handoffs/actor-reset-full200-p1r-20261010/validate_p1r_outputs.py`
  - `research/handoffs/actor-reset-full200-p1r-20261010/CACHE_LIFECYCLE_CPU.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/IMPLEMENTATION_DIFF.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/LOGIT_AND_COST_SCHEMA.md`
  - `research/handoffs/actor-reset-full200-p1r-20261010/GATE_STATUS.json`
- Runtime evidence source hashes, complete stderr/partial traces, actual episode budgets and artifact hashes are mandatory. Do not fabricate passing results.
- P2 OFF200/ON200 remains **NOT_AUTHORIZED**. Completion of P1R never auto-approves it.
- If authorized later, Codex follows `research/HANDOFF_PROTOCOL.md`: ensure latest approval green, atomic claim with `scripts/research_loop_claim.py`, verify claim CI green, one command, publish terminal handoff, STOP.

## PI decision

P1R is a narrow engineering recovery trial, **not a new scientific hypothesis about SR**. Accepted 001A mechanistic evidence is unchanged. No PI or Executor may treat the prior invalid run as a P1 PASS or a performance observation.
