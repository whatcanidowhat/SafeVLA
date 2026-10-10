# PI Cycle — 001D P1R2: four-GPU shared queue, offline logger and OFF/OFF preflight

Experiment ID: EXP-ACTOR-RESET-FULL200-001D-P1R2
Status: DRAFT
Cycle ID: actor-reset-full200-p1r2-20261010
Authorization: NOT_AUTHORIZED — PI_REVIEW, 0 active GPU/episodes; no queue, model or episode before independent authorization and successful CODEX claim CI.

## Rationale and approvals

User requested **GPU 0,1,2,3 as candidates**, permitted GPU hold with absolutely no killing others' processes, and offered to launch the approved runner manually on the server. This control proposal adopts all three preferences. Terminal P1R `b65cb07d87dc3321f79b89d9b1ffcde0ff2e1db3` was signed by PI `cb579ffdc10c86999be68600e788d938cdb7a89c` (control CI #38064189115 SUCCESS): CPU Gate A PASS in 4 genuine Attention lifecycle cases / 24 Attention calls, but GPU0 free 3954MiB < strict frozen >6823.390767MiB, so Gate B/C NOT_STARTED and official SR/Cost NOT_MEASURED. Original P1 and P1R are terminal and not executable.

Scope is a NEW, one-shot P1R2 cycle: **reuse accepted Gate A evidence, queue safely for one of four shared GPUs, run unchanged Gate B GPU offline engineering controls, then unchanged Gate C OFF/OFF 5+5 only if B PASS.** Zero ON live treatments, no P2. All original B0 source hashes, the P1R shape-aware snapshot restore, checkpoint/DINO/dataset and official metrics identities remain frozen.

## Gate Q — shared-lab GPU queue, before any CUDA/model initialization

1. Candidate **physical GPUs 0,1,2,3**, no other GPU IDs. Use `nvidia-smi` to observe each card's compute PIDs, memory.used, utilization and free memory; require **no foreign compute job**, **used<=4000 MiB**, **util<=30%**, and **free>6823.390767 MiB**, plus same supported CUDA/model capability. Observe three qualifying polls spaced 30 seconds apart; prefer freest qualified GPU; record physical id and UUID.
2. **Max queue wait = 12 hours**, ending earlier if grant expires. Poll CPU-only (no Torch CUDA initialization, checkpoint load, simulator start). Record full timestamps/observations and queue_wait_seconds. Never change scientific episode/sampling settings.
3. **Optional own-GPU hold is permitted:** after an idle card is identified and while preparing launch, spawn at most one child with `CUDA_VISIBLE_DEVICES=<chosen physical id>`; allocate no more than 4096MiB for no longer than 120 seconds. No active busy-loop or extra Torch calls. Record own PID and GPU UUID. `trap` cleanup on termination/interrupt and kill/wait **only the exact spawned own PID**, never any foreign PID, `pkill`, `killall`, `sudo` or GPU reset. A hold is advisory, not exclusive server ownership.
4. Release/reap own hold BEFORE the GPU model child starts. Immediately recheck selected device: foreign compute absent, original strict free>6823.390767MiB. If another process wins the race, go back to bounded queue **before any model child/episode starts**; no retries of a launched phase, no cross-card failover after any GPU forward. Once selected for actual Gate B, keep the same physical GPU for Gate C A/B sessions.
5. Export `CUDA_VISIBLE_DEVICES=<physicalGPU>` for **all** GPU children. Inside torch and original `--gpu_devices 0`, logical CUDA device 0 refers to this mapped single physical GPU. Remove inherited `CUDA_VISIBLE_DEVICES='0'` hardcode from new isolated runner ONLY; assert child GPU UUID matches chosen physical ID. This is transport/scheduling infrastructure only.
6. `GPU_WAIT_INTERVAL=30`, `GPU_WAIT_STABLE_CHECKS=3`, `GPU_WAIT_MAX_MEM_MB=4000`, `GPU_WAIT_MAX_UTIL=30`, `GPU_WAIT_MAX_SECONDS=43200`, `GPU_HOLD_ENABLED=1` optional, `GPU_HOLD_MEM_MB<=4096`, `GPU_HOLD_MAX_SECONDS<=120`. Never implement infinite wait or kill other jobs.
7. **Separate clocks:** queue max12h; experiment GPU/episode gate max8h starts only at GPU work launch. Waiting before GPU work does not exhaust the 8h experiment budget, but nothing may proceed after grant expires. If a new queue wait occurs before later phases, log it separately and do not reset the 8h experiment clock.

## Gate A provenance (NOT rerun)

- Read/verify original P1R `CACHE_LIFECYCLE_CPU.md`, `cpu.json`, `IMPLEMENTATION.patch.txt` and code digests at `b65cb07d...`; accepted genuine frozen B0 3 Attention layers, float32/float64 × empty/allocated, shape-aware K/V clone-rebind restoration and output max_abs 0. This is trusted CPU engineering evidence only, not GPU scientific PASS.
- If prior CPU file/code hashes fail or restored snapshot logic changes, STOP; a new Gate A rerun would need PI review. No need to perform another GPU or online event for Gate A.

## Gate B and C (unchanged original scientific protocol)

**Gate B GPU offline:** 600 historical accepted 001A saved Actor inputs, same input/RNG and copied state for logger OFF/ON; max_abs <=1e-5 for raw actor logits, normalized logits, distribution probs and root decoder K/V, exact sampled actions/mode/history/RNG/counters/forward counts. Root-Actor-only cache/counter reset OFF vs ON offline; critic reward/cost, RNG and augmentation exact invariants. B failure STOP before online.

**Gate C online OFF/OFF:** if and only if B PASS, original TaskSpec 200-row `shuffle=true,seed=123` manifest SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0` with independently verified original loader, first five source indices 127,103,16,6,50, one persistent Agent per independent session, 2×5 live episodes max, `greedy=false`, `test_augmentation=true`, horizon600. Original sampled-action history, official success >0.1 and five-part **pre-step Safety Cost** aggregation preserved. Both OFF sessions fixed on the same chosen physical GPU. Stochastic differences documented, not misattributed to reset. No live ON.

## Isolated worktree, script and hard limits

- New worktree: `/nvme2/user/qyy/SafeVLA_p1r2_001d`, must be fresh and physically isolated from original accepted B0, development, P1 and P1R artifacts.
- New runner command (to be implemented by CODEX only after authorized claim and green claim CI):

```bash
/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1r2-20261010/run_p1r2_preflight.py
```

- Environment knobs above control GPU queue/hold only, never scientific sampling parameters. Manual user Bash startup is allowed **after** CODEX claim and claim CI; user manual start must not bypass the run's one-shot lock or Python supervisor verification. Neither this DRAFT nor approved-by-PI label permits direct running until those gates pass.
- **Active in draft = 0 GPU / 0 live episodes. Proposed maximum after PI approval = 1 concurrent GPU / 10 live starts / 6000 online decisions / 2400 offline combined forwards / 8h experiment GPU ceiling / max 12h queue / 50GiB raw artifacts.**
- Original GPU free guard >6823.390767 MiB remains strict. If time/foreign PID/OOM/capacity violates guard, preserve evidence and STOP without restarting launched experiment, switching GPU, or altering B0.
- All required Git handoff files (also terminal BLOCKED/INVALID, with NOT_STARTED gates):
  - `research/handoffs/actor-reset-full200-p1r2-20261010/RESULT_SUMMARY.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/RUN_MANIFEST.json`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/ARTIFACT_INDEX.json`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/REVIEW_NOTES.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/FROZEN_PROTOCOL.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/TASK_MANIFEST.csv`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/AA_VALIDATION.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/LOGGER_EQUIVALENCE.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/RESET_INVARIANTS.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/P1R2_RESOURCE_PROFILE.json`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/run_p1r2_preflight.py`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/validate_p1r2_outputs.py`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/GATE_A_PROVENANCE.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/GPU_QUEUE_REPORT.json`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/GPU_QUEUE_LOG.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/IMPLEMENTATION_DIFF.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/LOGIT_AND_COST_SCHEMA.md`
  - `research/handoffs/actor-reset-full200-p1r2-20261010/GATE_STATUS.json`
- P2 OFF200/ON200 remains NOT_AUTHORIZED. Completion returns result to PI and STOP. No extra seeds, automatic rerun, new task, or safety-metric alteration.

## PI decision

**Proposal staged, no execution yet.** Any actual task needs a second green approval control commit followed by CODEX's unique claim and green claim CI. Reusing old P1R/P1 claim IDs is forbidden.
