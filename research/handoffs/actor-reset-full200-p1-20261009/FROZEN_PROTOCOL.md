# Frozen P1 engineering protocol

EXP-ACTOR-RESET-FULL200-001D; cycle actor-reset-full200-p1-20261009.
Instruction 67c42b0fd7d485fdaa9fe5687099537c2c28035a; published claim
334e80351ff6cb2cfb0dac166fa7227ae055a5bd; claim_id b19afa0bbe1b4957b22ce081a44ee3e0.
Claim CI 38016589200 succeeded. Authorization expires 2026-10-12T01:30:05.821Z.
This file freezes implementation before first invocation. It records a plan, not successful gates.

Run once from /nvme2/user/qyy/SafeVLA_p1_001d:
`/home/amax/.conda/envs/safevla/bin/python research/handoffs/actor-reset-full200-p1-20261009/run_p1_preflight.py`

Baseline: accepted 001A executable B0, upstream 2aa82559d272b5f888e53433e258914057f15bed plus approved local DINO loader adaptation.
112 source files are byte matched to accepted source hashes. Checkpoint SHA256
05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301;
DINO weights b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9.
Check Python, dependencies, DINO 157 source files and runtime module roots; no fallback.

TASK_MANIFEST.csv is actually generated (200 real rows), SHA256
8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0.
The original loader must separately cross-check all 200 raw TaskSpecs and order before an online episode.
The frozen selected source indices are 127,103,16,6,50. A and B each consume this prefix with one persistent Agent, in separate processes.
Full queue remains 200 so original video selection and RNG progression remain intact. Stop after collecting fifth official task result before a sixth initialization.
Pre-model Python/NumPy/PyTorch CPU+GPU seeds 123; retain subsequent official seeds/shuffle and augmentation progression. No episode reseed.
Official stochastic sample, greedy false, augmentation true, 20 actions, end index4, task horizon600 and Actor rollover500.

Offline stage precedes live episodes. Load an independent Agent; replay accepted real saved root Actor inputs from 001A donor episode_order2, 600 sequential inputs.
Captured-input SHA256 417e684e96563b043138d328776d693f05dc74bb7c074099dea39c3b5881500e.
Replace only isolated offline visual encoder with saved feature delivery, assert reconstructed decoder input matches captured input exactly.
For each input restore identical root cache/counter and all RNG, compare logger absent/present, 1200 root forwards total and zero combined forwards.
Raw logits, normalized logits, probabilities and next cache max_abs <=1e-5; sample/mode/history/counter/RNG/call-count equality exact.
Common passive oracle observer is present in both arms. This gate does not establish frontend or simulator equivalence.
Offline reset writes root counter and root decoder-layer K/V only. Nonzero reward/cost critic fixtures test against accidental recursive reset; fixtures are engineering sentinels never forwarded, not claimed as real critic rollouts.
Compare both critic caches/counters directly, state_dict, non-target module attributes, augmentation and RNG. No live ON.

Hard ceilings: 1 visible GPU, 10 initialization attempts (5+5), 6000 live decisions, 2400 offline combined forwards, 8 hours elapsed conservative GPU upper bound, 50GiB raw output.
Resource availability check uses three checkpoint-size copies plus1GiB headroom on GPU0; no reservation or guarantee against concurrent users.
Exclusive launch and phase markers prevent duplicate invocation. Failure stops subsequent phases; preserve evidence and hand back, no restart/retry, no P2.
Independent OFF differences require PI interpretation; they cannot establish reset benefit. P1 is not a full benchmark SR/Cost trial.
