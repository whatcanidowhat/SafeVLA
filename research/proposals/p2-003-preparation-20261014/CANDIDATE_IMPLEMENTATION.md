# P2-003 ON-only candidate — NOT AUTHORIZED

Base: 06cde9adeb91d47c062a11e72e90c7edf56fb398 (includes accepted ndarray fix). New experiment EXP-ACTOR-RESET-ON200-003; cycle actor-reset-on200-off002-reuse-003-20261014. Final PI code pin, approval/expiry, approval CI and unique claim/claim CI remain outstanding.

Only evaluate.py authorization/pre-model orchestration and analyze.py explicit identity-validator interface changed among historical files. Model execution body after the initial guard is AST-identical. Analysis metric body is unchanged. Root Actor reset, Critic behavior, Logger/storage, TaskSpecs, official metrics and checkpoint remain byte-identical. New recovery003.py, run_p2_on200_only.sh and CPU tests implement gates and provenance. The old full400 launcher remains historical and is not a valid 003 command.

Candidate checks: 41 CPU unit tests PASS (27 inherited plus 14 new recovery boundary tests), no CUDA initialized; shell syntax and scientific body parity PASS. Existing OFF200 strict validator rechecked 200 Episodes / 19270 decisions / 1203 raw files PASS, with pinned manifest/metrics/index hashes. No GPU, model or Episode was run during development.

Historical OFF /nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200 is read-only. New worktree /nvme2/user/qyy/SafeVLA_actor_logits_p2_on200_003_20261014. New result root /nvme2/user/qyy/SafeVLA_p2_003_on200_results_20261014 and launch evidence root /nvme2/user/qyy/SafeVLA_p2_003_launch_20261014 must be absent. Launcher invokes ON200 exactly once, no OFF replay, no retry. Budget gate 1 GPU / 200 Episodes; original 600-step horizon implies <=120000 decisions; inherited 168h / 100GiB / 12h queue guards and future approval expiry remain in effect.

Transport candidate: DIRECT HTTPS, explicitly no inherited HTTP(S)/ALL proxy; git http.proxy cleared per invocation; urllib ProxyHandler({}). Preparation exact Git+CI probes passed across two disconnected/reconnected SSH sessions. Initial github.com DNS timeout and a subsequent full-page download timeout were observed and recorded, so this is not a guarantee of stable future network. No tunnel/proxy service installed. No live remote guard bypass.

## Required future execution sequence (only after final approval)

1. PI pins this candidate's final commit in research/P2_CODE_SHA.txt and updates the approved 003 design. Wait approval CI; publish one claim through research_loop_claim.py; wait its distinct CI. Never reuse 002 claim.
2. In two separate SSH sessions, run the CPU-only readiness command below for the exact new claim. End session A before session B. Reports must be <=30 minutes old at launcher preflight, bind live HEAD+successful CI, and have different SSH session fingerprints. Replace CLAIM_SHA with the actual new claim; these files do not exist for a future claim now.

```bash
cd /nvme2/user/qyy/SafeVLA_actor_logits_p2_on200_003_20261014
/home/amax/.conda/envs/safevla/bin/python -B -m diagnostics.actor_logits.recovery003 network-readiness --control-root /nvme2/user/qyy/SafeVLA_loop_control_p2_codex_20261013 --report /nvme2/user/qyy/SafeVLA_p2_003_network_claim_20261014/CLAIM_SHA/session_a.json
# End this SSH session. In a fresh session use the same command with session_b.json.
```

3. From the exact clean PI-pinned tree run once:

```bash
bash diagnostics/actor_logits/run_p2_on200_only.sh
```

The launcher validates approval/claim CI, prior OFF bytes, scientific source parity, two cross-session receipts and three new live direct-network observations before GPU queue. Immediately before model construction it rechecks live control/CI, successful preflight binding, OFF hashes/strict integrity and exact TaskSpecs. It validates ON200, then re-verifies immutable OFF and performs explicitly cross-cycle comparison. Any failure STOP; preserve partial evidence. GPU Logger equivalence NOT_RUN_PI_WAIVED, never PASS.

Cross-cycle comparison checks exact task order/action map, weights, data/seed/settings and all old scientific source hashes; only explicit orchestration changes/new files are allowed. Physical GPU may differ from old GPU0 per draft 0-3 selection, and both identities are reported. Time/GPU load/simulator randomness/observation divergence remain confounds, not erased by aligned TaskSpecs. Default existing same-cycle analyze.compare behavior remains strict.

Preparation evidence is published separately on research-loop so the runtime source does not change when the result/code SHA is recorded. This candidate does not authorize execution or fill any future runtime handoff with simulated PASS.
