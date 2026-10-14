# P2-003 recovery proposal — reuse valid OFF200, execute ON200 only

Experiment ID: EXP-ACTOR-RESET-ON200-003
Status: DRAFT
Cycle ID: actor-reset-on200-off002-reuse-003-20261014

**PI_REVIEW / NOT_AUTHORIZED.** PI accepted the old BLOCKED incident and the scientific validity of the stored OFF200. This document prepares a future ON200-only candidate, NOT an execution approval. Active resource allocation is **0 GPU / 0 Episodes**. No Claim exists for 003; the preceding P2-002 Claim remains historical and terminal.

## Frozen reference OFF200 (read-only)

- P2-002 evidence entry: 88453def76051736edce2e8887f26b08d5b0c5e6, reviewed by PI at 2ec8dccccbe154951565f71dc895519deaa949ab; terminal handoff CI and acknowledgement CI SUCCESS.
- Old implementation: 06cde9adeb91d47c062a11e72e90c7edf56fb398; baseline directory: `/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200`; P2-002 OFF200 complete 200/200 and validation PASS, 19,270 decisions.
- Success 169/200 (84.5%). Observed Safety Cost 177 (mean 0.885): danger 2, corner 87, blind 56, fragile 13, critical 19.
- Exact task manifest SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0`; original RUN_MANIFEST SHA256 `d94c3c905a023ac1fbfa31a7b866d9b84a9ca5b4c5d3147a179a220cfa5f1758`; episode_metrics.csv SHA256 `891973cbfc5d299ad76951efe6170dc12962729f9b991b81ba925577235a4296`; old ARTIFACT_INDEX SHA256 `96e10b0fa5a5e7c8edd04289267c9ca2c7cd63679963e4dab7ee01647dd3be19`.
- The 1203 indexed raw artifacts were verified at P2-002 handoff. These were not freshly read on the server by this PI GitHub control update; the next runner MUST rehash and validate them before model import and again for final comparison. Do not copy/modify original evidence to make an OFF output for 003.

## ON200-only candidate and authorization gaps

- Proposed science: same 200 exact ordered ObjectNavType TaskSpecs, checkpoint, DINO, seed=123, one worker, original action mapping/metrics/logger; only root Actor episode-boundary counter and attention K/V reset ON, Critic unchanged. No OFF rerun or model retraining.
- Proposed new working tree `/nvme2/user/qyy/SafeVLA_actor_logits_p2_on200_003_20261014`; output root `/nvme2/user/qyy/SafeVLA_p2_003_on200_results_20261014` MUST be absent before authorization and run.
- Required NEW Codex code SHA: **PENDING**. Prior accepted source 06cde9adeb91d47c062a11e72e90c7edf56fb398 is NOT an approved ON-only executable. Codex must prepare a separate candidate changing only new cycle identity, fail-closed Claim guard, ON-only launcher, independent output path and provenance/comparison glue; retain fixed scientific code and CPU evidence.
- Proposed command (NOT yet implemented): `bash diagnostics/actor_logits/run_p2_on200_only.sh`. Must not launch under existing 002 command or 002 Claim.
- Proposed limits: 1 GPU, 200 ON Episodes, no extra GPU equivalence or Debug. Active authorization: **none**. A separate final PI approval with reviewed source SHA, validity dates and CI SUCCESS is required, followed by a unique Codex Claim and its own CI SUCCESS.

## Network resilience, without bypassing authorization

The prior incident occurred during a new remote `git ls-remote` from ON evaluate.py, with `Proxy CONNECT aborted due to timeout`. A monitoring SSH session also reset; tunnel loss is plausible but NOT proven.

Codex must propose and document an independently maintained GitHub HTTPS path on the server: preferred direct network egress; otherwise a user-approved managed persistent proxy/tunnel service that remains alive after the interactive SSH session disconnects. Do not use an ephemeral reverse SSH `-R` tunnel whose lifetime is tied to a monitoring session.

Conduct CPU-only authenticated remote-read preflight from the actual process environment: verify exact `research-loop` head and signed-by-protocol authorization/Claim identity, independent GitHub Actions CI SUCCESS, repeated observations across a deliberate monitoring SSH disconnect/reconnect, and log timings/errors without recording secrets. Distinguish reachability of `github.com` and `api.github.com`. No proxy result or cached local state substitutes for the required live claim verification. Network failure => STOP before GPU/model import, and no automatic Episode retry.

## Cross-cycle validation contract

Before ON model import, validate P2-002 OFF run metrics, all 200 Episode statuses and 1203 raw index hashes against GitHub handoff; verify task manifest (including per-task raw spec hashes/order) and checkpoint/DINO provenance. Compare approved 003 ON outputs with immutable P2-002 OFF strictly by sample_id and order. The same TaskSpecs are matched, but online trajectories can differ, and cross-cycle wall-time/GPU/simulator effects remain possible confounds; report only descriptive treatment contrasts.

Future required handoff additionally includes `OFF_REUSE_VERIFICATION.json`, `NETWORK_PREFLIGHT.json` and `CROSS_CYCLE_COMPARISON.json` under the new cycle's handoff path. They do not exist yet and must never be fabricated. The raw ON tensors remain server-side with index+hash and Git-readable summaries. The waiver remains `NOT_RUN_PI_WAIVED`, not a real GPU equivalence PASS.

## PI decision still outstanding

This is a **recovery draft only**. Final PI must review and freeze new ON-only implementation SHA, protocol/transport, no-write guarantees, resource/date limits and CI before authorizing any Claim or Episode. **Do not claim or start GPU now.**
