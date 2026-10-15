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
- Reviewed ON-only executable candidate **PI STAGED**: `013508992c07443d2cf25e9d2099916b9ba57bd8` (six-path code diff, 41 CPU tests reported PASS). This is an *exact source pin*, not a GPU/Claim approval. Actor/reset/Critic, Logger/storage and official science metric math remain unchanged relative to frozen P2-002 source.
- Code-reviewed command **exists but is NOT YET AUTHORIZED**: `bash diagnostics/actor_logits/run_p2_on200_only.sh`. New 003 Claim and distinct Claim CI still required. Never run old 002 launcher or reuse 002 Claim.
- Proposed limits: 1 GPU, 200 ON Episodes, no extra GPU equivalence or Debug. Active authorization: **none**. A separate final PI approval with reviewed source SHA, validity dates and CI SUCCESS is required, followed by a unique Codex Claim and its own CI SUCCESS.

## Network resilience, without bypassing authorization

The prior incident occurred during a new remote `git ls-remote` from ON evaluate.py, with `Proxy CONNECT aborted due to timeout`. A monitoring SSH session also reset; tunnel loss is plausible but NOT proven.

The reviewed 003 candidate uses direct HTTPS and rejects inherited SSH proxy settings. Direct connectivity has been intermittent, so the existing CPU-only live remote Claim+CI, cross-session and repeated preflight checks are **mandatory at actual startup**; a failed check STOPs before GPU/model import. We do **not** require a separate long-term network-infrastructure project as a precondition to final human authorization. Never bypass the remote Claim/CI guard or silently use a session-bound SSH `-R` proxy.

At execution time run the candidate's existing CPU-only guard in the exact launcher environment: live `research-loop` HEAD, distinct approved Claim/CI evidence, new cross-SSH-session direct-read receipts (<=30 min old), and three preflight network observations. No cache, bypass, token disclosure, or retrospective replacement of these checks. Failure => STOP before GPU/model import and truthful BLOCKED handoff, not an automatic Episode retry.

## Cross-cycle validation contract

Before ON model import, validate P2-002 OFF run metrics, all 200 Episode statuses and 1203 raw index hashes against GitHub handoff; verify task manifest (including per-task raw spec hashes/order) and checkpoint/DINO provenance. Compare approved 003 ON outputs with immutable P2-002 OFF strictly by sample_id and order. The same TaskSpecs are matched, but online trajectories can differ, and cross-cycle wall-time/GPU/simulator effects remain possible confounds; report only descriptive treatment contrasts.

Future required handoff additionally includes `OFF_REUSE_VERIFICATION.json`, `NETWORK_PREFLIGHT.json` and `CROSS_CYCLE_COMPARISON.json` under the new cycle's handoff path. They do not exist yet and must never be fabricated. The raw ON tensors remain server-side with index+hash and Git-readable summaries. The waiver remains `NOT_RUN_PI_WAIVED`, not a real GPU equivalence PASS.

## PI decision still outstanding

**Code review completed; execution still DRAFT.** The exact candidate SHA is pinned, and 41 CPU tests/validated OFF200 are recorded as Codex evidence. A separate explicit human final GPU authorization, approved resource/date limits, PI approval commit+CI and new Claim+CI must precede any GPU/ON Episode. Direct HTTPS will be tested *live* under the existing fail-closed guard; no open-ended network remediation is required. **Do not Claim or start GPU now.**

## 2026-10-15 — scoped PI code acceptance, NOT execution authorization

PI reviewed candidate `013508992c07443d2cf25e9d2099916b9ba57bd8` and staged it in `research/P2_CODE_SHA.txt`. The prior unimplemented-code gap is closed. Intermittent direct network remains an *execution-time availability risk*: start only after the actual claim-specific live guard passes; failure stops before GPU. No blanket requirement for a long-term proxy project. Current design remains DRAFT and LOOP_STATE remains PI_REVIEW / NOT_AUTHORIZED. The candidate's 168h/100GiB/12h proposed physical limits are **not approved**; final human authorization must establish definite wall/deadline/caps (prefer a bounded ON200 run, not an indefinite queue). Revalidate OFF raw 1,203 files at pre-model and final comparison. Historical OFF/ON across cycles is descriptive, not same-input causal proof.
