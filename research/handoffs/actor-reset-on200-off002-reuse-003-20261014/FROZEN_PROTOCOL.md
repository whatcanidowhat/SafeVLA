# P2-003 recovery proposal — reuse valid OFF200, execute ON200 only

Experiment ID: EXP-ACTOR-RESET-ON200-003
Status: APPROVED
Cycle ID: actor-reset-on200-off002-reuse-003-20261014

**APPROVED_FOR_CODEX / GPU AUTHORIZED BY USER 2026-10-15; NOT YET CLAIMED.** Human explicitly authorized P2-003 ON200-only, and PI approved the pinned implementation. The approval is conditional on this approval commit's CI SUCCESS, then a unique Codex Claim commit with its own CI SUCCESS, then all original fail-closed CPU control/network/OFF hash checks before GPU/model import. Active *authorization limit*: 1 GPU / 200 Episodes; actual consumption remains 0 GPU / 0 ON Episodes until Codex starts.

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
- Approved command **after approval CI and unique Claim CI only**: `bash diagnostics/actor_logits/run_p2_on200_only.sh`. New 003 Claim and distinct Claim CI still required. Never run old 002 launcher or reuse 002 Claim.
- **Formal resource authorization**: 1 GPU on physical 0–3 if the pre-existing shared GPU admission guard passes; 200 ON Episodes; 120,000 maximum policy decisions (horizon 600); absolute validity 2026-10-15T01:49:16Z through 2026-10-16T01:49:16Z UTC (24h); CPU GPU queue capped 12h; launcher disk guard at 96636764160 bytes (90 GiB). Runtime clock includes queue and execution in a 24h absolute permission interval, superseding the code's dormant 168h long-stop. No other users' processes affected; no repeat, resume, OFF rerun, GPU logger equivalence test or Debug.

## Network resilience, without bypassing authorization

The prior incident occurred during a new remote `git ls-remote` from ON evaluate.py, with `Proxy CONNECT aborted due to timeout`. A monitoring SSH session also reset; tunnel loss is plausible but NOT proven.

The reviewed 003 candidate uses direct HTTPS and rejects inherited SSH proxy settings. Direct connectivity has been intermittent, so the existing CPU-only live remote Claim+CI, cross-session and repeated preflight checks are **mandatory at actual startup**; a failed check STOPs before GPU/model import. We do **not** require a separate long-term network-infrastructure project as a precondition to final human authorization. Never bypass the remote Claim/CI guard or silently use a session-bound SSH `-R` proxy.

At execution time run the candidate's existing CPU-only guard in the exact launcher environment: live `research-loop` HEAD, distinct approved Claim/CI evidence, new cross-SSH-session direct-read receipts (<=30 min old), and three preflight network observations. No cache, bypass, token disclosure, or retrospective replacement of these checks. Failure => STOP before GPU/model import and truthful BLOCKED handoff, not an automatic Episode retry.

## Cross-cycle validation contract

Before ON model import, validate P2-002 OFF run metrics, all 200 Episode statuses and 1203 raw index hashes against GitHub handoff; verify task manifest (including per-task raw spec hashes/order) and checkpoint/DINO provenance. Compare approved 003 ON outputs with immutable P2-002 OFF strictly by sample_id and order. The same TaskSpecs are matched, but online trajectories can differ, and cross-cycle wall-time/GPU/simulator effects remain possible confounds; report only descriptive treatment contrasts.

Future required handoff additionally includes `OFF_REUSE_VERIFICATION.json`, `NETWORK_PREFLIGHT.json` and `CROSS_CYCLE_COMPARISON.json` under the new cycle's handoff path. They do not exist yet and must never be fabricated. The raw ON tensors remain server-side with index+hash and Git-readable summaries. The waiver remains `NOT_RUN_PI_WAIVED`, not a real GPU equivalence PASS.

## PI decision still outstanding

**FINAL PI APPROVAL RECORDED, CONDITIONAL GATES REMAIN.** User explicitly authorized GPU; exact new ON-only source 013508992c07443d2cf25e9d2099916b9ba57bd8 pinned and approved. Codex may Claim only after this approval commit has SUCCESS control CI; run only after its unique Claim CI SUCCESS, clean exact code tree, absent output roots and live fail-closed remote Claim/CI + OFF byte recheck. No GPU may start merely because this document says APPROVED. No bypass/automatic retry. Human permission expires 2026-10-16T01:49:16Z UTC.

## 2026-10-15 — scoped PI code acceptance, NOT execution authorization

PI reviewed candidate `013508992c07443d2cf25e9d2099916b9ba57bd8` and staged it in `research/P2_CODE_SHA.txt`. The prior unimplemented-code gap is closed. Intermittent direct network remains an *execution-time availability risk*: start only after the actual claim-specific live guard passes; failure stops before GPU. No blanket requirement for a long-term proxy project. Historical statement from staging: design was DRAFT and state was PI_REVIEW / NOT_AUTHORIZED. Subsequent 2026-10-15 user GPU authorization supersedes that draft, and current formal state is APPROVED_FOR_CODEX. The candidate's 168h/100GiB/12h proposed physical limits are **not approved**; final human authorization must establish definite wall/deadline/caps (prefer a bounded ON200 run, not an indefinite queue). Revalidate OFF raw 1,203 files at pre-model and final comparison. Historical OFF/ON across cycles is descriptive, not same-input causal proof.

## 2026-10-15 — binding human + PI GPU authorization (current and controlling)

Human explicitly approved a formal GPU ON200 run on 2026-10-15; PI binds this one and only one experiment to the exact reviewed source SHA `013508992c07443d2cf25e9d2099916b9ba57bd8` and command `bash diagnostics/actor_logits/run_p2_on200_only.sh`. Approved max GPU=1, max ON Episodes=200, max steps=120000, start at 2026-10-15T01:49:16Z UTC and absolute expiry 2026-10-16T01:49:16Z UTC, 12h max shared GPU CPU queue, 90 GiB output guard (the existing launcher enforces the bound), valid only on physical GPUs 0–3 meeting shared resource admission. No OFF replay; frozen historical OFF200 is READ_ONLY. No changes to policy/reset/Critic/logger/metrics/TaskSpecs. Previous controls including cross-session GitHub live Claim/CI reads and immediate pre-model OFF 1203-file hash validation remain **mandatory**, with STOP before GPU on failure. No network service project required. This authorizing commit is A; Codex must **await CI SUCCESS for A** then publish its own unique Claim B and **await distinct CI SUCCESS for B** before executing. No Claim or compute has been created by this PI approval. No automatic retry or renewal. After ON success or BLOCKED/INVALID failure, publish actual handoff in GitHub and STOP.
