# Formal P2-004 human + PI GPU authorization (2026-10-15)

**User consent:** The user expressly answered “确认” to a proposal for P2-004 up to one GPU, 24h authorization, one ON200 max 200 Episodes and immutable OFF200 reuse. PI's decision: **APPROVE ONE P2-004 ON200**. No implicit authorization for repeat/other experiment.

Approved source: [`67c7051ed4287968897282f915b8b076851562a1`](https://github.com/whatcanidowhat/SafeVLA/commit/67c7051ed4287968897282f915b8b076851562a1); preparatory control HEAD `693ed4870dd476c692200b0e00c220161d65f204`, CI [#38690170141](https://github.com/whatcanidowhat/SafeVLA/actions/runs/38690170141) SUCCESS.

Cycle `actor-reset-on200-off002-reuse-004-20261015`, experiment `EXP-ACTOR-RESET-ON200-004`. Control v84 PI_REVIEW / NOT_AUTHORIZED -> v85 APPROVED_FOR_CODEX / APPROVED, next_actor CODEX, no instruction commit or Claim at approval. Human/PI absolute approval 2026-10-15T06:08:16Z — 2026-10-16T06:08:16Z UTC (24h), physical GPUs 0–3 subject to safe GPU admission, max 1 GPU and 200 Episodes, theoretical 120000 decisions at max horizon600, 12h CPU queue, 90GiB outputs.

**Mandatory chain:** wait THIS authorization commit's control CI SUCCESS, then create a unique 004 Claim B through the existing control script, publish B atomically/non-force, verify remote exact SHA and independent successful B control CI, then provide source-required fresh cross-session claim-bound real TLS proxy HEAD/CI probes and other fail-closed runtime checks. No GPU/model import/Episode before all prerequisites. The user's managed Windows SSH reverse tunnel to `http://127.0.0.1:7899` must remain available; loss fails closed. Do not regenerate/reuse P2-003 Claim or bypass the guard.

**Science preserved:** ON200 only; immutable validated P2-002 OFF200 root and 1203-file rehash; exact 200 TaskSpecs/checkpoint/DINO/science source hashes, no Critic/Actor/scientific logic change other than original reset treatment; validate official SR/safety components and descriptive cross-cycle comparison with stochasticity limitations. GPU equivalence `NOT_RUN_PI_WAIVED`.

Execution worktree `/nvme2/user/qyy/SafeVLA_actor_logits_p2_on200_004_20261015`, fresh output `/nvme2/user/qyy/SafeVLA_p2_004_on200_results_20261015`, evidence `/nvme2/user/qyy/SafeVLA_p2_004_launch_20261015`, network-witness root `/nvme2/user/qyy/SafeVLA_p2_004_network_claim_20261015`. Launch **once** only through approved `bash diagnostics/actor_logits/run_p2_on200_only.sh`. On failure STOP and publish actual handoff; no retry, no off re-execution, no code substitutions.

No Claim publication or GPU launch was performed by this PI approval.
