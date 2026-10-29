# P2-004 — authorized ON200-only recovery using TLS-verified user proxy

Experiment ID: EXP-ACTOR-RESET-ON200-004
Status: DRAFT
Cycle ID: actor-reset-on200-off002-reuse-004-20261015

## Historical PI authorization (expired; no execution permission)

**APPROVED_FOR_CODEX, NOT YET CLAIMED.** User explicitly confirmed final P2-004 GPU authorization on 2026-10-15. This PI approval is bound to the exact scientific executable [`67c7051ed4287968897282f915b8b076851562a1`](https://github.com/whatcanidowhat/SafeVLA/commit/67c7051ed4287968897282f915b8b076851562a1), staged at `research/P2_CODE_SHA.txt`. The code-stage control CI [#38690170141](https://github.com/whatcanidowhat/SafeVLA/actions/runs/38690170141) succeeded prior to authorization.

Resource ceiling: **one physical GPU 0–3 only if safe occupancy checks pass**, **200 ON Episodes**, up to **120,000 policy decisions** under horizon 600, 12h CPU GPU queue and 90GiB output guard. Absolute PI permission window: **2026-10-15T06:08:16Z to 2026-10-16T06:08:16Z UTC** (24h). The launcher's earlier fallback 168h wall limit is superseded by the 24h authorization expiry enforced during queue and runtime.

The PI approval commit itself is **NOT** an executable Claim. Executor must wait for this approval commit A CI SUCCESS, then publish precisely one fresh P2-004 Claim B using the established control protocol, then wait for distinct B CI SUCCESS and exact live remote HEAD before any work. Do not reuse P2-003 Claim `fa4877c4550be75e9726172421d76468bcc920c5`.

## Scientific program (frozen)

- Reuse historical, validated **P2-002 OFF200** immutable files at `/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200`: 200 Episodes, 19,270 Actor decisions, SR169/200=84.5%, observed Safety Cost 177. Do NOT rerun OFF or modify those bytes. At new ON pre-model and final comparison, verify all 1,203 indexed OFF raw files and TaskSpec manifest with the real strict validator.
- Exactly **one new ON200** on the identical 200 ordered ObjectNavType TaskSpecs (CSV SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0`), same B0 checkpoint/DINO, seed=123, worker=1, stochastic sampling and test augmentation, horizon=600. Only root Actor Episode-boundary time counter and decoder K/V reset ON; Critic unchanged. No retraining, debug Episodes, OFF rerun or GPU logger equivalence replay. Logger waiver remains `NOT_RUN_PI_WAIVED`, never PASS.
- Source is fixed at `67c7051ed4287968897282f915b8b076851562a1`, with cycle/experiment and fresh output/evidence bindings, GitHub `http://127.0.0.1:7899` proxy and retained Git+Python HTTPS TLS certificate/hostname checks. No source edits after approval.
- Compare OFF/ON matched by TaskSpec and order **descriptively**. Different execution cycles and divergent trajectories are confounds; no identical-input causal claims from diverged observations. Report official SR, five-component Safety Cost, early END/corner oscillation proxies, Actor outputs and uncertainty.

## Execution identities and safety

- Control checkout: `/nvme2/user/qyy/SafeVLA_loop_control_p2_codex_20261013`
- New clean source worktree: `/nvme2/user/qyy/SafeVLA_actor_logits_p2_on200_004_20261015`
- New absent-before-first-run output root: `/nvme2/user/qyy/SafeVLA_p2_004_on200_results_20261015`
- New absent-before-first-run launch evidence: `/nvme2/user/qyy/SafeVLA_p2_004_launch_20261015`
- New 004 Claim-specific network witnesses: `/nvme2/user/qyy/SafeVLA_p2_004_network_claim_20261015`
- Only authorized command: `bash diagnostics/actor_logits/run_p2_on200_only.sh`, run once in the clean exact source worktree and only after successful unique Claim CI.
- User maintains SSH reverse proxy to Windows Clash. Before GPU/model import retain the source's actual fresh two independent SSH-session Claim-specific receipts, live remote GitHub HEAD/CI checks through TLS, source/weight/TaskSpec/OFF hashes, vacant output roots and safe shared-GPU admission. Proxy loss, missing CI, expiry, hash drift, GPU occupancy or any guard failure => **STOP before unsafe compute**, preserve truthful evidence, no automatic retry.

## Handoff

After ON200 completes or terminates BLOCKED/INVALID, publish genuine Git-readable result artifacts to `research/handoffs/actor-reset-on200-off002-reuse-004-20261015/` following the existing PI–Codex protocol, and STOP for PI review. No autonomous new experiment or permission extension.

**This is a one-shot approval: no Claim, GPU process or Episode was started by the PI control commit.**
