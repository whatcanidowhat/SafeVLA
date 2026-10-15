# P2-004 — ON200-only recovery via user-managed TLS-verified proxy

Experiment ID: EXP-ACTOR-RESET-ON200-004
Status: DRAFT
Cycle ID: actor-reset-on200-off002-reuse-004-20261015

**PI_REVIEW / NOT_AUTHORIZED. This is a preparation brief, not GPU permission.** Prior P2-003 terminated BLOCKED on a 60.07s *direct GitHub* timeout before any GPU/model/ON Episode. PI accepted terminal handoff at [8592d70d6ba714aacce59b05f9208edb154e874d](https://github.com/whatcanidowhat/SafeVLA/commit/8592d70d6ba714aacce59b05f9208edb154e874d). User approved P2-003 GPU previously; that specific permission/Claim cannot be reused here. A **new explicit final human GPU authorization** is required for P2-004.

## Fixed scientific experiment

- Keep the **historical OFF200** from `/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200` completely read-only: 200/200 Episodes, 19,270 Actor decisions, SR 169/200=84.5%, observed Safety Cost total 177. Repeat strict byte/index/hash verification at actual new ON pre-model boundary and final comparison; do NOT rerun OFF200.
- Launch **ON200 only**, same ordered 200 ObjectNavType TaskSpecs, B0 checkpoint/DINO, seed 123, one worker, sampling/augmentation, horizon 600. Sole scientific treatment: reset root Actor episode-boundary time counter and decoder K/V, Critic unchanged. Do not edit model, official metrics, logger or TaskSpecs.
- Cross-cycle OFF/ON is **TaskSpec matched but not same-input causal**. Report runtime/GPU/simulator and stochastic-trajectory confounds, SR/Cost differences, uncertainty, early END and corner oscillation.

## Codex CPU-only binding task — one minimal source commit

- Base accepted transport-only candidate: [d9e092eba73da7bdc64683a4de45d0c12aaeec2d](https://github.com/whatcanidowhat/SafeVLA/commit/d9e092eba73da7bdc64683a4de45d0c12aaeec2d), 7 focused CPU tests PASS; changes proxy routing, TLS and real GitHub Claim/CI checks, not scientific execution bodies.
- Candidate still contains **old 003** constants/paths. Rebind experiment identity `EXP-ACTOR-RESET-ON200-004`, cycle `actor-reset-on200-off002-reuse-004-20261015`, new control target, output, evidence and session witness roots ONLY. Final 004 **executable source SHA PENDING** until Codex submits a new exact commit and PI reviews it. Maintain original future run command `bash diagnostics/actor_logits/run_p2_on200_only.sh`.
- Execution worktree: `/nvme2/user/qyy/SafeVLA_actor_logits_p2_on200_004_20261015`
- ON result root (must be absent): `/nvme2/user/qyy/SafeVLA_p2_004_on200_results_20261015`
- Launch evidence root (must be absent): `/nvme2/user/qyy/SafeVLA_p2_004_launch_20261015`
- Claim-session reports: `/nvme2/user/qyy/SafeVLA_p2_004_network_claim_20261015`
- Historical OFF root (do not modify): `/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200`

## Transport and hard controls

- Explicit GitHub proxy `http://127.0.0.1:7899` through user's Windows SSH reverse tunnel to Clash. Keep Git `http.sslVerify=true` and Python TLS certificate/hostname verification; no ignored certificate, cached CI substitutes or bypass of the real remote HEAD/Claim/CI.
- Use existing Claim-bound independent session readiness and preflight with a truthful FAIL/STOP if proxy unavailable. **No open-ended network engineering** and no debug/model Episodes before approval.
- Proposed **not yet granted** resource envelope: one available physical GPU 0–3, <=200 ON Episodes, <=120000 decisions, <=90 GiB result storage, <=12h queue, new 24h authorization expiry if human approves; preserve shared-GPU admission/stop guards. No GPU logger equivalence replay; `NOT_RUN_PI_WAIVED` stays a waiver.

## Release and authorization sequence

1. Codex prepares **only** final 004 code identity/paths and scoped CPU tests, publishes exact SHA and STOPs. No Claim or GPU.
2. PI checks final SHA/science parity and GitHub evidence. Human separately gives **explicit P2-004 GPU approval**.
3. PI pins executable source in `research/P2_CODE_SHA.txt`, changes both designs to APPROVED and LOOP_STATE to APPROVED_FOR_CODEX with new validity window (control approval A). Wait **A CI SUCCESS**.
4. Codex/authorized Executor creates **one unique Claim B** and waits **B CI SUCCESS**; one-shot P2-004 ON200 launcher with real-time proxy TLS checks, OFF data hashes and GPU admission.
5. Validate new ON200 and compare to old OFF200 descriptively; archive actual evidence to `research/handoffs/actor-reset-on200-off002-reuse-004-20261015/` and STOP. No retry on failed gates.

No 003 claim/approval inherited; no model/ON Episode before full new approval.
