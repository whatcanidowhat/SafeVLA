# P2 ndarray-metrics Retry1 — Codex executor CANDIDATE (NOT AUTHORIZED)

This is a fresh, **Codex-only** candidate branch for next experiment `EXP-ACTOR-RESET-OFF200-ON200-002`, cycle `actor-reset-off200-on200-20261013-retry1`. It cannot run under original P2 claim `e3f688f832924a1dbb6a105b117b5551`.

Original OFF200 failed after 27 policy decisions, Episode order 0, at Logger end_episode with `TypeError: Unsupported JSON type: ndarray`. Preserve the full original results/episode shards, do not overwrite, delete, re-run or resume the prior claim. First finalize the old incident as a truthfully evidenced terminal BLOCKED handoff, wait for its own CI, and require separate PI acknowledgement to PI_REVIEW. If missing archived evidence or any remaining process is detected, STOP and report, do not bypass checks or fabricate artifacts.

Source parent fix commit: `49d434f42e2d636edeee5c2cd379df198a4bd41f`; NumPy ndarray metadata converted to JSON lists; raw Actor logits continue lossless NPZ. Zero-GPU CPU regression check passed. This candidate switches the *next-cycle* claim gate from OPERATOR to CODEX, requiring a fresh PI-approved unique CODEX_RUNNING Claim, exact new approved code SHA, and a distinct Claim CI SUCCESS. These are **not** obtained yet.

Science unchanged: OFF200 original Actor state, ON200 only root Actor time_step_counter and decoder K/V reset, same fixed TaskSpecs, one worker seed123, greedy=false, test_augmentation, original SR/five Safety Cost, full raw root Actor logits, critics unchanged. `GPU_EQUIVALENCE=NOT_RUN_PI_WAIVED` stays truthful; the PI may repeat this waiver for the next cycle, never claim GPU equivalence PASS.

Future fresh result directory `/nvme2/user/qyy/SafeVLA_p2_full400_results_20261013_retry1`; frozen source must be newly approved, NOT automatically run. Codex may prepare CPU-only checks, then launch only after all review/authorization/Claim gates pass. No unauthorized GPU preflight/retry/resume or shell bypass. Record every outcome and hand off to PI.
