# P2 retry1 after the original ndarray metrics crash — NOT EXECUTION AUTHORIZATION

This branch is a proposed **human-operator** source tree for a future fresh cycle, not a continuation of the old claim.

- Source parent: fixed Logger JSON metadata codec commit 49d434f42e2d636edeee5c2cd379df198a4bd41f (4 CPU regression tests PASS).
- Scientific comparison unchanged: original OFF200 and root Actor-only Reset ON200, frozen identical ordered ObjectNavType TaskSpecs, seed123, greedy=false, augmentation, 1 worker, full raw Actor logits, Reward/Cost Critic unchanged.
- Identifiers for the new fresh cycle are EXP-ACTOR-RESET-OFF200-ON200-002 / actor-reset-off200-on200-20261013-retry1.
- Future runner is diagnostics/actor_logits/run_p2_full400.sh, writing **only fresh** /nvme2/user/qyy/SafeVLA_p2_full400_results_20261013_retry1.
- It must not execute until the prior claimed P2 has been truthfully marked BLOCKED, PI has acknowledged the terminal result, a distinct cycle 002 has been approved with this exact code commit SHA and green CI, and the user manually claims OPERATOR_RUNNING with a separate successful claim CI.
- GPU Logger equivalence remains NOT_RUN_PI_WAIVED subject to explicit fresh PI approval in the new cycle; not PASS.
- Code-level change from the old P2 treatment is NumPy ndarray serialization for official metrics. New authority checks only bind to the new cycle. No scientific model/Actor/critic/loader source drift.
- **DO NOT** hot-patch the original results directory/worktree, force push research-loop, reset the old claim, or run under the original 001 claim.

Current branch is a **preparation proposal**; these instructions do not grant automatic authority to execute GPU or benchmark Episodes.
