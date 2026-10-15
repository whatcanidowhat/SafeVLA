# PI handoff to Codex — P2-004 CPU-only code rebinding

**Not a GPU or Claim authorization.** One task only: derive final unclaimed ON200 P2-004 executable source from proxy candidate `d9e092eba73da7bdc64683a4de45d0c12aaeec2d`. Keep Actor, Critic, scientific evaluate body, TaskSpecs, loader, logger, metrics, checkpoint, OFF200 untouched.

Change **only** old 003 cycle/experiment constants, ON output root, launcher/evidence/network-witness roots, fixed `execution_worktree` and any identity-bound text/test cases required to pass CPU control parsing. New experiment `EXP-ACTOR-RESET-ON200-004`; cycle `actor-reset-on200-off002-reuse-004-20261015`; worktree `/nvme2/user/qyy/SafeVLA_actor_logits_p2_on200_004_20261015`; result `/nvme2/user/qyy/SafeVLA_p2_004_on200_results_20261015`; evidence `/nvme2/user/qyy/SafeVLA_p2_004_launch_20261015`; network witnesses `/nvme2/user/qyy/SafeVLA_p2_004_network_claim_20261015`. The pre-existing Python module names may remain; no risky broad rename required.

Review why the existing `github_opener()` sets global `os.environ['NO_PROXY']`; keep an explicit verified loopback proxy while avoiding unnecessary process-global mutation if this can be done safely in a tiny targeted fix. Do not widen scope. Maintain HTTPS certificate and hostname checks, live real remote HEAD + exact Claim CI check, no retry and failure before GPU.

Report exact new code SHA, minimal diff/file listing, focused CPU tests and science parity, whether code/checkpoint and OFF paths match. **Do not Claim, start GPU, run episodes, rerun OFF200, or start persistent services.**

Next action after code submission: independent PI review and **separate human final GPU authorization**. Prior P2-003's authority is terminal.
