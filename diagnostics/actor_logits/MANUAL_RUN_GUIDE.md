# P2 full400 CODEX executor — immutable science and explicit PI waiver

Status: CODEX execution code candidate only. Do not launch before new PI control approval and a unique green CODEX claim CI.

Source pinned via `research/P2_CODE_SHA.txt` on the control branch. The accepted full-logging worktree uses the same OFF/ON evaluator and lossless logger, and only the actor-reset flag differs between the two official 200-Episode runs.

This command does not perform the 600-pair GPU Logger equivalence test, by explicit PI waiver `NOT_RUN_PI_WAIVED`, and must never fabricate a PASS report.

For the authorised Codex task:
1. Read `research/LOOP_STATE.json`, `research/NEXT_EXPERIMENT.json`, `research/HANDOFF_PROTOCOL.md`, `research/P2_CODE_SHA.txt` on `research-loop`.
2. In an unmodified control tree at approval HEAD, run `python scripts/research_loop_claim.py` EXACTLY ONCE. After push wait for distinct claim CI SUCCESS. If transport or CI fails, STOP and escalate to PI. Never self-approve or manually mark a claim.
3. Use exact clean source commit in the frozen independent execution worktree and **record** all real local runtime provenance and GPU usage. Run `bash diagnostics/actor_logits/run_p2_full400.sh` once, on a single allowed shared physical GPU.
4. The launcher waits CPU-only for eligible GPU 0-3; OFF200 first, validate all logits and original metrics, ON200 second only after OFF PASS; no retries or parallel sessions; respect authorization expiry, budgets and unknown GPU/report failures.
5. Commit accurate small handoff evidence to the GitHub control branch, keep large logs on server with hashes, status `AWAITING_PI_REVIEW` or terminal failure status, then STOP.

Do not execute from the `research-loop` control tree. Do not edit B0, the old result folders, renderer mapping, action selection, seed, 200 TaskSpec manifest or Critic state.
