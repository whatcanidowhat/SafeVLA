# PI P2 formal experiment — OFF200 vs Actor-only ON200, full logits

Experiment ID: EXP-ACTOR-RESET-OFF200-ON200-001
Status: DRAFT
Cycle ID: actor-reset-off200-on200-20261012

## Explicit PI direction (2026-10-12)

**Proceed directly to the 400-Episode experiment; GPU Logger equivalence test is explicitly waived.** This is a conscious PI risk acceptance, not a GPU equivalence PASS. Set `GPU_EQUIVALENCE=NOT_RUN_PI_WAIVED` in all manifests; do not forge, fill, or simulate a PASS evidence file.

## Frozen scientific protocol

- **OFF200:** original Actor episode-boundary behavior; **ON200:** only root Actor time_step_counter + each decoder layer attention K/V reset at Episode boundary; no Reward/Cost Critic reset or model/augmentation/RNG differences.
- Same accepted B0 source 2aa82559d272b5f888e53433e258914057f15bed, approved local DINO loading adapter, checkpoint 05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301; logger branch initial code bb71a65609f59e961049029805ec16c06be7240a.
- Each group 200 original ObjectNavType/minival TaskSpecs, same frozen full CSV sha 8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0; 1 worker, seed123, shuffle=true, greedy=false, test_augmentation=true, original horizon600. Preserve original official task success and pre-step five-component Safety Cost.
- Every live decision must have **lossless complete raw root Actor logits**, normalized logits, all action probabilities, sampled/executed action, task/episode/step and boundary root/critic state record. OFF and ON use identical Logger.
- No extra GPU preflight, no 600-pair offline replay, no OFF/OFF mini run, no test Episode. A real OFF200 is first executable scientific session after valid control claim, then ON200 only after OFF complete/validated.

Approval: DRAFT — awaiting PI reapproval for CODEX; OPERATOR unclaimed grant revoked and is no longer executable.

## CODEX execution and exact source (replacing human-operator route)

- PI switched back to **CODEX**, based on user's explicit request on 2026-10-13. The human `APPROVED_FOR_OPERATOR` grant was unclaimed and revoked at `cb44b00cdd9cbe421ab7cb64ae4c9ce392d1ea7c`; no human claim, no GPU work is implied by this protocol.
- Exact implementation code: `33ffa28000522e3715a0920bf3679c27e79b423e` on `feature/p2-actor-logits-logger-20261012`; also frozen in `research/P2_CODE_SHA.txt`. This has the **CODEX_RUNNING** control evidence check and one-shot `diagnostics/actor_logits/run_p2_full400.sh`. The code branch is not the control branch.
- After fresh PI approval commit **and its own CI SUCCESS**, CODEX alone must run `scripts/research_loop_claim.py` in the clean `research-loop` control tree, publish a single unique Claim, and wait for distinct Claim CI SUCCESS before launching any GPU/model/simulator process. CODEX must never run the human claim script or self-approve.
- In isolated `/nvme2/user/qyy/SafeVLA_actor_logits_p2_full400_20261012` execution worktree at exactly frozen code SHA, CODEX may run CPU-only static checks and the approved command **once**: `bash diagnostics/actor_logits/run_p2_full400.sh`. The launcher checks true CODEX claim CI and one free physical GPU from 0–3, then **OFF200 → integrity validation → ON200 → integrity validation → analyze** sequentially on the same GPU.
- **No GPU Logger equivalence test** by explicit PI waiver. The label is `NOT_RUN_PI_WAIVED` and is not PASS. No new OFF/OFF Gate, GPU replay, short episode dry-run, unapproved model edit or change to frozen task/metric/reset definition.
- On completion or STOP, CODEX must write the required review/handoff evidence in `research/handoffs/actor-reset-off200-on200-20261012/`, publish status AWAITING_PI_REVIEW or truthful terminal BLOCKED/INVALID/ABORTED, and STOP pending PI review.

Approved max **1 GPU, 400 Episode starts, 240000 online decisions, 168h experiment runtime, 100GiB total output, 12h CPU-only queue**, plus authorization expiry; no hidden retry or seed. If OFF incomplete or any guard fails, **do not start ON**. No additional P2 authorization invented by a shell variable.

Output control handoff includes `research/handoffs/actor-reset-off200-on200-20261012/RESULT_SUMMARY.md`, `research/handoffs/actor-reset-off200-on200-20261012/RUN_MANIFEST.json`, `research/handoffs/actor-reset-off200-on200-20261012/ARTIFACT_INDEX.json`, `research/handoffs/actor-reset-off200-on200-20261012/REVIEW_NOTES.md`, `research/handoffs/actor-reset-off200-on200-20261012/FROZEN_PROTOCOL.md`, `research/handoffs/actor-reset-off200-on200-20261012/LOGGER_WAIVER.md`, `research/handoffs/actor-reset-off200-on200-20261012/TASK_MANIFEST.csv`, `research/handoffs/actor-reset-off200-on200-20261012/OFF_ON_RESULTS.md`, `research/handoffs/actor-reset-off200-on200-20261012/VALIDATION_SUMMARY.md`, `research/handoffs/actor-reset-off200-on200-20261012/GPU_QUEUE_REPORT.json`; large raw logits/tensors/videos remain on server with hashes and referenced paths. CODEX publishes C handoff and STOP, PI separately reviews.

Prior P1R2 approval `a88d5ab1f29127753d5b6c1c42bf7b15f891b566` had max10 episodes, claim null and was revoked without execution in `763d7e76b72248b968f4f5d564185221da11d5b8`. Never use it for this experiment.


## PI CODEX source pin (2026-10-13)

Only exact source SHA `33ffa28000522e3715a0920bf3679c27e79b423e` is accepted for this CODEX executor. `research/P2_CODE_SHA.txt` is canonical. Historical manual source pin is superseded, without rewriting historical Git commits. The same science and all budget ceilings remain frozen.
