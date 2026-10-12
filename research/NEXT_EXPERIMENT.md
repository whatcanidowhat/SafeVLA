# PI P2 formal experiment — OFF200 vs Actor-only ON200, full logits

Experiment ID: EXP-ACTOR-RESET-OFF200-ON200-001
Status: APPROVED
Cycle ID: actor-reset-off200-on200-20261012

## Explicit PI direction (2026-10-12)

**Proceed directly to the 400-Episode experiment; GPU Logger equivalence test is explicitly waived.** This is a conscious PI risk acceptance, not a GPU equivalence PASS. Set `GPU_EQUIVALENCE=NOT_RUN_PI_WAIVED` in all manifests; do not forge, fill, or simulate a PASS evidence file.

## Frozen scientific protocol

- **OFF200:** original Actor episode-boundary behavior; **ON200:** only root Actor time_step_counter + each decoder layer attention K/V reset at Episode boundary; no Reward/Cost Critic reset or model/augmentation/RNG differences.
- Same accepted B0 source 2aa82559d272b5f888e53433e258914057f15bed, approved local DINO loading adapter, checkpoint 05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301; logger branch initial code bb71a65609f59e961049029805ec16c06be7240a.
- Each group 200 original ObjectNavType/minival TaskSpecs, same frozen full CSV sha 8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0; 1 worker, seed123, shuffle=true, greedy=false, test_augmentation=true, original horizon600. Preserve original official task success and pre-step five-component Safety Cost.
- Every live decision must have **lossless complete raw root Actor logits**, normalized logits, all action probabilities, sampled/executed action, task/episode/step and boundary root/critic state record. OFF and ON use identical Logger.
- No extra GPU preflight, no 600-pair offline replay, no OFF/OFF mini run, no test Episode. A real OFF200 is first executable scientific session after valid control claim, then ON200 only after OFF complete/validated.

Approval: APPROVED_FOR_OPERATOR — execute only after manual claim and its independent green control CI.

## Development exception and actual launch

The currently submitted `diagnostics/actor_logits/evaluate.py` hard-requires an equivalence PASS JSON. CODEX must make a **minimal explicit PI-waiver code patch**, eliminate that obsolete requirement only for the exact approved P2 scope, and truthfully annotate `NOT_RUN_PI_WAIVED`; update tests/docs and implement single-sequential-launch `diagnostics/actor_logits/run_p2_full400.sh` in a **new isolated experiment worktree**, not B0 nor control worktree. Commit and hash changes; no fake PASS report. Require approval SHA/unique claim/green CI in executor/launch check; `--execute-authorized` alone is insufficient.

**After control approval + its green CI, and CODEX claim + its green CI**, prepare code and CPU-only syntax/unit checks, use the bounded non-destructive GPU queue 0–3, then execute **OFF200 → validate complete → ON200 → validate/compare** on one selected physical GPU. The runner must honor no-foreign-PID, used<=4000MiB/util<=30%/free>6823.390767MiB 3 samples x30s, max12h queue. Optional short own GPU hold only with lab compliance, default disabled. Never terminate another user's PID or switch physical GPU after model starts.

Approved max **1 GPU, 400 Episode starts, 240000 online decisions, 168h experiment runtime, 100GiB total output, 12h CPU-only queue**, plus authorization expiry; no hidden retry or seed. If OFF incomplete or any guard fails, **do not start ON**. No additional P2 authorization invented by a shell variable.

Output control handoff includes `research/handoffs/actor-reset-off200-on200-20261012/RESULT_SUMMARY.md`, `research/handoffs/actor-reset-off200-on200-20261012/RUN_MANIFEST.json`, `research/handoffs/actor-reset-off200-on200-20261012/ARTIFACT_INDEX.json`, `research/handoffs/actor-reset-off200-on200-20261012/REVIEW_NOTES.md`, `research/handoffs/actor-reset-off200-on200-20261012/FROZEN_PROTOCOL.md`, `research/handoffs/actor-reset-off200-on200-20261012/LOGGER_WAIVER.md`, `research/handoffs/actor-reset-off200-on200-20261012/TASK_MANIFEST.csv`, `research/handoffs/actor-reset-off200-on200-20261012/OFF_ON_RESULTS.md`, `research/handoffs/actor-reset-off200-on200-20261012/VALIDATION_SUMMARY.md`, `research/handoffs/actor-reset-off200-on200-20261012/GPU_QUEUE_REPORT.json`; large raw logits/tensors/videos remain on server with hashes and referenced paths. CODEX publishes C handoff and STOP, PI separately reviews.

Prior P1R2 approval `a88d5ab1f29127753d5b6c1c42bf7b15f891b566` had max10 episodes, claim null and was revoked without execution in `763d7e76b72248b968f4f5d564185221da11d5b8`. Never use it for this experiment.


## Manual source-code pin (PI)

Operator implementation is **exact git commit** `637b3aee56fd5dacd0013eded65256ff714e2e53` on `feature/p2-actor-logits-logger-20261012`. PI approves the unaltered, clean detached or worktree checkout at this SHA only, including logger, explicit waiver entry, static checks, and `diagnostics/actor_logits/run_p2_full400.sh`. The SHA is also recorded in `research/P2_MANUAL_CODE_SHA.txt`. No CODEX executor and no fake GPU-equivalence PASS.
