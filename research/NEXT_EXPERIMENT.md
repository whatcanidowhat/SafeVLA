# FAST-TRACK Actor memory unmask successor

Experiment ID: EXP-ACTOR-MEMORY-UNMASK-002
Status: DRAFT
Cycle: actor-memory-unmask-002-20261029
Frozen candidate source: 2689081ec324c67a35f2a4692cfa009826fcf9d4

Measure delta_SR=SR_OFF−SR_ON and delta_mean_cost=Cost_OFF−Cost_ON on the same ordered200 tasks. Run MASK_ON200→MASK_OFF200, one worker,seed123, stochastic sampling/test augmentation, same checkpoint and allocated GPU UUID. Both Actor reset OFF; max_steps500/cache500/horizon600. Keep Actor/Critic/prev_actions/cache/counter/rollover and official success/five-cost rules unchanged. Limits:1 GPU,400 starts,240000 decisions,12h runtime,90GiB output.

Only execution bindings differ from base 1a77276ae8ee08cb34c67492a2a77c9a8c2cc0d8: batch replaces inaccessiblev100; fresh experiment/cycle/worktree/output/log names. All205 other source Git entries are byte-identical, including scientific code. Reuse existing38-check/417-load CPU evidence; no new CPU/logits or standalone GPU logger study.

Old001 terminal d670852bf08ed3d14d55dc1e4c631ccf678c6eeb, original Claim81219671 and its artifacts remain historical. Human-authorized PI acknowledgement e6d2b20a152a760b78d30549178f3f51378c0a63 clears old authorization; no old Claim/job/output/waiver is reusable.

New PI approval A must explicitly accept the common target dependencies/rendering and scoped logger waiver for this experiment. Proposed flags are in JSON; current executable flags remain PI_DECISION_REQUIRED. No execution is authorized by this DRAFT, source publication or CI.

After actual A and its CI, obtain one unique B with the original complete Claim tool, verify B CI/true remote HEAD/expiry, then re-read actual batch/account/QOS permissions, resource inventory and time limits. No unverified --test-only flag, debug Episode or extra policy forward. Submit only sbatch diagnostics/actor_memory_unmask/unmask_pair.slurm. Actual allocated GPU and loaded three-branch500 guards remain mandatory.

Any scheduler/access, rendering, loading, serialization, source, authorization, network/CI or data-integrity error means immediate STOP and evidence; no automatic retry or configuration change. Never run OFF after ON failure. Report SR/Cost deltas, all five cost components, episode-paired differences, failed earlyEND/exploration proxies and comparability limitations; missing data stays unavailable.
