# FAST-TRACK Actor memory unmask successor

Experiment ID: EXP-ACTOR-MEMORY-UNMASK-002
Status: APPROVED
Cycle: actor-memory-unmask-002-20261029
Frozen candidate source: 2689081ec324c67a35f2a4692cfa009826fcf9d4

Measure delta_SR=SR_OFF−SR_ON and delta_mean_cost=Cost_OFF−Cost_ON on the same ordered200 tasks. Run MASK_ON200→MASK_OFF200, one worker,seed123, stochastic sampling/test augmentation, same checkpoint and allocated GPU UUID. Both Actor reset OFF; max_steps500/cache500/horizon600. Keep Actor/Critic/prev_actions/cache/counter/rollover and official success/five-cost rules unchanged. Limits:1 GPU,400 starts,240000 decisions,12h runtime,90GiB output.

Only execution bindings differ from base 1a77276ae8ee08cb34c67492a2a77c9a8c2cc0d8: batch replaces inaccessiblev100; fresh experiment/cycle/worktree/output/log names. All205 other source Git entries are byte-identical, including scientific code. Reuse existing38-check/417-load CPU evidence; no new CPU/logits or standalone GPU logger study.

Old001 terminal d670852bf08ed3d14d55dc1e4c631ccf678c6eeb, original Claim81219671 and its artifacts remain historical. Human-authorized PI acknowledgement e6d2b20a152a760b78d30549178f3f51378c0a63 clears old authorization; no old Claim/job/output/waiver is reusable.

The designated PI now accepts the common target dependencies and renderer for both arms, and scopes a NEW logger waiver to EXP002. No GPU/pixel equivalence PASS is asserted. A CI, unique Claim B, B CI, live remote HEAD, expiry and actual runtime checks remain mandatory.

After actual A and its CI, obtain one unique B with the original complete Claim tool, verify B CI/true remote HEAD/expiry, then re-read actual batch/account/QOS permissions, resource inventory and time limits. No unverified --test-only flag, debug Episode or extra policy forward. Submit only sbatch diagnostics/actor_memory_unmask/unmask_pair.slurm. Actual allocated GPU and loaded three-branch500 guards remain mandatory.

Any scheduler/access, rendering, loading, serialization, source, authorization, network/CI or data-integrity error means immediate STOP and evidence; no automatic retry or configuration change. Never run OFF after ON failure. Report SR/Cost deltas, all five cost components, episode-paired differences, failed earlyEND/exploration proxies and comparability limitations; missing data stays unavailable.

## PI decision on successor

Approved source: `2689081ec324c67a35f2a4692cfa009826fcf9d4`.
Approved UTC: `2026-10-29T09:09:51Z`; expiry UTC: `2026-10-31T09:09:51Z`.
Delegation: the project user designated this AI collaborator as PI; this is not an independent human signature.
Limits: 1 GPU, 400 Episodes, 240000 decisions, 12h runtime, 90GiB output. Scheduler batch/zhangyh/user_zhangyh; recheck live ACL/capacity before submitting once.
Environment and renderer accepted for both arms only. Logger equivalence PI-waived for EXP002, not PASS. Approval alone does not authorize a GPU launch; require A/B independent successful push CI, unique Claim B, live HEAD/expiry and frozen runtime gates.
