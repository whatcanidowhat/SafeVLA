# EXP-ACTOR-MEMORY-UNMASK-001 — BLOCKED before GPU allocation

Claim B `81219671d89762437d576e6daf20f38f87fc1271` was published by the original full Claim tool and its exact push CI40833565006 attempt1 passed. Approval A `077edc718e1a6eddfd3850e0d264461dad8f555f` CI40817284620 also passed. The final live permission/source/expiry check passed; frozen source remains `1a77276ae8ee08cb34c67492a2a77c9a8c2cc0d8`.

The only `sbatch diagnostics/actor_memory_unmask/unmask_pair.slurm` attempt returned exit1: `Invalid account or account/partition combination specified`. No Job ID, GPU allocation, model, simulator, policy forward or Episode was started. MASK_ON200=0/200, MASK_OFF200=0/200. SR_ON, SR_OFF, delta_SR, Cost_ON, Cost_OFF, delta_Cost, all five cost components and failed early END are **NOT_AVAILABLE**, not zero.

Read-only scheduler evidence identifies `v100` AllowAccounts=`slurmtest,root`, excluding approved script account `zhangyh`. The account association and QOS exist; an unrelated pre-existing job246255 uses accountzhangyh/QOSuser_zhangyh in partitionbatch. No existing job was touched. This is a scheduler access failure, not evidence about Actor memory performance.

Execution stopped immediately. No partition/account override, source edit, resubmission, extra Claim or debug Episode followed. The current Claim must not be reused or resumed. PI should review this terminal result and resolve the scheduler restriction in a freshly frozen, newly approved cycle; no additional logits study is needed.
