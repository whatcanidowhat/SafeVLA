# Unmask online Slurm candidate — 2026-10-28

Status: **ONLINE_NOT_STARTED_BLOCKED**. User requested execution; this packet completes new online entrypoint preparation without impersonating PI or using an old Claim. Current true GitHub control HEAD is recorded in READINESS.json; old terminal BLOCKED and expired authorization do not cover this experiment.

New code: evaluate_online.py (derived from immutable P2 evaluator), online_gate.py, online_logger.py, online_pair.py, online_results.py and unmask_pair.slurm; test_online_gate.py has14 CPU tests PASS. Default evaluator plan reconstructs200 frozen TaskSpecs, no model/simulator/output directory. Slurm shell syntax PASS. Actor switch, Critic, attention cache/counter/rollover, prev_actions, historical evaluator, worker and inference source remain byte-identical to b0ab97e.

The future Slurm pair preserves the scheduler CUDA mapping, runs MASK_ON200 then MASK_OFF200 with Actor reset OFF in both, one persistent worker per arm, seed123/shuffle/stochastic sample+mode/test augmentation and original official metrics. Fresh process per arm. Full200 means400 starts, max240000 decisions. No old OFF200 used as control, no extra policy forward. Raw logits/probs/actions and official metrics use inherited passive storage; root mask/critic mode assertions added without forward calls.

Guard: exact new experiment/cycle/worktree/source pin, current remote claimed control, distinct approval/Claim CI, expiry/budget, exact dependency versions, reviewed asset inventory, reviewed renderer environment, new logger-equivalence decision, one Slurm GPU, capacity/no foreign compute, absent output and disk/time ceilings. No old reset Claim/waiver, no extra forward, no automatic retry/GPU switch or foreign process termination. Child cleanup targets only the coordinator's own process group. Historical CPU/logger evidence does not become GPU PASS.

Current blockers: PyTorch2.5.1 vs2.4.1, NumPy1.23.5 vs2.1.2, jsonschema4.25.1 absent; asset content inventory not pinned; Mesa/native rendering not reviewed; no independently approved new cycle/source/Claim/CI. Do not submit sbatch merely because this script exists. No job/model/GPU/simulator/Episode was started by this preparation.

PI_REQUEST.json is a draft request, not an applied control state or approval. Only PI can review the exact shared candidate, handle prior terminal control, and approve a fresh cycle according to HANDOFF_PROTOCOL. Candidate remains CPU/static validated; online path is untested and must be reviewed before execution. Existing control files and old evidence are unchanged.

Analysis reports official SR, five cost components, full cost distribution/median/p90/max, task-paired OFF-minus-ON changes, descriptive paired bootstrap and exact McNemar. Cross-Episode history and stochastic trajectory dependence limit causal/statistical interpretation; no nonsignificance-as-equivalence claim. Results/analysis code has not been exercised on a new live run.
