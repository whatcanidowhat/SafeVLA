# Full-400 manual run: execution now routed via shell supervisor after a green human claim

The approved frozen plan is `OFF200 -> validate -> ON200 -> validate -> analyze`.
The actual one-shot entry is `bash diagnostics/actor_logits/run_p2_full400.sh`.
The script verifies exact manual OPERATOR_RUNNING claim in /nvme2/user/qyy/SafeVLA_loop_control,
the corresponding GitHub control CI SUCCESS, freezes one shared physical GPU (0-3),
checks source/weights/task identity through evaluate.py, and monitors deadline and output caps.

Environment:
- `P2_OUTPUT_ROOT`: choose one **nonexistent** external results directory; if omitted script uses `/nvme2/user/qyy/SafeVLA_p2_full400_results_20261012`.
- `P2_CONTROL_ROOT`: default `/nvme2/user/qyy/SafeVLA_loop_control`.
- `P2_PYTHON`: default `/home/amax/.conda/envs/safevla/bin/python`.
- `HTTPS_PROXY`/`HTTP_PROXY`: retain your already working authenticated GitHub transport configuration for claim CI check.
The runner is deliberately not compatible with the old --gpu-equivalence-report requirement.
Never invent a PASS report, append to existing results, or run OFF and ON concurrently.
