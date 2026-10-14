# P2 cycle 002 entry bindings (preparation only)

Base: 55ecae3ba6e1fc834ea1976adab91640f15fcdd8, accepted by human PI with 27 CPU regressions. No additional Debug experiments or regression expansion.

New experiment: EXP-ACTOR-RESET-OFF200-ON200-002
New cycle: actor-reset-off200-on200-002-20261014
Execution worktree: /nvme2/user/qyy/SafeVLA_actor_logits_p2_002_20261014
Fresh output root: /nvme2/user/qyy/SafeVLA_p2_002_results_20261014 (absent at preparation; launcher rejects any existing path)
Control default: /nvme2/user/qyy/SafeVLA_loop_control_p2_codex_20261013

Only evaluate.py experiment/cycle constants and launcher experiment/cycle/default paths changed. Exact source SHA continues to be read from the clean claimed control tree's research/P2_CODE_SHA.txt; the PI draft must pin the resulting commit, never the old code SHA. Old claim fails identity checks. Environment path overrides retain existing authorization and no-overwrite guards; use the defaults above for this cycle.

Static CPU validation: bash -n PASS; Python AST parse PASS; exact replacement-only diff assertion PASS. No imports of model/simulator; no GPU, Episodes, synthetic Debug experiment or launch. Storage/Logger/reset/scientific runtime source and frozen manifest are byte-identical to accepted base. GPU equivalence remains NOT_RUN_PI_WAIVED, not PASS.

The full 27-test results remain in regressions/ndarray-20261013/ as historical evidence for the accepted base. They were not rerun or presented as new-cycle execution. A distinct final PI authorization plus its CI and a fresh CODEX claim plus its CI are mandatory before one-shot bash diagnostics/actor_logits/run_p2_full400.sh. No approval/claim is created by this code commit.
