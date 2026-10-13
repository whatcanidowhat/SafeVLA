# ndarray serialization repair: CPU validation

Reused existing repair commit49d434f42e2d636edeee5c2cd379df198a4bd41f unchanged for all runtime code. json_value handles np.ndarray with tolist(), then np.generic with item(). Relative to prior33ffa280, the existing repair touches only the JSON codec plus regression tests/docs/CI; model, Actor Reset, task/seed/worker configuration and tensor shard encoding are unchanged.

Base full CPU suite:25/26 PASS. Sole failure was an outdated exception-message assertion in the authorization guard test. The actual guard raised the correct missing PI approval/claim/control-tree error before runtime import. Original test output is retained; it is not reported as PASS.

This supplementary commit changes tests and evidence only: update that expected diagnostic and add a two-synthetic-Episode integration regression exercising real Agent/Worker method bodies with mocked CPU dependencies, ndarray percentage_collision plus nested arrays/scalars, atomic EpisodeWriter completion, validator acceptance and raw Actor logits byte equality. It does not run a real Episode or load weights.

Final full CPU suite:27 tests PASS in0.701s. AST checks and git diff --check PASS. Native FP16/FP32/FP64/BF16/integer/bool roundtrips, JSON arrays/scalars/nesting, nonfinite/unsupported-value rejection, call counts/RNG and full root logits remain covered. Runtime files were compared byte-for-byte with49d434f and match. CUDA_VISIBLE_DEVICES was empty and the suite asserts CUDA uninitialized.

Command:
```
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/amax/.conda/envs/safevla/bin/python -B -m unittest discover -s diagnostics/actor_logits/tests -v
```

GPU/model/simulator/live Episode usage for this repair task:0. GPU_EQUIVALENCE=NOT_RUN_PI_WAIVED is the PI-requested label for future authorized execution, NOT a PASS from these CPU tests.

Execution authorization remains pending. Last verified research-loop HEAD19dcb30e4858be81be25620ebe151860f7720b2c is old cycle INVALID/next_actor=PI, with no PI acknowledgement or fresh approval yet. No control files/claims/old outputs changed. Do not reuse its retained authorization fields or Claim.

Before any rerun, PI must independently acknowledge the old INVALID, then freeze and approve a fresh experiment/cycle ID, source SHA and new execution/result directories. The inherited evaluate.py and run_p2_full400.sh still bind the old experiment/cycle (and default old output path): they cannot simply be launched under a new cycle. A separately reviewed entry binding update must accompany the new design/source pin; this test-only commit does not silently alter those bindings or grant execution. New approval CI and a unique new Claim CI must pass. No retry or GPU replay is performed here.
