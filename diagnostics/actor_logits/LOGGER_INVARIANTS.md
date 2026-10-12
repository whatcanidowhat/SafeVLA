# Logger invariants and integration review

1. Default Agent/Worker execution has no logger unless explicitly attached. One
   ActorLogitsLogger class serves OFF and ON. The independent reset.py is attached
   only for ON, and touches only root counter and root decoder K/V at Episode boundaries.
2. Hooks observe root actor.linear, root forward/decoder and both critic forwards.
   Critic actor heads are never used as root logits. Hooks return None and do not
   invoke model, distribution sample/mode, augmentation, controller or environment.
3. Agent.act retains its original forward, sample, mode, history and return logic.
   Additional observations copy already available values. Sampling counters mark the
   original invocation sites; independent CPU mock counters corroborate them. They
   are not a global interceptor capable of detecting every hypothetical hidden call.
   Independent real-runtime call-count verification is still required.
4. Worker observes final action after existing skip_done replacement, immediately
   before the original step, and acknowledges its return. It copies existing five
   pre-step cost locals. No extra observation/controller query is introduced.
5. Before/after treatment boundaries prove recorded critic counter/cache equality.
   CPU tests additionally compare weights and Python/NumPy/Torch RNG. These tests
   do not prove all real GPU state equivalence. Logger contains no seeding/reset logic.
6. Failures propagate and latch INCOMPLETE; subsequent calls are rejected. A disk
   failure can prevent additional error files, but the previous INCOMPLETE marker
   and original raised exception remain. No silent drop, retry, overwrite or zero fill.
7. Full evaluation preserves original single-worker main-process scheduling, one
   persistent Agent, seed123, shuffled 200 TaskSpecs, stochastic actions, augmentation
   and official max_eps_len=-1 (ObjectNav horizon600). Original video-selection
   shuffle, evaluator metrics, terminal semantics and normal Agent.reset are retained.
8. CPU manifest reconstruction executes the original loader selection method body
   with an identity-only normalizer and local dataset adapter. It restores Python RNG.
   It reproduces the frozen CSV hash and indices127,103,16,6,50. The historical
   `p1_selected` CSV column is retained solely for byte identity; it NEVER filters P2.
   Real normalization/TaskSpec identity is separately checked before Agent build.
9. reference_identity.json retains the accepted B0 source/dependency/weight/DINO
   hashes. The two integration files are declared changes. The accepted local DINO
   adaptation is copied into this isolated worktree and independently identified;
   the original B0 directory is untouched. All other accepted source files must match.
10. evaluate.py defaults to PLAN_ONLY. The future execution path requires an explicit
    operator acknowledgement, approval reference, clean committed code, exact hashes
    and a separately produced real-GPU PASS report bound to the implementation commit,
    600 pairs and atol1e-5. These are evidence consistency checks, NOT a replacement
    for PI authorization, a unique control claim and green control CI. No report is
    fabricated by development tests. Existing P1R2 allowance cannot authorize P2.

Integration source locations: InferenceAgentVIDA.act in
architecture/models/allenact_transformer_models/inference_agent.py;
OnlineEvaluatorWorker.evaluate_on_task in online_evaluation/online_evaluator_worker.py.
The original numerical model forward implementations and official metric function
remain unchanged. CPU tests execute these actual two modified method bodies via AST,
with small synthetic CPU dependencies, not real SafeVLA weights or simulator imports.

Real GPU output/RNG/cache equivalence, numerical tolerance, cache rollover, full
runtime import/load behavior, graphics GPU mapping, IO throughput and all live
Episodes remain NOT_RUN_AWAITING_AUTHORIZATION. Per-step copies/hashes and fsync
can affect wall time; numerical noninterference on real GPU must be measured before
scientific use. No claim of zero overhead is made.
