# Isolated implementation and static audit

No accepted B0 file is edited. Changes are the standalone runner and CPU validator in this handoff directory.
frozen_identity.json indexes all 112 copied B0 files and original development/B0 diff hashes; implementation_freeze.json binds supplemental code and protocol hashes.
Original B0 /nvme2/user/qyy/SafeVLA_baseline_clean and development /nvme2/user/qyy/SafeVLA remain preserved.

Inspected B0 source locations (line numbers refer to frozen B0):
- architecture/models/allenact_transformer_models/inference_agent.py:94 build_agent;169 initial reset;172 reset;229 act;269 single combined forward;276 sample;278 mode;281 sampled action history.
- architecture/models/allenact_transformer_models/allenact_dino_transformer.py:286 forward_encoder;326 forward; rollover at counter500. Runtime decoder: training/online/third_party_models/llama/model.py.
- online_evaluation/online_evaluator.py:340 original full200 loader/shuffle; singleworker synchronous dispatch preserves one Agent.
- online_evaluation/online_evaluator_worker.py:53 start_worker;274 task reset;301 pre-step cost reads;318 existing task step;505 success;506 cost.
- architecture/allenact_preprocessors/dino_preprocessors.py:217 augmentation interval;243-245 transform sampling and counter carry.
- tasks/multi_task_eval_sampler.py:206-215 timeout retry path: runner rejects recursive task initialization.

Runtime modifications are scoped: passive hooks/profile observations; wrapper checks the original loader result without replacing samples; a private completion exception stops after fifth official result and preserves worker cleanup. Offline-only SavedEncoder uses actual frozen features; root_reset is never called online.
Logger reads existing outputs, tensor copies, hashes and RNG states. It does not query simulator/controller or request additional forward/sample calls. Actual online Categorical sample/mode calls and root/critic forwards are counted.
Static validation compiles without imports and checks manifest count/hash, reset write targets and explicit budget guards. Runtime identity, numerical equivalence, actual task order, counts and metrics remain NOT_VERIFIED until execution reports exist.
