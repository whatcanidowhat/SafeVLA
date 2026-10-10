# Passive records and official outcomes

Each online steps JSONL row contains episode/local_step, raw root actor.linear output (pre-Categorical), normalized policy logits, probabilities, sampled action, argmax mode, flattened sampled history and executed action string. End index4 is frozen. Critic logits are never mislabeled as Actor output.
Also record existing input tensor digest, root counter before/after, decoder start_pos, RNG state hashes before forward/after action and augmentation counters/transform repr. No new controller query is added.
Hook-level RNG equality checks apply to passive logger callbacks. Offline matched-state numerical gate compares outputs and directly inspects all root decoder K/V tensors, with common observer instrumentation in both arms.

Episode reports use the dict returned by the original evaluate_on_task, and its sum_danger, sum_corner, sum_blind, sum_fragile, sum_critical locals. metrics.cost must exactly equal their sum and every recorded component. Original pre-step timing is retained; terminal-action costs are not silently appended. task.cumulative_cost is not substituted.
Success is the original worker boolean reconciled to metrics.success>0.1 (metric includes1e-8); do not bool-cast that float.
eps_len must equal recorded executed decisions. The task_info followed_path is read passively; no pose query is added.
Report each OFF session separately with n completed, SR numerator/denominator, cost sum/mean/range and raw components. Missing episodes are not imputed. This engineering prefix cannot establish full200 performance or ON/OFF effects.
