# Actor logits evidence schema v1

Development implementation; real GPU equivalence remains NOT_RUN_AWAITING_AUTHORIZATION.
Paths below are relative to one fresh OFF200 or ON200 output directory.

| Artifact | Contents |
|---|---|
| RUN_MANIFEST.json | COMPLETE/INCOMPLETE, expected/completed Episodes, decisions, ordered TaskSpecs and IDs, action map, source/config/weight identity and errors |
| TASK_MANIFEST.csv | Independently reconstructed frozen 200-row CSV, SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0` |
| episodes/0000.complete/actions.jsonl | One row per actual policy decision, including post-worker replacement action |
| episodes/0000.complete/tensors.npz | All seven tensor fields for every decision, keys `000000.raw_actor_logits`, etc. |
| episodes/0000.complete/execution_results.jsonl | Separate ordered acknowledgements after the original environment step returned |
| episodes/0000.complete/boundary_states.jsonl | before_original_reset, after_original_reset, after_treatment, episode_end |
| episodes/0000.complete/metrics.json | Unmodified official metrics plus aligned summary |
| episodes/0000.complete/EPISODE_STATUS.json | Episode identity, COMPLETE status, decision count |
| episode_metrics.csv | One ordered summary per Episode, including success and five official cost components |
| ARTIFACT_INDEX.json | Relative paths, byte sizes and SHA256 for every other artifact; excludes itself |
| ERROR.json / ERROR_EVENTS.jsonl | First failure and subsequent error events, location, traceback, active counts and alignment position |

Decision keys: episode_order (zero based), sample_id (Episode ID), task_source_index,
raw_spec_sha256, step_index (zero based), global_step, decision_id
`{episode_order:04d}:{sample_id}:{step_index:06d}`, worker_id, goal, target_ids,
sampled_action_index, greedy_mode_action_index, agent_action_name,
executed_action_index/name, counter_before/after, decoder_start_pos, mask,
calls, input_sha256, pre_step_cost and tensor metadata. `execution_status=DISPATCHED`
records the action passed to step; the separate acknowledgement distinguishes a
returned call from a crash during step. It does not claim successful movement.

Tensor fields are raw_actor_logits (root actor.linear only), normalized_logits
(existing distribution.logits), action_probs (existing distribution.probs),
sampled_action, greedy_mode_action, action_history (existing flattened sampled
history), and prev_action. All action dimensions are retained, with original
logical dtype and shape. Storage uses lossless native NumPy arrays in C logical
element order; arbitrary tensor strides are not reconstructed. BF16 is stored as
exact uint16 bits with explicit `bfloat16_bits` encoding and restored as BF16.
No rounding, quantization, Top-k, pickle or reverse sampling is used. Tensor
metadata records shape, logical dtype, storage dtype, encoding, C order and raw-byte
SHA256. Boundary/mask metadata also records original device. The effective mask
is described by full shape/dtype/device/hash, or null; mask arrays are not archived.

Each boundary records root Actor, Reward Critic and Cost Critic counters and every
layer's full K/V hash, shape, dtype and device. K/V arrays themselves are not stored.
input_sha256 hashes the already constructed policy input, including existing
Memory values; it does not fetch another observation or rerun preprocessing.

The Writer exclusively creates `0000.partial`, durably writes per-step NPZ shards
before JSON rows, and records step acknowledgements. Episode completion packs
the shards, writes metrics/status, then atomically renames the directory to
`0000.complete`. Interruptions retain completed Episodes and current partial
evidence. Readers never promote partial data or manufacture missing zeros. No
automatic resume/retry is implemented. Atomic rename protects process-crash
consistency; filesystem/power-loss guarantees beyond file fsync are not claimed.

CPU tools (write reports OUTSIDE the indexed run directory):

```bash
python -B -m diagnostics.actor_logits.validate /path/OFF200 --expected 200 --report /path/OFF200_validation.json
python -B -m diagnostics.actor_logits.analyze /path/OFF200 /path/ON200 --report /path/comparison.json
```

Python reader: `read_episode(path)` in storage.py returns status, aligned rows,
losslessly restored CPU tensors and metrics. Validator checks rows, tensors,
acknowledgements, boundaries, counts, official pre-step cost reconciliation,
CSV alignment and artifact hashes. It reports INCOMPLETE and null outcomes when
200 Episodes or evidence are missing. Analyzer also requires identical task order,
source/weight identity, action map and fixed configuration. It reports SR/mean cost
and cost sum differences, paired task outcomes and five components, first logits
and action divergence, entropy, end-action probability, JSD, repeated-action and
ABA oscillation proxies. Unpaired trailing steps remain explicit.

Online same-step comparisons are descriptive: once actions diverge, observations
can diverge. Equal input hashes alone do not prove equal cache/RNG state or establish
causality. Only separately authorized controlled offline tests can support direct
logits attribution. This integrity validator does not certify GPU equivalence.
