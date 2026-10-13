# Validation summary

Prelaunch: frozen clean source SHA 33ffa28000522e3715a0920bf3679c27e79b423e; bash -n PASS;9 Python AST PASS;200 task manifest SHA 8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0 PASS, first indices127,103,16,6,50. Historical22 CPU mock tests belong to bb71a65 development, not a newly run frozen-code suite. No new GPU equivalence tests.

Postmortem CPU-only:27 rows/acks,189 native tensor fields, packed NPZ vs all27 shards, shapes/dtypes/SHA, complete20-action vectors, task/order/counter metadata and original artifact-index hashes checked. OFF after_original_reset vs after_treatment boundary states identical. These partial evidence checks PASS; run integrity correctly returns INCOMPLETE with null outcomes. They do not promote the partial directory.

Root cause: storage.py:23-26 json_value supports NumPy scalar, not ndarray. Original Worker.calculate_metrics:571 produces percentage_collision from observation values; metrics.json.tmp stops at that key. Logger.end_episode:217 -> EpisodeWriter.finish:133 -> write_json:33 raised. Exact ndarray shape/value was not serialized and is not inferred. Existing CPU mocks used scalar metrics and did not cover this carrier. No source fix or experiment retry performed.

Original manifest decisions0 and launch context RUNNING were preserved unchanged; terminal handoff reconciles27 actual decisions and exit1. SIGTERM did not release failed own process within60s; verified exact own PID was killed after88.6s. GPU2 subsequently had no compute PID,5MiB used. Independent renderer UUID tracing and600-input GPU equivalence were not performed.
