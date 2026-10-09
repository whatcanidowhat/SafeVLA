# EXP-RESET-EARLYEND-001B — B4 / BLOCKED

P0 confirms 16 historical unsuccessful sub-horizon cases with executed terminal `end`.
All 16 have UNKNOWN historical worker-local counter_start. No EXPOSED, BOUNDARY or
UNEXPOSED assignment is justified. The frozen P0 stopping condition is met.

The retained original four-worker run persists global result-arrival tables and
aggregate worker counts, not per-task worker_id/worker-local iteration. Its W&B
binary journal (history, summary and reassembled raw console), launcher/output/debug
logs and table schemas contain no recoverable mapping. Historical source creates
worker_id and iter in an in-memory queue payload; Linux disables its per-task print,
and the persisted tables omit those fields. This is an evidence limitation, not a
proof that every possible external worker log is absent. See source_audit.json and
historical_source_excerpts.json for the exact bounded search and source references.

Eligibility is verified against the aligned full200 table, accepted terminal-action
traces and their independent validation; all 16 original video hashes match the
accepted input manifest. Terminal end is not inferred from episode length alone.
Historical end steps are zero based. Missing counters and rollover steps stay blank.
No counter, worker assignment, cache or order is invented from timestamps or global
completion order. The prescribed 500-counter_start rule therefore cannot be applied.

P1 and subsequent online comparisons: NOT_STARTED. GPU use 0; live episodes 0;
policy checkpoint loads 0; simulator starts 0. paired_episode_results.csv is header
only; paired_step_trace.parquet is a typed zero-row table marked NOT_STARTED.
These empty outputs are missing experimental observations, not zero treatment effects.

No causal conclusion B1/B2/B3 is supported, and the accepted 001A R1 result remains
unchanged. This audit neither supports nor refutes reset effects on early termination,
success or official Safety Cost. It cannot identify valid negative-control cases.

Next actor PI. Specific next action: determine whether an authentic task-to-worker
ordered log can be supplied; if unavailable, PI must decide a new design and approval.
No successor experiment, rerun or replacement claim is authorized by this handoff.
Executor STOP after publication.
