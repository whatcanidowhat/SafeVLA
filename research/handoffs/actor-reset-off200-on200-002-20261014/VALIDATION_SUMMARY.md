# Actual validation

- Frozen source 06cde9adeb91d47c062a11e72e90c7edf56fb398, clean before and after run; all runtime source hashes rechecked equal.
- Checkpoint/DINO/task CSV hashes matched before run. Real loader and loading events recorded in OFF200_RUN_MANIFEST.json, including 417 policy keys and no missing/unexpected entries.
- Existing official run validator: PASS, 200 Episodes and 19270 decisions. Checks include full 20-action vectors, tensor hashes/shapes/dtypes, action acknowledgements, forward/sample counts, task order, boundaries, unchanged OFF treatment/critic state, official cost aggregation and artifact coverage. See OFF200_VALIDATION.json.
- After termination, 1203 raw index entries matched size and SHA256; no additional GPU/model call or Debug experiment was run.
- ON failed at remote claim lookup before model imports and result directory creation. No ON numerical, reset or comparison validation occurred.
- NOT_RUN_PI_WAIVED remains a waiver, not GPU equivalence PASS.
- Full console logs preserved as text copies; source/runtime manifest, actual per-task metrics and failure traceback are Git-readable. Large raw files are server-only, optional for this compact PI review and hashed; this does not claim upload or independent PI review.
