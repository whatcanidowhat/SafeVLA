# P2 Logger ndarray-metrics exception — isolated fix proposal

## Observed runtime failure (2026-10-13)

First OFF200 episode reached logger `end_episode` and crashed at `storage.py:json_value` with `TypeError: Unsupported JSON type: ndarray` while writing `episodes/0000.partial/metrics.json`. User-supplied `ERROR.json` states episode_order=0, global_step=27, location=end_episode. This establishes **the environment reached its episode-finalization path**, but **not** that the logger wrote any `.complete` Episode; no validated 200-episode results exist in this evidence.

## Scope of this branch

This is an isolated change to **JSON metadata serialization only**. NumPy `ndarray` values are mapped to nested native lists via `.tolist()`. Existing `np.generic` scalar conversion remains. `allow_nan=False` continues rejecting NaN/Infinity; unsupported complex types remain errors. Crucially, complete raw root Actor logits are stored separately in `lossless NPZ`; there is **no downcast, rounding, truncation, reinterpretation of action logits, sampling, reset, Critic or official Safety Cost**.

The regression test uses only NumPy + the storage helper and does not import the SafeVLA model, create an Episode, initialize CUDA, or use a GPU. Passing this regression cannot prove any real GPU policy behavior.

## Mandatory treatment of prior attempt

The production code SHA `33ffa28000522e3715a0920bf3679c27e79b423e` is frozen for the existing CODEX Claim on `research-loop` (`e27a5d25bba4e6a991960a733d64fd295480b7b9`). This fix branch **is not approved for running under that claim**. Do not hot-patch that worktree, tamper with frozen source identities, change `P2_CODE_SHA` under an active claim, manufacture a PASS report, overwrite prior output dirs, or run the 400 Episodes again under the same claim.

First read only and preserve the original output: `OFF200/ERROR.json`, `RUN_MANIFEST.json`, `OFF200/episodes/0000.partial/` (including lossless step npz/actions, if present), official output and any own running process state. Report actual started/complete counts and exact failure in a truthful terminal BLOCKED/INVALID handoff for PI review. Only after a **new PI cycle / new authorization / fresh CODEX claim**, with explicitly reviewed and frozen new code SHA and fresh output directory, can a new 400-Episode run start. The original ATTEMPT is not a complete, valid baseline.
