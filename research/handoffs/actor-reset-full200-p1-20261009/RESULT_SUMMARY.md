# 001D P1 terminal result: INVALID

One approved command invocation ended at 2026-10-10T07:14:04.223395+00:00. No retry, restart, live ON or P2 execution occurred. Next actor is PI.

## Verified facts
- Published claim `334e80351ff6cb2cfb0dac166fa7227ae055a5bd`, claim_id `b19afa0bbe1b4957b22ce081a44ee3e0`, instruction `67c42b0fd7d485fdaa9fe5687099537c2c28035a`; [claim CI SUCCESS](https://github.com/whatcanidowhat/SafeVLA/actions/runs/38016589200).
- Isolated B0 source/weight/DINO/dependency hashes passed pre-run verification. Policy load contained417 keys, no missing/unexpected keys; two DINO175-key strict loads succeeded. Imported project modules were under the isolated root.
- Actual real200 TaskSpec manifest exists, SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0`; planned source indices127,103,16,6,50. The original online loader cross-check was NOT_REACHED.
- One offline root Actor forward completed on the first saved real input. The next operation failed restoring the pre-forward cache snapshot: shape `[1,500,8,64]` destination versus `[0,500,8,64]` saved cache.
- Logger-present replay never started; zero complete equivalence pairs. Reset invariant test NOT_REACHED. Both online OFF sessions NOT_STARTED: zero simulator starts, zero episode initialization attempts, zero online decisions.
- Budget used:1 GPU; elapsed conservative GPU upper bound 0.010825 hours; peak Torch allocated 2894834176 bytes. This is not measured full-session GPU capacity.
- B0 runtime source and original tracked development/B0 diffs remain unchanged; see preservation_check.json.

## Cause and interpretation
The executor's snapshot restoration assumed a fixed cache shape. Frozen B0 `training/online/third_party_models/llama/model.py:280-284` lazily materializes a zero-batch cache on the first single-step forward. Runner `restore_temporal` at75 used in-place copy and could not restore the initial empty shape. Traceback at169 occurs before the logger-present branch. This is an executor harness defect, not evidence of logger numerical drift, Actor reset efficacy, or a Baseline defect.

Syntax/static guards passed but did not exercise this lazy-allocation lifecycle. No numerical tolerance or behavior was changed after failure. Expected P1 gates remain unproven: logger equivalence, Actor-only reset, original online task cross-check, A/A trajectories and official SR/Cost reconciliation.

SR and Safety Cost: NOT_MEASURED, n=0/0 online sessions. Do not report zeros as outcomes or infer reset benefit. Accepted001A conclusions are unchanged.

## Handoff and next decision
All14 required outputs are provided, including explicit not-started gate reports and the exact failed runner. Full error traceback, policy load events, input manifest, source hashes and frozen implementation are Git-readable. Original stdout is retained server-side as optional evidence. Large historical captured inputs remain server-only optional evidence with hash/size.
P1 readiness FAIL; P2 BLOCKED/NOT_AUTHORIZED. Sole proposed next action: PI review of an isolated snapshot-restoration correction and CPU lifecycle test, then a fresh cycle/approval for any rerun. No correction or rerun is authorized by this handoff.
