# Next Experiment — Clean B0 full-200 premature-end reproduction

Experiment ID: EXP-CLEAN-B0-PREMATURE-END-REPRO-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: clean-b0-premature-end-repro-001-20260923

## Research question
Under clean executable B0, does a complete 200-task ObjectNav evaluation still contain sub-horizon failures whose final executed action is `end`?

## Why this experiment
The historical audit verified 16/16 sub-horizon failures ended with `end`, but it did not show a gradual search-triggered rise in `p(end)`; the probability increase was terminal-only and also appeared in successful controls. More importantly, the historical 2026-08-03 evaluator used commit `60bc54f...` and called `successful_if_done()` before `agent.get_action`, which can issue `GetVisibleObjects` on a visibility-cache miss. That extra live query violates the current clean-B0 contract. Its behavioral effect is unknown.

Therefore the historical run remains hypothesis-generating. Before SafeVLA-vs-FLaRe or hidden-state attribution, establish the invalid-end phenotype under clean B0.

## Competing hypotheses
- H-PERSIST: clean B0 still has confirmed sub-horizon invalid-end failures.
- H-HISTORICAL-PROVENANCE: clean B0 has few/no such failures or a substantially different profile.

## Protocol
One full 200-task ObjectNav evaluation in `/nvme2/user/qyy/SafeVLA_baseline_clean`.

Freeze: official reference `2aa82559...` plus only accepted local-DINO adaptation; 200 tasks; horizon600; seed123; workers4; shuffle on; test augmentation on; stochastic sampling (`greedy=false`). Preserve official success/end/metrics/action semantics.

Forbidden: PT-Guard, Done Gate, shadow logger, steering, reranking, Probe, action rewrite, extra policy forward, or any extra pre-decision success/visibility/controller query.

## Metrics
Report official SR/Safety Cost/SEL; failure and episode-length distributions; count `eps_len<600`; final action for every sub-horizon failure; confirmed invalid-end prevalence; horizon failures; and offline video-derived quantized `p(end)` for clean invalid-end cases. Report outcomes of the 16 historical invalid-end IDs descriptively only.

## Decision
If clean invalid-end persists, premature termination remains a clean-B0 mechanism target. If zero confirmed invalid-end cases occur, weaken H-PERSIST and investigate provenance/run variability before safety-alignment attribution.

Incomplete 200-task run or any identity/protocol violation => BLOCKED; do not silently resume or change settings.

## Required outputs
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/RESULT_SUMMARY.md`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/RUN_MANIFEST.json`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/ARTIFACT_INDEX.json`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/REVIEW_NOTES.md`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/episode_results.csv`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/termination_summary.csv`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/historical_overlap.csv`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/b0_provenance.json`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/analyze_clean_b0_termination.py`
- `research/handoffs/clean-b0-premature-end-repro-001-20260923/run_clean_b0_full200.sh`

After handoff publication, STOP.
