# Next Experiment — Terminal end-legality calibration audit

Experiment ID: EXP-END-LEGALITY-CALIBRATION-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: end-legality-calibration-001-20260923

## Research question
When SafeVLA actually executes `end` in the historical full-200 run, does its Actor confidence distinguish a legal successful stop from an illegal failed stop?

## Why this replaces reproduction
The researcher explicitly decided that neither a full-200 rerun nor a targeted clean-B0 reproduction is worth the current mainline cost. The 16 historical invalid-end cases are sufficient as a discovery set. Their provenance limitation remains recorded, so this experiment does not promote them to formal clean-B0 prevalence evidence.

The previous audit already showed that the terminal `p(end)` jump is not failure-specific. The next useful question is therefore whether **terminal confidence itself encodes legality**.

## Population
- Illegal end: the 16 confirmed sub-horizon failures with final executed `end`.
- Legal end: all successful episodes with recoverable terminal probability frames from the same historical run.

## Hypotheses
- H-CALIBRATION-GAP: legal and illegal end confidence substantially overlap.
- H-LOW-CONFIDENCE-END: illegal ends are systematically lower-confidence than legal ends.

## Metrics
Recover terminal quantized `p(end)`, end-vs-next-action margin, t-1 / previous-five-step `p(end)`, terminal jump, and entropy when reliable. Report distributions plus AUROC/bootstrap uncertainty. Because all cases are conditioned on executing `end`, high p(end) in both groups is expected; the informative question is separation between legal and illegal groups.

## Fixed conditions
Existing historical artifacts only; 0 GPU; 0 episodes; no simulator/model/replay/Probe/reset/extra forward. Quantized video probabilities are approximate. Historical provenance remains a limitation and no SafeRL/clean-B0 causal claim is allowed.

## Decision rule
If legal and illegal terminal confidence overlap strongly, next diagnose whether stop-legality information is absent from representation or present but unused by Actor. Hidden-state work must first respect H-RESET Gate B.

If illegal end confidence is systematically lower, prioritize stochastic sampling / end-threshold calibration as the next intervention branch.

## Required outputs
- `research/handoffs/end-legality-calibration-001-20260923/RESULT_SUMMARY.md`
- `research/handoffs/end-legality-calibration-001-20260923/RUN_MANIFEST.json`
- `research/handoffs/end-legality-calibration-001-20260923/ARTIFACT_INDEX.json`
- `research/handoffs/end-legality-calibration-001-20260923/REVIEW_NOTES.md`
- `research/handoffs/end-legality-calibration-001-20260923/terminal_end_events.csv`
- `research/handoffs/end-legality-calibration-001-20260923/legality_calibration_summary.csv`
- `research/handoffs/end-legality-calibration-001-20260923/legality_calibration.md`
- `research/handoffs/end-legality-calibration-001-20260923/analyze_end_legality.py`

After handoff publication, STOP.
