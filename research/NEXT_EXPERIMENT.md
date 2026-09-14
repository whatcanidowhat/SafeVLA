# Next Experiment — 001B protocol preflight

Experiment ID: EXP-B0-REPRO-001B-PREFLIGHT
Status: APPROVED
Authorization: APPROVED_FOR_CODEX — RESEARCH_EXPERIMENT / 0 GPU / 0 episode
Cycle ID: b0-repro-001b-preflight-20260914

## Purpose

001A has already passed provenance in the reviewed single-worker two-task smoke. The full 200-task A/A run is still not authorized because its task manifest, worker/scheduling semantics, exact commands, budget, and comparison criteria are not yet frozen. This preflight exists only to resolve those protocol questions from read-only evidence before any full run.

## Research question

Can the current verified executable B0 support an unambiguous, auditable full-200 A/A protocol definition without starting any model, simulator, GPU inference, or episode?

## Hypothesis

Read-only inspection of the executable B0 candidate, 001A provenance packet, dataset/task metadata, evaluator/launcher source, and historical full-200 records is sufficient to freeze the exact 200 tasks, stable pairing key, worker/scheduling semantics, two new run commands, resource plan, and pre-registered comparison criteria.

## Competing explanation

The protocol may remain underdetermined because task enumeration, worker topology, launcher defaults, or metric collection differ across paths or require runtime execution to resolve. If so, report BLOCKED rather than guessing or launching 001B.

## Scope and unique variable

There is no policy treatment. The only scope change versus 001A is from a two-task single-worker smoke to defining the full-200 protocol. Executable B0 behavior must remain unchanged.

## Fixed conditions

- Candidate: `/nvme2/user/qyy/SafeVLA_baseline_clean`, official reference HEAD `2aa82559d272b5f888e53433e258914057f15bed`, with only the already approved local-DINO infrastructure adaptation.
- Do not modify policy, actor/decoder, success/end logic, horizon, official metrics/cost semantics, checkpoint, reset/counter/cache, sensors, augmentation, or evaluator behavior.
- 0 GPU, 0 episode, no model load, no AI2-THOR, no SafeVLA rollout.
- Do not rerun 001A. Do not start full 001B.
- Historical 86.5% remains context only; it is not a Reference Run.
- Runtime/code/dataset inspection is read-only.

## Required evidence

The handoff must either freeze or explicitly block each of these: executable-B0 identity; exact 200-task manifest and source hash; stable task pairing key; dynamic episode-ID handling; worker count; task assignment/scheduling; seed; shuffle; stochastic/greedy setting; test augmentation; horizon; process boundaries; two new full-run command templates; output/provenance requirements; resource estimate; and pre-registered comparison plan for SR and official Safety Cost.

For performance comparison, propose criteria before observing new 001B outcomes. At minimum include SR delta, paired success agreement with uncertainty, and official Safety Cost total/distribution/nonzero-episode behavior with paired differences. Do not require bitwise-identical trajectories.

Preflight PASS requires zero unresolved ambiguity that could change task identity, policy behavior, evaluation semantics, or comparability. Otherwise return BLOCKED.

## Formal command anchor

`git -C /nvme2/user/qyy/SafeVLA_baseline_clean rev-parse HEAD`

This command is only an identity anchor. The audit itself may use additional read-only shell/source inspection commands, all of which must be recorded in the run manifest. No command may load the model, allocate inference GPU, start AI2-THOR, or start an episode.

## Required outputs

- `research/handoffs/b0-repro-001b-preflight-20260914/RESULT_SUMMARY.md`
- `research/handoffs/b0-repro-001b-preflight-20260914/RUN_MANIFEST.json`
- `research/handoffs/b0-repro-001b-preflight-20260914/ARTIFACT_INDEX.json`
- `research/handoffs/b0-repro-001b-preflight-20260914/REVIEW_NOTES.md`

Full EXP-B0-REPRO-001B remains NOT_AUTHORIZED after this preflight until a later PI review explicitly approves it.
