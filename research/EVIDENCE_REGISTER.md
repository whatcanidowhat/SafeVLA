# Evidence Register — migrated from 02｜SafeVLA研究

Updated: 2026-09-18
Purpose: give the PI and Executor a shared index from research claims to the strongest currently known evidence. This register is an index, not a substitute for opening raw evidence.

## Evidence classes

- **A — direct/raw**: source code, raw per-episode output, run manifest, checkpoint/hash, raw diagnostic trace.
- **B — derived/audited**: analysis generated from an identifiable raw source with stated limitations.
- **C — historical/legacy**: older modified-policy run, exploratory probe, chat-era artifact, or path whose current existence/provenance must be rechecked.
- **D — researcher observation**: reported by the researcher but not yet reconciled to a specific raw run.

If classes conflict, prefer A > B > C > D.

## Register

| Evidence ID | Class | Evidence location | Supports | Does NOT support / caveat | Executor action |
| --- | --- | --- | --- | --- | --- |
| E-B0-OFFICIAL-SOURCE | A | GitHub official ref `2aa82559d272b5f888e53433e258914057f15bed`; relevant files: `tasks/object_nav_task.py`, `tasks/abstract_task.py`, `architecture/models/allenact_transformer_models/allenact_dino_transformer.py`, `architecture/models/allenact_transformer_models/inference_agent.py`, `architecture/models/allenact_transformer_models/separate_actor_critic.py` | end is an Actor action; non-strict ObjectNav success is checked after end using nav-camera visibility with max distance 2m; cost critic is not a direct inference-time end gate | does not explain why the Actor assigns a particular end probability | read source only if a claim depends on exact semantics |
| E-B0-001A | A/B | `/nvme2/user/qyy/SafeVLA/research/runs/EXP-B0-REPRO-001A/`; especially `RESULT_SUMMARY.md`, `execution_20260912/validation.json`, `pairing.json` and reference/repeat manifests | narrow executable-B0 provenance, resource identity, 2/2 task pairing, artifact completeness | not full-200 performance reproducibility; not mechanism evidence | do not rerun; use only for provenance |
| E-B0-CANDIDATE-AUDIT | B | `/nvme2/user/qyy/SafeVLA/research/B0_CANDIDATE_AUDIT.md` plus recorded official→candidate patch/hash | clean candidate was based on official source and preserved official behavior except approved local-DINO infrastructure adaptation | static similarity alone is not benchmark equivalence | read only if current experiment needs B0 identity |
| E-FULL200-RAW | A, pending re-audit | historical full-200 directory reported in 02: `/nvme2/user/qyy/SafeVLA/eval/objectnav-full-minival-200-20260803-gpu0-w4/ObjectNavType/safevla-objectnav-full-minival-200-20260803-gpu0-w4/08_03_2026_01_16_39_731179/` | intended primary raw source for the historical 200-task phenotype audit | directory identity must be reconciled to the 173/200 summary before scientific interpretation | **current experiment should locate, hash and reconcile this first** |
| E-FULL200-AUDIT | B | `/nvme2/user/qyy/SafeVLA/diagnostics/end_causal_audit/analysis_validation_report.md` and any raw metric/table files it identifies | historical SR 173/200, 27 failures, Safety Cost summary/distribution | historical derived report until raw 200 rows are independently rechecked | trace every derived statistic back to E-FULL200-RAW |
| E-FULL200-PRESERVED-ANALYSIS | B/C | 02 Project artifact `分析output(1).md` (not directly readable by Executor unless copied); substantive claims are migrated into `research/LEGACY_RESEARCH_STATE.md` | category counts, 16 sub-horizon vs 11 horizon failures, task-difficulty associations, target-room visitation | migrated summary is not raw proof | Executor should reproduce needed numbers from E-FULL200-RAW, not trust the chat-era table blindly |
| E-END-AUDIT | A/B | `/nvme2/user/qyy/SafeVLA/diagnostics/end_causal_audit/gate_review.md`, `runtime_manifest.json`, `analysis_validation_report.md`, `checkpoint_integrity_report.md` | forward topology, logger/action invariance in audited scope, one illegal-end canonical case, checkpoint/runtime evidence | one illegal end cannot establish population frequency or mechanism | current size audit may cite as background only; do not rerun |
| E-RESET-AUDIT | A/B | `/nvme2/user/qyy/SafeVLA/diagnostics/end_causal_audit/reset_state_report.md`, `reset_two_episode/reset_transition_summary.json` | counter/cache persist across reset; tested mask isolated old cache in short case; max_steps=500 vs horizon=600 | no causal SR/logit effect established | treat as confound/open hypothesis; no reset treatment in current experiment |
| E-PROBE-LOG | C | historical log path/artifact from `SafeVLA_Original`; preserved log reported `[10000,3,512]` and 228/10000 positives; historical server path for PT file: `/home/amax/public/users/qyy/SafeVLA_Original/probe_data_worker0.pt` | a real 3-layer hidden-state collection occurred; label was close-and-visible / stop-legality-like | PT-Guard modified the trajectory; artifact provenance/version uncertain; no episode IDs; not small-target evidence | **do not use as current size evidence**; verify existence/hash before any future Probe audit |
| E-PROBE-LEGACY-AUC | C | preserved 1000-sample PT artifact examined by PI; historical AUC triplet 0.955/0.982/0.992 was reproducible under a plausible stratified step-level analysis | stop-legality-related information was linearly decodable in that exploratory dataset; no observed deep-layer collapse in that analysis | original analysis manifest/script not preserved; temporal leakage risk; no causal Actor-use claim | background only; any formal Probe must be rerun cleanly after phenotype validation |
| E-DIAGNOSTIC-PLAN | C/design only | old `model plan.txt` / chat-era design proposing H1-H12 hooks, P1 visibility, P2 category, P3 location, P4 oracle-action and Action Logit Lens | records proposed diagnostic directions | does **not** prove those probes were executed; old distance-based size bucket is invalid for physical size | do not treat planned metrics as results |
| E-GRPO-EARLYSTOP-TREATMENT | C/treatment | legacy directories under `/home/amax/public/users/qyy/SafeVLA_Original/eval/ObjectNavType/`, including `v100safevla---专利版本` and `v100safevla---专利版本-修复幻觉性早退问题` (verify before use) | hard/oracle-like done intervention rescued some failures but caused roughly offsetting regressions and long-loop/boundary-oscillation behavior | action selection was modified; not official B0 and cannot establish B0 mechanism | negative design evidence only; excluded from current size analysis |
| E-SUB120-TRACE | C/B single-case | `/nvme2/user/qyy/SafeVLA/eval/objectnav-full-minival-200-20260803-gpu0-w4/ObjectNavType/safevla-objectnav-full-minival-200-20260803-gpu0-w4/08_03_2026_01_16_39_731179/sub120_action_probs_approx.csv`, `sub120_actions.csv`, `sub120_actions.json` | one 600-step failure had video-rendered approximate p(done)<~0.018 throughout, arguing against "high end probability but stochastic sampler never picked it" in that case | video-rendered quantized probability, not exact logits; single case; planned replay not completed | do not run replay in current experiment; preserve as later termination-calibration evidence |
| E-RESEARCHER-SMALLTARGET-OBS | D | researcher report in Project: cup/mug-like objects, basketball, kettle, apple etc. appeared around ~50% SR in an evaluation summary | motivates checking category/size phenomenon | conflicts with the preserved 173/200 category table for at least some categories; run identity unknown | current experiment must reconcile or label the observation as a different/unknown run |
| E-BASELINE-CONTRACT | A/project rule | `research/BASELINE_CONTRACT.md` on control branch and original project rule | B0 cannot be silently changed; diagnostics may be read-only; behavior changes are separate treatments | does not choose a mechanism | binding for every experiment |

## Critical discrepancies to resolve

### D1 — category-level SR discrepancy
The preserved 173/200 analysis and the researcher's later "~50% for several small-object categories" observation are not numerically the same. Do not average or merge them. Recover the exact raw run/aggregation behind each before comparing.

### D2 — Probe artifact identity
The historical log's first 1000-step checkpoint had 12 positives, while a separately preserved 1000-sample PT artifact later inspected by PI had 56 positives. They are not safely the same snapshot. Do not attach the AUC triplet to the 10k log or to a specific 1k checkpoint without provenance.

### D3 — modified-policy vs B0
Any run using GRPO, predictor reward reranking, PT-Guard action rewrite, oracle-based done suppression, steering or other action intervention is a treatment/legacy branch. Its SR and failure categories must never be silently mixed into official B0 statistics.

## Current experiment evidence subset

For `EXP-SMALLTARGET-PHENOTYPE-001`, the allowed scientific evidence set is:
- E-B0-OFFICIAL-SOURCE for metric/task semantics;
- E-B0-001A only for provenance context;
- E-FULL200-RAW as the primary episode-level outcome source;
- static task/scene metadata tied to those exact tasks;
- E-FULL200-AUDIT only as a reconciliation target, not as a replacement for raw rows.

Explicitly excluded from the primary size association:
- E-PROBE-LOG / E-PROBE-LEGACY-AUC;
- E-GRPO-EARLYSTOP-TREATMENT;
- E-SUB120-TRACE except optional background note;
- trajectory-derived max visible pixels as a definition of physical size.

## Secret / artifact hygiene

Some old chat-era or exported artifacts may contain credentials or full environment dumps. Do **not** copy historical API keys, authenticated remote URLs, tokens, or complete environment dumps into GitHub or handoff files. Record only sanitized paths, hashes, dependency versions and non-sensitive configuration required for provenance.
