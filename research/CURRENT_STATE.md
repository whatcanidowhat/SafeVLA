# Current State

Updated: 2026-09-18（Asia/Shanghai）
Mode: PI has approved EXP-SMALLTARGET-PHENOTYPE-001 design; execution authority is controlled exclusively by LOOP_STATE. Old 001B preflight remains revoked.

## Verified Facts

- EXP-B0-REPRO-001A is COMPLETED / PROVENANCE PASS for its narrow goal: two Reference + two Repeat episodes on the same executable B0, stable 2/2 task pairing, and traceable source/resource identity. It does not prove full-200 performance equivalence.
- Executable B0 remains official reference 2aa82559d272b5f888e53433e258914057f15bed plus the manually accepted local-DINO infrastructure adaptation only.
- Historical full-200 summary reports ObjectNav SR=173/200=0.865 and 27 failures; this remains historical evidence until episode-level raw records are re-audited.
- A historical canonical state recorded stop_legal=false while Actor end probability was approximately 0.993365 and greedy action was end. This establishes an illegal/premature-end phenotype in at least one state, not its mechanism.
- Source audit confirms end is a learned Actor action. Actor hidden beliefs are mapped by a LinearActorHead into an action distribution; end is sampled or mode-selected like other actions. Only after end is executed does ObjectNav successful_if_done() check whether a valid target is visible in the navigation camera within maximum_distance=2.
- Therefore end_prob≈0.993365 is not an "end Probe"; it is the Actor probability assigned to the end action. The historical Probe AUC values 0.955/0.982/0.992 are a separate exploratory layer-wise decoder Probe for a close-and-visible / stop-legality-like label.
- Those historical Probe results are hypothesis-generating only: the run contained PT-Guard action rewriting, the saved artifact lacks full episode/task provenance, and step-level splitting risks temporal leakage. They do not prove small-target representation.
- A later topology audit established that a formal Actor Probe must use the root actor_critic.decoder branch; reward-critic and cost-critic hidden states must not be mixed into Actor representation claims.
- The old EXP-B0-REPRO-001B-PREFLIGHT approval expired on 2026-09-14T07:34:44Z and control state showed claim_id=null. It has been revoked rather than executed.

## Researcher Observation — not yet independently audited

The researcher reports several target categories such as cup/mug-like objects, basketball, kettle and apple showing navigation SR around 50% in evaluation summaries.

This is a valid phenomenon candidate. It is not yet evidence that "small physical size" is the cause: category, sample size, house/scene difficulty, expert path length, initial distance and occlusion remain alternatives. Basketball/kettle also demonstrate that low-SR category and "small object" cannot be treated as synonyms.

## Active Hypotheses

- H-SIZE: policy-independent target physical size contributes to lower ObjectNav SR.
- H-CATEGORY: semantic category explains low SR independent of size.
- H-DIFFICULTY: hard houses/tasks, longer expert paths, farther starts or occlusion explain the pattern.
- H-SAMPLE: category-level ~50% SR is unstable because n is small.
- H-REPRESENTATION / H-READOUT / H-EXPLORATION / H-TERMINATION / H-SAFETY remain downstream mechanism hypotheses. They should not be selected before the target-size/category phenotype is made precise.
- H-RESET remains a known confound/risk hypothesis; its causal effect on SR is still unproven and it is not the current mainline.

## Unsupported Interpretations

- "Small targets are already proven to cause low SR."
- "AUC 0.992 proves the model represents small objects."
- "end_prob≈0.993 is a Probe score or a success probability."
- "High end probability proves representation loss, readout mismatch, or safety-induced termination."
- "Safety critic directly gates inference and forces end."
- "Trajectory max visible pixels is target physical size." It is policy-dependent and may be a consequence/mediator of search and approach behavior.

## Highest-Value Uncertainty

Before more Probe or intervention work, determine whether the reported low-SR categories are statistically credible and whether policy-independent physical target size explains any of the failure pattern beyond sample-size and task-difficulty alternatives.

## Unique Next Experiment

EXP-SMALLTARGET-PHENOTYPE-001: zero-rollout audit of the historical full-200 episode-level evidence plus static target-object 3D size metadata. It must not start a model, GPU inference, AI2-THOR or any episode. If the raw evidence cannot support this analysis, return BLOCKED rather than rerunning evaluation.

## Control State

PI has approved EXP-SMALLTARGET-PHENOTYPE-001. The authoritative execution gate is LOOP_STATE; once it transitions to APPROVED_FOR_CODEX, Codex may claim only this zero-rollout audit. Full 001B, Probe reruns, video-mechanism studies, reset treatment and Safe-vs-IL comparisons remain NOT_AUTHORIZED.
