# EXP-SIZE-RUNTIME-METADATA-001: BLOCKED

The run stopped at 2026-09-20T13:25:25Z after 44 scene initializations (24 preflight + 20 full). Recorded exception: BrokenPipeError at the full-loop progress print after task 20 was saved. This is an output-channel interruption, not an observed mapping or AABB failure. The exact pipe closure trigger is not established. The final print also uses the same pipe; the process exit code was not recovered.

Preflight passed: 12 tasks loaded twice, 18 target comparisons, exact geometry equality and maximum absolute/relative dimension difference 0. Partial full-pass coverage is 26/368 valid exact-mapped targets and 20/200 complete tasks. The remaining 342 targets / 180 tasks were not attempted; task medians remain empty. Offline validation independently cross-checks snapshots, CSV geometry, repeatability and medians; see validation_report.json and coverage_report.md.

AI2-THOR build: 966bd7758586e05d18f6181f459c0e90ba318bec, CloudRendering. Archived requirements and committed initialization source bind the historical build; current executable/assembly hashes identify this run. Historical executable bytes were not separately hashed contemporaneously, so historical binary byte equality is not independently proven.

The descriptor is world-axis scene-instance AABB immediately after CreateHouse, with autoSimulation=False and no post-load physics advancement, task teleport, navigation or policy. It is creation-state extent, not canonical intrinsic volume or demonstrated post-settling evaluation geometry. Applicability to later phenotype work remains a PI decision.

Actual resources: one simulator graphics GPU (physical GPU 3, observed Vulkan device argument 4), 44/224 scene initializations, 0 SafeVLA/ObjectNav episodes, 0 model/checkpoint loads, 0 Actor/Critic forwards. Geometry extraction read no outcome files/columns and performed no size-success association. Frozen control documents read for handoff are not extraction inputs. No simulator process remained at handoff verification; original development HEAD, tracked diff and status are unchanged.

H-RUNTIME-AABB is not established for complete coverage. H-RUNTIME-GAP is not established by this output failure. H-SIZE remains untested. No imputation, category, visible-pixel or distance proxy was used.

Next actor: PI. One proposed next action (not approved): review this partial handoff and, if warranted, issue a fresh cycle/design for transport-resilient extraction, explicitly handling prior measurements and a new scene budget. No retry, resume or further experiment was started. Executor STOP after handoff publication.
