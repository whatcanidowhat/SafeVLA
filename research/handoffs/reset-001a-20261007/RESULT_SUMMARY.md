# EXP-RESET-001A result

Status: AWAITING_PI_REVIEW; next actor PI; executor STOP after publication.
Instruction: `0493548a70aa192bb04457bc15e9343d934bb259`. Unique claim: `c4ab4b395817f439a0239514f3469480aa0d8500` / `24f414396c8f4ef39b34b179dee8c6fa`.
Claim recovery used one explicit non-force push of the pre-existing claim after identity and dry-run checks. [Claim CI](https://github.com/whatcanidowhat/SafeVLA/actions/runs/37752169988) succeeded before experiment execution.

Outcome: **R1**. A/A OFF passed; both offline conditions passed repeat stability.
Fixed real input sequence: 600 decisions. CARRY first rollover: local step 200 (zero based).
First logit divergence above declared tolerance: 200.
First argmax divergence: 205.
First common-RNG sampled-action divergence: 200.
Before/boundary/after maxima: `{"pre_rollover": {"n": 200, "max_abs_logit": 3.814697265625e-06, "max_abs_hidden": 2.86102294921875e-06}, "boundary": {"n": 1, "max_abs_logit": 5.018692970275879, "max_abs_hidden": 3.9560751914978027}, "post_rollover": {"n": 399, "max_abs_logit": 12.482795715332031, "max_abs_hidden": 6.683389663696289}}`.

A valid Actor carry package reproducibly changes the policy distribution and at least one decision under the tested identical current inputs. This supports the state-carry-to-Actor-decision pathway; it does not establish SR or Safety Cost impact.

## Execution and validity

Live budget used: 1 GPU, 3 episodes started, 3 completed. Live capture used an isolated byte-verified copy of accepted official B0 under the approved execution worktree, with only the accepted local-DINO infrastructure adaptation. The original development tracked diff and official runtime snapshot remained unchanged.

P0 identified independent Actor/Reward/Cost decoder state. The local review patch adds an explicit OFF/ON root-Actor-only counter plus K/V reset; it was not applied or committed to B0. The treatment method is loaded in the offline harness only. The exact imported decoder uses scaled-dot-product attention; the similarly named architecture decoder is not the runtime implementation.

Task selection was frozen by expert_length metadata, not observed outcomes. It is a stress-input selection, not an unbiased benchmark sample. One fixed real input sequence is required; no performance rate is inferred from capture episodes. A/A comparisons use saved inputs and common RNG, not equality of independent stochastic live trajectories.

The preparation-only FileExistsError occurred before model imports and consumed no episode/GPU experiment budget; original source and log remain archived. The directory-copy fix did not change the experimental design, and an exclusive model-phase marker prevents restart. Any scientific-stage failure is retained without automatic retry.

## Evidence and limits

See `decoder_reset_audit.md`, `decoder_state_map.json`, `reset_fix_design.md`, `reset_fix.patch`, `aa_off_equivalence.json`, `rollover_stress_result.json`, `rollover_trace.csv`, `RUN_MANIFEST.json`, `validation.json`, and the runnable executor. ARTIFACT_INDEX binds Git-readable small evidence and optional server-side tensors/source snapshots/logs by SHA256 and byte count. Raw tensors are server-only and are not represented as independently reviewed by PI.

Counter and K/V form one state package; this does not separately attribute their roles. Fixed teacher forcing prevents action divergence from changing subsequent inputs. No Probe, Stop Gate, size experiment, online reset-treatment performance comparison or 200-task evaluation was run. SR and official Safety Cost effects remain untested.

Next action: PI reviews validity and the permitted interpretation. No next experiment or PI acknowledgement is issued by this executor.
