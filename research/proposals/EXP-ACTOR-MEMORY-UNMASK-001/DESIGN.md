# EXP-ACTOR-MEMORY-UNMASK-001

Status: DRAFT_FOR_PI_REVIEW. Preparation/CPU-only authorized by user on 2026-10-28. Online/GPU NOT_AUTHORIZED. No claim/instruction for this experiment. Control state remains untouched.

Research Question: Does exposing pre-Episode cached Actor K/V improve official ObjectNav SR and Safety Cost?

Current Evidence: P2-002 source 06cde9adeb91d47c062a11e72e90c7edf56fb398 retains cache/counter across episodes, but masks prefix tokens. Genuine decoder uses boolean True=accessible SDPA masks. Separate root/reward/cost models exist. Cache persistence alone is not evidence of use. Old reset experiments concern a different intervention and are not controls here.

Hypothesis: Useful prior episode information changes root Actor decisions and improves SR without increasing Safety Cost.

Competing Explanation: Retained observations/goal/action history from unrelated tasks causes interference; effects arise from position/rollover interaction or stochastic trajectories rather than useful memory. No benefit is assumed.

Control: Mask ON (original episode isolation), cache/counter retained.

Treatment: Mask OFF on root Actor only, cache/counter retained; decoder default causal mask. Actor reset OFF in both arms. The default remains ON for every instance; configure_root_actor changes only the already constructed root. Never pass a constructor flag via shared Critic kwargs.

Unique Variable: root Actor cross-Episode attention mask.

Fixed Conditions: policy safe_objnav.pt SHA256 05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301; DINO weights b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9, source hashes/reference dependencies from diagnostics/actor_logits/reference_identity.json. Official ObjectNavType/minival dataset SHA fa0dffc2dece071716dcf648e19c6944caf1621ee24dadc6996eadcd46eee04a and ordered 200 TaskSpecs manifest SHA 8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0, seed123, shuffle=true, one persistent worker0, stochastic sample then mode exactly as original agent, test_augmentation=true, original sensors/action mapping (end index4 of20), horizon600, max_steps1000, same GPU/dependencies/assets/scene ordering per arm. Restart process with seed123 before each complete arm; no per-Episode Actor reset. Critic handling and prev_actions masks unchanged. Official success and five cost components unchanged.

Metrics: official SR successes/200; Safety Cost sum/200 and total, five component sums (dangerous/corner/blind/fragile/critical using actual COST_KEYS), median/p90/max and full distribution; Episode length and early-end descriptive. Task-aligned SR/cost differences, paired bootstrap 95% intervals (10000 replicates, analysis seed20261028), discordant success pairs/exact McNemar; descriptive because episode histories and stochastic environments differ. Joint success criterion: positive SR difference and nonpositive mean cost difference; stronger confirmatory claim requires SR lower CI>0 and cost upper CI<=0. No equivalence claim from nonsignificance.

Expected Result: CPU fixed-input old history mutation leaves Mask ON logits unchanged, alters Mask OFF logits; fresh paired online arms may improve both endpoints.

Falsifying Result: no logit sensitivity with mask OFF, history sensitivity under mask ON, Critic/state/RNG/forward invariance failure, or online no improvement/worse cost. CPU sensitivity proves access/use only, not beneficial memory or benchmark improvement.

Alternative Explanations: synthetic post-perception replay lacks actual scene semantics; different previous episode lengths alter later exposed histories and rollover phase; same seed does not guarantee identical observations/actions/augmentation RNG; scene/goal leakage and wrong-goal interference; GPU numeric differences; logger noninterference unproven on GPU. CPU test fixtures cannot waive GPU evidence.

Allowed Code Changes: actor file per-instance switch and conditional mask; new diagnostics/actor_memory_unmask/ only and new proposal packet. New entrypoint must use root-only configurator after construction, no reset callback; retain original evaluator/official metrics behavior and logger interception without extra forward.

Forbidden Code Changes: historical files/outputs, prev_actions masks, reward/cost Critic, attention cache implementation, counter/max_steps/rollover, weights/training/action map/metrics, old control or claim bindings; extra live policy forward; reusing old reset/OFF results as new controls.

Stop Conditions: any unapproved GPU/online work, wrong/missing fresh approval+claim+CI, old Claim, source/resource/TaskSpec/hash/import drift, multiple variables, incomplete/misaligned episodes, logger/cache/critic invariants fail, unavailable disk/GPU/network/expiry, output exists, budget reached or process failure. Preserve failed evidence, stop; no retries, automatic GPU switch, guard relaxation or foreign-process interruption.

Required Artifacts: immutable source revision+patch; manifest/config/full commands/exit status/allowlisted env/module paths; checkpoint/DINO/asset/source hashes; saved fixed input+cache provenance; logits/P(end)/JS/counter/cache checks and tests; raw logs; resource preflight; 200 task manifest; PI review notes and SHA256 artifact index. No checkpoints/tensors in Git; large replay state remains server optional with hashes.

## Stage 1 CPU protocol (authorized)

CPU unit tests use genuine decoder and source-extracted forward, original positional encoder and supplied embeddings. Compare default vs exact original forward, causal batch behavior, prev-actions null token, genuine warm-up cache mutation, root-only configuration, original separate Critic forward outputs/cache/one-call counts, real 1000 rollover. Fixed-input replay independently loads checkpoint root decoder, last_actions_embed, Actor head/time encoder with strict subsets; retains real 512-d/3-layer architecture. Deterministic synthetic post-perception embeddings seed123, 4-step Episode A warmup then same Episode B first input. Compare ON/OFF and final-layer K/V mutations separately; save raw logits, p(end), JS natural logs, warmup cache and inputs, SHA256. This is isolated Actor component replay, not full vision-language policy or online benchmark.

## Full 200-task online plan (future independent authorization)

200 unique TaskSpecs per arm: MASK_ON200 + MASK_OFF200 = 400 live Episode starts total, max240000 decisions (600 each). A total200 budget would yield100/arm and is a different smaller protocol requiring PI amendment. Run fresh Mask ON first then OFF as predeclared; sequential on same selected physical GPU, clean process+same seed/manifest each arm. Preserve order, worker assignment and all failures. Old OFF200 is historical only. Do not alternate masks per episode, because that contaminates history. No opportunistic stopping based on endpoint direction.

Before launch PI must acknowledge outstanding terminal control, stage new unique cycle, review exact source SHA/design, independently authorize 1 GPU/400 starts with expiry and maximum walltime/output, approve control CI, then executor publishes unique Claim and passes distinct Claim CI. A new reviewed launcher/evaluator must bind this experiment, not use run_p2_full400.sh or alter old scripts. Current preparation intentionally provides NO executable GPU bypass. Final entrypoint must verify source clean/pinned and true remote Claim/CI before heavyweight imports, check source diff allowlist/resources, then attach configurator once at root build and no reset callback; log mode and masks with original one-forward path. Pre-launch CPU checks plus future GPU logger equivalence need separate scoped approval; historical waiver does not transfer.

Resource request (proposal, not allocation): 1 V10032GB from0–3, same UUID both arms, no foreign compute PID, used<=4000MiB, util<=30%, free>=6824MiB (historical conservative floor, revalidate peak for this new treatment), 3 stable polls30s apart plus immediate admission check. No hold/kill. Host RAM>=16GiB available, output capacity>=90GiB plus reserve10GiB; budget output90GiB. Queue<=12h; execution<=12h for400 episodes (historical OFF200 wall bound4.16h is sizing evidence only), absolute approval window must contain both; if insufficient bound STOP before second arm, preserve partial. Peak cache shape remains unchanged, access length grows within retained prefix only. Resource budget and GPU equivalence remain PI decisions.

Run integrity: each arm must validate200 complete archived episodes and exact task raw_spec/order/worker; assert every decision one root/reward/cost decoder call, mask mode, no boundary Actor reset, source/checkpoint/assets identities and official CSV totals. If either incomplete do not present full benchmark contrast. Analyze all200 paired tasks only after both integrity gates; release own processes, archive report+index and return PI, STOP.
