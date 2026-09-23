# Decision Log

## 2026-09-11 — DEC-LOOP-001：建立最小闭环

依据：用户要求遵照“分支 · 02｜SafeVLA研究”（ID 6a7b4d04-abf0-83ee-991b-e3e9a9821032）末尾构建要求，先完成共享事实层和skill，不启动新实验，再进行人工监督的两轮dry run。

决策：创建AGENTS、五份research文件及safevla-research-loop。中文正文保留工程字段；约束证据程序，不固定研究机制。实际服务器路径为 /nvme2/user/qyy/SafeVLA。

范围：协议构建、已有证据只读核对和文件验证。保留原有代码/产物/工作区修改，不扩大为提交发布、发送聊天消息或后台持续运行。

## 2026-09-11 — DEC-LOOP-002：继承Gate B阻断

证据：diagnostics/end_causal_audit/gate_review.md、reset_state_report.md、reset_two_episode/reset_transition_summary.json。当前tracked diff哈希与历史manifest一致；本轮未重新运行模型或哈希checkpoint。

决策：Gate B继续FAIL。counter/cache保留为已有记录，rollover对action/SR/cost的影响仍需隔离验证，不能静默修复。

为何不直接Probe/扩大诊断：状态来源未隔离会削弱hidden-state解释。为何不直接oracle：不能优先解决当前运行状态不确定性。

下一步：唯一草案EXP-RESET-001，先核实重放输入/调用语义，明确单变量、参数和执行范围。这是候选设计，不是已批准的最终实验。

## 后续记录

记录日期、证据路径/版本、研究问题、所选与未选方案理由、结果如何改变假设、停止/继续条件、唯一下一实验。负责人读取代码和原始产物，不只接收聊天总结。

## 2026-09-11 — DEC-REPRO-001：用户批准仅001A并拆分A/A设计

用户最新评审覆盖旧protocol-only范围，仅授权EXP-B0-REPRO-001A的provenance smoke；001B和完整200-task禁止。
采用Reference Run / Repeat Run；历史86.5%只作historical reference。预注册每run两个task、两次共最多四个episode、stochastic/augmentation/shuffle及显式参数；不追求SR结论。
NEXT执行前已更新，旧设计与旧状态保存在001A/preflight。原skill中一般性的Control/Treatment字段不强加于本A/A设计。

## 2026-09-11 — DEC-REPRO-002：按身份停止条件终止于preflight

比较官方共同祖先、HEAD已提交变更和working tree后，发现worker的预决策successful_if_done/dist_to_target_func调用在关闭诊断/影子开关时仍存在，可见性缓存未命中路径会调用底层环境。
尚未建立该调用路径与官方行为等价证据；既有RNG/forward/action链验证不充分覆盖环境事件语义。不能通过冻结相同修改版哈希直接宣称官方B0成立。
用户明确要求B0身份无法冻结即停止，因此Reference Run和Repeat Run均不启动。没有静默修复、没有扩大实验；checkpoint/DINO只进行文件SHA256计算。
greedy history修改在本stochastic设置不激活，不把它误报成stochastic已受影响。原Gate B保留，但本轮停止不等于要求先修reset。
交付research/runs/EXP-B0-REPRO-001A/RESULT_SUMMARY.md；001A BLOCKED_PREFLIGHT，001B NOT_AUTHORIZED。人工评审B0适配集合和无侵入性证据后才可恢复001A。

## 2026-09-12 — DEC-B0-CANDIDATE-001

用户仅授权B0-CANDIDATE-AUDIT：通过官方GitHub固定commit API及tree SHA核验reference，创建独立worktree codex/b0-candidate-audit，不修改开发诊断代码。
候选保留所有官方policy/worker/task/reset源码；不复制开发分支行为修复。只保留路径配置与本地DINO加载适配，后者分类B，需要另行获准的runtime验证。
输出完整官方→候选patch和逐文件相等检查，研究报告research/B0_CANDIDATE_AUDIT.md。静态检查不等于运行等价；001A不自动恢复。
不运行模型、不启动AI2-THOR、不跑episode、不commit/push。完成后等待人工评审候选及下一次授权。

## 2026-09-12 人工批准：executable B0的DINO基础设施适配

因服务器网络限制，人工批准candidate使用本地DINOV2_REPO源码和DINOV2_CKPT权重，作为executable B0必要infrastructure adaptation，不作为treatment或研究变量。
保留source="local"、pretrained=False、显式checkpoint、load_state_dict(..., strict=True)及文件缺失fail-fast。
不执行EXP-B0-DINO-EQ-001，不要求与在线torch.hub loader比较RNG/bitwise equivalence；此决定不声称两加载路径已实测等价。
每个正式run仍须保存DINOV2_REPO、DINO源码snapshot/hash、DINOV2_CKPT及SHA256、实际runtime module paths、strict load成功状态。失败或资源漂移必须停止，不回退为随机初始化。
其他Baseline规则不变；不得修reset/counter或增加预决策environment/controller查询。001A仅Reference/Repeat各最多2个episode，001B/200-task仍禁止。

DEC-DINO-INFRA-APPROVED：本决定覆盖此前DINO运行等价阻塞，保留旧审计为历史记录。恢复001A，不运行新DINO等价实验。使用独立runpy/profiling观察器，仅在官方已有load_state_dict及episode函数返回时记录现有返回值，不hook forward、不额外查询环境。候选源码保持不变。


## 2026-09-12 — EXP-B0-REPRO-001A完成并停止

Decision: 接受001A为COMPLETED / PROVENANCE PASS（限单worker、两个task的来源与产物验证）；保留全部结果，停止等待人工评审。
Evidence: runs/EXP-B0-REPRO-001A/execution_20260912/validation.json全部检查PASS，pairing.json配对2/2；两run各2 episodes、exit0。
Identity: official HEAD 2aa82559d272b5f888e53433e258914057f15bed，candidate patch f8ff5b1c07a07d5ff717f5f262734dc3d7345ca0d88e5cd5c1607a245e6927db一致；DINO/source/checkpoint/runtime路径均一致且strict加载成功。
Interpretation: 仅支持最小smoke的可追溯性。轨迹长度Reference28/56、Repeat27/62，不解释因果，不要求逐bit相同，不作SR/Safety Cost结论，不修reset/counter。
Adaptation: 本地DINO继续作为人工接受的executable B0基础设施；未执行DINO-EQ，也不将RNG/bitwise/在线等价作为阻塞点。
Preservation: 原开发目录166文件未变；baseline合同原文逐字保留，只追加获批适配；official policy/environment/evaluation代码本轮未修改。
Next action: 唯一下一动作是人工审阅RESULT_SUMMARY并冻结001B完整任务、worker/调度、command、预算与容差；001B仍未授权，未启动。
No commit/push；未执行001B、200-task、DINO-EQ或reset实验。历史preflight失败仍归档，不删除或冒充本次run。

## 2026-09-13 — DEC-HANDOFF-BOOTSTRAP-001

人工评审接受H1 PASS及001A已完成；不得重复001A，001B未授权。授权独立research-loop控制面建设、白名单commit/push，不授权任何实验。创建orphan控制分支以排除开发/runtime源码与旧提交历史。初始LOOP_STATE=PI_REVIEW/next_actor=PI，EXP-LOOP-HANDOFF-001仅DRAFT。旧索引按历史身份保留，canonical ARTIFACT_INDEX另建。构建测试仅合成Git fixture，绝不冒充真实PI批准、claim或ack。完成发布后STOP等待PI独立读取。


## CONTROL-PROTOCOL-FIX-001 — 2026-09-14

用户仅授权修复PI_REVIEW逐文件设计staging校验。只有PI_REVIEW、next_actor=PI、
NOT_AUTHORIZED、instruction_commit=null、claim_id=null全部成立时，允许两份设计
各自为DRAFT或APPROVED；不授予执行权。其他状态必须保持两份设计APPROVED及
authorization.status=APPROVED，既有完整性和状态转换检查不放宽。
新增真实顺序合成回归和可执行状态拒绝不一致回归；完整验证真实research-loop历史，
包括93e1f66、18eef394、72b9fa929。不修改LOOP_STATE或实验设计，
不增加状态转换，不rewrite/squash/force push，不领取或执行真实handoff任务。

## 2026-09-14 — PI复核CONTROL-PROTOCOL-FIX-001

PI已通过GitHub独立读取并审查commit f2d477c1befb74acb5fc7f76b89ab6cd41e5e4c9、修复后的validator与成功CI。确认修复仅放宽PI_REVIEW且NOT_AUTHORIZED的分阶段设计写入；可执行状态仍要求Markdown/JSON均为APPROVED且authorization为APPROVED。原EXP-LOOP-HANDOFF-001的CONTROL_ONLY_NOOP授权在其既有时限和0 GPU/0 episode预算内继续有效；本记录不修改LOOP_STATE、不扩大任务范围、不授权001B。Codex如领取任务，instruction_commit必须绑定本PI复核记录所在的最新research-loop HEAD，且仅在该HEAD的CI成功后执行。

## 2026-09-14 — EXP-LOOP-HANDOFF-001 Executor handoff

Explicit PI/user resumption authorized only CONTROL_ONLY_NOOP at 0dba6b748f26a782a595edeb53f02083850631cb. Approval CI and claim 16748eab9993b568faf2e78213df29a6ebb31d81 CI succeeded before one print-only no-op (exit0). Publish AWAITING_PI_REVIEW / next_actor=PI with instruction_commit and claim_id unchanged, then STOP. PI acknowledgement remains pending; no full-loop completion claim, baseline change, second experiment or 001B. Evidence: research/handoffs/handoff-001-20260913/.


## 2026-09-18 — DEC-SMALLTARGET-MAINLINE-001：从001B协议工程切回小目标现象验证

Evidence review:
- 001A已满足当前只读诊断所需的最小B0 provenance；继续优先完善full-200 A/A协议的边际科学价值低于验证主研究现象。
- 研究者报告若干类别SR约50%，但类别不等于尺寸；basketball/kettle等例子暴露“低SR类别=小物体”的逻辑跳跃。
- 历史0.955/0.982/0.992 AUC来自close-and-visible/stop-legality类Probe，不是small-object-size Probe，且存在PT-Guard轨迹与step级切分泄漏风险，不能用于证明小目标representation。
- 源码核对确认end由Actor action distribution选择；<=2m+nav可见只在end执行后作为successful_if_done判据。因此单个end_prob≈0.993365只证明非法end phenotype，不能定位representation/readout/safety机制。
- 旧EXP-B0-REPRO-001B-PREFLIGHT授权已过期且control branch无claim/result，已撤销。

Decision:
唯一下一实验改为EXP-SMALLTARGET-PHENOTYPE-001。先用既有历史full-200原始结果和静态scene/task metadata回答“低SR类别是否真实、physical target size是否与失败相关”。不运行模型，不启动AI2-THOR，不新增episode。

Causal discipline:
- 物理尺寸必须来自policy-independent 3D metadata；distance和trajectory max visible pixels不得冒充size。
- 先报告每类别n/SR/Wilson CI，再做连续size关联和有限难度控制。
- visible pixels / premature end / horizon failure只作为下游failure morphology。
- 若原始200条episode证据、stable task identity或静态size metadata不能复原，则BLOCKED；不得以重新跑评估替代。

Next:
完成PI staged review后，若设计文件与control validator一致，则仅授权0 GPU / 0 episode离线审计。Probe、视频机制、readout干预、安全对照、reset treatment均等待本轮结果。


## 2026-09-18 — PI批准 EXP-SMALLTARGET-PHENOTYPE-001 执行

用户明确回复“批准执行 EXP-SMALLTARGET-PHENOTYPE-001”。PI接受此前冻结设计，不修改研究问题、主假设、竞争解释、唯一解释变量、指标或停止条件。执行预算保持0 GPU / 0 episode；仅允许对既有historical full-200原始结果、task specs和静态scene metadata做只读/离线审计。

授权不包含full 001B、任何新rollout、Probe重跑、视频机制实验、reset treatment、Safe-vs-IL比较或baseline/runtime修改。最终执行权仅由随后单独的LOOP_STATE PI_REVIEW→APPROVED_FOR_CODEX状态提交授予；Codex必须先claim并等待claim CI成功，再执行一次并提交handoff后STOP。


## 2026-09-18 — DEC-LEGACY-MIGRATION-001：将 02｜SafeVLA研究 迁入共享事实层

背景：网页版PI可以检索Project中的02历史研究对话，但桌面端Executor不能自动读取这些聊天。若关键实验、Probe限制、被否定解释和服务器证据位置只留在聊天中，会导致PI与Executor对研究状态理解不一致。

决策：不上传完整聊天原文，而是进行证据化迁移：
- 新建 `research/LEGACY_RESEARCH_STATE.md`，按已证实事实、观察、待验证假设、被否定/更正解释和历史branch inventory整理。
- 新建 `research/EVIDENCE_REGISTER.md`，把研究主张绑定到raw/server/Git证据位置，并标明每项证据可支持与不可支持的结论。
- 原始服务器证据优先于迁移摘要；冲突必须在handoff中报告，禁止用聊天摘要覆盖raw evidence。
- 明确保留三类关键边界：历史Probe不是small-object Probe；PT-Guard/GRPO干预分支不是B0；后来的“若干小类别约50% SR”观察与保存的173/200类别表存在run identity待核对问题。

对当前实验的影响：不改变 `EXP-SMALLTARGET-PHENOTYPE-001` 的研究问题、变量、预算或停止条件，也不增加任何rollout/Probe/视频/replay授权。Executor在claim后执行前必须先读两份迁移文档，并从既有raw full-200证据重新核对需要的统计。

安全：历史导出材料中可能含敏感环境/认证信息；迁移文档只记录非敏感路径、哈希和研究结论边界，不复制API key、token或认证URL。


## 2026-09-18 — DEC-CONTROL-REPAIR-002：修复 small-target 授权链 CI 失败

Failure evidence:
- GitHub Actions run 35326062502 failed in `python scripts/validate_research_loop.py --history`.
- First observed error: `LOOP_STATE schema violation: deque(['required_outputs'])`.
- Root cause 1: commit `902a99eb...` opened the new small-target PI_REVIEW cycle with `required_outputs=[]`, violating the existing schema's minimum four handoff outputs. This invalid state was then preserved across several staged design commits.
- Root cause 2: the first `APPROVED_FOR_CODEX` state had scientifically adequate Markdown instructions, but `NEXT_EXPERIMENT.json` lacked several validator-required machine-readable fields (`competing_explanation`, `reference`, `repeat_or_treatment`, `alternative_explanations`, `command`).
- No Executor claim occurred: `instruction_commit=null`, `claim_id=null`; budget remained 0 GPU / 0 episode. Therefore this was a control-plane staging error, not a research execution incident.

Repair:
1. Do not rewrite/squash/force-push the bad history.
2. Add a narrow append-only-history exception only for the exact 2026-09-18 unclaimed small-target staging incident; all future/current states remain subject to normal validation.
3. Add and test a PI-only `APPROVED_FOR_CODEX -> PI_REVIEW` revocation edge that is legal only before claim; claimed/running work cannot use it.
4. Revoke the unclaimed approval, preserve the seven required output paths, return to `PI_REVIEW/NOT_AUTHORIZED`.
5. Complete the missing machine-readable design fields without changing the scientific question, variable, metrics, resource budget or stop conditions.
6. Re-authorize only after the repaired control history is green.

Scientific scope unchanged:
`EXP-SMALLTARGET-PHENOTYPE-001` remains zero-rollout / 0 GPU / 0 episode, using only existing historical full-200 evidence and static metadata. No Probe, replay, video-mechanism, reset treatment, Safe-vs-IL comparison or B0 modification is added.


## 2026-09-18 — DEC-CONTROL-REPAIR-003：repair CI通过并恢复small-target执行授权

Repair validation passed before re-authorization:
- synthetic control tests: PASS;
- full append-only control-history validation: PASS;
- no force-push/rewrite/squash;
- no Executor claim, GPU/model/simulator/episode execution occurred during the incident.

PI then re-authorized the unchanged `EXP-SMALLTARGET-PHENOTYPE-001` at commit `cd411d523bcfa5f335240b266c43bdfe14ada6d6`.
Budget remains 0 GPU / 0 episode. Executor must claim the latest green `origin/research-loop` HEAD and bind `instruction_commit` to that pre-claim HEAD. Earlier failed approval commits are historical incident records only and must not be claimed.


## 2026-09-20 — Historical server evidence aligned

User-authorized evidence archival only; no experiment claim or execution and no change to LOOP_STATE/NEXT_EXPERIMENT.
Packet: [history/evidence-alignment-20260919/README.md](history/evidence-alignment-20260919/README.md).
The inventory records 684 paths (680 existing/readable, 4 missing names); 645 existing sources lack byte-identical copies on the four inspected GitHub branches.
Final W&B raw table is 200 rows / 173 successes / sum_cost 145 and matches its recorded SHA256. Stable task-path normalization pairs 200/200 tasks, with 200/200 expert_length == gt_episode_len and room-visitation values present.
Static scene/asset sources exist, but annotated size is not yet validated as transformed per-target simulator bounding-box size. Four recovered Probe worker tensor hashes differ; the original AUC artifact/analysis identity remains unresolved.
Raw tensors, media, large logs and datasets remain server-side with SHA256/size. Existing experimental source copies are referenced rather than duplicated into the control plane. Old development status snapshots inside the archive are historical, not current authority.


## 2026-09-20 — Control validator UTF-8 path repair

Evidence archive commit `06a8265051298b103a76ff26db490d262da3882c` preserved the approved 158-file packet exactly. Its history CI exposed a pre-existing parser defect: `git ls-tree` quotes non-ASCII filenames by default, but the validator treated that display-escaped string as a literal path and rejected a research Markdown artifact as outside the whitelist. Working-tree validation uses NUL-delimited paths and did not expose this defect before publication.

Use `git ls-tree -z` and parse NUL-delimited records for committed snapshots. Regression tests cover UTF-8 research filenames with either core.quotePath setting, and ensure non-ASCII names cannot bypass the whitelist or disallowed artifact extensions. Preserve the original evidence commit and its packet bytes; append this validator repair without rewriting history. Experiment state/design, runtime and execution authorization remain unchanged.


## 2026-09-20 — DEC-EVIDENCE-ALIGNMENT-REVIEW-001：接受历史证据包并保持small-target实验不变

PI通过GitHub独立核验：
- `06a8265051298b103a76ff26db490d262da3882c` 归档历史证据；
- `fed5d8fe282bff0d3c119033ccb565f1574753d9` 修复UTF-8路径校验；
- Actions run `35463659449` 全绿，synthetic control tests和完整history/artifact validation均通过。

接受为共享证据层的事实：
- 2026-08-03 historical full-200最终W&B表=200行/173 success/sum_cost=145；
- stable task-path 200/200唯一配对；
- expert_length == gt_episode_len 200/200；
- room-visitation字段200/200可用；
- Probe、end audit、sub120及大文件均已有Git副本或server_path+SHA256登记。

仍未解决：
- annotations/scene metadata中的size字段尚未证明等同于当前任务中目标实例经缩放/变换后的真实3D bounding box；
- 四个Probe worker tensor哈希不同，历史AUC三元组的精确原始artifact/analysis identity仍未恢复。

Decision:
不改 `EXP-SMALLTARGET-PHENOTYPE-001` 的研究问题、唯一变量、指标或预算。正式执行时先验证policy-independent physical-size变量能否由静态metadata合法构造；若不能，按预注册stop condition返回BLOCKED，不以distance、visible pixels或category名称替代。旧授权已过期且从未claim，先撤销再对同一未执行cycle续发新授权。


## 2026-09-20 — EXP-SMALLTARGET-PHENOTYPE-001 metadata gate BLOCKED

Claim 72634cf7c3a954dd99456b2d75c3af539da9e68e bound renewed PI approval 290b426ac65d35ef1f8cb4819e9662c4fae1bb31; claim CI 35499476480 succeeded before execution. Budget used: 0 GPU / 0 episode / 0 model load / 0 simulator launch.

The metadata-first audit mapped 200 tasks and all 368 broad-synset target IDs to scene objects. Candidate asset bounding boxes cover only 42 targets; 326 lack candidates in inspected static sources. Only 31 tasks are candidate-complete, and no candidate has been promoted to validated scene-instance physical size: units/scale/transformation equivalence remains unestablished. Thus validated size count and analyzable n are 0. The bounded search does not establish global nonexistence of another static source.

The pre-registered stop condition is met: return BLOCKED to PI. No distance, visible-pixel or category-name proxy, incomplete-target median, category statistical analysis, tertile, association or regression was used. episode_table.csv preserves 200 historical raw rows with empty size cells; category_sr.csv explicitly records NOT_EXECUTED with blank statistics. H-SIZE remains untested.

Handoff: [RESULT_SUMMARY.md](handoffs/smalltarget-phenotype-001-20260918/RESULT_SUMMARY.md), with metadata coverage, source hashes, raw identity checks and all required outputs. Proposed prerequisite only: a version-bound static geometry and instance-transform source covering all broad targets, including built-in THOR assets. No new experiment is approved; executor STOP after publication. Existing development changes and frozen NEXT_EXPERIMENT files are preserved.


## 2026-09-20 — DEC-SMALLTARGET-BLOCKED-REVIEW-001：接受BLOCKED并将唯一下一实验改为runtime geometry recovery

PI独立读取commit `9910cb2c5645a3549d6e8e474827ee12de1df8bd`、RESULT_SUMMARY、RUN_MANIFEST、REVIEW_NOTES、size_analysis和200行episode table，并确认最终CI `35500291745` 为success。

Accepted result:
- 200/200 historical tasks与368个broad target IDs均成功静态匹配；
- 仅42/368 targets存在candidate asset bbox；326缺失；
- 仅31/200 tasks candidate-complete；
- units / scale / scene transform / instance-bound equivalence未验证，因此validated task size=0；
- H-SIZE未被支持也未被否定，因为核心解释变量没有被合法测量；
- Executor遵守0 GPU / 0 episode / 0 model / 0 simulator，未用distance、visible pixels、category或不完整subset替代尺寸。

Additional PI exploratory audit of the archived raw 200-row table:
- mug SR=8/13=61.5%，basketball SR=6/9=66.7%；
- sub_house_id<20恰好包含11个mug、8个basketball、1个vase；该20条中的8个failure全部属于mug/basketball；
- 在sub_house_id>=20中仅剩mug 2条、basketball 1条，且均成功，因此现有200条对“category effect vs early task allocation”缺乏有效overlap，不能把两种解释当独立证据；
- failure的expert_length均值108.26，success为49.06，H-DIFFICULTY仍是强竞争解释。
这些统计仅用于PI选择下一实验，不作为未经预注册的论文结论。

Decision:
不进入Probe、readout、安全、reset或新B0性能实验。唯一下一实验为 `EXP-SIZE-RUNTIME-METADATA-001`，先解决测量工具：使用**exact historical simulator runtime**在scene初始化后、任何policy action之前读取target runtime AABB。该实验不读取success labels，不做size-success association。

Rationale:
AI2-THOR公开接口说明object metadata可包含axisAlignedBoundingBox及size/cornerPoints，但world-axis AABB会随对象pose/orientation变化；因此本轮只验证“初始场景中的scene-instance extent”，不把它宣传成canonical intrinsic size。 exact historical build是否具有并稳定提供该字段，必须由实验本身验证。

Authorization:
当前仅DRAFT / PI_REVIEW / NOT_AUTHORIZED。预计预算最多1 GPU用于simulator graphics，0 SafeVLA/ObjectNav episodes，0 model loads，最多224次scene initialization。等待用户/PI显式批准后才可进入APPROVED_FOR_CODEX。


## 2026-09-20 — PI批准 EXP-SIZE-RUNTIME-METADATA-001 执行

用户明确回复“批准执行 EXP-SIZE-RUNTIME-METADATA-001”。PI批准此前冻结的measurement-recovery设计，不修改研究问题、H-RUNTIME-AABB、H-RUNTIME-GAP、唯一变量、preflight、指标或stop conditions。

Authorization:
- scope=RESEARCH_EXPERIMENT；
- max_gpu=1，仅允许simulator graphics/runtime；
- max_episodes=0；
- SafeVLA model/checkpoint loads=0；
- Actor/Critic forwards=0；
- 最多224次scene initialization（12-task preflight双次加载=24，preflight通过后最多200次full extraction）；
- geometry extraction不得读取success/failure、episode length、Safety Cost、visible pixels等outcome字段；
- 不得进行size-success association、Probe、replay、reset、Safe-vs-IL或新B0评估。

Executor必须先等待本approval commit control CI成功，再claim最新绿色HEAD并等待claim CI成功后执行。任何preflight stop condition触发即BLOCKED并STOP。

## 2026-09-21 — Runtime metadata handoff: BLOCKED / next actor PI

EXP-SIZE-RUNTIME-METADATA-001 completed 24 preflight and 20 full scene initializations before a BrokenPipeError in its stdout progress print. No retry or resume was performed. Preflight: 12 tasks loaded twice, 18 exact-equal target comparisons, max absolute/relative dimension difference 0. Partial full pass: 26/368 exact-mapped valid AABBs, 20/200 complete tasks; 342 targets / 180 tasks unattempted, not observed missing geometry. The complete-coverage hypothesis remains unresolved.

One simulator graphics GPU, 44/224 scene initializations, 0 episodes, 0 model/checkpoint loads, 0 Actor/Critic forwards. Geometry extraction read no outcomes and made no size-success association. Creation-state world-axis AABB is not intrinsic volume or demonstrated settled evaluation-state geometry. Development HEAD/diff/status and frozen NEXT_EXPERIMENT files are preserved.

Handoff: [RESULT_SUMMARY.md](handoffs/size-runtime-metadata-001-20260920/RESULT_SUMMARY.md), [coverage_report.md](handoffs/size-runtime-metadata-001-20260920/coverage_report.md), [ARTIFACT_INDEX.json](handoffs/size-runtime-metadata-001-20260920/ARTIFACT_INDEX.json). Original traceback, script, raw snapshots, partial CSVs and independent validation are Git-readable. Required outputs are complete; scientific extraction is incomplete.

The only next action is PI review. A fresh transport-resilient extraction cycle is a proposal only, requiring an explicit design and budget that account for this partial run. The current claim is terminal and must not auto-resume. No further experiment or PI acknowledgement is authorized or fabricated. Executor STOP after handoff publication.


## 2026-09-22 — DEC-RUNTIME-GEOMETRY-001-REVIEW：接受输出管道中断的BLOCKED结果

PI独立审阅 `ac2c9fe80f113345f5092205b07192a47a0e0358` 的RESULT_SUMMARY、RUN_MANIFEST、coverage_report、runtime identity、repeatability、partial geometry CSV与原始sample。

Evidence accepted:
- exact historical build commit=966bd7758586e05d18f6181f459c0e90ba318bec，CloudRendering；
- 12-task preflight双次加载共18个target comparison，18/18 exact-equal，最大绝对/相对差异均0；
- full阶段实际完成20/200 tasks、26 targets；26/26 exact ID mapping + valid creation-state AABB，observed mapping/geometry failure=0；
- 剩余342 targets / 180 tasks是UNATTEMPTED，不得标记为missing；
- BLOCKED由full-loop progress stdout的BrokenPipeError触发，当前证据不支持把它解释为simulator/geometry失败；
- 44/224 scene initializations，1 simulator GPU，0 episode，0 model load，0 Actor/Critic forward；未读取outcome字段。

Decision:
H-RUNTIME-AABB保持UNRESOLVED，H-RUNTIME-GAP也未被支持。旧claim关闭，不resume。下一轮若继续，只允许fresh cycle，以完全相同measurement semantics重跑完整200 tasks，同时把唯一工程变化限制为transport-resilient output/checkpoint机制，并用上一轮已完成的20 tasks作为cross-cycle overlap control。仍禁止任何size-success association。


## 2026-09-22 — DEC-RUNTIME-GEOMETRY-002-DRAFT：唯一下一实验为transport-resilient full extraction

Reasoning:
- 001的12-task repeatability与first-20 full pass均未出现mapping/AABB失败；
- BLOCKED发生在task20已保存后的一次stdout progress print，属于已观察到的transport failure；
- 剩余180 tasks完全未尝试，因此不能从001推断complete coverage；
- 直接进入size-success分析仍然越过measurement gate。

Decision:
新建fresh cycle `size-runtime-metadata-002-20260922` / `EXP-SIZE-RUNTIME-METADATA-002`。全新重跑200 tasks一次，最终dataset全部来自新cycle；前20 tasks与上一轮冻结结果做cross-cycle exact overlap control。唯一工程变化是移除inherited stdout/stderr依赖并增加atomic per-task checkpoint/progress manifest。scientific geometry semantics保持不变。

Budget:
max 1 simulator GPU；0 ObjectNav/SafeVLA episodes；0 model load；0 Actor/Critic forward；最多200 scene initializations。当前仅DRAFT / NOT_AUTHORIZED，等待用户显式批准。


## 2026-09-22 — DEC-MAINLINE-MECHANISM-001：停止继续验证physical size，先定位failure stage

User decision:
研究者明确认为目标物体相对更小属于足够明确的常识性观察，不希望继续把主线时间投入physical-size measurement；下一步优先回答“低SR到底在哪个机制阶段发生”，并尽快澄清“更难任务”的含义。

PI boundary:
接受停止size measurement，但不把“size causally causes failure”升级为论文结论。正式统计对象改称low-SR target group；small-object-like仅作为研究者定性描述。

Correction:
此前基于`sub_house_id<20`的“更难任务”解释错误。历史审计已确认`sub_house_id`是shuffle后的sample index。真实house为`house_index`。当前可用的pre-policy difficulty proxy是`expert_length=task_info["expert_length"]`，但它是expert trajectory length，不是纯环境难度或已证明的shortest path。

Exploratory evidence:
- low-SR group(mug/basketball/laptop/bowl) SR=34/48=70.8%，other=139/152=91.4%；
- 14个low-SR failures中9个never target-room，10个never nav-visible，4个nav-visible；
- expert_length分层后gap仍存在：<=50 92.3% vs97.8%，51–100 57.1% vs90.7%，>100 25.0% vs58.8%。
因此当前leading stage hypothesis是pre-visibility exploration/room-arrival，但尚未形成正式结果。

Decision:
supersede未批准的EXP-SIZE-RUNTIME-METADATA-002。唯一下一DRAFT为`EXP-LOW-SR-STAGE-LOCALIZATION-001`，0GPU/0episode，仅用现有full-200证据正式复现difficulty-stratified SR与S0/S1/S2 stage taxonomy。结果决定下一内部诊断走exploration branch还是post-visibility representation/readout branch。


## 2026-09-22 — DEC-02-REDESIGN-001：按02｜SafeVLA主线重设计下一实验

User request:
“根据02|SafeVLA内容重新设计实验。”

PI review:
上一DRAFT `EXP-LOW-SR-STAGE-LOCALIZATION-001` 只做S0/S1/S2统计，虽然能整理表型，但没有直接回答02长期主问题：信息是没进入、表示丢失、还是表示存在但Actor未使用，以及premature end和termination hesitation是否来自同一语义决策失配。

Historical constraints from 02:
- old Probe AUC约0.955/0.982/0.992，不支持简单Layer3信息消失；
- old Probe存在PT-Guard、无episode ID、step split泄漏风险，必须clean重做；
- Done Gate没有提高净SR，而是把部分premature done转成hesitation/loop；
- 已有一个illegal end_prob≈0.993 case与一个600-step low-p(done) case，说明termination存在相反表型；
- cost critic源码不直接gate Actor end。

Decision:
supersede未批准的stage-only audit。唯一下一DRAFT改为 `EXP-SEMANTIC-DECISION-MISMATCH-001`：6个历史失败任务（3个pre-visibility、3个post-visibility）+ 每个一个same-category expert-length-nearest success control + 一个cross-category expert-length-nearest success control，形成18个诊断task；两固定seed共最多36 episode。B0行为不改，只读hook actor causal path。

Fresh analysis:
- 先按fresh run重新判P0/P1，不用historical label冒充复现；
- clean episode/house-split Probe测fusion/L1/L2/L3 target-visible information；
- 若L3信息强，再做唯一causal manipulation：offline L3沿probe方向±alpha，与等范数random/shuffled方向比较Actor end-logit sensitivity；
- P0轨迹分析end_prob、action entropy、room/search、loop/oscillation，定位exploration vs premature termination；
- 结果直接决定下一干预属于exploration、representation、readout/calibration还是history/objective。

Current status remains DRAFT / PI_REVIEW / NOT_AUTHORIZED.


## 2026-09-23 — DEC-02-REREAD-GATEB-001：按02原始Gate顺序重排实验

User asked to first read `02｜SafeVLA研究` and then design the experiment.

PI reread found the decisive omitted constraint:
- Gate A PASS；
- Gate C PASS（限定审计范围）；
- Gate B FAIL：official reset不清Actor/Reward/Cost counter与KV cache；
- short decoder control只证明新episode开头旧cache被mask，不能排除累计counter在500 rollover后的history/logit/action影响；
- 02明确要求在Gate B未关闭前停止正式Probe/hidden-state机制解释。

Decision:
supersede未批准的 `EXP-SEMANTIC-DECISION-MISMATCH-001`。唯一下一DRAFT改为 `EXP-RESET-ROLLOVER-CAUSAL-001`。

Design:
1. 两条clean-B0只读capture trace（seed123/456），每条累计>=540 decisions，最多16 episodes；
2. 保存真实Actor-decoder输入与episode masks，不额外forward、不改变RNG/action；
3. offline对完全相同输入做OFFICIAL-CARRY vs CLEAN-RESET counterfactual replay；
4. 先要求official replay复现实况，再检查rollover前后hidden/logit/end/action差异；
5. live环境中不执行reset treatment。

Decision consequence:
- 若rollover后无差异，Gate B2可在限定证据下通过，下一轮恢复semantic-decision mismatch / clean Probe + Actor-use实验；
- 若rollover后出现可复现差异，H-RESET成为已证实的机制通路，下一轮先做行为影响量化；
- 不论哪种结果，本轮都不修改B0、不训练Probe、不做Stop Gate。


## 2026-09-23 — DEC-PREMATURE-END-001：优先定位提前终止动态，暂缓reset rollover

Current GitHub state before this decision:
- `EXP-RESET-ROLLOVER-CAUSAL-001` was DRAFT / NOT_AUTHORIZED;
- LOOP_STATE was PI_REVIEW / next_actor=PI / claim_id=null;
- no reset-rollover capture episode had been executed.

New evidence/review:
- aligned historical full-200 has 27 failures, including 16 sub-horizon failures;
- legacy canonical evidence contains one illegal 8-step termination state with stop_legal=false and Actor p(end)≈0.993365;
- the recovered sub120 horizon case shows the opposite morphology, rendered p(done)<~0.018 through 600 steps;
- official ObjectNav RL reward configuration uses step_penalty=0, failed_stop_reward=0, reached_horizon_reward=0, goal_success_reward=10. This makes an early-exit objective loophole plausible but does not prove that safety alignment causes it;
- the paper's cautious/extreme-failure behavior is hypothesis support only, not proof for the historical B0 failures;
- there is still no evidence that the authors deliberately trained exact test scenes/tasks to early-exit.

Methodological decision:
Gate B reset/counter rollover remains a real unresolved implementation confound for future hidden-state/Probe/readout claims. However it is not a prerequisite for an artifact-only audit of already-recorded executed actions and rendered Actor probabilities. Because the current project objective is rapid syndrome localization, spending up to 32 fresh episodes on reset before establishing the systematic premature-end morphology has lower information value.

Decision:
- supersede `EXP-RESET-ROLLOVER-CAUSAL-001` before approval and defer it as a future hidden-state diagnostic gate;
- stage `EXP-PREMATURE-END-DYNAMICS-001` as the unique next DRAFT;
- use all 16 historical eps_len<600 failures, first verify final executed action=end from video, then recover p(end) dynamics and one expert_length-nearest historical success control per case;
- do not preselect low-SR categories and do not infer SafeRL causation from this audit.

Critical correction retained:
`sub_house_id` is the original dataset sample index assigned before evaluation-order shuffle; it is not house identity or difficulty.

Current status remains PI_REVIEW / NOT_AUTHORIZED. No Executor claim or execution is authorized by this decision.


## 2026-09-23 — CONTROL-PROTOCOL-REPAIR-004：premature-end staging history mismatch

Failure evidence:
- approval HEAD `3af48dc297468cca0b564ef234c2c1c6bd4a435c` failed GitHub Actions run `35841714309`;
- synthetic protocol tests passed;
- full history validation failed with `NEXT_EXPERIMENT mismatch`;
- no Executor claim occurred and no research execution started.

Root cause:
PI changed the new experiment design in multiple append-only commits before moving `LOOP_STATE` from the old reset-rollover cycle to the new premature-end cycle. Three intermediate commits therefore intentionally contained a staged NEXT_EXPERIMENT identity different from the still-old PI_REVIEW LOOP_STATE:
- `149ab525a99aa02afd0f4aa61472d3fd1ec2680e`
- `0151b678a822f780190130690f1eb75955e2eca6`
- `7fba4aedfaa9978581ba3bee643c5ffe953ddb35`

This was a control-plane publication-order defect only. All three snapshots remained PI_REVIEW / NOT_AUTHORIZED with instruction_commit=null and claim_id=null. No GPU, simulator, model, episode, or artifact analysis was executed.

Repair:
- preserve the three commits; do not rewrite/squash/force-push history;
- add a narrow validator exception tied only to the exact three commit SHAs and exact old/new experiment identities/statuses;
- keep normal snapshot validation strict, so the same mismatch is still rejected anywhere else;
- add a synthetic regression test proving the ordinary validator rejects the mismatch and only the exact legacy exception accepts it;
- do not modify the approved scientific design, budgets, or LOOP_STATE while repairing validation.

Executor remains forbidden to claim until a newer research-loop HEAD has green control CI. After green CI, claim must bind to that latest pre-claim HEAD, not the failed approval commit.

## 2026-09-23 — EXP-PREMATURE-END-DYNAMICS-001 completed / awaiting PI review

All 16 historical sub-horizon failure videos were recovered reliably and confirm final executed end. Frozen morphology counts: LATE_RISE 13/16, EARLY_HIGH_PRIOR 1/16, OTHER_OR_UNCLEAR 2/16, LOW_PROB_END 0/16. All initial end bars were sub-resolution; all preterminal maxima were below 0.5. The 14 first crossings of 0.5 occurred at the terminal decision itself. The one early-high case is a two-step episode, not evidence of an elevated first-decision prior.

All 16 same-category expert-length matches were recovered (11 unique successes, reused). Ten of those 11 successes also first crossed 0.5 at successful termination. Thirteen prescribed control windows include the success terminal frame; expert-length matching gaps have median 34 and range 2–133. The abrupt terminal jump is therefore not failure-specific and does not establish SafeRL causation. Target scope: STRICT 12 / AMBIGUOUS 4; narrow visibility cannot establish broad-target invisibility in ambiguous cases.

This was historical CPU-only video/table analysis: 0 GPU, 0 episodes, 0 simulator, 0 model/checkpoint loads, 0 Actor/Critic forwards. All labels agree under the secondary pixel method and declared quantization sensitivity checks. Zero-width and saturated bars are approximate, not exact 0/exact 1. Independent validation passed for sources, metrics, frame alignment, matches and unchanged development identity.

Handoff: [RESULT_SUMMARY.md](handoffs/premature-end-dynamics-001-20260923/RESULT_SUMMARY.md), [analysis](handoffs/premature-end-dynamics-001-20260923/premature_end_analysis.md), [artifact index](handoffs/premature-end-dynamics-001-20260923/ARTIFACT_INDEX.json). Claim a5045c5255b67f3e0f1d4f559d2d6078f70f9544 and its green CI preceded execution. Frozen designs and claim/authorization identity are preserved.

Next actor PI. One proposal follows the frozen LATE_RISE branch: review a matched hard-task SafeVLA versus comparable non-safety/base-policy causal comparison, with comparability and confounds explicitly controlled. This proposal is not approved or executed. Gate B remains unresolved for future hidden-state interpretation. No next experiment or PI acknowledgement is self-issued. Executor STOP after publication.


## 2026-09-23 — CONTROL-PROTOCOL-REPAIR-005：PI acknowledgement sparse-tree incident

During PI acknowledgement of result commit `f79b3cfd19e8f537715d2c321e13885e7303ba1f`, a low-level Git tree construction mistake created commit `0d91c85df9cd5be8ac9de9d1ab3b6f10cc4a9d5d` with only three tracked files: `research/LOOP_STATE.json`, `research/NEXT_EXPERIMENT.md`, and `research/NEXT_EXPERIMENT.json`. This accidentally omitted the rest of the control tree.

No execution authorization was introduced: the sparse snapshot is PI_REVIEW / NOT_AUTHORIZED and preserves the completed experiment's historical claim only. No Executor claim, GPU, simulator, episode, model load or analysis followed this commit.

Repair:
- append-only restoration commit `7a63f05e1925ef7322f217132939dd943aee91f8` reconstructs the exact full parent tree from result commit `f79b3cfd...` and overlays only the intended acknowledgement files;
- no force push, reset, squash or history rewrite;
- normal validation remains strict;
- a narrow history exception applies only to exact sparse commit `0d91c85d...`, verifies its exact parent/state/design identity and exact three-file tree shape;
- regression coverage added before opening any new research cycle.

Scientific result and acknowledgement decision are unchanged. No next experiment is authorized by this repair.


## 2026-09-23 — DEC-PREMATURE-END-RESULT-REVIEW-001：接受历史 invalid-end，否定对 LATE_RISE 的过强机制解释

Independent PI review of result commit `f79b3cfd19e8f537715d2c321e13885e7303ba1f`:
- 16/16 historical sub-horizon failures are reliably recovered and final action=end;
- all first-decision p(end) bars are sub-resolution;
- all preterminal maxima <0.5; every >=0.5 crossing occurs at terminal end;
- frozen labels: LATE_RISE13 / EARLY_HIGH_PRIOR1 / OTHER2 / LOW_PROB_END0;
- the sole EARLY_HIGH_PRIOR is a two-decision episode with terminal end inside the first-five window;
- 10/11 unique matched successes also first cross0.5 at successful termination.

PI interpretation:
The labels are mechanically correct, but they do not support “search failure gradually raises p(end).” The robust observation is an abrupt terminal switch. Because successful termination shows the same switch, the unresolved question is why an end is legal in success and illegal in failure.

Provenance audit:
The 2026-08-03 run used commit `60bc54fbdedaf5745d0476c25321e808708273aa`. Its evaluator calls `successful_if_done(strict_success=False)` before `agent.get_action`; the visibility path can issue `GetVisibleObjects(maxDistance=2)` on a cache miss. Observations were captured before this query and a behavioral effect is not established, but the extra simulator query violates the current clean-B0 contract.

Decision:
Do not proceed directly to SafeVLA-vs-FLaRe. The unique next DRAFT is `EXP-CLEAN-B0-PREMATURE-END-REPRO-001`: one full-200 evaluation under accepted executable B0, followed by offline termination classification. Historical 173/200 remains descriptive reference only.

No execution is authorized by this decision.


## 2026-09-23 — DEC-SKIP-REPRO-001：不再为 historical invalid-end 做重复复现

Researcher decision:
Full-200 clean-B0 rerun and targeted clean-B0 reproduction are both removed from the current mainline. The purpose of the project is mechanism diagnosis, and the 16 historical invalid-end cases already provide a sufficient discovery set for the next diagnostic question.

Boundary retained:
The historical 2026-08-03 run is not promoted to formal clean-B0 prevalence evidence. Its evaluator provenance limitation remains documented. Skipping reproduction trades formal baseline generality for research speed; any final paper claim about official-B0 prevalence will require separate evidence later if that claim becomes necessary.

Next question:
The previous audit showed terminal p(end) rises abruptly in both failures and successes. Therefore the next useful distinction is legal versus illegal terminal decision, not reproduction of the same failures.

Unique next DRAFT:
`EXP-END-LEGALITY-CALIBRATION-001`, zero-rollout / 0 GPU / 0 episode. Compare all recoverable successful terminal end events with the 16 confirmed illegal terminal ends using video-rendered quantized action probabilities.

No execution is authorized.
