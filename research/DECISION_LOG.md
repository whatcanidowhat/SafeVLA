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
