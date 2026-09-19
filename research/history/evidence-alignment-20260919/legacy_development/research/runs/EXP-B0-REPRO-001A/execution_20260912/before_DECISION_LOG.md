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
