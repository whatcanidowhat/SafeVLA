---
name: safevla-research-loop
description: SafeVLA ObjectNav 研究与受控交接入口。按 STATUS/GATE、开发、科学分析、执行或审核加载证据，遵守 PI 授权、Claim CI、Baseline 合约和停止条件。
---

# SafeVLA Research Loop

目标：提高 SR，同时保持或降低官方 Safety Cost。路径相对仓库根目录。

## Router：按当前任务加载

先读 AGENTS.md、research/LOOP_STATE.json；使用同一已核实 revision 的文件。初次进入协议或执行状态变更前完整读 research/HANDOFF_PROTOCOL.md；熟悉且协议内容未变时，只读任务适用章节。协议明确要求的 NEXT_EXPERIMENT.md / .json 和 current evidence 仍须读取；不把精简路由当作跳过协议的理由。

| MODE | 读取与边界 |
| --- | --- |
| STATUS / GATE（status、ci、claim 检查） | 当前 remote HEAD、LOOP_STATE、NEXT_EXPERIMENT.json 授权/身份/预算/截止期字段及适用协议章节；核验 CI 时读 workflow 和必要控制脚本。只读查询不调用会创建 Claim 的命令，不加载完整长期研究史。实际领取前转 EXECUTOR。 |
| DEV（控制文档/CPU 测试） | 任务相关指令、协议和被改文件；运行局部 CPU 控制测试。若涉及科学代码，另读下述科学参考和 Baseline 合约。开发任务不授予实验权限。 |
| RESEARCH / REVIEW | 先确定实验与问题，再读 [科学流程](../../../research/agent_workflow/references/scientific-workflow.md) 和其指定的原始证据、源码、相关研究记录。独立 PI 审计需要时完整阅读，不以摘要替代证据。 |
| EXECUTOR | 完整协议、完整批准设计、科学流程、相关 Baseline/运行证据和控制脚本。只有当前角色获准且全部门禁通过才进入现有一次性执行流程；此 Skill 不新增启动器。 |
| HANDOFF | 协议 Result C、LOOP_STATE.required_outputs、当前 run 的真实 manifest、artifact index、哈希与退出证据。若解释科学结果或决定下一实验，转 RESEARCH / REVIEW；纯交接不重读无关历史。 |

STATUS/GATE 可按字段检索设计；任何身份/范围矛盾、字段缺失或协议要求均须扩大读取并 STOP 受影响动作。远端不可读时明确 REMOTE_UNVERIFIED；本地缓存永不授予执行许可。状态/CI/network CLI 只有已实现且验证后才能调用；不得假定规划中的工具存在。

STATUS/GATE 的只读查询可使用 scripts/research_loop_status.py 和 scripts/research_loop_ci_wait.py；首次使用读 [参数与结果契约](../../../research/agent_workflow/READONLY_TOOLS.md)。CI 等待交给一次有界脚本调用；其 SUCCESS/exit 0 不是执行许可，不代替 Claim、完整历史、身份或 runtime 门禁。

## Policy：授权与停止条件

- GitHub research-loop 的实时 HEAD + LOOP_STATE + 对应 exact-SHA CI 共同判定当前控制状态；历史 001A/001B、bootstrap、protocol-only、dry-run 或草案描述都不是现时授权。终止周期不可自动重跑。
- CODEX 只在 APPROVED_FOR_CODEX + next_actor=CODEX 及有效 PI 授权下领取；先验证批准 A 的 CI，再用 scripts/research_loop_claim.py 原子发布唯一 Claim B，确认远端 HEAD=B 和 B 的独立 CI 成功。不能用另一 SHA、PR CI 或本地 Claim 代替。完整历史校验由 A/B 的 exact-SHA 控制 CI 执行；本地 Claim 校验当前快照与本次状态转换。
- 人工 OPERATOR 只能使用协议中明确的 APPROVED_FOR_OPERATOR 路线和 scripts/research_loop_manual_claim.py；不能借用 CODEX 批准或冒充 PI。角色字符串不是身份认证。已领取周期不得转换角色或恢复运行。
- 执行前仍须核实 deadline、源码/TaskSpec/checkpoint、frozen OFF 只读哈希、实际环境/TLS/代理 witness、资源和预算；CI 成功不替代运行门禁。显式选择 SSH repo、Python、依赖/import paths；另一 CMD 会话的环境不会自动继承。控制 worktree 不得作为 SafeVLA runtime。
- Claim/push/CI/TLS/网络失败或结果不确定即 STOP；按协议保留本地 Claim 和证据，不自动重试、补领、force push、reset 或恢复中断实验。不扩大样本、参数或预算。
- Executor 不自我批准、不改 READY、不代写 PI 回执。完成或阻断均保存 required handoff outputs，交回 PI 并 STOP；未同步证据不能称为 PI 已可访问或已审核。
- 保留 B0、未提交工作、旧结果与归档；不静默改冻结科学协议，不伪造 PASS。消息发送、commit/push 遵循现有明确授权，仅选择本轮文件。
- Skill 不创建后台进程、定时任务或无限循环。已有授权内的步骤不反复询问；阶段/角色切换、阻断解除及新实验遵守协议。

## Stop Conditions

以下情况停止自主实验执行并报告，可继续不依赖该问题的只读排查：
- Baseline意外变化，runtime不明，manifest与实际矛盾。
- 必要数据/产物缺失，结果达不到所需复现标准，代码错误令结果不可解释。
- 多个核心变量变化、泄漏、episode/control不一致。
- Gate B等未解除，却需要解释其影响范围内的结果。
- 达到授权轮数/时间/计算预算，或下一动作超出现有执行范围。

报告已完成工作、证据缺口、阻断影响和一个具体下一动作。隔离验证阻断本身时先形成独立设计并取得所需执行授权，不能改标签后自行继续。


## Output contract：控制任务默认四项

VERDICT: 当前判定；本地状态标 REMOTE_UNVERIFIED，非执行状态 CAN_EXECUTE=false
EVIDENCE: exact SHA、Claim ID、CI workflow/run ID/attempt、证据路径（仅已核实项）
BLOCKER: 阻断或缺失证据；无则 none
NEXT_AUTHORIZED_ACTION: 当前已获准的一个动作，或等待 PI

同一 SHA、Claim ID、CI run/attempt/status/conclusion、状态及授权有效性/网络证据均未变化时，不重复读取、解释或主动汇报；用户询问仍简短回应。截止期到达、网络异常或新证据也触发更新，即使 SHA 未变。缓存仅减少重复解释，不替代每次操作必须的实时门禁。

已验证的只读脚本可有界轮询并只输出最终结构化结果或异常；没有该工具时报告缺口，不让 LLM 无界轮询。科学报告仍完整给出因果链、竞争解释、不确定性及原始证据。
