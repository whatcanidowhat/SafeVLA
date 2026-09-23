---
name: safevla-research-loop
description: 在 SafeVLA ObjectNav 项目中执行可审计研究闭环：核查证据、比较竞争假设、设计单变量实验、验证运行并更新状态。用于失败机制诊断及 SR/官方 Safety Cost 干预评估，遵循 Baseline 合约、当前实验状态及执行授权。
---

# SafeVLA Research Loop

目标：提高SR，同时保持或降低官方Safety Cost。约束证据和实验纪律，不限制科学创意，不规定必须Probe、研究end或采用某种机制。

文内路径均相对SafeVLA仓库根目录。通过SSH工作时显式选择repo和Python/环境；用户另一CMD会话激活的环境不会自动传入本会话。

## 执行模式与角色

初始化为protocol-only。协议审稿后执行已授权、人工监督的两轮小规模dry run，每轮闭合“设计→真实证据→审查→状态更新→唯一下一实验”。通过两轮不自动启动无限循环；连续模式必须在已有用户授权内明确轮数或时间/计算预算和停止条件。

已有负责人时，由负责人读代码/产物并决定唯一下一实验；执行者实现、运行、验证和提出建议，不自行成为第二负责人。版本化research状态和run artifacts是共享事实；GitHub版本不能替代服务器runtime provenance。

本项目负责人参考任务为“分支 · 02｜SafeVLA研究”（6a7b4d04-abf0-83ee-991b-e3e9a9821032）。仅在已有消息发送授权且工具可用时发送结果/请求指令；没有可靠桥接时生成可交接文件与摘要，报告缺口，不声称已经发送。收到设计后仍核验其证据、唯一变量、合约和授权范围。

Skill是工作流说明，不是后台调度器；加载skill不自动创建定时任务或常驻进程。已有授权覆盖的步骤不反复确认，缺少授权的执行或阻断解除须明确记录。

## 1. Read State

读取AGENTS.md及research/下BASELINE_CONTRACT.md、CURRENT_STATE.md、NEXT_EXPERIMENT.md、EXPERIMENT_REGISTER.md、DECISION_LOG.md，再读取相关最新原始产物、源码和git diff。

确认服务器路径、HEAD/dirty tree、实际Python/依赖/import paths、执行模式。分支名不能证明B0身份，历史数字不能写成新结果。输入缺失先查可用来源，明确缺口。

## 2. Audit Evidence

分开verified fact、observation、hypothesis、conclusion及证据范围。核对runtime、baseline协议、产物、标签、episode集合/顺序/worker分配、泄漏、不平衡、seed稳定性和混杂。摘要或图表不能代替原始结果。

检查当前停止条件，特别是Gate B。阻断期间可继续只读溯源、文档和隔离设计，不能扩大运行或解释受影响Probe。旧token被mask隔离不排除rollover风险，cache保留也不直接证明性能影响。

## 3. Generate Competing Explanations

不自动沿用前一假设。沿input→vision/language→V-L fusion→temporal state/history→actor→action→environment outcome追踪。按证据比较感知、语言、融合、记忆、探索、readout、termination、训练目标与安全约束，提出可区分机制的预测。

## 4. Select One Discriminative Experiment

在选择下一实验前，先判断当前阻塞项是否仍处于主研究目标的关键路径上；不要仅因为某个历史假设已经写进 CURRENT_STATE 就自动继续。
选择最能降低当前不确定性的最小实验。动代码前填写NEXT_EXPERIMENT的问题、证据、假设/竞争解释、control/treatment、唯一变量、固定条件、指标、预期/证伪结果、替代解释、允许/禁止改动、停止条件、产物和资源范围。

只读诊断不改变行为；离线状态干预和线上干预均明确归类。正式实验只改变一个核心变量。关键输入/字段不全则保持DRAFT/BLOCKED；设计完整且已有执行授权覆盖时才置READY。重试不能混入未注册变量。

## 5. Execute

只执行READY且在授权范围内的实验。分配唯一run ID，冻结设计，保存代码/配置/输入身份；只改设计允许的文件，不覆盖已有结果或未提交工作，不在B0上静默修bug。

Diagnosis的hook/logging不改变RNG、cache、forward次数、动作与环境。Actor/critic特征分开；counterfactual使用独立模型/cache或离线replay。Intervention保留B0，显式隔离treatment。触发停止条件即保留结果并停本run，不自动改参数、扩大样本或跨阶段。

## 6. Validate Run

按research/BASELINE_CONTRACT.md保存manifest、command、commit/status/diff、参与执行的untracked源码身份、checkpoint hash、seed、worker数、episode manifest、runtime import paths、results/logs、退出状态。

核对实际与声明、产物计数、episode对齐、变量隔离。进程成功不代表科学有效；不可解释时标INVALID/BLOCKED并记录原因。日志非侵入性只适用于验证范围。

## 7. Interpret

对照expected result、falsifying result、competing explanations。区分可解码性、相关性、机制干预与最终指标；方向符合预期不足以确认假设。

SR和官方Safety Cost共同报告样本量、分母、适当不确定性、cost分布/长尾。离线或小样本诊断不能宣称完整benchmark提升，无显著差异不自动等于等价。

## 8. Update State and Handoff

保存RESULT_SUMMARY.md及原始证据链接；更新CURRENT_STATE、EXPERIMENT_REGISTER、DECISION_LOG。归档已执行设计，再提出恰好一个下一实验，明确建议与已批准设计的区别。

交接experiment/run ID、revision/diff、manifest、results/logs路径与哈希、有效性判定、停止原因和待决问题。负责人审阅真实证据后更新设计。commit/push和消息发送按已有授权，只选择本轮文件；未同步的本地产物不称为负责人已可访问。

## Stop Conditions

以下情况停止自主实验执行并报告，可继续不依赖该问题的只读排查：
- Baseline意外变化，runtime不明，manifest与实际矛盾。
- 必要数据/产物缺失，结果达不到所需复现标准，代码错误令结果不可解释。
- 多个核心变量变化、泄漏、episode/control不一致。
- Gate B等未解除，却需要解释其影响范围内的结果。
- 达到授权轮数/时间/计算预算，或下一动作超出现有执行范围。

报告已完成工作、证据缺口、阻断影响和一个具体下一动作。隔离验证阻断本身时先形成独立设计并取得所需执行授权，不能改标签后自行继续。

## GitHub PI–Codex Handoff Protocol

先读research/LOOP_STATE.json和research/HANDOFF_PROTOCOL.md。只有APPROVED_FOR_CODEX + next_actor=CODEX才允许领取；使用scripts/research_loop_claim.py绑定instruction_commit并成功push唯一claim后，待CI通过才可执行一个approved experiment。完成或阻断必须产生required handoff outputs，交回PI并STOP。Executor不得自行批准下一实验、改READY或代写PI回执。GitHub共享状态不能替代服务器runtime provenance。控制worktree不得作为SafeVLA runtime。001A已完成不重跑；001B未授权；当前handoff草案未授权。本控制分支的这个明确授权门优先于正文旧的通用READY步骤。
