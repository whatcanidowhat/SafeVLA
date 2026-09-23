# SafeVLA Research Agent Instructions

目标：诊断官方 SafeVLA 在 ObjectNav 上的失败机制，寻找提高 SR、同时保持或降低官方 Safety Cost 的干预。

## 项目地图

研究决策和代码改动前，依次读取：
1. research/BASELINE_CONTRACT.md
2. research/CURRENT_STATE.md
3. research/NEXT_EXPERIMENT.md
4. research/EXPERIMENT_REGISTER.md
5. research/DECISION_LOG.md

研究闭环入口：.codex/skills/safevla-research-loop/SKILL.md。即使客户端未自动发现 skill，执行研究循环前也须读取。可通过 .agents/skills/safevla-research-loop 发现同一 skill，不维护第二份正文。

评测入口：scripts/eval.sh → training/online/online_eval.py → online_evaluation/；策略源码在 architecture/，任务和环境规则在 tasks/、environment/。已有审计位于 diagnostics/end_causal_audit/。

## 核心约束

- Baseline/诊断阶段不得静默改变 success definition、end、horizon、checkpoint、官方指标或核心评测协议。疑似 bug 先记录，行为改动作为隔离 treatment。
- 先提出研究问题、竞争解释、唯一变量、对照和证伪条件，再改代码。不要默认用户或上一轮的假设正确。
- 自由探索感知、语言条件、V-L fusion、时间表征、记忆、探索、actor readout、终止、训练目标和安全约束，不强制 Probe 或 end 方向。
- 区分事实、观察、假设、结论；指标朝预期变化不足以证明因果。
- 每个 run 记录实际 working tree、runtime provenance 和 artifacts；HEAD 不涵盖未提交代码，须保存相关 diff 和参与执行的 untracked 源码身份。
- 每轮只有一个正式待执行实验；尊重 NEXT_EXPERIMENT 状态和停止条件。构建协议不代表启动实验或授权无限循环。
- 保留现有修改与原始结果。commit/push 仅选择本轮文件并遵循已有授权；创建 skill 不自动授权向其他聊天发送消息。

初始化模式为 protocol-only；现有 Gate B 阻断见状态文件，解除必须有新证据和决策记录。

## GitHub PI–Codex Handoff Protocol

先读research/LOOP_STATE.json和research/HANDOFF_PROTOCOL.md。只有APPROVED_FOR_CODEX + next_actor=CODEX才允许领取；使用scripts/research_loop_claim.py绑定instruction_commit并成功push唯一claim后，待CI通过才可执行一个approved experiment。完成或阻断必须产生required handoff outputs，交回PI并STOP。Executor不得自行批准下一实验、改READY或代写PI回执。GitHub共享状态不能替代服务器runtime provenance。控制worktree不得作为SafeVLA runtime。001A已完成不重跑；001B未授权；当前handoff草案未授权。本控制分支的这个明确授权门优先于正文旧的通用READY步骤。
