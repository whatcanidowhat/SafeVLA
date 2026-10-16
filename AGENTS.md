# SafeVLA Research Agent Instructions

目标：诊断官方 SafeVLA 在 ObjectNav 上的失败机制，提高 SR，同时保持或降低官方 Safety Cost。

## 任务入口与读取范围

研究闭环入口为 .codex/skills/safevla-research-loop/SKILL.md；客户端未自动发现时也须读取，不维护第二份正文。
先明确任务模式和实验/范围，再按入口路由加载。控制事实以实时 research-loop remote HEAD、research/LOOP_STATE.json 和对应 exact-SHA CI 为准；本地缓存标 REMOTE_UNVERIFIED，不授予执行许可。

- status / claim / ci / handoff：读取 LOOP_STATE、NEXT_EXPERIMENT.json 的授权/identity/预算/期限及 HANDOFF_PROTOCOL.md 适用章节和必要脚本。初次进入协议或状态变更前完整读协议；协议要求的 Markdown 设计和当前证据不能省略。实际 Claim/执行前使用 Skill 的 EXECUTOR 路由；交接核对本 run 的全部 required_outputs。无需默认读取完整 CURRENT_STATE、DECISION_LOG、EXPERIMENT_REGISTER。
- 控制文档 / CPU 开发：读取相关文件和测试，不加载无关科学历史，不据此启动实验。
- research-analysis / PI-review / scientific-code：读 research/agent_workflow/references/scientific-workflow.md、BASELINE_CONTRACT 和当前设计，按实验定位 CURRENT_STATE、EXPERIMENT_REGISTER、DECISION_LOG 及原始证据/源码。独立审计、冲突或依赖需要时完整阅读，摘要不能替代原始证据。

## 所有模式的边界

- Baseline/诊断不得静默改变 success definition、end、horizon、checkpoint、官方指标或核心评测协议。疑似 bug 先记录，行为改动作为隔离 treatment。
- 科学改动前提出研究问题、竞争解释、唯一变量、对照和证伪条件；不默认上一轮假设正确，不强制 Probe/end 方向。区分事实、观察、假设、结论，指标方向不足以证明因果。
- 每个 run 保存实际 working tree、runtime provenance、diff、参与执行的 untracked 源码身份和 artifacts；GitHub HEAD 不能替代运行身份。控制 worktree 不运行 SafeVLA。
- 每轮只有一个正式待执行实验。PI 批准 A 的 CI、唯一 Claim B 的成功发布、B 的 exact CI 与 remote HEAD、当前身份/有效期/范围及所有运行门禁均不可跳过。CODEX/OPERATOR 路线不可互借授权，Executor 不批准、不改 READY、不代写 PI 回执。
- BLOCKED/INVALID/ABORTED、PI_REVIEW 或 NOT_AUTHORIZED 均不授予执行权；历史静态状态不是当前许可。网络/推送/Claim/CI 不确定即 STOP，保留证据，不自动重试、force push 或恢复实验。终止周期的恢复须由 PI 按协议处理。
- 完成或阻断均提供 required handoff outputs，交回 PI 并 STOP。保护现有修改和旧结果；commit/push 仅选择本轮文件并遵循已有授权；Skill 不授权消息发送或后台调度。
- Skill 的全部 Stop Conditions 在所有模式生效；只读排查可继续，但不能解释被阻断问题影响的科学结果或扩大执行。

控制任务按 Skill 四字段输出；相同证据不重复解释，截止期/网络/状态变化仍须响应。研究审核保留完整证据链。评测入口 scripts/eval.sh → training/online/online_eval.py → online_evaluation/，源码 architecture/、tasks/、environment/ 仅在相应 runtime/source tree 查阅。
