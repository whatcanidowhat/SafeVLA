# Scientific workflow (conditional reference)

适用：RESEARCH / REVIEW / scientific-code，以及获准执行前的科学有效性检查。文内路径相对仓库根目录；控制授权以 research/HANDOFF_PROTOCOL.md 为准。

已有负责人时，由负责人读代码/产物并决定唯一下一实验；执行者实现、运行、验证和提出建议，不自行成为第二负责人。版本化research状态和run artifacts是共享事实；GitHub版本不能替代服务器runtime provenance。

本项目负责人参考任务为“分支 · 02｜SafeVLA研究”（6a7b4d04-abf0-83ee-991b-e3e9a9821032）。仅在已有消息发送授权且工具可用时发送结果/请求指令；没有可靠桥接时生成可交接文件与摘要，报告缺口，不声称已经发送。收到设计后仍核验其证据、唯一变量、合约和授权范围。

## 1. Read State

先明确实验 ID、问题与范围。读取 research/BASELINE_CONTRACT.md 和当前设计，再从 CURRENT_STATE.md、EXPERIMENT_REGISTER.md、DECISION_LOG.md 定位相关记录，读取关联最新原始产物、源码和 git diff。证据依赖、冲突或独立审计需要时完整阅读；不能以状态摘要、图表或前一轮结论代替原始证据。

确认服务器路径、HEAD/dirty tree、实际Python/依赖/import paths、执行模式。分支名不能证明B0身份，历史数字不能写成新结果。输入缺失先查可用来源，明确缺口。

## 2. Audit Evidence

分开verified fact、observation、hypothesis、conclusion及证据范围。核对runtime、baseline协议、产物、标签、episode集合/顺序/worker分配、泄漏、不平衡、seed稳定性和混杂。摘要或图表不能代替原始结果。

检查当前停止条件，特别是Gate B。阻断期间可继续只读溯源、文档和隔离设计，不能扩大运行或解释受影响Probe。旧token被mask隔离不排除rollover风险，cache保留也不直接证明性能影响。

## 3. Generate Competing Explanations

不自动沿用前一假设。沿input→vision/language→V-L fusion→temporal state/history→actor→action→environment outcome追踪。按证据比较感知、语言、融合、记忆、探索、readout、termination、训练目标与安全约束，提出可区分机制的预测。

## 4. Select One Discriminative Experiment

在选择下一实验前，先判断当前阻塞项是否仍处于主研究目标的关键路径上；不要仅因为某个历史假设已经写进 CURRENT_STATE 就自动继续。
选择最能降低当前不确定性的最小实验。动代码前填写NEXT_EXPERIMENT的问题、证据、假设/竞争解释、control/treatment、唯一变量、固定条件、指标、预期/证伪结果、替代解释、允许/禁止改动、停止条件、产物和资源范围。

只读诊断不改变行为；离线状态干预和线上干预均明确归类。正式实验只改变一个核心变量。关键输入/字段不全则保持DRAFT/BLOCKED；设计建议不构成授权，只有 PI 可按 HANDOFF_PROTOCOL 批准。READY 不是控制分支的执行许可，Executor 不得自行置 READY。重试不能混入未注册变量。

## 5. Execute

只有当前角色的 PI 批准 A、独立 Claim B、各自 exact-SHA CI 和全部运行门禁通过后，才执行唯一获准实验一次；严格遵守 HANDOFF_PROTOCOL。分配唯一run ID，冻结设计，保存代码/配置/输入身份；只改设计允许的文件，不覆盖已有结果或未提交工作，不在B0上静默修bug。

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
