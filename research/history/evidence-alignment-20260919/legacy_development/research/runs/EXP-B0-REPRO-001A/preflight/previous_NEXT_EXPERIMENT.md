# Next Experiment

Experiment ID: EXP-RESET-001
Status: DRAFT_BLOCKED
Phase: B — isolated hypothesis test（未执行）
Execution authorization: 本轮仅构建协议；已有 Gate B 阻断未解除，本草案不构成运行授权。

Research Question: 固定全部当前序列输入和随机状态时，继承 counter 起点是否通过 rollover 改变 Actor logits/动作？

Current Evidence: diagnostics/end_causal_audit/reset_state_report.md、reset_two_episode/reset_transition_summary.json 记录 reset 后 counter/cache 保留；29 decisions 未观察 rollover。

Hypothesis: H-RESET；counter 起点改变边界时刻，导致当前历史截断与输出差异。
Competing Explanation: 旧 token 已被 mask 隔离；历史差异来自增强、RNG 或设备执行。

Control: 独立离线执行副本，保留官方 counter 推进规则及明确记录的继承 counter 起点。
Treatment: 相同副本，仅将序列起点 counter 设为0；不同时改变 cache 内容、输入、mask、权重或官方 rollout 代码。
Unique Variable: 序列开始时 time_step_counter 的初始化值。涉及的分支和初始化时点须依据实际调用代码补全并固定，不混入清 cache 或其他 reset 修改。

Fixed Conditions: 相同代码/权重、重放输入及哈希、cache 初值、mask/traj_index、history、模型模式、设备/dtype和RNG；同条件重复以估计数值波动。继承 counter 值和序列长度须在核实输入后预注册，覆盖边界前/边界/边界后，目前尚无可执行参数。

Metrics: 每步 counter/start position、可用历史范围、Actor logits 最大绝对差/分布差、argmax一致率、重复运行噪声。不以离线结果宣称SR/Safety Cost提升。
Expected Result: control 特有的 rollover 邻域出现超过预设数值容差的可重复差异，并能追踪历史范围变化。
Falsifying Result: 确实覆盖边界、对照有效且误差受控时无差异，削弱该输入范围内的H-RESET，不证明所有任务都无影响。
Alternative Explanations: 输入/增强/cache不一致、分支错误、RNG差异、重放不符合真实调用语义；未排除则不足以解释机制。

Allowed Code Changes: 完成设计并获得执行授权后，仅在独立实验目录增加离线 harness 和记录代码，设置独立副本状态。
Forbidden Code Changes: 不改当前B0、不清live actor cache、不改success/end/horizon/官方指标、不在线额外forward、不训练或加入oracle。

Stop Conditions: 重放输入缺失；无法固定唯一变量；runtime/权重不明；差异不可复现；需同时改多项状态；manifest与实际不符；达到预注册时间/资源上限。保留部分结果，不自动扩大实验。
Required Artifacts: 按BASELINE_CONTRACT保存run manifest、代码/输入哈希、参数和command、逐步结果、logs、RESULT_SUMMARY.md；离线无环境指标须注明。

Readiness: 补齐真实输入、分支范围、counter值、长度、容差、资源/时间上限和command，并确认执行授权覆盖后才能置READY。下一步仅为只读检查输入与调用语义，形成最终设计；不要先运行后补字段。
