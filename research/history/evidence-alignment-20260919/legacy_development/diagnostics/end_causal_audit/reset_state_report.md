# Reset / State-Carrier Audit

结论：FAIL。短序列中的旧 cache 隔离已得到支持，但跨 episode 累积 counter 会使历史截断点依赖此前 episode 长度，未满足 Gate B。

- 【已证实事实】`InferenceAgentVIDA.reset()` 复位 `steps_taken_in_task=0`、`memory=None`，递增 `num_evaluated_traj`；rollout storage 已初始化时调用 `after_updates()`。
- 【已证实事实】reset 不清零 Actor、Reward Critic、Cost Critic 的 `time_step_counter` 或 K/V cache。两 episode runtime 中，episode 1 结束后三分支 counter=8、cache 非零，reset 后保持。
- 【已证实事实】episode 2 首步 `time_step=0`、`mask=0`、`traj_index` 更新；29 steps 均满足 one-top-level/three-branch、RNG 不变、action chain 一致。
- 【已证实事实】默认启用 trajectory indexing；新 episode 首步的 decoder mask 只允许当前 key。受控实际 decoder 测试中，同 start position 下 dirty old cache 与 zero old cache 输出最大绝对差为 0；continued position 与 fresh position 当前输出最大差也为 0。当前 Attention.forward 未应用源码中存在的 RoPE helper。
- 【合理推断】在 counter 未回卷的短序列中，旧 episode token 不被新 episode 当前 token 注意；不能据此声称不存在所有跨 episode 依赖。
- 【已证实事实】模型配置 `max_steps=500`，环境 episode horizon=600。forward 在累计 counter 到 500 时自动归零；counter 的累计起点跨 episode 保留。
- 【合理推断】因此回卷可在后续 episode 的任意相对位置发生，并截断该 episode 的更早历史；位置由此前 episode 长度决定。这是明确的 cross-episode state-carrier 依赖，而不是仅凭 `memory=None` 可排除的问题。
- 【待验证假设 H-RESET】该依赖是否实质改变 action/logits、是否影响官方 full-200 指标，需要单独、受控、与 Baseline 隔离的最小 A/B 实验。本阶段禁止静默清零后继续称为 Baseline。

原始证据：`reset_state_audit.json`、`reset_two_episode/reset_state_audit.json`、`reset_two_episode/reset_transition_summary.json`、`reset_two_episode/smoke_steps.jsonl`。
