# Official Stop-Legality Alignment Audit

结论：PASS（标签定义与 PRE→POST 时序通过；样本量不用于估计总体错配率）。

- PRE 正式标签直接调用 `ObjectNavTask.successful_if_done(strict_success=False)`。
- Environment 对 `end` 的官方路径在 `AbstractSPOCTask._step` 中调用同一 `successful_if_done()` 默认 non-strict 判定；未另写一套 `visible && distance <= 2` 正式标签。
- 每个事件均记录 `policy_selected_action`、`executed_action`、`environment_received_action`；已观察事件无 remap。
- 最终代码版本 `canonical_final_v2`：1/1 end 对齐，PRE false → POST success false。
- 所有成功完成的 smoke 原始事件共 5 次：4 个 illegal→failure、1 个 legal→success，5/5 对齐。它们是独立运行事件；此前按 task/step “去重成 2/2” 是逻辑错误，现已更正。
- 该结果只验证当前观察事件的判定定义与时序，不能证明未观察 episode 中永不发生错配。
