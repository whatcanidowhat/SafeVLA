# Official Result Calculation & Reasonableness Audit

## Existing 200-task run

- 【已证实事实】Success = 173/200 = 0.865。代码在 episode 层 macro mean；内部人为加 1e-8，所以 W&B 显示 0.86500001。
- 【已证实事实】SEL = episode macro mean of success * expert_length / max(actual_length, expert_length)，失败为 0，结果 0.7761195002。它按 action count 而非几何路径长度衡量效率，不能当 SPL。
- 【已证实事实】Episode length 为 len(all_actions) 的 episode macro mean，92.515；成功 episode 条件均值 55.16185，失败 episode 条件均值 331.85185。
- 【已证实事实】Safety Cost 为每 episode 对五类 cost 的累计，再做 episode macro mean：总 cost 145，均值 0.725。分量总和 corner=4, danger=15, blind=68, fragile=5, critical=53。
- 【已证实事实】cost 极度集中：188/200 episode 为 0；12/200 非零；最大单 episode 66，占总 cost 45.52%；top-5 占 86.90%；median=0。只报均值会掩盖长尾集中。
- 【已证实事实】percentage_collision 实际公式是 1 - mean(last_action_success)，分母为所有非 -1 action observation；它测 action failure rate，不是 collided 字段的纯物理碰撞率。输出还因上游容器成为单元素 list。
- 【已证实事实】Room coverage 是每 episode 的 visited-room percentage 再 macro mean（0.54478）；total rooms visited macro mean 为 2.195。
- 【已证实问题】num_eps 被传成 eps_idx+1，而 eps_idx 已在每 step 开始递增，因此实际为 episode step count+1；200-run 的 93.515 不是 episode 数。命名/实现不合理，但未修改 Baseline。
- 【已证实问题】“Uploading results from 2 tasks out of 1 emitted”使用 len(results)，而 results 是二元组，因此该日志计数不是任务数。未修改 Baseline。
- 【合理性结论】Success/SEL 的宏平均与常见 episode-level 报告一致；Safety Cost 必须同时报告分布与集中度；collision 命名不准确；num_eps 与 upload-count 日志不应用于科研结论。建议另建 audit_metric_v2，不替换 official metric。
