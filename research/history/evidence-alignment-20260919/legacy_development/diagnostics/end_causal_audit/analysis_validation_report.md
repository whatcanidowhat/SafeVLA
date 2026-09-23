# Analysis Validation Report

## Overall assessment

- 运行真实性、checkpoint 完整性、forward topology、动作时序和 observed end 标签：**可共享，但必须带样本量与非确定性 caveat**。
- 任何 Probe/hidden-state 机制结论：**Needs revision / 当前阻断**，因为 Gate B FAIL。
- 既有 full-200 官方指标：公式核验可共享；Safety Cost 必须同时报告分布，`percentage_collision`、`num_eps`、upload-count 名称不可按字面解释。

## 计算 spot-check

- Success：`173 / 200 = 0.865`，episode-level macro；已验证。
- Safety Cost：五分量总计 `4+15+68+5+53=145`，`145/200=0.725`；已验证。
- 非零 cost episode：12/200；最大 episode 66，`66/145=45.52%`；top-5 占 86.90%。均值被长尾主导，单报 0.725 会误导。
- SEL：episode macro mean of `success * expert_length / max(actual_length, expert_length)`；失败为 0，使用 action count 而非几何路径，不能称 SPL。
- Episode length：`len(all_actions)` 的 episode macro mean；成功/失败条件均值的分母分别为 173/27，不应和总体均值混作同一总体。

## 高影响问题

1. **Gate B blocker**：三分支 counter/cache 跨 episode 保留，累计 counter 在模型 max_steps=500 回卷；评测 horizon=600。历史截断位置可依赖前序 episode 长度。
2. **样本量**：final canonical 只有 1 个 end；所有 smoke 共 5 个独立 end。5/5 标签对齐足以检查已观察时序，不足以估计总体错配率。
3. **重复运行非确定性**：固定前缀、seed、greedy 仍得到不同长度/概率；官方 test augmentation 与 GPU 执行使 seed 不等于 bitwise reproducibility。
4. **Evidence Gaps**：repo 内无 expert trajectory generator/stop rule 证据，训练数据未在本机，checkpoint 不保存 IL 来源与完整 SafeRL 启动配置。

## 因果措辞审查

- `policy_end_prob` 仅为策略对 end 动作的概率，不是 success probability。
- `reward_value_estimate` / `cost_value_estimate` 是未来累计 reward/cost 估计，不是成功/危险概率。
- 当前不能从 illegal end、value 相关性或奖励结构推出 H-SAFEQUIT 因果结论。
- 当前不能从单例推出 representation-readout mismatch；须先解决/隔离 Gate B，再按 house/episode 划分做 probe 与基线/control。

## 交付判断

P0–P7 的审计产物可供研究审阅；研究结论必须停在“Gate B 已定位失败、Gate C 对已观察事件通过”。在未获授权前，不进入更多 episode、正式 Probe、Oracle 或 200-task。
