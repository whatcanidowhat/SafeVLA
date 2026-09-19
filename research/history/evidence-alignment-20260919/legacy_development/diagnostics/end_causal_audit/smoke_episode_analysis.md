# Final One-Episode Greedy Diagnostic

最终证据版本：`canonical_final_v2/`；episode `ObjectNavType_420_1786530346_findamug`，target `mug.n.04`，8 decisions，final failure。

- action sequence：`l, m, r, r, m, m, m, end`。
- 【已证实事实】8/8 steps 均为 exactly one top-level decision；Actor decoder/linear、Reward Critic decoder、Cost Critic decoder 每步各一次；20-D logits/probabilities 完整。
- 【已证实事实】8/8 steps 的 logger 前后 RNG 指纹一致；selected/executed/environment/history action 全链一致。
- 【已证实事实】全程 `stop_legal_exact_pre=false`、target pixel count=0；第 8 步环境真正收到 `end`，官方 final success=false。
- 【已证实事实】第 8 步 `policy_end_prob=0.9933653474`、`end_rank=1`、`end_vs_best_nonend_margin=0.9868370532`、entropy=`0.0404556319`。
- 【已证实事实】第 8 步 target distance=`2.4648081325`，episode 从未进入 official legal stop state。
- 【已证实事实】第 8 步 `reward_value_estimate=0.5708865523`、`cost_value_estimate=1.2387325764`；二者分别是训练目标下的未来累计 reward/cost 估计，不是成功/危险概率。
- 【合理推断】该单例显示的不是“end 仅略微胜出”，而是 Actor 在该帧对 end 高度相对集中；它符合错误高置信 end 与 H-EXPLORATION 的表面现象，但不能区分感知、表征、readout、history、reset 或 SafeRL 目标机制。
- 【待验证假设】H-PERCEPTION/H-REPRESENTATION/H-READOUT/H-HISTORY/H-RESET/H-SAFEQUIT/H-EXPLORATION；由于 Gate B FAIL，本轮不升级任何机制解释。
- 【暂不允许结论】一次 illegal end 不能证明系统性 premature end，也不能证明 SafeRL 安全目标导致 end。

复现注意：相同固定前缀、seed=123、greedy 的多次 smoke 曾在 8 或 11 steps 结束，`policy_end_prob` 也不同。由于命令保留官方 `--test_augmentation` 且 GPU kernel 不保证逐 bit 确定，这些运行是诊断样本，不应被描述为严格确定性复现。
