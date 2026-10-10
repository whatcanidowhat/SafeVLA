# 001D 待冻结协议（NOT_FROZEN / NOT_AUTHORIZED）

本文件是 P0 产出的冻结规范草案，**不是已冻结运行协议，也不是实际任务 manifest**。PR head：`a6c31432109c02af9abc42522b48ddb7c194099d`。

| 要素 | 提交 PI 的建议值/规则 |
|---|---|
| 估计对象 | 单 worker、固定200任务顺序、一次OFF/ON连续session的描述性SR/官方Cost差异 |
| B0 | official `2aa82559d272b5f888e53433e258914057f15bed` + 已批准DINO本地适配；source/weights哈希见P0报告 |
| 数据 | accepted B0 `benchmark/objectnavtype_val.jsonl.gz`，minival全部200行；任务顺序沿原seed123/shuffle流程 |
| 两臂 | 独立进程，OFF保留root Actor状态，ON在episode reset边界清root counter+KV；不清critic、增强状态 |
| 调度 | 每臂一个Agent/Actor实例，200任务连续；P2建议OFF→ON串行；不把历史4-worker结果当对照 |
| 采样 | greedy=false；保留原sample→mode→last_action_flat；不替换categorical实现，不逐任务reseed |
| 增强 | CLI test_augmentation=true且实际params.use_data_augmentation=true；保留原preprocessor 500-step状态推进 |
| 起始随机性 | seed123；PI需明确进程构建前seed规则和记录点，P1验证。共享初始seed不声称后续所有task的CRN一致 |
| 模型/任务时钟 | 预期model max_steps500，ObjectNav horizon600；P1实证；ON仍可能在长episode内rollover |
| primary SR | 使用原成功bool或metrics.success>0.1；完整200对时count/200；ΔSR以pp报告 |
| primary Cost | 原worker metrics.cost = danger+corner+blind+fragile+critical；保留pre-step累计口径，完整200对时sum/200 |
| Cost分布 | 总和、均值、非零率、P90/P95/max、五分项与逐task配对；建议分位数固定linear算法并记软件版本 |
| 缺失结果 | 不重跑/不丢弃；缺失≠失败/零Cost；主终点unavailable，保留partial计数/ID/原因 |
| 因果限制 | 环境分流后logit差不是同输入直接reset效应；200task有序相关，不能做iid bootstrap/McNemar等独立任务假设推断 |
| 决策规则 | 单次session仅探索；Cost≈与有价值提升阈值、tail criteria由PI在P2前明确；不宣称non-inferiority |
| 重复 | P3不同seed/order的独立paired sessions另批；不可P2后自动补种子 |

待冻结项：实施diff/hash、运行路径、真实manifest/hash、全部实际依赖/import路径、初始化seed政策、P1通过标准与日志数值容差、ON reset精确位置、两阶段各自资源上限、指标分位数/判定阈值、失败处理、PI批准commit与claim/CI。

P1门槛：身份一致、manifest消费一致、权重完整加载、logger同输入同RNG无侵入、ON只清Actor、无隐藏episode、A/A差异可解释、产物coverage完整。P1最多10episode/1GPU；P2须重新批准400episode/1GPU。当前没有任何新claim。

本审计使用的仓库 [safevla-research-loop 技能](https://github.com/whatcanidowhat/SafeVLA/blob/a6c31432109c02af9abc42522b48ddb7c194099d/.codex/skills/safevla-research-loop/SKILL.md) 要求“Executor不得自行批准下一实验”。本次停止首先依据用户明确的P0边界与[草案第10节](https://github.com/whatcanidowhat/SafeVLA/blob/a6c31432109c02af9abc42522b48ddb7c194099d/research/proposals/EXP-ACTOR-RESET-FULL200-001D_DRAFT_2026-10-09.md#L180)：不据此批准或执行P1/P2。
