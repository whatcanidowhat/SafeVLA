# EXP-TOPO-MEMORY-001 — 房间访问只读影子验证协议（草案）

**状态**：DESIGNED / SYNTHETIC_UNIT_TESTED；真实历史轨迹逐步回放 NOT_RUN；SafeVLA 在线 Episode NOT_RUN；动态 Prompt 干预 NOT_RUN。尚未申请 Claim，未获 PI GPU 授权。

## 研究假设

H1：在历史无效 END 中，存在显著的目标房间未覆盖现象。历史 16 例的狭义目标房间未访问 9 例；目标集合严格一致 8 例。仅为描述性证据，不证明因果。

H2：使用只读 room ID 序列，能无损地记录每个 Episode 的独立房间首次访问、重访、房间切换和未取得有效 room ID 的时段。

H3（待后续独立授权）：在策略未被修改的条件下，拓扑日志识别的未覆盖区域及探索停滞具有预后价值；语言子目标与恢复动作需要独立 treatment 验证。

## 样本

`exploration_candidates.csv`：8 个 STRICT 核心候选（sub4/6/13/55/117/125/176/196）+1 个 AMBIGUOUS 扩展候选（sub175）。
同一行保留两个**不同实验周期**的字段，勿视为同一次评测。
负对照建议：sub3、sub46、sub68、sub87（旧历史已进入目标房间但失败），以及当前 OFF200 已成功的 sub13、sub55。
后者不是同一次运行内的成功配对，只用于展示同一任务的波动性。

## 只读 TopologyMemory 的契约

输入：episode_id、step、current_agent_room（整型或 room|N）；可选 room_type。
输出：visited_room_ids、entry_sequence、first_visit_step、observed_room_changes、unknown_observations、steps_since_last_new_room。
从不访问 task.successful_if_done、目标对象房间、完整 house topology、动作概率、Cost Critic，也不修改 prompt 或环境动作。
房间 ID 而非房间类别用于唯一标识；两间 Bedroom 不能合并。
UNKNOWN (-1) 必须保留，不能自动归入某一房间；跳过未知帧后的房间变更不能宣称相邻可通行。
“连续 50 步未进入新房间”是探索覆盖的**特征**，不是已证实的停滞或失败判定。

`topology_memory.py` 是独立纯 Python 原型，可通过 `python -m unittest -v test_topology_memory.py` 检验。
当前单元测试使用合成 room ID 序列，只证明模块逻辑一致，不证明可正确重放旧任务或提升 SR。

## 门禁 D0 — 证据与身份（只读 CPU）

1. 逐条核查候选 `sub_house_id`、`house_index`、`task_key` 与历史 `premature_end_cases.csv` 一致。
2. 固定原历史代码身份 `60bc54fbdedaf5745d0476c25321e808708273aa`；记录源文件/任务规格哈希与算法版本。
3. 目标合法性以 broad_target_ids 为准；狭义观测只能做候选筛选。sub175 不进入主因果结论。
4. 不得用 2026-10 OFF200 的终止步数覆盖 2026-08 旧历史数据。

## 门禁 D1 — 真实拓扑日志重建

缺少的必要输入：每个历史 Episode 的每步 room ID 或可确定 room ID 的位置轨迹（`last_agent_location` / `followed_path`）、对应房屋 `floorPolygon`、房间语义标签、原 Episode 身份。仅有 GitHub 的 `trace_sub*.csv` 是动作与量化概率，不包含每步位姿/房间序列，因此 **不能从当前公开归档恢复原始房间拓扑**。
如果可访问原始服务器归档：优先离线读取轨迹，不额外推理模型；每步通过 `get_room_id_from_location` 生成 room ID，再运行 tracker。
若位置轨迹不可获得，应报告 `INSUFFICIENT_TRACE`，不能把汇总 `has_agent_been_in_room=False` 伪造成访问序列。

比对：独立房间访问率、首入步骤、目标房间何时被覆盖、是否重复访问同一房间、失效 END 前未探索区域（仅 ground-truth analysis）、
是否出现大房间内 50 步以上正常搜索等误报。
目标房间成员关系只准在评测后离线 join，不得馈送给 Actor 或候选房间调度器。

## 门禁 D2 — 在线只读 shadow（需要单独审批，不是当前运行）

使用 `CurrentAgentRoom` 或 `get_current_room()` 的后观测，保持策略输入集、动作、终止、缓存、采样、RNG、forward 次数完全不变。
**注意**：读取 Controller 可能触发额外环境动作/缓存刷新；最佳方式是复用既有环境观察中已取得的 room ID，不要新增策略前 Controller 查询。
同一 Episode 内：
- before_action 记录已有 room ID 状态；
- after_action 记录执行后 room 及官方增量 cost；
- 真正终止后记录访问摘要；
- 逐步日志包含 `sample_id, house_index, sub_house_id, step, room_id, room_type, entered_new, steps_since_new, chosen_action, executed_action, last_action_success, cumulative_cost`。
仅在影子模式记录、不得改变 `agent.get_action` 的执行调用、Actor 状态、END 映射。
与独立 B0 严格核对 action digest / episode metrics / checkpoint、任务调度 / 每次运行身份。
若不等价则 `INVALID`，不能声称 shadow 无影响。

## 门禁 D3 — 控制接口先验验证（独立处理组，需要审批）

冻结当前 Checkpoint、相同视觉/历史输入；在隔离 Actor 实例上比较原指令与 RoomNav / ObjectNavRoom / 组合指令对 logits 和动作的影响。
结果指标为房间级推进率，而不是仅比较文本 embedding 距离。
对提示变化同时控制 actor cache / time_step_counter / prev_action / RNG；新样本对照不能是额外 live Actor forward。
若没有可重复的房间级推进，停止 Prompt-only 架构，转向显式 waypoint / 子目标接口。

## 门禁 D4 — 干预消融（独立新授权）

E0 = 原始 B0；E1 = 仅拓扑 Shadow；E2 = 无拓扑，仅预注册停滞触发+文本改写；E3 = 拓扑 + 文本引导。
需要额外测试：终止校验与重规划（不能仅覆盖 END 为 sub_done）、以及安全预算/风险约束，
以隔离 SR 提升与 Cost 变化机制。
相同 TaskSpec/Seed/运行环境，报告有效样本数、episode 身份/worker、配对 SR、总 Safety Cost / 每类 Cost、错误拦截成功 END、房间覆盖、新增动作、误触发、成本尾部；
至少一个新房间实际进入才能视为成功子目标，不以 Prompt 变化作为成功证据。
**现有 Actor Reset ON200 未实际测量，不能混入本实验**。

## 研究判定

- `PASS-D1`：只读房间记忆与真实轨迹和独立统计一致；不表示 SR 提升。
- `PASS-D3`：语言确实驱动 room-level 行为；不表示安全收益。
- `PASS-D4`：相同协议下效果经过独立复验，SR ↑ 且 Safety Cost ↓（含尾部、任务迁移与代价）才能支持最终主张。
- 若真实轨迹缺失、官方 B0 不变量被破坏、目标房间 oracle 泄露、提示无效、增加导航危险，应明确报告 BLOCKED/INVALID 或负结果。

## 直接源码链接

- https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/handoffs/premature-end-dynamics-001-20260923/premature_end_cases.csv
- https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/handoffs/actor-reset-off200-on200-002-20261014/OFF200_BEHAVIOR_PROXIES.csv
- https://github.com/whatcanidowhat/SafeVLA/blob/baseline-experiment/environment/navigation_sensors.py
- https://github.com/whatcanidowhat/SafeVLA/blob/baseline-experiment/utils/data_generation_utils/navigation_utils.py
- https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/BASELINE_CONTRACT.md
