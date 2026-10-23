# PI → Codex: EXP-TOPO-MEMORY-FOURWAY-001 | 四线并行执行交接

## 本轮 PI 意图

时间优先，**允许四条验证工作流并行开发、CPU 核验和资源预检**，但每个实验产生独立 manifest、输入、运行目录、状态和结论。
不要将“同时开展研究”误解为“四条 GPU 策略运行可以在未验证资源容量、未完成预检时无条件并发”。不要自行杀其它进程。没有可用服务器 GPU / Checkpoint 时如实 BLOCKED，禁止提交虚构 SR/Cost。不得改 B0 的 reset、END、奖励、动作映射或官方 metric。

## 输入、身份与冻结样本

- repo: `whatcanidowhat/SafeVLA`；研究分支 `research-loop`；B0 历史开发身份 `60bc54fbdedaf5745d0476c25321e808708273aa`（每次运行再记录精确当前 HEAD/工作区）。
- 16 个历史 invalid END：`research/handoffs/premature-end-dynamics-001-20260923/premature_end_cases.csv`，该组来自 2026-08 历史评测。
- 2026-10 OFF200：`research/handoffs/actor-reset-off200-on200-002-20261014/OFF200_EPISODE_METRICS.csv`、`OFF200_BEHAVIOR_PROXIES.csv`、`TASK_MANIFEST.csv`。这和前述 2026-08 是两次不同运行，不得直接比长度/Cost 断言干预效应。
- 冻结 CORE_STRICT = `4,6,13,55,117,125,176,196`；EXT_AMBIGUOUS = `175`；NEG_CONTROL_VISITED_STRICT = `3,46,68,87`。
- 历史统计：16 invalid END 中 9 narrow target room 未访问，8 STRICT + 1 AMBIGUOUS；这是**观测关联不是因果**。
- OFF200：169/200 SR, cost 177, 19,270 decisions, ON200 缺失。

## 并行 Lane V1 | 历史轨迹拓扑重建（CPU）

**输入**：原服务器保存的每步位置（`followed_path` / `last_agent_location` / observation history）、house `rooms[].floorPolygon` 和所有 success-eligible 的 `broad_target_ids`。优先采用只读原始归档，验证 SHA 与历史 task_key。对 8 CORE + 4 negative controls 输出逐步 room ID。

**产物**：`V1_ROOM_TIMELINE.csv`, `V1_ROOM_COVERAGE.csv`, `V1_INPUT_MANIFEST.json`, `V1_VALIDATION.json`。对齐帧、步数、END、房间 ID；目标房间只能**离线评价时 join**，不能作为策略输入。无法恢复真实位置就 `BLOCKED_MISSING_TRAJECTORY`；不能由 `has_agent_been_in_room=False` 合成 room 序列。

## 并行 Lane V2 | 拓扑记忆 Shadow 等价验证

**输入**：已有 `CurrentAgentRoom`，优先复用 `task.get_observations()` 的已取得字段、`task.observation_history`；勿添加每步 Controller 查询。用户提供的 `topology_memory.py` 是纯内存基准实现。

**产物**：`V2_SHADOW_VISITS.jsonl`, `V2_REPLAY_EQUIVALENCE.json`, `V2_ROOM_MEM_EVENTS.csv`。测 episode reset / unknown room / rerevisit / 同类房间多个 ID / 顺序唯一性。真实 B0 vs Shadow 必须比较每一步**执行动作**、action logit/prob、RNG、Actor cache/counter、episode 指标和 official Safety Cost，保持原始 worker/任务/seed/增强。日志只读，不喂给 Actor。仅合成单元测试 PASS 不能写 B0_EQUIVALENT。

## 并行 Lane V3 | 语言引导的房间级行为测试

冻结 exact SafeVLA Checkpoint，使用独立 Actor 实例/缓存进行反事实前向；不能在 live Actor 额外 forward。离线对照见 `prompt_probe_plan.csv`：原指令 vs RoomNav、ObjectNavRoom、复合指令、分阶段指令。

1. **离线判别**：固定图像、历史、prev action 和 cache，比对 logits 差异与方向一致性。
2. **独立 rollout**：测试真实“到达目标房间”的比例、进度、是否提前 END、额外 Cost，不把房间真值作为 Prompt 路由输入。
3. **中途切换**：分别研究保留旧 K/V cache 和独立的受控缓存处理；Actor reset 是独立变量，绝不能混在第一组。

**通过门禁**：不是 `P(end)` 变化或者文本可被编码，而是能可重复地前往指定**已感知到**的房间。若无方向响应，不要通过提示词拼凑出“有效”，报告 `PROMPT_ONLY_NOT_SUPPORTED`。

## 并行 Lane V4 | 联合拓扑 + 语言干预

设计至少 B0、仅停滞判定、仅拓扑（影子不改变动作）、仅 Prompt、拓扑 + Prompt、必要时再分离 END 拦截/安全恢复。每个 treatment 独立的 checkpoint、agent cache、任务、seed、日志。

- topology graph 的“Unvisited”必须是**已发现但未探索**，不可读仿真器完整真实房间图作为可部署策略输入。
- semantic Room Type 不等于 room ID；未知房间的存在不能假装已知。
- 原始 `end` 不能简单替换成 `sub_done` 并宣称实现重规划；被拦截后必须有可观察的真实导航动作。
- 保留原任务指令和成功判定；oracle `task.successful_if_done()` 只能记录 GT，禁止决策侧读取。
- 若连续 50 步未进新房，也不能单独作为停滞真值；需结合原地位移/动作循环/视觉新增信息/碰撞证据。
- 输出 matched task SR、每类 official Cost、总 Cost、P95/P99、任务步数、房间覆盖、进入新房比率、成功 END 被错误干预的比例、恢复额外风险与配对任务状态。严格区分 8-case diagnostic 与 full200 headline。
- 任何换 checkpoint、worker 顺序、随机种子、reset、代价口径或目标可见性定义都必须另列实验组，不能与 B0 单变量干预混称。

## 资源与最小任务调度

先并行处理四条线的代码、历史资料和 CPU/模型资源预检。V1 全 CPU。V2 的纯逻辑全 CPU；在线等价需要策略运行。V3 用独立模型实例按实际可用 GPU 顺序排队；V4 GPU 评测必须在 GPU、Checkpoint、Dataset、任务 Identity PASS 后启动。不要抢卡、杀进程或超过实际可用资源。

若 PI 已明示新的 GPU 授权范围，则在范围内执行；否则冻结相关门禁并一次性汇报资源缺口，避免重复 Claim/CI 请求。既有 OFF200 数据可在来源身份一致时作为**历史参考**，不可未经核对直接与新 treatment 成对比较。

## 最终四份交付

每条线输出：`RUN_MANIFEST.json`, `RESULT_SUMMARY.md`, `VALIDATION.json`, 原始可复查数据，和一个 status (PASS / BLOCKED / INVALID / FAIL)。四线统一 `FOURWAY_MATRIX.md`：逐条列已证实观察、反例、前提、混杂、下一实验。若没有真实 treatment 数据，必须显示 `SR delta = NOT_MEASURED`、`Cost delta = NOT_MEASURED`，不得填写 0。
