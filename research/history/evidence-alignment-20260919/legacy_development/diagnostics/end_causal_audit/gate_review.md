# Gate Review — P0–P7 Final

本报告以 `canonical_final_v2/` 为最终代码版本的单 episode 证据，并以 `reset_two_episode/` 为补充 reset 证据。此前草稿中的 “Gate B: PASS（带 rollover 风险）” 和 “2/2 去重后的 end events” 不成立，现明确更正如下。

## Gate A: PASS

证据：

- repo `/nvme2/user/qyy/SafeVLA`，branch `baseline-experiment`，commit `60bc54fbdedaf5745d0476c25321e808708273aa`。
- 原始 dirty worktree 已保留；审计前 tracked patch SHA256 为 `d132497013a6f933995656635c2a3a99db6b2b7fb09f6a16890dd3e7b3084756`，既有 `full_diff.patch` SHA256 为 `672d53cb6f7db66dc29838ccd9ba1fd1d80a2f03f2c910f7d518a267670ad89b`，未覆盖。
- checkpoint `/home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt` SHA256=`05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301`；模型与 checkpoint 均为 417 tensors，loaded=417，missing/unexpected/shape mismatch 均为 0，关键 Actor/视觉/decoder/reward critic/cost critic 参数均来自 checkpoint。
- DINO SHA256=`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`；20 个 action names 与动态 `end_idx=4` 已记录。
- Python 3.10.19、torch 2.4.1+cu121、AllenAct 0.5.5a0、AI2-THOR commit `966bd7758586e05d18f6181f459c0e90ba318bec`、命令及 GPU 已记录。

## Gate B: FAIL

已通过部分：

- 最终 canonical 8 environment steps = 8 top-level policy decisions。
- 每步 Actor decoder、Actor linear、Reward Critic decoder、Cost Critic decoder 各恰好 1 次；没有额外 policy forward。
- 每步 20-D logits/probabilities 完整，logger 前后 Python/NumPy/Torch CPU/CUDA RNG 指纹一致。
- `selected_action == executed_action == environment_received_action == next_step_history_action`；无 remap。
- 两 episode、29 decisions 的补充 smoke 中，上述断言也全部通过。

失败证据：

- 【已证实事实】`InferenceAgentVIDA.reset()` 不清零 Actor、Reward Critic、Cost Critic 的 `time_step_counter`，也不清空 K/V cache；两 episode runtime 直接观察到 episode 1 结束后三分支 counter=8、cache 非零，reset 后保持不变。
- 【已证实事实】新 episode 的 `time_step=0` 与 trajectory mask 会隔离旧 cache；受控 decoder 测试中 dirty-old-cache 与 zero-old-cache 当前输出最大差为 0。因此不能声称每个新 episode 的第一步立即读取旧 token。
- 【已证实事实】当前模型 `max_steps=500`；forward 在累计 `time_step_counter >= 500` 时把 counter 归零。由于 counter 跨 episode 累积，某一 episode 的历史截断点取决于此前 episode 的总长度；评测环境 horizon 为 600，单 episode 本身也可越过该边界。
- 【合理推断】回卷时 decoder 从 position 0 重写并只保留回卷后的历史，因而 Actor 决策的可用历史窗口会被进程此前 episode 长度影响。这构成已定位但尚未做干预验证的 cross-episode state-carrier 依赖。

按本研究的严格停止条件，“reset 存在影响决策语义的跨 episode 状态残留”时不得继续解释 Probe/hidden-state 结果，因此 Gate B 判 FAIL。未静默修复该官方行为。

## Gate C: PASS（定义/时序验证，不代表统计充分）

证据：

- PRE 正式标签直接调用 `ObjectNavTask.successful_if_done(strict_success=False)`；环境接收 `end` 后走同一官方 non-strict 判定，没有另写 `visible && distance <= 2` 标签。
- 最终 canonical：1/1 实际 end 对齐（PRE false → POST success false）。
- 所有成功完成的 smoke 原始事件：5/5 对齐；4 个 illegal→failure，1 个 legal→success。这里是 5 次独立运行事件，不能按相同 task/step “去重”。

## 是否允许进入固定小规模诊断

NO。

P0–P7 到此停止。先将 counter/cache reset 行为作为候选实验变量，与官方 Baseline 严格隔离并设计最小证伪实验；未经再次授权，不扩大 episode 数、不训练正式 Probe、不运行 Oracle、不运行 200-task。

## 当前新增事实

- checkpoint 与 DINO 身份、参数加载完整性通过。
- 顶层与三分支 forward topology、动作映射、logger RNG 非侵入性通过。
- `end` 是 20 分类动作之一，应称 `policy_end_prob`，不是 success probability。
- 最终 canonical 在从未进入 official legal stop state 的情况下于第 8 步 greedy 选择 `end`：`policy_end_prob=0.993365`、margin=`0.986837`、distance=`2.464808`、pixel count=`0`，最终失败。
- 既有 full-200 结果的官方 Success=173/200=0.865；Safety Cost 共 145，集中于 12/200 episodes，top-1 episode 占 45.52%，top-5 占 86.90%。
- `diagnostic-action-history-20260802/launcher.log` 只证明运行启动至 Controller 初始化附近；没有完成标志和指标，相关 PID 已不存在，不能当作完成评估。

## 当前合理推断

- 最终单例同时符合 H-EXPLORATION 与错误 end 现象，但不能由单例判断系统性频率或机制。
- SafeRL 的 reward/cost/Lagrangian 目标结构允许 H-SAFEQUIT，但目前只有目标结构与相关性层面的可检验预测，没有因果证据。
- greedy 固定前缀重复运行仍出现 8/11 step 等轨迹差异，和 test augmentation/CUDA 非完全确定性相容；seed 123 不等于逐 bit 可复现。

## 当前待验证假设

H-PERCEPTION、H-REPRESENTATION、H-READOUT、H-ACTION-COMPETITION、H-SAMPLING、H-HISTORY、H-RESET、H-SAFEQUIT、H-EXPLORATION。

## 已否定或暂不允许的结论

- 【已否定于本 smoke】logger 引入额外 forward、改变 RNG、改变 action、混入 critic hidden 作为 Actor 表征。
- 【暂不允许结论】SafeVLA “因为害怕危险所以提前退出”。
- 【暂不允许结论】一次高 `policy_end_prob` 的 illegal end 证明 representation/readout mismatch。
- 【暂不允许结论】Gate B 失败前的 hidden-state/Probe 机制解释。
