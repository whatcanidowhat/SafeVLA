# EXP-ACTOR-RESET-FULL200-001D — PI 实验设计草案

> Status: **DRAFT / PI_REVIEW / NOT_AUTHORIZED / NOT_EXECUTED**  
> Draft date: 2026-10-09 (Asia/Shanghai)  
> Scientific priority: **Success Rate (SR) 与官方 Safety Cost**  
> Protocol: Single-worker, fixed task manifest, full 200-task ObjectNavType, paired OFF/ON sessions  
> **重要：本文件是独立设计草案，尚未写入 GitHub research-loop 控制面，不代表已获得运行授权；不得据此启动模型/模拟器、提交 claim 或修改既有 001C 控制状态。**

## 0. 控制面现状与隔离边界（2026-10-09 只读核对）

- `whatcanidowhat/SafeVLA` 的 `research-loop/research/LOOP_STATE.json`：`state_version=47`，`experiment_id=EXP-COUNTER-DRIFT-EARLYEND-001C`，`status=APPROVED_FOR_CODEX`，`next_actor=CODEX`，`claim_id=null`，`max_gpu=1`，`max_episodes=32`。GitHub 上未检索到 `research/handoffs/counter-drift-earlyend-001c-20261009/RESULT_SUMMARY.md`。这**不证明**服务器上绝对没有运行活动。
- `NEXT_EXPERIMENT.md` 仍为 001C 的正式授权设计；**不得在 001D 的只读阶段覆盖它**。
- 若决定将 001D 改为正式唯一下一实验，必须先通过现行控制程序：核查 001C 是否已经 claim 或开始执行；若已 claim，不能直接撤销并覆盖；若未 claim，依照协议做撤回/过期处理、append-only 决策记录、CI 验证，然后另行注册 001D 并单独批准 P1/P2 预算。
- [LOOP_STATE.json](https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/LOOP_STATE.json) · [NEXT_EXPERIMENT.md](https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/NEXT_EXPERIMENT.md) · [BASELINE_CONTRACT.md](https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/BASELINE_CONTRACT.md)

## 1. 科学问题和估计对象

**主要研究问题**：对于冻结的 200 个 ObjectNavType 测试任务和固定处理顺序，保持官方 B0 其他设置不变，在每个 episode boundary 清理 root Actor 的 `time_step_counter` 和各层 K/V cache，能否提高成功率，同时不增加官方 Safety Cost？

**干预 (treatment)**：只在 episode boundary 上施加 root Actor temporal-state reset。不是 Stop Gate、Oracle、end mask、rerank、训练、改 checkpoint、改 horizon 或改评测定义。

**估计对象**：在 **单 Worker 连续 200 任务的特定顺序与随机初始化**下，ON 与 OFF 的 session-level SR、Safety Cost 和其他预注册结果之差。它不是官方默认 8-Worker 结果，也不是对历史四 Worker 173/200 结果的同协议复现，更不是本轮对自然 rollover 单独机制的识别。

**假设**：
- H-SR：`SR_ON > SR_OFF`。
- H-COST：`MeanOfficialCost_ON <= MeanOfficialCost_OFF` 是偏好的安全方向；单次 point estimate 不构成统计非劣效证明。
- H0：Reset 对真实结果没有稳定的净收益；观察差异可能由随机 rollout、顺序或环境波动产生。
- 其他结果（premature failed-end、rollover exposure、trajectory/logits）仅为探索性解释变量/过程指标，不可取代 SR 与 Cost 主终点。

## 2. 代码依据（只读静态证据）

- 官方参考 Git commit：`PKU-Alignment/SafeVLA@2aa82559d272b5f888e53433e258914057f15bed`。官方 `scripts/eval.sh` 默认 `num_workers=8`、`seed=123`、`shuffle=true`、`test_augmentation=true`；Python evaluator `--greedy` 默认为 false。单 Worker 是**新评测协议差异**。
- `online_evaluation/online_evaluator.py` 在一个 worker 时进入主进程执行分支，已选任务按 `tasks_queue.put(sample)` 的次序入队。
- `online_evaluation/online_evaluator_worker.py` 的 `start_worker` 每 worker 构建一个 Agent；`evaluate_on_task` 在每任务开头调用 `agent.reset()`。
- 官方 `InferenceAgentVIDA.reset()` 重置 task-local bookkeeping / `memory`，未显式清 root Actor decoder 的 model-side counter 与各层 K/V。
- Actor `forward` 中若 `time_step_counter >= max_steps` 则先归零，episode mask 由 model-side counter 与 episode-local timestep 构成。具体 runtime `max_steps=500` 来源于已接受的 EXP-RESET-001A 运行审计，须在本轮 runtime 再确认。
- 此前 001A 的相同输入 teacher-forced 实验接受 R1：状态携带引起的 rollover 可改变 Actor logits/动作偏好；**尚无 SR 或 Cost 收益证据**。
- 001A 使用 `reset_fix.patch` 的 root Actor-only reset package，并记录了 critic state unchanged；001D 在采用同一逻辑前仍需审核和做 A/A 无侵入验证。

## 3. 两个实验条件——必须保持唯一干预

| 属性 | OFF（Control） | ON（Treatment） |
|---|---|---|
| 代码基础 | 同一 accepted executable B0 | 同一 B0，仅注入经审核的 Actor-reset hook |
| Checkpoint / DINO | 同一 hash 和 runtime 加载规则 | 同一 hash 和 runtime 加载规则 |
| 数据 | 同一冻结 200 TaskSpec | 同一冻结 200 TaskSpec |
| 顺序 | 固定 `task_order_001d.csv` | 字节一致的任务列表与顺序 |
| Worker | 1 | 1 |
| 初始 seed | 123 | 123（同一初始化条件，不保证逐步相同） |
| Stochastic sampling | 官方默认，greedy=false | 同左 |
| Test augmentation | true | true |
| Horizon、Action、Success / Cost 规则 | 官方原样 | 官方原样 |
| Episode-boundary root Actor state | 保留 model-side counter/KV | `counter=0`、各 decoder layer `cache_k/cache_v` 清零 |
| Reward/Cost Critic、环境与其他策略状态 | 不额外干预 | 不额外干预 |
| Logger | 同一无侵入 trace 逻辑 | 同一无侵入 trace 逻辑 |

严格指定：**保留官方 `shuffle=true` 与 `seed=123` 的任务选择协议，但导出确切且稳定的任务列表/顺序；不要通过 `--no_shuffle` 或改成 greedy 来取得所谓“可复现”。**

每臂使用独立、全新进程/模型实例，同一 Python 和依赖环境；不能在同一个持续运行的 OFF 模型之后直接切换到 ON。

## 4. 任务集合与配对设计

1. 通过 accepted B0 评测器读取任务表，以稳定 `sample_id`、task specification digest、house ID、目标、初始位姿/seed 和原始顺序生成 `task_order_001d.csv`。
2. 要求 task manifest 恰好 **200 行、200 个稳定任务标识**，并且 OFF 和 ON 的 task spec 哈希一致；动态 episode ID 不能替代稳定 task ID。
3. 任务顺序必须在启动任何 experimental rollout 前冻结，且两个臂从同一 manifest 顺序消费。若当前 evaluator 代码不能保证一致，先只读定位并提出**最小独立 launcher/manifest adapter**，不能悄悄修改 B0 的 task sampling。
4. 保留 200 对全部结果；即使某任务失败、异常、提前终止或日志缺失，也必须归档，不因其不利而排除或重跑。
5. 单 Worker 解决 worker-local 的状态归属，不等于 stochastic rollout bitwise deterministic；任何重启、截断、数据缺失必须报告。

## 5. 分阶段资源与停止条件

### P0 — Read-only identity/protocol audit（0 GPU，0 episode；当前允许准备文档）

核查：
- accepted B0 的确切 source identity、Git HEAD + tracked/staged/untracked diff、DINO 本地加载批准范围、checkpoint/DINO SHA；本轮仅记录历史 SHA，未重新哈希服务器资源；
- 真实 Python / dependency / imported module paths、模型参数和 `max_steps` 的预期来源，运行前必须实证；
- 任务总数/顺序从哪里产生，`shuffle=true` 下如何在两臂保持稳定 manifest；
- root Actor 及 reward/cost critic 的实际 state ownership；reset hook 与官方 `agent.reset()` 的位置/次数；
- trace hook 是否可只读抓取真实一次 forward 的 policy distribution 与计数，不额外 forward/环境 query/随机数消费；
- 输出文件目录、故障恢复边界、最大 GPU/episode 上限及已存在 001C 控制授权的当前状态。

P0 交付：`P0_READ_ONLY_AUDIT.md`、`DRAFT_IMPLEMENTATION_PLAN.md`、`FROZEN_PROTOCOL_DRAFT.md`、任务 manifest 的**生成规范**（未产生真实 server manifest 时不得伪称已冻结）。若依赖任何 GPU/model load 或服务器文件变动则停止，进入单独请求授权阶段。

### P1 — A/A 验证与 instrument non-invasiveness（拟申请最多 10 live episodes / 1 GPU）

- **仅在另行批准后**，以 OFF 完整原行为分别运行固定任务前缀 5 个任务两次（5+5=10）；从干净模型进程初始化。
- 检查 5/5 task ID 和任务顺序、task specification、初始环境与 RNG provenance、同输入/同 RNG 条件下 policy outputs、实际动作、SR/Cost 记录、最终产物完整性。
- 不强制要求所有独立 stochastic online trajectories bitwise 相同：若存在随机性，报告全部差异、来源和可配对范围；不能把 A/A 失败解释为 reset 效应。
- 另外在**离线相同 Actor 输入**情况下做 logger OFF/ON 兼容核查，验证添加记录程序不引入额外 forward、RNG/counter/cache/执行动作变化；这一步不能自行扩大 P1 live 预算。
- Gate FAIL：路径身份或 checkpoint 不一致、episode 集合/顺序无法冻结、日志改变策略/环境行为、出现额外 forward 或指标缺失、或 A/A 的差异足以使单轮 A/B 结果无法解释。中止进入 P2，保留原始证据。

### P2 — Full-200 reset-performance A/B（拟申请 400 live episodes / 1 GPU）

- **仅在 P1 审查通过并再次获得明确授权后**运行：OFF 200 + ON 200；两臂互不并行、全新进程，预先冻结执行顺序（第一轮可 OFF→ON，后续种子复验再做顺序平衡）。
- 每臂最多 200 个真实 ObjectNavType live episodes，绝不追加挑选性重试、额外 seed 或隐藏 smoke。
- 保存每 episode 原始 task、动作、结果、详细官方 Cost 项、模型/state/rollover 元数据、stderr、exit code 和全套 manifests。
- 出现 manifest 身份漂移、额外 forward、代码/权重/依赖变化、超过任务预算、日志不可追溯或策略/环境语义更改，立即停止并保留 partial run；不能为了凑满 200 静默修复后继续。

**P1+P2 拟议预算：最多 410 live episodes，1 GPU。该预算目前没有执行授权。**

### P3 — Independent-seed replication（本轮不授权、不启动）

P2 结束后先进行 PI review。若观察到有意义的 SR 提升且 Cost 未明显恶化，再预注册至少两个额外完整 paired sessions（不同 seed/order，必要时 AB/BA 平衡），用 independent sessions 估计变异性和提高结果可信度。重复次数需根据波动和资源另行审批。

## 6. Primary outcomes 与固定判断规则

### 6.1 Success Rate

`SR_arm = count(success)/200`，`ΔSR = SR_ON − SR_OFF`，百分比差写成 percentage points (pp)。success 使用 B0 官方判定，无自定义替代定义。

保存 2×2 paired outcomes：`OFF failure→ON success`、`OFF success→ON failure`、两者均成功、两者均失败。所有 200 task 按冻结 manifest 分母配对，不用仅展示成功的任务。

### 6.2 Official Safety Cost

`MeanCost_arm = Σ official_cost / 200`；`ΔMeanCost = MeanCost_ON − MeanCost_OFF`。同时报告总和、非零 Cost 比例、P90/P95、最大值及各 cost component 的 per-task pairing。

**安全判断不可仅用均值是否小于 0 的 point estimate 宣称已证明 non-inferiority。** 本轮把 Cost 非增加作为优先探索方向，若进入确认阶段必须预先定义可接受的 Cost margin/uncertainty criteria，评审尾部安全风险。

### 6.3 结果审查优先级

- SR↑ 且 MeanCost↓、尾部未恶化：最佳候选，进入独立种子复验；
- SR↑ 且 MeanCost≈、尾部未恶化：潜在获益，需复验；
- SR↑ 但 Cost↑或长尾风险↑：不能宣称安全性能改进；
- SR 无净提高但 Cost 降低：可作为独立安全改进候选；
- 两者无有价值改善：下调 reset 作为主线性能改进的优先级。

P2 一对连续 session **仅提供探索性差异**。任务序列通过共享 Actor 状态互相关联，不能把 200 个 task 当成 200 个独立随机分配的实验单元，并据此做过度确定的显著性/置信结论。报告 per-task paired difference 主要用于描述，跨 seed/order 的重复 session 才适合做总体稳定性推断。

## 7. Secondary/process outcomes（用于解释，不能替代主终点）

- failed-end 数量与 time-to-end；success→failure 与 failure→success 的完整转移；episode lengths/horizon timeouts；
- OFF/ON 每任务开头 counter、实际 rollover 位置、当前 episode context 是否在中途断裂；不能把“OFF 与 ON 后续 logits 差异”自动等同于 context-collapse 的直接效应；
- 每步 `logits[20]`、`probs[20]`、sampled/argmax/executed actions、episode-local timestep、实际 Actor input digest；
- 首次同输入 policy divergence / first executed-action divergence / first pose divergence；之后视为环境反馈混合区；
- 如果 task 超过 500 step，ON 自己也可能发生 rollover，不得把 ON 描述为始终完整历史；
- 通过已有 passive fields 获取碰撞与可见性，不许为分析额外调用 simulator/环境控制器。

## 8. 输出与可追溯性

**每个 run 必备 server raw 资料**（Git 仓库只保存小文件索引和摘要，原始大量 traces 保留服务器）：

- 完整 command、实际工作目录、Git HEAD、branch、dirty diff、参与执行的 untracked code hash、Python/依赖、模型与 checkpoint/DINO hashes、模块真实 import path；
- 任务 manifest 与 seed/shuffle/stochastic/test augmentation/worker/horizon 配置；
- per-episode outcomes、每步真实 policy logits/probs/action、counter/rollover、与官方 cost components；
- 带 SHA256、bytes、coverage 的 `trace_index.json` 和 `ARTIFACT_INDEX.json`；
- 失败和 partial logs 同样归档。

拟议 Git-readable 交付：

```
research/handoffs/actor-reset-full200-001d-<cycle-id>/
  P0_READ_ONLY_AUDIT.md
  FROZEN_PROTOCOL.md
  TASK_MANIFEST.csv
  A_A_VALIDATION.md
  RUN_MANIFEST_OFF.json
  RUN_MANIFEST_ON.json
  paired_episode_results.csv
  aggregate_sr_cost.json
  cost_distribution.csv
  rollover_exposure.csv
  trace_index.json
  ARTIFACT_INDEX.json
  RESULT_SUMMARY.md
  REVIEW_NOTES.md
```

这些文件名是**计划产物，不是现有执行证据**；任何未执行阶段不得生成伪造成功行或空结果当成零效应。

## 9. 本轮只读审计需要回答的 8 个实际问题

1. 当前服务器上的 accepted B0 身份与已接受 001A 使用的执行源码是否一致？
2. `research-loop` 的 001C 是否已经 claim/开始运行？若无法查看服务器，仅报告 Git 已知状态与缺口。
3. 单 Worker 在 200 个任务中是否持续持有**同一个 Agent/Actor 实例**？新任务何处触发原始 `reset()`？
4. 固定 `shuffle=true, seed=123` 时，能否列出并哈希确切的 200 个 task IDs 与 task specs？task 队列顺序是否稳定？
5. Review patch 的 ON 是否真的只重置 root Actor 的 counter/KV，critic 等状态原封不动？
6. 官方的 `sample()`/`mode()`/`last_action_flat` 行为能否在相同命令下不变？
7. 每步 policy logits 与官方 Safety Cost 能否无侵入地采集？实际 logits 来源是哪里？
8. P1 A/A 预检需要何种最小 launcher/trace 实现？有哪些不可验证项与确切停止条件？

## 10. 下一交接指令（给 Codex / Agent；只读）

> 只做 EXP-ACTOR-RESET-FULL200-001D 的 P0_READ_ONLY_AUDIT。依次读取 AGENTS.md、research/BASELINE_CONTRACT.md、research/CURRENT_STATE.md、research/LOOP_STATE.json、research/NEXT_EXPERIMENT.md、research/EXPERIMENT_REGISTER.md、research/DECISION_LOG.md，核实任何未同步执行状态。随后审查 accepted B0/001A reset patch、任务队列、root Actor state reset、critic state、logits 与 official Safety Cost 记录路径。交付上面 8 个问题的证据、精确文件/函数/行号、P1 最小实现草案与执行预算。**不得 claim 001C、不得改 LOOP_STATE/NEXT_EXPERIMENT、不得写入或推送 GitHub、不得启动模型或 AI2-THOR、不得占 GPU、不得运行 Episode、不得宣称 001D 已批准。** 遇到无法只读核实的运行时信息，明确列为 P1 待验证，不要猜测。审计结束后停止并等待 PI 评审。

## 11. 证据参考

- `research-loop/research/LOOP_STATE.json` 与 `research-loop/research/NEXT_EXPERIMENT.md`：001C 现有控制状态，未切换到 001D。
- `research-loop/research/BASELINE_CONTRACT.md`：baseline 不变量与不改变 source semantics 的限制。
- `research-loop/research/handoffs/reset-001a-20261007/RESULT_SUMMARY.md`、`decoder_state_map.json`、`reset_fix_design.md`：Actor state carry/rollover 的 R1 实验与 reset package。
- 官方源码 `online_evaluation/online_evaluator_worker.py`、`online_evaluation/online_evaluator.py`、`architecture/models/allenact_transformer_models/inference_agent.py`、`architecture/models/allenact_transformer_models/allenact_dino_transformer.py` 与 `training/online/online_eval.py`。

---

**PI 最终状态：设计科学方向获认可；001D 保持 DRAFT / NOT_AUTHORIZED。当前仅完成已有 GitHub 文件的只读审查及设计文档草拟；未进行服务器运行时核验、未创建实际 task manifest、未启动实验、未向 GitHub 写入任何文件。**