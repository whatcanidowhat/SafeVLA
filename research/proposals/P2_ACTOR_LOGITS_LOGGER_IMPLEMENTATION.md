# P2 准备工作｜Actor logits 全量记录模块（Codex 开发任务）

**任务状态：IMPLEMENTATION_REQUESTED；未实现、未运行、未验收。**

PI 已采纳“先实现 OFF/ON 共用的无侵入 Actor logits Logger，验证日志开启前后输出一致，再冻结 OFF200/ON200 命令”的下一步建议。本文件仅是**开发任务书**，并非运行授权。此开发分支从已验收 B0 `PKU-Alignment/SafeVLA@2aa82559d272b5f888e53433e258914057f15bed` 派生，不能直接修改已验收 B0、P1、P1R、P1R2 的目录和实验产物。

## 0. 控制边界（不可绕过）

- GitHub `research-loop` 当前 HEAD `a88d5ab1f29127753d5b6c1c42bf7b15f891b566`（2026-10-12 07:46 UTC 核实）是 **P1R2** 的 `APPROVED_FOR_CODEX`、最多 **10 Episodes**，尚未 claim。**它不是 P2 正式 400-Episode 批准，不得复用其额度启动 OFF200/ON200，也不允许改动已批准的实验设计。**
- Codex 可以在本开发分支/全新隔离 worktree 实现代码与 CPU 静态/单元测试。任何 GPU checkpoint 真实回放、模拟器、Episode 测试均需符合独立的有效 PI 批准和 CODEX claim+绿色控制 CI。无批准时测试记录为 `NOT_RUN_AWAITING_AUTHORIZATION`。
- 目标最后是 **OFF 原始 B0 200 + Actor-only Reset ON200，共 400 Episodes**；在日志等价性验证与独立 P2 授权之前不得启动。不得自动转移 P1R2 未用预算、不得覆盖旧交接、不得触发 ON/P2 live evaluation。
- 已验收 B0 的本地 DINO 权重加载适配属于预先接受的单独环境事实，Codex 必须先核实服务器的真实工作树和适配差异；本 GitHub 派生分支本身并不包含那些未提交的本地改动。任何开发改动只能放隔离 worktree。

## 1. 本次两组唯一科学变量

| 项目 | OFF200 | ON200 |
|---|---|---|
| B0 模型/权重/DINO/数据集 | 相同 | 相同 |
| 单 worker / 持续 Agent | 1 | 1 |
| ObjectNavType | minival 前 200 个冻结任务 | 完全相同的 200 个任务、顺序、TaskSpec |
| RNG / 配置 | seed=123，shuffle=true，greedy=false，test_augmentation=true，horizon=600 | 相同 |
| Actor `time_step_counter` / decoder K/V | B0 原始跨 Episode 状态行为 | **仅在 Episode 边界**重置 root Actor 计数器和各层 K/V |
| Reward / Cost Critic 及 RNG/augmentation | 原始逻辑 | 原始逻辑，绝不可主动重置 |
| Logger | 完全相同的实现和输出 schema | 完全相同的实现和输出 schema |
| 官方 Safety Cost | 环境的五项 `danger+corner+blind+fragile+critical`（原始 pre-step 计法） | 相同 |

冻结的 200-row 原始 TaskSpec manifest SHA256 为 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0`，前 5 个原始源索引是 `127,103,16,6,50`；需从 B0 loader 独立再生并严格比对。种子能固定任务选择/顺序，但并不证明所有随机动作或环境轨迹一致。

真实代码入口：`architecture/models/allenact_transformer_models/allenact_dino_transformer.py` 根 Actor 输出来自 `self.actor(beliefs)`；根 Actor head 是 `agent.actor_critic.actor.linear`。评估执行 `inference_agent.py` 的 `self.actor_critic(**agent_input)` 后 `distributions.sample()`；两个 Critic 在完整 forward 中仍运行，但不负责选择真实动作。**不得误把两个嵌套 Critic 的 actor.linear 输出或 Cost Critic 的 `c_values` 当成根 Actor logits 或实际 Safety Cost。**

## 2. Logger：完整、不改动作、逐步可审计

1. 一个共用可配置组件 `ActorLogitsLogger`，建议在**根 Actor `actor.linear` 注册只读 forward hook**，不要修改 `forward()` 数值路径。默认完全关闭；启用也不得额外调用 policy、`.sample()`、`.mode()`、环境 step、控制器、数据增强预处理或 RNG 操作。
2. 每一个实际 Actor policy decision 一条完整记录（每个有效动作维度均保存，禁止 Top-k、四舍五入、降精度），严格区分：
   - `raw_actor_logits`：根 Actor 线性 head 的**原始、未规范化张量**，保留真实 shape 和 dtype，不能拿 `Categorical.logits` 冒充；
   - `normalized_logits`：最终动作分布的规范化 logits；`action_probs`：全量动作概率（不能从日志中反向采样）；
   - `sampled_action_index`、`greedy_mode_action_index`、`executed_action_index`/`executed_action_name`（注意评估器可能在 agent 之后修改或替换动作，必须区分）；
   - `episode_order`、`sample_id`、`task_source_index`、`step_index`、`global_step`、稳定的 `decision_id`、`prev_action`、目标标识、root `time_step_counter` before/after forward、root decoder 起始位置、有效 mask 描述、前向/采样调用数量。
3. 每个 Episode 边界额外保存 root Actor `time_step_counter`，每层 K/V 的 shape/dtype/device 与完整张量哈希（允许仅边界做哈希）；Critic 和 Cost Critic 同样只读记录 K/V 和计数器作为**不受干预证明**。不要每步保存巨大 KV 张量；不要用 logger 执行 reset。Actor-only Reset ON 的实际 reset 实现仍为**另一个单变量模块**。
4. 精确数值以 per-episode **lossless NPZ/NPY** 保存，保留原生 dtype/shape/order；`actions.jsonl`、`episode_metrics.csv`、`boundary_states.jsonl`、`RUN_MANIFEST.json`、`ARTIFACT_INDEX.json` 记录对齐键、哈希、元数据及输出路径。分离 `OFF200/` 和 `ON200/`，按 Episode 原子写入或临时文件完成后重命名，崩溃时保留已完成证据与 `INCOMPLETE` 明示状态；绝不能静默补零、跳过或重复记录。
5. 验证无额外模型 forward、无额外随机数、无新 episode reseed、无 action map 改动；无害的 `detach().cpu().clone()` 仅用于**复制已有输出**，不可改变 Actor/Decoder/Memory/RNG。若 hook 截取/logging 失败，应 fail-closed，记录准确失败位置，不能继续运行并声称完整日志。
6. 对日志做源文件/权重/数据哈希；checkpoint 原始 SHA256 `05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301`，DINO SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。全量 numpy/tensor/权重文件留服务器，GitHub 只提交小型校验摘要及 SHA，不能上传 checkpoint。

## 3. 开发阶段与硬性验收（此任务不启动 GPU）

**Phase 1：独立实现/CPU 单元测试。** 提交 Logger、Writer、Reader、Validator、分析器与明确启停选项；测试原生 dtype/shape lossless 回读、多 Episode / crash partial recovery / 缺失行或重复行检测、actor head 与 critic head 区分、前向次数/动作数量一致、禁止多次 sample、无额外随机操作、OFF/ON 使用同一个 Logger 类。可以使用人工数据／CPU mock 进行功能单测，但**不能将 CPU mock 测试误称真实 SafeVLA 权重验证**。

**Phase 2：真实 GPU 离线等价性（须另有已批准执行授权后才可开始）。** 历史已保存的 600 次 Actor 输入 `captured_actor_inputs.pt` SHA256 `417e684e96563b043138d328776d693f05dc74bb7c074099dea39c3b5881500e`。逐对对齐相同输入、Actor/critic/cache 状态、RNG，Logger OFF vs Logger ON：
- `max_abs(raw_logits, normalized_logits, probs, next K/V) <= 1e-5`；
- sampled action、mode、executed action/历史、RNG 完全相同；
- Actor、Reward Critic、Cost Critic forward 次数与原始执行相同，no extra model/augmentation/controller forward；
- 任一失败 STOP 并保留错误证据；不得静默重试、换 seed 或偷改 B0。

**Phase 3：正式 P2 冻结建议（待 PI 审查，不运行）。** 准备未来独立批准后可用的两条参数完全一致、唯 Actor-reset 不同的 `OFF200` / `ON200` CLI；单 worker、完整 frozen task manifest 与样本 ID、seed123、shuffle=true、greedy=false、test_augmentation=true、同一个 checkpoint、同一个 logger、独立输出目录、原始 official SR/Cost。保留 GPU0-3 公共卡非破坏性队列方案，严禁杀他人进程。

## 4. 数据分析要求

- 官方终局汇总 `ΔSR=SR_ON-SR_OFF`、`ΔCost=Cost_ON-Cost_OFF`，任务级配对变化及 `danger/corner/blind/fragile/critical` 五项拆分。
- 每个 `sample_id` 首次分歧的 actor logits、早停动作概率、动作熵、Jensen-Shannon divergence、dead-corner 相关重复动作/振荡指标；统一每一步动作名映射。
- **区分两种结论级别：**实际 OFF/ON 在线同位置 `step_index` 比对，动作不同导致后续观察不同，不能自动解释为 reset 的直接因果作用；真实相同输入/状态的离线配对测试才能支撑直接的 logits 差异归因。
- `episodes_completed != 200` 或日志/manifest 对齐不全应标为 `INCOMPLETE`，不能伪造成功率或将 n=0 的 Safety Cost 写成 0。

## 5. 交付物与 STOP

- 开发分支代码和变更 diff、静态/CPU 单测命令与结果、`LOG_SCHEMA.md`、`LOGGER_INVARIANTS.md`、`VALIDATION_SUMMARY.md`、日志读写器/检查器/分析器，以及未来 OFF200/ON200 两条**拟定但未执行**的 Shell 命令。
- 如果尚未授权 GPU 等价性测试，报告必须清楚写 `GPU_EQUIVALENCE=NOT_RUN_AWAITING_AUTHORIZATION`，不能把 CPU PASS 当成正式 Logging 的无侵入证明。
- 先向 PI 交付 branch、commit SHA、文件 hash、预期数据量、实际用量和所有未验证项。**不要修改 `research-loop` 的授权或执行 Claim；不要启动 200+200、10-episode 或其他 GPU 任务。CODEX 在实现交付后 STOP。**

> 本文件是 PI 采纳下一步建议后的可执行开发任务书，**不是**“Codex 已开始开发”“Logger 已通过”或“P2 已获授权”的声明。
