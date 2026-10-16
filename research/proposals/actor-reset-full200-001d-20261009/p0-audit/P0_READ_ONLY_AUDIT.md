# PR #1 · EXP-ACTOR-RESET-FULL200-001D P0 只读审计

审计日期：2026-10-09（Asia/Shanghai）。结论：**P0 静态身份与可实施性审计完成；001D 仍为 DRAFT / NOT_AUTHORIZED，P1/P2 未执行。建议先补齐下列实施门槛，由 PI 独立审批 P1。**

本次未 claim 001C、未修改服务器文件/分支/控制状态、未评论或推送 GitHub；未导入模型、加载 checkpoint、启动 AI2-THOR、查询/占用 GPU 或运行 Episode。checkpoint/DINO SHA 核验仅流式读取文件字节。四份交付仅保存于本地。

## 1. PR 与 CI

- PR：[whatcanidowhat/SafeVLA #1](https://github.com/whatcanidowhat/SafeVLA/pull/1)，OPEN，GitHub `is_draft=true`；结束前复核 head/base 均未推进。
- Head：`a6c31432109c02af9abc42522b48ddb7c194099d`；base `research-loop`：`0a152f88e8eaf78b32ea6b09124782db25e2c436`。
- 变更仅 `research/proposals/EXP-ACTOR-RESET-FULL200-001D_DRAFT_2026-10-09.md`，新增 193 行，无 runtime/reset 实现。
- [CI 37939431668](https://github.com/whatcanidowhat/SafeVLA/actions/runs/37939431668)：`pull_request`，head 精确匹配；`control-protocol` SUCCESS，2026-10-09 21:50 左右完成。
- 实际检查了 synthetic control fixtures 与完整 control history/artifact validation；两步均成功。CI 未加载 SafeVLA，也未证明 B0 等价、日志无侵入、随机可配对、reset 效果或实验授权。

## 2. 001C 的授权与服务器状态

GitHub 最新 `research-loop` 仍为上述 base，LOOP_STATE v47：

| 字段 | 核验值 |
|---|---|
| experiment_id | EXP-COUNTER-DRIFT-EARLYEND-001C |
| status / next_actor | APPROVED_FOR_CODEX / CODEX |
| instruction_commit / claim_id | null / null |
| authorization | APPROVED，PI，max_gpu=1，max_episodes=32 |
| 授权窗口（北京时间） | 2026-10-09 11:41 至 2026-10-10 11:41 |

服务器 `/nvme2/user/qyy/SafeVLA_loop_control` 仍停在 `8acdb18f12a83d3e9ca3711313337f4ca232ae21`，worktree clean，保存的是 **001B BLOCKED v44**。本次未 fetch/fast-forward，不能把本地旧状态当成 GitHub 最新状态。

检查共享仓库四个登记 worktree、local branch refs、已知 development `research/handoffs`/`research/runs` 下 001C 目录、以及匹配 claim/001C/evaluator/THOR 的进程：未发现 001C claim、产物或运行迹象。此结论限于已检查路径和进程快照，不证明其他机器/目录绝无活动。

用户本次明确禁止 claim，优先于旧 001C 执行授权。001D PR 也不撤销或替代 001C。若 PI 转向 001D，现行 `scripts/validate_research_loop.py:478–492` 已支持 **未 claim** 的 `APPROVED_FOR_CODEX → PI_REVIEW`：清除批准元数据、改 NOT_AUTHORIZED、保留 reviewed_result_commit 和原 execution_worktree/required_outputs，增加 state_version，NEXT 退回 DRAFT；再以新周期注册独立 P1。必须再次核对无并发 claim。本审计不执行该转换，也不建议先 claim 再人为中止 001C。

## 3. Baseline 身份

以下均为服务器只读重新核验，而不是仅引用旧报告：

| 对象 | 结果 |
|---|---|
| accepted B0 | `/nvme2/user/qyy/SafeVLA_baseline_clean`，HEAD `2aa82559d272b5f888e53433e258914057f15bed`，branch `codex/b0-candidate-audit` |
| B0 tracked diff | 仅 `architecture/allenact_preprocessors/dino_preprocessors.py`；SHA256 `c4a8194d337c37a7a8765a01a11175778301b284211b3b32cd87de55088d3816`；staged diff 空 |
| B0 untracked source | `scripts/b0_candidate_env.sh`；不是 HEAD 所覆盖的执行身份，未来使用须另行保存/hash |
| 与 001A 比较 | B0 的 112 个 tracked 文件均与 001A `official_runtime` 字节相同，也全部匹配已归档 `RUN_MANIFEST.source_frozen` |
| development | `/nvme2/user/qyy/SafeVLA`，HEAD `60bc54fbdedaf5745d0476c25321e808708273aa`，4 个 tracked 修改；diff SHA256 `f69354e479fdc1f7ad78ab0f30f5ee57e5b4b50a3cbfff2c53e5f69792d7599b`；不是干净 B0 |
| policy checkpoint | `/home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt`，2,027,033,991 bytes，SHA256 `05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301`，匹配 001A |
| DINO checkpoint | `/home/amax/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`，88,283,115 bytes，SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`，匹配 001A |
| DINO source | `/home/amax/.cache/torch/hub/facebookresearch_dinov2_main`；001A 索引内 157 个文件全部匹配；不代表索引外文件已全面审计 |
| 实际审计 Python | `/home/amax/.conda/envs/safevla/bin/python`；通过包元数据读取依赖，未 import 模型 |
| 已安装依赖 | torch 2.4.1+cu121，numpy 2.1.2，transformers 4.46.1，ai2thor 0+966bd7758586e05d18f6181f459c0e90ba318bec，allenact 0.5.5a0，jsonschema 4.25.1，pyarrow 22.0.0 |

DINO 本地适配与合约批准范围一致：`source="local"`、`pretrained=False`、本地路径校验、`load_state_dict(strict=True)`；P1 必须同时指定两项 DINO 环境变量并禁止静默网络/随机初始化回退。policy loader 本身为 `strict=False`（`inference_agent.py:162–165`），因此必须在加载时验证 missing/unexpected 均为空，不能把“没有报错”当作完整加载。001A 历史记录为 policy 417 keys、DINO 175 keys；本次没有重新加载验证。

001A 的真实 decoder import 为 `training/online/third_party_models/llama/model.py`。未来 001D 的 import paths、GPU/dtype/cache layout、strict-load 事件仍须 P1 实证。本次没有生成或证明新的运行时身份。

## 4. 八项审计问题的答案与代码路径

本节行号取自当前已核对 B0 字节。源码根为 `/nvme2/user/qyy/SafeVLA_baseline_clean`；未改动文件可对照 [官方 commit](https://github.com/PKU-Alignment/SafeVLA/tree/2aa82559d272b5f888e53433e258914057f15bed)。DINO 适配行号以服务器副本为准。

| 问题 | P0 结论与证据 |
|---|---|
| 1. B0 与 001A 是否一致？ | 是，范围为上述 112 tracked 文件、已索引 DINO 源码及两份权重字节。001A 的实验选择/teacher forcing 不是 001D 的线上协议；不能复制旧 harness 后声称协议相同。 |
| 2. 001C 是否 claim/运行？ | GitHub null claim；已知服务器范围无 claim/运行迹象。本地 control 为旧 001B，详见第 2 节。 |
| 3. 一个 Worker 是否持续同一 Actor？ | 是，静态路径明确：`online_evaluator.py:553–566` 单 worker 同步调用 `start_worker`；`online_evaluator_worker.py:65–75` 只构造一次 Agent，再进入 `distribute_evaluate:583–632`。每任务 `evaluate_on_task:274` 调 `agent.reset()`。构建阶段另有 `inference_agent.py:169` 初始化 reset，应单独计数。 |
| 4. 200 个任务与顺序是否可冻结？ | 数据存在且恰好 200 行；`online_evaluator.py:258,282–283` 从 cwd 的 `./benchmark` 读 minival；`:352–367` 对原行号先 seed(123)、shuffle、再取前缀；`:541–543` 按已选 samples 顺序入队。`TaskSpecQueue.next_task_spec` (`tasks/task_specs.py:238–244`) 顺序消费。当前仅核验数据和生成路径，未生成实际 001D manifest。 |
| 5. reset 是否只影响 Actor？ | 001A patch 只写 `self.actor_critic.time_step_counter` 与 `root.decoder.layers[*].attention.cache_k/cache_v`。`separate_actor_critic.py:8–37` 显示 `critic_tsfm`/`c_critic_tsfm` 是独立实例；返回的 action distribution 来自 root Actor。禁止 `.modules()` 递归清理。P1 还须对实际实现检查 critic counter/cache tensor，不可只比较 state_dict。 |
| 6. 采样/动作历史如何保留？ | `inference_agent.py:269–296` 每 decision 一次组合 forward，然后原有 `sample()`、`mode()`，`last_action_flat` 始终取 sample；greedy 分支只改变返回动作。必须保留调用次数/顺序、原 sample、原 mode、原 last_action；记录不得再 sample。 |
| 7. logits 与 Cost 如何捕获？ | raw Actor logits 是 root `actor.linear` 的输出（AllenAct `algorithms/onpolicy_sync/policy.py:174–186`）；最终分布在 `allenact_dino_transformer.py:470–475`，不是 `extras` 中 critic logits。Cost 直接读取 worker 已算好的 `metrics["cost"]`/五分项，详见第 5 节。被动读取可实现，但无侵入性尚未运行验证。 |
| 8. P1 最小实现是什么？ | 隔离 B0 snapshot + manifest 断言器 + root reset 开关 + 被动 logger + 独立验证器；两次 OFF 前五任务 A/A，另加预算内离线 logger 与 ON reset 检验。见 `DRAFT_IMPLEMENTATION_PLAN.md`。 |

任务数据：`benchmark/objectnavtype_val.jsonl.gz` SHA256 `fa0dffc2dece071716dcf648e19c6944caf1621ee24dadc6996eadcd46eee04a`；解压 bytes SHA256 `f10ce169dc71e0ee2b4f9bb803a464babf8c72e28ad7d7d2a28b0d80a174863e`，与历史 aligned task specs 相同。文件数据身份已核验，未据此声称 house/assets 全量哈希一致。

## 5. 风险清单与 P1 前必须固定的细节

优先级针对实施前的风险，不表示这份纯文档 PR 已引入运行时 bug。

| 优先级 | 风险与触发位置 | 最小处理 |
|---|---|---|
| 高 | 001C 仍占当前批准状态；草案不能自行成为唯一实验 | PI 按未 claim 撤回规则单独处理，再批准独立 P1；本次不改状态 |
| 高 | 只做 OFF/OFF A/A 不验证 ON 是否误清 critic；001A patch 在原 reset **之前**，现有001C设计则描述原 reset 后清理，不能无说明混用 | 为001D固定调用顺序与初始化 reset 分类；P1 加离线 ON/OFF边界状态不变量检查，禁止增加 live episode |
| 高 | raw logits 与 `Categorical.logits`（归一化 log-probs）混用，或抓到 critic 输出；草案第 7 节未给出字段语义 | 分别记录 `actor_linear_logits_raw`、`distribution_logits_normalized`、probs；hook 只绑定 root `actor.linear`；runtime 校验 action count/order/end index |
| 高 | Cost 时点与 success 浮点语义可悄悄改指标：worker 在 step 前累计上一步五分项 (`301–310`)，step 后 done 即退出 (`363–364`)；`calculate_metrics:505` success 加 `1e-8` | primary Cost 用原 worker `metrics.cost`，不改为 `task.cumulative_cost`；保留 pre-step 原分项。末个 horizon 动作 Cost 可能未进 evaluator 总和，应披露并保持两臂同口径，不能顺手修复。success 复用官方 bool/`>0.1`，不能 `bool(metrics.success)` |
| 高 | 相同 seed 不等于未来每任务 CRN：增强自身有跨步计数器；episode 长度变后后续 RNG/增强状态分流 | `dino_preprocessors.py:217–245` 以 500 步更新增强，来源 `dinov2_vits_tsfm_base.py:123–124,151–152`；只记录、不清零。估计对象维持连续 session 效应，不新增逐任务 reseed 或伪称全程同 RNG |
| 中 | `--test_augmentation` 字段本身不足以证明实际增强路径；源码实际依赖 `params.use_data_augmentation` 与 sensor graph | P1 保存真实 preprocessor paths、enabled、num_steps_to_change、计数器和 transform 身份。禁止只保存 CLI flag |
| 中 | 模型构建发生在 lazy sampler `set_seed(123)` 前；仅传 CLI seed 不能证明加载期 RNG 相同 | 保存进程起点、model build、sampler seed、首决策 RNG 摘要；如需进程入口统一 seed，写入 PI 批准的两臂 launcher 协议，不声称原 CLI 已保证；不改采样算法 |
| 中 | manifest adapter 绕过原 shuffle 或改变 RNG 消耗；视频标记还有第二次 shuffle (`online_evaluator.py:535–539`) | 保留原生成与入队路径，比较并导出 manifest；第二次 shuffle 标记 needs_video，不重排任务，但不得无记录删除其 RNG 消耗 |
| 中 | 单 worker 先完成全部任务才收集 results_queue (`evaluate:557–588`)；仅依赖最终 W&B 表容易丢失 partial 进度 | 每 episode 被动落盘独立 journal；记录原队列结果，不替换 evaluator 调度；不增加 worker |
| 中 | 重启“继续第 N 题”会丢 OFF cache、critic、增强和 RNG 连续性 | P2 不自动恢复。异常停止并交付 partial；新增 session/重试需 PI 批准，不能隐藏重跑 |
| 中 | sampler timeout 路径可能重新建 controller/递归 next_task (`multi_task_eval_sampler.py:206–215`) | 对消费、初始化尝试、开始/完成分别记账；任务跳过或环境重建触发预注册停止条件，不替换 task 凑 200 |
| 中 | `SR/200` 在缺失 episode 时会把未知误当失败或漏掉分母 | 只有 200 完整配对才给主终点；partial 报完整数、缺失 ID 和描述性数据，主终点置 unavailable |
| 中 | “Cost≈ / 尾部未恶化 / 有价值改善”未量化 | P1 后、P2 前由 PI 冻结探索阈值、分位数算法；单 session 不做 200 独立样本显著性，不宣称非劣效 |

ON 仍可能在自身第 500 步 rollover。模型 max_steps 静态来源 `training/online/base.py:129` → `dinov2_vits_tsfm_base.py:248`；task horizon 为 600（`max_episode_configs.py:10`），两者不可混淆。

成功定义：`AbstractSPOCTask.is_successful:177–178` 要求 `successful_if_done()` 且执行过 end；ObjectNav `successful_if_done:119–134` 采用 broad target、导航相机可见、距离阈值 2，不能换成额外中心视野/自定义 proximity 判定。Logger 不额外调用这些可能访问 controller 的函数，复用原计算结果。

## 6. 可解释范围、门槛与预算建议

001D 科学方向与草案一致：**新单 worker 连续序列协议**下的 session-level reset 效应。不能与默认 8-worker 或历史 4-worker 173/200 直接当成同协议提升；不能把 200 tasks 当独立随机化单元。后续不同长度导致环境/RNG/增强分流是该连续协议的后果，应披露，不能归为同输入的直接 rollover 效应。

建议仅独立申请 P1：最多 **1 GPU 同时占用，10 live episodes（5+5），最多 6,000 live decision steps**；离线 logger/reset 检验另列有界 forward 预算（建议最多四条 600-step replay，即 2,400 次组合调用，或明确等价的 root-only预算），不新增 simulator episode。CPU 文档/manifest处理不占 GPU。可申请 8 GPU 小时、50 GiB raw 产物上限作为保守停止上限，非完成时间预测；触顶保留 partial 并停止。

P2 保持未批准：1 GPU、两臂串行、200+200=400 live episodes，最大 240,000 live steps；P1+P2 总 episode 上限 410。不要根据 001A 混合了离线 replay 的总耗时推算 full200。P1 应实测加载耗时、每步耗时、每任务初始化耗时、产物 bytes，再向 PI 提交独立 P2 GPU-hour/disk 上限；P2 不因 P1 通过自动启动。P3 额外种子不在上述预算内。

P1 仍须实证：完整权重加载、真实 import/cache layout、Agent/Actor 实例生命周期、初始环境/seed、logger无侵入、ON critic不变、200 manifest 生成一致和前五任务实际消费。P0 不能替代这些运行验证。

**审计结束，STOP；等待 PI 对 P1/P2 的独立批准。**
