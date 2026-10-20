# 001D P1 最小实施方案（建议稿，未批准/未实施）

依据 PR head `a6c31432109c02af9abc42522b48ddb7c194099d`。本文件不是可执行授权，不替换 001C。实现前由 PI 处理当前未 claim 的 001C 状态，再给 P1 新周期、预算、产物与明确批准；P2 另行审批。

## 最小五个组件

1. **隔离 launcher**：从已核验的 accepted B0 建独立执行 snapshot；记录 112 source hashes、DINO适配、新增 runner/hash、两份权重和所有实际 import paths。保留开发目录原修改。显式限定一个 GPU/一个 worker，每次独立新进程。使用官方参数：ObjectNavType/minival、shuffle=true、seed=123、greedy=false、test augmentation=true、horizon=600、原输入传感器与 action map。确认 `ACTION_DICT` 等覆盖开关未意外改变协议；不依赖另一 CMD 的环境。
2. **manifest 导出/断言**：保留原 evaluator 读取、shuffle、normalize、needs_video 和入队路径，在入队前复制为可序列化记录。每次启动比较预冻结 200 行的 task spec/order digest；消费时验证序号与 sample_id。P1 取完整冻结序列前五项，保留一致前置生成语义，不为便捷重新抽五题或改 no_shuffle。
3. **episode-boundary reset 开关**：沿用 001A reviewed package。ON 在原 `InferenceAgentVIDA.reset` 的同一入口只设 root counter=0、逐层 root K/V zero；随后原 reset 恰好执行一次。OFF 完整原行为。构建期 reset 和任务期 reset 分开记；不额外调用 reset，不清 critic/augmentation/rollout之外状态。若最终实现改成原 reset 后 hook，须明确登记该差异并离线验证，不能声称逐字沿用原 patch。
4. **同一次 forward 的被动 logger**：root `actor.linear` 输出捕获 raw logits；组合 forward 后读取 final distribution 的 normalized logits/probs；保存原 sample/mode/实际 executed action，绝不重 sample/forward。只拷贝 detach 后的小 tensor，不返回替代 hook output。Actor-input digest 覆盖实际 tensors（含 dtype/shape、episode-local timestep、mask、previous-action），不再调用 encoder/preprocessor。counter前后、decoder start_pos/mask元数据、实例ID、augmentation counters、RNG摘要都来自已有状态。环境 pose/collision/visibility 仅复用已有 observation/event/task fields；缺失则标 unavailable，不增加 controller query。
5. **独立 CPU 汇总/验证器**：核对 starts/completed/failures、日志 coverage、一次 agent调用对应一次 root/reward/cost分支原 forward、原 sample 次数、原 Cost 五分项与原 worker metrics 一致、success bool与阈值一致、任务无重复/丢失。把 task.cumulative_cost 与 evaluator Cost 作为不同字段，不替换主指标。每 episode 落盘，完成后生成 hash/bytes index。

## P1 顺序与 Gates

- 实现后先做无模型静态检查：参数/路径白名单、hook仅root、manifest生成逻辑、异常即停；保存完整 diff 供审核。
- 按 PI 批准的初始化种子规则，两个独立 OFF session 各跑固定前五任务，合计最多10 live episodes；连续性在每个5任务session内保留。不跑 ON live，不添加 smoke/补跑。
- 同时记录第一决策前 RNG 与环境身份、实际 preprocessor参数、权重 missing/unexpected keys、decoder max_steps/shape/dtype/device、每类 reset/forward 次数。CLI seed 不足以代替实际初态核验。
- 利用预算内 capture 的固定 Actor inputs，在独立模型/状态副本上做 **logger OFF vs ON** 的同输入/同RNG重放；比较 logits/probs、实际采样结果、RNG状态、cache/counter和forward次数，覆盖episode边界及rollover。只在离线副本重放，绝不向 live actor 加 forward。
- 另做 **reset OFF vs ON** 的离线边界不变量检查：ON root counter/KV清零；OFF未变；ON 的 reward/cost critic counter、逐层K/V及其他非目标状态逐值不变。K/V不是注册 buffer，state_dict 相等不足以证明。可用同一有界600步输入和状态快照；不得增加live预算。
- 预算建议将上述离线检查的 forward次数单独计数、封顶。建议最多四条600-step组合重放（2,400次组合调用；每次内部原有root/reward/cost各一次），若采用root-only replay，分别记录且不混称。
- A/A 不以“两个随机轨迹没完全一致”直接判reset失败，也不能以“允许随机”跳过差异分析。同输入/同状态/同RNG的logger验证应严格相等；独立online差异须保留首次输入/动作/pose分歧和来源。若差异足以淹没一次A/B解释，Gate FAIL，不启动P2。
- PI 审查任务一致性、加载身份、无侵入检查、ON边界不变量、coverage和资源统计后，才可能批准P2。

## 失败与恢复

任何身份漂移、额外forward/query/sample、critic误改、重复/跳过任务、初始化失败、日志写失败、预算超限：停止并保留partial/error/exit code。episode从初始化尝试开始保守记账，不能以“没有成功产出结果”规避预算。P2连续session不自动从第N题续跑；重建所有cache/RNG/增强/环境状态未获批准时不能声称恢复等价。

## 拟申请资源

| 阶段 | 并发GPU | live episodes | live steps上限 | 其他限制 |
|---|---:|---:|---:|---|
| 当前P0 | 0 | 0 | 0 | 已结束，服务器只读 |
| P1（待批准） | 1 | 10 | 6,000 | 离线组合forward≤2,400；建议8 GPU小时硬上限、raw≤50 GiB，触顶停止 |
| P2（再次独立批准） | 1 | 400 | 240,000 | OFF→ON串行新进程；GPU-hour/disk上限根据P1实测速率再申请 |

P1 cap 是停止预算，不是吞吐预测或批准。每步仅20维 raw logits+20维probs 的 float32 基础量，P2 最坏约38.4 MB；实际 metadata、视频、观测capture与模型快照可能大得多，不可用这一数字承诺总磁盘量。完整encoder输入仅限离线验证所需capture，不能默认记录240,000步高维全量输入。

P1需交付实现diff、manifest及生成器身份、A/A报告、logger兼容报告、Actor-only reset不变量报告、资源/coverage统计和artifact索引。所有阶段保持未执行，等待PI。
