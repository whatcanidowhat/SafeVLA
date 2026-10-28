# EXP-ACTOR-MEMORY-UNMASK-001 CPU preparation

已完成独立分支、root Actor mask 开关、CPU 单元测试、checkpoint Actor 组件重放和资源预检。在线/GPU 未授权且未执行；SR / Safety Cost 均未测量。控制状态、历史 Claim、历史源文件和原结果未改动。

分支 `codex/actor-memory-unmask-001-20261028`，基线 `06cde9adeb91d47c062a11e72e90c7edf56fb398`，本地代码提交 `0d0ec865de555c2d13d591d666730b45e7b1e20f`。源码修改仅 actor 文件8行新增/1行条件替换，以及5个新 CPU/配置脚本；详见 SOURCE.patch / SOURCE_HASHES.json。

默认每个实例 mask ON。root 完整构造后，`configure_root_actor(root, mask_enabled=False)` 仅切换 root。原单步 mask 在四步旧历史之后的 Episode 首步为 `[false,false,false,false,true]`；OFF 经原 decoder 默认因果 mask 变为 `[0,0,0,0,0]` additive mask，五个有效位置均可访问。多步输入仍使用下三角因果约束。没有修改 prev_actions masks、Critic、cache 分配/写入、counter/max_steps rollover，也没有加 live forward。

CPU 单元测试8/8 PASS：默认与原提交 forward/cache 逐值一致；历史扰动 ON 被隔离/OFF 生效；root 配置不改 cache 对象/值、counter、RNG/Critic；首步 prev_actions 仍 null token20；批量因果性；真实1000步 rollover；原 separate 类 Critic 输出/cache/每分支一次decoder调用不变。

checkpoint 哈希匹配，严格加载 root decoder29项、Actor head2项、prev_actions embedding1项、time encoder1项。独立三层512维 Actor，4次有效 warmup，克隆状态后6次边界重放。固定输入是合成的感知后 embedding，不是完整 DINO/T5 观测或线上轨迹。

| 对照 | max absolute logit difference | JS(nats) |
|---|---:|---:|
| Mask ON / OFF，同一输入与cache |9.711416|0.463700|
| ON 原历史 / 改历史K |0|0|
| ON 原历史 / 改历史V |0|0|
| OFF 原历史 / 改历史K |3.619738|0.219407|
| OFF 原历史 / 改历史V |5.297393|0.522715|

P(end)：ON=2.866716e-7，OFF=1.298557e-4；OFF 改K=1.187123e-3，改V=9.403175e-4。完整20维raw logits与各层实际mask见 REPLAY_RESULTS.json。正对照支持“历史K/V可访问并影响Actor logits”，不证明历史语义有益，也不证明SR或Cost改善。

首次重放因 Attention.forward 绕过模块hook，观测计数断言失败。修正离线观测点到TransformerBlock后重放PASS，保留首轮日志、输入、cache及哈希；未更改生产行为。CPU fixture只替代感知、避免T5和模拟器依赖；Critic数值不变证据来自源forward+小维fixture，不是完整checkpoint GPU等价证明。

资源：policy/DINO权重和DINO源码哈希、5项依赖、200 TaskSpecs manifest均匹配；cache实现、separate Critic、inference_agent、worker与指定提交逐字节相同。四张V10032GB均忙；磁盘剩余约34.14GiB，低于建议100GiB（90GiB产物+10GiB余量）。资源快照不构成预约或GPU准入。资产目录存在，内容版本尚未完成全量pin。

完整评测方案：Mask ON200 + OFF200，200相同有序TaskSpecs/臂，共400 Episode starts、最多240000 decisions。固定checkpoint/seed123/worker0/随机采样+mode调用/augmentation/官方指标/horizon600/max_steps1000。两个全新进程同seed，单臂内部持久cache；不交替mask，不复用旧OFF200。请求最多1张V100、执行12h、队列12h、产物90GiB，绝对授权时限另由PI决定。每臂完整性门后才做配对SR/Cost、bootstrap区间及长尾报告。详见 DESIGN.md/json。

主要混杂：合成输入分布外；错目标/跨场景历史干扰；Episode长度与rollover相位；在线随机轨迹/augmentation RNG分叉；同seed不保证同轨迹；顺序/墙钟差异；GPU数值和logger等价尚未验证。停止：任何身份/指标/任务/唯一变量不一致、CPU机制失败、资源或期限不足、数据不完整、额外forward、旧Claim、输出目录已存在、网络/CI失败即保存并停，不自行重试。

下一动作仅为PI独立审核。GPU前仍需新cycle、明确独立授权、source pin、新在线入口绑定审查、批准CI、新Claim及其CI、实时资源/网络/资产门；旧GPU waiver不继承。首轮交付没有可绕过授权的GPU启动器，在线入口当前是设计范围，不声称已经通过GPU执行就绪验证。
