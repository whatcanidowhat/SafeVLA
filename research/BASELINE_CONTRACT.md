# Baseline Contract

## 研究边界

以官方 SafeVLA 及其原论文评测协议为 B0，诊断真实失败并提出改进。不能仅凭分支名或一次分数认定官方 Baseline。

## B0 不变量

Baseline 和只读诊断阶段保持：
- 官方 success definition、end 执行逻辑、horizon、指标分母和聚合方式。
- checkpoint、加载规则、输入传感器、动作集合及映射。
- 采样/greedy、test augmentation、seed、worker 数、episode 集合/顺序/分配、数据版本、核心评测协议。
- policy forward 次数、RNG、cache/counter/history 更新及环境交互语义。

小样本、固定前缀、greedy smoke 必须标明诊断协议，不与完整官方 B0 混称。worker 和 episode 调度需记录，因为历史状态可能跨 episode 保留。

## 允许的诊断与隔离干预

允许路径适配、runtime verification、日志、activation hook、hidden-state collection、Probe 和离线分析，前提是不改变策略输出、内部状态推进、RNG、环境行为及评价语义。资源身份或处理结果发生变化的适配不能直接视为只读。

Actor、reward critic、cost critic 特征须标明实际来源。counterfactual 不得在 live actor 上额外 forward；使用独立模型/cache 或离线重放。Probe 按 episode/house 划分训练与测试，防止逐帧泄漏；信息可解码不等于策略使用了它。

任何行为修改，包括疑似 bug 修复、counter/cache reset、oracle mask 或更换 checkpoint，均先注册为独立 treatment，保留 B0，明确唯一变量与固定条件。修改评价规则属于另一协议分析，不能称为官方指标提升。发现混杂时保留无效结果并重新设计，不顺手加入第二个变量。

## 当前身份与限制

2026-09-11 只读核对：
- repo: /nvme2/user/qyy/SafeVLA
- branch: baseline-experiment
- HEAD: 60bc54fbdedaf5745d0476c25321e808708273aa
- tracked binary diff SHA256: f69354e479fdc1f7ad78ab0f30f5ee57e5b4b50a3cbfff2c53e5f69792d7599b

该 diff 哈希与历史 diagnostics/end_causal_audit/runtime_manifest.json 一致，但不覆盖 untracked 源文件。

历史 manifest 报告 checkpoint /home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt，SHA256 05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301，417/417 tensors 加载完整。本次未重新哈希权重或加载模型，后续运行前重新核实。

历史 Gate B = FAIL。不得静默 reset 后称为 B0，或以未隔离的 hidden-state 结果宣称机制成立。

## 每个正式 run 的证据要求

使用独立目录，建议 research/runs/<experiment-id>/<run-id>/，保留：
- run_manifest.json：experiment/run ID、状态、UTC 时间、host/cwd、完整 command、实际 Python、依赖版本、相关环境变量、HEAD/branch/status、tracked patch 和参与执行的 untracked 源码快照/哈希、checkpoint 与外部模型资源哈希、实际 import paths。
- 精确配置、seed、worker 数、GPU、采样和增强开关、数据版本、episode IDs/顺序/worker 分配。离线运行保存重放输入及来源哈希；无环境 episode 时注明不适用及理由。
- 原始 results、logs、退出状态、计划/完成/失败数量；保留失败尝试、部分运行及所有预注册 seed。
- RESULT_SUMMARY.md：预期和证伪结果对照、实际结果、竞争解释、限制、证据路径、停止原因和下一决策。

只保存允许列表内的相关环境变量，不转储 token、认证 remote URL 或完整环境。GitHub 共享状态/代码；被 .gitignore 排除的 eval/ 等产物须另有可访问位置与哈希，不能假设 push 会同步。

## 2026-09-12 人工批准：executable B0的DINO基础设施适配

因服务器网络限制，人工批准candidate使用本地DINOV2_REPO源码和DINOV2_CKPT权重，作为executable B0必要infrastructure adaptation，不作为treatment或研究变量。
保留source="local"、pretrained=False、显式checkpoint、load_state_dict(..., strict=True)及文件缺失fail-fast。
不执行EXP-B0-DINO-EQ-001，不要求与在线torch.hub loader比较RNG/bitwise equivalence；此决定不声称两加载路径已实测等价。
每个正式run仍须保存DINOV2_REPO、DINO源码snapshot/hash、DINOV2_CKPT及SHA256、实际runtime module paths、strict load成功状态。失败或资源漂移必须停止，不回退为随机初始化。
其他Baseline规则不变；不得修reset/counter或增加预决策environment/controller查询。001A仅Reference/Repeat各最多2个episode，001B/200-task仍禁止。
