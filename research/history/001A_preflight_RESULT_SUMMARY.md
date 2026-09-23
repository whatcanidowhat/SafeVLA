# EXP-B0-REPRO-001A — RESULT_SUMMARY

Status: BLOCKED_PREFLIGHT
Reference Run: NOT_STARTED
Repeat Run: NOT_STARTED
实际运行数: 0；实际episode数: 0。
结论: 前置核查已完成并触发用户规定的停止条件；两次provenance smoke没有执行，不能标记001A smoke通过。
本轮未修改策略/环境/评测源码，未加载模型或启动模拟器，未commit/push，未执行001B或完整200-task。

## 1. 两个smoke run是否确认使用同一B0 identity？

未确认，因为两个run均未启动。已冻结当前候选working tree的可审计快照，但“能保存代码身份”不等于“已确认官方B0行为等价”。
HEAD为60bc54fbdedaf5745d0476c25321e808708273aa；tracked diff SHA256为f69354e479fdc1f7ad78ab0f30f5ee57e5b4b50a3cbfff2c53e5f69792d7599b。
本轮重新读取文件计算checkpoint和DINO SHA256，分别为：
- safe_objnav.pt: 05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301
- dinov2_vits14_pretrain.pth: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9

这些哈希与历史审计记录一致，但不表示本轮已实际加载权重。静态源码路径也不冒充runtime import/module paths。

## 2. 是否能稳定按episode ID配对？

未验证。没有本轮动态episode ID、实际顺序或per-episode results。
设计规定两run各2个task、seed=123、num_workers=1，保留shuffle/test augmentation、stochastic模式，按同一源task稳定ID/规格哈希配对，另外保存动态episode ID。
历史动态episode ID带运行生成部分，不能直接作为跨run配对主键。实际ID生成/规范化后的对应关系仍需在获准恢复001A时验证。

## 3. 是否存在provenance缺口？

存在：
- 官方参考worker与当前worker的无条件预决策环境查询，尚无行为等价证据。
- 无实际run，故没有runtime import/module paths、实际command/启动环境、episode manifest、原始运行日志或per-episode结果；这些明确标记未产生，未伪造。
- 候选命令保存于preflight/candidate_commands_NOT_EXECUTED.json；它们不是已执行命令，启动环境与runtime记录包装未定稿。
- 本地DINO源码无.git，已保存源码哈希及快照，可追溯当前资源；与官方加载来源的对应关系仍需核实。

已保存：HEAD、status、tracked/staged patch、相对官方参考的patch、100份候选源码哈希及快照、DINO源码快照、模型文件路径/大小/SHA256、已注册设计和行为审计。

## 4. 是否满足进入EXP-B0-REPRO-001B的条件？

不满足。001A未完成两次smoke；B0官方行为等价、实际import来源、稳定task配对和完整产物链均未全部验证。
即使将来001A通过，也不自动授权001B；单worker两task不证明4-worker Full A/A协议、SR/cost稳定性或统计充分性。
001B本轮NOT AUTHORIZED；历史86.5% full-200始终只作historical reference，不是Reference Run。

## 5. 具体阻塞点是什么？

当前online_evaluation/online_evaluator_worker.py每步在策略调用前新增successful_if_done(strict_success=False)及dist_to_target_func()；这组调用不受END_CAUSAL_DIAGNOSTICS/SHADOW_PREDICTOR_ENABLE开关保护。
可见性调用在缓存未命中时进入底层controller.step('GetVisibleObjects')并更新缓存。尚未证明实际调用顺序下始终命中缓存，或额外环境事件对last_event/后续观测和环境语义无影响。
这不是“已证明SR/cost被改变”；这是用户要求的“官方行为不变”尚未建立，不能仅靠两份相同修改版A/A来补足。

另有greedy action-history改动确实改变greedy模式语义，但本设计greedy=false，该分支差异按静态分析不激活；它不是本stochastic smoke的单独停止理由。
原counter/reset问题保留，未静默修复；本次不是要求先解决H-RESET，而是执行用户明确的B0身份前置停止规则。

## 人工评审对象与下一动作

只评审恢复001A所需的候选B0身份：确定官方reference revision及允许适配集合，对额外环境查询建立无侵入性证据，或另行批准准备与官方调用路径一致的候选执行版本。不得为继续smoke而静默修改当前源码。
本轮到此停止，未注册或执行新的实验。

## Evidence

- preflight/registered_design.md：执行前设计
- preflight/BEHAVIOR_AUDIT.md：具体调用链、官方差异与结论边界
- preflight/preflight_manifest.json：机器可读前置状态及资源身份
- preflight/git_head.txt、git_status.preflight.txt、git_diff.patch、git_staged.patch
- preflight/official_reference_to_worktree.patch、official_reference.txt
- preflight/source_sha256.json、runtime_source_snapshot.tar.gz
- preflight/dino_source_sha256.json、dino_source_snapshot.tar.gz
- preflight/candidate_commands_NOT_EXECUTED.json：未执行候选命令
