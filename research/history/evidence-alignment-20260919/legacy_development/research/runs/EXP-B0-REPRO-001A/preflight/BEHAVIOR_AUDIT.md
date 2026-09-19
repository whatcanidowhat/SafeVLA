# EXP-B0-REPRO-001A — Baseline behavior preflight

Result: BLOCKED_PREFLIGHT. 两个smoke均未启动；不是smoke失败，也不是完成两run。

## Reference identity
Local HEAD: 60bc54fbdedaf5745d0476c25321e808708273aa.
Local upstream/main: 2aa82559d272b5f888e53433e258914057f15bed.
共同官方祖先: 6464cbdc0ef4d2948157c43f90304e1261d4209f.
祖先到本地upstream/main只有README一行差异，相关行为源码相同；未进行网络fetch，不声称本地upstream是远端最新版本。
相对官方祖先的工作区差异涉及9个tracked文件，包含HEAD中已提交改动；不能只核对git diff HEAD。

## Findings
1. online_evaluation/online_evaluator_worker.py:371-388 在每步策略调用前无条件执行 successful_if_done(strict_success=False) 和 dist_to_target_func()。即使SHADOW_PREDICTOR_ENABLE=0、END_CAUSAL_DIAGNOSTICS=0，查询仍执行。官方参考worker没有这组预决策调用。
2. tasks/object_nav_task.py:119-129 → environment/stretch_controller.py:504-512 → get_visible_objects():430-489。可见性查询在缓存未命中时调用底层controller.step('GetVisibleObjects')，并填充缓存。当前静态核查尚未建立实际sensor顺序下每次都命中缓存、或额外查询对controller.last_event/后续观测和环境行为无影响的证据。
3. tasks/object_nav_task.py:82-108 使用L2距离和controller对象位置接口。新增调用不是仅从已记录日志读值，其传递副作用需要纳入等价性核查。
4. architecture/models/allenact_transformer_models/inference_agent.py:325-350 改变greedy时的last_action_flat：从sampled action改为实际greedy action。此项确实改变greedy行为；本设计greedy=false，按静态分支逻辑该差异不激活，不能夸大为本stochastic smoke已被证明受影响。
5. DINO改为本地源码/权重加载；可以保存文件哈希，但缓存源码没有.git。源码快照可冻结当前资源，不自动证明其与官方加载来源等价。
6. shadow开启会要求renderDepthImage=True；本计划明确关闭。eval.sh默认worker/shuffle/greedy与HEAD不同；必须由完整显式参数规避默认值歧义。
7. 未改变官方counter/reset是本实验的要求。原Gate B不自动阻断所有Baseline工作；本轮停止依据是用户本轮明确要求的B0身份/行为等价前置条件，非坚持先修reset。

## Evidence scope
已有forward/RNG/action-chain审计支持被观察smoke内logger的相应性质，但未提供上述官方参考与当前worker的环境查询/事件语义等价证明。
以上不证明新增查询已经改变SR、Safety Cost或策略输出；证明的是该行为差异路径确实存在，当前等价性未建立。
不能把两份相同的修改版代码A/A相同当作官方B0身份认证。停止后不静默修改源码，也不自行创建clean工作区改变候选B0。

## Review decision needed
先确定官方参考revision及允许的适配集合，对无条件环境查询给出无侵入性证据，或另行批准准备与官方调用路径一致的候选B0。之后再评审恢复001A；不得直接进入001B。
