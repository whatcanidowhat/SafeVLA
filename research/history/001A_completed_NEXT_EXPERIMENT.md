# Next Experiment

Experiment ID: EXP-B0-REPRO-001A
Name: Provenance Smoke
Status: COMPLETED / PROVENANCE PASS — STOPPED, AWAITING_HUMAN_REVIEW
Authorization: 用户仅授权001A；若B0身份无法冻结或协议不明确立即停止。001B和完整200-task均禁止执行。结束后停止等待人工评审，不commit/push。
Historical Reference: 86.5% full-200仅为historical reference；不作为任何Reference Run。

Research Question: 两个最小smoke能否确认同一可追溯B0 identity，并按稳定task ID配对，且产物完整？
Hypothesis: 在代码/资源/协议可冻结且不改变官方行为的前提下，两次独立运行的identity一致，两个task均可一一配对，全部原始产物可查。
Competing Explanation: 工作区行为差异、资源/import漂移、动态episode ID或不完整记录导致不能确认identity/配对。

Reference Run: 本轮新启动的reference-smoke，最多2个episode。
Repeat Run: 同一冻结身份的新进程repeat-smoke，最多2个episode。
Unique Variable: 运行实例；无策略干预。两次共最多4个episode，不自动重试/扩样。2个task是验证顺序和跨episode ID记录的最小设计；1个task不足以检验顺序。
Fixed Conditions: seed=123；num_workers=1；stochastic=true，greedy=false；test_augmentation=true；shuffle=true；minival同一数据哈希；使用既有seed shuffle后前2个task；保持官方checkpoint、DINO权重、传感器、success/end/horizon、counter/cache/reset、forward/RNG语义。单worker/2-task为明确smoke协议，不能外推4-worker Full A/A。实际命令不得依赖eval.sh可能变化的默认值。
B0 identity: 使用已审计独立candidate，reference=2aa82559d272b5f888e53433e258914057f15bed，candidate patch SHA256=f8ff5b1c07a07d5ff717f5f262734dc3d7345ca0d88e5cd5c1607a245e6927db。本地DINO由本次人工决定接受为基础设施适配，不再要求在线loader等价。
Protocol preflight: 确认eval_set_size=2确实限制任务队列；以源数据sample_id/task-spec哈希等稳定键配对，保留动态episode ID、选择顺序和实际顺序，不用运行时生成ID代替稳定task ID。

Metrics: identity一致性、task ID唯一性/配对覆盖率(2/2)、顺序记录、2/2 episode产物完整率、日志/退出码/manifest字段完整性；不检验SR/Safety Cost显著性，不要求轨迹或成功标志相同。
Expected Result: 两run冻结的行为身份与资源一致；task一一配对；每episode有结果、原始日志和来源。
Falsifying Result: identity不一致、缺失/重复task、无法追溯runtime或产物不完整；不能因结果不同改B0。

Stop Conditions:
- 无法冻结官方B0身份、未分类行为修改或协议不明确：前置停止，不跑smoke。
- 任一run出现身份漂移、任务数超2、episode配对失败、缺失权重/import信息、运行错误/产物不完整：停止，不扩样。
- 每run最多60分钟，两run运行预算最多120分钟；GPU资源不可用则停止，不占卡/杀其他任务。
- 两run结束立即停止；无论通过与否都不执行001B，不运行200-task，不commit/push。

Required Artifacts:
每run保存git rev-parse HEAD、git status --short、git diff(含staged补充)、binary patch和SHA256、参与执行的untracked源码快照/哈希、完整command、checkpoint path+SHA256、DINO path+SHA256、runtime import/module paths、Python/依赖、seed、num_workers、greedy/stochastic、test augmentation、shuffle、dataset身份、episode manifest、稳定task ID/动态episode ID及选择/实际顺序、原始日志、per-episode results、退出码。
若前置阻断，明确run未启动，不能伪造runtime module paths、episode结果或宣称两run通过。
完成后更新CURRENT_STATE/EXPERIMENT_REGISTER/DECISION_LOG，并生成RESULT_SUMMARY回答identity、配对、缺口、001B准备条件及阻塞点。001B即便技术条件满足也需人工另行授权。


Execution update:
- 两个新run目录位于research/runs/EXP-B0-REPRO-001A/execution_20260912/，角色Reference Run / Repeat Run。
- physical GPU=1（运行前确认空闲），CUDA_VISIBLE_DEVICES=1，官方--gpu_devices=0；每run新进程。
- W&B采用offline本地存储；HF/Transformers采用已有本地缓存且offline fail-fast，不改变模型/增强/采样配置。
- 记录器位于研究产物目录，通过sys.setprofile只读取已有函数return时的参数/返回值及已加载模块路径；不额外forward、不调用controller，不改变官方源码。
- 若Reference Run发生错误、资源身份漂移或记录缺失，停止且不启动Repeat Run；不自动重试扩样。
- 运行前保存完整command/环境允许列表/identity/patch/源码snapshot及DINO snapshot。运行结束记录strict load、实际imports、选中task顺序和per-episode结果。


Completion update (2026-09-12):
- Reference Run / Repeat Run各2 started/2 completed，exit 0；总计4次episode执行，无重试/扩样。
- 同一获准B0 identity与runtime路径、DINO strict load、2/2稳定sample_id配对和产物完整性均PASS；见execution_20260912/validation.json。
- 原始轨迹长度不同，不视为provenance失败，不修改baseline，不作性能结论。
- 本设计运行前版本已冻结为execution_20260912/approved_design.md；当前页面标注已完成状态。
- 本轮已经停止。不存在READY的下一实验。唯一下一动作是人工评审RESULT_SUMMARY和EXP-B0-REPRO-001B_DESIGN.md；001B仍NOT_AUTHORIZED / PROTOCOL_NOT_FROZEN，不启动200-task。
- 001B需独立冻结worker数、完整task manifest/调度、两次新run command、资源预算及容差；DINO在线/RNG/bitwise等价不是前置项。
