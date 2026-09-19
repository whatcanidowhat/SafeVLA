# Next Experiment

Experiment ID: EXP-B0-REPRO-001A
Name: Provenance Smoke
Status: BLOCKED_PREFLIGHT — no smoke launched
Authorization: 用户仅授权001A；若B0身份无法冻结或协议不明确立即停止。001B和完整200-task均禁止执行。结束后停止等待人工评审，不commit/push。
Historical Reference: 86.5% full-200仅为historical reference；不作为任何Reference Run。

Research Question: 两个最小smoke能否确认同一可追溯B0 identity，并按稳定task ID配对，且产物完整？
Hypothesis: 在代码/资源/协议可冻结且不改变官方行为的前提下，两次独立运行的identity一致，两个task均可一一配对，全部原始产物可查。
Competing Explanation: 工作区行为差异、资源/import漂移、动态episode ID或不完整记录导致不能确认identity/配对。

Reference Run: 本轮新启动的reference-smoke，最多2个episode。
Repeat Run: 同一冻结身份的新进程repeat-smoke，最多2个episode。
Unique Variable: 运行实例；无策略干预。两次共最多4个episode，不自动重试/扩样。2个task是验证顺序和跨episode ID记录的最小设计；1个task不足以检验顺序。
Fixed Conditions: seed=123；num_workers=1；stochastic=true，greedy=false；test_augmentation=true；shuffle=true；minival同一数据哈希；使用既有seed shuffle后前2个task；保持官方checkpoint、DINO权重、传感器、success/end/horizon、counter/cache/reset、forward/RNG语义。单worker/2-task为明确smoke协议，不能外推4-worker Full A/A。实际命令不得依赖eval.sh可能变化的默认值。
B0 identity: 必须核对官方参考代码与当前working tree差异，区分路径/记录适配和行为修改；只保存哈希不能证明官方等价。未明确时禁止启动。
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

Preflight Result: 当前官方行为等价性未建立，按用户停止条件终止；详见research/runs/EXP-B0-REPRO-001A/preflight/BEHAVIOR_AUDIT.md。两次smoke均未启动；仅准备证据和最终报告。
