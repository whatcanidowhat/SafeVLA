# Current State

Updated: 2026-09-13（Asia/Shanghai）
Mode: EXP-B0-REPRO-001A = COMPLETED / provenance PASS / awaiting PI review. 不得重复运行。001B NOT_AUTHORIZED。当前唯一待评审草案是EXP-LOOP-HANDOFF-001，LOOP_STATE=PI_REVIEW，next_actor=PI。

## Verified Facts

- 最新人工决定：本地DINO加载已纳入executable B0，是必要基础设施适配而非treatment；不执行DINO-EQ/RNG/bitwise比较。001A已验证每run两次DINO strict=True加载成功、各175 keys且无missing/unexpected keys，实际路径来自获准DINOV2_REPO。
- 001A execution_20260912：Reference Run 001A-20260912-reference和Repeat Run 001A-20260912-repeat各2 started/2 completed，退出码均0；固定B0/source/DINO/checkpoint/dataset/runtime paths一致，稳定sample_id配对2/2且顺序一致，validation.json全部PASS。
- 实验已停止；原开发目录166个受保护文件保持不变，candidate源码运行前后无变化；无commit/push。原始证据见research/runs/EXP-B0-REPRO-001A/RESULT_SUMMARY.md及execution_20260912/。

- 2026-09-12：GitHub API固定commit SHA/tree SHA与本地对象一致，已核实官方reference 2aa82559d272b5f888e53433e258914057f15bed，不仅依据共同祖先。
- 独立candidate /nvme2/user/qyy/SafeVLA_baseline_clean，branch codex/b0-candidate-audit：111/112官方tracked文件字节一致；唯一tracked适配为DINO加载入口，另新增路径env文件。原开发源码/diagnostics保持。
- 候选worker不含开发版无条件预决策查询，greedy/history与reset/counter原官方行为全部保留；DINO本地加载已由最新人工决定接受为基础设施，取代旧审计中的待在线等价验证分类。
- 完整diff和初始适配分类见research/B0_CANDIDATE_AUDIT.md（历史静态审计）；后续人工决定及实际smoke记录以本状态和DECISION_LOG为准。

- 历史001A前置尝试曾仅保存静态身份并BLOCKED_PREFLIGHT，0 episodes；旧摘要保存至execution_20260912/before_RESULT_SUMMARY.md，不与本次4次episode执行混淆。
- 旧前置核查使用共同祖先6464cbdc0ef4d2948157c43f90304e1261d4209f比较开发worker的预决策查询差异；共同祖先本身不证明official身份。当前实际candidate基于已核验官方2aa82559；不在开发worker上恢复smoke。
- 历史86.5% full-200仅作historical reference，不能充当本轮Reference Run。

- 本次 SSH 确认 repo 为 amax@192.168.134.211:/nvme2/user/qyy/SafeVLA；/nvme2/qyy/SafeVLA 不存在。
- branch=baseline-experiment，HEAD=60bc54fbdedaf5745d0476c25321e808708273aa。
- 当前 tracked diff SHA256=f69354e479fdc1f7ad78ab0f30f5ee57e5b4b50a3cbfff2c53e5f69792d7599b，与历史 manifest 一致。
- 构建前已有 4 个 tracked 修改：architecture/allenact_preprocessors/dino_preprocessors.py、architecture/models/allenact_transformer_models/inference_agent.py、online_evaluation/online_evaluator_worker.py、scripts/eval.sh；untracked 包括 diagnostics/、full_diff.patch、online_evaluation/end_causal_diagnostic_logger.py。均须保留。
- 已读取 diagnostics/end_causal_audit/ 下 gate_review.md、analysis_validation_report.md、runtime_manifest.json、checkpoint_integrity_report.md、reset_state_report.md、reset_two_episode/reset_transition_summary.json。历史运行结论本轮未重跑验证。
- reset 原始摘要记录两个 episode 长度 8/21，reset 前后三分支 counter 均为 8 且 cache norm 不变；29 decisions 未观察 rollover。旧 token 被 mask 隔离的结论来自既有受控测试报告，不能外推为所有跨 episode 影响均不存在。
- 历史 Gate A PASS、Gate B FAIL、Gate C PASS（限已观察时序）。创建协议不解除 Gate B。

## Observations

- 最新001A只验证provenance：两run各2/2成功标志、cost总计0；eps_len为Reference 28/56，Repeat 27/62。轨迹不同，不要求bitwise一致，不归因于DINO/reset，不作SR或Safety Cost性能结论。原始success=1.00000001与num_eps字段均未修正。

- 历史 full-200 报告 SR=173/200=0.865；Safety Cost 总计145、均值0.725；12/200 episodes 非零，top-1占45.52%、top-5占86.90%。来源 analysis_validation_report.md。本轮未重算原始 full-200 结果，不冒充最新结果。
- 历史 canonical_final_v2 单 episode 8 steps，在 PRE stop_legal=false 时 greedy 选择 end，policy_end_prob≈0.993365，结果失败。来源 runtime_manifest.json；单例不足以推断系统性机制。
- reset_state_report.md 记录 model max_steps=500、环境 horizon=600 和累计 counter rollover。历史固定 seed/greedy smoke 仍有轨迹差异，固定 seed 不等于逐 bit 可复现。

## Active Hypotheses

- H-RESET：跨 episode 累积 counter 改变回卷时刻，进而改变当前 episode 可用历史、logits/动作；SR/Safety Cost 因果影响尚未建立。
- 竞争解释：短序列 mask 隔离旧状态，差异来自 test augmentation、GPU 非确定性或 episode 调度。
- 感知、语言条件、融合、时间表征、readout、采样、探索与训练目标/安全约束均为开放候选，不自动延续某一个假设。

## Rejected / Unsupported Hypotheses

- 既有 smoke 未发现 logger 多余 forward、RNG 或动作改变；只限被观察范围，不能无限外推。
- “害怕危险所以提前退出”“单个高 end 概率证明 representation/readout mismatch”无充分因果证据。
- “cache 保留所以新 episode 第一步一定使用旧 token”不受现有证据支持。
- Gate B 解除或被合法隔离前，不接受受影响 hidden-state 的机制结论。

## Highest-Value Uncertainty

001A的同一executable B0身份、稳定task配对及产物完整性已通过。下一关键不确定性是拟议001B完整任务集/worker调度/资源预算/差异容差协议是否充分冻结，之后才可研究该B0的full-200 SR与官方Safety Cost可复现范围。
DINO在线/本地loader等价不再是阻塞点；001A不证明full-200性能或多worker可复现。
唯一下一动作：人工评审001A结果与001B草案。本轮已停止，不执行下一实验；reset机制假设保持开放且DEFERRED，旧Gate B范围不因此自动解除。

## Control-plane bootstrap — latest scope

用户批准本轮建设并commit/push research-loop控制面；这覆盖此前仅对控制文件的no-commit限制，不授权任何实验。原开发与candidate worktree保持原样。GitHub评审包位于research/review_packets/EXP-B0-REPRO-001A/。当前没有获批真实研究实验，也没有获批handoff dry run。
