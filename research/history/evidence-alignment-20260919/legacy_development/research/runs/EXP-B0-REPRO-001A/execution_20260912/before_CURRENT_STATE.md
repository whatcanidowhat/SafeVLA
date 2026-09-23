# Current State

Updated: 2026-09-12（Asia/Shanghai）
Mode: B0-CANDIDATE-AUDIT完成静态审计，等待人工评审。用户本轮禁止恢复smoke；001A仍BLOCKED_PREFLIGHT、0 runs / 0 episodes，001B禁止执行。

## Verified Facts

- 2026-09-12：GitHub API固定commit SHA/tree SHA与本地对象一致，已核实官方reference 2aa82559d272b5f888e53433e258914057f15bed，不仅依据共同祖先。
- 独立candidate /nvme2/user/qyy/SafeVLA_baseline_clean，branch codex/b0-candidate-audit：111/112官方tracked文件字节一致；唯一tracked适配为DINO加载入口，另新增路径env文件。原开发源码/diagnostics保持。
- 候选worker不含开发版无条件预决策查询，greedy/history与reset/counter原官方行为全部保留；DINO本地加载仍B类，未进行runtime验证。
- 完整diff和适配分类见research/B0_CANDIDATE_AUDIT.md；smoke未获本轮授权，未启动。

- 本轮001A前置核查已保存HEAD、tracked/staged/官方参考patch、100份候选源码快照/哈希、DINO源码快照及模型文件新计算SHA256；没有运行时加载/import或episode结果。
- 官方参考共同祖先6464cbdc0ef4d2948157c43f90304e1261d4209f与当前worker存在无条件预决策环境查询差异，其行为等价性尚未建立。不是已证明SR/cost改变。
- 历史86.5% full-200仅作historical reference，不能充当本轮Reference Run。

- 本次 SSH 确认 repo 为 amax@192.168.134.211:/nvme2/user/qyy/SafeVLA；/nvme2/qyy/SafeVLA 不存在。
- branch=baseline-experiment，HEAD=60bc54fbdedaf5745d0476c25321e808708273aa。
- 当前 tracked diff SHA256=f69354e479fdc1f7ad78ab0f30f5ee57e5b4b50a3cbfff2c53e5f69792d7599b，与历史 manifest 一致。
- 构建前已有 4 个 tracked 修改：architecture/allenact_preprocessors/dino_preprocessors.py、architecture/models/allenact_transformer_models/inference_agent.py、online_evaluation/online_evaluator_worker.py、scripts/eval.sh；untracked 包括 diagnostics/、full_diff.patch、online_evaluation/end_causal_diagnostic_logger.py。均须保留。
- 已读取 diagnostics/end_causal_audit/ 下 gate_review.md、analysis_validation_report.md、runtime_manifest.json、checkpoint_integrity_report.md、reset_state_report.md、reset_two_episode/reset_transition_summary.json。历史运行结论本轮未重跑验证。
- reset 原始摘要记录两个 episode 长度 8/21，reset 前后三分支 counter 均为 8 且 cache norm 不变；29 decisions 未观察 rollover。旧 token 被 mask 隔离的结论来自既有受控测试报告，不能外推为所有跨 episode 影响均不存在。
- 历史 Gate A PASS、Gate B FAIL、Gate C PASS（限已观察时序）。创建协议不解除 Gate B。

## Observations

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

独立候选已排除开发worker的额外预决策查询；下一审阅重点是官方reference及DINO/路径B类适配能否作为限定001A验证对象，以及无侵入runtime产物记录方式。
不要再用当前开发worker的查询差异推断新candidate仍有同一差异；也不要把source-level静态一致升级成runtime-equivalent。
001A保持不启动，等待人工审阅research/B0_CANDIDATE_AUDIT.md；001B和完整200-task禁止。
