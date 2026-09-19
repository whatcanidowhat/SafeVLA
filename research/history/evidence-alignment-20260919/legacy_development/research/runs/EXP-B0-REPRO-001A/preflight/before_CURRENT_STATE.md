# Current State

Updated: 2026-09-11（Asia/Shanghai）
Mode: protocol-only；正式实验状态 DRAFT_BLOCKED。本轮构建协议并读取已有证据，未启动评测、Probe、Oracle 或训练。

## Verified Facts

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

固定输入和随机状态、隔离现有 B0 后，counter 起点是否通过 rollover 改变当前序列策略输出？唯一草案见 NEXT_EXPERIMENT.md。先核实可用重放输入、runtime 和隔离方案，形成可审阅设计；本文件不触发新实验。
