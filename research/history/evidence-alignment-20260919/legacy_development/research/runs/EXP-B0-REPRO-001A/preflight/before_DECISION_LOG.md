# Decision Log

## 2026-09-11 — DEC-LOOP-001：建立最小闭环

依据：用户要求遵照“分支 · 02｜SafeVLA研究”（ID 6a7b4d04-abf0-83ee-991b-e3e9a9821032）末尾构建要求，先完成共享事实层和skill，不启动新实验，再进行人工监督的两轮dry run。

决策：创建AGENTS、五份research文件及safevla-research-loop。中文正文保留工程字段；约束证据程序，不固定研究机制。实际服务器路径为 /nvme2/user/qyy/SafeVLA。

范围：协议构建、已有证据只读核对和文件验证。保留原有代码/产物/工作区修改，不扩大为提交发布、发送聊天消息或后台持续运行。

## 2026-09-11 — DEC-LOOP-002：继承Gate B阻断

证据：diagnostics/end_causal_audit/gate_review.md、reset_state_report.md、reset_two_episode/reset_transition_summary.json。当前tracked diff哈希与历史manifest一致；本轮未重新运行模型或哈希checkpoint。

决策：Gate B继续FAIL。counter/cache保留为已有记录，rollover对action/SR/cost的影响仍需隔离验证，不能静默修复。

为何不直接Probe/扩大诊断：状态来源未隔离会削弱hidden-state解释。为何不直接oracle：不能优先解决当前运行状态不确定性。

下一步：唯一草案EXP-RESET-001，先核实重放输入/调用语义，明确单变量、参数和执行范围。这是候选设计，不是已批准的最终实验。

## 后续记录

记录日期、证据路径/版本、研究问题、所选与未选方案理由、结果如何改变假设、停止/继续条件、唯一下一实验。负责人读取代码和原始产物，不只接收聊天总结。
