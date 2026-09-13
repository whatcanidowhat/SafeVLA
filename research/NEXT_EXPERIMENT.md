# Next Experiment — PI review complete / closed handoff cycle

Experiment ID: EXP-LOOP-HANDOFF-001
Status: DRAFT
Authorization: NOT_AUTHORIZED
Cycle ID: handoff-001-20260913

该 handoff dry run 已完成并由 PI 独立读取结果 commit `0e9c3bceac3c9b4dbe9daed6d37307080c2d0b3e` 后回执。当前 DRAFT 仅表示关闭后的非执行归档状态，不得重新 claim 或重跑。任何后续实验都必须使用新的 cycle_id 并由 PI 重新设计、审查和批准。001A 已完成，不得重复；001B 仍未授权。

Research Question: 实际PI与Executor是否能完成一次GitHub批准→claim→no-op结果→PI回执？
Hypothesis: instruction_commit绑定批准HEAD，单次claim，四项结果完整，PI独立读取并回执。
Competing Explanation: 仅本地文件或Executor代写回执，被误认为双向交接。
Reference / Repeat: 批准commit A与一次no-op结果C，无policy treatment。
Unique Variable: 一次控制交接周期。
Fixed Conditions: 0 GPU、0 episode、不import模型、不启动AI2-THOR、不进入runtime worktree、单任务、非force push。
Metrics: 指令SHA正确、成功claim唯一、required outputs可读、真实PI reviewed_result_commit回执。
Expected Result: A→B→C→D且控制CI通过。
Falsifying Result: 未获PI批准/回执、重复或过期claim、结果缺失或身份漂移。
Alternative Explanations: API可读不等于PI已阅读；CI成功不等于PI批准。
Stop Conditions: 未批准/过期、stale/push不确定、校验失败、任何模型/episode调用；发布结果后STOP。
Required Artifacts: RESULT_SUMMARY.md、RUN_MANIFEST.json、ARTIFACT_INDEX.json、REVIEW_NOTES.md；准确路径见LOOP_STATE。

PI acknowledgement D 将 `reviewed_result_commit` 绑定到结果 commit C；当前状态回到 `PI_REVIEW / NOT_AUTHORIZED`。本周期已关闭，不授权001B或任何真实研究实验。
