# PI review checklist

本包是新实验提案与CPU准备结果，不是正式research-loop Claim结果，也不是PI回执。现有BLOCKED控制状态仍由PI拥有。

- 核对 SOURCE.patch、8项测试和原始失败/成功重放，确认仅root Actor mask发生变化；任一反例均要求修订。
- 接受CPU组件重放的证据范围：证明历史K/V影响logits，不能推导SR/Cost效果。固定输入与warm cache二进制仅在服务器可访问，RAW_REPLAY_INDEX列哈希。
- 决定是否批准新在线入口实现/审查及GPU logger验证；历史NOT_RUN_PI_WAIVED不是新实验授权，也不是PASS。
- 完整200任务对照是200/臂、共400次starts；若希望总200，必须明确另订100/臂方案。
- 明确资源/expiry/预算，解决GPU繁忙与磁盘不足，固定资产版本和真实online入口，再作独立控制协议批准。不得复用旧Claim、旧实验数字或代写PI批准。
- 线上结果只有在两臂各200完整/身份对齐后可解释；不足或任一门失败均保留部分结果并STOP。

待PI决定，不附自我批准。
