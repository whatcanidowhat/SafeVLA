# diagnostic-action-history-20260802/launcher.log 审稿

## 事实与因果链

- 【已证实事实】日志 946 行、93,543 bytes；00:18:51 起反复轮询四张卡，阈值为“无外部 compute、显存≤4GB、util≤30%”。
- 【已证实事实】第一次轮询阶段没有启动评估；00:25 左右第二次流程选择 GPU1，命令为单 episode、greedy、no-shuffle、seed123。
- 【已证实事实】日志停在 AI2-THOR controller 初始化；目录只有 W&B 流文件，没有 metrics/FullAggregatedResults；pid 已不存在。
- 【已证实事实】因此该日志不能证明完成了一次导航任务，也不能支持 action-history 修复有效或模型机制结论。
- 【合理推断】进程可能在 controller 初始化后异常退出、被终止或会话断开；日志没有 traceback/exit code，无法区分。
- 【待验证假设】旧 run 是否因资源竞争、超时、外部 kill、AI2-THOR 卡死或未记录异常而终止。

## 元认知审查

- “GPU wait 叙事很长、模型结构 dump 很长”不等于“实验成功证据很多”。日志体量会诱发完成感，核心终态证据实际为空。
- GPU util 的瞬时波动不是稳定空闲的充分证据；真正相关的是外部 compute PID、显存和连续稳定窗口。旧 run 的 stable_checks=1，存在抢卡竞态。
- 单 episode 固定前缀只能做判别性/管线诊断，不能外推整体 SR 或 action-history 因果效果。

## 道—法—术—器

- 道：先证伪运行真实性，再谈模型；不让预设假设驱动证据选择。
- 法：按 PRE→decision→branches→action mapping→environment→POST 建立可审计链，并区分事实/推断/假设。
- 术：记录 exit code、结构化 step JSONL、checkpoint/hash、RNG/forward assertions、end-label alignment。
- 器：SSH、nvidia-smi、W&B 文件、AI2-THOR、diagnostic logger 都只是取证工具；工具输出本身不等于结论。
