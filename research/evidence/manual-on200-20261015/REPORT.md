# 手动 ON200 只读审计与历史 OFF200 对照

结论：**存储证据完整性 PASS；可作带限制的同 TaskSpec 配对描述性比较。不是 P2-004 Claim 的成功执行，也不是因果等价验证。**

ON 身份为 `MANUAL_OUTSIDE_RESEARCH_LOOP`，执行源码 `67c7051ed4287968897282f915b8b076851562a1`。原 P2-004 仍为 BLOCKED（终结提交 `d478be22efea7b1fc94d5dbe8750d09fd903e44a`）；本独立证据分支未改变控制状态、授权或 Claim。

## 严格验证和身份

现有 `diagnostics.actor_logits.validate.validate_run` 对每组各执行一次，作为现有 `analyze.compare` 的内部严格验证调用，并只捕获其返回报告，不放宽规则。OFF：200/200、19,270 决策；ON：200/200、18,424 决策；各 1203 个索引文件均通过。验证包括逐步动作/全动作向量/计数、官方成功与五项成本逐步累加、完整 Episode 边界、Actor counter/KV reset 和 Critic 状态隔离、完整文件集合/大小/SHA256。

两组 200 个 TaskSpecs 内容、顺序、action map、checkpoint/DINO SHA、seed=123、worker=1、shuffle=true、greedy=false、test_augmentation=true、官方评估参数（除输出目录）一致，物理 GPU 编号均记为 0。Task manifest SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0`。109/112 个记录源码哈希分别与各自精确 Git 提交核对；科学执行主体在初始授权门禁之后 AST 相同，模型/Actor/Critic/reset/logger/官方指标文件未变化。源码差异只限已登记的编排/控制/分析接口与相关新文件，详见 RUN_CONFIG.json。本次未重新读取权重文件进行哈希，权重一致性依据运行 manifest 的相同 SHA、冻结执行代码的实际加载前检查及完整 load_events。

## 结果

SR：**84.5% → 87.0%，+2.5 个百分点**；11 个任务由失败变成功，6 个由成功变失败，163 个两次成功、20 个两次失败。

| 指标 | OFF200 | 手动 ON200 | ON−OFF |
|---|---:|---:|---:|
| 成功数 / 200 | 169 | 174 | +5.000 |
| Safety Cost 总计 | 177 | 501 | +324.000 |
| 平均 Safety Cost | 0.885 | 2.505 | +1.620 |
| 决策数 | 19270 | 18424 | -846.000 |
| 失败且提前 END（代理指标） | 19 | 14 | -5.000 |
| 相邻重复动作次数 | 13140 | 12257 | -883.000 |
| ABA 震荡次数 | 1838 | 1806 | -32.000 |
| corner 成本状态下重复动作次数 | 61 | 77 | +16.000 |
| danger 成本分量 | 2 | 0 | -2 |
| corner 成本分量 | 87 | 118 | +31 |
| blind 成本分量 | 56 | 327 | +271 |
| fragile 成本分量 | 13 | 14 | +1 |
| critical 成本分量 | 19 | 42 | +23 |

ON 总成本增加 324，其中 blind 增加 271。ON 最高三个任务成本为 148、148、81，合计 377/501（75.25%），存在明显长尾。每任务成本样本标准差 OFF=4.270，ON=16.278。相邻重复动作比例（相邻转移为分母）68.90% → 67.26%；单看重复总数下降不能消除步数不同的影响。提前 END 定义仅为“失败、长度<600、最后执行 end”；ABA 定义为 a[i]=a[i−2]≠a[i−1]；corner 重复按现有工具定义，不代表已诊断死角或非法终止。

配对任务正态近似 95% 区间：SR 差 [-1.54,+6.54] 个百分点；平均成本差 [-0.610,+3.850]。这是单 seed、固定任务集的描述性区间，未建模共享房屋依赖或跨 seed 不确定性，不支持确定改善或因果结论。当前观测为 SR 小幅提高、Safety Cost 明显增加，不满足“提高 SR 且保持/降低安全成本”的联合目标。

## logits 与轨迹混杂

现有分析器比较 15,023 个同任务同时间步位置；仅 1 对输入哈希相同，15,022 对不同。均值 JSD=0.179341，最大 raw-logit 绝对差=28.359582；153 个任务在共同前缀内存在动作分歧。额外未配对步数 OFF=4,247、ON=3,401。以上不是相同输入/状态的 replay 因果比较；同任务配对、相同 seed、同 GPU 编号不能排除跨日运行、模拟器/增强/采样差异或轨迹分叉。

## 手动门禁替换与退出状态

原始 manual_on200.py 单独归档于 MANUAL_LAUNCH_SOURCE.txt（SHA 见索引），它替换 `recovery.before_model`，保留 OFF 完整性、TaskSpec 和科学源码一致性检查，返回明确手动身份及空 claim/approval。它绕过了正式 Claim/CI、网络凭证及冻结 launcher 的排队/资源监管路径；不能追认其为受控实验完成。

代码检查显示替换函数在模型加载前执行，后续随机种子重设、权重/DINO/依赖检查、模型加载、原 Actor reset、Critic、logger 和评估计算保持原代码。没有发现改变科学计算的脚本修改，存储边界验证也通过，因此结果可用于上述受限比较。但脚本在运行树之外，未纳入原 source_sha256；本次单独哈希只能证明现存脚本身份，不能证明所有未记录运行时行为。GPU logger 仍为 `NOT_RUN_PI_WAIVED`，不是 GPU 无侵入性 PASS。

日志含 “Worker 0 processed 200 tasks” 和 “Evaluation finished”，manifest 为 COMPLETE；未找到独立 shell 退出码文件，故退出码记为 **NOT_RECORDED**，不伪造 exit=0。日志北京时间 2026-10-15 14:56 至 18:52 左右。原始日志、配置和数据支持应用层完成，不补造启动/GPU 调度凭证。

## 证据位置与边界

- 手动原始数据：`/nvme2/user/qyy/SafeVLA_manual_on200_20261015/ON200`；完整日志、手动脚本和视频：`/nvme2/user/qyy/SafeVLA_manual_on200_20261015`。
- 历史 OFF：`/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200`，保持只读。
- 全量验证、逐步对照、任务表和完整服务器哈希索引：`/nvme2/user/qyy/SafeVLA_manual_on200_audit_20261015`。
- Git 仅收录本报告、必要配置/验证与统计摘要、日志摘录、手动脚本文本、哈希索引。原视频、张量和完整逐步对照不上传。
- 原始 manifest/index/task/metrics 哈希审计前后不变。此次审计没有 GPU 操作、模型/模拟器启动、Episode 或 Claim；派生产物写在独立新目录。本分支仅供 PI 阅读，不是新实验授权。
