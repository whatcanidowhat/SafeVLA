# SafeVLA 历史证据对齐包

登记时间（UTC）：2026-09-19T17:24:15.008732+00:00  
归档基于 research-loop：6d41b40012142022f4a8eec245a2f2218084b0f2。

本包落实用户的“先盘点、分三类、一次性提交共享证据”要求。它是历史文件归档与完整性核验，不是 EXP-SMALLTARGET-PHENOTYPE-001 的执行结果；没有领取实验，没有训练、模型加载、GPU、AI2-THOR、episode、重放或尺寸关联分析。LOOP_STATE 与两份 NEXT_EXPERIMENT 保持原字节。所有 legacy_development 文件都是旧版本快照，不能覆盖当前控制文件或充当当前授权。

## 1. 历史研究但未入 Git 的文件

本轮限定范围内登记 684 项路径，其中 680 项存在且可读，4 项为明确缺失的旧名称。按 Git blob 字节身份与四个远端分支比对，35 项来源已有完全相同的 GitHub 副本，不能再称为“未入 Git”；剩余 645 项列于下表文件。

- [HISTORICAL_NOT_IN_GIT.csv](HISTORICAL_NOT_IN_GIT.csv)：逐文件路径、三类归属、大小、SHA256、共享副本路径与理由。
- [FILE_INVENTORY.json](FILE_INVENTORY.json)：完整来源清单、GitHub commit/path 对照、已扫描范围和缺失记录。
- [ARTIFACT_INDEX.json](ARTIFACT_INDEX.json)：Git 可读证据副本及哈希，受仓库 CI 校验。
- [DEVELOPMENT_PRESERVATION.json](DEVELOPMENT_PRESERVATION.json)：原开发 HEAD、dirty diff 和状态前后相同。

范围完整覆盖开发目录 diagnostics/end_causal_audit、research，以及指定 full-200 run 顶层媒体和最终 W&B 表；另外定点核对两个 Original 副本的 Probe 文件/日志和指定数据源。本包不声称穷尽整个服务器的所有历史文件或所有 W&B 中间快照。

## 2. 三类处置

| 分类 | 完整清单中的路径数（含已有 Git/缺失项） | 本次处置 |
|---|---:|---|
| 必须入 Git / MUST_GIT | 154 | 小型原始结果表、必要任务规范、报告、结构化诊断和离线提取脚本，保存原字节或明确标记转换方式 |
| 只登记路径和 hash / PATH_HASH_ONLY | 502 | 视频/图片、Probe 张量、大日志、压缩数据集、原始配置与已有 Git 副本，登记 SHA256、大小和访问边界 |
| 不应进入共享控制面 / DO_NOT_SHARE_CONTROL | 28 | 开发/runtime 代码、launch/replay 文件、dirty patch、环境转储和临时 PID；仅引用身份，不复制内容 |

密钥、认证 remote URL、完整环境转储不进入共享副本。已在实验分支的源码以固定 commit/path/blob 引用。旧 runtime 源码不会因为改放在 research 下就变成控制数据。

## 3. 当前小目标实验：九类输入核对

| 旧证据 | 核对结果 | 共享入口 / 限制 |
|---|---|---|
| historical full-200 原始 200 行 | 存在；200 行、173 success、sum_cost=145，与历史报告一致 | [完整性核验](full200/INTEGRITY_CHECK.json)、[200 行 CSV](full200/episode_results.csv)；原始 W&B JSON 同时保留 |
| W&B table / episode-level | 最终 VideoTable 有 22 列；summary 所引最终表的 SHA256 一致 | [原 summary](full200/wandb-summary.json) 与 full200/media/table；197/199 行中间表不是完整结果 |
| task spec / stable identity | 原 benchmark 200 行；规范化 task_path 后 200/200 唯一配对 | [原任务规范解压副本](task_specs/objectnavtype_val.jsonl)、[身份检查](task_specs/task_identity_check.json)；不用动态 episode ID 或行序作为身份 |
| gt_episode_len / expert_length | 200/200 非缺失，结果与任务规范逐条相等 | 上述原表与任务规范 |
| target-room visitation | has_agent_been_in_room 200/200 已保存 | 原表；该字段是轨迹之后的观察量，不是先验任务难度或物理尺寸 |
| scene/object metadata | val.jsonl.gz 与 annotations.json.gz 可读且已 SHA256；前者有 room/object/assetId | [字段与限制](metadata/STATIC_METADATA_AVAILABILITY.json)；annotations 36813 条 size 非空、2851 条为空，但尚未验证尺寸单位、实例缩放、bounding box 与 broad-target 绑定 |
| Probe .pt / log / analysis scripts | 两副本共四个 worker 张量均可读且哈希不同；两个 11 MB 日志哈希相同 | [Probe 来源说明](probe/PROVENANCE.md)；原 AUC 三元组的精确 1000-sample 文件/分析 manifest 尚未识别 |
| end_causal_audit report/json | 已找到并归档各版本小型报告与结构化轨迹 | legacy_development/diagnostics/end_causal_audit；Gate B 结论不在本次解除 |
| sub120 单案例证据 | actions JSON/CSV、近似概率 CSV 和两个提取脚本存在 | sub120/；概率为视频量化近似，不是精确 logits，也不是群体机制证据 |

仅完成源文件身份/完整性核验。没有生成当前实验的 category_sr、size effect、回归或 size_analysis 结果；静态目标实例尺寸能否完整构建仍需在正式实验中验证。

## 4. 网页研究与 GitHub/服务器的对齐

[WEB_RESEARCH_CONTEXT.md](WEB_RESEARCH_CONTEXT.md) 记录已读取范围与研究边界。可直接读取的“分支 · 02｜SafeVLA研究”共 42 轮，涵盖继承的早期研究和后续分支；主“02｜SafeVLA研究”独立链接未获得，不能声称另行读取了主对话全文。GitHub 已迁移的 LEGACY_RESEARCH_STATE / EVIDENCE_REGISTER 用作索引，实际原始文件优先。

- 173/200 的 2026-08-03 run 与旧 169/200 或修改策略的 174/200 run 严格区分。
- 后来“某些小类别约 50%”的回忆尚未绑定到同一 run，不与该 173/200 表混合。
- 历史 Probe 标签是 close-and-visible / stop-legality-like；AUC 不能当作小目标尺寸证据。
- 广义 synset 的 success-eligible target IDs 与旧 narrow visibility/room 指标可能不同；不得静默等同。
- 所有旧报告中的后续实验建议都只是历史文本，不是当前执行授权。

## 5. 复核与使用

运行 tools/verify_archive.py 可离线复核共享副本、原始表、200 个稳定键及专家步长配对。加 --server 可重新校验库存中的全部现存来源哈希（会读取大文件）。tools/build_archive.py 保存本轮原始取证/转换逻辑；源路径与审计时分支身份以本包 manifest 为准。

完整源码 hash、checkpoint 与历史协议的存在不证明当前运行可复现。此次归档不推进正式实验状态，不重跑 001A/001B，不替 PI 续期或批准。
