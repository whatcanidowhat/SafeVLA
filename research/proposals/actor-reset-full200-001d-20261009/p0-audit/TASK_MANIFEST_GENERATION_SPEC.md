# 001D task manifest 生成规范（未生成实际 manifest）

数据已只读核验：200行；compressed SHA256 `fa0dffc2dece071716dcf648e19c6944caf1621ee24dadc6996eadcd46eee04a`；uncompressed bytes SHA256 `f10ce169dc71e0ee2b4f9bb803a464babf8c72e28ad7d7d2a28b0d80a174863e`。尚未执行官方 evaluator、未生成或冻结200行001D manifest。

## 生成流程（P1获批后实施）

1. 在隔离B0 snapshot的正确cwd读取 `./benchmark/objectnavtype_val.jsonl.gz`。不要误以为CLI dataset_path替代了该硬编码相对路径。
2. 读取全部原始JSONL行，原始行号为稳定 `sub_house_id`。保存数据bytes hash、解析器/Python版本。不得按house、expert长度或已知success排序。
3. 严格复用 `online_evaluator.py:352–367`：创建0..199，原 `random.seed(123); random.shuffle`，随后原normalize流程；最终200项全部保留。不用NumPy shuffle替代，不改变runtime全局RNG消耗。
4. `sample_id` 按 `online_evaluation_types_and_utils.py:75`：`task=ObjectNavType,house=<house_index>,sub_house_id=<original_index>`。完整raw TaskSpec保留目标synsets、broad/narrow对象ID、初始xyz/yaw、goal、extras、task_path等，不能只hash显式显示的几列。
5. 复用 `normalized_eval_sample_to_task_spec:94–119` 的实际转换；在原队列入队前旁路复制，保存原始TaskSpec和normalized TaskSpec两个digest。不要为导出manifest构造model/controller。metadata-only工具若必须import会触发runtime初始化，应拆分/审查后执行。
6. 原第二次shuffle (`online_evaluator.py:535–539`) 决定needs_video标记，保留其调用和结果；它不改变 `for sample in samples` 入队顺序。task内容与操作元数据可分别hash，避免把timestamp当stable ID。
7. 写 `TASK_SPECS.jsonl`（完整规范对象）和 `TASK_MANIFEST.csv`（200行索引）。建议JSON规范为UTF-8、sort_keys=true、separators=(',',':')、ensure_ascii=false、禁止NaN，每行末LF；NumPy arrays按原数值转list，显式保存转换规则。每臂生成后先验证与冻结文件字节/semantic digest一致，再允许第一个episode。
8. 记录worker_id=0、session内ordinal、task dequeue/initialization/start/completion。assert 200 unique stable IDs、200specs、集合不缺失、排序一致；动态timestamp episode ID只能附加，不能作主键。

## CSV建议字段

`ordinal_0based, original_row_index, sample_id, task_type, house_index, task_path, goal, initial_x, initial_y, initial_z, initial_yaw, raw_spec_sha256, normalized_spec_sha256, needs_video, worker_id, run_seed, dataset_sha256`

另存generator_source_sha256、B0_source_manifest_sha256、Python/依赖、serialization规则、全manifest SHA256。初始environment **实际** pose/seed/scene身份只在P1运行时采集，禁止将计划值标成实际核验值。

P1两个OFF session使用完整序列前5个任务。取前缀不意味着在官方shuffle之前截取前5行，也不应因eval_set_size=5改变未登记的初始化/视频选择RNG路径。优先对完整队列生成做对照并在已批准launcher中明确前缀消费边界，首个模型forward前验证。

## 失败条件

数据hash/TaskSpec/order改变、重复任务、缺任务、未知ACTION_DICT覆盖、初始化失败递归跳题、任一臂从不同manifest读取：立即停止，保留partial和错误，不重新shuffle、不替换task、不补跑。两臂task内容一致不等于其后续随机流或初始模拟器实际状态已经验证相同。
