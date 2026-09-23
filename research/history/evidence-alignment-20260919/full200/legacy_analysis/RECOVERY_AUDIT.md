# ARTIFACT-RECOVERY-AUDIT-001：原始动作序列可恢复性审计

> 时间：2026-09-18 ｜ 类型：纯重分析（不跑模型、不启 simulator）｜ 限时审计
> 目的：判断原 200-task run 的 nav-visible 失败能否从现有 artifacts 恢复动作序列，以支持原轨迹 replay。

## 结论

**RECOVERABLE（可恢复）**——原始失败轨迹的动作序列可从 W&B 视频逐帧恢复，满足 replay 的前提条件。下一步走 **EXP-LEGAL-STOP-REPLAY-001（轨迹重放）**，而非 stochastic rerun。

## POC-002 结果：sub120 动作序列完整恢复（已通过）

| 检查项 | 通过标准 | 结果 |
|---|---|---|
| 视频完整性 | 精确 600 帧 | ✅ 600 帧 @5fps |
| Action schema | 明确 ACTION_DICT 状态 | ✅ 无 ACTION_DICT/LONG_ACTION_NAME → 默认 ALL_STRETCH_ACTIONS，nav 8 动作序固定 |
| 单帧解码 | 每帧唯一 1 动作 | ✅ 600/600 唯一，0 歧义 |
| 序列长度 | 600 recovered | ✅ 600 |
| End 一致性 | 末动作明确 | ✅ 末动作 = rotate_right（**非 done**）→ **真 horizon** |
| 抽样复核 | 起/中/高像素/末 | ✅ frame 0/50/100/300/599 与自动结果一致 |
| 产物 | json/csv + 脚本 | ✅ sub120_actions.json + sub120_actions.csv + extract_actions.py |

**关键发现**：sub120 全程 600 步**从未执行 `done`/`end`**，动作分布 move_ahead=445 / rotate_left=102 / rotate_right=51 / move_back=2，末动作 rotate_right。→ 之前 taxonomy 中「eps_len=600 是 horizon 还是第 600 步 end」的悬案，对 sub120 判定为**真 horizon**。

> **定性更正**：sub120 不是「纯探索/导航失败」。它 `vis_nav=4683`（目标曾明显进入 nav 相机）+ horizon + 无执行 end，正确标签是 **「Nav-visible horizon failure with no executed stop」**——问题可能发生在「目标出现之后，策略没有利用这个信息」，而非「找到目标之前」。

## P0 结果：sub120 近似 p(done) trace（已恢复）

概率条编码：`action_dist=probs` 传给渲染器，ObjectNav 每 action 一根蓝色条，条宽 = `int(55×prob)`（bar_width=55）。分辨率 ≈ 0.018。

| 指标 | 值 |
|---|---|
| p(done) > 0 的帧数 | **0 / 600** |
| p(sub_done) > 0 | **0 / 600** |
| 策略分布实际只覆盖 | move_ahead / rotate_right / rotate_left（move_back 在 sub-resolution 尾部） |

- 产物：`sub120_action_probs_approx.csv` + `extract_action_probs.py`（**video-rendered quantized probability approximation，非精确 logits/probs**）。
- **严格结论（已纠正越界）**：
  - `rendered done bar` 600/600 帧为 0 → 原始 `p(done)` 每帧都 **低于约 0.018 的可视化分辨率**（**不是** `p(done)=0`）。
  - 只有 move_ahead / rotate_left / rotate_right 的概率条曾超过量化阈值；其他动作可能存在小于约 0.018 的概率质量（**不是**「分布只在三动作非零」）。
  - 能排除的是「done 本来概率很高、只是 stochastic sampling 恰好一直没采到」；**不能**说 sampling 完全不参与（真实 p(done) 可能是 0.001/0.005/0.017 等，视频无法分辨）。
- **对 sub120 的严格状态**：`Nav-visible horizon failure with persistently low rendered stop probability`。
- **尚未证实**：是否 ever official_stop_legal（vis_nav=4683 ≠ successful_if_done）、是否认出目标、是否 temporal memory 丢失、是否 readout 没利用表示。**这些不能提前写进因果链，需 replay 回答。**

## 四项检查

### 1. 结构化动作信息（wandb 表 / 本地文件 / JSON）

- **无**。
- 依据：`task.task_info["taken_actions"/"agent_poses"/"action_successes"]` 仅内存维护；`to_log["agent_path"]` 被构造但未写入 wandb 表；VideoTable 22 列无动作/位姿；`grep` 全 run 文件无 `taken_actions/agent_poses/followed_path`。

### 2. 视频是否 1 帧=1 step，taken_action 是否可恢复

- **是**。
- **帧对齐**：sub68（eps_len=16）视频恰好 **16 帧 @5fps**（ffprobe：`nb_frames=16`, `duration=3.2s`），即 1 帧=1 env step。
- **编码方式**：worker 将 `taken_action=action` 传给 `get_video_frame`（[online_evaluator_worker.py:541](online_evaluation/online_evaluator_worker.py#L541)），渲染为**黑色文本**，非执行动作灰色（[visualization_utils.py:480](utils/visualization_utils.py#L480)：`fill="gray" if action != taken_action else "black"`）。
- **检测验证（proof-of-concept）**：对 sub68 逐帧检测右侧 action 列表的黑文本行，black rows 随步变化：
  - frame 0 → row 6-7；frame 5 → row 3-4；frame 15 → row 9（其余 action 为灰色）。
  - 证明 taken action 可逐帧定位，且随步变化。

### 3. benchmark 是否提供 house / initial pose / task spec

- **是**。`objectnavtype_val.jsonl.gz` 含 `house_index`、`agent_starting_position`、`agent_y_rotation`、`natural_language_spec`、`synsets`、`expert_length`。

### 4. 恢复的动作数 == eps_len，末动作与终止分类一致

- **部分完成**：帧对齐已证 16=16；逐帧「黑文本行→动作名」的完整映射与末动作校验**未完成**（见下）。

## 剩余工作（完整提取）

1. 黑文本行号 → 动作名映射：ObjectNav 的 nav 动作渲染顺序由 `action_names` 顺序 + `navigation_actions` 过滤确定（move_ahead/rotate_right/rotate_left/move_back/done/sub_done/rotate_left_small/rotate_right_small），见 [visualization_utils.py:394-403](utils/visualization_utils.py#L394-L403)。
2. 对 9 个 nav-visible 失败逐一提取，校验动作数 == eps_len。
3. 校验末动作：`eps<600` 应末动作 = done；`eps=600` 末动作待查（区分「第 600 步 end」vs「horizon 无 end」）。

## 硬停止条件（若触发则判 RECOVERY_NOT_SUFFICIENT）

- 视频丢帧 / 帧数 != eps_len
- 黑文本无法唯一确定（一帧多个黑行 / 无黑行）
- 一帧对应关系不明确
- 缺 initial state

## 对后续实验的影响

- RECOVERABLE → 优先走 **EXP-LEGAL-STOP-REPLAY-001**：固定原 sample + 原动作序列 + initial state，做 trajectory replay，逐 step 记录 `official_stop_legal`，直接回答「原失败轨迹是否 ever legal-stop」。
- replay 必须先过 fidelity gate（episode length / action_success / agent pose / termination 与原 run 吻合），否则即便动作相同也不能假定 simulator replay 就是原轨迹。
