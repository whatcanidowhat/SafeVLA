# SafeVLA 四线并行实验包 | 2026-10-23

当前提供的并发操作是 **4 个本地 evidence/preflight gate**，不是 4 次已经完成的 AI2-THOR 模型评测。

## 立即运行

```bash
python run_four_validation_lanes.py
```

四个进程任务通过 `ThreadPoolExecutor(max_workers=4)` 并发产生独立 `results/*.json`，汇总为 `results/FOURWAY_STATUS.json`。

当前预期状态：
- `V1`: historical cohort audit PASS；逐步真实 room 轨迹缺失，拓扑重建 BLOCKED。
- `V2`: topology memory synthetic unit tests PASS；B0 行为/状态等价待真实模型验证。
- `V3`: 动态语言字符串基本输入兼容 PASS；Checkpoint/仿真器/T5 权重缺失，rollout BLOCKED。
- `V4`: 缺少新 treatment，SR/Cost 差分 NOT_MEASURED。

## 在有资源的 SafeVLA 服务器上

```bash
python run_four_validation_lanes.py \
  --safevla-root /path/to/SafeVLA \
  --checkpoint /path/to/safe_objnav.pt \
  --house-dataset /path/to/house_dataset \
  --out-dir ./research/runs/EXP-TOPO-MEMORY-FOURWAY-001/local-preflight
```

⚠ 这个命令只会检查资源存在性与本地模块逻辑，不会自动启动 GPU/仿真任务。
服务器 Codex 按 `CODEX_SERVER_HANDOFF.md` 分别接入真实历史轨迹、策略推理与 treatment；每条线另行进行来源验证并产出真实评测结果。

本仓库交接包仅供 Codex 审阅、CPU 预检和后续经批准的服务器实验；脚本不会领取 Claim 或启动 GPU。

## 来源记录

- https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/handoffs/premature-end-dynamics-001-20260923/premature_end_cases.csv
- https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/handoffs/actor-reset-off200-on200-002-20261014/OFF200_BEHAVIOR_PROXIES.csv
- https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/handoffs/actor-reset-off200-on200-002-20261014/TASK_MANIFEST.csv
- https://github.com/whatcanidowhat/SafeVLA/blob/research-loop/research/BASELINE_CONTRACT.md
