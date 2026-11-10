# NVIDIA ObjectNav mask/unmask evaluation — 20261103T163308Z-live-3148

这是服务器 `/public/home/zhangyh/QYY` 中已完成实验的产物归档（不含视频）。“cost 约0.55”对应的准确结果是 **MASK_ON200 平均 Safety Cost 0.535**。

## 已完成的结果

| 组别 | 任务数 | 成功数 / SR | 成本总和 / 均值 | 决策数 |
|---|---:|---:|---:|---:|
| MASK_ON200 | 200 | 169 / 84.5% | 107 / **0.535** | 17,754 |
| MASK_OFF200 | 200 | 150 / 75.0% | 1,076 / **5.380** | 29,124 |

五项成本总和（danger / corner / blind / fragile / critical）：

- MASK_ON200：9 / 18 / 39 / 3 / 38。
- MASK_OFF200：15 / 384 / 615 / 9 / 53。

`PAIR_STATUS.json` 为 COMPLETE，400/400 个 episode 完成。`SUMMARY.json` 和两个 `episode_metrics.csv` 从原始完成记录独立复算；`COMPARISON.json`、验证 JSON 和控制台日志保留原始字节。本包发布于实验完成之后，不是预注册设计或新的运行许可。

## 主要代码与实际启动方式

本归档分支基于实际运行源码 **2689081ec324c67a35f2a4692cfa009826fcf9d4**，完整科学代码保留在仓库根目录。关键文件：

- `architecture/models/allenact_transformer_models/allenact_dino_transformer.py`：root Actor 的跨 episode attention mask 开关。
- `diagnostics/actor_memory_unmask/configure.py`、`evaluate_online.py`、`online_pair.py`：配对评估和配置。
- `diagnostics/actor_logits/logger.py`、`storage.py`：逐步 logits、概率、动作、边界和张量记录。
- `tasks/abstract_task.py`、`online_evaluation/online_evaluator_worker.py`：原有安全成本和评估语义。
- `deployment/unmask_nvidia_20261103/`：这次真正使用的 NVIDIA 适配、手动运行门禁、实时日志、输出扫描竞态修复与测试。
- `deployment/vulkan_repair_20261031/`：Vulkan 环境激活入口和配置；不包含 NVIDIA 驱动二进制。

实际启动命令（历史记录，不是下载本分支后可无条件直接执行的通用安装命令）：

```bash
nohup bash /public/home/zhangyh/QYY/unmask_nvidia_20261103/run_pair_live.sh \
  --run --gpu 0 > "$LOG" 2>&1 < /dev/null &
```

原始结果根目录：

```text
/public/home/zhangyh/QYY/runs/EXP-ACTOR-MEMORY-UNMASK-NVIDIA/20261103T163308Z-live-3148
```

入口链：`run_pair_live.sh → run_pair.sh → pair_entry.py → online_pair.py → evaluate_online.py`。实际 Python 为 `/public/home/zhangyh/QYY/envs/safevla_unmask500_20261028/bin/python`，并不由终端显示的 conda 环境名决定。

`pair_entry.py` 只在内存中替换输出目录大小扫描，容忍 `.partial → .complete` 原子改名导致的 FileNotFoundError；原冻结科学源码没有被编辑。两组均关闭额外 Actor 边界 reset；ON/OFF 指 attention mask，不是 reset。两组均保留通常的 `agent.reset()`。

固定配置：ObjectNavType / minival200 / objaverse，worker=1，seed=123，shuffle=true，greedy=false，test_augmentation=true，模型有效 cache=500，episode horizon=600。相同的200条 TaskSpec、checkpoint、DINO 权重及首次模型构造前 RNG 记录均在原 manifest 中。GPU 映射和 NVIDIA 535.104.05 渲染配置见 `PAIR_LAUNCH.json`。

## 归档范围（不含视频）

原始结果共3215个文件；按用户要求排除400个视频，发布2815个原文件。`ORIGINAL_ARTIFACT_INDEX.json` 的 `files` 映射已发布文件，`excluded_files` 仅保留被排除视频的名称、大小和校验值。9个独立 ZIP 包含：

- 400个 episode 的全部 `actions.jsonl`、`execution_results.jsonl`、`boundary_states.jsonl`、`metrics.json`、`EPISODE_STATUS.json` 和 `tensors.npz`。
- 两组完整 RUN_MANIFEST、Task manifest、原始逐任务 CSV。
- **400张俯视图**；不包含任何视频文件。
- 顶层比较结果、验证结果、启动记录和控制台日志。

`archives/*.zip` 每个都是可以单独打开的 ZIP，不是需要二进制拼接的 `.zip.001`。全部解压到同一个空目录即可恢复原结果层级：

```bash
python - <<'PY'
from pathlib import Path
import zipfile
target = Path('restored_run')
target.mkdir(exist_ok=False)
for archive in sorted(Path('archives').glob('*.zip')):
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
PY
```

视频按用户要求排除，俯视图重新打包。已发布原文件字节与 SHA256 保持一致；`EXPORT_INVENTORY.json` 保留服务器初始未筛选导出的记录，不代表 GitHub 发布范围。

GitHub 不在线预览这些 ZIP 中的全部内容，下载后可查看原始张量/俯视图。`ARCHIVE_VOLUMES.json` 记录各卷 SHA256，`ORIGINAL_ARTIFACT_INDEX.json` 记录所有原文件 SHA256、大小与卷内路径。

先执行完整性与指标复核，无需导入 torch、CUDA 或模拟器：

```bash
python verify_archive.py
```

没有打包 checkpoint、DINO 权重、Conda 环境、Objaverse 数据/资产或 NVIDIA 驱动安装包。这些是外部依赖，不是本次生成的结果。checkpoint 为官方 `SafetyEmbodiedAI/safety-model/safe_objnav.pt`，SHA256 为 `05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301`；DINO 权重 SHA256 为 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。

## 结果解释边界

此运行是 `MANUAL_OFFLINE_NVIDIA`；原 manifest 明确记录未获得 formal control / GitHub CI 验证，不能改称正式门禁 PASS。全资产启动哈希扫描按用户要求关闭；没有在此归档任务中恢复扫描数据集。

原始0.535/5.38保留不变，作为实测描述，而非论文1.854的已验证复现或单一变量因果效应。后续审计发现：

1. `--seed` 未传入模拟器内部运动噪声的 `System.Random`；同 Python/Torch seed 不保证相同实际运动轨迹。
2. 旧对照服务器 DINO 可用 xFormers，新评估环境缺少 xFormers，会选不同 attention 后端。`ENVIRONMENT_OBSERVED_AT_EXPORT.json` 是导出时观察，不伪造为运行当时的完整环境快照。
3. `is_corner_unsafe` 将可达点的 `(x,z)` 与 agent 的 `(x,y)` 比较；独立坐标平移反例证明其几何不变性问题。这里保留原官方实现，未悄悄重算或替换指标；对本次分差的实际影响尚未测量。
4. 两组首个 episode 在尚无跨 episode 历史时已出现输入/动作分叉，不能把全部均值差异归因于 mask。

归档没有修复上述科学问题、重新执行实验、修改旧结果、变更正式研究控制分支或授予继续运行权限。
