# B0-CANDIDATE-AUDIT

Audit date: 2026-09-12 (Asia/Shanghai)
Status: CANDIDATE_PREPARED / STATIC_AUDIT_COMPLETE / AWAITING_HUMAN_REVIEW
No model execution, AI2-THOR launch, episode, smoke, commit or push was performed.

## 1. 审计结论

已在独立worktree建立候选B0：/nvme2/user/qyy/SafeVLA_baseline_clean，branch codex/b0-candidate-audit。
从独立核实的官方commit开始，而非从diagnostic working tree复制。112个官方tracked文件中111个字节相同，只有DINO加载入口有适配；另新增scripts/b0_candidate_env.sh。候选的完整可应用patch见b0_candidate_audit/official_to_candidate.patch及本文末尾。

当前开发目录 /nvme2/user/qyy/SafeVLA 中原有tracked代码、diagnostics、skill和既有诊断logger保持原样；Git共享元数据增加了worktree/branch，研究文档记录本次审计。具体逐文件保护验证见final_validation.json。

候选消除了上一轮指出的“当前开发worker新增无条件预决策查询”差异，因为候选worker直接取自官方reference；不是以关闭开关冒充去除该路径。
候选保留所有官方已知行为，包括greedy模式中sampled action写入history，以及reset不清counter/cache的原始逻辑。

本轮不能宣称已完成runtime equivalence。DINO本地加载、实际checkpoint加载、动态import路径与episode产物链仍未运行验证。
EXP-B0-REPRO-001A保持不启动；001B/完整200-task不在本轮范围。

## 2. Official reference identity 与核对依据

- 官方仓库remote URL: https://github.com/PKU-Alignment/SafeVLA.git
- 当前开发仓库origin（已脱敏）: https://github.com/whatcanidowhat/SafeVLA.git
- 当前开发仓库upstream（已脱敏）: https://github.com/PKU-Alignment/SafeVLA.git
- 选定reference commit: 2aa82559d272b5f888e53433e258914057f15bed
- Commit title: Add badge for models and datasets availability
- 官方提交链接: https://github.com/PKU-Alignment/SafeVLA/commit/2aa82559d272b5f888e53433e258914057f15bed
- 独立核验API: https://api.github.com/repos/PKU-Alignment/SafeVLA/commits/2aa82559d272b5f888e53433e258914057f15bed

核验链：
1. 官方仓库README列出了同一仓库的clone URL及SafeVLA论文/作者信息，说明项目关联来源。
2. 服务器直接读取GitHub REST API固定commit，返回sha与选定commit一致；返回tree SHA与本地git rev-parse reference^{tree}一致。原始JSON保存为b0_candidate_audit/official_commit_api.json。
3. 本地upstream/main指向同一commit。这是补充一致性证据，不是唯一依据。
4. 不因为共同祖先身份自动认定官方。先前祖先6464cbd...只用于历史diff对比；本候选直接基于API核实的2aa8255...。
5. git ls-remote未及时返回已中断，网页commit抓取不可用；固定commit由成功的服务器API核验。不声称该commit必然是截至今天最新main或论文实验精确版本。
6. API核对时间、tree SHA、脱敏URL保存在reference_identity.json。保存的reference_README.md和reference_commit_object.txt来自已核实的Git对象。

## 3. Candidate B0 identity

- Worktree: /nvme2/user/qyy/SafeVLA_baseline_clean
- Branch: codex/b0-candidate-audit
- HEAD: 2aa82559d272b5f888e53433e258914057f15bed
- Candidate = reference commit + uncommitted official_to_candidate.patch
- Full patch SHA256: f8ff5b1c07a07d5ff717f5f262734dc3d7345ca0d88e5cd5c1607a245e6927db
- 两个差异文件：architecture/allenact_preprocessors/dino_preprocessors.py；新增scripts/b0_candidate_env.sh。
- 没有新commit；不能只用HEAD表示候选身份。新增untracked shell也已经纳入完整patch，并保存单独SHA256。
- 外部资源identity详见resource_identity.json；全部官方文件逐项比对见official_file_comparison.json。

本轮以文件读取重新计算：
- checkpoint: /home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt
  SHA256: 05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301
- DINO checkpoint: /home/amax/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
  SHA256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
- DINO source: /home/amax/.cache/torch/hub/facebookresearch_dinov2_main，157份Python源文件哈希已保存，无.git；没有伪造源码commit。
- benchmark/objectnavtype_val.jsonl.gz直接来自reference且字节一致；SHA256 fa0dffc2dece071716dcf648e19c6944caf1621ee24dadc6996eadcd46eee04a。
- houses val.jsonl.gz与annotations.json.gz均已记录大小/哈希；未遍历哈希全部Objaverse资产，运行期资源一致性仍须核实。

文件哈希不证明checkpoint为论文特定训练run，不证明当前环境已成功加载它。历史86.5%仅historical reference，不充当Reference Run。

## 4. 每项适配的A/B/C分类

A = behavior-preserving with direct evidence（注明证据成立的范围）。
B = likely behavior-preserving but requires runtime verification。
C = behavior-changing / not allowed。

| 项目 | 用途/具体diff | 分类 | Policy/environment行为影响及证据 |
| --- | --- | --- | --- |
| 新增import os；DINO未配置本地路径时的fallback | 读取路径；未配置时仍调用原torch.hub.load(repository, model_type)，随后同一wrapper/.to/BatchNorm/eval | A（代码路径范围） | 未增加forward/随机采样；构造调用参数及后处理保持。不能把fallback静态等价外推成本地分支运行等价。 |
| DINO本地分支 | source=local/pretrained=False创建相同model_type，torch.load(path,map_location=cpu,weights_only=True)，strict=True加载，然后再包装/to(device) | B | 资源来源与加载机制变化；预期仅替代下载，但源码版本、权重键值、dtype、构造RNG消耗和实际输出相等未测试。没有改DINO后的process/normalize/augmentation。 |
| 本地路径成对配置及文件存在检查 | 路径缺失直接报错，避免随机初始化继续 | A（有效输入范围） | 检查不接触策略/环境；无效配置的报错行为有意不同，不能声称异常路径等价。 |
| b0_candidate_env.sh路径选择 | 从脚本位置解析candidate root，将PYTHONPATH只指向candidate；声明B0_PYTHON、data/NLTK/DINO路径 | A（shell只设置变量）+ B（资源解析） | bash -n通过，文件无模型启动指令；实际模块来源、NLTK/资产兼容性须运行验证。B0_PYTHON没有自动替换系统Python，后续必须显式使用。 |
| HF_ENDPOINT/ALLENACT_DEBUG/timeout | 保持官方eval.sh中相同值 | A（配置字面值） | 与reference相同；未设置任何策略参数、deterministic算法或线程优化。 |
| checkpoint路径 | 计划通过官方--ckpt_path传既有绝对路径；不改load_state_dict规则 | A（代码原样）+ B（实际加载） | inference_agent.py整体字节一致，strict=False仍是官方值；本轮未验证missing/unexpected/shape/dtype，不能用旧开发版本加载审计替代候选验证。 |
| provenance/manifest | 在research/b0_candidate_audit独立读文件、Git和API保存证据 | A | 不在agent/worker/controller中加入hook或额外调用；这些是本次审计产物，不冒充runtime manifest。 |
| 开发worker中的额外预决策查询 | 不带入候选 | C/不允许带入（等价性未建立） | candidate worker与官方字节一致，不能因函数名看似读取就认定环境调用无副作用。 |
| greedy history修复、reset/counter修复、shadow、rerank/steering/GRPO/probe等 | 全部不带入 | C/不允许 | 这些超出候选适配范围；无新实现。只排除开发改动，不删除官方原有实现。 |

候选保留的适配中没有C类。原有官方bug/特殊语义不属于本轮修复范围。

## 5. 专项审计

### DINO官方加载与本地加载
reference使用torch.hub.load("facebookresearch/dinov2", model_type)，未固定DINO commit。
本地DINO源码的dinov2/hub/backbones.py显示默认pretrained=True路径先构造模型，再load_state_dict(strict=True)。候选本地分支维持“构造→加载权重→SafeVLA wrapper→设备→eval”的顺序，不沿用开发版本先to(device)后手工加载的顺序。
该源码对应的预期权重URL形式为https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth；本轮未下载或重新证明官方发布权重与本地文件的对应关系。
源码无.git且官方hub ref未固定，因此当前source snapshot可追溯，但不能虚构其等于某个历史DINO commit。保留B类结论；验证需另获允许加载模型，不能靠本轮静态审计升级为A。

### Checkpoint loading
candidate inference_agent.py与reference相同，保留state_dict/model_state_dict/direct state_dict自动识别、原key转换、map_location和strict=False。
没有移植开发版critical-key guard或诊断collector。文件哈希存在不等于候选运行时417/417已验证；后续须在既有加载动作上记录结果，不能为了检查额外forward。

### Stochastic / greedy / history
reference先sample()，再mode()；last_action_flat始终写sampled action；greedy=true时环境使用mode action。
该官方greedy/history不一致原样保留。smoke建议greedy=false，但配置草案不是执行授权。没有修改采样、动作映射或策略模块。

### Evaluator worker与controller
candidate online_evaluation/online_evaluator_worker.py、online_evaluator.py、environment/stretch_controller.py均逐字节等于reference。
无开发分支新增的预决策successful_if_done/dist_to_target_func查询，无shadow_predictor_logger/shadow_safety_predictor/end_causal_diagnostic_logger文件。
官方自身为sensor、task.step、success或metrics所做的controller调用原样保留；不能声称整个官方程序没有controller查询。

### Task success / end / horizon / official metrics
tasks/object_nav_task.py、tasks/abstract_task.py以及task配置、环境和metric计算源码均字节相同。
end分支仍使用官方successful_if_done，未改strict_success或重新计算标签；horizon配置传递不变，不用提前中断episode充当smoke样本限制。
后续--eval_set_size=2限制task数量，--max_eps_len=-1保留官方默认解析。实际task.max_steps/model.max_steps仍须runtime记录，不能把历史600/500当成已实测的候选值。

### Input sensors / augmentation
官方脚本四个输入sensor字符串不变：raw_navigation_camera、raw_manipulation_camera、last_actions、an_object_is_in_hand。
training/online/online_eval.py与inference_agent.py整体未改；test_augmentation和官方v2实现保持。没有新增depth sensor或重新实现图像变换。
注意直接Python入口默认input_sensors只有raw_navigation_camera，后续command必须显式给出上述四项，不能依赖默认值。

### Worker / shuffle / seed
官方scripts/eval.sh保留8 workers、seed=123、shuffle=true、test_augmentation=true默认；本轮没有修改它。
001A后续候选计划使用官方Python入口显式num_workers=1、seed=123、shuffle和test_augmentation开启、eval_set_size=2；这是已标明的小规模smoke协议，不是对8-worker官方脚本默认或历史4-worker full-200的性能复现。
官方minival选择逻辑按seed shuffle，再截取eval_set_size。代码没改，源task ID/实际顺序/动态episode ID配对仍需运行记录。

### Reset / counter
inference_agent.reset()、allenact_dino_transformer.py和decoder源码与reference完全一致。
reset不清time_step_counter/KV cache，单步推理counter递增，达到max_steps或多时间步输入触发归零等官方行为均保留。没有通过clean候选偷偷修reset。

## 6. Launcher与provenance使用边界

scripts/b0_candidate_env.sh只供source来设置路径，不运行模型。
后续从candidate根目录工作，显式使用B0_PYTHON和training/online/online_eval.py；不要source后再调用未经路径替换的官方scripts/eval.sh，因为它仍含/path/to占位符，且不转发eval_set_size。
没有生成自动执行或自动下一轮的launcher。候选smoke参数只保存为smoke_plan_NOT_AUTHORIZED.json，不是已执行command。
provenance静态证据已保存；实际import/module paths、完整实际启动环境、episode IDs/results/logs尚未产生。site-packages的.pth静态检查未发现开发目录字符串，但这不能证明运行时绝无import串线。

## 7. 尚未证明等价的项目

1. 本地DINO源码/权重与选定官方运行依赖的对应关系；实际tensor加载、输出及初始化RNG影响。
2. SafeVLA checkpoint在候选模型上的加载完整性与dtype/shape；保留官方strict=False不代表允许漏加载。
3. 实际Python/import来源、依赖版本及外部数据/资产解析；本轮没有import torch、模型或AI2-THOR。
4. 原始episode结果、稳定task ID/顺序、完整运行manifest的采集路径；未加入诊断collector，须先评审无侵入性的运行期记录方式。
5. 官方DINO hub源码未pin且历史环境信息有限，不能宣称本候选精确还原论文或历史86.5%执行状态。

## 8. 是否允许进入EXP-B0-REPRO-001A smoke？

现在不允许启动：用户明确要求本轮仅候选审计并等待人工评审；001A保持BLOCKED_PREFLIGHT，不自动恢复。
静态层面：独立候选已具备可审阅identity和完整diff，上一轮worker预决策查询差异已从候选排除；未发现带入C类开发改动。
下一项人工决定是是否接受这份reference及A/B适配边界，并批准用限定的001A运行验证B类项目、补齐无侵入runtime产物记录。批准前不能称候选已是runtime-equivalent B0。
即使001A未来通过，也不能自动执行001B或完整200-task。

## 9. 完整 official → candidate diff

以下包含全部tracked变化和新增untracked环境文件；没有省略修改。审计报告与证据保存在原研究目录，不注入候选执行代码，因此不属于候选源码patch。


```diff
diff --git a/architecture/allenact_preprocessors/dino_preprocessors.py b/architecture/allenact_preprocessors/dino_preprocessors.py
index 7bba5a3..083f8f9 100644
--- a/architecture/allenact_preprocessors/dino_preprocessors.py
+++ b/architecture/allenact_preprocessors/dino_preprocessors.py
@@ -1,3 +1,4 @@
+import os
 from typing import Any, Dict, List, Optional, Sequence, Union, cast
 
 import gym
@@ -102,9 +103,25 @@ class DinoViTPreprocessor(Preprocessor):
     @property
     def vit(self) -> DinoViTEmbedder:
         if self._vit is None:
-            self._vit = DinoViTEmbedder(
-                model=torch.hub.load("facebookresearch/dinov2", self.dino_model_type),
-            ).to(self.device)
+            local_repo = os.environ.get("DINOV2_REPO")
+            local_checkpoint = os.environ.get("DINOV2_CKPT")
+            if local_repo is not None or local_checkpoint is not None:
+                if not local_repo or not local_checkpoint:
+                    raise ValueError("Set both DINOV2_REPO and DINOV2_CKPT")
+                if not os.path.isdir(local_repo):
+                    raise FileNotFoundError(local_repo)
+                if not os.path.isfile(local_checkpoint):
+                    raise FileNotFoundError(local_checkpoint)
+                model = torch.hub.load(
+                    local_repo, self.dino_model_type, source="local", pretrained=False
+                )
+                model.load_state_dict(
+                    torch.load(local_checkpoint, map_location="cpu", weights_only=True),
+                    strict=True,
+                )
+            else:
+                model = torch.hub.load("facebookresearch/dinov2", self.dino_model_type)
+            self._vit = DinoViTEmbedder(model=model).to(self.device)
             for module in self._vit.modules():
                 if "BatchNorm" in type(module).__name__:
                     module.momentum = 0.0
diff --git a/scripts/b0_candidate_env.sh b/scripts/b0_candidate_env.sh
new file mode 100644
index 0000000..6fe8168
--- /dev/null
+++ b/scripts/b0_candidate_env.sh
@@ -0,0 +1,14 @@
+#!/usr/bin/env bash
+# Source this file to configure paths only. It does not start an evaluation.
+B0_CANDIDATE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
+export PYTHONPATH="$B0_CANDIDATE_ROOT"
+export B0_PYTHON=/home/amax/.conda/envs/safevla/bin/python
+export OBJAVERSE_HOUSES_DIR=/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28
+export OBJAVERSE_DATA_DIR=/home/amax/public/datasets/qyy/objaverse_assets
+export NLTK_DATA=/home/amax/public/datasets/qyy/nltk_data
+export DINOV2_REPO=/home/amax/.cache/torch/hub/facebookresearch_dinov2_main
+export DINOV2_CKPT=/home/amax/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
+# Same values as the official scripts/eval.sh; no policy flags are set here.
+export HF_ENDPOINT=https://hf-mirror.com
+export ALLENACT_DEBUG=True
+export ALLENACT_DEBUG_VST_TIMEOUT=2000

```
