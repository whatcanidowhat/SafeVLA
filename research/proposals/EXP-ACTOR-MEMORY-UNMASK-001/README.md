# EXP-ACTOR-MEMORY-UNMASK-001 PI 审核包

状态：CPU_VALIDATED / DRAFT_FOR_PI_REVIEW；GPU/在线未授权，0 Episode。

请按 RESULT_SUMMARY.md → DESIGN.md / DESIGN.json → SOURCE.patch / SOURCE_HASHES.json → cpu-tests-attempt1.log → REPLAY_RESULTS.json / replay-attempt1.log / replay-attempt2.log → RESOURCE_PREFLIGHT.json → RUN_MANIFEST.json → REVIEW_NOTES.md → ARTIFACT_INDEX.json 阅读。

源码位于 diagnostics/actor_memory_unmask/，唯一生产文件变化位于 architecture/models/allenact_transformer_models/allenact_dino_transformer.py。历史实验文件和控制状态均保留。

复现CPU（同源worktree，输出目录必须全新）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/amax/.conda/envs/safevla/bin/python -B -m diagnostics.actor_memory_unmask.test_cpu
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/amax/.conda/envs/safevla/bin/python -B -m diagnostics.actor_memory_unmask.replay --checkpoint /home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt --output /NEW/ABSOLUTE/REPLAY_DIR
```

没有GPU执行授权或可执行线上Claim；不要运行历史P2 launcher。在线资源门、source pin与新入口仍待独立PI决定。
