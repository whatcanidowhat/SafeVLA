# SafeVLA Vulkan repair (2026-10-31)

This bundle supplies user-space NVIDIA graphics libraries matching the allocated
host driver 535.104.05, an explicit Vulkan ICD discovery directory, EGL vendor
configuration, and the conda-forge Vulkan Tools 1.3.250 executable. The newer
Vulkan Tools 1.4.357 executable failed while querying headless display interfaces.
The original repository and system drivers were not edited.

Use in the GPU container:

```bash
# One navigation task:
bash /public/home/zhangyh/QYY/vulkan_repair_20261031/run_objectnav.sh --eval_set_size 1 --num_workers 1 --seed 123

# Current complete local minival navigation benchmark (200 tasks):
bash /public/home/zhangyh/QYY/vulkan_repair_20261031/run_objectnav.sh --num_workers 1 --seed 123
```

Alternatively, source activate.sh in the current shell, then run the repository's
scripts/eval.sh normally. The launcher checks the host driver version before use.

Vulkan validation log: vulkaninfo.log (real GPU names and UUIDs, exit code 0).
Single-task evaluation log: smoke.log.
Evaluation smoke outputs: smoke_output/.

Sources:
- https://download.nvidia.com/XFree86/Linux-x86_64/535.104.05/
- https://conda.anaconda.org/conda-forge/linux-64/vulkan-tools-1.3.250-hac7e632_1.conda

Both downloaded archives were verified against their published SHA-256 values.
Only selected graphics libraries are exposed through lib/. System kernel modules,
system library directories, and the existing SafeVLA Conda environment were not changed.

## Verified in container 6d4cebf35794 (Slurm job 247600, gpu119)

- Full vulkaninfo: exit 0; both V100 UUIDs match nvidia-smi.
- Navigation evaluation: one task processed, video and aggregate results saved, SMOKE_EXIT=0.
- The model did not complete the target in this single episode; this check verifies execution, not model performance.
