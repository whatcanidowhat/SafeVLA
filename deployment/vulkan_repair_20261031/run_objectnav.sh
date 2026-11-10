#!/bin/bash
set -euo pipefail
source /public/home/zhangyh/QYY/vulkan_repair_20261031/activate.sh
runtime_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u)
if [ "$runtime_driver" != '535.104.05' ]; then
    printf 'This graphics-library bundle requires host driver 535.104.05; detected: %s\n' "$runtime_driver" >&2
    exit 2
fi
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
cd /public/home/zhangyh/QYY/SafeVLA
exec bash scripts/eval.sh --task_type objectnav --ckpt_path /public/home/zhangyh/QYY/data/checkpoints/safe_objnav.pt "$@"
