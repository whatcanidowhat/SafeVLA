#!/bin/bash
set -euo pipefail
BASE=/public/home/zhangyh/QYY
"$BASE/envs/safevla_native/bin/python" -S "$BASE/unmask_nvidia_20261103/cleanup_stopped_run.py"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
bash "$BASE/unmask_nvidia_20261103/run_pair.sh" --preflight --gpu 0
