#!/bin/bash
set -euo pipefail
source /public/home/zhangyh/QYY/unmask_nvidia_20261103/environment.sh
cd /public/home/zhangyh/QYY/unmask_nvidia_20261103
printf '===REGRESSION-TESTS===\n'
"$SAFEVLA_TARGET_PYTHON" -B -m unittest -v test_output_scan
printf '===PATCHED-PAIR-PLAN-ONLY===\n'
"$SAFEVLA_TARGET_PYTHON" -B pair_entry.py --output /tmp/unmask_plan_only_not_created --checkpoint /public/home/zhangyh/QYY/data/checkpoints/safe_objnav.pt --control-root "$SAFEVLA_CONTROL_ROOT" --approval-reference MANUAL_NVIDIA_NO_FORMAL_APPROVAL --claim-reference MANUAL_NVIDIA_NO_CLAIM
printf '===FROZEN-SOURCE-CHECK===\n'
cd "$SAFEVLA_ROOT"
git status --porcelain
git rev-parse HEAD
printf '===FULL-PREFLIGHT===\n'
bash /public/home/zhangyh/QYY/unmask_nvidia_20261103/run_pair.sh --preflight --gpu 0
