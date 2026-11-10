#!/bin/bash
set -euo pipefail
exec /public/home/zhangyh/QYY/envs/safevla_native/bin/python -S \
  /public/home/zhangyh/QYY/unmask_nvidia_20261103/live_console/main.py "$@"
