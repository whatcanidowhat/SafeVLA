#!/bin/bash
set -Eeuo pipefail
MODE=preflight
export UNMASK_GPU_INDEX=0
OUT=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run) MODE=run; shift ;;
        --preflight) MODE=preflight; shift ;;
        --gpu) UNMASK_GPU_INDEX="${2:?GPU index required}"; shift 2 ;;
        --output) OUT="${2:?Output path required}"; shift 2 ;;
        *) echo 'Usage: bash run_pair.sh [--preflight|--run] [--gpu 0|1] [--output NEW_ABSOLUTE_PATH]' >&2; exit 2 ;;
    esac
done
[[ "$UNMASK_GPU_INDEX" =~ ^[0-9]+$ ]] || { echo 'GPU index must be nonnegative' >&2; exit 2; }
BASE=/public/home/zhangyh/QYY
source "$BASE/unmask_nvidia_20261103/environment.sh"
export UNMASK_NVIDIA_DEADLINE_UTC
UNMASK_NVIDIA_DEADLINE_UTC=$(date -u -d '+11 hours 50 minutes' +%Y-%m-%dT%H:%M:%SZ)
cd "$SAFEVLA_ROOT"
if [ "$MODE" = preflight ]; then
    exec "$SAFEVLA_TARGET_PYTHON" -B "$BASE/unmask_nvidia_20261103/preflight.py"
fi
umask 077
mkdir -p "$BASE/runs/EXP-ACTOR-MEMORY-UNMASK-NVIDIA" "$BASE/unmask_nvidia_20261103/locks"
exec 9>"$BASE/unmask_nvidia_20261103/locks/$UNMASK_GPU_UUID.lock"
flock -n 9 || { echo 'Another paired run holds this GPU lock' >&2; exit 2; }
if [ -z "$OUT" ]; then
    OUT="$BASE/runs/EXP-ACTOR-MEMORY-UNMASK-NVIDIA/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[[ "$OUT" = /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || {
    echo 'Output must be an absent absolute path with an existing parent directory' >&2; exit 2;
}
echo "PAIR_OUTPUT=$OUT"
exec "$SAFEVLA_TARGET_PYTHON" -B "$BASE/unmask_nvidia_20261103/pair_entry.py" \
    --output "$OUT" --checkpoint "$BASE/data/checkpoints/safe_objnav.pt" \
    --control-root "$SAFEVLA_CONTROL_ROOT" \
    --approval-reference MANUAL_NVIDIA_NO_FORMAL_APPROVAL \
    --claim-reference MANUAL_NVIDIA_NO_CLAIM --execute-authorized
