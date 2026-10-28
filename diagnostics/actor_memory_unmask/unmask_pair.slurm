#!/bin/bash
# DRAFT executable: sbatch only after independent PI approval + unique Claim CI.
#SBATCH --job-name=unmask-pair500
#SBATCH --account=zhangyh
#SBATCH --qos=user_zhangyh
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/zhangyh/QYY/migration_logs/unmask-pair500-%j.out
#SBATCH --error=/public/home/zhangyh/QYY/migration_logs/unmask-pair500-%j.err
set -Eeuo pipefail
umask 077
ROOT=/public/home/zhangyh/QYY/SafeVLA_actor_memory_unmask_001_20261028
source "$ROOT/diagnostics/actor_memory_unmask/target_environment.sh"
UNMASK_PYTHON="${UNMASK_PYTHON:-$SAFEVLA_TARGET_PYTHON}"
: "${UNMASK_APPROVAL_SHA:?New PI approval SHA required}"
: "${UNMASK_CLAIM_SHA:?New unique Claim SHA required}"
: "${UNMASK_OUTPUT:?New absent output root required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"
# Renderer environment must match the independently reviewed design exactly.
# No silent Mesa/lavapipe switch, scheduler CUDA remapping or HOME override.
cd "$ROOT"
"$UNMASK_PYTHON" -B -m diagnostics.actor_memory_unmask.online_pair \
  --output "$UNMASK_OUTPUT" \
  --checkpoint /public/home/zhangyh/QYY/data/checkpoints/safe_objnav.pt \
  --control-root "$SAFEVLA_CONTROL_ROOT" \
  --approval-reference "$UNMASK_APPROVAL_SHA" --claim-reference "$UNMASK_CLAIM_SHA" \
  --execute-authorized
