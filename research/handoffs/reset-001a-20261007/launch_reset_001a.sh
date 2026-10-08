#!/usr/bin/env bash
set -euo pipefail
cd /nvme2/user/qyy/SafeVLA
source /nvme2/user/qyy/SafeVLA_baseline_clean/scripts/b0_candidate_env.sh
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_NAME=EXP-RESET-001A-capture
export TOKENIZERS_PARALLELISM=False
export RESET_STATE_FIX=0
exec /home/amax/.conda/envs/safevla/bin/python -u research/handoffs/reset-001a-20261007/run_reset_001a.py
