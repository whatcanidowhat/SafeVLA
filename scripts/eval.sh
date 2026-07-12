#!/bin/bash
export NLTK_DATA="/home/amax/public/datasets/qyy/nltk_data"
export PYTHONPATH=/nvme2/user/qyy/SafeVLA  # change to your own path
export OBJAVERSE_HOUSES_DIR=/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28  # change to your own path
export OBJAVERSE_DATA_DIR=/home/amax/public/datasets/qyy/objaverse_assets  # change to your own path
export HF_ENDPOINT=https://hf-mirror.com
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000

# Default values
task_type=""
ckpt_path=""
eval_subset="minival"
output_basedir="./eval"
num_workers=1
seed=123
shuffle=flase #固定一下每次跑的任务顺序
test_augmentation=true
house_set="objaverse"
input_sensors="raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand "
greedy=true
eval_set_size=""

# GPU wait defaults — aggressive grab: poll fast, claim on first idle, hold large mem.
# Override from the command line environment when needed.
WAIT_FOR_GPU=${WAIT_FOR_GPU:-1}
GPU_WAIT_INTERVAL=${GPU_WAIT_INTERVAL:-2}
# Idle threshold: GPU is claimable when no foreign compute process AND
# memory/util are below these caps. Too strict -> wait forever on residual usage;
# too loose -> may race a nearly-full GPU. V100 32GB: 4GB / 30% is a practical default.
GPU_WAIT_MAX_MEM_MB=${GPU_WAIT_MAX_MEM_MB:-4000}
GPU_WAIT_MAX_UTIL=${GPU_WAIT_MAX_UTIL:-30}
GPU_WAIT_STABLE_CHECKS=${GPU_WAIT_STABLE_CHECKS:-1}
GPU_WAIT_STATUS_EVERY=${GPU_WAIT_STATUS_EVERY:-1}

# GPU hold (占卡): reserve enough memory + light compute so others' wait scripts skip us.
GPU_HOLD_ENABLED=${GPU_HOLD_ENABLED:-1}
GPU_HOLD_MEM_MB=${GPU_HOLD_MEM_MB:-4096}
GPU_HOLD_PID=""

list_candidate_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '
    else
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null
    fi
}

gpu_query_value() {
    local gpu_id="$1"
    local field="$2"
    nvidia-smi -i "$gpu_id" --query-gpu="$field" --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

# True if GPU has compute apps other than our own hold process.
gpu_has_foreign_compute_process() {
    local gpu_id="$1"
    local pid
    local pids

    pids=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [ -z "$pids" ] && return 1

    for pid in $(echo "$pids" | tr ',' ' '); do
        [ -z "$pid" ] && continue
        if [ -n "$GPU_HOLD_PID" ] && [ "$pid" = "$GPU_HOLD_PID" ]; then
            continue
        fi
        return 0
    done
    return 1
}

gpu_is_idle() {
    local gpu_id="$1"
    local mem_used
    local util
    local hold_mem_adjust=0

    if gpu_has_foreign_compute_process "$gpu_id"; then
        return 1
    fi

    mem_used=$(gpu_query_value "$gpu_id" "memory.used")
    util=$(gpu_query_value "$gpu_id" "utilization.gpu")

    if [ -z "$mem_used" ] || [ -z "$util" ]; then
        return 1
    fi

    # Ignore our own hold reservation when re-checking the selected GPU.
    if [ -n "$GPU_HOLD_PID" ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$gpu_id" ]; then
        hold_mem_adjust=$GPU_HOLD_MEM_MB
        util=0
    fi

    [ "$((mem_used - hold_mem_adjust))" -le "$GPU_WAIT_MAX_MEM_MB" ] && [ "$util" -le "$GPU_WAIT_MAX_UTIL" ]
}

print_gpu_wait_status() {
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo "Current compute processes:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
}

print_gpu_poll_summary() {
    local candidate_gpus="$1"
    local gpu_id
    local mem_used
    local util
    local status
    local summary=""

    for gpu_id in $candidate_gpus; do
        mem_used=$(gpu_query_value "$gpu_id" "memory.used")
        util=$(gpu_query_value "$gpu_id" "utilization.gpu")
        if gpu_is_idle "$gpu_id"; then
            status="idle"
        elif gpu_has_foreign_compute_process "$gpu_id"; then
            status="busy(compute)"
        else
            status="busy(mem/util)"
        fi
        summary="${summary} GPU${gpu_id}=${status} mem=${mem_used:-?}MB util=${util:-?}%"
    done

    echo "[$(date '+%F %T')] polling:${summary}"
}

release_gpu_hold() {
    if [ -n "$GPU_HOLD_PID" ]; then
        kill "$GPU_HOLD_PID" 2>/dev/null || true
        wait "$GPU_HOLD_PID" 2>/dev/null || true
        GPU_HOLD_PID=""
        echo "Released GPU hold process."
    fi
}

hold_selected_gpu() {
    local ready_file
    local waited

    if [ "$WAIT_FOR_GPU" != "1" ] || [ "$GPU_HOLD_ENABLED" != "1" ]; then
        return 0
    fi

    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        return 0
    fi

    release_gpu_hold

    ready_file=$(mktemp /tmp/safevla_gpu_hold_XXXXXX)
    rm -f "$ready_file"

    echo "Holding GPU ${CUDA_VISIBLE_DEVICES} aggressively (占卡, ${GPU_HOLD_MEM_MB}MB + light compute)..."
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -c "
import os
import signal
import time

import torch

def _exit_handler(*_):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _exit_handler)
signal.signal(signal.SIGINT, _exit_handler)

device = torch.device('cuda:0')
mem_mb = ${GPU_HOLD_MEM_MB}
num_floats = max(1, int(mem_mb * 1024 * 1024 / 4))
hold = torch.zeros(num_floats, device=device)
# Touch memory so allocation is committed immediately.
hold.fill_(1)
torch.cuda.synchronize()
ready_path = '${ready_file}'
with open(ready_path, 'w', encoding='utf-8') as f:
    f.write(f'{os.getpid()}\\n')
print(f'GPU hold active on cuda:0 ({mem_mb}MB reserved, pid={os.getpid()})', flush=True)

# Keep mild utilization so util-based waiters also skip this GPU.
x = torch.randn(1024, 1024, device=device)
while True:
    x = x @ x
    x = x / (x.norm() + 1e-6)
    torch.cuda.synchronize()
    time.sleep(0.05)
" &
    GPU_HOLD_PID=$!
    echo "GPU hold process PID: $GPU_HOLD_PID"

    # Wait briefly until CUDA allocation succeeds; otherwise keep searching.
    waited=0
    while [ "$waited" -lt 30 ]; do
        if [ -f "$ready_file" ] && kill -0 "$GPU_HOLD_PID" 2>/dev/null; then
            rm -f "$ready_file"
            echo "GPU ${CUDA_VISIBLE_DEVICES} hold confirmed."
            return 0
        fi
        if ! kill -0 "$GPU_HOLD_PID" 2>/dev/null; then
            break
        fi
        sleep 0.2
        waited=$((waited + 1))
    done

    rm -f "$ready_file"
    echo "Warning: GPU hold failed to confirm on ${CUDA_VISIBLE_DEVICES}; will keep waiting."
    release_gpu_hold
    return 1
}

cleanup_on_exit() {
    release_gpu_hold
}
on_interrupt() {
    echo "Interrupted. Stopping GPU wait..."
    release_gpu_hold
    exit 130
}
trap cleanup_on_exit EXIT
trap on_interrupt INT TERM

wait_for_free_gpu() {
    local stable_gpu=""
    local stable_count=0
    local candidate_gpus
    local gpu_id
    local loops_since_status=0
    local best_gpu=""
    local best_mem=""
    local mem_used

    release_gpu_hold

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "Error: WAIT_FOR_GPU=1 but nvidia-smi was not found."
        exit 1
    fi

    candidate_gpus=$(list_candidate_gpus)
    if [ -z "$candidate_gpus" ]; then
        echo "Error: no candidate GPUs found."
        exit 1
    fi

    echo "Waiting for an idle GPU from candidates: $candidate_gpus"
    echo "Idle threshold: no foreign compute process, memory <= ${GPU_WAIT_MAX_MEM_MB}MB, util <= ${GPU_WAIT_MAX_UTIL}%"
    echo "Aggressive grab: stable_checks=${GPU_WAIT_STABLE_CHECKS}, interval=${GPU_WAIT_INTERVAL}s, hold=${GPU_HOLD_MEM_MB}MB"

    while true; do
        best_gpu=""
        best_mem=""

        # Prefer the freest idle GPU in this poll (lowest memory.used).
        for gpu_id in $candidate_gpus; do
            if gpu_is_idle "$gpu_id"; then
                mem_used=$(gpu_query_value "$gpu_id" "memory.used")
                if [ -z "$best_gpu" ] || [ "$mem_used" -lt "$best_mem" ]; then
                    best_gpu="$gpu_id"
                    best_mem="$mem_used"
                fi
            fi
        done

        if [ -n "$best_gpu" ]; then
            if [ "$stable_gpu" = "$best_gpu" ]; then
                stable_count=$((stable_count + 1))
            else
                stable_gpu="$best_gpu"
                stable_count=1
            fi

            echo "GPU $best_gpu idle check ${stable_count}/${GPU_WAIT_STABLE_CHECKS} (mem=${best_mem}MB)"
            if [ "$stable_count" -ge "$GPU_WAIT_STABLE_CHECKS" ]; then
                export CUDA_VISIBLE_DEVICES="$best_gpu"
                echo "Selected GPU $best_gpu. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
                if hold_selected_gpu; then
                    return 0
                fi
                # Hold failed (race); reset and keep polling immediately.
                stable_gpu=""
                stable_count=0
                continue
            fi
        else
            stable_gpu=""
            stable_count=0
            print_gpu_poll_summary "$candidate_gpus"
            loops_since_status=$((loops_since_status + 1))
            if [ "$loops_since_status" -ge "$GPU_WAIT_STATUS_EVERY" ]; then
                print_gpu_wait_status
                loops_since_status=0
            fi
        fi

        sleep "$GPU_WAIT_INTERVAL"
    done
}

ensure_selected_gpu_is_still_idle() {
    local selected_gpu="$CUDA_VISIBLE_DEVICES"

    if [ "$WAIT_FOR_GPU" != "1" ]; then
        return 0
    fi

    # Already holding: treat as successfully claimed.
    if [ -n "$GPU_HOLD_PID" ] && kill -0 "$GPU_HOLD_PID" 2>/dev/null; then
        return 0
    fi

    if [ -z "$selected_gpu" ]; then
        wait_for_free_gpu
        return 0
    fi

    if gpu_is_idle "$selected_gpu"; then
        # Re-assert hold if it was released earlier.
        if [ "$GPU_HOLD_ENABLED" = "1" ] && [ -z "$GPU_HOLD_PID" ]; then
            hold_selected_gpu || wait_for_free_gpu
        fi
        return 0
    fi

    echo "Selected GPU $selected_gpu became busy before eval started. Waiting again."
    wait_for_free_gpu
}

# Function to print usage
print_usage() {
    echo "Usage: $0 --task_type <type> --ckpt_path <path> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --task_type           Task type: objectnav | pickup | fetch"
    echo "  --ckpt_path           Path to checkpoint file"
    echo ""
    echo "Optional arguments:"
    echo "  --eval_subset         Evaluation subset (default: minival)"
    echo "  --output_basedir      Output base directory (default: ./eval)"
    echo "  --num_workers         Number of workers (default: 8)"
    echo "  --seed                Random seed (default: 123)"
    echo "  --no_shuffle          Disable shuffling (default: enabled)"
    echo "  --no_test_augmentation Disable test augmentation (default: enabled)"
    echo "  --house_set           House set to use (default: objaverse)"
    echo "  --input_sensors       Input sensors (default: raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand)"
    echo ""
    echo "Environment variables:"
    echo "  CUDA_VISIBLE_DEVICES  Candidate GPU list, e.g. 0,1,2,3"
    echo "  WAIT_FOR_GPU=0        Disable GPU waiting and keep the previous behavior"
    echo "  GPU_WAIT_INTERVAL     Seconds between idle checks (default: 2)"
    echo "  GPU_WAIT_MAX_MEM_MB   Max used memory to treat GPU as idle (default: 4000)"
    echo "  GPU_WAIT_MAX_UTIL     Max GPU util %% to treat GPU as idle (default: 30)"
    echo "  GPU_WAIT_STABLE_CHECKS Consecutive idle polls before claim (default: 1)"
    echo "  GPU_WAIT_STATUS_EVERY Full nvidia-smi dump every N polls (default: 1)"
    echo "  GPU_HOLD_ENABLED=0    Disable GPU hold/占卡 after selection"
    echo "  GPU_HOLD_MEM_MB       Memory to reserve while holding GPU (default: 4096)"
    echo "  --help                Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_type)
            task_type="$2"
            shift 2
            ;;
        --ckpt_path)
            ckpt_path="$2"
            shift 2
            ;;
        --eval_subset)
            eval_subset="$2"
            shift 2
            ;;
        --output_basedir)
            output_basedir="$2"
            shift 2
            ;;
        --num_workers)
            num_workers="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --no_shuffle)
            shuffle=false
            shift
            ;;
            --no_test_augmentation)
            test_augmentation=false
            shift
            ;;
        --greedy)
            greedy=true
            shift
            ;;
        --eval_set_size)
            eval_set_size="$2"
            shift 2
            ;;
        --house_set)
            house_set="$2"
            shift 2
            ;;
        --input_sensors)
            input_sensors="$2"
            shift 2
            ;;
        --help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Check required arguments
if [ -z "$task_type" ] || [ -z "$ckpt_path" ]; then
    echo "Error: Missing required arguments"
    echo ""
    print_usage
fi

# Convert task type to internal format
if [ "$task_type" == "objectnav" ]; then
    task_type_internal="ObjectNavType"
elif [ "$task_type" == "pickup" ]; then
    task_type_internal="PickupType"
elif [ "$task_type" == "fetch" ]; then
    task_type_internal="FetchType"
else
    echo "Error: Invalid task type '$task_type'"
    echo "Valid options: objectnav, pickup, fetch"
    exit 1
fi

# Used by shadow/GRPO HeuristicSafetyPredictor nav_only gating (ObjectNav).
export TASK_TYPE_INTERNAL=$task_type_internal

# B1 shadow logging (optional; does not change baseline actions):
#   SHADOW_PREDICTOR_ENABLE=1 \
#   SHADOW_PREDICTOR_LOGDIR=./eval/ObjectNavType/B1-shadow/shadow_logs \
#   WANDB_NAME=v100safevla-B1-shadow \
#   bash scripts/eval.sh --task_type objectnav --ckpt_path ... --seed 123
# Then:
#   python scripts/analyze_shadow_log.py --logdir ./eval/ObjectNavType/B1-shadow/shadow_logs
# Optional: add --greedy to online_eval via editing cmd, or pass through python directly.

if [ "$WAIT_FOR_GPU" = "1" ]; then
    wait_for_free_gpu
fi

# Build the command
cmd="python training/online/online_eval.py"

# Add shuffle flag if enabled
if [ "$shuffle" = true ]; then
    cmd="$cmd --shuffle"
fi

# Add test_augmentation flag if enabled
if [ "$test_augmentation" = true ]; then
    cmd="$cmd --test_augmentation"
fi

# Add greedy flag if enabled (recommended for B1 shadow A/B)
if [ "$greedy" = true ]; then
    cmd="$cmd --greedy"
fi

# Add other parameters
cmd="$cmd \
    --eval_subset $eval_subset \
    --output_basedir $output_basedir/$task_type_internal \
    --task_type $task_type_internal \
    --input_sensors $input_sensors \
    --house_set $house_set \
    --num_workers $num_workers \
    --seed $seed \
    --ckpt_path $ckpt_path"

if [ -n "$eval_set_size" ]; then
    cmd="$cmd --eval_set_size $eval_set_size"
fi

# Execute the command
echo "Executing command:"
echo "$cmd"
echo ""
ensure_selected_gpu_is_still_idle
release_gpu_hold
eval $cmd
