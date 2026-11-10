#!/bin/bash
UNMASK_HOME=/public/home/zhangyh/QYY/unmask_nvidia_20261103
source /public/home/zhangyh/QYY/SafeVLA_actor_memory_unmask_002_20261029/diagnostics/actor_memory_unmask/target_environment.sh
# Use the frozen experiment Python, with a deterministic NVIDIA graphics prefix.
unset LD_LIBRARY_PATH DISPLAY SDL_VIDEODRIVER QT_QPA_PLATFORM VK_DRIVER_FILES
export XDG_DATA_DIRS=/usr/local/share:/usr/share
source /public/home/zhangyh/QYY/vulkan_repair_20261031/activate.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/public/home/zhangyh/QYY/envs/safevla_native/lib/python3.10/site-packages/nvidia/cudnn/lib:/public/home/zhangyh/QYY/envs/safevla_native/lib/python3.10/site-packages/nvidia/cuda_cupti/lib"
export VK_ICD_FILENAMES=/public/home/zhangyh/QYY/vulkan_repair_20261031/share/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/public/home/zhangyh/QYY/vulkan_repair_20261031/share/glvnd/egl_vendor.d/10_nvidia.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$UNMASK_HOME:$SAFEVLA_ROOT"
export UNMASK_NVIDIA_MANUAL=ACKNOWLEDGED_MANUAL_NVIDIA_RUN
export UNMASK_GPU_INDEX="${UNMASK_GPU_INDEX:-0}"
export UNMASK_GPU_UUID
UNMASK_GPU_UUID=$(nvidia-smi -i "$UNMASK_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$UNMASK_GPU_UUID"
# A private cache prevents sharing AI2-THOR's device-index mapping across nodes.
export HOME="$UNMASK_HOME/runtime/$(hostname)/gpu-$UNMASK_GPU_INDEX"
export XDG_RUNTIME_DIR="$HOME/xdg-runtime"
mkdir -p "$HOME/.ai2thor" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
if [ ! -e "$HOME/.ai2thor/releases" ]; then
    ln -s /public/home/zhangyh/QYY/runtime_home/.ai2thor/releases "$HOME/.ai2thor/releases"
fi
export SDL_AUDIODRIVER=dummy AUDIODEV=null
