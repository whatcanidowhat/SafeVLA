# Source this file in the GPU container before running scripts/eval.sh.
# NVIDIA userspace graphics libraries match the host driver 535.104.05.
export PATH="/public/home/zhangyh/QYY/vulkan_repair_20261031/tools-1.3.250/bin:/public/home/zhangyh/QYY/envs/vulkan_tools_probe/bin:$PATH"
export LD_LIBRARY_PATH="/public/home/zhangyh/QYY/vulkan_repair_20261031/lib:/public/home/zhangyh/QYY/envs/safevla_native/lib:/public/home/zhangyh/QYY/envs/vulkan_tools_probe/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_DATA_DIRS="/public/home/zhangyh/QYY/vulkan_repair_20261031/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
export SAFEVLA_VULKAN_BACKEND=nvidia
unset VK_ICD_FILENAMES VK_DRIVER_FILES
export __EGL_VENDOR_LIBRARY_FILENAMES=/public/home/zhangyh/QYY/vulkan_repair_20261031/share/glvnd/egl_vendor.d/10_nvidia.json
