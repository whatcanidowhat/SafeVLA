# Frozen target paths; source only. No job/model/simulator or GPU initialization.
export SAFEVLA_ROOT=/public/home/zhangyh/QYY/SafeVLA_actor_memory_unmask_001_20261028
export SAFEVLA_CONTROL_ROOT=/public/home/zhangyh/QYY/SafeVLA_research_control_20261028
export SAFEVLA_DATA_ROOT=/public/home/zhangyh/QYY/data
export SAFEVLA_TARGET_PYTHON=/public/home/zhangyh/QYY/envs/safevla_unmask500_20261028/bin/python
export PATH=/public/home/zhangyh/QYY/envs/git_unmask_20261028/bin:$PATH
export DINOV2_REPO=/public/home/zhangyh/QYY/cache/torch/hub/facebookresearch_dinov2_main
export DINOV2_CKPT=/public/home/zhangyh/QYY/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
export OBJAVERSE_HOUSES_DIR=/public/home/zhangyh/QYY/data/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_DIR=/public/home/zhangyh/QYY/data/objaverse_assets
export NLTK_DATA=/public/home/zhangyh/QYY/data/nltk_data
export TORCH_HOME=/public/home/zhangyh/QYY/cache/torch
export HF_HOME=/public/home/zhangyh/QYY/cache/huggingface
export HF_HUB_CACHE=/public/home/zhangyh/QYY/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/public/home/zhangyh/QYY/cache/huggingface
export TRANSFORMERS_CACHE=/public/home/zhangyh/QYY/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
# Keep scheduler CUDA mapping and HOME. Renderer variables require PI review.
