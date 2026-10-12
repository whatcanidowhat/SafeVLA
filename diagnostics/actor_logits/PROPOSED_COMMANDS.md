# Proposed OFF200 / ON200 commands — NOT EXECUTED

These commands are a PI review deliverable, not execution authorization. Current
GPU_EQUIVALENCE=NOT_RUN_AWAITING_AUTHORIZATION. No online Episode was started.
Before executing either command, obtain independent valid P2 authorization, unique
claim and green control CI, plus a reviewed real GPU logger equivalence PASS on
600 captured identical inputs/states at original atol1e-5. The historical input
SHA256 is `417e684e96563b043138d328776d693f05dc74bb7c074099dea39c3b5881500e`.
The equivalence JSON must contain status=PASS,
scope=REAL_GPU_IDENTICAL_INPUT_LOGGER_EQUIVALENCE, implementation_commit equal to
this clean checkout HEAD, pairs=600, atol=1e-5, and actual supporting evidence.
The entry checks those binding fields; PI must review the underlying evidence and
current control authorization. A manually asserted PASS is not proof.

Shared setup (same fresh shell environment for both independent sessions):

```bash
cd /nvme2/user/qyy/SafeVLA_actor_logits_logger_20261012
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export DINOV2_REPO=/home/amax/.cache/torch/hub/facebookresearch_dinov2_main
export DINOV2_CKPT=/home/amax/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
export OBJAVERSE_HOUSES_DIR=/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_DIR=/home/amax/public/datasets/qyy/objaverse_assets
export NLTK_DATA=/home/amax/public/datasets/qyy/nltk_data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=False
# PI/executor must supply these only after independent approval and GPU validation:
: "${P2_APPROVAL_COMMIT:?independent valid P2 approval required}"
: "${GPU_EQUIVALENCE_REPORT:?reviewed real GPU evidence path required}"
: "${P2_PHYSICAL_GPU:?one approved physical GPU index 0-3 required}"
: "${P2_OUTPUT_ROOT:?fresh external output directory for this approved run required}"
```

OFF200 (proposal only):

```bash
/home/amax/.conda/envs/safevla/bin/python -B -m diagnostics.actor_logits.evaluate \
  --actor-reset off --output "${P2_OUTPUT_ROOT:?}/OFF200" \
  --checkpoint /home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt \
  --physical-gpu "${P2_PHYSICAL_GPU:?}" --num-workers 1 --seed 123 \
  --shuffle --greedy false --test-augmentation --episodes 200 \
  --execute-authorized --approval-reference "${P2_APPROVAL_COMMIT:?}" \
  --gpu-equivalence-report "${GPU_EQUIVALENCE_REPORT:?}"
```

ON200 (proposal only; do not automatically chain after OFF):

```bash
/home/amax/.conda/envs/safevla/bin/python -B -m diagnostics.actor_logits.evaluate \
  --actor-reset on --output "${P2_OUTPUT_ROOT:?}/ON200" \
  --checkpoint /home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt \
  --physical-gpu "${P2_PHYSICAL_GPU:?}" --num-workers 1 --seed 123 \
  --shuffle --greedy false --test-augmentation --episodes 200 \
  --execute-authorized --approval-reference "${P2_APPROVAL_COMMIT:?}" \
  --gpu-equivalence-report "${GPU_EQUIVALENCE_REPORT:?}"
```

Only actor-reset and output name differ. Same implementation/weights, seed,
physical GPU and exact ordered 200 TaskSpecs are required. The runtime independently
regenerates the CSV (SHA256 `8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0`),
then checks the original loader's 200 normalized identities/raw specs before model
build. Its copy is written as TASK_MANIFEST.csv. No alternative tasks or seeds and
no prefix selection. Output must be outside the code checkout to keep it clean.
Original official evaluation output is a separate sibling OFF200_official or
ON200_official directory with the evaluator's timestamped subdirectory.

For CPU plan inspection, omit --execute-authorized, --approval-reference and
--gpu-equivalence-report. This regenerates/compares the manifest and prints original
arguments; it does not import the real evaluator/model, initialize CUDA or create
an output directory. Original max_eps_len=-1 retains ObjectNav's600 decision horizon.

Public GPU queue proposal: inspect GPU0-3 read-only, request/wait for one available
approved slot, then freeze that physical index for both sessions. CUDA exposes only
that device as logical0. Do not kill other users' processes, allocate a probe model,
automatically fall back to another GPU or launch concurrent OFF/ON sessions. Confirm
THOR renderer-to-physical-GPU mapping during independently approved validation;
CUDA_VISIBLE_DEVICES alone is not proof of graphics-device isolation. If no suitable
slot exists, wait or report BLOCKED. Failure ends the session; no automatic retries.

Capacity estimate for400*600=240,000 decisions: three20-action FP32 vectors plus
four scalar int64 values are272 bytes/decision, about65.28 MB total numeric payload.
NPZ headers, per-step metadata, target IDs and hashes dominate overhead. Reserve
several GiB for logger artifacts (initial planning allowance2 GiB, revise from real
authorized profiling); this is an estimate, not a measured real-run bound. Boundary
K/V are hashes only. Original videos/frames and official outputs require separate
PI-approved disk/runtime budgeting. No GPU-hours or live throughput estimate has
been validated in this development task.
