#!/usr/bin/env bash
# P2-004 one-shot ON200 with immutable historical OFF reference. Final PI authorization required.
set -Eeuo pipefail
unset ALL_PROXY all_proxy GIT_SSL_NO_VERIFY
export HTTPS_PROXY=http://127.0.0.1:7899 HTTP_PROXY=http://127.0.0.1:7899
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export NO_PROXY="" no_proxy=""
umask 077
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
CONTROL_ROOT=$(printenv P2_CONTROL_ROOT || printf '%s' /nvme2/user/qyy/SafeVLA_loop_control_p2_codex_20261013)
OUTPUT_ROOT=/nvme2/user/qyy/SafeVLA_p2_004_on200_results_20261015
EVIDENCE_ROOT=/nvme2/user/qyy/SafeVLA_p2_004_launch_20261015
P2_PYTHON=$(printenv P2_PYTHON || printf '%s' /home/amax/.conda/envs/safevla/bin/python)
PHASE_PID=""
stop_own_process() {
  if [[ -n "$PHASE_PID" ]]; then
    kill -TERM "$PHASE_PID" 2>/dev/null || true
    wait "$PHASE_PID" 2>/dev/null || true
    PHASE_PID=""
  fi
}
trap 'stop_own_process; exit 130' INT TERM
die(){ printf 'STOP: %s\n' "$*" >&2; exit 1; }
[[ -x "$P2_PYTHON" ]] || die "Python executable not found"
[[ -f "$REPO_ROOT/diagnostics/actor_logits/evaluate.py" ]] || die "Wrong source tree"
[[ "$REPO_ROOT" != "$CONTROL_ROOT" ]] || die "Never execute from control checkout"
[[ ! -e "$OUTPUT_ROOT" ]] || die "Output directory already exists. No overwrite or retry."
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || die "Source tree dirty"
[[ -r "$CONTROL_ROOT/research/P2_CODE_SHA.txt" ]] || die "Missing frozen source SHA in PI control"
CODE_SHA=$(tr -d '[:space:]' < "$CONTROL_ROOT/research/P2_CODE_SHA.txt")
[[ "$CODE_SHA" =~ ^[0-9a-f]{40}$ ]] || die "Invalid approved source SHA"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$CODE_SHA" ]] || die "P2 source checkout is not the PI-frozen commit"
[[ "$(git -C "$CONTROL_ROOT" branch --show-current)" == "research-loop" ]] || die "Wrong control checkout"
[[ -z "$(git -C "$CONTROL_ROOT" status --porcelain)" ]] || die "Dirty control checkout"
export P2_CONTROL_ROOT="$CONTROL_ROOT" P2_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONPATH="$REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"
[[ ! -e "$EVIDENCE_ROOT" ]] || die "Launch evidence exists; no automatic retry"
mkdir -m 700 "$EVIDENCE_ROOT" || die "Cannot create unique launch evidence"
# Strict live authorization, original OFF bytes and three proxy network probes, CPU only.
CUDA_VISIBLE_DEVICES="" "$P2_PYTHON" -B -m diagnostics.actor_logits.recovery003 preflight \
  --control-root "$CONTROL_ROOT" --report "$EVIDENCE_ROOT/NETWORK_PREFLIGHT.json" \
  || die "P2-004 CPU control/network/OFF gate failed"
# Query the actual GitHub Actions run for the exact CODEX claim; never trust a shell boolean.
CLAIM_CONTEXT=$("$P2_PYTHON" - <<'PY'
import json, os, subprocess, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from diagnostics.actor_logits.recovery003 import PROXY_URL, transport_environment, github_opener
p = Path(os.environ["P2_CONTROL_ROOT"])
def git(*args):
    return subprocess.check_output(["git", "-c", "http.proxy="+PROXY_URL, "-c", "http.sslVerify=true", "-C", str(p), *args], text=True, timeout=60, env=transport_environment()).strip()
s = json.loads((p/"research/LOOP_STATE.json").read_text())
sha = git("rev-parse", "HEAD")
remote = git("ls-remote", "--heads", "origin", "refs/heads/research-loop").split()
if not remote or remote[0] != sha: raise SystemExit("Remote claim HEAD mismatch")
if s.get("status") != "CODEX_RUNNING" or s.get("next_actor") != "CODEX" or s.get("updated_by") != "CODEX":
    raise SystemExit("Missing CODEX_RUNNING claim")
if s.get("experiment_id") != "EXP-ACTOR-RESET-ON200-004" or s.get("cycle_id") != "actor-reset-on200-off002-reuse-004-20261015":
    raise SystemExit("Wrong P2 experiment")
if not s.get("instruction_commit") or not s.get("claim_id"): raise SystemExit("Missing claim identity")
a = s["authorization"]
if a.get("status") != "APPROVED" or a.get("max_gpu") != 1 or a.get("max_episodes") != 200:
    raise SystemExit("Wrong PI authorization/budget")
expires = datetime.fromisoformat(a["expires_at_utc"].replace("Z","+00:00"))
if expires <= datetime.now(timezone.utc): raise SystemExit("Authorization expired")
d = json.loads((p/"research/NEXT_EXPERIMENT.json").read_text())
if d.get("status") != "APPROVED" or d.get("command") != ["bash","diagnostics/actor_logits/run_p2_on200_only.sh"]:
    raise SystemExit("Different/unapproved research command")
request = urllib.request.Request(
    "https://api.github.com/repos/whatcanidowhat/SafeVLA/actions/runs?head_sha="+sha+"&per_page=20",
    headers={"Accept": "application/vnd.github+json", "User-Agent": "SafeVLA-codex-executor"})
try:
    with github_opener().open(request, timeout=25) as response:
        runs = json.load(response).get("workflow_runs", [])
except Exception as e:
    raise SystemExit("Cannot verify claim CI over GitHub API; STOP: "+type(e).__name__)
if not any(r.get("head_sha")==sha and r.get("name")=="research-loop-validate"
           and r.get("status")=="completed" and r.get("conclusion")=="success" and r.get("event")=="push"
           for r in runs):
    raise SystemExit("Distinct CODEX claim CI not SUCCESS")
print(s["instruction_commit"], sha, int(expires.timestamp()), sep=" ")
PY
) || die "Control claim or CI check failed; no GPU initialized"
read -r APPROVAL_SHA CLAIM_SHA EXPIRES_EPOCH <<< "$CLAIM_CONTEXT"
[[ "$APPROVAL_SHA" =~ ^[a-f0-9]{40}$ && "$CLAIM_SHA" =~ ^[a-f0-9]{40}$ ]] || die "Invalid commit identities"
export PYTHONPATH="$REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
export DINOV2_REPO=/home/amax/.cache/torch/hub/facebookresearch_dinov2_main
export DINOV2_CKPT=/home/amax/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
export OBJAVERSE_HOUSES_DIR=/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_DIR=/home/amax/public/datasets/qyy/objaverse_assets
export NLTK_DATA=/home/amax/public/datasets/qyy/nltk_data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
command -v timeout >/dev/null || die "GNU timeout unavailable"
# CPU-only shared GPU queue. Conservative no-foreign-compute and memory guards.
selected="" stable="" stable_count=0
queue_start=$(date -u +%s)
while true; do
  now=$(date -u +%s)
  (( now < EXPIRES_EPOCH )) || die "PI authorization expired during queue"
  (( now - queue_start < 43200 )) || die "12h GPU queue deadline"
  best="" best_used=9999999 best_uuid=""
  for gpu in 0 1 2 3; do
    row=$(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || continue
    IFS=, read -r uuid used free util <<< "$row"
    uuid=$(echo "$uuid" | tr -d '[:space:]')
    used=$(echo "$used" | tr -d '[:space:]')
    free=$(echo "$free" | tr -d '[:space:]')
    util=$(echo "$util" | tr -d '[:space:]')
    [[ "$used" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || continue
    compute=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null) || continue
    [[ -z "$(echo "$compute" | tr -d '[:space:]')" ]] || continue
    if (( used <= 4000 && free >= 6824 && util <= 30 && used < best_used )); then
      best="$gpu"; best_used="$used"; best_uuid="$uuid"
    fi
  done
  if [[ -n "$best" && "$best" == "$stable" ]]; then
    stable_count=$((stable_count + 1))
  elif [[ -n "$best" ]]; then
    stable="$best"; stable_count=1
  else
    stable=""; stable_count=0
  fi
  printf '[%s] GPU queue candidate=%s stable=%s/3\n' "$(date -u +%FT%TZ)" "$stable" "$stable_count"
  if (( stable_count >= 3 )); then selected="$stable"; selected_uuid="$best_uuid"; break; fi
  sleep 30
done
# Last check before any GPU work; never kill another user's process.
row=$(nvidia-smi -i "$selected" --query-gpu=uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits) || die "Selected GPU vanished"
IFS=, read -r uuid used free util <<< "$row"
uuid=$(echo "$uuid" | tr -d '[:space:]')
used=$(echo "$used" | tr -d '[:space:]')
free=$(echo "$free" | tr -d '[:space:]')
util=$(echo "$util" | tr -d '[:space:]')
compute=$(nvidia-smi -i "$selected" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null) || die "Cannot inspect compute apps"
[[ "$uuid" == "$selected_uuid" && -z "$(echo "$compute" | tr -d '[:space:]')" ]] || die "GPU UUID changed or foreign process started"
[[ "$used" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || die "Malformed GPU metrics"
(( used <= 4000 && free >= 6824 && util <= 30 )) || die "GPU capacity race"
export CUDA_VISIBLE_DEVICES="$selected"
printf 'Chosen physical GPU %s UUID %s; logical cuda:0, CODEX claim %s\n' "$selected" "$selected_uuid" "$CLAIM_SHA"
mkdir -m 700 "$OUTPUT_ROOT" || die "Cannot create fresh results root"
cat > "$OUTPUT_ROOT/P2_LAUNCH_CONTEXT.json" <<EOF
{"approval_commit":"$APPROVAL_SHA","codex_claim_commit":"$CLAIM_SHA","physical_gpu":$selected,"gpu_uuid":"$selected_uuid","gpu_logger_equivalence":"NOT_RUN_PI_WAIVED","implementation_commit":"$(git -C "$REPO_ROOT" rev-parse HEAD)","planned_groups":["ON200"],"OFF_reference":"/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200","outcome":"RUNNING"}
EOF
GPU_TIME_START=$(date -u +%s)
run_phase(){
  local treatment="$1" label="$2"
  local now remaining disk pid
  now=$(date -u +%s)
  (( now < EXPIRES_EPOCH )) || die "Approval expired before $label"
  remaining=$(( 604800 - (now - GPU_TIME_START) ))
  (( remaining > 0 )) || die "168h cumulative runtime exhausted"
  printf 'START %s on physical GPU %s at %s\n' "$label" "$selected" "$(date -u +%FT%TZ)"
  timeout -s TERM -k 60 "$remaining" "$P2_PYTHON" -B -m diagnostics.actor_logits.evaluate \
    --actor-reset "$treatment" --output "$OUTPUT_ROOT/$label" \
    --checkpoint /home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt \
    --physical-gpu "$selected" --num-workers 1 --seed 123 --shuffle \
    --greedy false --test-augmentation --episodes 200 \
    --execute-authorized --approval-reference "$APPROVAL_SHA" \
    --claim-reference "$CLAIM_SHA" --control-root "$CONTROL_ROOT" \
    --pi-waive-gpu-equivalence > "$OUTPUT_ROOT/$label.console.log" 2>&1 &
  PHASE_PID=$!
  while kill -0 "$PHASE_PID" 2>/dev/null; do
    sleep 30
    now=$(date -u +%s)
    disk=$(du -sb "$OUTPUT_ROOT" | awk '{print $1}')
    if (( disk >= 96636764160 || now >= EXPIRES_EPOCH || now - GPU_TIME_START >= 604800 )); then
      printf 'STOP resource/deadline guard during %s disk=%s\n' "$label" "$disk" >&2
      stop_own_process
      die "$label resource ceiling reached, partial evidence preserved"
    fi
  done
  pid="$PHASE_PID"; PHASE_PID=""
  wait "$pid" || die "$label failed, see console log; later group NOT_STARTED"
  "$P2_PYTHON" -B -m diagnostics.actor_logits.validate "$OUTPUT_ROOT/$label" \
    --expected 200 --report "$OUTPUT_ROOT/$label.validation.json" || die "$label incomplete; later group NOT_STARTED"
  printf 'COMPLETE/VALIDATED %s at %s\n' "$label" "$(date -u +%FT%TZ)"
}
cd "$REPO_ROOT"
run_phase on ON200
"$P2_PYTHON" -B -m diagnostics.actor_logits.recovery003 compare \
  --control-root "$CONTROL_ROOT" --report "$OUTPUT_ROOT/CROSS_CYCLE_COMPARISON.json" \
  || die "Cross-cycle comparison validation failed"
printf 'P2-004 ON COMPLETE; immutable OFF compared. Outputs: %s\n' "$OUTPUT_ROOT"
# User must inspect logs and publish actual result handoff for PI review; no auto reruns.
