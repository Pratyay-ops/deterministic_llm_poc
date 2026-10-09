#!/usr/bin/env bash
# Serve an LLM deterministically on the GB10: vLLM's batch-invariant mode with the settings validated in
# experiment_results/2026-10-08_gb10 (Qwen3-8B and Qwen3-30B-A3B at temperature 0: every completion of
# 8 prompts x 100-200 identical requests bitwise identical while one server handled all prompts at once).
#
#   bash scripts/gb10/serve_deterministic.sh [start]   # OpenAI-compatible API on http://127.0.0.1:8000/v1
#   bash scripts/gb10/serve_deterministic.sh status
#   bash scripts/gb10/serve_deterministic.sh stop      # stop and remove the container
#   python3 scripts/gb10/check_determinism.py          # verify: same prompt N times at once -> one answer
#
# Works from the repo (weights from ~/.cache/huggingface) and from a saved package, where it uses
# ./model/Qwen3-8B and loads ./image/vllm-openai-v0.31.0.tar.zst if the image is missing.
#
# Environment:
#   MODEL_DIR      local weights directory (default: ./model/Qwen3-8B next to this script, if present)
#   MODEL          HF repo id when there is no MODEL_DIR (default Qwen/Qwen3-8B, from the HF cache)
#   REVISION       HF revision to serve (default for Qwen/Qwen3-8B: the validated b968826d9c46...)
#   SERVED_NAME    model name for clients (default qwen3-8b-deterministic)
#   PORT, BIND     host port and address (default 8000 on 127.0.0.1; see API_KEY before using 0.0.0.0)
#   API_KEY        require "Authorization: Bearer <key>" (strongly advised with BIND=0.0.0.0: Docker-published
#                  ports bypass ufw)
#   MAX_MODEL_LEN  context length (default 16384)     MAX_NUM_SEQS  concurrent sequences (default 64)
#   KV_CACHE_GB    KV-cache size in GB (default 16)    THINKING      Qwen3 thinking by default: true|false (false)
#   IMAGE          container image (default vllm/vllm-openai:v0.31.0, the validated build)
#   DROP_CACHES=1  drop the page cache first via a privileged container (not needed by vLLM 0.31, which counts
#                  reclaimable page cache as free on GB10, but frees memory for other CUDA programs)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${NAME:-detllm-deterministic}"
VALIDATED_QWEN3_8B_REVISION=b968826d9c46dd6066d109eabc6255188de91218

die() { printf '\033[1;31m[serve] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '\033[1;34m[serve]\033[0m %s\n' "$*"; }
usage() { sed -n '2,9p' "$0"; exit "${1:-0}"; }

ACTION="${1:-start}"
case "$ACTION" in
  start|stop|status) ;;
  -h|--help|help) usage 0 ;;
  *) echo "unknown command: $ACTION" >&2; usage 2 ;;
esac

if ! docker info >/dev/null 2>&1; then
  # Added to the docker group after this login session started? Re-run under that group - no re-login needed.
  if [ -z "${DETLLM_VIA_SG:-}" ] && command -v sg >/dev/null 2>&1 \
     && getent group docker | cut -d: -f4 | tr ',' '\n' | grep -qx "$(id -un)"; then
    export DETLLM_VIA_SG=1
    exec sg docker -c "$(printf '%q ' bash "$0" "$@")"
  fi
  die "cannot talk to Docker - join the docker group ('sudo usermod -aG docker \$USER') and log in again, or run: sg docker -c \"bash $0 ${*:-start}\""
fi

PORT="${PORT:-8000}"
BIND="${BIND:-127.0.0.1}"
host="$BIND"; [ "$host" = "0.0.0.0" ] && host=127.0.0.1

if [ "$ACTION" = "stop" ]; then
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker rm -f "$NAME" >/dev/null && log "stopped $NAME"
  else
    log "$NAME is not running"
  fi
  exit 0
fi
if [ "$ACTION" = "status" ]; then
  state="$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" || state="not created"
  log "$NAME: ${state:-not created}"
  if curl -fs "http://$host:$PORT/health" >/dev/null 2>&1; then log "healthy on http://$host:$PORT/v1"; else log "not answering on http://$host:$PORT"; fi
  exit 0
fi

IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
[ -z "${MODEL_DIR:-}" ] && [ -d "$HERE/model/Qwen3-8B" ] && MODEL_DIR="$HERE/model/Qwen3-8B"
REVISION="${REVISION:-}"
[ -z "$REVISION" ] && [ -z "${MODEL_DIR:-}" ] && [ "$MODEL" = "Qwen/Qwen3-8B" ] && REVISION="$VALIDATED_QWEN3_8B_REVISION"
SERVED_NAME="${SERVED_NAME:-qwen3-8b-deterministic}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
KV_CACHE_GB="${KV_CACHE_GB:-16}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
case "$(printf '%s' "${THINKING:-false}" | tr '[:upper:]' '[:lower:]')" in
  true|1|yes|on) THINKING=true ;;
  false|0|no|off) THINKING=false ;;
  *) die "THINKING must be true or false, got '$THINKING'" ;;
esac
for v in PORT MAX_MODEL_LEN MAX_NUM_SEQS KV_CACHE_GB; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a whole number, got '${!v}'"
done

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [ -f "$HERE/image/vllm-openai-v0.31.0.tar.zst" ] && [ "$IMAGE" = "vllm/vllm-openai:v0.31.0" ]; then
    log "loading $IMAGE from $HERE/image (a few minutes)"
    zstd -dc "$HERE/image/vllm-openai-v0.31.0.tar.zst" | docker load
  else
    log "pulling $IMAGE"
    docker pull "$IMAGE"
  fi
fi

# Replace a running instance first, so its memory is free for the check below.
docker rm -f "$NAME" >/dev/null 2>&1 || true

# Weights on disk decide the memory plan: weights + KV cache + ~6 GB for activations, CUDA graphs, server.
if [ -n "${MODEL_DIR:-}" ]; then
  [ -f "$MODEL_DIR/config.json" ] || die "MODEL_DIR=$MODEL_DIR has no config.json"
  weights_dir="$MODEL_DIR"
else
  snapshots="$HF_CACHE_DIR/hub/models--${MODEL//\//--}/snapshots"
  if [ -n "$REVISION" ] && [ -d "$snapshots/$REVISION" ]; then weights_dir="$snapshots/$REVISION"
  else weights_dir="$(ls -d "$snapshots/"* 2>/dev/null | head -n 1 || true)"; fi
fi
weights_gb=""
if [ -n "$weights_dir" ]; then
  weights_gb=$( (du -cLb "$weights_dir"/*.safetensors 2>/dev/null || true) | tail -n 1 | awk '$1 > 0 {printf "%.0f", $1 / 1e9 + 0.5}')
fi
weights_gb="${weights_gb:-20}"
mem_gb=$((weights_gb + KV_CACHE_GB + 6))
total_gb=$(awk '/^MemTotal/ {printf "%.1f", $2 * 1024 / 1e9}' /proc/meminfo)
util=$(awk -v m="$mem_gb" -v t="$total_gb" 'BEGIN {printf "%.4f", m / t}')
kv_bytes=$((KV_CACHE_GB * 1000000000))

if [ "${DROP_CACHES:-0}" = "1" ]; then
  docker run --rm --privileged ubuntu:24.04 sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' \
    || die "dropping the page cache failed (needs the ubuntu:24.04 image and Docker's --privileged)"
  log "dropped the page cache"
fi
avail_gb=$(awk '/^MemAvailable/ {printf "%.0f", $2 * 1024 / 1e9}' /proc/meminfo)
log "plan: ${weights_gb} GB weights + ${KV_CACHE_GB} GB KV cache + 6 GB = ${mem_gb} GB of ${total_gb} GB (MemAvailable ${avail_gb} GB)"
[ "$avail_gb" -ge "$mem_gb" ] || die "not enough free memory - stop other GPU jobs or lower KV_CACHE_GB"

if [ "$BIND" != "127.0.0.1" ] && [ -z "${API_KEY:-}" ]; then
  log "WARNING: BIND=$BIND exposes an unauthenticated API to the network (Docker ports bypass ufw); set API_KEY"
fi

run=(run -d --name "$NAME" --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
     -p "$BIND:$PORT:8000" -e VLLM_BATCH_INVARIANT=1)
[ -n "${API_KEY:-}" ] && run+=(-e "VLLM_API_KEY=$API_KEY")
if [ -n "${MODEL_DIR:-}" ]; then
  run+=(-v "$MODEL_DIR:/models/model:ro" -e HF_HUB_OFFLINE=1)
  model_args=(/models/model)
else
  run+=(-v "$HF_CACHE_DIR:/root/.cache/huggingface" -e HF_TOKEN -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}")
  model_args=("$MODEL")
  [ -n "$REVISION" ] && model_args+=(--revision "$REVISION" --tokenizer-revision "$REVISION")
fi
# Validated: bf16, batch-invariant kernels, TRITON_ATTN attention (vLLM would otherwise pick FLASH_ATTN on
# GB10), no prefix caching. Daily-use defaults: greedy decoding unless a request sets temperature; Qwen3
# thinking off unless a request passes chat_template_kwargs.
docker "${run[@]}" "$IMAGE" "${model_args[@]}" \
  --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --seed 0 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$util" --kv-cache-memory-bytes "$kv_bytes" \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --override-generation-config '{"temperature": 0.0}' \
  --default-chat-template-kwargs "{\"enable_thinking\": $THINKING}" >/dev/null
log "starting $NAME ($IMAGE, ${model_args[*]}) - VLLM_BATCH_INVARIANT=1, TRITON_ATTN"

for _ in $(seq 1 240); do
  if curl -fs "http://$host:$PORT/health" >/dev/null 2>&1; then
    log "ready: http://$host:$PORT/v1  (model \"$SERVED_NAME\")"
    echo "  check:  python3 \"$HERE/check_determinism.py\" --url http://$host:$PORT${API_KEY:+ --api-key \$API_KEY}"
    echo "  logs:   docker logs -f $NAME        stop: bash \"$0\" stop"
    exit 0
  fi
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ] || { docker logs --tail 40 "$NAME" >&2; die "server exited during start-up"; }
  sleep 5
done
docker logs --tail 40 "$NAME" >&2
die "server not ready after 20 minutes (still starting? see: docker logs -f $NAME)"
