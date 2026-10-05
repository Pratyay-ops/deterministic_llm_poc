#!/usr/bin/env bash
# Shared settings for the GB10 helper scripts. Override any of them through the environment, e.g.
#   BASE_IMAGE=vllm/vllm-openai:v0.31.0 bash scripts/gb10/setup.sh

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/vllm:26.09-py3}"
IMAGE="${IMAGE:-detllm:latest}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
DETLLM_CACHE_DIR="${DETLLM_CACHE_DIR:-$HOME/.cache/detllm}"
CONTAINER_REPO=/workspace/detllm

log() { printf '\033[1;34m[detllm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[detllm] WARNING:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[detllm] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Build the `docker run` argument list into the DOCKER_ARGS array.
docker_run_args() {
  DOCKER_ARGS=(run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864)
  if [ "${DETACH:-0}" = "1" ]; then
    DOCKER_ARGS+=(-d --name "${CONTAINER_NAME:-detllm-run}")
  elif [ -t 0 ] && [ -t 1 ]; then
    DOCKER_ARGS+=(-it)
  fi
  DOCKER_ARGS+=(
    -v "$REPO_DIR:$CONTAINER_REPO"
    -v "$HF_CACHE_DIR:/root/.cache/huggingface"
    -v "$DETLLM_CACHE_DIR:/root/.cache/detllm"
    -w "$CONTAINER_REPO"
    -e HF_TOKEN
    -e HUGGING_FACE_HUB_TOKEN
    -e HF_HUB_OFFLINE
    -e TRITON_PTXAS_PATH
    -e "DETLLM_BASE_IMAGE=$BASE_IMAGE"
  )
}

write_host_info() {
  local out="$REPO_DIR/results/host_info.txt"
  mkdir -p "$REPO_DIR/results"
  {
    echo "# collected $(date -Is)"
    echo "## uname"; uname -a
    echo "## dgx-release"; cat /etc/dgx-release 2>/dev/null || echo "(none)"
    echo "## os-release"; grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null
    echo "## nvidia-smi"; nvidia-smi 2>&1 | head -n 25
    echo "## memory"; free -g
    echo "## docker"; docker version --format '{{.Server.Version}}' 2>&1
    echo "## base image"; echo "$BASE_IMAGE"
  } >"$out" 2>&1 || true
}
