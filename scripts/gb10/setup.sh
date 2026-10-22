#!/usr/bin/env bash
# One-time setup on the GB10 (Dell Pro Max GB10 / DGX Spark, DGX OS). Safe to re-run.
#
#   bash scripts/gb10/setup.sh                 # checks, submodule, image, model download for `parallel`
#   bash scripts/gb10/setup.sh --suite smoke   # download only what the smoke suite needs
#   bash scripts/gb10/setup.sh --skip-download
#   BASE_IMAGE=nvcr.io/nvidia/vllm:26.09-py3 bash scripts/gb10/setup.sh   # NVIDIA's NGC build instead
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SUITES=()
DO_BUILD=1
DO_DOWNLOAD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --suite) SUITES+=("$2"); shift 2 ;;
    --skip-build) DO_BUILD=0; shift ;;
    --skip-download) DO_DOWNLOAD=0; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done
[ ${#SUITES[@]} -eq 0 ] && SUITES=(smoke parallel)

log "1/6 Host checks"
arch="$(uname -m)"
[ "$arch" = "aarch64" ] || warn "expected aarch64 (GB10), found $arch"
if [ -f /etc/dgx-release ]; then grep -E 'DGX_(NAME|OTA_VERSION|SWBUILD_VERSION)' /etc/dgx-release || true; fi
command -v nvidia-smi >/dev/null || die "nvidia-smi not found - is the NVIDIA driver installed?"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || nvidia-smi
command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "cannot talk to Docker. Add yourself to the docker group: sudo usermod -aG docker \$USER && newgrp docker"
docker info 2>/dev/null | grep -qi nvidia || warn "NVIDIA container runtime not listed by 'docker info' (needed for --gpus all)"
free -g | sed -n '1,2p'
avail_gb="$(df -BG --output=avail "$HOME" | tail -n 1 | tr -dc '0-9')"
[ "${avail_gb:-0}" -ge 80 ] || warn "only ${avail_gb} GB free in \$HOME; image (~20 GB) + models (parallel ~20 GB, blog ~110 GB) need space"
mkdir -p "$HF_CACHE_DIR" "$DETLLM_CACHE_DIR" "$REPO_DIR/results"
write_host_info

log "2/6 Git submodule thinking-machines-lab/batch_invariant_ops"
if [ -d "$REPO_DIR/.git" ] || [ -f "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" submodule update --init --recursive
else
  warn "not a git checkout - make sure batch_invariant_ops/ contains the library"
fi
[ -f "$REPO_DIR/batch_invariant_ops/batch_invariant_ops/__init__.py" ] || die "batch_invariant_ops submodule is empty"

if [ "$DO_BUILD" = 1 ]; then
  log "3/6 Pull base image $BASE_IMAGE (large; first time takes a while)"
  docker pull "$BASE_IMAGE"
  log "4/6 Build $IMAGE"
  docker build -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR/docker"
else
  log "3-4/6 Skipping image build"
fi

log "5/6 GPU check inside the container"
bash "$REPO_DIR/scripts/gb10/run.sh" python3 -m detllm env

if [ "$DO_DOWNLOAD" = 1 ]; then
  log "6/6 Download models for: ${SUITES[*]}"
  dl_args=()
  for s in "${SUITES[@]}"; do dl_args+=(--suite "$s"); done
  bash "$REPO_DIR/scripts/gb10/run.sh" python3 -m detllm download "${dl_args[@]}"
else
  log "6/6 Skipping model download"
fi

log "Setup complete. Next:"
echo "  bash scripts/gb10/run_suite.sh smoke      # ~10-20 min sanity run"
echo "  bash scripts/gb10/run_suite.sh parallel   # main reproduction"
