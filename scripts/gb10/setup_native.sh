#!/usr/bin/env bash
# FALLBACK ONLY - native Python environment on DGX OS without Docker.
# The container route (setup.sh) is what NVIDIA supports on DGX Spark / GB10 and what this repo is
# tested against; use this only if Docker is unavailable to you.
#
#   bash scripts/gb10/setup_native.sh
#   VLLM_VERSION=0.31.0 bash scripts/gb10/setup_native.sh
#
# `pip install vllm` pulls a matching torch/triton. It works only if an aarch64 CUDA wheel exists
# for your vLLM version and its CUDA build supports sm_121 under your driver - if `detllm env` shows
# no GPU or the preflight fails, go back to the container.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VENV="${VENV:-$REPO_DIR/.venv}"
VLLM_VERSION="${VLLM_VERSION:-0.31.0}"
PYTHON="${PYTHON:-python3}"

command -v "$PYTHON" >/dev/null || die "$PYTHON not found"
git -C "$REPO_DIR" submodule update --init --recursive
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel
python -m pip install "vllm==${VLLM_VERSION}"
python -m pip install --no-deps -e "$REPO_DIR/batch_invariant_ops"
python -m pip install -e "$REPO_DIR[dev]" httpx huggingface_hub
python -m detllm env
log "native environment ready: source $VENV/bin/activate && python -m detllm suite smoke --resume-latest"
