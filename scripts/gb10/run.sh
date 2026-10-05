#!/usr/bin/env bash
# Run a command inside the detllm container: GPU, repo, Hugging Face cache and compile caches mounted.
#
#   bash scripts/gb10/run.sh                          # interactive shell
#   bash scripts/gb10/run.sh python3 -m detllm list
#   DETACH=1 bash scripts/gb10/run.sh python3 -m detllm suite parallel   # background container
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found - run: bash scripts/gb10/setup.sh"
mkdir -p "$HF_CACHE_DIR" "$DETLLM_CACHE_DIR" "$REPO_DIR/results"
[ $# -eq 0 ] && set -- bash

docker_run_args
if [ "${DETACH:-0}" = "1" ]; then
  docker "${DOCKER_ARGS[@]}" "$IMAGE" "$@" >/dev/null
  log "started container ${CONTAINER_NAME:-detllm-run} in the background"
  echo "  follow:  docker logs -f ${CONTAINER_NAME:-detllm-run}"
  echo "  stop:    docker stop ${CONTAINER_NAME:-detllm-run}     (finished jobs are kept; re-run resumes)"
else
  exec docker "${DOCKER_ARGS[@]}" "$IMAGE" "$@"
fi
