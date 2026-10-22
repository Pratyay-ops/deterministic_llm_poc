#!/usr/bin/env bash
# Run (or resume) a suite on the GB10.
#
#   bash scripts/gb10/run_suite.sh smoke
#   bash scripts/gb10/run_suite.sh parallel
#   bash scripts/gb10/run_suite.sh blog --exclude 'moe-*'
#   bash scripts/gb10/run_suite.sh parallel --only 'completions-*' 'sweep-*'
#   DETACH=1 bash scripts/gb10/run_suite.sh parallel        # keep running after you disconnect
#
# Re-running the same command continues the newest unfinished run of that suite: jobs whose
# result.json says "ok" are skipped. Pass --no-resume or --run-dir to change that.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SUITE="${1:-parallel}"
[ $# -gt 0 ] && shift
case "$SUITE" in
  *.yaml|*.yml) SUITE_FILE="$SUITE" ;;
  *) SUITE_FILE="configs/suites/$SUITE.yaml" ;;
esac
[ -f "$REPO_DIR/$SUITE_FILE" ] || die "suite file not found: $SUITE_FILE"

# Unified memory: the page cache competes with GPU allocations and CUDA does not count it as free
# (NVIDIA DGX Spark known issue). Flush it before starting engines.
if [ "${DROP_CACHES:-1}" = "1" ]; then
  if sudo -n true 2>/dev/null || [ -t 0 ]; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && log "dropped the page cache"
  else
    warn "could not drop the page cache (no sudo). Run: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
  fi
fi
write_host_info

CONTAINER_NAME="${CONTAINER_NAME:-detllm-$(basename "$SUITE_FILE" .yaml)}"
export CONTAINER_NAME
exec bash "$REPO_DIR/scripts/gb10/run.sh" python3 -m detllm suite "$SUITE_FILE" --resume-latest "$@"
