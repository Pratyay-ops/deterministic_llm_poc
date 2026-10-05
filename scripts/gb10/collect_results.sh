#!/usr/bin/env bash
# Pack a run directory (default: the newest one) into a .tar.gz you can copy off the GB10.
#
#   bash scripts/gb10/collect_results.sh                      # newest run in results/
#   bash scripts/gb10/collect_results.sh results/parallel-20261006-101500
#   LIGHT=1 bash scripts/gb10/collect_results.sh              # skip the large per-completion files
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$(ls -1dt "$REPO_DIR"/results/*/ 2>/dev/null | head -n 1)"
  [ -n "$RUN_DIR" ] || die "no run directories in results/"
fi
RUN_DIR="${RUN_DIR%/}"
[ -d "$RUN_DIR" ] || die "not a directory: $RUN_DIR"

name="$(basename "$RUN_DIR")"
out="$REPO_DIR/results/${name}.tar.gz"
excludes=()
if [ "${LIGHT:-0}" = "1" ]; then
  excludes=(--exclude='*/completions.jsonl' --exclude='*/needle_outputs.jsonl' --exclude='*/vllm_server.log')
fi
tar -C "$(dirname "$RUN_DIR")" "${excludes[@]}" -czf "$out" "$name" "$(basename "$REPO_DIR/results/host_info.txt")" 2>/dev/null \
  || tar -C "$(dirname "$RUN_DIR")" "${excludes[@]}" -czf "$out" "$name"
log "wrote $out ($(du -h "$out" | cut -f1))"
echo "  report: $RUN_DIR/report.md"
