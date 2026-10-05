#!/usr/bin/env bash
# Keep NVIDIA's entrypoint (banner, CUDA compatibility checks) when the base image has one, then
# run the requested command. Upstream vllm/vllm-openai images default to `vllm serve`; this
# entrypoint makes both bases behave the same way (run whatever command is given, default bash).
set -e
if [ -x /opt/nvidia/nvidia_entrypoint.sh ]; then
  exec /opt/nvidia/nvidia_entrypoint.sh "$@"
fi
exec "$@"
