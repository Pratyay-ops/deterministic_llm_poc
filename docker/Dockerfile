# Thin layer over a CUDA + PyTorch + Triton + vLLM image that runs on GB10 (aarch64, sm_121).
#
# Default base: the upstream vLLM image, v0.31.0 (multi-arch, arm64 included). It carries the
# newest batch-invariance code, including Triton matmul tiles tuned for compute capability 12.x
# (GB10 class). Built on CUDA 13.0 - the CUDA version of the DGX OS 7 driver (R580) - for
# compute capability 12.0, which runs on GB10's 12.1. Python 3.12.
# Alternative: NVIDIA's NGC build, validated by NVIDIA for DGX Spark / GB10:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/vllm:26.09-py3 ...   (vLLM 0.29.0, CUDA 13.4.1)
#
# Nothing from the GPU stack is reinstalled. The repository is bind-mounted at /workspace/detllm at
# run time (scripts/gb10/run.sh), so code changes never need an image rebuild.
ARG BASE_IMAGE=vllm/vllm-openai:v0.31.0
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

ENV DETLLM_BASE_IMAGE=${BASE_IMAGE} \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/detllm/src:/workspace/detllm/batch_invariant_ops \
    HF_HOME=/root/.cache/huggingface \
    DETLLM_CACHE_DIR=/root/.cache/detllm \
    TRITON_CACHE_DIR=/root/.cache/detllm/triton \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/detllm/torchinductor \
    VLLM_CACHE_ROOT=/root/.cache/detllm/vllm \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Only the test runner is added; the import check fails the build early if the base image lacks
# a library the harness needs.
RUN python3 -m pip install --no-cache-dir "pytest>=8" \
 && python3 -c "import importlib.metadata as m, torch, triton, transformers, yaml, httpx, huggingface_hub; print('vllm', m.version('vllm'), '| torch', torch.__version__, '| triton', triton.__version__, '| transformers', transformers.__version__)"

COPY entrypoint.sh /usr/local/bin/detllm-entrypoint.sh
RUN chmod +x /usr/local/bin/detllm-entrypoint.sh

WORKDIR /workspace/detllm
ENTRYPOINT ["/usr/local/bin/detllm-entrypoint.sh"]
CMD ["bash"]
