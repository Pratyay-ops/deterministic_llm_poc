# Thin layer over a CUDA + PyTorch + Triton + vLLM image that runs on GB10 (aarch64, sm_121).
#
# Default base: NVIDIA's NGC vLLM container, validated for DGX Spark / GB10.
#   26.09-py3 -> vLLM 0.29.0, PyTorch 2.14 (pre-release build), CUDA 13.4.1, Python 3.12, Transformers 5.15
# Alternative: the upstream image with the newest batch-invariance code (GB10/sm120-tuned tiles):
#   docker build --build-arg BASE_IMAGE=vllm/vllm-openai:v0.31.0 ...
#
# Nothing from the GPU stack is reinstalled. The repository is bind-mounted at /workspace/detllm at
# run time (scripts/gb10/run.sh), so code changes never need an image rebuild.
ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.09-py3
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
