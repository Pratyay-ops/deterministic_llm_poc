"""Engine / kernel modes, and the environment variables each one needs *before* CUDA starts.

Pure Python: the scheduler uses this to build each job's environment without importing torch.

Op-level modes (``op_invariance`` experiment):
  baseline       stock PyTorch kernels (cuBLAS / cuBLASLt with default heuristics)
  torch_flags    the "usual" determinism knobs: CUBLAS_WORKSPACE_CONFIG,
                 torch.use_deterministic_algorithms, cudnn.deterministic, no TF32. These fix
                 run-to-run nondeterminism but NOT batch variance - the blog's central point.
  cublaslt_ws1   what vLLM's batch-invariant mode does for GEMMs on Hopper/Blackwell (and, today,
                 on GB10): cuBLASLt with a ~1 KiB workspace so no split-K kernel can be picked.
  official       thinking-machines-lab/batch_invariant_ops as released (Triton mm/addmm,
                 log_softmax, mean).
  official_full  official + this repo's batch-invariant Triton bmm/softmax, which the library
                 lacks and which eager attention needs.
  custom         this repo's own approach: run vendor kernels on fixed-size, zero-padded chunks
                 so the kernel shape (and therefore its reduction strategy) never depends on the
                 batch size. A corrected version of the original "pad to a multiple of 32" idea.

HF transformers modes (``hf_invariance``) pick an attention implementation plus an op mode.
vLLM modes (``vllm_*``): ``baseline`` vs ``batch_invariant`` (``VLLM_BATCH_INVARIANT=1``).
"""

from __future__ import annotations

OP_MODES: tuple[str, ...] = (
    "baseline",
    "torch_flags",
    "cublaslt_ws1",
    "official",
    "official_full",
    "custom",
)
HF_MODES: tuple[str, ...] = ("baseline", "baseline_eager", "official", "official_full", "custom")
VLLM_MODES: tuple[str, ...] = ("baseline", "batch_invariant")

MODE_DESCRIPTIONS: dict[str, str] = {
    "baseline": "stock kernels, no determinism measures",
    "torch_flags": "classic PyTorch determinism flags (fix run-to-run, not batch variance)",
    "cublaslt_ws1": "cuBLASLt with 1 KiB workspace (no split-K) - vLLM's GEMM strategy on SM90+",
    "official": "thinking-machines-lab/batch_invariant_ops (mm, addmm, log_softmax, mean)",
    "official_full": "batch_invariant_ops + this repo's batch-invariant bmm/softmax (attention)",
    "custom": "fixed-size zero-padded chunks through vendor kernels (TorchDispatchMode)",
    "baseline_eager": "stock kernels with HF eager attention (isolates the attention backend)",
    "batch_invariant": "vLLM with VLLM_BATCH_INVARIANT=1",
    "default": "-",
}

# Op mode applied inside an HF job, and the HF attention implementation used with it.
HF_KERNEL_MODE: dict[str, str] = {
    "baseline": "baseline",
    "baseline_eager": "baseline",
    "official": "official",
    "official_full": "official_full",
    "custom": "custom",
}
HF_ATTN_IMPL: dict[str, str] = {
    "baseline": "sdpa",
    "baseline_eager": "eager",
    "official": "eager",
    "official_full": "eager",
    "custom": "eager",
}

# Variables cuBLAS reads once, when its handle is created - they must be in the environment
# before the job process touches CUDA. ``None`` means "make sure it is unset".
_OP_PRE_INIT_ENV: dict[str, dict[str, str | None]] = {
    "baseline": {"CUBLAS_WORKSPACE_CONFIG": None, "CUBLASLT_WORKSPACE_SIZE": None},
    "torch_flags": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUBLASLT_WORKSPACE_SIZE": None},
    "cublaslt_ws1": {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"},
}


def pre_init_env(family: str, mode: str | None) -> dict[str, str | None]:
    """Environment overrides for a job of the given experiment family and mode."""
    if family == "vllm":
        # vLLM reads VLLM_BATCH_INVARIANT at import / engine start-up.
        return {"VLLM_BATCH_INVARIANT": "1" if mode == "batch_invariant" else None}
    if family == "ops":
        return dict(_OP_PRE_INIT_ENV.get(mode or "", {}))
    if family == "hf":
        return dict(_OP_PRE_INIT_ENV.get(HF_KERNEL_MODE.get(mode or "", ""), {}))
    return {}


def apply_env(overrides: dict[str, str | None], env: dict[str, str] | None = None) -> dict[str, str]:
    """Apply overrides to ``env`` (default: ``os.environ``); ``None`` values delete the key."""
    import os

    target = os.environ if env is None else env
    for key, value in overrides.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = str(value)
    return dict(target)
