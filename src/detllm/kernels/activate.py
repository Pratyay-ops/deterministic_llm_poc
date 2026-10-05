"""Turn an op-level mode on inside the current process.

Variables that cuBLAS reads at handle creation (CUBLAS_WORKSPACE_CONFIG, CUBLASLT_WORKSPACE_SIZE)
are set by the scheduler before the process starts (``detllm.modes.pre_init_env``); this module
handles everything that can be switched from Python.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass
class KernelMode:
    name: str
    info: dict[str, Any] = field(default_factory=dict)
    _context: Callable[[], contextlib.AbstractContextManager] | None = None

    def context(self) -> contextlib.AbstractContextManager:
        """Wrap model / op calls in this (only the dispatch-mode based ``custom`` needs it)."""
        return self._context() if self._context else contextlib.nullcontext()

    def runtime_stats(self) -> dict[str, Any]:
        holder = self.info.get("_instance")
        return holder.stats() if holder is not None else {}


def _set_reduced_precision_off() -> None:
    for attr in ("allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction"):
        try:  # torch >= 2.10: (allow_reduced_precision, allow_splitk)
            setattr(torch.backends.cuda.matmul, attr, (False, False))
        except (TypeError, ValueError, RuntimeError):
            setattr(torch.backends.cuda.matmul, attr, False)


def _torch_flags() -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "use_deterministic_algorithms": True,
        "cudnn.deterministic": True,
        "allow_tf32": False,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _cublaslt_ws1() -> dict[str, Any]:
    # Mirrors vllm/model_executor/determinism/batch_invariant.py for GPUs outside the SM80 family.
    torch.backends.cuda.preferred_blas_library("cublaslt")
    _set_reduced_precision_off()
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
        "CUBLASLT_WORKSPACE_SIZE": os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def activate(mode: str) -> KernelMode:
    if mode in ("baseline", "default"):
        return KernelMode(mode)
    if mode == "torch_flags":
        return KernelMode(mode, _torch_flags())
    if mode == "cublaslt_ws1":
        return KernelMode(mode, _cublaslt_ws1())
    if mode in ("official", "official_full"):
        from detllm.kernels import official

        info: dict[str, Any] = {"official": official.enable()}
        if mode == "official_full":
            from detllm.kernels import extra_ops

            info["extra_ops"] = extra_ops.enable()
        return KernelMode(mode, info)
    if mode == "custom":
        from detllm.kernels.custom import FixedChunkMode

        instance = FixedChunkMode()
        return KernelMode(mode, {"custom": instance.describe(), "_instance": instance}, lambda: instance)
    raise ValueError(f"unknown kernel mode {mode!r}")


def public_info(mode: KernelMode) -> dict[str, Any]:
    """JSON-safe description of the active mode."""
    info = {k: v for k, v in mode.info.items() if not k.startswith("_")}
    stats = mode.runtime_stats()
    if stats:
        info["custom_stats"] = stats
    return {"mode": mode.name, **info}
