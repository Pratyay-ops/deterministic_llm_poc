"""Wrapper around thinking-machines-lab/batch_invariant_ops (the blog's companion library).

The library swaps ``aten::mm``, ``aten::addmm``, ``aten::_log_softmax`` and ``aten::mean.dim`` for
Triton kernels with one fixed tile configuration per dtype (no split-K, so every row is reduced
in the same order whatever the batch size). Two things matter on GB10:

* Shared memory. GB10 (sm_121) is a consumer-Blackwell-class part with ~99 KB of opt-in shared
  memory per block (H100: 227 KB). The library's fp16 tile (128x256x64, 3 stages) needs ~144 KB
  and cannot launch there; bf16 / fp32 tiles need ~96 KB and fit. vLLM hit the same limit and
  shrinks BLOCK_N for fp16 on such GPUs. This harness therefore runs models in bf16 and reports
  the fp16 failure instead of hiding it.
* fp32 precision. The library's ``tl.dot`` runs fp32 inputs in TF32 (open issue #23 / PR #24), so
  "batch-invariant" fp32 matmuls are less precise than stock IEEE fp32 matmuls.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

import torch

from detllm.env import device_smem_optin, read_git_head
from detllm.utils import REPO_ROOT

log = logging.getLogger(__name__)

SUBMODULE_DIR = REPO_ROOT / "batch_invariant_ops"

# Tile configuration hard-coded in batch_invariant_ops.matmul_persistent (commit f22b1fb).
LIBRARY_MATMUL_TILES: dict[torch.dtype, dict[str, int]] = {
    torch.bfloat16: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_stages": 3},
    torch.float16: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "num_stages": 3},
    torch.float32: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "num_stages": 3},
}


def import_library():  # noqa: ANN201
    try:
        return importlib.import_module("batch_invariant_ops")
    except ImportError:
        if not (SUBMODULE_DIR / "batch_invariant_ops" / "__init__.py").exists():
            raise ImportError(
                "batch_invariant_ops not found - run `git submodule update --init` "
                "(scripts/gb10/setup.sh does this)"
            ) from None
        sys.path.insert(0, str(SUBMODULE_DIR))
        return importlib.import_module("batch_invariant_ops")


def estimated_smem_bytes(dtype: torch.dtype) -> int:
    """Pipeline buffers of the library's matmul tile: (A tile + B tile) x stages."""
    t = LIBRARY_MATMUL_TILES[dtype]
    elem = torch.tensor([], dtype=dtype).element_size()
    per_stage = (t["BLOCK_SIZE_M"] * t["BLOCK_SIZE_K"] + t["BLOCK_SIZE_K"] * t["BLOCK_SIZE_N"]) * elem
    return per_stage * t["num_stages"]


def library_info() -> dict[str, Any]:
    lib = import_library()
    smem = device_smem_optin() if torch.cuda.is_available() else None
    return {
        "module_path": getattr(lib, "__file__", None),
        "version": getattr(lib, "__version__", None),
        "submodule_commit": read_git_head(SUBMODULE_DIR),
        "device_smem_optin_bytes": smem,
        "tile_smem_bytes": {
            str(dt).removeprefix("torch."): {
                "needed": estimated_smem_bytes(dt),
                "fits": None if smem is None else estimated_smem_bytes(dt) <= smem,
            }
            for dt in LIBRARY_MATMUL_TILES
        },
    }


def probe(dtype: torch.dtype) -> dict[str, Any]:
    """Run one small matmul through the (enabled) library and check it against fp64."""
    try:
        g = torch.Generator().manual_seed(0)
        a = torch.randn(33, 512, generator=g)
        b = torch.randn(512, 384, generator=g)
        ref = (a.double() @ b.double()).float()
        out = torch.mm(a.to("cuda", dtype), b.to("cuda", dtype)).float().cpu()
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        return {"ok": True, "max_rel_err": rel}
    except Exception as exc:  # noqa: BLE001 - e.g. triton OutOfResources on small-smem GPUs
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}


def enable(probe_dtypes: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float32)) -> dict[str, Any]:
    lib = import_library()
    lib.enable_batch_invariant_mode()
    info = {"enabled": True, **library_info(), "probes": {}}
    for dt in probe_dtypes:
        info["probes"][str(dt).removeprefix("torch.")] = probe(dt)
    bad = [k for k, v in info["probes"].items() if not v["ok"]]
    if bad:
        log.warning("batch_invariant_ops failed for %s: %s", bad, {k: info["probes"][k] for k in bad})
    return info


def disable() -> None:
    import_library().disable_batch_invariant_mode()
