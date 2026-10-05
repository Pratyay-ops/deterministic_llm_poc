"""Batch-invariant ``aten::bmm`` and ``aten::_softmax`` (Triton).

batch_invariant_ops covers mm/addmm/log_softmax/mean, which is enough for the linear layers and
RMSNorm of a transformer but not for attention: HF's eager attention is
``matmul(q, k^T) -> softmax(fp32) -> matmul(p, v)``, i.e. ``aten::bmm`` + ``aten::_softmax``.
cuBLAS batched GEMM may choose a different kernel (and split-K) when the number of matrices in
the batch changes, so the library alone is not batch-invariant end to end (see its issue #6).

Both kernels use one fixed configuration for every shape:
* bmm: every program owns one (matrix, M-tile, N-tile) and walks K in fixed BLOCK_K steps, so an
  output element is reduced the same way regardless of how many matrices are in the batch.
  ``input_precision="ieee"`` keeps fp32 inputs exact - Triton defaults to TF32, which corrupts the
  rotary-embedding phase ``inv_freq @ position_ids`` beyond position 2048 (batch_invariant_ops #23).
* softmax: one program per row, fixed BLOCK, vector accumulators reduced once at the end.

Tiles are small (12-24 KB of shared memory) so they fit GB10's ~99 KB/block limit with room to spare.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import triton
import triton.language as tl

log = logging.getLogger(__name__)

BMM_CONFIG = {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
SOFTMAX_BLOCK = 1024
_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles = num_pid_m * num_pid_n
    batch = (pid // tiles).to(tl.int64)
    tile = pid % tiles
    pid_m = tile // num_pid_n
    pid_n = tile % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + batch * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + batch * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        mask_k = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + batch * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride
    offs = tl.arange(0, BLOCK)

    running_max = tl.full([BLOCK], float("-inf"), tl.float32)
    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        v = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        running_max = tl.maximum(running_max, v)
    row_max = tl.max(running_max, axis=0)

    running_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        v = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        running_sum += tl.exp(v - row_max)
    denom = tl.sum(running_sum, axis=0)

    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        mask = cols < n_cols
        v = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(y_row + cols, (tl.exp(v - row_max) / denom).to(y_ptr.dtype.element_ty), mask=mask)


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch-invariant (B, M, K) @ (B, K, N) -> (B, M, N)."""
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError(f"bmm expects 3-D tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError(f"bmm shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.dtype != b.dtype or a.dtype not in _SUPPORTED:
        raise NotImplementedError(f"batch-invariant bmm supports fp16/bf16/fp32 pairs, got {a.dtype}, {b.dtype}")
    batch, m, k = a.shape
    n = b.shape[2]
    c = torch.empty((batch, m, n), device=a.device, dtype=a.dtype)
    if c.numel() == 0:
        return c
    if k == 0:
        return c.zero_()
    cfg = BMM_CONFIG
    tiles = triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"])
    _bmm_kernel[(batch * tiles,)](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return c


def softmax(x: torch.Tensor, dim: int = -1, half_to_float: bool = False) -> torch.Tensor:
    """Batch-invariant softmax (same semantics as ``aten::_softmax``)."""
    if x.dtype not in _SUPPORTED:
        raise NotImplementedError(f"batch-invariant softmax supports fp16/bf16/fp32, got {x.dtype}")
    out_dtype = torch.float32 if half_to_float else x.dtype
    if x.dim() == 0:
        return torch.ones((), dtype=out_dtype, device=x.device)
    dim = dim % x.dim()
    moved = dim != x.dim() - 1
    xt = x.movedim(dim, -1) if moved else x
    shape = xt.shape
    x2 = xt.reshape(-1, shape[-1]).contiguous()
    y2 = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
    if x2.numel():
        _softmax_kernel[(x2.shape[0],)](
            x2, y2, x2.stride(0), y2.stride(0), x2.shape[1], BLOCK=SOFTMAX_BLOCK, num_warps=4
        )
    y = y2.reshape(shape)
    return y.movedim(-1, dim).contiguous() if moved else y


def _bmm_impl(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return bmm(self, mat2)


def _softmax_impl(self: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    return softmax(self, dim, half_to_float)


_LIB: Any = None


def _register(lib: Any, op: str, fn: Any) -> None:
    try:  # torch >= 2.12 ships its own Triton bmm for CUDA, so we must be allowed to override it
        lib.impl(op, fn, "CUDA", allow_override=True)
    except TypeError:  # older torch: no allow_override keyword
        lib.impl(op, fn, "CUDA")


def enable() -> dict[str, Any]:
    global _LIB
    if _LIB is None:
        lib = torch.library.Library("aten", "IMPL")
        _register(lib, "aten::bmm", _bmm_impl)
        _register(lib, "aten::_softmax", _softmax_impl)
        _LIB = lib
    return {
        "enabled": True,
        "ops": ["aten::bmm", "aten::_softmax"],
        "bmm_config": BMM_CONFIG,
        "softmax_block": SOFTMAX_BLOCK,
        "dispatch_check": dispatch_check(),
    }


def disable() -> None:
    global _LIB
    if _LIB is not None:
        _LIB._destroy()
        _LIB = None


def is_enabled() -> bool:
    return _LIB is not None


def dispatch_check() -> dict[str, Any]:
    """Verify that torch.matmul / torch.softmax really reach our kernels (bitwise identical)."""
    try:
        g = torch.Generator().manual_seed(0)
        q = torch.randn(2, 4, 3, 64, generator=g).to("cuda", torch.bfloat16)
        k = torch.randn(2, 4, 64, 40, generator=g).to("cuda", torch.bfloat16)
        via_torch = torch.matmul(q, k)
        direct = bmm(q.reshape(8, 3, 64), k.reshape(8, 64, 40)).reshape(2, 4, 3, 40)
        s = torch.randn(5, 77, generator=g).to("cuda")
        return {
            "matmul_routed": bool(torch.equal(via_torch, direct)),
            "softmax_routed": bool(torch.equal(torch.softmax(s, dim=-1), softmax(s, -1))),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
