"""CUSTOM mode: batch invariance by construction, using the vendor (cuBLAS / PyTorch) kernels.

Why the previous implementation in this repo could not work:
* It replaced ``torch.matmul`` with a Python function that called ``torch.matmul`` again
  (infinite recursion), and Python-level patching never reaches ``nn.Linear`` (``F.linear`` ->
  ``aten::addmm``) or ops called from C++ - nor a vLLM engine running in another process.
* Padding the batch to the next multiple of 32 still gives the kernel a *different* shape for
  different batch sizes (32, 64, 96, ...), and cuBLAS picks its algorithm (tiles, split-K) per shape.
* Kahan summation changes the precision of a reduction, not its batch dependence.

The fix keeps the idea but applies it properly: every intercepted op is executed on fixed-size,
zero-padded chunks - always exactly ``rows`` rows for mm/addmm, ``mats`` matrices for bmm and
``reduce_rows`` rows for softmax/log_softmax/mean - copied into fresh buffers (same alignment
every time). The vendor kernel therefore always sees identical shapes and computes every row in
the same way, no matter how many other rows the batch contains. Interception happens at the
dispatcher with a ``TorchDispatchMode``, i.e. *below* F.linear / torch.matmul / softmax, and calling
the original kernel from inside the mode cannot recurse.

It is slower (Python dispatch, padding, copies) - the point is a correct, independent second
implementation to compare with the Triton kernels of batch_invariant_ops.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten


class FixedChunkMode(TorchDispatchMode):
    def __init__(self, rows: int = 64, mats: int = 64, reduce_rows: int = 256):
        super().__init__()
        self.rows = rows
        self.mats = mats
        self.reduce_rows = reduce_rows
        self.calls: Counter[str] = Counter()
        self.passthrough: Counter[str] = Counter()
        self._handlers = {
            aten.mm.default: self._mm,
            aten.addmm.default: self._addmm,
            aten.bmm.default: self._bmm,
            aten._softmax.default: self._softmax,
            aten._log_softmax.default: self._softmax,
            aten.mean.dim: self._mean_dim,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "mats": self.mats,
            "reduce_rows": self.reduce_rows,
            "ops": sorted(str(op) for op in self._handlers),
        }

    def stats(self) -> dict[str, Any]:
        return {"chunked_calls": dict(self.calls), "unchunked_calls": dict(self.passthrough)}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        handler = self._handlers.get(func)
        if handler is not None:
            out = handler(func, *args, **kwargs)
            if out is not NotImplemented:
                self.calls[str(func)] += 1
                return out
            self.passthrough[str(func)] += 1
        return func(*args, **kwargs)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _chunk(x: torch.Tensor, start: int, size: int) -> tuple[torch.Tensor, int]:
        """Rows [start, start+size) of ``x`` copied into a fresh zero-padded buffer of ``size`` rows."""
        part = x[start : start + size]
        n = part.shape[0]
        buf = torch.empty((size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        buf[:n].copy_(part)
        if n < size:
            buf[n:].zero_()
        return buf, n

    # ---------------------------------------------------------------- matmuls

    def _mm(self, func, a: torch.Tensor, b: torch.Tensor):  # noqa: ANN001
        if not a.is_cuda or a.dim() != 2 or a.shape[0] == 0:
            return NotImplemented
        outs = []
        for start in range(0, a.shape[0], self.rows):
            buf, n = self._chunk(a, start, self.rows)
            outs.append(func(buf, b)[:n])
        return torch.cat(outs, dim=0)

    def _addmm(self, func, bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, beta=1, alpha=1):  # noqa: ANN001
        if not a.is_cuda or a.dim() != 2 or a.shape[0] == 0:
            return NotImplemented
        rows_bias = bias.dim() == 2 and bias.shape[0] == a.shape[0] and a.shape[0] != 1
        outs = []
        for start in range(0, a.shape[0], self.rows):
            buf, n = self._chunk(a, start, self.rows)
            bias_part = self._chunk(bias, start, self.rows)[0] if rows_bias else bias
            outs.append(func(bias_part, buf, b, beta=beta, alpha=alpha)[:n])
        return torch.cat(outs, dim=0)

    def _bmm(self, func, a: torch.Tensor, b: torch.Tensor):  # noqa: ANN001
        if not a.is_cuda or a.dim() != 3 or a.shape[0] == 0:
            return NotImplemented
        outs = []
        for start in range(0, a.shape[0], self.mats):
            abuf, n = self._chunk(a, start, self.mats)
            bbuf, _ = self._chunk(b, start, self.mats)
            outs.append(func(abuf, bbuf)[:n])
        return torch.cat(outs, dim=0)

    # ---------------------------------------------------------------- row reductions

    def _softmax(self, func, x: torch.Tensor, dim: int, half_to_float: bool):  # noqa: ANN001
        if not x.is_cuda or x.dim() == 0 or dim % x.dim() != x.dim() - 1 or x.numel() == 0:
            return NotImplemented
        x2 = x.reshape(-1, x.shape[-1])
        outs = []
        for start in range(0, x2.shape[0], self.reduce_rows):
            buf, n = self._chunk(x2, start, self.reduce_rows)
            outs.append(func(buf, -1, half_to_float)[:n])
        return torch.cat(outs, dim=0).reshape(x.shape)

    def _mean_dim(self, func, x: torch.Tensor, dim=None, keepdim: bool = False, *, dtype=None):  # noqa: ANN001
        if not x.is_cuda or x.dim() == 0 or x.numel() == 0 or dim is None:
            return NotImplemented
        dims = [dim] if isinstance(dim, int) else list(dim)
        if [d % x.dim() for d in dims] != [x.dim() - 1]:
            return NotImplemented  # only last-dim reductions (RMSNorm) are chunked
        x2 = x.reshape(-1, x.shape[-1])
        outs = []
        for start in range(0, x2.shape[0], self.reduce_rows):
            buf, n = self._chunk(x2, start, self.reduce_rows)
            outs.append(func(buf, [-1], True, dtype=dtype)[:n])
        out = torch.cat(outs, dim=0)  # (rows, 1)
        shape = list(x.shape[:-1]) + ([1] if keepdim else [])
        return out.reshape(shape)
