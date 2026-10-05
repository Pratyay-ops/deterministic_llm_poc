"""Op-level experiments (blog: "Batch invariance and determinism", "How do we make kernels
batch-invariant?").

For every op the first row is computed with batch size 1 and then as part of batches of growing
size; the op is *batch-invariant* if that row is bitwise identical at every batch size. Each op is
also repeated at a fixed batch size to show that it is *run-to-run deterministic* - the blog's
point is that kernels are almost always the latter but rarely the former.

Also included verbatim: the blog's ``torch.mm(a[:1], b)`` vs ``torch.mm(a, b)[:1]`` demo (1669.25
on the authors' GPU) and its 1000x repeated bf16 matmul run-to-run check.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from detllm.kernels.activate import KernelMode, activate, public_info

BATCH_SIZES = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
QUICK_BATCH_SIZES = [1, 2, 4, 16, 64, 256]
BLOG_MM_DEMO_DIFF = 1669.25


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(_bits(a), _bits(b)))


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    d = torch.nan_to_num((a.double() - b.double()).abs(), nan=float("inf"))
    return float(d.max().item())


def _randn(shape: tuple[int, ...], dtype: torch.dtype, g: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    # Generated directly on the GPU: no multi-GB host copies (host RAM *is* GPU memory on GB10).
    return torch.randn(shape, generator=g, device="cuda", dtype=dtype) * scale


def hf_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exactly how HF Qwen2/Qwen3 compute RMSNorm (fp32 mean over the hidden dim)."""
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    return weight * (xf * torch.rsqrt(variance + eps)).to(x.dtype)


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """HF ``eager_attention_forward``: matmul -> fp32 softmax -> matmul."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if causal:
        lq, lk = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).triu(1 + lk - lq)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


@dataclass
class OpSpec:
    name: str
    dtypes: tuple[str, ...]
    max_batch: int
    shape: str
    make: Callable[[torch.dtype, int, torch.Generator], dict[str, torch.Tensor]]
    fn: Callable[[dict[str, torch.Tensor], int], torch.Tensor]


H, D, KV = 16, 128, 1024  # attention heads, head dim, cached KV length

OPS: dict[str, OpSpec] = {
    s.name: s
    for s in (
        OpSpec(
            "mm",
            ("bfloat16", "float32", "float16"),
            2048,
            "[B,4096] @ [4096,4096]",
            lambda dt, b, g: {"a": _randn((b, 4096), dt, g), "w": _randn((4096, 4096), dt, g, 0.02)},
            lambda x, b: torch.mm(x["a"][:b], x["w"]),
        ),
        OpSpec(
            "linear_bias",
            ("bfloat16",),
            2048,
            "F.linear([B,2048], W[6144,2048], bias)  (aten::addmm)",
            lambda dt, b, g: {
                "x": _randn((b, 2048), dt, g),
                "w": _randn((6144, 2048), dt, g, 0.02),
                "bias": _randn((6144,), dt, g),
            },
            lambda x, b: F.linear(x["x"][:b], x["w"], x["bias"]),
        ),
        OpSpec(
            "rmsnorm",
            ("bfloat16",),
            2048,
            "HF RMSNorm over hidden=2048 (aten::mean.dim)",
            lambda dt, b, g: {"x": _randn((b, 2048), dt, g), "w": _randn((2048,), dt, g)},
            lambda x, b: hf_rmsnorm(x["x"][:b], x["w"]),
        ),
        OpSpec(
            "log_softmax_vocab",
            ("float32",),
            512,
            "log_softmax over vocab=151936 (sampler logprobs)",
            lambda dt, b, g: {"x": _randn((b, 151936), dt, g, 4.0)},
            lambda x, b: torch.log_softmax(x["x"][:b], dim=-1),
        ),
        OpSpec(
            "softmax",
            ("float32", "bfloat16"),
            2048,
            "softmax over 4096",
            lambda dt, b, g: {"x": _randn((b, 4096), dt, g, 4.0)},
            lambda x, b: torch.softmax(x["x"][:b], dim=-1),
        ),
        OpSpec(
            "bmm_attention_scores",
            ("bfloat16",),
            256,
            f"q[B,{H},1,{D}] @ k^T[B,{H},{D},{KV}]  (aten::bmm, batch=B*heads)",
            lambda dt, b, g: {"q": _randn((b, H, 1, D), dt, g), "kt": _randn((b, H, D, KV), dt, g)},
            lambda x, b: torch.matmul(x["q"][:b], x["kt"][:b]),
        ),
        OpSpec(
            "attention_eager_decode",
            ("bfloat16",),
            256,
            f"eager attention, 1 query vs {KV} cached keys, {H} heads",
            lambda dt, b, g: {
                "q": _randn((b, H, 1, D), dt, g),
                "k": _randn((b, H, KV, D), dt, g),
                "v": _randn((b, H, KV, D), dt, g),
            },
            lambda x, b: eager_attention(x["q"][:b], x["k"][:b], x["v"][:b]),
        ),
        OpSpec(
            "attention_eager_prefill",
            ("bfloat16",),
            256,
            f"eager causal attention, 128 queries, {H} heads",
            lambda dt, b, g: {
                "q": _randn((b, H, 128, D), dt, g),
                "k": _randn((b, H, 128, D), dt, g),
                "v": _randn((b, H, 128, D), dt, g),
            },
            lambda x, b: eager_attention(x["q"][:b], x["k"][:b], x["v"][:b], causal=True),
        ),
        OpSpec(
            "attention_sdpa_decode",
            ("bfloat16",),
            256,
            f"F.scaled_dot_product_attention, 1 query vs {KV} keys (not overridden by any mode)",
            lambda dt, b, g: {
                "q": _randn((b, H, 1, D), dt, g),
                "k": _randn((b, H, KV, D), dt, g),
                "v": _randn((b, H, KV, D), dt, g),
            },
            lambda x, b: F.scaled_dot_product_attention(x["q"][:b], x["k"][:b], x["v"][:b]),
        ),
        OpSpec(
            "attention_sdpa_prefill",
            ("bfloat16",),
            256,
            f"F.scaled_dot_product_attention causal, 128 queries, {H} heads",
            lambda dt, b, g: {
                "q": _randn((b, H, 128, D), dt, g),
                "k": _randn((b, H, 128, D), dt, g),
                "v": _randn((b, H, 128, D), dt, g),
            },
            lambda x, b: F.scaled_dot_product_attention(x["q"][:b], x["k"][:b], x["v"][:b], is_causal=True),
        ),
    )
}


def blog_mm_demo(km: KernelMode) -> dict[str, Any]:
    """Verbatim from the blog (fp32, linspace inputs)."""
    try:
        with torch.inference_mode(), km.context():
            batch, dim = 2048, 4096
            a = torch.linspace(-1000, 1000, batch * dim, device="cuda").reshape(batch, dim)
            b = torch.linspace(-1000, 1000, dim * dim, device="cuda").reshape(dim, dim)
            out1 = torch.mm(a[:1], b)
            out2 = torch.mm(a, b)[:1]
            diff = (out1 - out2).abs().max().item()
        return {"max_abs_diff": diff, "blog_value": BLOG_MM_DEMO_DIFF, "batch_invariant": diff == 0}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}


def blog_run_to_run(km: KernelMode, iterations: int = 1000) -> dict[str, Any]:
    """Verbatim from the blog: the same bf16 matmul 1000 times is bitwise identical."""
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(2048, 2048, generator=g, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, generator=g, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), km.context():
            ref = torch.mm(a, b)
            mismatches = sum(int((torch.mm(a, b) - ref).abs().max().item() != 0) for _ in range(iterations))
        return {"iterations": iterations, "mismatches": mismatches, "run_to_run_deterministic": mismatches == 0}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}


def test_op(spec: OpSpec, dtype_name: str, km: KernelMode, batch_sizes: list[int], rtr_iters: int, seed: int) -> dict[str, Any]:
    dtype = getattr(torch, dtype_name)
    sizes = [b for b in batch_sizes if b <= spec.max_batch]
    result: dict[str, Any] = {"op": spec.name, "dtype": dtype_name, "shape": spec.shape, "batch_sizes": sizes}
    try:
        g = torch.Generator(device="cuda").manual_seed(seed)
        inputs = spec.make(dtype, max(sizes), g)
        with torch.inference_mode(), km.context():
            ref = spec.fn(inputs, 1)[:1].clone()
            diffs: dict[int, float] = {}
            equal: dict[int, bool] = {}
            for b in sizes:
                row = spec.fn(inputs, b)[:1]
                equal[b] = bitwise_equal(row, ref)
                diffs[b] = max_abs_diff(row, ref)
            full = spec.fn(inputs, max(sizes)).clone()
            rtr = all(bitwise_equal(spec.fn(inputs, max(sizes)), full) for _ in range(rtr_iters))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                spec.fn(inputs, max(sizes))
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 3 * 1e3
        variant = [b for b in sizes if not equal[b]]
        result.update(
            batch_invariant=not variant,
            first_variant_batch=variant[0] if variant else None,
            n_variant_batches=len(variant),
            max_abs_diff=max(diffs.values()),
            max_abs_diff_by_batch={str(k): v for k, v in diffs.items()},
            run_to_run_identical=rtr,
            ms_at_max_batch=round(ms, 3),
            error=None,
        )
        del inputs
    except Exception as exc:  # noqa: BLE001 - e.g. Triton OutOfResources for fp16 on GB10
        result.update(batch_invariant=None, run_to_run_identical=None, error=f"{type(exc).__name__}: {str(exc)[:400]}")
    torch.cuda.empty_cache()
    return result


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    mode = ctx.job.mode
    km = activate(mode)
    quick = bool(ctx.param("quick", False))
    batch_sizes = ctx.param("batch_sizes") or (QUICK_BATCH_SIZES if quick else BATCH_SIZES)
    ops = ctx.param("ops") or list(OPS)
    rtr_iters = int(ctx.param("run_to_run_iters", 5 if quick else 20))

    summary: dict[str, Any] = {
        "mode": mode,
        "blog_mm_demo": blog_mm_demo(km),
        "blog_run_to_run": blog_run_to_run(km, iterations=100 if quick else 1000),
        "results": [],
    }
    ctx.log.info("blog mm demo: %s", summary["blog_mm_demo"])
    for name in ops:
        spec = OPS[name]
        for dt in ctx.param("dtypes") or spec.dtypes:
            if dt not in spec.dtypes:
                continue
            res = test_op(spec, dt, km, list(batch_sizes), rtr_iters, ctx.seed)
            summary["results"].append(res)
            ctx.log.info(
                "%-26s %-9s invariant=%s run_to_run=%s max|diff|=%s %s",
                name,
                dt,
                res.get("batch_invariant"),
                res.get("run_to_run_identical"),
                res.get("max_abs_diff"),
                res.get("error") or "",
            )
    summary["kernels"] = public_info(km)
    ok = [r for r in summary["results"] if r.get("error") is None]
    summary["n_ops"] = len(summary["results"])
    summary["n_invariant"] = sum(1 for r in ok if r["batch_invariant"])
    summary["n_run_to_run"] = sum(1 for r in ok if r["run_to_run_identical"])
    summary["n_errors"] = len(summary["results"]) - len(ok)
    return summary
