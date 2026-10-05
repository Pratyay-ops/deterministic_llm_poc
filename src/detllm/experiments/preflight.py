"""Preflight: catch environment problems in ~1 minute instead of an hour into a suite.

Checks CUDA / GB10 detection, Triton compilation on sm_121, which batch_invariant_ops tiles fit in
shared memory, correctness of this repo's Triton kernels (incl. exact fp32), the CUSTOM mode, and
which batch-invariance features the installed vLLM has.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from detllm.env import device_smem_optin, gpu_info, nvidia_smi, package_versions
from detllm.utils import meminfo_gb


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def _check(name: str, fn, critical: bool = True) -> dict[str, Any]:  # noqa: ANN001
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {str(exc)[:600]}"
    return {"name": name, "ok": ok, "critical": critical, "detail": detail}


def _cuda() -> tuple[bool, Any]:
    info = gpu_info()
    return bool(info.get("cuda_available")), info


def _triton() -> tuple[bool, Any]:
    n = 4097
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    _add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return bool(torch.equal(out, x + y)), {"triton": triton.__version__}


def _official() -> tuple[bool, Any]:
    from detllm.kernels import official

    info = official.enable(probe_dtypes=(torch.bfloat16, torch.float32, torch.float16))
    official.disable()
    ok = info["probes"]["bfloat16"]["ok"] and info["probes"]["float32"]["ok"]
    if not info["probes"]["float16"]["ok"]:
        info["note"] = (
            "fp16 tile (128x256x64, 3 stages) does not fit this GPU's shared memory - expected on "
            "GB10; the harness uses bf16."
        )
    return ok, info


def _extra_ops() -> tuple[bool, Any]:
    from detllm.kernels import extra_ops

    g = torch.Generator().manual_seed(0)
    detail: dict[str, Any] = {}
    ok = True
    for dt, tol in ((torch.bfloat16, 2e-2), (torch.float32, 1e-5)):
        a = torch.randn(6, 5, 96, generator=g)
        b = torch.randn(6, 96, 70, generator=g)
        ref = a.double() @ b.double()
        out = extra_ops.bmm(a.to("cuda", dt), b.to("cuda", dt)).double().cpu()
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        detail[f"bmm_{dt}"] = err
        ok &= err < tol
    # exact fp32 (no TF32): position ids must survive a K=1 "matmul" unchanged (RoPE pattern)
    pos = torch.arange(8192, dtype=torch.float32, device="cuda").reshape(1, 1, -1)
    ones = torch.ones(1, 1, 1, device="cuda")
    exact = bool(torch.equal(extra_ops.bmm(ones, pos), pos))
    detail["fp32_ieee_exact"] = exact
    ok &= exact
    s = torch.randn(7, 3000, generator=g)
    err = (extra_ops.softmax(s.cuda(), -1).double().cpu() - torch.softmax(s.double(), -1)).abs().max().item()
    detail["softmax_max_abs_err"] = err
    ok &= err < 1e-5
    detail["dispatch"] = extra_ops.enable()["dispatch_check"]
    extra_ops.disable()
    ok &= detail["dispatch"].get("matmul_routed") is True and detail["dispatch"].get("softmax_routed") is True
    return bool(ok), detail


def _custom() -> tuple[bool, Any]:
    from detllm.kernels.custom import FixedChunkMode

    g = torch.Generator().manual_seed(0)
    a = torch.randn(130, 256, generator=g).to("cuda", torch.bfloat16)
    w = torch.randn(512, 256, generator=g).to("cuda", torch.bfloat16)
    ref = torch.nn.functional.linear(a, w)
    mode = FixedChunkMode()
    with mode:
        out = torch.nn.functional.linear(a, w)
        out_first = torch.nn.functional.linear(a[:1], w)
    err = (out.float() - ref.float()).abs().max().item()
    invariant = bool(torch.equal(out[:1], out_first))
    return err < 0.5 and invariant, {"max_abs_err_vs_stock": err, "row0_bs1_vs_bs130_equal": invariant, **mode.stats()}


def _vllm() -> tuple[bool, Any]:
    import dataclasses

    import vllm
    from vllm.engine.arg_utils import EngineArgs

    fields = {f.name for f in dataclasses.fields(EngineArgs)}
    detail: dict[str, Any] = {
        "version": getattr(vllm, "__version__", None),
        "kv_cache_memory_bytes": "kv_cache_memory_bytes" in fields,
        "attention_backend_arg": "attention_backend" in fields,
        "attention_config_arg": "attention_config" in fields,
    }
    module = None
    for name in ("vllm.model_executor.determinism.batch_invariant", "vllm.model_executor.layers.batch_invariant"):
        try:
            module = __import__(name, fromlist=["x"])
            detail["batch_invariant_module"] = name
            break
        except ImportError:
            continue
    try:
        from vllm.model_executor.determinism import batch_invariant_configs as bic
        from vllm.platforms import current_platform

        family = bic._get_tuned_matmul_arch_family(current_platform.get_device_capability())
        detail["tuned_matmul_family"] = family
        detail["tuned_matmul_table_for_this_gpu"] = bool(family and family in bic._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS)
    except Exception as exc:  # noqa: BLE001
        detail["tuned_matmul_family"] = f"n/a ({type(exc).__name__})"
    return module is not None, detail


def _memory() -> tuple[bool | None, Any]:
    mem = meminfo_gb()
    free, total = torch.cuda.mem_get_info()
    detail = {
        "MemTotal_GB": round(mem.get("MemTotal", 0), 1),
        "MemAvailable_GB": round(mem.get("MemAvailable", 0), 1),
        "PageCache_GB": round(mem.get("Cached", 0), 1),
        "cudaMemGetInfo_free_GB": round(free / 1e9, 1),
        "cudaMemGetInfo_total_GB": round(total / 1e9, 1),
    }
    if detail["PageCache_GB"] > 10:
        detail["hint"] = (
            "large page cache - on unified memory it reduces what CUDA reports as free. Run on the host: "
            "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
        )
    return None, detail


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    checks = [_check("cuda", _cuda)]
    if checks[0]["ok"]:
        checks += [
            _check("triton_compile_sm", _triton),
            _check("batch_invariant_ops", _official),
            _check("extra_triton_ops", _extra_ops),
            _check("custom_fixed_chunk_mode", _custom),
            _check("vllm", _vllm),
            _check("memory", _memory, critical=False),
        ]
    for c in checks:
        status = "INFO" if c["ok"] is None else ("OK" if c["ok"] else "FAIL")
        ctx.log.info("preflight %-24s %s  %s", c["name"], status, str(c["detail"])[:300])
    return {
        "ok": all(c["ok"] for c in checks if c["critical"]),
        "checks": checks,
        "packages": package_versions(),
        "nvidia_smi": nvidia_smi(),
        "smem_optin_bytes": device_smem_optin() if checks[0]["ok"] else None,
    }
