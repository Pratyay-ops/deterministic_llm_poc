"""vLLM offline engine (``vllm.LLM``) plus engine-argument helpers shared with the API server.

The same harness must work with the NGC image (vLLM 0.29) and upstream images (vLLM 0.31+), so
optional engine arguments are feature-detected from ``EngineArgs`` instead of assumed.

Batch-invariant mode is switched on by ``VLLM_BATCH_INVARIANT=1`` (set by the scheduler before
the process starts). What vLLM then does on GB10 (compute capability 12.x):
* GEMMs: cuBLASLt with a ~1 KiB workspace (no split-K); vLLM >= 0.31 also has Triton tiles tuned
  for sm120-class GPUs (Qwen3-1.7B/4B/8B shapes).
* Triton overrides for softmax / log_softmax / mean / bmm, batch-invariant RMSNorm and attention.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any

from detllm.config import ModelPreset
from detllm.engines.base import Completion
from detllm.jobs import RetryLater

log = logging.getLogger(__name__)

# Default attention backend in batch-invariant mode: vLLM tests batch invariance with FLASH_ATTN,
# TRITON_ATTN and FLEX_ATTENTION on CUDA; TRITON_ATTN is pure Triton and therefore available on
# sm_121 in every image. Baseline jobs keep vLLM's own default unless told otherwise.
DEFAULT_BI_ATTENTION_BACKEND = "TRITON_ATTN"

# Transient start-up shortages only. "kv cache memory" is not here: vLLM prints an INFO line with it on
# every start when kv_cache_memory_bytes is set, and "...larger than the available KV cache memory" /
# "no available memory for the cache blocks" are configuration errors that a retry cannot fix.
_MEMORY_ERRORS = (
    "less than desired gpu memory utilization",
    "not enough memory",
    "insufficient memory",
    "out of memory",
    "free memory on device",
)
# Boolean engine args whose default is True, so False must be passed explicitly on the CLI.
_DEFAULT_TRUE_BOOLS = {"enable_prefix_caching", "enable_chunked_prefill"}


def vllm_version() -> str:
    import vllm

    return getattr(vllm, "__version__", "unknown")


@lru_cache(maxsize=1)
def engine_arg_fields() -> frozenset[str]:
    from vllm.engine.arg_utils import EngineArgs

    return frozenset(f.name for f in dataclasses.fields(EngineArgs))


def supports_kv_cache_bytes() -> bool:
    return "kv_cache_memory_bytes" in engine_arg_fields()


def device_total_gb() -> float:
    import torch

    _free, total = torch.cuda.mem_get_info()
    return total / 1e9


def is_memory_error(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _MEMORY_ERRORS)


def engine_kwargs(preset: ModelPreset, params: dict[str, Any], *, mem_gb: float, mode: str) -> dict[str, Any]:
    """Engine arguments for ``LLM(**kw)``; ``to_cli_flags`` turns them into ``vllm serve`` flags."""
    fields = engine_arg_fields()
    total = device_total_gb()
    kw: dict[str, Any] = {
        "model": preset.hf_id,
        "dtype": preset.dtype,
        "seed": int(params.get("seed", 0)),
        "max_model_len": int(params.get("max_model_len", preset.max_model_len)),
        # Per-instance share of the unified memory; several engines can share the GPU.
        "gpu_memory_utilization": round(min(max(mem_gb / total, 0.05), 0.95), 4),
        "max_num_seqs": int(params.get("max_num_seqs", 64)),
        "enable_prefix_caching": bool(params.get("enable_prefix_caching", False)),
        "enforce_eager": bool(params.get("enforce_eager", False)),
    }
    if params.get("max_num_batched_tokens"):
        kw["max_num_batched_tokens"] = int(params["max_num_batched_tokens"])
    kv_gb = params.get("kv_cache_gb")
    if kv_gb and "kv_cache_memory_bytes" in fields:
        # Fixed KV size: no reliance on start-up profiling, which concurrent jobs would disturb.
        kw["kv_cache_memory_bytes"] = int(float(kv_gb) * 1e9)
    backend = params.get("attention_backend")
    if backend is None and mode == "batch_invariant":
        backend = params.get("bi_attention_backend", DEFAULT_BI_ATTENTION_BACKEND)
    if backend:
        if "attention_backend" in fields:
            kw["attention_backend"] = backend
        elif "attention_config" in fields:
            kw["attention_config"] = {"backend": backend}
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = backend
    kw.update(params.get("engine_kwargs") or {})
    return kw


def to_cli_flags(kw: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for key, value in kw.items():
        if key == "model" or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
            elif key in _DEFAULT_TRUE_BOOLS:
                flags.append("--no-" + key.replace("_", "-"))
        elif isinstance(value, (dict, list)):
            flags += [flag, json.dumps(value)]
        else:
            flags += [flag, str(value)]
    return flags


def make_sampling_params(spec: dict[str, Any]):  # noqa: ANN201 - vllm.SamplingParams
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": float(spec.get("temperature", 0.0)),
        "top_p": float(spec.get("top_p", 1.0)),
        "max_tokens": int(spec.get("max_tokens", 128)),
        "ignore_eos": bool(spec.get("ignore_eos", False)),
    }
    if spec.get("seed") is not None:
        kwargs["seed"] = int(spec["seed"])
    if spec.get("logprobs"):
        kwargs["logprobs"] = 0  # only the sampled token's logprob
    return SamplingParams(**kwargs)


def to_completion(index: int, request_output: Any, prompt_id: str) -> Completion:
    out = request_output.outputs[0]
    token_ids = [int(t) for t in out.token_ids]
    logprobs = None
    if out.logprobs is not None:
        logprobs = []
        for pos, tid in enumerate(token_ids):
            try:  # list[dict[token_id, Logprob]] or vLLM's FlatLogprobs (same indexing)
                logprobs.append(float(out.logprobs[pos][tid].logprob))
            except (KeyError, IndexError, TypeError, AttributeError):
                logprobs.append(float("nan"))
    return Completion(
        index=index,
        prompt_id=prompt_id,
        token_ids=token_ids,
        text=out.text,
        logprobs=logprobs,
        finish_reason=out.finish_reason,
        num_prompt_tokens=len(request_output.prompt_token_ids or []),
    )


def attention_backend_from_log(text: str) -> str | None:
    """Best effort: which attention backend vLLM reported choosing."""
    matches = re.findall(r"Using ([A-Za-z0-9_]+) (?:attention )?backend", text)
    return matches[-1] if matches else None


class VllmOfflineEngine:
    def __init__(self, preset: ModelPreset, params: dict[str, Any], *, mem_gb: float, mode: str):
        from vllm import LLM

        self.preset = preset
        self.mode = mode
        self.kwargs = engine_kwargs(preset, params, mem_gb=mem_gb, mode=mode)
        log.info("starting vLLM %s offline engine (%s): %s", vllm_version(), mode, self.kwargs)
        t0 = time.time()
        try:
            self.llm = LLM(**self.kwargs)
        except Exception as exc:  # noqa: BLE001
            if is_memory_error(str(exc)):
                raise RetryLater(f"vLLM could not reserve memory yet: {exc}") from exc
            raise
        self.startup_s = time.time() - t0
        log.info("vLLM engine ready in %.0fs", self.startup_s)

    def generate(
        self,
        prompts: list[list[int]],
        sampling: dict[str, Any] | list[dict[str, Any]],
        *,
        prompt_ids: list[str] | None = None,
    ) -> list[Completion]:
        from vllm.inputs import TokensPrompt

        specs = sampling if isinstance(sampling, list) else [sampling] * len(prompts)
        params = [make_sampling_params(s) for s in specs]
        inputs = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]
        outputs = self.llm.generate(inputs, params, use_tqdm=False)
        return [
            to_completion(i, o, prompt_ids[i] if prompt_ids else str(i)) for i, o in enumerate(outputs)
        ]

    def info(self) -> dict[str, Any]:
        return {"vllm_version": vllm_version(), "engine_kwargs": self.kwargs, "startup_s": round(self.startup_s, 1)}

    def close(self) -> None:
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:  # noqa: BLE001
            pass
