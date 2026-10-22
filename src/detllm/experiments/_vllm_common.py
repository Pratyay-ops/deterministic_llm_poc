"""Helpers shared by the vLLM experiments."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from detllm.engines.vllm_engine import VllmOfflineEngine, attention_backend_from_log, supports_kv_cache_bytes
from detllm.engines.vllm_server import VllmServer, completion_payload, send_requests
from detllm.utils import evict_file_cache, hf_model_cache_dir


def serialize_startup(params: dict[str, Any]) -> bool:
    """Engines started side by side are fine when the KV cache size is given explicitly; only
    profiling-based sizing (old vLLM) needs exclusive start-up."""
    value = params.get("serialize_startup", "auto")
    if value == "auto":
        return not supports_kv_cache_bytes()
    return bool(value)


def drop_model_page_cache(ctx, hf_id: str) -> None:  # noqa: ANN001
    freed = evict_file_cache(hf_model_cache_dir(hf_id))
    ctx.log.info("dropped %.1f GB of page cache for %s (unified memory)", freed / 1e9, hf_id)


@contextlib.contextmanager
def running_server(ctx, *, warmup_prompt: list[int] | None = None, warmup_requests: int = 8) -> Iterator[VllmServer]:  # noqa: ANN001
    preset = ctx.require_model()
    server = VllmServer(
        preset, ctx.job.params, mem_gb=ctx.job.mem_gb, mode=ctx.job.mode, log_path=ctx.path("vllm_server.log")
    )
    try:
        with ctx.startup_lock(enabled=serialize_startup(ctx.job.params)):
            server.start()
            if warmup_prompt is not None:
                payloads = [
                    completion_payload(server.served_model_name, warmup_prompt, max_tokens=16, seed=ctx.seed)
                    for _ in range(warmup_requests)
                ]
                results, _ = send_requests(server.base_url, "completions", payloads)
                bad = [r for r in results if not r.get("ok")]
                if bad:
                    raise RuntimeError(f"warm-up requests failed: {bad[0].get('error')}")
        drop_model_page_cache(ctx, preset.hf_id)
        yield server
    finally:
        server.stop()


@contextlib.contextmanager
def offline_engine(ctx) -> Iterator[VllmOfflineEngine]:  # noqa: ANN001
    preset = ctx.require_model()
    with ctx.startup_lock(enabled=serialize_startup(ctx.job.params)):
        engine = VllmOfflineEngine(preset, ctx.job.params, mem_gb=ctx.job.mem_gb, mode=ctx.job.mode)
    drop_model_page_cache(ctx, preset.hf_id)
    try:
        yield engine
    finally:
        engine.close()


def backend_in_logs(ctx, server: VllmServer | None = None) -> str | None:  # noqa: ANN001
    path = server.log_path if server is not None else ctx.path("job.log")
    try:
        return attention_backend_from_log(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
