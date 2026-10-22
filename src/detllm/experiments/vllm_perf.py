"""Blog experiment "Performance".

Blog setup: an API server on one GPU running Qwen3-8B, 1000 requests with output lengths between
90 and 110 tokens. vLLM default 26 s, unoptimized deterministic vLLM 55 s, with the improved
attention kernel 42 s. Run this job ``exclusive`` (alone on the GPU) - concurrent jobs would
distort the timing.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from detllm.engines.vllm_server import completion_payload, parse_choice, send_requests
from detllm.experiments._vllm_common import backend_in_logs, running_server
from detllm.prompts import load_prompt_file, load_tokenizer, render_chat_ids

BLOG = {"model": "Qwen/Qwen3-8B", "n": 1000, "output_tokens": "90-110", "default_s": 26, "deterministic_s": 55, "improved_attention_s": 42}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    preset = ctx.require_model()
    p = ctx.job.params
    n = int(p.get("n", 1000))
    lo, hi = (int(x) for x in p.get("output_len_range", [90, 110]))
    tok = load_tokenizer(preset.hf_id)
    texts = load_prompt_file(p.get("prompt_file", "configs/prompts/fillers.txt"))
    prompts = [render_chat_ids(tok, t, preset.chat_template_kwargs) for t in texts]
    rng = random.Random(ctx.seed)
    lengths = [rng.randint(lo, hi) for _ in range(n)]

    with running_server(ctx, warmup_prompt=prompts[0], warmup_requests=int(p.get("warmup_requests", 64))) as server:
        payloads = [
            completion_payload(
                server.served_model_name,
                prompts[i % len(prompts)],
                max_tokens=lengths[i],
                seed=ctx.seed,
                logprobs=False,
                ignore_eos=True,
            )
            for i in range(n)
        ]
        ctx.log.info("timing %d requests, %d-%d output tokens each", n, lo, hi)
        results, wall = send_requests(server.base_url, "completions", payloads, seed=ctx.seed)
        info = {**server.info(), "attention_backend_log": backend_in_logs(ctx, server)}

    ok = [r for r in results if r.get("ok")]
    if len(ok) < n:
        first_error = next((r.get("error") for r in results if not r.get("ok")), None)
        raise RuntimeError(f"{n - len(ok)}/{n} requests failed, timing would be meaningless; first error: {first_error}")
    out_tokens = 0
    for r, length in zip(results, lengths):
        if r.get("ok"):
            ids = parse_choice(r["json"], "completions")["token_ids"]
            out_tokens += len(ids) if ids is not None else length
    latencies = [r["latency_s"] for r in ok]
    return {
        "model": preset.key,
        "hf_id": preset.hf_id,
        "mode": ctx.job.mode,
        "n": n,
        "n_ok": len(ok),
        "output_len_range": [lo, hi],
        "wall_time_s": round(wall, 2),
        "output_tokens": out_tokens,
        "output_tokens_per_s": round(out_tokens / wall, 1) if wall else None,
        "latency_s": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": _pct(latencies, 0.5),
            "p90": _pct(latencies, 0.9),
            "p99": _pct(latencies, 0.99),
        },
        "engine_info": info,
        "blog_reference": BLOG,
    }
