"""Blog experiment "How nondeterministic are completions?".

Blog setup: Qwen/Qwen3-235B-A22B-Instruct-2507, 1000 completions at temperature 0 of
"Tell me about Richard Feynman" (non-thinking), 1000 tokens each -> 80 unique completions (most
common one 78 times), all identical for the first 102 tokens; with batch-invariant kernels all
1000 were identical.

Default here: the same prompt and sampling on a small model, sent as N concurrent requests to a
``vllm serve`` instance (what a real user of a server sees; also how batch_invariant_ops'
deterministic_vllm_inference.py does it). ``engine: offline`` instead submits everything to
``vllm.LLM`` at once, optionally mixed with filler requests of random lengths so that requests
start at different times. (Identical prompts submitted all at once tend to march through the
engine in lockstep and therefore see identical batches - which hides the effect.)
"""

from __future__ import annotations

import random
from typing import Any

from detllm.analysis.divergence import divergence_example, summarize_sequences
from detllm.engines.base import Completion
from detllm.engines.vllm_server import chat_payload, completion_payload, parse_choice, send_requests
from detllm.experiments._vllm_common import backend_in_logs, offline_engine, running_server
from detllm.prompts import filler_prompt_ids, load_tokenizer, render_chat_ids, resolve_prompt
from detllm.utils import JsonlWriter

BLOG = {
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "n": 1000,
    "max_tokens": 1000,
    "n_unique_default": 80,
    "modal_count_default": 78,
    "first_divergence_token_number": 103,
    "n_unique_batch_invariant": 1,
}


def _server_run(ctx, prompt_text: str, prompt_ids: list[int], n: int, max_tokens: int) -> tuple[list[Completion], dict]:  # noqa: ANN001
    p = ctx.job.params
    api = p.get("api", "completions")
    with running_server(ctx, warmup_prompt=prompt_ids) as server:
        if api == "chat":
            make = lambda: chat_payload(  # noqa: E731
                server.served_model_name,
                prompt_text,
                max_tokens=max_tokens,
                seed=ctx.seed,
                chat_template_kwargs=p.get("chat_template_kwargs", ctx.model.chat_template_kwargs),
                logprobs=bool(p.get("logprobs", True)),
            )
        else:
            make = lambda: completion_payload(  # noqa: E731
                server.served_model_name,
                prompt_ids,
                max_tokens=max_tokens,
                seed=ctx.seed,
                logprobs=bool(p.get("logprobs", True)),
                ignore_eos=bool(p.get("ignore_eos", False)),
            )
        payloads = [make() for _ in range(n)]
        ctx.log.info("sending %d concurrent %s requests (max_tokens=%d)", n, api, max_tokens)
        results, wall = send_requests(
            server.base_url,
            api,
            payloads,
            concurrency=int(p.get("concurrency") or n),
            jitter_s=float(p.get("arrival_jitter_s", 0.0)),
            seed=ctx.seed,
        )
        info = {**server.info(), "api": api, "wall_time_s": round(wall, 2), "attention_backend_log": backend_in_logs(ctx, server)}
    completions = []
    for i, r in enumerate(results):
        if r.get("ok"):
            parsed = parse_choice(r["json"], api)
            completions.append(Completion(index=i, prompt_id="target", latency_s=r["latency_s"], **parsed))
        else:
            completions.append(Completion(index=i, prompt_id="target", token_ids=None, text=None, error=r.get("error")))
    return completions, info


def _offline_run(ctx, prompt_ids: list[int], n: int, max_tokens: int) -> tuple[list[Completion], dict]:  # noqa: ANN001
    p = ctx.job.params
    n_fill = int(p.get("filler_requests", n // 2))
    rng = random.Random(ctx.seed)
    with offline_engine(ctx) as engine:
        fillers = filler_prompt_ids(
            load_tokenizer(ctx.model.hf_id),
            seed=ctx.seed,
            count=max(n_fill, 1),
            max_tokens=int(p.get("filler_max_prompt_tokens", 512)),
            chat_template_kwargs=ctx.model.chat_template_kwargs,
        )
        target_spec = {"temperature": 0.0, "max_tokens": max_tokens, "seed": ctx.seed, "logprobs": bool(p.get("logprobs", True))}
        items = [("target", prompt_ids, target_spec) for _ in range(n)]
        items += [
            ("filler", fillers[i % len(fillers)], {"temperature": 0.0, "max_tokens": rng.randint(8, max_tokens), "seed": ctx.seed})
            for i in range(n_fill)
        ]
        rng.shuffle(items)
        outs = engine.generate([it[1] for it in items], [it[2] for it in items], prompt_ids=[it[0] for it in items])
        info = {**engine.info(), "filler_requests": n_fill, "attention_backend_log": backend_in_logs(ctx)}
    targets = [c for c in outs if c.prompt_id == "target"]
    for i, c in enumerate(targets):
        c.index = i
    return targets, info


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    preset = ctx.require_model()
    p = ctx.job.params
    n = int(p.get("n", 1000))
    max_tokens = int(p.get("max_tokens", 1000))
    engine = p.get("engine", "server")
    tok = load_tokenizer(preset.hf_id)
    prompt_text = resolve_prompt(p.get("prompt"))
    prompt_ids = render_chat_ids(tok, prompt_text, p.get("chat_template_kwargs", preset.chat_template_kwargs))

    if engine == "server":
        completions, info = _server_run(ctx, prompt_text, prompt_ids, n, max_tokens)
    elif engine == "offline":
        completions, info = _offline_run(ctx, prompt_ids, n, max_tokens)
    else:
        raise ValueError(f"engine must be 'server' or 'offline', got {engine!r}")

    with JsonlWriter(ctx.path("completions.jsonl")) as out:
        for c in completions:
            out.write(c.to_dict())

    use_tokens = all(c.token_ids is not None for c in completions if c.error is None)
    seqs = [(c.token_ids if use_tokens else c.text) if c.error is None else None for c in completions]
    logprobs = [c.logprobs for c in completions] if use_tokens else None
    summary = summarize_sequences(seqs, logprobs=logprobs)

    example = None
    if summary["n_unique"] > 1 and summary["unique"][1]["first_divergence_vs_modal"] is not None:
        modal = seqs[summary["modal_index"]]
        other = seqs[summary["unique"][1]["example_index"]]
        decode = (lambda ids: tok.decode(ids)) if use_tokens else None
        example = divergence_example(decode, modal, other, summary["unique"][1]["first_divergence_vs_modal"])

    modal_text = completions[summary["modal_index"]].text if summary["modal_index"] is not None else None
    ctx.log.info(
        "%s/%s: %d unique of %d (modal x%d), earliest divergence at %s %s",
        preset.key,
        ctx.job.mode,
        summary["n_unique"],
        summary["n_valid"],
        summary["modal_count"],
        summary["unit"][:-1],
        summary["earliest_divergence"],
    )
    return {
        "model": preset.key,
        "hf_id": preset.hf_id,
        "mode": ctx.job.mode,
        "engine": engine,
        "prompt": prompt_text,
        "prompt_tokens": len(prompt_ids),
        "n": n,
        "max_tokens": max_tokens,
        "compared_on": "token_ids+logprobs" if use_tokens else "text",
        **summary,
        "divergence_example": example,
        "modal_text_head": (modal_text or "")[:600],
        "mean_output_tokens": (
            sum(len(c.token_ids) for c in completions if c.token_ids) / max(1, sum(1 for c in completions if c.token_ids))
        ),
        "engine_info": info,
        "blog_reference": BLOG,
    }
