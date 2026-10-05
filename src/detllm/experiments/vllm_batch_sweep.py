"""Engine-level batch invariance: the same "needle" request alone vs. inside batches.

Mirrors vLLM's own tests (tests/v1/determinism/test_batch_invariance.py): a reference generation
for the needle at batch size 1 (run twice: run-to-run check), then for every batch size the needle
is placed at a random position among filler requests of mixed lengths (short prompts and long,
chunked prefills) with random output lengths, so the batch around the needle keeps changing while
it decodes. The needle's tokens and per-token logprobs are compared bitwise with the reference.
"""

from __future__ import annotations

import random
from typing import Any

from detllm.analysis.divergence import compare_to_reference
from detllm.experiments._vllm_common import backend_in_logs, offline_engine
from detllm.prompts import filler_prompt_ids, load_tokenizer, render_chat_ids, resolve_prompt
from detllm.utils import JsonlWriter


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    preset = ctx.require_model()
    p = ctx.job.params
    batch_sizes = [int(b) for b in p.get("batch_sizes", [1, 2, 4, 8, 16, 32, 64, 128])]
    trials = int(p.get("trials", 2))
    max_tokens = int(p.get("max_tokens", 256))
    chat_kwargs = p.get("chat_template_kwargs", preset.chat_template_kwargs)
    tok = load_tokenizer(preset.hf_id)
    needle = render_chat_ids(tok, resolve_prompt(p.get("prompt")), chat_kwargs)
    pool = filler_prompt_ids(
        tok,
        seed=ctx.seed,
        count=int(p.get("filler_pool", 96)),
        max_tokens=int(p.get("filler_max_prompt_tokens", 768)),
        chat_template_kwargs=chat_kwargs,
    )
    needle_spec = {"temperature": 0.0, "max_tokens": max_tokens, "seed": ctx.seed, "logprobs": True}
    rng = random.Random(ctx.seed)

    rows = []
    with offline_engine(ctx) as engine, JsonlWriter(ctx.path("needle_outputs.jsonl")) as out:
        ref = engine.generate([needle], needle_spec)[0]
        again = engine.generate([needle], needle_spec)[0]
        run_to_run = compare_to_reference(ref.token_ids, ref.logprobs, again.token_ids, again.logprobs)
        out.write({"batch_size": 1, "trial": "reference", **ref.to_dict()})
        for size in batch_sizes:
            for trial in range(trials if size > 1 else 1):
                position = rng.randrange(size)
                prompts = [pool[rng.randrange(len(pool))] for _ in range(size - 1)]
                specs = [
                    {"temperature": 0.0, "max_tokens": rng.randint(8, 2 * max_tokens), "seed": ctx.seed}
                    for _ in range(size - 1)
                ]
                prompts.insert(position, needle)
                specs.insert(position, needle_spec)
                result = engine.generate(prompts, specs)[position]
                cmp = compare_to_reference(ref.token_ids, ref.logprobs, result.token_ids, result.logprobs)
                rows.append({"batch_size": size, "trial": trial, "position": position, **cmp})
                out.write({"batch_size": size, "trial": trial, "position": position, **result.to_dict()})
                ctx.log.info("batch %4d trial %d: identical=%s first_token_divergence=%s", size, trial, cmp["identical"], cmp["first_token_divergence"])
        info = {**engine.info(), "attention_backend_log": backend_in_logs(ctx)}

    token_mismatch = [r for r in rows if not r["tokens_equal"]]
    logprob_mismatch = [r for r in rows if r["tokens_equal"] and r.get("logprobs_bitwise_equal") is False]
    return {
        "model": preset.key,
        "hf_id": preset.hf_id,
        "mode": ctx.job.mode,
        "max_tokens": max_tokens,
        "reference_tokens": len(ref.token_ids or []),
        "reference_text_head": (ref.text or "")[:400],
        "run_to_run": run_to_run,
        "results": rows,
        "n_trials": len(rows),
        "n_identical": sum(1 for r in rows if r["identical"]),
        "n_token_mismatch": len(token_mismatch),
        "n_logprob_only_mismatch": len(logprob_mismatch),
        "variant_batch_sizes": sorted({r["batch_size"] for r in rows if not r["identical"]}),
        "earliest_token_divergence": min((r["first_token_divergence"] for r in token_mismatch), default=None),
        "engine_info": info,
    }
