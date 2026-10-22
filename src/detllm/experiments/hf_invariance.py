"""Model-level batch invariance with an in-process HF transformers model.

The target prompt is run alone (batch size 1) and then inside batches of real-text filler rows of
exactly the same length (so no padding, identical positions), with the target as the first and as
the last row. Two checks:

* forward: the target's logits at every prompt position must be bitwise identical;
* decode:  greedy decoding must produce identical tokens AND bitwise-identical logprobs.

Expected: ``baseline`` / ``baseline_eager`` vary with batch size; ``official`` (library only)
fixes linear layers + RMSNorm but not attention, so it can still vary; ``official_full`` and
``custom`` should be bitwise invariant end to end.
"""

from __future__ import annotations

from typing import Any

import torch

from detllm.analysis.divergence import compare_to_reference
from detllm.engines.hf_engine import HFEngine
from detllm.experiments.op_invariance import bitwise_equal, max_abs_diff
from detllm.prompts import equal_length_fillers, render_chat_ids, resolve_prompt
from detllm.utils import evict_file_cache, hf_model_cache_dir


def _batch(target: list[int], fillers: list[list[int]], size: int, position: str) -> tuple[list[list[int]], int]:
    rows = [list(f) for f in fillers[: size - 1]]
    row = 0 if position == "first" else size - 1
    rows.insert(row, list(target))
    return rows, row


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    preset = ctx.require_model()
    batch_sizes = [int(b) for b in ctx.param("batch_sizes", [1, 2, 3, 4, 8, 16, 32])]
    positions = list(ctx.param("target_positions", ["first", "last"]))
    max_new_tokens = int(ctx.param("max_new_tokens", 128))
    chat_kwargs = ctx.param("chat_template_kwargs", preset.chat_template_kwargs)

    with ctx.startup_lock(enabled=bool(ctx.param("serialize_startup", False))):
        engine = HFEngine(preset, ctx.job.mode, attn_implementation=ctx.param("attn_implementation"))
    freed = evict_file_cache(hf_model_cache_dir(preset.hf_id))
    ctx.log.info("model loaded (%s); dropped %.1f GB of page cache", engine.info(), freed / 1e9)

    tok = engine.tokenizer
    prompt_text = resolve_prompt(ctx.param("prompt"))
    target = render_chat_ids(tok, prompt_text, chat_kwargs)
    fillers = equal_length_fillers(
        tok, len(target), max(batch_sizes) - 1, seed=ctx.seed, chat_template_kwargs=chat_kwargs
    )

    # ---- forward pass (teacher-forced logits)
    ref_logits = engine.forward_logits([target], rows=[0])[0]
    forward = []
    for size in batch_sizes:
        for pos in positions if size > 1 else ["first"]:
            batch, row = _batch(target, fillers, size, pos)
            logits = engine.forward_logits(batch, rows=[row])[0]
            forward.append(
                {
                    "batch_size": size,
                    "position": pos,
                    "bitwise_equal": bitwise_equal(logits, ref_logits),
                    "max_abs_diff": max_abs_diff(logits, ref_logits),
                    "argmax_agreement": float((logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()),
                }
            )
            del logits
    del ref_logits
    torch.cuda.empty_cache()

    # ---- greedy decoding
    ref_tokens, ref_lps = (x[0] for x in engine.greedy([target], max_new_tokens))
    again_tokens, again_lps = (x[0] for x in engine.greedy([target], max_new_tokens))
    run_to_run = compare_to_reference(ref_tokens, ref_lps, again_tokens, again_lps)
    decode = []
    examples = []
    for size in batch_sizes:
        for pos in positions if size > 1 else ["first"]:
            batch, row = _batch(target, fillers, size, pos)
            toks, lps = engine.greedy(batch, max_new_tokens)
            cmp = compare_to_reference(ref_tokens, ref_lps, toks[row], lps[row])
            decode.append({"batch_size": size, "position": pos, **cmp})
            if not cmp["tokens_equal"] and len(examples) < 3:
                fd = cmp["first_token_divergence"]
                examples.append(
                    {
                        "batch_size": size,
                        "position": pos,
                        "token_number": fd + 1,
                        "reference": tok.decode(ref_tokens[max(0, fd - 16) : fd + 12]),
                        "batched": tok.decode(toks[row][max(0, fd - 16) : fd + 12]),
                    }
                )

    ctx.save_json(
        "generations.json",
        {"prompt": prompt_text, "prompt_ids": target, "reference_tokens": ref_tokens, "reference_text": tok.decode(ref_tokens)},
    )
    info = engine.info()
    engine.close()
    return {
        "model": preset.key,
        "hf_id": preset.hf_id,
        "mode": ctx.job.mode,
        "attn_implementation": info["attn_implementation"],
        "prompt_tokens": len(target),
        "max_new_tokens": max_new_tokens,
        "forward": forward,
        "forward_invariant": all(r["bitwise_equal"] for r in forward),
        "decode": decode,
        "decode_invariant": all(r["identical"] for r in decode),
        "decode_token_invariant": all(r["tokens_equal"] for r in decode),
        "first_token_divergence": min(
            (r["first_token_divergence"] for r in decode if r["first_token_divergence"] is not None), default=None
        ),
        "run_to_run": run_to_run,
        "examples": examples,
        "kernels": info["kernels"],
    }
