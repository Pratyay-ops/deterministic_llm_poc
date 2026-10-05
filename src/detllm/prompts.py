"""Prompt files, chat templating and equal-length filler batches."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from detllm.utils import resolve_path

DEFAULT_PROMPT = {"file": "configs/prompts/feynman.txt"}
DEFAULT_FILLERS = "configs/prompts/fillers.txt"


def load_prompt_file(path: str) -> list[str]:
    """One prompt per non-empty line; lines starting with '#' are comments."""
    text = resolve_path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def resolve_prompt(spec: Any = None) -> str:
    """Accepts a literal string, ``{"text": ...}`` or ``{"file": path, "index": i}``."""
    spec = DEFAULT_PROMPT if spec is None else spec
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        if "text" in spec:
            return str(spec["text"])
        if "file" in spec:
            prompts = load_prompt_file(spec["file"])
            return prompts[int(spec.get("index", 0))]
    raise ValueError(f"cannot interpret prompt spec {spec!r}")


@lru_cache(maxsize=8)
def load_tokenizer(hf_id: str):  # noqa: ANN201 - transformers type
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id)


def render_chat(tokenizer, user_text: str, chat_template_kwargs: dict[str, Any] | None = None) -> str:
    """Apply the model's chat template (e.g. ``enable_thinking=False`` for Qwen3 non-thinking mode)."""
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {})
    )


def render_chat_ids(tokenizer, user_text: str, chat_template_kwargs: dict[str, Any] | None = None) -> list[int]:
    """Token ids of the chat-templated prompt. Engines receive ids, so every engine and mode sees
    exactly the same input tokens."""
    text = render_chat(tokenizer, user_text, chat_template_kwargs)
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def equal_length_fillers(
    tokenizer,
    length: int,
    count: int,
    *,
    seed: int = 0,
    texts: list[str] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[list[int]]:
    """``count`` real-text token sequences of exactly ``length`` tokens (truncated or repeated).

    Equal lengths mean an HF batch needs no padding, so the target row is computed on exactly
    the same tokens and positions at every batch size.
    """
    texts = texts or load_prompt_file(DEFAULT_FILLERS)
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    out: list[list[int]] = []
    for i in range(count):
        ids = render_chat_ids(tokenizer, texts[order[i % len(order)]], chat_template_kwargs)
        reps = -(-length // max(len(ids), 1))
        out.append((ids * reps)[:length])
    return out


def filler_prompt_ids(
    tokenizer,
    *,
    seed: int = 0,
    min_tokens: int = 16,
    max_tokens: int = 1024,
    count: int = 64,
    texts: list[str] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Ragged batch partners for engine-level tests: real prompts, some stretched to long lengths
    so batches mix short decodes with (chunked) long prefills."""
    texts = texts or load_prompt_file(DEFAULT_FILLERS)
    rng = random.Random(seed)
    out: list[list[int]] = []
    for i in range(count):
        text = texts[rng.randrange(len(texts))]
        target = rng.randint(min_tokens, max_tokens)
        base = tokenizer(text, add_special_tokens=False)["input_ids"]
        body = (base * (-(-target // max(len(base), 1))))[:target]
        user_text = tokenizer.decode(body) if i % 2 else text  # half natural, half stretched
        out.append(render_chat_ids(tokenizer, user_text, chat_template_kwargs))
    return out
