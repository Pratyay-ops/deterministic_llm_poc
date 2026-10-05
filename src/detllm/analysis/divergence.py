"""Compare generated sequences (token ids, or text as a fallback) and their logprobs.

"Identical" in this project means bitwise identical: same token ids AND the same float bits for
every logprob. Two runs can agree on every token while their logprobs differ in the last bits -
that is still nondeterminism, just not yet large enough to flip an argmax.
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

Seq = Sequence[int] | str


def first_divergence(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    """0-based index of the first differing element; ``len(shorter)`` if one is a strict prefix
    of the other; ``None`` if identical."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def float_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def bitwise_equal(a: Sequence[float] | None, b: Sequence[float] | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and all(float_bits(x) == float_bits(y) for x, y in zip(a, b))


def first_bitwise_mismatch(a: Sequence[float], b: Sequence[float]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if float_bits(x) != float_bits(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        if math.isfinite(x) and math.isfinite(y):
            worst = max(worst, abs(x - y))
        elif not (x == y or (math.isnan(x) and math.isnan(y))):
            return math.inf
    return worst


def _key(seq: Seq) -> tuple[int, ...] | str:
    return seq if isinstance(seq, str) else tuple(seq)


def summarize_sequences(
    seqs: Sequence[Seq | None],
    *,
    logprobs: Sequence[Sequence[float] | None] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Unique-output statistics for N generations of the same request.

    ``earliest_divergence`` is the first position (0-based) at which any generation differs from
    the most common one - the blog reports "identical for the first 102 tokens, diverging at the
    103rd", i.e. index 102.
    """
    valid = [(i, s) for i, s in enumerate(seqs) if s is not None]
    counts: Counter = Counter()
    first_index: dict[Any, int] = {}
    for i, s in valid:
        k = _key(s)
        counts[k] += 1
        first_index.setdefault(k, i)
    ranked = counts.most_common()
    summary: dict[str, Any] = {
        "n": len(seqs),
        "n_valid": len(valid),
        "n_failed": len(seqs) - len(valid),
        "n_unique": len(ranked),
        "modal_count": ranked[0][1] if ranked else 0,
        "counts": [c for _, c in ranked],
        "earliest_divergence": None,
        "divergence_positions": [],
        "unique": [],
        "modal_index": None,
        "unit": "chars" if valid and isinstance(valid[0][1], str) else "tokens",
    }
    if not ranked:
        return summary
    modal_key = ranked[0][0]
    positions = []
    for rank, (k, c) in enumerate(ranked):
        fd = None if rank == 0 else first_divergence(modal_key, k)
        if fd is not None:
            positions.append(fd)
        if rank < top_k:
            summary["unique"].append(
                {"rank": rank, "count": c, "length": len(k), "first_divergence_vs_modal": fd, "example_index": first_index[k]}
            )
    summary["modal_index"] = first_index[modal_key]
    summary["divergence_positions"] = sorted(positions)
    summary["earliest_divergence"] = min(positions) if positions else None
    summary["modal_length"] = len(modal_key)

    if logprobs is not None:
        variants: Counter = Counter()
        for i, s in valid:
            lp = logprobs[i] if i < len(logprobs) else None
            if _key(s) == modal_key and lp is not None:
                variants[tuple(float_bits(x) for x in lp)] += 1
        summary["modal_logprob_variants"] = len(variants)
        summary["modal_logprob_counts"] = [c for _, c in variants.most_common(top_k)]
    return summary


def compare_to_reference(
    ref_tokens: Sequence[int] | None,
    ref_logprobs: Sequence[float] | None,
    tokens: Sequence[int] | None,
    logprobs: Sequence[float] | None,
) -> dict[str, Any]:
    """Compare one generation against a reference generation of the same request."""
    if ref_tokens is None or tokens is None:
        return {"tokens_equal": None, "first_token_divergence": None, "logprobs_bitwise_equal": None}
    fd = first_divergence(ref_tokens, tokens)
    out: dict[str, Any] = {"tokens_equal": fd is None, "first_token_divergence": fd}
    if ref_logprobs is not None and logprobs is not None:
        upto = len(ref_tokens) if fd is None else fd  # compare logprobs where the tokens agree
        a, b = list(ref_logprobs)[:upto], list(logprobs)[:upto]
        out["logprobs_bitwise_equal"] = fd is None and bitwise_equal(ref_logprobs, logprobs)
        out["first_logprob_mismatch"] = first_bitwise_mismatch(a, b)
        out["max_abs_logprob_diff"] = max_abs_diff(a, b)
    else:
        out["logprobs_bitwise_equal"] = None
    out["identical"] = bool(out["tokens_equal"]) and out["logprobs_bitwise_equal"] is not False
    return out


def divergence_example(
    decode: Callable[[Sequence[int]], str] | None,
    modal: Seq,
    other: Seq,
    position: int,
    *,
    before: int = 24,
    after: int = 12,
) -> dict[str, Any]:
    """Human-readable context around a divergence (like the blog's "Queens, New York" vs
    "New York City" example)."""

    def show(seq: Seq) -> str:
        if isinstance(seq, str):
            return seq
        return decode(list(seq)) if decode else " ".join(map(str, seq))

    return {
        "position": position,
        "token_number": position + 1,
        "shared_prefix_tail": show(modal[max(0, position - before) : position]),
        "modal_continuation": show(modal[position : position + after]),
        "other_continuation": show(other[position : position + after]),
    }
