"""Blog section "The original sin: floating-point non-associativity" - the blog's demos (CPU).

Note on Python versions: the blog's snippet uses the built-in ``sum()``. Up to Python 3.11 that
was a plain left-to-right double accumulation; since Python 3.12 ``sum()`` of floats uses
compensated (Neumaier) summation, which hides most of the effect. The containers ship Python 3.12,
so the blog's number is reproduced with an explicit left-to-right loop (what ``sum()`` used to do),
and the 3.12 built-in result is reported next to it - itself a nice illustration that changing the
reduction algorithm changes the answer.
"""

from __future__ import annotations

import random
import sys
from typing import Any

BLOG_UNIQUE_SUMS = 102


def naive_sum(values: list[float]) -> float:
    total = 0.0
    for v in values:
        total += v
    return total


def unique_sums(seed: int = 42, iterations: int = 10000) -> tuple[list[float], list[float]]:
    """The blog's snippet: shuffle 8 values and sum them 10,000 times.

    Returns (unique left-to-right sums, unique built-in ``sum()`` results), both sorted.
    """
    vals = [1e-10, 1e-5, 1e-2, 1]
    vals = vals + [-v for v in vals]
    naive, builtin = [], []
    random.seed(seed)
    for _ in range(iterations):
        random.shuffle(vals)
        naive.append(naive_sum(vals))
        builtin.append(sum(vals))
    return sorted(set(naive)), sorted(set(builtin))


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    associativity = {
        "(0.1 + 1e20) - 1e20": (0.1 + 1e20) - 1e20,
        "0.1 + (1e20 - 1e20)": 0.1 + (1e20 - 1e20),
    }
    naive, builtin = unique_sums(seed=int(ctx.param("seed_sum", 42)))
    ctx.save_json("unique_sums.json", {"left_to_right": naive, "builtin_sum": builtin})
    ctx.log.info("left-to-right: %d unique results (blog: %d); built-in sum(): %d", len(naive), BLOG_UNIQUE_SUMS, len(builtin))
    return {
        "associativity": associativity,
        "n_unique_sums": len(naive),
        "blog_n_unique_sums": BLOG_UNIQUE_SUMS,
        "matches_blog": len(naive) == BLOG_UNIQUE_SUMS,
        "min_sum": naive[0],
        "max_sum": naive[-1],
        "python": sys.version.split()[0],
        "n_unique_builtin_sum": len(builtin),
        "builtin_sum_note": "Python >= 3.12 sum() uses compensated summation" if sys.version_info >= (3, 12) else "plain summation",
    }
