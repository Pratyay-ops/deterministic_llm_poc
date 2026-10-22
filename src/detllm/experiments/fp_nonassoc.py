"""Blog section "The original sin: floating-point non-associativity" - the blog's demos (CPU).

Note on Python versions: the blog's snippet uses the built-in ``sum()``, whose algorithm changed twice:
up to 3.11 it is a plain left-to-right accumulation (258 unique results here); Python 3.12 and 3.13
compensate float additions (Neumaier) but add the blog's *int* ``1`` and ``-1`` without compensation,
and exactly that mix gives the blog's 102 (with ``1.0`` instead of ``1`` it gives 1); Python 3.14 also
compensates the ints (9 unique results). The containers ship Python 3.12, so the blog's snippet is
reproduced as written; the plain left-to-right loop is reported next to it - itself a nice illustration
that changing the reduction algorithm changes the answer.
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
    ctx.log.info("built-in sum(): %d unique results (blog: %d); left-to-right loop: %d", len(builtin), BLOG_UNIQUE_SUMS, len(naive))
    return {
        "associativity": associativity,
        "n_unique_sums": len(builtin),
        "blog_n_unique_sums": BLOG_UNIQUE_SUMS,
        "matches_blog": len(builtin) == BLOG_UNIQUE_SUMS,
        "min_sum": builtin[0],
        "max_sum": builtin[-1],
        "python": sys.version.split()[0],
        "n_unique_left_to_right": len(naive),
        "builtin_sum_note": (
            "plain left-to-right summation"
            if sys.version_info < (3, 12)
            else "compensated for floats, the int 1 added uncompensated (the blog's case)"
            if sys.version_info < (3, 14)
            else "compensated for floats and ints"
        ),
    }
