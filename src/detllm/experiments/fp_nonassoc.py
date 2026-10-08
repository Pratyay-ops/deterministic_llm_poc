"""Blog section "The original sin: floating-point non-associativity" - the blog's demos (CPU).

Note on Python versions: the blog's snippet uses the built-in ``sum()``. Since Python 3.12 that is
compensated (Neumaier) summation for floats, but the blog's list holds the *int* ``1`` (and ``-1``),
which CPython adds without compensation - and that exact mix gives the blog's 102 unique results
(with ``1.0`` instead of ``1`` it gives 1). The containers ship Python 3.12, so the blog's snippet is
reproduced as written; a plain left-to-right loop (what ``sum()`` did up to 3.11) is reported next to
it and gives 258 - itself a nice illustration that changing the reduction algorithm changes the answer.
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
            "Python >= 3.12 sum(): compensated for floats, the int 1 is added uncompensated"
            if sys.version_info >= (3, 12)
            else "plain left-to-right summation"
        ),
    }
