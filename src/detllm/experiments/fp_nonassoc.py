"""Blog section "The original sin: floating-point non-associativity" - verbatim demos (CPU)."""

from __future__ import annotations

import random
from typing import Any

BLOG_UNIQUE_SUMS = 102


def unique_sums(seed: int = 42, iterations: int = 10000) -> list[float]:
    """Exactly the blog's snippet: shuffle 8 values and sum them 10,000 times."""
    vals = [1e-10, 1e-5, 1e-2, 1]
    vals = vals + [-v for v in vals]
    results = []
    random.seed(seed)
    for _ in range(iterations):
        random.shuffle(vals)
        results.append(sum(vals))
    return sorted(set(results))


def run(ctx) -> dict[str, Any]:  # noqa: ANN001
    associativity = {
        "(0.1 + 1e20) - 1e20": (0.1 + 1e20) - 1e20,
        "0.1 + (1e20 - 1e20)": 0.1 + (1e20 - 1e20),
    }
    results = unique_sums(seed=int(ctx.param("seed_sum", 42)))
    ctx.save_json("unique_sums.json", results)
    ctx.log.info("There are %d unique results (blog: %d)", len(results), BLOG_UNIQUE_SUMS)
    return {
        "associativity": associativity,
        "n_unique_sums": len(results),
        "blog_n_unique_sums": BLOG_UNIQUE_SUMS,
        "matches_blog": len(results) == BLOG_UNIQUE_SUMS,
        "min_sum": results[0],
        "max_sum": results[-1],
    }
