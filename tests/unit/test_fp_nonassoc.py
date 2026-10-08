import sys

import pytest

from detllm.experiments.fp_nonassoc import BLOG_UNIQUE_SUMS, naive_sum, unique_sums


def test_associativity_examples():
    assert (0.1 + 1e20) - 1e20 == 0.0
    assert 0.1 + (1e20 - 1e20) == 0.1


def test_naive_sum_is_left_to_right():
    assert naive_sum([1e20, 0.1, -1e20]) == 0.0
    assert naive_sum([1e20, -1e20, 0.1]) == 0.1


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the blog's count comes from Python >= 3.12 sum()")
def test_blog_snippet_gives_102_unique_sums():
    naive, builtin = unique_sums()
    assert len(builtin) == BLOG_UNIQUE_SUMS
    assert len(naive) == 258  # plain left-to-right accumulation of the same orders
