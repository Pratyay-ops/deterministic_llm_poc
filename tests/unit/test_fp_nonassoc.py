from detllm.experiments.fp_nonassoc import BLOG_UNIQUE_SUMS, naive_sum, unique_sums


def test_associativity_examples():
    assert (0.1 + 1e20) - 1e20 == 0.0
    assert 0.1 + (1e20 - 1e20) == 0.1


def test_naive_sum_is_left_to_right():
    assert naive_sum([1e20, 0.1, -1e20]) == 0.0
    assert naive_sum([1e20, -1e20, 0.1]) == 0.1


def test_blog_reports_102_unique_left_to_right_sums():
    naive, builtin = unique_sums()
    assert len(naive) == BLOG_UNIQUE_SUMS
    assert builtin  # Python >= 3.12 sum() is compensated; its count is reported, not asserted
