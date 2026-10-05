from detllm.analysis.divergence import (
    bitwise_equal,
    compare_to_reference,
    divergence_example,
    first_divergence,
    summarize_sequences,
)


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence("abc", "abd") == 2


def test_bitwise_equal_is_strict():
    assert bitwise_equal([0.0, -1.5], [0.0, -1.5])
    assert not bitwise_equal([0.0], [-0.0])  # == says equal, the bits do not
    nan = float("nan")
    assert bitwise_equal([nan], [nan])
    assert not bitwise_equal([1.0], [1.0 + 2**-52])
    assert not bitwise_equal([1.0], [1.0, 2.0])


def test_summary_counts_and_earliest_divergence():
    a, b, c = [1, 2, 3, 4, 5], [1, 2, 3, 9, 9], [1, 7, 7, 7, 7]
    s = summarize_sequences([a, a, a, b, c, None])
    assert (s["n"], s["n_valid"], s["n_failed"]) == (6, 5, 1)
    assert s["n_unique"] == 3 and s["modal_count"] == 3
    assert s["counts"] == [3, 1, 1]
    assert s["divergence_positions"] == [1, 3]
    assert s["earliest_divergence"] == 1
    assert s["unique"][0]["first_divergence_vs_modal"] is None
    assert s["modal_index"] == 0


def test_summary_logprob_variants_detect_hidden_nondeterminism():
    seqs = [[1, 2]] * 3
    lps = [[-0.5, -1.0], [-0.5, -1.0], [-0.5, -1.0000001]]
    s = summarize_sequences(seqs, logprobs=lps)
    assert s["n_unique"] == 1  # same tokens ...
    assert s["modal_logprob_variants"] == 2  # ... different numerics


def test_summary_all_identical():
    s = summarize_sequences([[5, 6, 7]] * 10, logprobs=[[-1.0, -2.0, -3.0]] * 10)
    assert s["n_unique"] == 1
    assert s["earliest_divergence"] is None
    assert s["modal_logprob_variants"] == 1


def test_compare_to_reference():
    ref_t, ref_l = [1, 2, 3], [-1.0, -2.0, -3.0]
    same = compare_to_reference(ref_t, ref_l, [1, 2, 3], [-1.0, -2.0, -3.0])
    assert same["identical"] and same["tokens_equal"] and same["logprobs_bitwise_equal"]

    lp_only = compare_to_reference(ref_t, ref_l, [1, 2, 3], [-1.0, -2.0000001, -3.0])
    assert lp_only["tokens_equal"] and not lp_only["logprobs_bitwise_equal"] and not lp_only["identical"]
    assert lp_only["first_logprob_mismatch"] == 1

    diverged = compare_to_reference(ref_t, ref_l, [1, 5, 6], [-1.0, -2.5, -3.0])
    assert not diverged["tokens_equal"] and diverged["first_token_divergence"] == 1
    assert diverged["first_logprob_mismatch"] is None  # logprobs only compared where tokens agree


def test_divergence_example_on_text():
    ex = divergence_example(None, "born in Queens, New York", "born in New York City", 8, before=8, after=6)
    assert ex["token_number"] == 9
    assert ex["shared_prefix_tail"] == "born in "
    assert ex["modal_continuation"] == "Queens"
    assert ex["other_continuation"] == "New Yo"
