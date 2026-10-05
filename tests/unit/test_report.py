from detllm.analysis.report import write_report
from detllm.utils import read_json, write_json


def _job(run, job_id, experiment, model, mode, summary, status="ok"):
    d = run / job_id
    write_json(d / "job.json", {"id": job_id, "experiment": experiment, "model": model, "mode": mode})
    write_json(d / "result.json", {"status": status, "summary": summary, "error": None if status == "ok" else "boom"})


def test_report_renders_all_sections(tmp_path):
    run = tmp_path / "parallel-20261006-000000"
    write_json(run / "suite.json", {"name": "parallel", "started": "2026-10-06T00:00:00"})
    _job(
        run,
        "fp-nonassoc",
        "fp_nonassoc",
        None,
        "default",
        {"associativity": {"(0.1 + 1e20) - 1e20": 0.0}, "n_unique_sums": 102, "blog_n_unique_sums": 102},
    )
    base = {
        "model": "qwen3-1.7b",
        "mode": "baseline",
        "engine": "server",
        "n": 1000,
        "n_valid": 1000,
        "max_tokens": 1000,
        "n_unique": 37,
        "modal_count": 500,
        "earliest_divergence": 211,
        "modal_logprob_variants": 12,
        "engine_info": {"api": "completions"},
        "divergence_example": {
            "token_number": 212,
            "shared_prefix_tail": "born on May 11, 1918, in",
            "modal_continuation": " Queens, New York",
            "other_continuation": " New York City",
        },
    }
    bi = dict(base, mode="batch_invariant", n_unique=1, modal_count=1000, earliest_divergence=None, modal_logprob_variants=1, divergence_example=None)
    _job(run, "completions-qwen3-1.7b-baseline", "vllm_completions", "qwen3-1.7b", "baseline", base)
    _job(run, "completions-qwen3-1.7b-batch_invariant", "vllm_completions", "qwen3-1.7b", "batch_invariant", bi)
    _job(
        run,
        "ops-baseline",
        "op_invariance",
        None,
        "baseline",
        {
            "mode": "baseline",
            "blog_mm_demo": {"max_abs_diff": 1669.25},
            "blog_run_to_run": {"mismatches": 0},
            "results": [
                {
                    "op": "mm",
                    "dtype": "bfloat16",
                    "batch_invariant": False,
                    "first_variant_batch": 64,
                    "max_abs_diff": 0.0625,
                    "run_to_run_identical": True,
                    "error": None,
                }
            ],
            "n_ops": 1,
            "n_invariant": 0,
            "n_errors": 0,
        },
    )
    _job(run, "perf-qwen3-1.7b-baseline", "vllm_perf", "qwen3-1.7b", "baseline", {}, status="error")

    text = write_report(run).read_text(encoding="utf-8")
    assert "**37 unique**" in text and "**1 unique**" in text
    assert "token 212" in text  # 1-based token number, like the blog's "103rd token"
    assert "Queens" in text
    assert "102" in text
    assert "varies" in text
    assert "Jobs that did not finish" in text and "perf-qwen3-1.7b-baseline" in text
    summary = read_json(run / "summary.json")
    assert summary["completions-qwen3-1.7b-batch_invariant"]["summary"]["n_unique"] == 1
