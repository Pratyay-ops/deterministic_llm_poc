import pytest

from detllm.config import ConfigError, expand_jobs, list_suites, load_models, load_suite

GB10_USABLE_GB = 122


def test_models_load():
    models = load_models()
    m = models["qwen3-1.7b"]
    assert m.hf_id == "Qwen/Qwen3-1.7B"
    assert m.chat_template_kwargs == {"enable_thinking": False}
    assert m.vllm_mem_gb() == pytest.approx(m.weights_gb + m.kv_cache_gb + m.overhead_gb, abs=0.11)
    assert m.kv_tokens() and m.kv_tokens() > 50_000


@pytest.mark.parametrize("path", list_suites(), ids=lambda p: p.stem)
def test_shipped_suites_are_valid(path):
    suite = load_suite(str(path))
    assert suite.jobs
    ids = [j.id for j in suite.jobs]
    assert len(ids) == len(set(ids))
    for job in suite.jobs:
        assert 0 < job.mem_gb <= GB10_USABLE_GB, job.id
        if not job.exclusive:
            assert job.mem_gb <= suite.memory_budget_gb, job.id


def test_matrix_expansion_ids_and_params():
    models = load_models()
    jobs = expand_jobs(
        [
            {
                "experiment": "vllm_completions",
                "id": "c-{model}-{mode}",
                "matrix": {"model": ["qwen3-1.7b"], "mode": ["baseline", "batch_invariant"]},
                "params": {"n": 10},
            }
        ],
        defaults={"seed": 3},
        models=models,
    )
    assert [j.id for j in jobs] == ["c-qwen3-1.7b-baseline", "c-qwen3-1.7b-batch_invariant"]
    assert jobs[0].params["n"] == 10 and jobs[0].params["seed"] == 3
    assert jobs[0].params["kv_cache_gb"] == models["qwen3-1.7b"].kv_cache_gb
    assert jobs[0].mem_gb == models["qwen3-1.7b"].vllm_mem_gb()


def test_kv_override_changes_planned_memory():
    models = load_models()
    (job,) = expand_jobs(
        [{"experiment": "vllm_completions", "model": "qwen3-1.7b", "params": {"kv_cache_gb": 20}}],
        defaults={},
        models=models,
    )
    assert job.id == "vllm_completions-qwen3-1.7b-baseline"
    assert job.mem_gb == models["qwen3-1.7b"].vllm_mem_gb(20)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"experiment": "nope"}, "unknown experiment"),
        ({"experiment": "vllm_completions"}, "needs a model"),
        ({"experiment": "vllm_completions", "model": "gpt-9"}, "unknown model"),
        ({"experiment": "op_invariance", "mode": "batch_invariant"}, "not valid"),
    ],
)
def test_config_errors(entry, match):
    with pytest.raises(ConfigError, match=match):
        expand_jobs([entry], defaults={}, models=load_models())


def test_duplicate_ids_rejected():
    with pytest.raises(ConfigError, match="duplicate"):
        expand_jobs(
            [{"experiment": "fp_nonassoc", "id": "x"}, {"experiment": "fp_nonassoc", "id": "x"}],
            defaults={},
            models=load_models(),
        )


def test_unknown_dependency_rejected():
    with pytest.raises(ConfigError, match="depends on unknown job"):
        expand_jobs([{"experiment": "fp_nonassoc", "depends_on": ["ghost"]}], defaults={}, models=load_models())
