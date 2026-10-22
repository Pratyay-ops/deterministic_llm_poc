from detllm.config import JobSpec
from detllm.runner import JobState, select_startable

BUDGET = {"budget_gb": 100.0, "max_parallel": 8}


def job(job_id, mem=10.0, exclusive=False, deps=(), not_before=0.0):
    spec = JobSpec(id=job_id, experiment="fp_nonassoc", mem_gb=mem, exclusive=exclusive, depends_on=list(deps))
    return JobState(spec, not_before=not_before)


def ids(jobs):
    return [j.spec.id for j in jobs]


def test_packs_jobs_into_the_memory_budget_with_backfill():
    pending = [job("a", 40), job("b", 40), job("c", 40), job("d", 10)]
    start, skip = select_startable(pending, [], set(), set(), now=0, **BUDGET)
    assert ids(start) == ["a", "b", "d"] and not skip


def test_respects_max_parallel():
    pending = [job(str(i), 1) for i in range(10)]
    start, _ = select_startable(pending, [], set(), set(), now=0, budget_gb=100, max_parallel=3)
    assert len(start) == 3


def test_running_jobs_count_against_the_budget():
    start, _ = select_startable([job("a", 20), job("b", 5)], [job("r", 90)], set(), set(), now=0, **BUDGET)
    assert ids(start) == ["b"]


def test_oversized_job_only_runs_on_an_idle_gpu():
    start, _ = select_startable([job("big", 150)], [], set(), set(), now=0, **BUDGET)
    assert ids(start) == ["big"]
    start, _ = select_startable([job("big", 150)], [job("small", 5)], set(), set(), now=0, **BUDGET)
    assert start == []


def test_exclusive_jobs_wait_for_shared_work_and_an_idle_gpu():
    perf, shared = job("perf", 20, exclusive=True), job("a", 10)
    start, _ = select_startable([perf, shared], [], set(), set(), now=0, **BUDGET)
    assert ids(start) == ["a"]
    start, _ = select_startable([perf], [job("a", 10)], set(), set(), now=0, **BUDGET)
    assert start == []
    start, _ = select_startable([perf], [], {"a"}, set(), now=0, **BUDGET)
    assert ids(start) == ["perf"]


def test_nothing_starts_next_to_an_exclusive_job():
    start, _ = select_startable([job("a", 1)], [job("perf", 20, exclusive=True)], set(), set(), now=0, **BUDGET)
    assert start == []


def test_dependencies_wait_or_skip():
    b, c = job("b", deps=["a"]), job("c", deps=["x"])
    start, skip = select_startable([b, c], [], set(), {"x"}, now=0, **BUDGET)
    assert start == [] and ids(skip) == ["c"]
    start, _ = select_startable([b], [], {"a"}, set(), now=0, **BUDGET)
    assert ids(start) == ["b"]


def test_retry_backoff():
    start, _ = select_startable([job("a", not_before=100.0)], [], set(), set(), now=50.0, **BUDGET)
    assert start == []
    start, _ = select_startable([job("a", not_before=100.0)], [], set(), set(), now=150.0, **BUDGET)
    assert ids(start) == ["a"]
