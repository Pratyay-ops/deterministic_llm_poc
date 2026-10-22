"""Memory-aware parallel scheduler for one GPU with unified memory (GB10 / DGX Spark).

Running independent experiments side by side does not change their numerics: every process gets
its own kernels and its own batches, and kernel selection depends on shapes, not on GPU load. It
does change *timings*, so throughput jobs are marked ``exclusive`` and run alone at the end.

Each job runs as ``python -m detllm job`` in its own session (process group), so a timeout or
Ctrl+C can take down the job together with any vLLM server / engine-core children it spawned.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from detllm.config import JobSpec, SuiteSpec
from detllm.experiments.registry import get_experiment
from detllm.jobs import EXIT_RETRY_LATER
from detllm.modes import apply_env, pre_init_env
from detllm.utils import (
    REPO_ROOT,
    human_duration,
    meminfo_gb,
    now_iso,
    read_json_or_none,
    timestamp_slug,
    write_json,
)

log = logging.getLogger("detllm.runner")

RETRY_DELAY_S = 45.0
KILL_GRACE_S = 30.0
FINAL_STATES = {"ok", "failed", "timeout", "skipped", "cancelled"}


@dataclass
class JobState:
    spec: JobSpec
    status: str = "pending"  # pending | running | ok | failed | timeout | skipped | cancelled
    attempts: int = 0
    not_before: float = 0.0
    started: float | None = None
    finished: float | None = None
    message: str = ""
    resumed: bool = False
    proc: subprocess.Popen | None = None
    log_fh: IO[str] | None = None
    kill_reason: str | None = None
    term_sent: float | None = None

    def duration(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or time.time()) - self.started

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "experiment": self.spec.experiment,
            "model": self.spec.model,
            "mode": self.spec.mode,
            "status": self.status,
            "attempts": self.attempts,
            "mem_gb": self.spec.mem_gb,
            "exclusive": self.spec.exclusive,
            "duration_s": round(self.duration() or 0.0, 1),
            "resumed": self.resumed,
            "message": self.message[:500],
        }


def select_startable(
    pending: list[JobState],
    running: list[JobState],
    done_ok: set[str],
    done_failed: set[str],
    *,
    budget_gb: float,
    max_parallel: int,
    now: float,
) -> tuple[list[JobState], list[JobState]]:
    """Decide which pending jobs start now. Pure function; returns (to_start, to_skip).

    * A job whose dependency failed is skipped; one whose dependency is unfinished waits.
    * Jobs in retry back-off wait until ``not_before``.
    * Non-exclusive jobs are packed into the memory budget in suite order; a later, smaller job
      may backfill around one that does not fit yet. A job bigger than the whole budget only
      starts on an idle GPU.
    * Exclusive jobs start only on an idle GPU when no non-exclusive job is ready.
    """
    to_start: list[JobState] = []
    to_skip: list[JobState] = []
    if any(j.spec.exclusive for j in running):
        return to_start, to_skip
    ready_shared: list[JobState] = []
    ready_exclusive: list[JobState] = []
    for job in pending:
        deps = job.spec.depends_on
        if any(d in done_failed for d in deps):
            to_skip.append(job)
            continue
        if not all(d in done_ok for d in deps) or job.not_before > now:
            continue
        (ready_exclusive if job.spec.exclusive else ready_shared).append(job)

    used = sum(j.spec.mem_gb for j in running)
    slots = max_parallel - len(running)
    for job in ready_shared:
        if slots <= 0:
            break
        idle = not running and not to_start
        if used + job.spec.mem_gb <= budget_gb or idle:
            to_start.append(job)
            used += job.spec.mem_gb
            slots -= 1
    if not running and not to_start and not ready_shared and ready_exclusive:
        to_start.append(ready_exclusive[0])
    return to_start, to_skip


class SuiteRunner:
    def __init__(
        self,
        suite: SuiteSpec,
        run_dir: Path,
        *,
        resume: bool = True,
        only: list[str] | None = None,
        exclude: list[str] | None = None,
        max_parallel: int | None = None,
        budget_gb: float | None = None,
        poll_s: float = 2.0,
    ):
        self.suite = suite
        self.run_dir = run_dir
        self.max_parallel = max_parallel or suite.max_parallel
        self.budget_gb = budget_gb or suite.memory_budget_gb
        self.poll_s = poll_s
        self._stop_requested = 0
        selected = [j for j in suite.jobs if _selected(j.id, only, exclude)]
        if not selected:
            raise ValueError("no jobs selected (check --only / --exclude patterns)")
        self.jobs = [JobState(spec) for spec in selected]
        self._prepare(resume)

    # ------------------------------------------------------------------ setup

    def _prepare(self, resume: bool) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        selected_ids = {j.spec.id for j in self.jobs}
        for job in self.jobs:
            result = read_json_or_none(self.run_dir / job.spec.id / "result.json")
            if resume and result and result.get("status") == "ok":
                job.status, job.resumed = "ok", True
                job.message = "completed in a previous run"
            # Dependencies outside the selection: honour a finished result, otherwise drop them.
            kept = []
            for dep in job.spec.depends_on:
                if dep in selected_ids:
                    kept.append(dep)
                    continue
                dep_result = read_json_or_none(self.run_dir / dep / "result.json")
                if dep_result and dep_result.get("status") == "ok":
                    continue
                log.warning("%s: dependency %s is not part of this run - ignoring it", job.spec.id, dep)
            job.spec.depends_on = kept
        write_json(
            self.run_dir / "suite.json",
            {
                **self.suite.to_dict(),
                "run_dir": str(self.run_dir),
                "started": now_iso(),
                "max_parallel": self.max_parallel,
                "memory_budget_gb": self.budget_gb,
            },
        )

    # ------------------------------------------------------------------ main loop

    def run(self) -> int:
        old_int = signal.signal(signal.SIGINT, self._on_signal)
        old_term = signal.signal(signal.SIGTERM, self._on_signal)
        try:
            self._loop()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            self._write_status()
        self._print_summary()
        return 0 if all(j.status == "ok" for j in self.jobs) else 1

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self._stop_requested += 1
        if self._stop_requested == 1:
            log.warning("stop requested (signal %s): terminating running jobs ...", signum)
        else:
            log.warning("second stop request: killing running jobs")

    def _loop(self) -> None:
        resumed = sum(1 for j in self.jobs if j.resumed)
        log.info(
            "suite %s: %d jobs (%d already done), budget %.0f GB, max %d parallel, run dir %s",
            self.suite.name,
            len(self.jobs),
            resumed,
            self.budget_gb,
            self.max_parallel,
            self.run_dir,
        )
        while True:
            now = time.time()
            changed = self._poll_running(now)
            running = [j for j in self.jobs if j.status == "running"]
            pending = [j for j in self.jobs if j.status == "pending"]

            if self._stop_requested:
                for job in running:
                    self._terminate(job, "cancelled", force=self._stop_requested > 1)
                for job in pending:
                    job.status, job.message = "cancelled", "suite stopped"
                if not running:
                    break
                time.sleep(self.poll_s)
                continue

            if not pending and not running:
                break

            done_ok = {j.spec.id for j in self.jobs if j.status == "ok"}
            done_failed = {j.spec.id for j in self.jobs if j.status in FINAL_STATES - {"ok"}}
            to_start, to_skip = select_startable(
                pending,
                running,
                done_ok,
                done_failed,
                budget_gb=self.budget_gb,
                max_parallel=self.max_parallel,
                now=now,
            )
            for job in to_skip:
                job.status, job.message = "skipped", "a dependency did not succeed"
                log.warning("SKIP  %s (dependency failed)", job.spec.id)
                changed = True
            for job in to_start:
                self._launch(job)
                changed = True

            if not running and not to_start and pending and not to_skip:
                waiting = [j for j in pending if j.not_before > now]
                if not waiting:
                    for job in pending:  # blocked forever (should not happen after validation)
                        job.status, job.message = "skipped", "unsatisfiable dependencies"
                    changed = True
            if changed:
                self._write_status()
            time.sleep(self.poll_s)

    # ------------------------------------------------------------------ process handling

    def _launch(self, job: JobState) -> None:
        spec = job.spec
        out_dir = self.run_dir / spec.id
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = write_json(out_dir / "job.json", spec.to_dict())
        info = get_experiment(spec.experiment)

        env = dict(os.environ)
        apply_env(pre_init_env(info.family, spec.mode), env)
        cache_root = Path(os.environ.get("DETLLM_CACHE_DIR", Path.home() / ".cache" / "detllm"))
        env.update(
            {
                "DETLLM_RUN_DIR": str(self.run_dir),
                "DETLLM_STARTUP_LOCK": str(self.run_dir / ".startup.lock"),
                "PYTHONUNBUFFERED": "1",
                # Per-job compile caches: no races between concurrent engines, reuse on re-runs.
                "VLLM_CACHE_ROOT": str(cache_root / "vllm" / spec.id),
                "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor" / spec.id),
            }
        )
        src = str(REPO_ROOT / "src")
        if src not in env.get("PYTHONPATH", "").split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
        env.update({k: str(v) for k, v in spec.env.items()})

        job.attempts += 1
        job.log_fh = open(out_dir / "job.log", "a", encoding="utf-8")
        job.log_fh.write(f"\n===== attempt {job.attempts} at {now_iso()} =====\n")
        job.log_fh.flush()
        cmd = [sys.executable, "-m", "detllm", "job", "--spec", str(spec_path), "--out", str(out_dir)]
        job.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=job.log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job.status, job.started, job.finished = "running", time.time(), None
        job.kill_reason, job.term_sent = None, None
        running = [j for j in self.jobs if j.status == "running"]
        used = sum(j.spec.mem_gb for j in running)
        avail = meminfo_gb().get("MemAvailable")
        log.info(
            "START %-44s mem %5.1f GB | running %d/%d, planned %.0f/%.0f GB%s%s",
            spec.id,
            spec.mem_gb,
            len(running),
            self.max_parallel,
            used,
            self.budget_gb,
            f", MemAvailable {avail:.0f} GB" if avail else "",
            " [exclusive]" if spec.exclusive else "",
        )

    def _poll_running(self, now: float) -> bool:
        changed = False
        for job in self.jobs:
            if job.status != "running" or job.proc is None:
                continue
            rc = job.proc.poll()
            if rc is None:
                if job.kill_reason is None and job.duration() and job.duration() > job.spec.timeout_min * 60:
                    log.error("TIMEOUT %s after %s", job.spec.id, human_duration(job.duration()))
                    self._terminate(job, "timeout")
                elif job.term_sent and now - job.term_sent > KILL_GRACE_S:
                    self._signal_group(job, signal.SIGKILL)
                continue
            changed = True
            self._finish(job, rc)
        return changed

    def _signal_group(self, job: JobState, sig: int) -> None:
        if job.proc is None:
            return
        try:
            os.killpg(job.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _terminate(self, job: JobState, reason: str, force: bool = False) -> None:
        if job.kill_reason is None:
            job.kill_reason = reason
            job.term_sent = time.time()
            self._signal_group(job, signal.SIGTERM)
        if force:
            self._signal_group(job, signal.SIGKILL)

    def _finish(self, job: JobState, rc: int) -> None:
        job.finished = time.time()
        if job.log_fh is not None:
            job.log_fh.close()
            job.log_fh = None
        result = read_json_or_none(self.run_dir / job.spec.id / "result.json") or {}
        status = result.get("status")
        if job.kill_reason:
            job.status, job.message = job.kill_reason, f"terminated ({job.kill_reason})"
        elif rc == 0 and status == "ok":
            job.status, job.message = "ok", ""
        elif rc == EXIT_RETRY_LATER and job.attempts <= job.spec.max_retries:
            job.status = "pending"
            job.not_before = time.time() + RETRY_DELAY_S * job.attempts
            job.message = f"retry {job.attempts}/{job.spec.max_retries}: {result.get('error', '')}"
            log.warning("RETRY %s in %.0fs: %s", job.spec.id, RETRY_DELAY_S * job.attempts, result.get("error", ""))
            return
        else:
            job.status = "failed"
            job.message = result.get("error") or f"exit code {rc}"
        level = logging.INFO if job.status == "ok" else logging.ERROR
        log.log(
            level,
            "%-5s %-44s %s%s",
            "DONE" if job.status == "ok" else job.status.upper(),
            job.spec.id,
            human_duration(job.duration()),
            "" if job.status == "ok" else f" - {job.message[:200]} (see {self.run_dir / job.spec.id / 'job.log'})",
        )

    # ------------------------------------------------------------------ reporting

    def _write_status(self) -> None:
        write_json(
            self.run_dir / "status.json",
            {"suite": self.suite.name, "updated": now_iso(), "jobs": [j.to_dict() for j in self.jobs]},
        )

    def _print_summary(self) -> None:
        width = max(len(j.spec.id) for j in self.jobs)
        lines = ["", f"Suite {self.suite.name} - {self.run_dir}", "-" * (width + 36)]
        for job in self.jobs:
            note = " (resumed)" if job.resumed else ""
            lines.append(f"{job.spec.id:<{width}}  {job.status:<9} {human_duration(job.duration()):>9}{note}")
        counts: dict[str, int] = {}
        for job in self.jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
        lines.append("-" * (width + 36))
        lines.append(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
        print("\n".join(lines), flush=True)


def _selected(job_id: str, only: list[str] | None, exclude: list[str] | None) -> bool:
    if only and not any(fnmatch.fnmatch(job_id, pat) for pat in only):
        return False
    return not (exclude and any(fnmatch.fnmatch(job_id, pat) for pat in exclude))


def default_run_dir(results_root: Path, suite_name: str) -> Path:
    return results_root / f"{suite_name}-{timestamp_slug()}"


def latest_run_dir(results_root: Path, suite_name: str) -> Path | None:
    """Most recent run directory of this suite that still has unfinished jobs."""
    candidates = sorted(results_root.glob(f"{suite_name}-*"), reverse=True)
    for cand in candidates:
        status = read_json_or_none(cand / "status.json")
        if status is None:
            continue
        if any(j.get("status") != "ok" for j in status.get("jobs", [])):
            return cand
        return None  # the latest run finished cleanly -> start a new one
    return None
