"""One job = one experiment in one mode, executed in its own process.

The scheduler launches ``python -m detllm job --spec job.json --out <dir>``. A fresh process per job
is required: vLLM reads VLLM_BATCH_INVARIANT at start-up, cuBLAS reads its workspace variables
when the handle is created, and batch_invariant_ops patches aten ops process-wide.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from detllm.config import JobSpec, ModelPreset, load_models
from detllm.experiments.registry import get_experiment, load_runner
from detllm.modes import apply_env, pre_init_env
from detllm.utils import configure_logging, file_lock, now_iso, read_json, write_json

EXIT_RETRY_LATER = 75  # EX_TEMPFAIL: "not enough free memory right now, re-queue me"


class RetryLater(RuntimeError):
    """Raised when a job cannot start yet (e.g. vLLM's free-memory check failed)."""


@dataclass
class JobContext:
    job: JobSpec
    model: ModelPreset | None
    out_dir: Path
    run_dir: Path
    log: logging.Logger

    def param(self, key: str, default: Any = None) -> Any:
        return self.job.params.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.param("seed", 0))

    def require_model(self) -> ModelPreset:
        if self.model is None:
            raise ValueError(f"experiment {self.job.experiment} needs a model")
        return self.model

    def path(self, name: str) -> Path:
        return self.out_dir / name

    def save_json(self, name: str, obj: Any) -> Path:
        return write_json(self.out_dir / name, obj)

    @contextlib.contextmanager
    def startup_lock(self, enabled: bool = True) -> Iterator[None]:
        """Serialise engine start-up across concurrent jobs (only when it is really needed)."""
        path = os.environ.get("DETLLM_STARTUP_LOCK")
        if not enabled or not path:
            yield
            return
        with file_lock(path, log=self.log):
            yield


def _raise_on_signal(signum: int, _frame: Any) -> None:
    # Turn SIGTERM into an exception so `with` blocks run and child servers are shut down.
    raise KeyboardInterrupt(f"received signal {signum}")


def run_job(spec_path: Path, out_dir: Path) -> int:
    spec = JobSpec.from_dict(read_json(spec_path))
    info = get_experiment(spec.experiment)
    # The scheduler already set these, but `detllm job` can also be invoked by hand.
    apply_env(pre_init_env(info.family, spec.mode))
    apply_env({k: str(v) for k, v in spec.env.items()})

    signal.signal(signal.SIGTERM, _raise_on_signal)
    configure_logging()
    log = logging.getLogger(f"detllm.job.{spec.id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models()
    ctx = JobContext(
        job=spec,
        model=models.get(spec.model) if spec.model else None,
        out_dir=out_dir,
        run_dir=Path(os.environ.get("DETLLM_RUN_DIR", out_dir.parent)),
        log=log,
    )
    result: dict[str, Any] = {
        "job_id": spec.id,
        "experiment": spec.experiment,
        "model": spec.model,
        "hf_id": ctx.model.hf_id if ctx.model else None,
        "mode": spec.mode,
        "started": now_iso(),
    }
    log.info("job %s: experiment=%s model=%s mode=%s", spec.id, spec.experiment, spec.model, spec.mode)
    t0 = time.time()
    code = 1
    try:
        from detllm.env import collect_env

        write_json(out_dir / "env.json", collect_env(gpu=info.family != "cpu"))
        summary = load_runner(spec.experiment)(ctx)
        result.update(status="ok", summary=summary)
        code = 0
    except RetryLater as exc:
        log.warning("job asked to be retried later: %s", exc)
        result.update(status="retry", error=str(exc))
        code = EXIT_RETRY_LATER
    except KeyboardInterrupt as exc:
        result.update(status="cancelled", error=str(exc) or "interrupted")
        code = 130
    except BaseException as exc:  # noqa: BLE001 - record everything, including SystemExit
        log.error("job failed: %s", exc)
        result.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        code = 1
    finally:
        result["duration_s"] = round(time.time() - t0, 2)
        result["finished"] = now_iso()
        write_json(out_dir / "result.json", result)
        log.info("job %s finished: %s in %.0fs", spec.id, result.get("status"), result["duration_s"])
    return code
