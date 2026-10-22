"""Small dependency-free helpers shared by the scheduler, the job processes and the report.

Nothing here may import torch / vLLM / transformers.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Paths in configs are relative to the repository root unless absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# --------------------------------------------------------------------------- JSON / YAML I/O


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "tolist"):  # numpy / torch
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    return repr(obj)


def write_json(path: str | os.PathLike[str], obj: Any, *, indent: int | None = 2) -> Path:
    """Atomic JSON write (tmp file + rename) so a killed job never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_json_or_none(path: str | os.PathLike[str]) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


class JsonlWriter:
    """Append-only JSON-lines writer (one record per line)."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: Any) -> None:
        self._fh.write(json.dumps(record, default=_json_default) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_yaml(path: str | os.PathLike[str]) -> Any:
    with open(resolve_path(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; values from ``override`` win. Inputs are not modified."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_key_values(items: list[str] | None) -> dict[str, Any]:
    """``["n=100", "batch_sizes=[1,2,4]"]`` -> ``{"n": 100, "batch_sizes": [1, 2, 4]}`` (YAML-typed)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw) if raw.strip() else ""
    return out


def stable_hash(obj: Any, length: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


# --------------------------------------------------------------------------- locking


@contextlib.contextmanager
def file_lock(path: str | os.PathLike[str], *, poll_s: float = 1.0, log: logging.Logger | None = None) -> Iterator[None]:
    """Cross-process exclusive lock (flock). Released automatically if the holder dies."""
    import fcntl  # POSIX only - the harness runs on Linux

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        t0 = time.monotonic()
        announced = False
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if log is not None and not announced:
                    log.info("waiting for lock %s", path)
                    announced = True
                time.sleep(poll_s)
        if log is not None and announced:
            log.info("acquired lock %s after %.0fs", path, time.monotonic() - t0)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# --------------------------------------------------------------------------- GB10 unified memory


def hf_model_cache_dir(hf_id: str) -> Path:
    """Directory of a model inside the Hugging Face hub cache."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    return hub / ("models--" + hf_id.replace("/", "--"))


def evict_file_cache(root: str | os.PathLike[str]) -> int:
    """Drop the page cache of every regular file under ``root`` (``posix_fadvise(DONTNEED)``).

    On GB10 / DGX Spark the GPU and the OS page cache share the same 128 GB of LPDDR5x, and
    cudaMemGetInfo does not count reclaimable cache as free (NVIDIA DGX Spark known issue). Once
    a model is loaded into the engine the cached safetensors pages are dead weight that makes the
    *next* vLLM engine think there is not enough memory. Unlike ``drop_caches`` this needs no root
    and only touches the files we name. Returns the number of bytes advised.
    """
    if not hasattr(os, "posix_fadvise"):
        return 0
    total = 0
    for p in Path(root).rglob("*"):
        try:
            if p.is_symlink() or not p.is_file():
                continue
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                total += p.stat().st_size
            finally:
                os.close(fd)
        except OSError:
            continue
    return total


def meminfo_gb() -> dict[str, float]:
    """Parse /proc/meminfo (values in GB). Empty dict if unavailable."""
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    out[key] = int(parts[0]) * 1024 / 1e9
    except OSError:
        pass
    return out
