"""Model presets and suite definitions (YAML) -> typed job specs. Pure Python.

Suite YAML::

    name: parallel
    memory_budget_gb: 100        # share of GB10's ~122 GB unified memory the scheduler may plan
    max_parallel: 8
    defaults: {seed: 0}          # merged into every job's params
    jobs:
      - experiment: vllm_completions
        id: completions-{model}-{mode}     # optional template ({experiment}, {model}, {mode}, matrix keys)
        matrix: {model: [qwen3-1.7b], mode: [baseline, batch_invariant]}
        params: {n: 1000, max_tokens: 1000}
        mem_gb: 20                         # optional; defaults come from the model preset
        exclusive: false                   # true = run alone (performance measurements)
        depends_on: []
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from detllm.experiments.registry import get_experiment
from detllm.utils import deep_merge, load_yaml, resolve_path

DEFAULT_MODELS_FILE = "configs/models.yaml"
SUITES_DIR = "configs/suites"

# Memory planning defaults (GB) for jobs without a model.
_FAMILY_DEFAULT_MEM = {"cpu": 0.5, "ops": 6.0}


class ConfigError(ValueError):
    pass


@dataclass
class ModelPreset:
    key: str
    hf_id: str
    dtype: str = "bfloat16"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    max_model_len: int = 4096
    weights_gb: float = 0.0
    kv_kib_per_token: float = 0.0
    kv_cache_gb: float = 8.0  # default KV-cache allocation for vLLM jobs
    overhead_gb: float = 5.0  # activations, CUDA graphs, CUDA context, sampler, API server
    hf_mem_gb: float = 12.0
    notes: str = ""

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> ModelPreset:
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"model {key!r}: unknown fields {sorted(unknown)}")
        if "hf_id" not in data:
            raise ConfigError(f"model {key!r}: missing hf_id")
        return cls(key=key, **{k: v for k, v in data.items() if k != "key"})

    def vllm_mem_gb(self, kv_cache_gb: float | None = None) -> float:
        kv = self.kv_cache_gb if kv_cache_gb is None else kv_cache_gb
        return round(self.weights_gb + kv + self.overhead_gb, 1)

    def kv_tokens(self, kv_cache_gb: float | None = None) -> int | None:
        """How many tokens fit in the KV cache (for log messages / sanity checks)."""
        if not self.kv_kib_per_token:
            return None
        kv = self.kv_cache_gb if kv_cache_gb is None else kv_cache_gb
        return int(kv * 1e9 / (self.kv_kib_per_token * 1024))


@dataclass
class JobSpec:
    id: str
    experiment: str
    model: str | None = None
    mode: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    mem_gb: float = 4.0
    exclusive: bool = False
    depends_on: list[str] = field(default_factory=list)
    timeout_min: float = 240.0
    max_retries: int = 3
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SuiteSpec:
    name: str
    description: str = ""
    memory_budget_gb: float = 100.0
    max_parallel: int = 6
    jobs: list[JobSpec] = field(default_factory=list)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- loading


def load_models(path: str | None = None) -> dict[str, ModelPreset]:
    data = load_yaml(path or DEFAULT_MODELS_FILE) or {}
    models = data.get("models", data)
    return {key: ModelPreset.from_dict(key, dict(value or {})) for key, value in models.items()}


def list_suites() -> list[Path]:
    return sorted(resolve_path(SUITES_DIR).glob("*.yaml"))


def resolve_suite_path(name_or_path: str) -> Path:
    p = resolve_path(name_or_path)
    if p.exists():
        return p
    candidate = resolve_path(SUITES_DIR) / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise ConfigError(f"suite not found: {name_or_path} (looked in {SUITES_DIR}/)")


def load_suite(name_or_path: str, models: dict[str, ModelPreset] | None = None) -> SuiteSpec:
    path = resolve_suite_path(name_or_path)
    data = load_yaml(path) or {}
    models = models if models is not None else load_models(data.get("models_file"))
    jobs = expand_jobs(data.get("jobs") or [], defaults=data.get("defaults") or {}, models=models)
    return SuiteSpec(
        name=str(data.get("name") or path.stem),
        description=str(data.get("description") or "").strip(),
        memory_budget_gb=float(data.get("memory_budget_gb", 100.0)),
        max_parallel=int(data.get("max_parallel", 6)),
        jobs=jobs,
        source=str(path),
    )


_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: Any) -> str:
    return _SLUG.sub("-", str(value)).strip("-")


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _format(template: str, fields: dict[str, str], where: str) -> str:
    try:
        return template.format(**fields)
    except (KeyError, IndexError) as exc:
        raise ConfigError(f"{where}: bad placeholder in {template!r}: {exc}") from None


def default_mem_gb(family: str, preset: ModelPreset | None, params: dict[str, Any]) -> float:
    if family == "vllm" and preset is not None:
        return preset.vllm_mem_gb(params.get("kv_cache_gb"))
    if family == "hf" and preset is not None:
        return float(preset.hf_mem_gb)
    return _FAMILY_DEFAULT_MEM.get(family, 4.0)


def expand_jobs(
    entries: list[dict[str, Any]],
    *,
    defaults: dict[str, Any],
    models: dict[str, ModelPreset],
) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for index, raw in enumerate(entries):
        where = f"jobs[{index}]"
        if not isinstance(raw, dict) or "experiment" not in raw:
            raise ConfigError(f"{where}: each job needs an 'experiment'")
        entry = dict(raw)
        try:
            info = get_experiment(entry["experiment"])
        except KeyError as exc:
            raise ConfigError(f"{where}: {exc}") from None
        matrix = dict(entry.get("matrix") or {})
        keys = list(matrix)
        combos = list(itertools.product(*(_as_list(matrix[k]) for k in keys))) if keys else [()]
        for combo in combos:
            values = dict(zip(keys, combo))
            model = values.pop("model", entry.get("model"))
            mode = values.pop("mode", entry.get("mode")) or info.modes[0]
            if info.needs_model and not model:
                raise ConfigError(f"{where}: experiment {info.name!r} needs a model")
            if model is not None and model not in models:
                raise ConfigError(f"{where}: unknown model {model!r} (see configs/models.yaml)")
            if mode not in info.modes:
                raise ConfigError(f"{where}: mode {mode!r} not valid for {info.name} {info.modes}")
            params = deep_merge(deep_merge(defaults, entry.get("params") or {}), values)
            preset = models.get(model) if model else None
            if info.family == "vllm" and preset is not None:
                params.setdefault("kv_cache_gb", preset.kv_cache_gb)
            fields = {"experiment": info.name, "model": model or "", "mode": mode or ""}
            fields.update({k: _slug(v) for k, v in values.items()})
            if entry.get("id"):
                job_id = _slug(_format(str(entry["id"]), fields, where))
            else:
                parts = [info.name, model, mode, *(str(v) for v in values.values())]
                job_id = _slug("-".join(p for p in parts if p))
            mem = entry.get("mem_gb")
            jobs.append(
                JobSpec(
                    id=job_id,
                    experiment=info.name,
                    model=model,
                    mode=mode,
                    params=params,
                    mem_gb=float(mem) if mem is not None else default_mem_gb(info.family, preset, params),
                    exclusive=bool(entry.get("exclusive", False)),
                    depends_on=[_format(str(d), fields, where) for d in entry.get("depends_on") or []],
                    timeout_min=float(entry.get("timeout_min", 240.0)),
                    max_retries=int(entry.get("max_retries", 3)),
                    env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                )
            )
    seen: set[str] = set()
    for job in jobs:
        if job.id in seen:
            raise ConfigError(f"duplicate job id {job.id!r} - give the entries distinct 'id' templates")
        seen.add(job.id)
    for job in jobs:
        for dep in job.depends_on:
            if dep not in seen:
                raise ConfigError(f"job {job.id!r} depends on unknown job {dep!r}")
    return jobs
