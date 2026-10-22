"""Experiment registry. Pure Python - experiment modules are imported lazily by name."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from detllm.modes import HF_MODES, OP_MODES, VLLM_MODES


@dataclass(frozen=True)
class ExperimentInfo:
    name: str
    target: str  # "module:function"
    family: str  # cpu | ops | hf | vllm  (decides env + default memory budget)
    needs_model: bool
    modes: tuple[str, ...]
    description: str
    blog_section: str = ""


EXPERIMENTS: dict[str, ExperimentInfo] = {
    e.name: e
    for e in (
        ExperimentInfo(
            "preflight",
            "detllm.experiments.preflight:run",
            "ops",
            False,
            ("default",),
            "GPU/driver/library sanity checks: Triton on sm_121, shared memory, vLLM features",
        ),
        ExperimentInfo(
            "fp_nonassoc",
            "detllm.experiments.fp_nonassoc:run",
            "cpu",
            False,
            ("default",),
            "Floating-point non-associativity demos (the blog's 102 unique sums)",
            "The original sin: floating-point non-associativity",
        ),
        ExperimentInfo(
            "op_invariance",
            "detllm.experiments.op_invariance:run",
            "ops",
            False,
            OP_MODES,
            "Run-to-run determinism and batch invariance of matmul, RMSNorm, softmax and attention ops",
            "Batch invariance and 'determinism' / How do we make kernels batch-invariant?",
        ),
        ExperimentInfo(
            "hf_invariance",
            "detllm.experiments.hf_invariance:run",
            "hf",
            True,
            HF_MODES,
            "Model-level batch invariance of a HF transformers model (forward logits + greedy decode)",
            "Implementation (torch.Library kernel substitution)",
        ),
        ExperimentInfo(
            "vllm_completions",
            "detllm.experiments.vllm_completions:run",
            "vllm",
            True,
            VLLM_MODES,
            "N temperature-0 completions of one prompt; count unique outputs and first divergence",
            "Experiments: How nondeterministic are completions?",
        ),
        ExperimentInfo(
            "vllm_batch_sweep",
            "detllm.experiments.vllm_batch_sweep:run",
            "vllm",
            True,
            VLLM_MODES,
            "Needle prompt alone vs. mixed into batches of growing size; bitwise logprob comparison",
            "Batch invariance at the engine level (mirrors vLLM's own tests)",
        ),
        ExperimentInfo(
            "vllm_perf",
            "detllm.experiments.vllm_perf:run",
            "vllm",
            True,
            VLLM_MODES,
            "Throughput: 1000 requests with 90-110 output tokens through the API server",
            "Experiments: Performance",
        ),
    )
}


def get_experiment(name: str) -> ExperimentInfo:
    try:
        return EXPERIMENTS[name]
    except KeyError:
        known = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"unknown experiment {name!r} (known: {known})") from None


def load_runner(name: str) -> Callable[[Any], dict]:
    module_name, func_name = get_experiment(name).target.split(":")
    return getattr(importlib.import_module(module_name), func_name)
