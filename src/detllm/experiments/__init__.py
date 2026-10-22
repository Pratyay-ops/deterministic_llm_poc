"""Experiments. Each module exposes ``run(ctx: JobContext) -> dict`` (the job summary).

Only the registry is imported here; experiment modules import torch/vLLM and are loaded lazily.
"""

from detllm.experiments.registry import EXPERIMENTS, ExperimentInfo, get_experiment, load_runner

__all__ = ["EXPERIMENTS", "ExperimentInfo", "get_experiment", "load_runner"]
