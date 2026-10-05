"""Unit tests run anywhere Python runs; tests marked ``gpu`` need CUDA (run them in the container
on the GB10: ``bash scripts/gb10/run.sh python3 -m pytest -m gpu``)."""

import pytest


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):  # noqa: ANN001, ARG001
    if _has_cuda():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
