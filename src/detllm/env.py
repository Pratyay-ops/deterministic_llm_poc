"""Environment snapshot written next to every job's results (env.json).

GPU details are collected lazily (torch is only imported when ``gpu=True``).
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from detllm.utils import REPO_ROOT, meminfo_gb

PACKAGES = (
    "torch",
    "triton",
    "vllm",
    "transformers",
    "flashinfer-python",
    "batch-invariant-ops",
    "huggingface-hub",
    "httpx",
    "numpy",
)
ENV_PREFIXES = (
    "VLLM_",
    "CUBLAS",
    "CUBLASLT",
    "TRITON_",
    "TORCH",
    "PYTORCH_",
    "NCCL_",
    "CUDA_",
    "HF_",
    "DETLLM_",
    "NVIDIA_",
)
_SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def relevant_env() -> dict[str, str]:
    """Environment variables that influence numerics, minus anything that looks like a secret."""
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(ENV_PREFIXES) and not any(m in k for m in _SECRET_MARKERS)
    }


def read_git_head(repo_dir: Path) -> str | None:
    """Commit SHA of a checkout (works without a git binary; handles submodule .git files)."""
    git = repo_dir / ".git"
    try:
        if git.is_file():  # submodule / worktree: "gitdir: <path>"
            target = git.read_text(encoding="utf-8").strip().split(":", 1)[1].strip()
            git = (repo_dir / target).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        ref_file = git / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip()
        packed = git / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0]
    except (OSError, IndexError):
        return None
    return None


def nvidia_smi() -> dict[str, Any]:
    query = "name,driver_version,compute_cap,memory.total"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        fields = [f.strip() for f in out.stdout.strip().splitlines()[0].split(",")] if out.stdout else []
        return dict(zip(query.split(","), fields)) if fields else {"error": out.stderr.strip()[:300]}
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def gpu_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    info.update(
        name=props.name,
        capability=f"{props.major}.{props.minor}",
        sm_count=props.multi_processor_count,
        total_memory_gb=round(props.total_memory / 1e9, 2),
        mem_get_info_free_gb=round(free / 1e9, 2),
        mem_get_info_total_gb=round(total / 1e9, 2),
        shared_memory_per_block_optin=device_smem_optin(),
        cudnn=torch.backends.cudnn.version(),
        is_gb10=is_gb10(props.name, props.major, props.minor),
    )
    return info


def device_smem_optin(device: int = 0) -> int | None:
    """Max dynamic shared memory per block (opt-in), bytes. Decides which Triton tiles fit."""
    import torch

    props = torch.cuda.get_device_properties(device)
    for attr in ("shared_memory_per_block_optin", "max_shared_memory_per_block_optin"):
        value = getattr(props, attr, None)
        if value:
            return int(value)
    try:
        import triton

        return int(triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"])
    except Exception:  # noqa: BLE001
        return None


def is_gb10(name: str, major: int, minor: int) -> bool:
    return "GB10" in name.upper() or (major, minor) == (12, 1)


def collect_env(*, gpu: bool = True) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "packages": package_versions(),
        "repo_commit": read_git_head(REPO_ROOT),
        "batch_invariant_ops_commit": read_git_head(REPO_ROOT / "batch_invariant_ops"),
        "base_image": os.environ.get("DETLLM_BASE_IMAGE"),
        "meminfo_gb": {
            k: round(v, 1)
            for k, v in meminfo_gb().items()
            if k in ("MemTotal", "MemAvailable", "Cached", "SwapTotal", "SwapFree")
        },
        "env": relevant_env(),
    }
    if gpu:
        try:
            env["gpu"] = gpu_info()
        except Exception as exc:  # noqa: BLE001
            env["gpu"] = {"error": f"{type(exc).__name__}: {exc}"}
        env["nvidia_smi"] = nvidia_smi()
    return env
