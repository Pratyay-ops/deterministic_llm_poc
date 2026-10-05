"""``vllm serve`` launcher and a concurrent OpenAI-API client.

The blog's experiments are about what a *user of a server* sees: requests arrive concurrently,
the server batches them however its load dictates, and with non-batch-invariant kernels the same
request gets different numerics depending on what it was batched with. Sending N concurrent
HTTP requests (as thinking-machines-lab/batch_invariant_ops/deterministic_vllm_inference.py does)
reproduces exactly that.

The server runs in the job's process group, so if the scheduler kills the job (timeout / Ctrl+C)
the server and its engine-core process go down with it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from detllm.config import ModelPreset
from detllm.engines.vllm_engine import engine_kwargs, is_memory_error, to_cli_flags, vllm_version
from detllm.jobs import RetryLater

log = logging.getLogger(__name__)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _tail(path: Path, n_bytes: int = 6000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - n_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class VllmServer:
    def __init__(
        self,
        preset: ModelPreset,
        params: dict[str, Any],
        *,
        mem_gb: float,
        mode: str,
        log_path: Path,
        startup_timeout_s: float = 1800.0,
    ):
        self.preset = preset
        self.mode = mode
        self.kwargs = engine_kwargs(preset, params, mem_gb=mem_gb, mode=mode)
        self.port = int(params.get("port") or free_port())
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.served_model_name = preset.hf_id
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        exe = shutil.which("vllm")
        launcher = [exe] if exe else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
        self.cmd = [
            *launcher,
            "serve",
            preset.hf_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            *to_cli_flags(self.kwargs),
            *[str(x) for x in params.get("server_extra_args") or []],
        ]
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        self.startup_s: float | None = None

    def start(self) -> None:
        import httpx

        log.info("starting vLLM %s server (%s): %s", vllm_version(), self.mode, " ".join(self.cmd))
        self._log_fh = open(self.log_path, "ab")
        t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log_fh, stderr=subprocess.STDOUT, env=dict(os.environ))
        while time.time() - t0 < self.startup_timeout_s:
            rc = self.proc.poll()
            if rc is not None:
                tail = _tail(self.log_path)
                if is_memory_error(tail):
                    raise RetryLater(f"vllm serve could not reserve memory yet (exit {rc})")
                raise RuntimeError(f"vllm serve exited with code {rc}; log tail:\n{tail[-3000:]}")
            try:
                if httpx.get(self.base_url + "/health", timeout=5.0).status_code == 200:
                    self.startup_s = time.time() - t0
                    log.info("server ready on port %d after %.0fs", self.port, self.startup_s)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3.0)
        self.stop()
        raise TimeoutError(f"vllm serve not healthy after {self.startup_timeout_s:.0f}s (see {self.log_path})")

    def stop(self, timeout_s: float = 60.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> VllmServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def info(self) -> dict[str, Any]:
        return {
            "vllm_version": vllm_version(),
            "engine_kwargs": self.kwargs,
            "command": self.cmd,
            "startup_s": round(self.startup_s or 0.0, 1),
        }


# --------------------------------------------------------------------------- requests


def completion_payload(
    model: str,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    seed: int,
    logprobs: bool = True,
    ignore_eos: bool = False,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_ids,  # token ids -> identical input to the offline engine
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "ignore_eos": ignore_eos,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    if logprobs:
        payload["logprobs"] = 0
    return payload


def chat_payload(
    model: str,
    user_text: str,
    *,
    max_tokens: int,
    seed: int,
    chat_template_kwargs: dict[str, Any] | None = None,
    logprobs: bool = True,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 0
    return payload


def _as_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _token_id(token: Any) -> int | None:
    if isinstance(token, str) and token.startswith("token_id:"):
        try:
            return int(token.split(":", 1)[1])
        except ValueError:
            return None
    return None


def parse_choice(response: dict[str, Any], api: str) -> dict[str, Any]:
    """Token ids, text, per-token logprobs and finish reason of the first choice."""
    choice = response["choices"][0]
    if api == "chat":
        text = (choice.get("message") or {}).get("content")
        content = (choice.get("logprobs") or {}).get("content")
        tokens = [c.get("token") for c in content] if content else None
        logprobs = [_as_float(c.get("logprob")) for c in content] if content else None
    else:
        text = choice.get("text")
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens")
        logprobs = [_as_float(x) for x in lp["token_logprobs"]] if lp.get("token_logprobs") else None
    token_ids = choice.get("token_ids")
    if token_ids is None and tokens:
        parsed = [_token_id(t) for t in tokens]
        token_ids = parsed if all(t is not None for t in parsed) else None
    return {
        "token_ids": [int(t) for t in token_ids] if token_ids is not None else None,
        "text": text,
        "logprobs": logprobs,
        "finish_reason": choice.get("finish_reason"),
    }


async def _send_all(
    base_url: str,
    endpoint: str,
    payloads: list[dict[str, Any]],
    *,
    concurrency: int,
    jitter_s: float,
    seed: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    import httpx

    rng = random.Random(seed)
    delays = [rng.uniform(0.0, jitter_s) if jitter_s > 0 else 0.0 for _ in payloads]
    limit = max(1, min(concurrency, len(payloads)))
    sem = asyncio.Semaphore(limit)
    results: list[dict[str, Any]] = [{} for _ in payloads]
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit)
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s), limits=limits) as client:

        async def one(i: int, payload: dict[str, Any]) -> None:
            if delays[i]:
                await asyncio.sleep(delays[i])
            async with sem:
                t0 = time.perf_counter()
                try:
                    resp = await client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    results[i] = {"ok": True, "json": resp.json(), "latency_s": time.perf_counter() - t0}
                except Exception as exc:  # noqa: BLE001
                    body = getattr(getattr(exc, "response", None), "text", "")
                    results[i] = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc} {body[:300]}",
                        "latency_s": time.perf_counter() - t0,
                    }

        await asyncio.gather(*(one(i, p) for i, p in enumerate(payloads)))
    return results


def send_requests(
    base_url: str,
    api: str,
    payloads: list[dict[str, Any]],
    *,
    concurrency: int | None = None,
    jitter_s: float = 0.0,
    seed: int = 0,
    timeout_s: float = 7200.0,
) -> tuple[list[dict[str, Any]], float]:
    """Fire all payloads concurrently. Returns (results in payload order, wall-clock seconds)."""
    endpoint = "/v1/chat/completions" if api == "chat" else "/v1/completions"
    t0 = time.perf_counter()
    results = asyncio.run(
        _send_all(
            base_url,
            endpoint,
            payloads,
            concurrency=concurrency or len(payloads),
            jitter_s=jitter_s,
            seed=seed,
            timeout_s=timeout_s,
        )
    )
    return results, time.perf_counter() - t0
