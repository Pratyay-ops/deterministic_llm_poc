#!/usr/bin/env python3
"""Check a running deterministic server: send every prompt N times at once (mixed, like several users)
and verify that all copies of a prompt come back bitwise identical - same text and the same logprob for
every token. Half of the copies leave out temperature and seed, so the server's own default (greedy, as
set by serve_deterministic.sh) is checked too. Python 3 stdlib only.

    python3 scripts/gb10/check_determinism.py                         # 3 built-in prompts x 30, 300 tokens
    python3 scripts/gb10/check_determinism.py --prompts-file configs/prompts/my_prompts.txt -n 50

Exit status: 0 = every prompt gave one bitwise-identical answer; 1 = some prompt gave several answers;
2 = inconclusive (server unreachable, failed requests, or no logprobs to compare).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request

DEFAULT_PROMPTS = [
    "Tell me about Richard Feynman",
    "Write a Python function that merges two sorted lists, with tests.",
    "Explain the trade-offs between TCP and UDP for a real-time game.",
]


def load_prompts(path: str) -> list[str]:
    """One prompt per non-empty line; lines starting with '#' are comments (as in configs/prompts/)."""
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.lstrip().startswith("#")]


def _call(url: str, body: dict | None, api_key: str | None, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=timeout) as resp:
        return json.load(resp)


def ask(url: str, model: str, prompt: str, max_tokens: int, api_key: str | None, server_default: bool):
    """(text, logprobs or None). JSON carries the server's float32 logprobs exactly, so tuple equality
    is a bitwise comparison."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 0,
    }
    if not server_default:
        body.update(temperature=0, seed=0)
    choice = _call(url + "/v1/chat/completions", body, api_key, 3600)["choices"][0]
    content = (choice.get("logprobs") or {}).get("content")
    logprobs = tuple(t["logprob"] for t in content) if content else None
    return choice["message"]["content"] or "", logprobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", help="served model name (default: the first one the server lists)")
    ap.add_argument("--api-key", help="for servers started with API_KEY (Authorization: Bearer)")
    ap.add_argument("--prompts-file", help="prompt file, one prompt per line (default: 3 built-in prompts)")
    ap.add_argument("-n", type=int, default=30, help="copies of every prompt, all sent at once (default 30)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--no-default-path", action="store_true",
                    help="send temperature 0 and seed 0 on every request instead of leaving half to the server default")
    args = ap.parse_args()
    if args.n < 1:
        ap.error("-n must be at least 1")

    prompts = load_prompts(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS
    if not prompts:
        ap.error(f"no prompts in {args.prompts_file}")
    url = args.url.rstrip("/")
    try:
        model = args.model or _call(url + "/v1/models", None, args.api_key, 30)["data"][0]["id"]
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as exc:
        print(f"cannot query {url}/v1/models: {exc} - is the server running (serve_deterministic.sh status)?")
        return 2

    work = [(k, prompt, not args.no_default_path and j % 2 == 1) for k, prompt in enumerate(prompts) for j in range(args.n)]
    print(f"{model} at {url}: {len(prompts)} prompts x {args.n} copies = {len(work)} requests at once, "
          f"max_tokens {args.max_tokens}, "
          + ("all at temperature 0" if args.no_default_path else "half at temperature 0, half at the server default"))

    variants: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(work), 256)) as pool:
        futures = {pool.submit(ask, url, model, p, args.max_tokens, args.api_key, d): k for k, p, d in work}
        for fut in concurrent.futures.as_completed(futures):
            try:
                variants[futures[fut]][fut.result()] += 1
            except Exception as exc:  # noqa: BLE001 - report every failure kind
                errors.append(f"prompt {futures[fut] + 1}: {type(exc).__name__}: {exc}")

    worst = 0  # 0 identical, 2 inconclusive, 1 varies (1 wins)
    for k, prompt in enumerate(prompts):
        counts = variants.get(k, collections.Counter())
        n_ok = sum(counts.values())
        texts = {text for text, _ in counts}
        text_only = any(lps is None for _, lps in counts)
        if len(counts) > 1:
            verdict, code = "VARIES", 1
        elif n_ok < args.n:
            verdict, code = "INCOMPLETE", 2
        elif text_only:
            verdict, code = "TEXT-ONLY", 2  # same text, but no logprobs to compare bitwise
        else:
            verdict, code = "identical", 0
        worst = 1 if 1 in (worst, code) else max(worst, code)
        label = " ".join(prompt.split())[:60]
        print(f"  {k + 1:2d}. {verdict:10s} {n_ok}/{args.n} ok, {len(texts)} distinct texts, "
              f"{len(counts)} distinct (text, logprobs)  | {label}")
    for err in errors[:5]:
        print("  error:", err)
    if errors:
        print(f"{len(errors)} requests failed")
    print({0: "DETERMINISTIC: every copy of every prompt was bitwise identical",
           1: "NOT deterministic: some prompt produced more than one answer",
           2: "INCONCLUSIVE: failed requests or missing logprobs (see above)"}[worst])
    return worst


if __name__ == "__main__":
    sys.exit(main())
