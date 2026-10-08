#!/usr/bin/env python3
"""Check a running deterministic server: send every prompt N times at once (mixed, like several users)
and verify that all copies of a prompt come back bitwise identical - same text and the same logprob for
every token. Python 3 stdlib only.

    python3 scripts/gb10/check_determinism.py                         # 3 built-in prompts x 20, 200 tokens
    python3 scripts/gb10/check_determinism.py --prompts-file configs/prompts/my_prompts.txt -n 50 --max-tokens 400

Exit status: 0 = every prompt gave one variant, 1 = some prompt gave several, 2 = requests failed.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import sys
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


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def ask(url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, tuple[float, ...]]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 0,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        choice = json.load(resp)["choices"][0]
    tokens = (choice.get("logprobs") or {}).get("content") or []
    # JSON carries the server's float32 logprobs exactly, so tuple equality is a bitwise comparison.
    return choice["message"]["content"] or "", tuple(t["logprob"] for t in tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", help="served model name (default: the first one the server lists)")
    ap.add_argument("--prompts-file", help="prompt file, one prompt per line (default: 3 built-in prompts)")
    ap.add_argument("-n", type=int, default=20, help="copies of every prompt, all sent at once (default 20)")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    url = args.url.rstrip("/")
    model = args.model or _get(url + "/v1/models")["data"][0]["id"]
    prompts = load_prompts(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS
    work = [(k, prompt) for k, prompt in enumerate(prompts) for _ in range(args.n)]
    print(f"{model} at {url}: {len(prompts)} prompts x {args.n} copies = {len(work)} requests at once, "
          f"temperature 0, max_tokens {args.max_tokens}")

    variants: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(work), 256)) as pool:
        futures = {pool.submit(ask, url, model, prompt, args.max_tokens): k for k, prompt in work}
        for fut in concurrent.futures.as_completed(futures):
            try:
                variants[futures[fut]][fut.result()] += 1
            except Exception as exc:  # noqa: BLE001 - report every failure kind
                errors.append(f"prompt {futures[fut] + 1}: {type(exc).__name__}: {exc}")

    deterministic = True
    for k, prompt in enumerate(prompts):
        counts = variants.get(k, collections.Counter())
        n_ok = sum(counts.values())
        texts = {text for text, _ in counts}
        label = " ".join(prompt.split())[:60]
        verdict = "identical" if len(counts) == 1 and n_ok == args.n else "VARIES"
        deterministic &= verdict == "identical"
        print(f"  {k + 1:2d}. {verdict:9s} {n_ok}/{args.n} ok, {len(texts)} distinct texts, "
              f"{len(counts)} distinct (text, logprobs)  | {label}")
    for err in errors[:5]:
        print("  error:", err)
    if errors:
        print(f"{len(errors)} requests failed")
        return 2
    print("DETERMINISTIC: every copy of every prompt was bitwise identical" if deterministic
          else "NOT deterministic: some prompt produced more than one variant")
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
