"""Turn a run directory into report.md (+ summary.json). Pure Python."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from detllm.utils import now_iso, read_json_or_none, write_json

OK, BAD, ERR = "yes", "**NO**", "error"


def load_run(run_dir: Path) -> dict[str, Any]:
    jobs = []
    for job_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        spec = read_json_or_none(job_dir / "job.json")
        if spec is None:
            continue
        jobs.append(
            {
                "dir": job_dir,
                "spec": spec,
                "result": read_json_or_none(job_dir / "result.json") or {"status": "missing"},
                "env": read_json_or_none(job_dir / "env.json") or {},
            }
        )
    return {
        "run_dir": run_dir,
        "suite": read_json_or_none(run_dir / "suite.json") or {},
        "status": read_json_or_none(run_dir / "status.json") or {},
        "jobs": jobs,
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _ok(flag: Any) -> str:
    return ERR if flag is None else (OK if flag else BAD)


def _by_experiment(run: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [j for j in run["jobs"] if j["spec"].get("experiment") == name and j["result"].get("status") == "ok"]


def _summary(job: dict[str, Any]) -> dict[str, Any]:
    return job["result"].get("summary") or {}


def _bi_status(s: dict[str, Any]) -> str:
    """'deterministic' (1 token sequence, 1 logprob variant, no failed requests), 'nondeterministic' (several
    token sequences or logprob variants), or 'inconclusive' (failed requests, logprobs not compared, no data)."""
    lp = s.get("modal_logprob_variants")
    if (s.get("n_unique") or 0) > 1 or (lp or 0) > 1:
        return "nondeterministic"
    if s.get("n_unique") == 1 and lp == 1 and s.get("n_valid") and s.get("n_valid") == s.get("n"):
        return "deterministic"
    return "inconclusive"


def _is_bitwise_identical(s: dict[str, Any]) -> bool:
    return _bi_status(s) == "deterministic"


def _bi_verdict(s: dict[str, Any]) -> str:
    lp = s.get("modal_logprob_variants")
    if not s.get("n_valid"):
        return "no valid completions - inconclusive"
    if s.get("n_unique") != 1:
        verdict = "NOT all identical - batch invariance violated"
    elif lp and lp > 1:
        verdict = f"same tokens but {lp} logprob variants - NOT bitwise identical"
    elif lp == 1:
        verdict = "all bitwise identical, tokens and logprobs"
    else:
        verdict = "same tokens, logprobs not compared"
    if s.get("n") and s.get("n_valid") != s.get("n"):
        verdict += f"; INCOMPLETE: only {s.get('n_valid')}/{s.get('n')} requests succeeded"
    return verdict


def _modal_variants(s: dict[str, Any]) -> str:
    """Logprob variants among the modal completions - meaningless when the modal one occurs once."""
    lp, count = s.get("modal_logprob_variants"), s.get("modal_count") or 0
    if not lp:
        return "not compared"
    return f"{lp} of {count}" if count > 1 else f"n/a (modal x{count})"


def _is_prompt_comparison(job: dict[str, Any]) -> bool:
    p = job["spec"].get("params") or {}
    return (
        p.get("api", "completions") == "completions"
        and p.get("engine", "server") == "server"
        and not p.get("enable_prefix_caching")
        and not p.get("attention_backend")
    )


def _completion_summaries(run: dict[str, Any]) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    """(job, label, summary) per prompt: multi-prompt jobs (``prompts:``) give one entry per prompt."""
    out = []
    for job in _by_experiment(run, "vllm_completions"):
        s = _summary(job)
        if s.get("multi_prompt"):
            for entry in s.get("per_prompt") or []:
                entry = {**entry, "engine_info": s.get("engine_info", {})}
                out.append((job, f"{job['spec']['id']} #{entry.get('prompt_index', 0) + 1}", entry))
        else:
            out.append((job, job["spec"]["id"], s))
    return out


def _comparison_setup(job: dict[str, Any]) -> str:
    """Jobs compared with each other share every parameter except the mode-specific ones."""
    p = {k: v for k, v in (job["spec"].get("params") or {}).items() if k not in ("prompt", "bi_attention_backend")}
    return json.dumps(p, sort_keys=True, default=str)


def _prompt_pairs(run: dict[str, Any]) -> dict[str, dict[tuple[str, str], list[Any]]]:
    """{model: {(setup, prompt text): [baseline summary, batch-invariant summary]}} for plain completion jobs;
    jobs with different settings (n, max_tokens, one server per prompt vs shared ...) are never paired."""
    out: dict[str, dict[tuple[str, str], list[Any]]] = {}
    for job, _label, s in _completion_summaries(run):
        if not _is_prompt_comparison(job):
            continue
        slot = {"baseline": 0, "batch_invariant": 1}.get(s.get("mode"))
        if slot is None or not s.get("prompt"):
            continue
        pair = out.setdefault(s.get("model", "?"), {}).setdefault((_comparison_setup(job), s["prompt"]), [None, None])
        pair[slot] = s
    return out


# --------------------------------------------------------------------------- sections


def _environment(run: dict[str, Any]) -> str:
    env = next((j["env"] for j in run["jobs"] if j["env"].get("gpu", {}).get("name")), None)
    if env is None:
        env = next((j["env"] for j in run["jobs"] if j["env"]), {})
    gpu = env.get("gpu", {})
    pkgs = env.get("packages", {})
    smi = env.get("nvidia_smi", {})
    rows = [
        ["GPU", f"{gpu.get('name')} (cc {gpu.get('capability')}, {gpu.get('sm_count')} SMs)"],
        ["Shared memory / block (opt-in)", gpu.get("shared_memory_per_block_optin")],
        ["Unified memory (cudaMemGetInfo total)", f"{gpu.get('mem_get_info_total_gb')} GB"],
        ["Driver", smi.get("driver_version")],
        ["CUDA (torch build)", gpu.get("torch_cuda")],
        ["Container base image", env.get("base_image")],
        ["torch / triton", f"{pkgs.get('torch')} / {pkgs.get('triton')}"],
        ["vLLM / transformers", f"{pkgs.get('vllm')} / {pkgs.get('transformers')}"],
        ["batch_invariant_ops commit", env.get("batch_invariant_ops_commit")],
        ["repo commit", env.get("repo_commit")],
    ]
    return "## Environment\n\n" + _table(["", ""], rows)


def _headline(run: dict[str, Any]) -> str:
    lines = []
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for job in _by_experiment(run, "vllm_completions"):
        s = _summary(job)
        if s.get("multi_prompt"):
            continue  # per-prompt results; summarised by the per-prompt line below
        if job["spec"]["id"].startswith(("completions-", "moe-completions-", "run-", "vllm_completions-")):
            by_model.setdefault(s.get("model", "?"), {})[s.get("mode", "?")] = s
    for model, modes in by_model.items():
        base, bi = modes.get("baseline"), modes.get("batch_invariant")
        parts = []
        if base:
            div = base.get("earliest_divergence")
            parts.append(
                f"baseline vLLM produced **{base.get('n_unique')} unique** completions out of {base.get('n_valid')}"
                + (f" (first divergence at token {div + 1})" if div is not None else "")
            )
        if bi:
            parts.append(f"batch-invariant vLLM produced **{bi.get('n_unique')} unique** ({_bi_verdict(bi)})")
        if parts:
            lines.append(f"- **{model}**: " + "; ".join(parts) + ".")
    for model, pairs in _prompt_pairs(run).items():
        complete = [(b, i) for b, i in pairs.values() if b and i]
        if complete:
            varied = sum(1 for b, _ in complete if _bi_status(b) == "nondeterministic")
            status = [_bi_status(i) for _, i in complete]
            inconclusive = status.count("inconclusive")
            lines.append(
                f"- **{model}, per prompt**: default vLLM was nondeterministic (tokens or logprobs) for "
                f"**{varied}/{len(complete)}** prompts; batch-invariant vLLM was bitwise identical for "
                f"**{status.count('deterministic')}/{len(complete)}** prompts"
                + (f" ({inconclusive} inconclusive: failed requests or logprobs not compared)" if inconclusive else "")
                + " (details below)."
            )
    for job in _by_experiment(run, "vllm_batch_sweep"):
        s = _summary(job)
        lines.append(
            f"- needle sweep {s.get('model')} / {s.get('mode')}: {s.get('n_identical')}/{s.get('n_trials')} batched runs "
            f"bitwise identical to batch size 1"
            + (f" (differences at batch sizes {s.get('variant_batch_sizes')})" if s.get("variant_batch_sizes") else "")
        )
    if not lines:
        return ""
    return (
        "## Headline\n\n"
        + "\n".join(lines)
        + "\n\nBlog (Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens): default vLLM 80 unique completions, "
        "first divergence at token 103; batch-invariant kernels: 1 unique."
    )


def _fp(run: dict[str, Any]) -> str:
    jobs = _by_experiment(run, "fp_nonassoc")
    if not jobs:
        return ""
    s = _summary(jobs[0])
    assoc = "; ".join(f"`{k}` = {v}" for k, v in (s.get("associativity") or {}).items())
    snippet, loop = s.get("n_unique_sums"), s.get("n_unique_left_to_right", "-")
    if "n_unique_left_to_right" not in s and "n_unique_builtin_sum" in s:
        # results written before 2026-10-08 stored the loop count in n_unique_sums
        snippet, loop = s.get("n_unique_builtin_sum"), s.get("n_unique_sums")
    return (
        "## Floating-point non-associativity (CPU)\n\n"
        f"- {assoc}\n- Summing the same 8 numbers in 10,000 random orders gives **{snippet}** distinct results "
        f"with the blog's snippet on Python {s.get('python', '?')} (blog: {s.get('blog_n_unique_sums')}); a plain "
        f"left-to-right loop over the same orders gives {loop}."
    )


def _ops(run: dict[str, Any]) -> str:
    jobs = _by_experiment(run, "op_invariance")
    if not jobs:
        return ""
    order = ["baseline", "torch_flags", "cublaslt_ws1", "official", "official_full", "custom"]
    jobs.sort(key=lambda j: order.index(j["spec"].get("mode")) if j["spec"].get("mode") in order else 99)
    modes = [j["spec"]["mode"] for j in jobs]
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for job in jobs:
        for r in _summary(job).get("results", []):
            key = (r["op"], r["dtype"])
            if r.get("error"):
                text = "error: " + r["error"].split(":")[0]
            elif r.get("batch_invariant"):
                text = "invariant"
            else:
                text = f"varies (B>={r.get('first_variant_batch')}, max {r.get('max_abs_diff', 0):.2g})"
            if r.get("run_to_run_identical") is False:
                text += ", NOT run-to-run det."
            cells.setdefault(key, {})[job["spec"]["mode"]] = text
    rows = [[f"{op} ({dt})", *[cells[(op, dt)].get(m, "-") for m in modes]] for (op, dt) in cells]
    demo_rows = []
    for job in jobs:
        s = _summary(job)
        demo, rtr = s.get("blog_mm_demo", {}), s.get("blog_run_to_run", {})
        demo_rows.append(
            [
                job["spec"]["mode"],
                demo.get("max_abs_diff", demo.get("error")),
                rtr.get("mismatches", rtr.get("error")),
                f"{s.get('n_invariant')}/{(s.get('n_ops') or 0) - (s.get('n_errors') or 0)}",
            ]
        )
    return (
        "## Op level: run-to-run vs batch invariance\n\n"
        "Row 0 of each op computed at batch size 1 vs. inside larger batches; *invariant* = bitwise identical "
        "at every batch size.\n\n"
        + _table(["op", *modes], rows)
        + "\n\nBlog demos - `torch.mm(a[:1], b)` vs `torch.mm(a, b)[:1]` (blog: 1669.25 on their GPU) and 1000 "
        "repeated bf16 matmuls (blog: 0 mismatches):\n\n"
        + _table(["mode", "mm demo max |diff|", "run-to-run mismatches /1000", "invariant ops"], demo_rows)
    )


def _hf(run: dict[str, Any]) -> str:
    jobs = _by_experiment(run, "hf_invariance")
    if not jobs:
        return ""
    rows = []
    examples = []
    for job in jobs:
        s = _summary(job)
        diffs = [r["max_abs_diff"] for r in s.get("forward", [])]
        rows.append(
            [
                s.get("model"),
                s.get("mode"),
                s.get("attn_implementation"),
                _ok(s.get("forward_invariant")),
                max(diffs) if diffs else None,
                _ok(s.get("decode_invariant")),
                None if s.get("first_token_divergence") is None else s["first_token_divergence"] + 1,
                _ok((s.get("run_to_run") or {}).get("identical")),
            ]
        )
        for ex in s.get("examples", [])[:1]:
            examples.append(
                f"- {s.get('model')} / {s.get('mode')}, batch {ex['batch_size']}, token {ex['token_number']}: "
                f"`{ex['reference']}` vs `{ex['batched']}`"
            )
    return (
        "## HF transformers model: forward logits and greedy decoding vs batch size\n\n"
        + _table(
            [
                "model",
                "mode",
                "attention",
                "logits invariant",
                "max |dlogit|",
                "decode bitwise invariant",
                "first diverging token",
                "run-to-run identical",
            ],
            rows,
        )
        + ("\n\nExamples of divergence:\n\n" + "\n".join(examples) if examples else "")
    )


def _prompts(run: dict[str, Any]) -> str:
    """Default vs batch-invariant vLLM, side by side for every prompt that was run in both modes."""
    sections = []
    verdicts = {"deterministic": "**deterministic**", "nondeterministic": "**NOT deterministic**", "inconclusive": "inconclusive"}
    for model, pairs in _prompt_pairs(run).items():
        rows, examples = [], []
        setups = list(dict.fromkeys(setup for setup, _ in pairs))
        for i, ((setup, prompt), (base, bi)) in enumerate(pairs.items()):
            label = " ".join(prompt.split())
            label = label[:60] + ("..." if len(label) > 60 else "")
            if len(setups) > 1:  # several comparison setups in one run: say which row is which
                label = f"[setup {setups.index(setup) + 1}] {label}"
            row = [i + 1, label]
            for s in (base, bi):
                if s is None:
                    row += ["not run (or not ok)", "-", "-", "-"]
                    continue
                div = s.get("earliest_divergence")
                row += [
                    f"{s.get('n_valid')}/{s.get('n')}",
                    f"{s.get('n_unique')} (modal x{s.get('modal_count')})",
                    "-" if div is None else div + 1,
                    _modal_variants(s),
                ]
            row.append(verdicts[_bi_status(bi)] if base and bi else "incomplete")
            rows.append(row)
            ex = (base or {}).get("divergence_example")
            if ex:
                examples.append(
                    f"- prompt {i + 1}, token {ex['token_number']}: ...{ex['shared_prefix_tail']!r} -> "
                    f"{ex['modal_continuation']!r} vs {ex['other_continuation']!r}"
                )
        sections.append(
            f"## Per-prompt comparison on {model}: default vs batch-invariant vLLM\n\n"
            "Each prompt was sent N times at temperature 0 as concurrent requests to a `vllm serve` instance, first "
            "with default kernels, then with `VLLM_BATCH_INVARIANT=1`. *unique* counts distinct token sequences; "
            "*first divergence* is the earliest token (1-based) where any two completions differ; *logprob variants* "
            "counts distinct bitwise logprob vectors among the copies of the most common completion. "
            "**deterministic** = batch-invariant run gave 1 unique completion with bitwise identical logprobs and no "
            "failed requests.\n\n"
            + _table(
                [
                    "#",
                    "prompt",
                    "default: ok",
                    "default: unique",
                    "default: first divergence",
                    "default: logprob variants",
                    "batch-inv: ok",
                    "batch-inv: unique",
                    "batch-inv: first divergence",
                    "batch-inv: logprob variants",
                    "verdict",
                ],
                rows,
            )
            + (
                "\n\nExamples of where default-vLLM completions split (the most common completion vs the second most "
                "common):\n\n" + "\n".join(examples)
                if examples
                else ""
            )
        )
    return "\n\n".join(sections)


def _completions(run: dict[str, Any]) -> str:
    entries = _completion_summaries(run)
    if not entries:
        return ""
    rows = []
    examples = []
    for _job, label, s in entries:
        info = s.get("engine_info", {})
        div = s.get("earliest_divergence")
        rows.append(
            [
                label,
                s.get("model"),
                s.get("mode"),
                f"{s.get('engine')}/{info.get('api', 'offline')}",
                f"{s.get('n_valid')}/{s.get('n')}x{s.get('max_tokens')}",
                s.get("n_unique"),
                s.get("modal_count"),
                None if div is None else div + 1,
                _modal_variants(s),
                info.get("attention_backend_log") or (info.get("engine_kwargs") or {}).get("attention_backend"),
            ]
        )
        ex = s.get("divergence_example")
        if ex:
            examples.append(
                f"- `{label}` token {ex['token_number']}: ...{ex['shared_prefix_tail']!r} -> "
                f"{ex['modal_continuation']!r} vs {ex['other_continuation']!r}"
            )
    blog = (
        "\n\nBlog: Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens -> 80 unique (modal x78), first divergence "
        "at token 103 (\"Queens, New York\" vs \"New York City\"); batch-invariant: 1 unique. "
        "batch_invariant_ops' own vLLM example (Qwen3-8B, 100 tokens): 18 unique -> 1 unique."
    )
    return (
        "## vLLM: N temperature-0 completions of one prompt\n\n"
        + _table(
            [
                "job",
                "model",
                "mode",
                "engine/api",
                "ok/n x max_tokens",
                "unique",
                "modal count",
                "first divergence (token #)",
                "logprob variants of modal",
                "attention backend",
            ],
            rows,
        )
        + blog
        + ("\n\nWhere completions first diverge:\n\n" + "\n".join(examples) if examples else "")
    )


def _sweeps(run: dict[str, Any]) -> str:
    jobs = _by_experiment(run, "vllm_batch_sweep")
    if not jobs:
        return ""
    rows = []
    for job in jobs:
        s = _summary(job)
        rows.append(
            [
                s.get("model"),
                s.get("mode"),
                f"{s.get('n_identical')}/{s.get('n_trials')}",
                s.get("n_token_mismatch"),
                s.get("n_logprob_only_mismatch"),
                ", ".join(map(str, s.get("variant_batch_sizes") or [])) or "-",
                _ok((s.get("run_to_run") or {}).get("identical")),
            ]
        )
    return "## vLLM: needle request alone vs inside batches\n\n" + _table(
        ["model", "mode", "bitwise identical", "token mismatches", "logprob-only mismatches", "batch sizes that differ", "bs=1 run-to-run"],
        rows,
    )


def _perf(run: dict[str, Any]) -> str:
    jobs = _by_experiment(run, "vllm_perf")
    if not jobs:
        return ""
    base_time = {(_summary(j).get("model")): _summary(j).get("wall_time_s") for j in jobs if _summary(j).get("mode") == "baseline"}
    rows = []
    for job in jobs:
        s = _summary(job)
        bt = base_time.get(s.get("model"))
        rows.append(
            [
                s.get("model"),
                s.get("mode"),
                f"{s.get('n_ok')}/{s.get('n')}",
                s.get("wall_time_s"),
                s.get("output_tokens_per_s"),
                (s.get("latency_s") or {}).get("p50"),
                (s.get("latency_s") or {}).get("p99"),
                None if not bt else round(s["wall_time_s"] / bt, 2),
            ]
        )
    return (
        "## vLLM throughput (exclusive runs)\n\n"
        + _table(["model", "mode", "ok", "wall time (s)", "output tok/s", "p50 latency", "p99 latency", "x baseline"], rows)
        + "\n\nBlog (Qwen3-8B, one GPU, 1000 requests of 90-110 tokens): default 26 s, unoptimized deterministic 55 s, "
        "with improved attention 42 s."
    )


def _preflight(run: dict[str, Any]) -> str:
    jobs = _by_experiment(run, "preflight")
    if not jobs:
        return ""
    s = _summary(jobs[0])
    # The vllm and memory rows carry the facts the runbook asks to check (tuned tiles, page cache): keep them whole.
    rows = [
        [c["name"], "info" if c["ok"] is None else _ok(c["ok"]), str(c["detail"]) if c["name"] in ("vllm", "memory") else str(c["detail"])[:220]]
        for c in s.get("checks", [])
    ]
    return "## Preflight\n\n" + _table(["check", "ok", "detail"], rows)


def _failures(run: dict[str, Any]) -> str:
    bad = [j for j in run["jobs"] if j["result"].get("status") != "ok"]
    if not bad:
        return ""
    rows = [[j["spec"]["id"], j["result"].get("status"), (j["result"].get("error") or "")[:200], f"{j['dir'].name}/job.log"] for j in bad]
    return "## Jobs that did not finish\n\n" + _table(["job", "status", "error", "log"], rows)


def render(run: dict[str, Any]) -> str:
    suite = run["suite"]
    jobs = run["jobs"]
    n_ok = sum(1 for j in jobs if j["result"].get("status") == "ok")
    header = (
        f"# Determinism report - suite `{suite.get('name', '?')}`\n\n"
        f"Run directory `{run['run_dir']}` - started {suite.get('started', '?')}, report written {now_iso()} - "
        f"{n_ok}/{len(jobs)} jobs ok.\n\n"
        "Reproduction of Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025) on NVIDIA GB10."
    )
    sections = [
        header,
        _headline(run),
        _prompts(run),
        _environment(run),
        _preflight(run),
        _fp(run),
        _ops(run),
        _hf(run),
        _completions(run),
        _sweeps(run),
        _perf(run),
        _failures(run),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    path = run_dir / "report.md"
    path.write_text(render(run), encoding="utf-8")
    write_json(
        run_dir / "summary.json",
        {
            j["spec"]["id"]: {
                "experiment": j["spec"].get("experiment"),
                "model": j["spec"].get("model"),
                "mode": j["spec"].get("mode"),
                "status": j["result"].get("status"),
                "summary": j["result"].get("summary"),
            }
            for j in run["jobs"]
        },
    )
    return path
