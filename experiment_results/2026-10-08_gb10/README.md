# GB10 runs of 2026-10-08: default vs batch-invariant vLLM on our own prompts

**Full write-up of both models: [COMBINED_REPORT.md](COMBINED_REPORT.md).** Short version for stakeholders:
`docs/FINDINGS_2026-10-08.md`.

Dell Pro Max GB10 (DGX OS 7.2.3, GB10 cc 12.1, 121 GiB unified memory, driver 595.91.07) - container
`vllm/vllm-openai:v0.31.0` (vLLM 0.31.0, torch 2.13.0+cu130, Triton 3.7.1, transformers 5.17.0) -
repo `8e0e74a` plus the changes in each run's `code-changes-vs-8e0e74a.diff`.

The run files record commit IDs from before the history was rewritten (commit metadata only): `8e0e74a` is
now `f15762a` and `2fc6a27` is now `1b12b5f` (same file contents).

## Headline: Qwen3-8B, `configs/prompts/my_prompts.txt` (run `qwen3-8b-mine/`)

All 8 prompts x 200 requests, temperature 0, up to 1000 tokens, **mixed in one `vllm serve` instance**
(1600 concurrent requests, up to 256 decoding at once - what a shared server sees), first with default
kernels, then with `VLLM_BATCH_INVARIANT=1`. Completions compared by token ids and bitwise logprobs.
1600/1600 requests succeeded in both modes.

| # | prompt | default: distinct completions | default: first divergence (token) | batch-invariant: distinct completions | batch-invariant: logprob variants |
|---|---|---|---|---|---|
| 1 | distributed rate limiter | 172 / 200 | 29 | **1 / 200** | **1** |
| 2 | thread-safe LRU cache (code) | 97 / 200 | 62 | **1 / 200** | **1** |
| 3 | longest increasing subsequence | 116 / 200 | 50 | **1 / 200** | **1** |
| 4 | microservices failure analysis | 154 / 200 | 41 | **1 / 200** | **1** |
| 5 | database paradigms | 126 / 200 | 95 | **1 / 200** | **1** |
| 6 | haiku (25 tokens) | 1 / 200 | - | **1 / 200** | **1** |
| 7 | Richard Feynman | 143 / 200 | 59 | **1 / 200** | **1** |
| 8 | SQL source/target-table extraction | 5 / 200 (141 logprob variants among the 165 copies of the most common answer) | 45 | **1 / 200** | **1** |

- Default vLLM was nondeterministic for 7 of 8 prompts at temperature 0, often in substance, not just
  wording: different code (`Tuple, Union` vs `Tuple, List, Union`), a different root-cause diagnosis
  ("Inefficient Load Balancing" vs "Load-Dependent Throttling"), "filtering by" vs "filtered by" in the
  SQL JSON. Examples are in `qwen3-8b-mine/report.md`.
- Batch-invariant vLLM was bitwise identical for 8 of 8 prompts: one completion per prompt, one logprob
  vector across all 200 copies.
- Speed for the same work (~1.23 M generated tokens each): default 896 tok/s, batch-invariant 1019 tok/s
  (vLLM 0.31 ships GB10-tuned batch-invariant matmul tiles for Qwen3-8B and auto-enables breakable CUDA
  graphs). One measurement each, under this mixed load - not the dedicated `vllm_perf` test.

## Qwen3-30B-A3B-Instruct-2507 (mixture of experts, run `qwen3-30b-a3b-mine/`)

Same method, 8 prompts x 100 requests per mode (800 requests per mode, 0 failed):

| # | prompt | default: distinct completions | default: first divergence (token) | batch-invariant: distinct completions | batch-invariant: logprob variants |
|---|---|---|---|---|---|
| 1 | distributed rate limiter | 80 / 100 | 33 | **1 / 100** | **1** |
| 2 | thread-safe LRU cache (code) | 88 / 100 | 17 | **1 / 100** | **1** |
| 3 | longest increasing subsequence | 89 / 100 | 76 | **1 / 100** | **1** |
| 4 | microservices failure analysis | 81 / 100 | 38 | **1 / 100** | **1** |
| 5 | database paradigms | 84 / 100 | 11 | **1 / 100** | **1** |
| 6 | haiku | 20 / 100 | 52 | **1 / 100** | **1** |
| 7 | Richard Feynman | 85 / 100 | 23 | **1 / 100** | **1** |
| 8 | SQL source/target-table extraction | 7 / 100 (48 logprob variants among the 59 copies of the most common answer) | 32 | **1 / 100** | **1** |

Default vLLM was nondeterministic for 8 of 8 prompts - more than the dense Qwen3-8B, plausibly because a
small numeric difference can change which experts a token is routed to. Batch-invariant vLLM (fixed-config
fused MoE kernels) was bitwise identical for 8 of 8. Speed for the same work (~645 k tokens): default 467,
batch-invariant 464 output tok/s. Server start: 588 s (default) and 431 s (batch-invariant) for 61 GB of weights.

## Other evidence

- **One prompt per server** (`qwen3-8b-one-prompt-per-server-partial/`, default vLLM only, 200 x 1000
  tokens, `max_num_seqs` 96): rate limiter 8, LRU cache 7, LIS 9, microservices 12 distinct completions out
  of 200. Fewer than in the mixed run - identical requests move through the engine in near-lockstep - but
  still nondeterministic. Stopped after 4 prompts when the suite was restructured into `qwen3-8b-mine`.
- **Engine-level needle sweep** (in `qwen3-8b-mine/`, cancelled after batch size 32/64 to free the GPU):
  the rate-limiter prompt alone vs. hidden among 2..N other requests - default vLLM differed from the
  bs=1 answer in all 10 batched trials, at every batch size from 2 to 32 (first differing token 141, 133 or
  210; one trial in the logprobs only); batch-invariant was bitwise identical at every size it reached (1-64).
  Per-trial lines are in `sweep-*/job.log`.
- **Smoke suite** (`smoke-qwen3-1.7b/`): 100 x 256 tokens - default 3 distinct completions (divergence at
  token 127), batch-invariant 1; needle sweep 1/3 vs 3/3 identical.
- **Op level** (`qwen3-8b-mine/report.md`): stock and `torch_flags` matmul/linear vary with batch size while
  being run-to-run deterministic (0/1000 mismatches); `official`, `official_full` and `custom` make every
  matmul invariant; PyTorch SDPA decode varies in every mode. `torch.mm(a[:1], b)` vs `torch.mm(a, b)[:1]`:
  1641 (blog: 1669.25).
- **Floating-point demo**: the blog's snippet gives exactly 102 distinct sums on Python 3.12 (its `sum()`
  adds the int literal `1` without compensation); a plain left-to-right loop gives 258.

## Known limitations found on this device

- `custom` kernel mode (fixed-size chunks through cuBLAS): fixed in this run for `torch.inference_mode`
  (it was a no-op), but a row's result still depends on its position inside the 64-row chunk on GB10 -
  e.g. Qwen3 HF forward at batch 4 with the target last. Shapes alone do not pin cuBLAS's arithmetic.
- vLLM 0.31.0 refuses to start batch-invariant mode for hybrid linear-attention / Mamba models (Qwen3.5,
  Qwen3.6, Qwen3.8, Nemotron-H): their attention backends do not declare batch-invariance support.
- Prefix caching with batch-invariant mode was not measured here; the deterministic server config keeps it off.

## Files

Each run directory keeps the harness output: `report.md`, `summary.json`, `status.json`, `suite.json`, and
per job `job.json`, `result.json`, `env.json`, `job.log`, `vllm_server.log`. Raw completions (token ids,
text, per-token logprobs) are `completions.jsonl.zst` (`zstd -d` to unpack). `host_info.txt` is the host
snapshot. The full uncompressed run directories stay in `results/` on the GB10 (gitignored).

Reproduce: `bash scripts/gb10/run_suite.sh configs/suites/qwen3-8b-mine.yaml` (see docs/GB10_RUNBOOK.md).
