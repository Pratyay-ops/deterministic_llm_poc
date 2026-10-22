# Deterministic LLM inference on an NVIDIA GB10 - combined experiment report

**Models:** Qwen3-8B (dense) and Qwen3-30B-A3B-Instruct-2507 (mixture of experts) -
**Runs:** 2026-10-08 on one Dell Pro Max GB10 - **Code:** branch `deterministic_llm_poc` (runs at commit
`2fc6a27`) - **Data:** this directory. Every number below is computed from the run files in this directory.

## 1. Executive summary

- **The problem is real and large.** At temperature 0, with the same prompt and settings, a standard vLLM
  server returned **up to 172 different answers out of 200 identical requests** (Qwen3-8B) and **up to 89 out of
  100** (Qwen3-30B-A3B) once other requests shared the server. Default vLLM was nondeterministic for **7 of 8**
  prompts on Qwen3-8B and **8 of 8** on Qwen3-30B-A3B.
- **The differences in answer are substantive:** different code for implementation, a different root-cause diagnosis to same problem, different JSON answer for same questions given to the normal llm.
  content - not only wording.
- **The fix works on both models.** With vLLM's batch-invariant mode (`VLLM_BATCH_INVARIANT=1`) every one of the
  16 prompt-model combinations returned **exactly one answer, bitwise identical down to every logprob**, across
  all 2,400 requests of that mode (0 failed).
- **At no speed cost on this machine:** Qwen3-8B 896 -> 1019 output tokens/s, Qwen3-30B-A3B 467 -> 464.
- **Root cause confirmed:** an answer changes with the batch it is computed in (the same prompt diverged from
  its stand-alone answer at every batch size from 2 to 32), while every kernel is repeatable for a fixed batch -
  exactly the mechanism described by Thinking Machines Lab.

| model | requests per mode | default: nondeterministic prompts | default: distinct answers per prompt | batch-invariant: bitwise-identical prompts | failed requests (default / batch-inv.) | output tok/s default -> batch-invariant | job time default / batch-invariant (server start) |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 8 x 200 = 1600 | 7 / 8 | 1-172 of 200 | **8 / 8** | 0 / 0 | 896 -> 1019 | 27 min (229 s) / 23 min (136 s) |
| Qwen3-30B-A3B-Instruct-2507 | 8 x 100 = 800 | 8 / 8 | 7-89 of 100 | **8 / 8** | 0 / 0 | 467 -> 464 | 33 min (588 s) / 31 min (431 s) |

## 2. Background and objective

LLM inference at temperature 0 is greedy and is widely assumed to be deterministic. Thinking Machines Lab
(*Defeating Nondeterminism in LLM Inference*, Horace He et al., 2025) showed why it is not in practice: the
GPU kernels of a forward pass are run-to-run deterministic, but **not batch-invariant** - the result for one
request depends on how many other requests are processed with it, because kernels choose different tilings
and reduction orders for different batch sizes, and floating-point addition is not associative. A server's
batch size changes with load, so users see different answers. Batch-invariant kernels remove the dependence.

Objective of this work: measure the effect **on our own prompts and on models we could use day to day**, on
our own hardware (a GB10 workstation), verify that vLLM's batch-invariant mode removes it, and measure what it costs.

## 3. Setup

**Hardware.** Dell Pro Max GB10 (DGX Spark class): NVIDIA GB10 Blackwell GPU (compute capability 12.1, 48 SMs),
20-core Arm CPU, 128 GB unified CPU/GPU memory (130.6 GB reported), DGX OS 7.2.3, driver 595.91.07.

**Software.** Container `vllm/vllm-openai:v0.31.0`: vLLM 0.31.0, PyTorch 2.13.0+cu130, Triton 3.7.1,
transformers 5.17.0, CUDA 13.0. Harness: this repository (`detllm`), batch_invariant_ops submodule `f22b1fb`.

**Models.**

| model | architecture | weights (bf16) | KV cache per token | why |
|---|---|---|---|---|
| Qwen3-8B | dense transformer, 36 layers | 16.4 GB | 144 KiB | the blog's own performance model; vLLM ships batch-invariant kernels tuned for its shapes on this GPU class |
| Qwen3-30B-A3B-Instruct-2507 | mixture of experts, 128 experts, ~3B of 30B parameters active per token | 61.1 GB | 96 KiB | closest model to the blog's Qwen3-235B-A22B that fits the GB10; tests MoE routing |

Hybrid linear-attention / Mamba models (Qwen3.5, 3.6, 3.8, Nemotron-H) were excluded: vLLM 0.31's
batch-invariant mode refuses to start for them.

**Prompts** (`configs/prompts/my_prompts.txt`; prompt length in tokens after the chat template, Qwen3-8B / 30B):

| # | prompt | prompt tokens | typical answer length (tokens, 8B / 30B) |
|---|---|---|---|
| 1 | Distributed rate limiter design (1M req/s, 100 servers) | 99 / 95 | 1000 (cap) / 1000 |
| 2 | Thread-safe LRU cache in Python, with tests | 123 / 119 | 1000 / 1000 |
| 3 | Longest increasing subsequence, four solutions | 151 / 147 | 1000 / 1000 |
| 4 | Microservices cascading-failure analysis | 179 / 175 | 1000 / 1000 |
| 5 | PostgreSQL vs MongoDB vs Neo4j for 50M users | 210 / 206 | 1000 / 1000 |
| 6 | Haiku about programming | 34 / 30 | 25 / ~170 |
| 7 | "Tell me about Richard Feynman" | 19 / 15 | ~790 / ~920 |
| 8 | SQL lineage: extract source/target tables as JSON | 502 / 498 | ~330 / ~360 |

**Method.**
- Each prompt was sent N times (Qwen3-8B: 200, Qwen3-30B-A3B: 100). All 8 x N requests were posted **at once,
  in a shuffled order, to one `vllm serve` instance** - the way a shared server sees several users - with up to
  256 (8B) or 160 (30B) sequences decoding together.
- Temperature 0, up to 1000 generated tokens, seed 0, bf16, prefix caching off, prompts sent as token ids
  (both modes get identical input). Qwen3-8B in non-thinking mode.
- Two modes, one after the other, each alone on the GPU: **default** vLLM (FLASH_ATTN attention, stock
  kernels), then **batch-invariant** vLLM (`VLLM_BATCH_INVARIANT=1`, TRITON_ATTN attention, vLLM's Triton
  matmul with GB10-tuned tiles, fixed-config fused-MoE kernels).
- Comparison per prompt: *distinct answers* = distinct token sequences among the N copies; *first divergence*
  = earliest token (1-based) where any two copies differ; *logprob variants* = distinct per-token logprob
  vectors, compared bit for bit, among the copies of the most common answer. A prompt is **deterministic**
  when there is 1 answer, 1 logprob variant, and no failed request.

## 4. Results

### 4.1 Qwen3-8B

| # | prompt | default: distinct answers | most common answer | first divergence (token) | logprob variants of most common | batch-invariant: distinct answers | logprob variants | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Distributed rate limiter | 172 / 200 | 4x | 29 | 1 of 4 | **1 / 200** | **1** | deterministic |
| 2 | Thread-safe LRU cache (Python code) | 97 / 200 | 14x | 62 | 10 of 14 | **1 / 200** | **1** | deterministic |
| 3 | Longest increasing subsequence | 116 / 200 | 13x | 50 | 9 of 13 | **1 / 200** | **1** | deterministic |
| 4 | Microservices failure analysis | 154 / 200 | 9x | 41 | 9 of 9 | **1 / 200** | **1** | deterministic |
| 5 | Database paradigms comparison | 126 / 200 | 9x | 95 | 7 of 9 | **1 / 200** | **1** | deterministic |
| 6 | Haiku about programming | 1 / 200 | 200x | - | 1 of 200 | **1 / 200** | **1** | deterministic |
| 7 | Richard Feynman | 143 / 200 | 6x | 59 | 5 of 6 | **1 / 200** | **1** | deterministic |
| 8 | SQL source/target-table extraction (JSON) | 5 / 200 | 165x | 45 | 141 of 165 | **1 / 200** | **1** | deterministic |

### 4.2 Qwen3-30B-A3B-Instruct-2507

| # | prompt | default: distinct answers | most common answer | first divergence (token) | logprob variants of most common | batch-invariant: distinct answers | logprob variants | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Distributed rate limiter | 80 / 100 | 4x | 33 | 1 of 4 | **1 / 100** | **1** | deterministic |
| 2 | Thread-safe LRU cache (Python code) | 88 / 100 | 3x | 17 | 1 of 3 | **1 / 100** | **1** | deterministic |
| 3 | Longest increasing subsequence | 89 / 100 | 3x | 76 | 1 of 3 | **1 / 100** | **1** | deterministic |
| 4 | Microservices failure analysis | 81 / 100 | 4x | 38 | 1 of 4 | **1 / 100** | **1** | deterministic |
| 5 | Database paradigms comparison | 84 / 100 | 4x | 11 | 1 of 4 | **1 / 100** | **1** | deterministic |
| 6 | Haiku about programming | 20 / 100 | 22x | 52 | 22 of 22 | **1 / 100** | **1** | deterministic |
| 7 | Richard Feynman | 85 / 100 | 3x | 23 | 1 of 3 | **1 / 100** | **1** | deterministic |
| 8 | SQL source/target-table extraction (JSON) | 7 / 100 | 59x | 32 | 48 of 59 | **1 / 100** | **1** | deterministic |

### 4.3 What the differences look like

Where the most common default answer and the second most common one part ways:

| model | prompt | at token | one answer continues | the other continues |
|---|---|---|---|---|
| Qwen3-8B | LRU cache (code) | 184 | `from typing import Any, Dict, Optional, Tuple, Union` | `... Tuple, List, Union` |
| Qwen3-8B | Microservices failure | 41 | "Root Cause: ... Due to **Inefficient Load Balancing and Resource Starvation**" | "... Due to **Load-Dependent Throttling and Resource Contention**" |
| Qwen3-8B | SQL lineage (JSON) | 56 | `"... as pres_name, filtering by cycl_time_id = ..."` | `"... as pres_name, filtered by cycl_time_id = ..."` |
| Qwen3-8B | Rate limiter | 141 | "### 1. **High-Level Architecture**" | "### 1. **Distributed Architecture** - 100 API servers" |
| Qwen3-30B-A3B | LIS | 220 | "Generate all possible subsequences and check which ones are increasing" | "Use recursion to explore all possible subsequences" |
| Qwen3-30B-A3B | Rate limiter | 92 | `[Client] → [Global Load Balancer] → [...]` | `Client → Load Balancer → API Gateway (Rate Limiter)` |
| Qwen3-30B-A3B | Feynman | 184 | "2. **Feynman Path Integral Formulation**" | "- For this work, Feynman, Schwinger ..." |
| Qwen3-30B-A3B | SQL lineage (JSON) | 136 | "renamed as bp_id) from p1_rem_list_imm..." | "as bp_id) from p1_rem_list_imm_plg" |

Even where the text agrees, the numbers underneath often do not: the 165 copies of Qwen3-8B's most common SQL
answer carried 141 different logprob vectors.

### 4.4 Speed and resources

| | Qwen3-8B default | Qwen3-8B batch-inv. | Qwen3-30B-A3B default | Qwen3-30B-A3B batch-inv. |
|---|---|---|---|---|
| generated tokens | 1,229,396 | 1,229,000 | 644,141 | 647,600 |
| generation wall time | 1372 s | 1207 s | 1379 s | 1397 s |
| output tokens/s | 896 | **1019** | 467 | **464** |
| server start | 229 s | 136 s | 588 s | 431 s |
| planned memory (weights + KV cache + overhead) | 82.4 GB | 82.4 GB | 92.1 GB | 92.1 GB |

Batch-invariant mode was not slower - unlike the 1.6-2.1x slowdown the blog reports for data-center GPUs.
Plausible reasons: vLLM 0.31 ships batch-invariant matmul tiles tuned for this GPU class (confirmed present
by the preflight check), auto-enables CUDA graphs for them, and at these batch sizes the GB10 is limited by
memory bandwidth (273 GB/s) more than by kernel efficiency. One measurement per mode under this mixed load -
not the dedicated throughput benchmark.

## 5. Supporting experiments

**Batch size alone changes the answer** (Qwen3-8B, rate-limiter prompt alone vs. hidden at a random position
among 2-128 other requests, 256 tokens, compared with the stand-alone answer; vLLM's own test method; stopped
after batch 32/64 to free the GPU):

| batch size | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| default vLLM | identical | differs (tok 141, 141) | differs (141, 133) | differs (141, 141) | differs (logprobs only; tok 210) | differs (141, 141) | not reached |
| batch-invariant | identical | identical x2 | identical x2 | identical x2 | identical x2 | identical x2 | identical |

**Mixed load exposes more** (Qwen3-8B, default vLLM, 200 copies of one prompt on its own server vs. the
same prompts in the mixed run):

| prompt | own server: distinct answers | mixed server: distinct answers |
|---|---|---|
| Distributed rate limiter | 8 / 200 | 172 / 200 |
| Thread-safe LRU cache | 7 / 200 | 97 / 200 |
| Longest increasing subsequence | 9 / 200 | 116 / 200 |
| Microservices failure analysis | 12 / 200 | 154 / 200 |

**Kernels are repeatable but not batch-invariant** (op-level test, `qwen3-8b-mine/report.md`): a bf16 matrix
multiplication gave 0 mismatches in 1000 repeats, yet one row's result changed with the number of rows in the
batch (max difference 1641 in the blog's `torch.mm(a[:1], b)` vs `torch.mm(a, b)[:1]` demo; blog: 1669.25).
PyTorch's "deterministic" flags do not change this; batch-invariant kernels (Thinking Machines'
batch_invariant_ops, the repo's Triton bmm/softmax) make every matrix op invariant. PyTorch's fused SDPA decode
attention varies in every mode - which is why vLLM uses its own Triton attention in batch-invariant mode.

**Root cause in miniature:** summing the same 8 numbers in 10,000 different orders gives 102 different results
with the blog's own snippet (Python 3.12), 258 with a plain left-to-right loop.

**Smoke test** (Qwen3-1.7B, 100 x 256 tokens): default 3 distinct answers (first divergence at token 127),
batch-invariant 1.

### 4.5 Deployment check (2026-10-09)

The saved deterministic package (`/home/user1/deterministic-llm/qwen3-8b/`: pinned Qwen3-8B weights, frozen
`vllm/vllm-openai:v0.31.0` image, `serve.sh` with the validated settings) was started on the GB10 and checked
through the everyday **chat-completions API** (chat template applied by the server, thinking off by default):
all 8 prompts x 25 copies sent at once (200 concurrent requests, temperature 0, 300 tokens) - **every prompt
returned one answer, bitwise identical in text and logprobs** (`check_determinism.py`: DETERMINISTIC). Server
start: ~2.5 min; log confirms `VLLM_BATCH_INVARIANT=1` and TRITON_ATTN.

## 6. Discussion

- **Realistic load is the condition to test.** Identical requests alone march through the engine almost in
  lockstep and showed 7-12 variants per 200; mixed with other prompts the same prompts showed 97-172. Testing
  determinism with repeated identical requests understates the problem.
- **Mixture-of-experts models are more sensitive.** Qwen3-30B-A3B diverged on all 8 prompts, often within the
  first 11-38 tokens, and even the haiku varied. A small numeric difference can change which experts a token is
  routed to, after which the computation differs substantially. (The 8B haiku answer is 25 tokens - too short to
  diverge; the 30B one is ~170.)
- **Text equality is not enough.** The SQL prompt's answers looked nearly stable (5-7 distinct texts) while
  their logprobs varied in almost every copy - a downstream system that uses probabilities (ranking, confidence,
  evaluation) would see nondeterminism even then.
- **Determinism was free here.** On the GB10 with these models it cost no throughput, which makes it a
  reasonable default wherever repeatable answers matter.

## 7. Limitations

- One machine, one run per mode; throughput comes from the mixed-load runs, not a dedicated benchmark.
- Determinism holds for the same weights, vLLM build, GPU type and server settings. A different GPU or vLLM
  version can give different (again deterministic) answers; re-validate after upgrades.
- Prefix caching was off in all runs; its interaction with batch-invariant mode was not measured.
- Hybrid linear-attention / Mamba models are not supported by vLLM 0.31's batch-invariant mode.
- The repository's own alternative (the "custom" mode: vendor kernels on fixed-size chunks) is not sufficient
  on the GB10: a row's result can still depend on its position inside a chunk. vLLM's batch-invariant mode and
  Thinking Machines' library do not have this problem.
- The batch-size sweep was stopped at batch size 32 (default) / 64 (batch-invariant).

## 8. Deliverables and reproducibility

- **Data:** each run directory here holds `report.md`, `summary.json`, `status.json`, per-job `job.json`,
  `result.json`, `env.json`, logs, and every completion with its token ids and logprobs
  (`completions.jsonl.zst`, `zstd -d` to unpack). Runs: `qwen3-8b-mine/`, `qwen3-30b-a3b-mine/`,
  `qwen3-8b-one-prompt-per-server-partial/`, `smoke-qwen3-1.7b/`; `code-changes-vs-8e0e74a.diff` records the
  code each run used (identical to commit `2fc6a27`).
- **Reproduce** (see `docs/GB10_RUNBOOK.md`, `docs/RESOURCES.md`):

  ```bash
  bash scripts/gb10/setup.sh --suite smoke --suite qwen3-8b-mine
  bash scripts/gb10/run_suite.sh configs/suites/qwen3-8b-mine.yaml --exclude 'sweep-*'   # ~50 min
  bash scripts/gb10/run_suite.sh configs/suites/qwen3-30b-mine.yaml                      # ~65 min
  ```

- **Use it:** `scripts/gb10/serve_deterministic.sh` starts a deterministic, OpenAI-compatible Qwen3-8B server
  with the validated settings; `scripts/gb10/check_determinism.py` verifies a running server. A self-contained
  copy (pinned weights, frozen vLLM image) is kept on the GB10 at `/home/user1/deterministic-llm/qwen3-8b/`.
- Short version for stakeholders: `docs/FINDINGS_2026-10-08.md`.

## Appendix: job parameters

| | Qwen3-8B | Qwen3-30B-A3B-Instruct-2507 |
|---|---|---|
| suite | `configs/suites/qwen3-8b-mine.yaml` | `configs/suites/qwen3-30b-mine.yaml` |
| requests | 8 prompts x 200 | 8 prompts x 100 |
| max_tokens / temperature / seed | 1000 / 0 / 0 | 1000 / 0 / 0 |
| max_num_seqs | 256 | 160 |
| KV cache | 60 GB | 24 GB |
| attention (default / batch-invariant) | FLASH_ATTN / TRITON_ATTN | FLASH_ATTN / TRITON_ATTN |
| prefix caching | off | off |
