# Experiments

How each experiment maps to the blog, what it measures, and what to expect on a GB10. Parameters
are set per job in `configs/suites/*.yaml` (`params:`) or with `--param key=value` on
`python3 -m detllm run`.

Terminology (from the blog):

* **Run-to-run deterministic** - the same kernel on the same inputs gives the same bits every time.
  Almost every kernel in an LLM forward pass is.
* **Batch-invariant** - a row's result does not depend on how many other rows are in the batch.
  Most kernels are *not*: they choose tiles, split-K / split-KV, or even different instructions by
  shape. Because an inference server's batch size depends on load, a user sees nondeterminism.
* **Identical** here always means bitwise identical: same token ids *and* the same float bits for
  every logprob.

---

## `preflight` - is this machine ready? (about 1 min)

Checks CUDA and GB10 detection (compute capability 12.1, SM count, shared memory per block);
compiles a Triton kernel for sm_121; enables batch_invariant_ops and probes bf16/fp32/fp16
matmuls (fp16 is expected to fail on GB10, see below); checks this repo's Triton `bmm`/`softmax`
against fp64 references and that `torch.matmul`/`torch.softmax` really dispatch to them; checks the
CUSTOM mode; and reports which batch-invariance features the installed vLLM has (including
whether it has tuned batch-invariant matmul tiles for this GPU). Also reports unified-memory and
page-cache numbers.

## `fp_nonassoc` - "The original sin: floating-point non-associativity"

The blog's two demos: `(0.1 + 1e20) - 1e20 = 0` vs `0.1 + (1e20 - 1e20) = 0.1`, and summing
`[1e-10, 1e-5, 1e-2, 1]` and their negatives in 10,000 random orders, which gives **102 distinct
results**. The snippet is run as written: Python 3.12's built-in `sum()` (the containers' Python)
compensates float additions, but the list holds the *int* `1`, which is added uncompensated - and
that mix gives exactly the blog's 102 (with `1.0` it would give 1). A plain left-to-right loop over
the same orders (what `sum()` did up to 3.11) is reported next to it and gives 258: a different
reduction algorithm, a different answer.

## `op_invariance` - kernels are deterministic, but not batch-invariant

For each op, row 0 is computed at batch size 1 and then inside batches of 2..2048 rows
(attention: 2..256 sequences). The op is **batch-invariant** if row 0 is bitwise identical at every
size. Each op is also repeated at the largest size to check **run-to-run determinism**.

| op | why it matters |
|---|---|
| `mm` (bf16, fp32, fp16) | every linear layer; fp16 also exposes the shared-memory limit |
| `linear_bias` | `F.linear` with bias → `aten::addmm` (Qwen2.5 q/k/v projections) |
| `rmsnorm` | HF's RMSNorm: fp32 `mean` over the hidden dimension |
| `log_softmax_vocab` | sampler logprobs over 151,936 tokens |
| `softmax` | attention probabilities |
| `bmm_attention_scores` | `q @ k^T` as `aten::bmm` with batch = sequences x heads |
| `attention_eager_decode/prefill` | HF eager attention end to end (bmm → softmax → bmm) |
| `attention_sdpa_decode/prefill` | PyTorch SDPA (flash / efficient / cuDNN backends) - never overridden |

It also runs the blog's two snippets verbatim: `torch.mm(a[:1], b)` vs `torch.mm(a, b)[:1]` with
`linspace` inputs (the blog printed `1669.25`) and 1000 repetitions of a bf16 2048x2048 matmul
(the blog asserts 0 differences).

Modes, one process each:

| mode | expected |
|---|---|
| `baseline` | run-to-run deterministic everywhere; matmul-like ops vary with batch size |
| `torch_flags` | same as baseline. The classic determinism flags do **not** give batch invariance (the blog's point) |
| `cublaslt_ws1` | vLLM's GEMM strategy on SM90+ and GB10 (cuBLASLt, no split-K). This tests, on your hardware, whether that alone makes GEMMs batch-invariant |
| `official` | mm / linear / RMSNorm / log_softmax invariant; bmm / softmax / attention not (the library does not cover them); fp16 mm → `OutOfResources` on GB10 |
| `official_full` | all matmul, normalisation and eager-attention ops invariant |
| `custom` | same, via fixed-size chunks through cuBLAS |

Notes: batch_invariant_ops runs fp32 matmuls in TF32 (its open issue #23), so its fp32 results are
batch-invariant but less precise than stock fp32. This repo's own `bmm` forces IEEE fp32 because
TF32 corrupts the rotary-embedding phase beyond position 2048.

## `hf_invariance` - a whole model, in-process (HF transformers)

The prompt ("Tell me about Richard Feynman", chat template, non-thinking) is run alone and inside
batches of 2..32 with filler rows of exactly the same length (so no padding and identical
positions), as the first and as the last row:

* **forward** - the target's logits at every prompt position, compared bitwise with batch size 1;
* **decode** - 128 greedy tokens (manual KV-cache loop), tokens and logprobs compared bitwise.

| mode | attention | expected |
|---|---|---|
| `baseline` | SDPA | logits differ with batch size; decoding may diverge |
| `baseline_eager` | eager | same, isolates the attention backend |
| `official` | eager | linear layers + RMSNorm fixed, attention (bmm/softmax) not, so differences can remain |
| `official_full` | eager | bitwise invariant |
| `custom` | eager | bitwise invariant (slower) |

The reference (batch size 1) is also run twice to confirm run-to-run determinism.

## `vllm_completions` - "How nondeterministic are completions?"

The blog sent "Tell me about Richard Feynman" (non-thinking) 1000 times at temperature 0 with
1000 output tokens to Qwen3-235B-A22B-Instruct-2507: **80 unique completions**, the most common
78 times, all identical for 102 tokens and diverging at token 103 ("Queens, New York" vs
"New York City"). With batch-invariant kernels, **all 1000 identical**.

Here the same request goes to `vllm serve` as N concurrent HTTP requests. That is what a user of a
server sees, and it is how batch_invariant_ops' `deterministic_vllm_inference.py` tests it. Prompts
are sent as token ids, so both modes receive the same input tokens. Reported: number of unique
completions (by token ids), count of the most common one, the first token position where
completions diverge (1-based, comparable to the blog's "103rd token") with decoded text around it,
and **logprob variants among the most common completion**. Even when the tokens agree, baseline
logprobs usually differ in the last bits; in batch-invariant mode there is exactly 1 variant.

Parameters: `n`, `max_tokens`, `prompt`, `engine: server|offline`, `api: completions|chat`,
`concurrency`, `arrival_jitter_s`, `max_num_seqs`, `kv_cache_gb`, `enable_prefix_caching`,
`attention_backend` / `bi_attention_backend`, `enforce_eager`, `ignore_eos`.

`engine: offline` uses `vllm.LLM` and mixes the targets with filler requests of random lengths.
Identical prompts submitted all at once otherwise move through the engine in lockstep, see
identical batches, and hide the effect.

Suite variants:

* `tml-poc-*` - batch_invariant_ops' own example: chat API, "Generate 1000 random numbers ...
  /no_think", 100 tokens, 1000 requests. Their README reports 18 unique → 1 unique (Qwen3-8B).
* `ablate-prefix-cache-*` - prefix caching on (identical prompts then hit the cache, a different
  code path the blog says attention must handle invariantly).
* `ablate-triton-attn-*` - baseline with the Triton attention backend: changing the backend alone
  does not make the engine batch-invariant.
* `ablate-flex-attn-*` - batch-invariant mode with FlexAttention, the backend of the blog's
  original implementation.

## `vllm_batch_sweep` - engine-level batch invariance (vLLM's own test method)

A "needle" request is generated alone (twice: run-to-run check), then placed at a random position
in batches of 2..128 requests. The other requests are real prompts of mixed lengths (some long
enough for chunked prefill) with random output lengths, so the batch around the needle keeps
changing while it decodes. The needle's tokens and per-token logprobs are compared bitwise with
the reference. This mirrors `tests/v1/determinism/test_batch_invariance.py` in vLLM.

Expected: baseline differs at some batch sizes (often only in logprobs at first); batch-invariant
mode is identical at every size.

## `vllm_perf` - "Performance"

Blog: an API server on one GPU running Qwen3-8B, 1000 requests with 90-110 output tokens:
vLLM default 26 s, unoptimized deterministic vLLM 55 s, with an improved attention kernel 42 s.
Here: the same load (`ignore_eos`, output lengths drawn uniformly from 90..110, real prompts), timed
after a warm-up. Reports wall time, output tokens/s, p50/p90/p99 latency, and the slowdown
relative to baseline. These jobs are `exclusive`: they run alone after everything else, because
concurrent jobs would distort timing. GB10 has far less memory bandwidth than the data-center GPU
in the blog (273 GB/s vs 3.35 TB/s on H100), so compare the *ratio*, not the seconds.

---

## Adding your own runs

```yaml
# configs/suites/mine.yaml
name: mine
memory_budget_gb: 100
max_parallel: 6
jobs:
  - experiment: vllm_completions
    id: sql-{model}-{mode}
    matrix: {model: [qwen2.5-1.5b-instruct, qwen3-1.7b], mode: [baseline, batch_invariant]}
    params: {n: 500, max_tokens: 400, prompt: {file: configs/prompts/my_prompts.txt, index: 7}}
```

```bash
bash scripts/gb10/run_suite.sh configs/suites/mine.yaml
```

New models go into `configs/models.yaml` (`hf_id`, `weights_gb`, `kv_cache_gb`, `overhead_gb`,
`hf_mem_gb`, optional `chat_template_kwargs`).
