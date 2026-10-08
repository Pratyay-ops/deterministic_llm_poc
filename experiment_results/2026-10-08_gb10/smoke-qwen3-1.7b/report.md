# Determinism report - suite `smoke`

Run directory `/workspace/detllm/results/smoke-20261008-120458` - started 2026-10-08T12:04:58+0000, report written 2026-10-08T12:08:26+0000 - 13/13 jobs ok.

Reproduction of Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025) on NVIDIA GB10.

## Headline

- **qwen3-1.7b**: baseline vLLM produced **3 unique** completions out of 100 (first divergence at token 127); batch-invariant vLLM produced **1 unique** (all identical, logprobs bitwise identical).
- needle sweep qwen3-1.7b / baseline: 1/3 batched runs bitwise identical to batch size 1 (differences at batch sizes [8, 32])
- needle sweep qwen3-1.7b / batch_invariant: 3/3 batched runs bitwise identical to batch size 1

Blog (Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens): default vLLM 80 unique completions, first divergence at token 103; batch-invariant kernels: 1 unique.

## Environment

|  |  |
|---|---|
| GPU | NVIDIA GB10 (cc 12.1, 48 SMs) |
| Shared memory / block (opt-in) | 101376 |
| Unified memory (cudaMemGetInfo total) | 130.59 GB |
| Driver | 595.91.07 |
| CUDA (torch build) | 13.0 |
| Container base image | vllm/vllm-openai:v0.31.0 |
| torch / triton | 2.13.0+cu130 / 3.7.1 |
| vLLM / transformers | 0.31.0 / 5.17.0 |
| batch_invariant_ops commit | f22b1fbe534b1fe045080f44a970a441caaf3c4f |
| repo commit | 8e0e74a7585c4e1b6637ba1ca23b0dcffa56cdba |

## Preflight

| check | ok | detail |
|---|---|---|
| cuda | yes | {'torch_cuda': '13.0', 'cuda_available': True, 'name': 'NVIDIA GB10', 'capability': '12.1', 'sm_count': 48, 'total_memory_gb': 130.59, 'mem_get_info_free_gb': 108.97, 'mem_get_info_total_gb': 130.59, 'shared_memory_per_b |
| triton_compile_sm | yes | {'triton': '3.7.1'} |
| batch_invariant_ops | yes | {'enabled': True, 'module_path': '/workspace/detllm/batch_invariant_ops/batch_invariant_ops/__init__.py', 'version': '0.1.0', 'submodule_commit': 'f22b1fbe534b1fe045080f44a970a441caaf3c4f', 'device_smem_optin_bytes': 101 |
| extra_triton_ops | yes | {'bmm_torch.bfloat16': 0.0038072405967843866, 'bmm_torch.float32': 3.304062081959647e-07, 'fp32_ieee_exact': True, 'softmax_max_abs_err': 1.0554311044921882e-09, 'dispatch': {'matmul_routed': True, 'softmax_routed': True |
| custom_fixed_chunk_mode | yes | {'max_abs_err_vs_stock': 0.0, 'row0_bs1_vs_bs130_equal': True, 'chunked_calls': {'aten.mm.default': 2}, 'unchunked_calls': {}} |
| vllm | yes | {'version': '0.31.0', 'kv_cache_memory_bytes': True, 'attention_backend_arg': True, 'attention_config_arg': True, 'batch_invariant_module': 'vllm.model_executor.determinism.batch_invariant', 'tuned_matmul_family': 'sm120 |
| memory | info | {'MemTotal_GB': 130.6, 'MemAvailable_GB': 108.2, 'PageCache_GB': 5.4, 'cudaMemGetInfo_free_GB': 103.4, 'cudaMemGetInfo_total_GB': 130.6} |

## Floating-point non-associativity (CPU)

- `(0.1 + 1e20) - 1e20` = 0.0; `0.1 + (1e20 - 1e20)` = 0.1
- Summing the same 8 numbers in 10,000 random orders gives **258** distinct results (blog: 102).

## Op level: run-to-run vs batch invariance

Row 0 of each op computed at batch size 1 vs. inside larger batches; *invariant* = bitwise identical at every batch size.

| op | baseline | cublaslt_ws1 | official_full | custom |
|---|---|---|---|---|
| mm (bfloat16) | varies (B>=2, max 0.0039) | invariant | invariant | invariant |
| mm (float32) | varies (B>=2, max 4.7e-06) | varies (B>=2, max 9.5e-06) | invariant | invariant |
| mm (float16) | varies (B>=2, max 0.002) | invariant | invariant | invariant |
| linear_bias (bfloat16) | varies (B>=2, max 7.6e-06) | varies (B>=2, max 0.0039) | invariant | varies (B>=2, max 7.6e-06) |
| rmsnorm (bfloat16) | invariant | invariant | invariant | invariant |
| log_softmax_vocab (float32) | invariant | invariant | invariant | invariant |
| softmax (float32) | invariant | invariant | invariant | invariant |
| softmax (bfloat16) | invariant | invariant | invariant | invariant |
| bmm_attention_scores (bfloat16) | invariant | error: RuntimeError | invariant | invariant |
| attention_eager_decode (bfloat16) | invariant | error: RuntimeError | invariant | invariant |
| attention_eager_prefill (bfloat16) | invariant | invariant | invariant | invariant |
| attention_sdpa_decode (bfloat16) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) |
| attention_sdpa_prefill (bfloat16) | invariant | invariant | invariant | invariant |

Blog demos - `torch.mm(a[:1], b)` vs `torch.mm(a, b)[:1]` (blog: 1669.25 on their GPU) and 1000 repeated bf16 matmuls (blog: 0 mismatches):

| mode | mm demo max |diff| | run-to-run mismatches /1000 | invariant ops |
|---|---|---|---|
| baseline | 1641 | 0 | 8/13 |
| cublaslt_ws1 | 606.6 | 0 | 8/11 |
| official_full | 0 | 0 | 12/13 |
| custom | 0 | 0 | 11/13 |

## HF transformers model: forward logits and greedy decoding vs batch size

| model | mode | attention | logits invariant | max |dlogit| | decode bitwise invariant | first diverging token | run-to-run identical |
|---|---|---|---|---|---|---|---|
| qwen3-1.7b | baseline | sdpa | **NO** | 0.75 | **NO** | - | yes |
| qwen3-1.7b | custom | eager | **NO** | 0.8301 | **NO** | - | yes |
| qwen3-1.7b | official_full | eager | yes | 0 | yes | - | yes |

## vLLM: N temperature-0 completions of one prompt

| job | model | mode | engine/api | n x max_tokens | unique | modal count | first divergence (token #) | logprob variants of modal | attention backend |
|---|---|---|---|---|---|---|---|---|---|
| completions-qwen3-1.7b-baseline | qwen3-1.7b | baseline | server/completions | 100x256 | 3 | 68 | 127 | 6 | FLASH_ATTN |
| completions-qwen3-1.7b-batch_invariant | qwen3-1.7b | batch_invariant | server/completions | 100x256 | 1 | 100 | - | 1 | TRITON_ATTN |

Blog: Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens -> 80 unique (modal x78), first divergence at token 103 ("Queens, New York" vs "New York City"); batch-invariant: 1 unique. batch_invariant_ops' own vLLM example (Qwen3-8B, 100 tokens): 18 unique -> 1 unique.

Where completions first diverge:

- `completions-qwen3-1.7b-baseline` token 127: ...' of California, Berkeley, and later at the University of Chicago, where he was a leading figure in the development of the' -> ' theory of quantum electrodynamics (QED). His work' vs ' modern understanding of quantum electrodynamics (QED). His'

## vLLM: needle request alone vs inside batches

| model | mode | bitwise identical | token mismatches | logprob-only mismatches | batch sizes that differ | bs=1 run-to-run |
|---|---|---|---|---|---|---|
| qwen3-1.7b | baseline | 1/3 | 0 | 2 | 8, 32 | yes |
| qwen3-1.7b | batch_invariant | 3/3 | 0 | 0 | - | yes |
