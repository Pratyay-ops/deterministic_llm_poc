# Determinism report - suite `qwen3-8b`

Run directory `/workspace/detllm/results/qwen3-8b-20261008-123024` - started 2026-10-08T12:30:24+0000, report written 2026-10-08T13:03:39+0000 - 12/12 jobs ok.

Reproduction of Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025) on NVIDIA GB10.

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
| cuda | yes | {'torch_cuda': '13.0', 'cuda_available': True, 'name': 'NVIDIA GB10', 'capability': '12.1', 'sm_count': 48, 'total_memory_gb': 130.59, 'mem_get_info_free_gb': 111.35, 'mem_get_info_total_gb': 130.59, 'shared_memory_per_b |
| triton_compile_sm | yes | {'triton': '3.7.1'} |
| batch_invariant_ops | yes | {'enabled': True, 'module_path': '/workspace/detllm/batch_invariant_ops/batch_invariant_ops/__init__.py', 'version': '0.1.0', 'submodule_commit': 'f22b1fbe534b1fe045080f44a970a441caaf3c4f', 'device_smem_optin_bytes': 101 |
| extra_triton_ops | yes | {'bmm_torch.bfloat16': 0.0038072405967843866, 'bmm_torch.float32': 3.304062081959647e-07, 'fp32_ieee_exact': True, 'softmax_max_abs_err': 1.0554311044921882e-09, 'dispatch': {'matmul_routed': True, 'softmax_routed': True |
| custom_fixed_chunk_mode | yes | {'max_abs_err_vs_stock': 0.0, 'row0_bs1_vs_bs130_equal': True, 'chunked_calls': {'aten.mm.default': 2}, 'unchunked_calls': {}, 'decomposed_calls': {}} |
| vllm | yes | {'version': '0.31.0', 'kv_cache_memory_bytes': True, 'attention_backend_arg': True, 'attention_config_arg': True, 'batch_invariant_module': 'vllm.model_executor.determinism.batch_invariant', 'tuned_matmul_family': 'sm120 |
| memory | info | {'MemTotal_GB': 130.6, 'MemAvailable_GB': 111.7, 'PageCache_GB': 4.1, 'cudaMemGetInfo_free_GB': 107.9, 'cudaMemGetInfo_total_GB': 130.6} |

## Floating-point non-associativity (CPU)

- `(0.1 + 1e20) - 1e20` = 0.0; `0.1 + (1e20 - 1e20)` = 0.1
- Summing the same 8 numbers in 10,000 random orders gives **102** distinct results with the blog's snippet (blog: 102); a plain left-to-right loop over the same orders gives 258.

## Op level: run-to-run vs batch invariance

Row 0 of each op computed at batch size 1 vs. inside larger batches; *invariant* = bitwise identical at every batch size.

| op | baseline | torch_flags | cublaslt_ws1 | official | official_full | custom |
|---|---|---|---|---|---|---|
| mm (bfloat16) | varies (B>=2, max 0.0078) | varies (B>=2, max 0.0078) | varies (B>=32, max 0.0078) | invariant | invariant | invariant |
| mm (float32) | varies (B>=2, max 5.7e-06) | varies (B>=2, max 5.7e-06) | varies (B>=2, max 1e-05) | invariant | invariant | invariant |
| mm (float16) | varies (B>=2, max 0.002) | varies (B>=2, max 0.002) | varies (B>=32, max 0.002) | invariant | invariant | invariant |
| linear_bias (bfloat16) | varies (B>=2, max 0.031) | varies (B>=2, max 0.031) | varies (B>=2, max 0.0039) | invariant | invariant | invariant |
| rmsnorm (bfloat16) | invariant | invariant | invariant | invariant | invariant | invariant |
| log_softmax_vocab (float32) | invariant | invariant | invariant | invariant | invariant | invariant |
| softmax (float32) | invariant | invariant | invariant | invariant | invariant | invariant |
| softmax (bfloat16) | invariant | invariant | invariant | invariant | invariant | invariant |
| bmm_attention_scores (bfloat16) | invariant | invariant | error: RuntimeError | invariant | invariant | invariant |
| attention_eager_decode (bfloat16) | invariant | invariant | error: RuntimeError | invariant | invariant | invariant |
| attention_eager_prefill (bfloat16) | invariant | invariant | invariant | invariant | invariant | invariant |
| attention_sdpa_decode (bfloat16) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) | varies (B>=2, max 0.00098) |
| attention_sdpa_prefill (bfloat16) | invariant | invariant | invariant | invariant | invariant | invariant |

Blog demos - `torch.mm(a[:1], b)` vs `torch.mm(a, b)[:1]` (blog: 1669.25 on their GPU) and 1000 repeated bf16 matmuls (blog: 0 mismatches):

| mode | mm demo max |diff| | run-to-run mismatches /1000 | invariant ops |
|---|---|---|---|
| baseline | 1641 | 0 | 8/13 |
| torch_flags | 1641 | 0 | 8/13 |
| cublaslt_ws1 | 606.6 | 0 | 6/11 |
| official | 0 | 0 | 12/13 |
| official_full | 0 | 0 | 12/13 |
| custom | 0 | 0 | 12/13 |

## vLLM: N temperature-0 completions of one prompt

| job | model | mode | engine/api | n x max_tokens | unique | modal count | first divergence (token #) | logprob variants of modal | attention backend |
|---|---|---|---|---|---|---|---|---|---|
| mine-p0-rate-limiter-qwen3-8b-baseline | qwen3-8b | baseline | server/completions | 200x1000 | 8 | 41 | 133 | 1 | FLASH_ATTN |
| mine-p1-lru-cache-qwen3-8b-baseline | qwen3-8b | baseline | server/completions | 200x1000 | 7 | 33 | 184 | 1 | FLASH_ATTN |
| mine-p2-lis-qwen3-8b-baseline | qwen3-8b | baseline | server/completions | 200x1000 | 9 | 27 | 50 | 2 | FLASH_ATTN |
| mine-p3-microservices-qwen3-8b-baseline | qwen3-8b | baseline | server/completions | 200x1000 | 12 | 23 | 41 | 1 | FLASH_ATTN |

Blog: Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens -> 80 unique (modal x78), first divergence at token 103 ("Queens, New York" vs "New York City"); batch-invariant: 1 unique. batch_invariant_ops' own vLLM example (Qwen3-8B, 100 tokens): 18 unique -> 1 unique.

Where completions first diverge:

- `mine-p0-rate-limiter-qwen3-8b-baseline` token 198: ...'** (for storing rate limit counters)\n- **Message queue** (for coordination and event propagation)\n- **Monitoring and' -> ' analytics** (for real-time metrics)\n\n### 2.' vs ' logging** (for audit and debugging)\n\n### 2.'
- `mine-p1-lru-cache-qwen3-8b-baseline` token 184: ...'\n\n```python\nimport threading\nimport time\nimport heapq\nfrom typing import Any, Dict, Optional, Tuple,' -> ' List, Union\nfrom collections import OrderedDict\n\nclass LRUC' vs ' Union\nfrom collections import OrderedDict\n\nclass LRUCache:\n'
- `mine-p2-lis-qwen3-8b-baseline` token 176: ...'Space Complexity**: $ O(n) $\n  - Recursion stack depth.\n\n### **Recurrence Relation**\n- Let `' -> 'f(i)` be the length of the LIS ending at index' vs 'LIS(i)` be the length of the longest increasing sub'
- `mine-p3-microservices-qwen3-8b-baseline` token 85: ...' failures** under load due to a **load-dependent connection pool exhaustion** in the **API Gateway**, which leads to **' -> 'HTTP 500 errors**, **latency spikes**,' vs 'unrecoverable state** in the **backend services** and'
