# Determinism report - suite `qwen3-8b-mine`

Run directory `/workspace/detllm/results/qwen3-8b-mine-20261008-130408` - started 2026-10-08T13:25:01+0000, report written 2026-10-08T14:14:52+0000 - 10/12 jobs ok.

Reproduction of Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025) on NVIDIA GB10.

## Headline

- **qwen3-8b, per prompt**: default vLLM was nondeterministic (tokens or logprobs) for **7/8** prompts; batch-invariant vLLM was bitwise identical for **8/8** prompts (details below).

Blog (Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens): default vLLM 80 unique completions, first divergence at token 103; batch-invariant kernels: 1 unique.

## Per-prompt comparison on qwen3-8b: default vs batch-invariant vLLM

Each prompt was sent N times at temperature 0 as concurrent requests to a `vllm serve` instance, first with default kernels, then with `VLLM_BATCH_INVARIANT=1`. *unique* counts distinct token sequences; *first divergence* is the earliest token (1-based) where any two completions differ; *logprob variants* counts distinct bitwise logprob vectors among the copies of the most common completion. **deterministic** = batch-invariant run gave 1 unique completion with bitwise identical logprobs and no failed requests.

| # | prompt | default: ok | default: unique | default: first divergence | default: logprob variants | batch-inv: ok | batch-inv: unique | batch-inv: first divergence | batch-inv: logprob variants | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Design a distributed rate limiter for an API handling 1 mill... | 200/200 | 172 (modal x4) | 29 | 1 of 4 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |
| 2 | Implement a thread-safe LRU cache in Python that supports O(... | 200/200 | 97 (modal x14) | 62 | 10 of 14 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |
| 3 | Analyze this complex algorithmic problem in detail: Given an... | 200/200 | 116 (modal x13) | 50 | 9 of 13 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |
| 4 | Debug and completely analyze this distributed system failure... | 200/200 | 154 (modal x9) | 41 | 9 of 9 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |
| 5 | Compare three database paradigms for a social media platform... | 200/200 | 126 (modal x9) | 95 | 7 of 9 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |
| 6 | you are a creative poet and you are tasked with Writting a h... | 200/200 | 1 (modal x200) | - | 1 of 200 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |
| 7 | Tell me about Richard Feynman | 200/200 | 143 (modal x6) | 59 | 5 of 6 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |
| 8 | You are SQL code parser agent that extracts all the source t... | 200/200 | 5 (modal x165) | 45 | 141 of 165 | 200/200 | 1 (modal x200) | - | 1 of 200 | **deterministic** |

Examples of where default-vLLM completions split (the most common completion vs the second most common):

- prompt 1, token 141: ...'- **Pseudocode for the core algorithm**\n\n---\n\n## 🧠 **Architecture Overview**\n\n### 1. **' -> 'High-Level Architecture**\n\nThe system is composed of the following components' vs 'Distributed Architecture**\n- **100 API servers**'
- prompt 2, token 184: ...'\n\n```python\nimport threading\nimport time\nimport heapq\nfrom typing import Any, Dict, Optional, Tuple,' -> ' Union\nfrom collections import OrderedDict\n\nclass LRUCache:\n' vs ' List, Union\nfrom collections import OrderedDict\n\nclass LRUC'
- prompt 3, token 386: ...' inputs.\n\n---\n\n## 2. **Dynamic Programming Solution**\n\n### **Approach**\nWe use a DP array `dp' -> '` where `dp[i]` represents the length of the' vs '[i]` where `dp[i]` represents the length'
- prompt 4, token 41: ...'**\n\n---\n\n## **1. Root Cause Analysis**\n\n### **Root Cause: Connection Pool Exhaustion in API Gateway Due to' -> ' Inefficient Load Balancing and Resource Starvation**\n\n**Failure' vs ' Load-Dependent Throttling and Resource Contention**\n\n'
- prompt 5, token 591: ...' COMMITTED` and `REPEATABLE READ`.\n\n### **Data Modeling Complexity**\n- **Normalized Schema**: Requires careful design' -> ' to avoid redundancy and ensure referential integrity.\n- **Schema' vs ' with many tables and relationships.\n- **Schema Evolution**: Ch'
- prompt 7, token 567: ...' through:\n- His groundbreaking work in physics.\n- His influential books and lectures.\n- His role as a mentor to many' -> ' young scientists.\n- His enduring influence on how science is taught' vs ' future scientists.\n- His enduring influence on how science is taught'
- prompt 8, token 56: ...'_imm",\n      "transformation_description": "Select distinct records with concatenation of first and last name as pres_name,' -> ' filtering by cycl_time_id = 20241' vs ' filtered by cycl_time_id = 20241'

## Environment

|  |  |
|---|---|
| GPU | NVIDIA GB10 (cc 12.1, 48 SMs) |
| Shared memory / block (opt-in) | 101376 |
| Unified memory (cudaMemGetInfo total) | 130.59 GB |
| Driver | 595.91.07 |
| CUDA (torch build) | 13.2 |
| Container base image | vllm/vllm-openai:v0.31.0 |
| torch / triton | 2.13.0+cu130 / 3.7.1 |
| vLLM / transformers | 0.31.0 / 5.17.0 |
| batch_invariant_ops commit | f22b1fbe534b1fe045080f44a970a441caaf3c4f |
| repo commit | 8e0e74a7585c4e1b6637ba1ca23b0dcffa56cdba |

## Preflight

| check | ok | detail |
|---|---|---|
| cuda | yes | {'torch_cuda': '13.0', 'cuda_available': True, 'name': 'NVIDIA GB10', 'capability': '12.1', 'sm_count': 48, 'total_memory_gb': 130.59, 'mem_get_info_free_gb': 114.48, 'mem_get_info_total_gb': 130.59, 'shared_memory_per_b |
| triton_compile_sm | yes | {'triton': '3.7.1'} |
| batch_invariant_ops | yes | {'enabled': True, 'module_path': '/workspace/detllm/batch_invariant_ops/batch_invariant_ops/__init__.py', 'version': '0.1.0', 'submodule_commit': 'f22b1fbe534b1fe045080f44a970a441caaf3c4f', 'device_smem_optin_bytes': 101 |
| extra_triton_ops | yes | {'bmm_torch.bfloat16': 0.0038072405967843866, 'bmm_torch.float32': 3.304062081959647e-07, 'fp32_ieee_exact': True, 'softmax_max_abs_err': 1.0554311044921882e-09, 'dispatch': {'matmul_routed': True, 'softmax_routed': True |
| custom_fixed_chunk_mode | yes | {'max_abs_err_vs_stock': 0.0, 'row0_bs1_vs_bs130_equal': True, 'chunked_calls': {'aten.mm.default': 2}, 'unchunked_calls': {}, 'decomposed_calls': {}} |
| vllm | yes | {'version': '0.31.0', 'kv_cache_memory_bytes': True, 'attention_backend_arg': True, 'attention_config_arg': True, 'batch_invariant_module': 'vllm.model_executor.determinism.batch_invariant', 'tuned_matmul_family': 'sm120', 'tuned_matmul_table_for_this_gpu': True} |
| memory | info | {'MemTotal_GB': 130.6, 'MemAvailable_GB': 114.5, 'PageCache_GB': 3.2, 'cudaMemGetInfo_free_GB': 112.0, 'cudaMemGetInfo_total_GB': 130.6} |

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

| job | model | mode | engine/api | ok/n x max_tokens | unique | modal count | first divergence (token #) | logprob variants of modal | attention backend |
|---|---|---|---|---|---|---|---|---|---|
| mine-all-qwen3-8b-baseline #1 | qwen3-8b | baseline | server/completions | 200/200x1000 | 172 | 4 | 29 | 1 of 4 | FLASH_ATTN |
| mine-all-qwen3-8b-baseline #2 | qwen3-8b | baseline | server/completions | 200/200x1000 | 97 | 14 | 62 | 10 of 14 | FLASH_ATTN |
| mine-all-qwen3-8b-baseline #3 | qwen3-8b | baseline | server/completions | 200/200x1000 | 116 | 13 | 50 | 9 of 13 | FLASH_ATTN |
| mine-all-qwen3-8b-baseline #4 | qwen3-8b | baseline | server/completions | 200/200x1000 | 154 | 9 | 41 | 9 of 9 | FLASH_ATTN |
| mine-all-qwen3-8b-baseline #5 | qwen3-8b | baseline | server/completions | 200/200x1000 | 126 | 9 | 95 | 7 of 9 | FLASH_ATTN |
| mine-all-qwen3-8b-baseline #6 | qwen3-8b | baseline | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | FLASH_ATTN |
| mine-all-qwen3-8b-baseline #7 | qwen3-8b | baseline | server/completions | 200/200x1000 | 143 | 6 | 59 | 5 of 6 | FLASH_ATTN |
| mine-all-qwen3-8b-baseline #8 | qwen3-8b | baseline | server/completions | 200/200x1000 | 5 | 165 | 45 | 141 of 165 | FLASH_ATTN |
| mine-all-qwen3-8b-batch_invariant #1 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |
| mine-all-qwen3-8b-batch_invariant #2 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |
| mine-all-qwen3-8b-batch_invariant #3 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |
| mine-all-qwen3-8b-batch_invariant #4 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |
| mine-all-qwen3-8b-batch_invariant #5 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |
| mine-all-qwen3-8b-batch_invariant #6 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |
| mine-all-qwen3-8b-batch_invariant #7 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |
| mine-all-qwen3-8b-batch_invariant #8 | qwen3-8b | batch_invariant | server/completions | 200/200x1000 | 1 | 200 | - | 1 of 200 | TRITON_ATTN |

Blog: Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens -> 80 unique (modal x78), first divergence at token 103 ("Queens, New York" vs "New York City"); batch-invariant: 1 unique. batch_invariant_ops' own vLLM example (Qwen3-8B, 100 tokens): 18 unique -> 1 unique.

Where completions first diverge:

- `mine-all-qwen3-8b-baseline #1` token 141: ...'- **Pseudocode for the core algorithm**\n\n---\n\n## 🧠 **Architecture Overview**\n\n### 1. **' -> 'High-Level Architecture**\n\nThe system is composed of the following components' vs 'Distributed Architecture**\n- **100 API servers**'
- `mine-all-qwen3-8b-baseline #2` token 184: ...'\n\n```python\nimport threading\nimport time\nimport heapq\nfrom typing import Any, Dict, Optional, Tuple,' -> ' Union\nfrom collections import OrderedDict\n\nclass LRUCache:\n' vs ' List, Union\nfrom collections import OrderedDict\n\nclass LRUC'
- `mine-all-qwen3-8b-baseline #3` token 386: ...' inputs.\n\n---\n\n## 2. **Dynamic Programming Solution**\n\n### **Approach**\nWe use a DP array `dp' -> '` where `dp[i]` represents the length of the' vs '[i]` where `dp[i]` represents the length'
- `mine-all-qwen3-8b-baseline #4` token 41: ...'**\n\n---\n\n## **1. Root Cause Analysis**\n\n### **Root Cause: Connection Pool Exhaustion in API Gateway Due to' -> ' Inefficient Load Balancing and Resource Starvation**\n\n**Failure' vs ' Load-Dependent Throttling and Resource Contention**\n\n'
- `mine-all-qwen3-8b-baseline #5` token 591: ...' COMMITTED` and `REPEATABLE READ`.\n\n### **Data Modeling Complexity**\n- **Normalized Schema**: Requires careful design' -> ' to avoid redundancy and ensure referential integrity.\n- **Schema' vs ' with many tables and relationships.\n- **Schema Evolution**: Ch'
- `mine-all-qwen3-8b-baseline #7` token 567: ...' through:\n- His groundbreaking work in physics.\n- His influential books and lectures.\n- His role as a mentor to many' -> ' young scientists.\n- His enduring influence on how science is taught' vs ' future scientists.\n- His enduring influence on how science is taught'
- `mine-all-qwen3-8b-baseline #8` token 56: ...'_imm",\n      "transformation_description": "Select distinct records with concatenation of first and last name as pres_name,' -> ' filtering by cycl_time_id = 20241' vs ' filtered by cycl_time_id = 20241'

## Jobs that did not finish

| job | status | error | log |
|---|---|---|---|
| sweep-qwen3-8b-baseline | cancelled | received signal 15 | sweep-qwen3-8b-baseline/job.log |
| sweep-qwen3-8b-batch_invariant | cancelled | received signal 15 | sweep-qwen3-8b-batch_invariant/job.log |
