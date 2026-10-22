# Determinism report - suite `qwen3-30b-mine`

Run directory `/workspace/detllm/results/qwen3-30b-mine-20261008-141458` - started 2026-10-08T14:14:58+0000, report written 2026-10-08T15:18:45+0000 - 2/2 jobs ok.

Reproduction of Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025) on NVIDIA GB10.

## Headline

- **qwen3-30b-a3b-instruct-2507, per prompt**: default vLLM was nondeterministic (tokens or logprobs) for **8/8** prompts; batch-invariant vLLM was bitwise identical for **8/8** prompts (details below).

Blog (Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens): default vLLM 80 unique completions, first divergence at token 103; batch-invariant kernels: 1 unique.

## Per-prompt comparison on qwen3-30b-a3b-instruct-2507: default vs batch-invariant vLLM

Each prompt was sent N times at temperature 0 as concurrent requests to a `vllm serve` instance, first with default kernels, then with `VLLM_BATCH_INVARIANT=1`. *unique* counts distinct token sequences; *first divergence* is the earliest token (1-based) where any two completions differ; *logprob variants* counts distinct bitwise logprob vectors among the copies of the most common completion. **deterministic** = batch-invariant run gave 1 unique completion with bitwise identical logprobs and no failed requests.

| # | prompt | default: ok | default: unique | default: first divergence | default: logprob variants | batch-inv: ok | batch-inv: unique | batch-inv: first divergence | batch-inv: logprob variants | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Design a distributed rate limiter for an API handling 1 mill... | 100/100 | 80 (modal x4) | 33 | 1 of 4 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |
| 2 | Implement a thread-safe LRU cache in Python that supports O(... | 100/100 | 88 (modal x3) | 17 | 1 of 3 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |
| 3 | Analyze this complex algorithmic problem in detail: Given an... | 100/100 | 89 (modal x3) | 76 | 1 of 3 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |
| 4 | Debug and completely analyze this distributed system failure... | 100/100 | 81 (modal x4) | 38 | 1 of 4 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |
| 5 | Compare three database paradigms for a social media platform... | 100/100 | 84 (modal x4) | 11 | 1 of 4 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |
| 6 | you are a creative poet and you are tasked with Writting a h... | 100/100 | 20 (modal x22) | 52 | 22 of 22 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |
| 7 | Tell me about Richard Feynman | 100/100 | 85 (modal x3) | 23 | 1 of 3 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |
| 8 | You are SQL code parser agent that extracts all the source t... | 100/100 | 7 (modal x59) | 32 | 48 of 59 | 100/100 | 1 (modal x100) | - | 1 of 100 | **deterministic** |

Examples of where default-vLLM completions split (the most common completion vs the second most common):

- prompt 1, token 92: ...'-offs, and pseudocode.\n\n---\n\n## 🔧 **1. Architecture Overview**\n\n### **High-Level Architecture**\n```\n' -> '[Client] → [Global Load Balancer] → [' vs 'Client → Load Balancer → API Gateway (Rate Limiter'
- prompt 2, token 59: ...'(1) average time complexity for get and put operations\n- Configurable maximum capacity with automatic eviction\n- Per-entry' -> ' Time-To-Live (TTL) expiration\n- Background' vs ' TTL (Time-To-Live) expiration\n- Background cleanup'
- prompt 3, token 220: ...')\n\n---\n\n# ✅ Solution 1: Brute Force (Recursive with Memoization)\n\n### 🧠 Approach\n' -> 'Generate all possible subsequences and check which ones are increasing.' vs 'Use recursion to explore all possible subsequences. At each index'
- prompt 4, token 38: ...' Microservices Failure Under Load**\n\n---\n\n## **1. Root Cause Analysis (RCA) with Specific Failure Mode**\n\n###' -> ' **Root Cause:**\n**Connection Pool Exhaustion in API' vs ' ✅ **Root Cause:**\n**Connection Pool Exhaustion'
- prompt 5, token 75: ...' million posts, high write/read throughput.\n- **Core Features**:\n  - Real-time feed generation (personalized,' -> ' chronological, algorithmic).\n  - User relationships (follows' vs ' dynamic).\n  - User relationships (follows, friends,'
- prompt 6, token 162: ...' "Dreams compile to life" - The powerful culmination: the intangible vision (dreams) is transformed into tangible' -> ', functioning reality (life) through the compilation process.<|im_end|>' vs ' reality (life) through the compilation process, the ultimate goal'
- prompt 7, token 184: ...' and mathematical tool that simplifies the calculation of particle interactions. These diagrams revolutionized theoretical physics and are still widely used today' -> '.\n\n2. **Feynman Path Integral Formulation' vs '.\n   - For this work, Feynman, Schw'
- prompt 8, token 136: ...'target_table_name": "pres_del_table", "transformation_description": "Selects cycl_time_id and pres_id (' -> 'renamed as bp_id) from p1_rem_list_imm' vs 'as bp_id) from p1_rem_list_imm_plg'

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

## vLLM: N temperature-0 completions of one prompt

| job | model | mode | engine/api | ok/n x max_tokens | unique | modal count | first divergence (token #) | logprob variants of modal | attention backend |
|---|---|---|---|---|---|---|---|---|---|
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #1 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 80 | 4 | 33 | 1 of 4 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #2 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 88 | 3 | 17 | 1 of 3 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #3 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 89 | 3 | 76 | 1 of 3 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #4 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 81 | 4 | 38 | 1 of 4 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #5 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 84 | 4 | 11 | 1 of 4 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #6 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 20 | 22 | 52 | 22 of 22 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #7 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 85 | 3 | 23 | 1 of 3 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-baseline #8 | qwen3-30b-a3b-instruct-2507 | baseline | server/completions | 100/100x1000 | 7 | 59 | 32 | 48 of 59 | FLASH_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #1 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #2 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #3 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #4 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #5 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #6 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #7 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |
| mine-all-qwen3-30b-a3b-instruct-2507-batch_invariant #8 | qwen3-30b-a3b-instruct-2507 | batch_invariant | server/completions | 100/100x1000 | 1 | 100 | - | 1 of 100 | TRITON_ATTN |

Blog: Qwen3-235B-A22B-Instruct-2507, 1000 x 1000 tokens -> 80 unique (modal x78), first divergence at token 103 ("Queens, New York" vs "New York City"); batch-invariant: 1 unique. batch_invariant_ops' own vLLM example (Qwen3-8B, 100 tokens): 18 unique -> 1 unique.

Where completions first diverge:

- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #1` token 92: ...'-offs, and pseudocode.\n\n---\n\n## 🔧 **1. Architecture Overview**\n\n### **High-Level Architecture**\n```\n' -> '[Client] → [Global Load Balancer] → [' vs 'Client → Load Balancer → API Gateway (Rate Limiter'
- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #2` token 59: ...'(1) average time complexity for get and put operations\n- Configurable maximum capacity with automatic eviction\n- Per-entry' -> ' Time-To-Live (TTL) expiration\n- Background' vs ' TTL (Time-To-Live) expiration\n- Background cleanup'
- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #3` token 220: ...')\n\n---\n\n# ✅ Solution 1: Brute Force (Recursive with Memoization)\n\n### 🧠 Approach\n' -> 'Generate all possible subsequences and check which ones are increasing.' vs 'Use recursion to explore all possible subsequences. At each index'
- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #4` token 38: ...' Microservices Failure Under Load**\n\n---\n\n## **1. Root Cause Analysis (RCA) with Specific Failure Mode**\n\n###' -> ' **Root Cause:**\n**Connection Pool Exhaustion in API' vs ' ✅ **Root Cause:**\n**Connection Pool Exhaustion'
- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #5` token 75: ...' million posts, high write/read throughput.\n- **Core Features**:\n  - Real-time feed generation (personalized,' -> ' chronological, algorithmic).\n  - User relationships (follows' vs ' dynamic).\n  - User relationships (follows, friends,'
- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #6` token 162: ...' "Dreams compile to life" - The powerful culmination: the intangible vision (dreams) is transformed into tangible' -> ', functioning reality (life) through the compilation process.<|im_end|>' vs ' reality (life) through the compilation process, the ultimate goal'
- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #7` token 184: ...' and mathematical tool that simplifies the calculation of particle interactions. These diagrams revolutionized theoretical physics and are still widely used today' -> '.\n\n2. **Feynman Path Integral Formulation' vs '.\n   - For this work, Feynman, Schw'
- `mine-all-qwen3-30b-a3b-instruct-2507-baseline #8` token 136: ...'target_table_name": "pres_del_table", "transformation_description": "Selects cycl_time_id and pres_id (' -> 'renamed as bp_id) from p1_rem_list_imm' vs 'as bp_id) from p1_rem_list_imm_plg'
