# Deterministic LLM inference on NVIDIA GB10
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Pratyay-ops/deterministic_llm_poc)

A complete, runnable reproduction of Thinking Machines Lab's
[*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
(Horace He et al., 2025) for a **Dell Pro Max with NVIDIA GB10** (same platform as DGX Spark:
aarch64, Blackwell GPU with compute capability 12.1, ~122 GB unified CPU/GPU memory, DGX OS).

The blog's claim, in one line: LLM inference at temperature 0 is not reproducible because the
**batch size changes with server load** and common kernels (matmul, RMSNorm, attention) are not
*batch-invariant*. The forward pass is deterministic run to run; it just gives a request different
bits depending on what it was batched with. Batch-invariant kernels fix that.

> **Running it:** follow [docs/GB10_RUNBOOK.md](docs/GB10_RUNBOOK.md) on the GB10.
> **What each experiment shows:** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Quick start (on the GB10)

```bash
git clone --recurse-submodules -b deterministic_llm_poc https://github.com/pratyay-roy_ZSinc/Deterministic_llm_poc.git
cd Deterministic_llm_poc
bash scripts/gb10/setup.sh                 # host checks, container image, model download
bash scripts/gb10/run_suite.sh smoke       # ~10-20 min: is everything working?
bash scripts/gb10/run_suite.sh parallel    # main reproduction (small models, run side by side)
# -> results/parallel-<timestamp>/report.md
```

Everything runs inside NVIDIA's vLLM container (`nvcr.io/nvidia/vllm:26.09-py3` by default); the
repository is mounted into it, so nothing is installed on the host apart from the Docker image.

## What is reproduced

| Blog section | Experiment | Engine |
|---|---|---|
| The original sin: floating-point non-associativity | `fp_nonassoc` - the 102 unique sums | Python |
| Why kernels are run-to-run deterministic but not batch-invariant | `op_invariance` - matmul, linear, RMSNorm, (log-)softmax, bmm, eager and SDPA attention, plus the blog's `torch.mm(a[:1], b)` vs `torch.mm(a, b)[:1]` demo | PyTorch |
| How to make kernels batch-invariant / Implementation (torch.Library) | `op_invariance`, `hf_invariance` - six kernel modes incl. thinking-machines-lab/batch_invariant_ops | PyTorch, HF transformers |
| How nondeterministic are completions? (1000 x "Tell me about Richard Feynman", T=0) | `vllm_completions` - default vs `VLLM_BATCH_INVARIANT=1` | vLLM server |
| batch_invariant_ops' vLLM proof of concept (18 unique -> 1) | `vllm_completions` with `api: chat` | vLLM server |
| Batch invariance at the engine level | `vllm_batch_sweep` - needle request alone vs. in batches of 2..128, bitwise logprobs | vLLM offline |
| Performance (1000 requests, 90-110 tokens) | `vllm_perf` - runs alone on the GPU | vLLM server |

The blog's own model (Qwen3-235B-A22B, ~470 GB) does not fit a GB10. The suites use Qwen3-1.7B
(vLLM's own batch-invariance test model), Qwen3-4B-Instruct-2507, Qwen3-8B (the blog's
performance model) and optionally the Qwen3-30B-A3B-Instruct-2507 MoE.

## Kernel modes

| Mode | What it is |
|---|---|
| `baseline` | stock kernels |
| `torch_flags` | `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms`, cuDNN deterministic, no TF32 - fixes run-to-run nondeterminism, **not** batch variance |
| `cublaslt_ws1` | cuBLASLt with a ~1 KiB workspace (no split-K) - what vLLM's batch-invariant mode uses for GEMMs on Hopper/Blackwell and GB10 |
| `official` | [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) (git submodule) |
| `official_full` | the library + this repo's batch-invariant Triton `bmm`/`softmax` (attention, which the library lacks) |
| `custom` | this repo's own approach: vendor kernels on fixed-size zero-padded chunks via a `TorchDispatchMode` |
| `batch_invariant` (vLLM) | `VLLM_BATCH_INVARIANT=1` |

## Repository layout

```
configs/
  models.yaml            model presets + unified-memory budgets
  suites/                smoke.yaml, parallel.yaml, blog.yaml
  prompts/               the blog prompt, the library's PoC prompt, batch fillers, your prompts
docker/                  Dockerfile (thin layer on the NGC / upstream vLLM image) + entrypoint
scripts/gb10/            setup.sh, run_suite.sh, run.sh, collect_results.sh, setup_native.sh
src/detllm/
  cli.py, runner.py      `python3 -m detllm ...`; memory-aware parallel scheduler (resumable)
  jobs.py, config.py     one process per job; suites/presets
  kernels/               batch_invariant_ops wrapper, extra Triton ops, CUSTOM mode, mode switches
  engines/               HF transformers, vLLM offline, vLLM server + concurrent client
  experiments/           preflight, fp_nonassoc, op_invariance, hf_invariance, vllm_*
  analysis/              divergence statistics, report.md generation
tests/                   unit tests (CPU) and kernel tests (GPU, marker `gpu`)
batch_invariant_ops/     submodule: thinking-machines-lab/batch_invariant_ops @ f22b1fb
```

## Changes from the previous version of this repo

* One GPU, many jobs: the old runner pinned BASELINE/OFFICIAL/CUSTOM to GPUs 0/1/2, which a GB10
  does not have. Jobs now run side by side on the single GPU within a memory budget; only the
  throughput measurements run alone.
* vLLM determinism now uses vLLM's native batch-invariant mode. Wrapping `llm.generate` in
  `set_batch_invariant_mode()` had no effect: vLLM V1 runs the model in a separate engine-core
  process.
* The old CUSTOM ops could not work: `torch.matmul` was replaced by a function that called
  `torch.matmul` again (infinite recursion). Python-level patching also never reaches `nn.Linear`.
  And padding the batch to a multiple of 32 still changes the kernel shape. The new `custom` mode
  keeps the idea but intercepts at the dispatcher and always runs identical shapes.
* bf16 instead of fp16: the library's fp16 matmul tile needs ~144 KB of shared memory per block;
  GB10 has ~99 KB.

## Citation

```bibtex
@article{he2025nondeterminism,
  author = {Horace He and Thinking Machines Lab},
  title = {Defeating Nondeterminism in LLM Inference},
  journal = {Thinking Machines Lab: Connectionism},
  year = {2025},
  note = {https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/},
  doi = {10.64434/tml.20250910}
}
```
