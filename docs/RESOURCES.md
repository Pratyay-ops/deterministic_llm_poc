# Resources to understand and run the deterministic-LLM experiment

**The experiment in one sentence:** an LLM served at temperature 0 still gives different answers to the
same question when other requests share the server; we show this on our own prompts and remove it with
batch-invariant kernels (vLLM `VLLM_BATCH_INVARIANT=1`), on a single NVIDIA GB10 workstation.

**Results (2026-10-08):** Qwen3-8B - default vLLM gave up to 172 different answers per 200 identical
requests (7 of 8 prompts); batch-invariant vLLM gave exactly 1 per prompt, bitwise identical, and was not
slower (batch-invariant 1019 vs default 896 output tokens/s for the same ~1.23 M tokens). Qwen3-30B-A3B (MoE) -
default 7-89 different answers per 100 (8 of 8 prompts); batch-invariant exactly 1 per prompt, at the same
speed (464 vs 467 tokens/s). One measurement each, not a dedicated throughput test. Details:
`docs/FINDINGS_2026-10-08.md`, `experiment_results/2026-10-08_gb10/README.md`.

## At a glance

| resource | what is needed | what we used / measured |
|---|---|---|
| Hardware | 1 NVIDIA GB10 machine (Dell Pro Max GB10 or DGX Spark) | Dell Pro Max GB10: Blackwell GPU (cc 12.1), 20-core Arm CPU, 128 GB unified memory, 273 GB/s |
| GPU memory (unified) | ~40 GB for an everyday deterministic server; ~85-95 GB during the experiment | Qwen3-8B: 39 GB (server plan: 17 weights + 16 KV cache + 6) to 82 GB (256 parallel requests); Qwen3-30B-A3B: 92 GB |
| Disk | ~60 GB free for Qwen3-8B; ~125 GB if Qwen3-30B-A3B is included | image 24 GB, Qwen3-8B 16 GB, Qwen3-30B-A3B 61 GB, results < 0.1 GB per run, frozen package +8.4 GB |
| Downloads | ~10 GB container image + model weights | 10.1 GB image; 16 GB (8B, ~3 min) and 61 GB (30B, ~10 min) at ~100 MB/s |
| Software | DGX OS 7 (Ubuntu 24.04), NVIDIA driver >= 580, Docker + NVIDIA Container Toolkit | driver 595.91.07, Docker 29.6, image `vllm/vllm-openai:v0.31.0` |
| Code | this repo, branch `deterministic_llm_poc`, with the `batch_invariant_ops` submodule | runs used commit `2fc6a27`; launcher scripts added after |
| Access | sudo once (Docker group); GitHub access to the repo; Docker Hub and huggingface.co reachable | Hugging Face token optional (public models) |
| People | 1 engineer: Linux, Docker, Python; no CUDA expertise needed | |
| Time | ~1-2 h reading; ~30 min setup; ~1 h device time for the Qwen3-8B experiment; +~1 h for Qwen3-30B-A3B | measured 50 min (8B) and 64 min (30B) on 2026-10-08 |
| Cost | no cloud spend; the GB10 workstation only | |

## 1. To understand it (reading, ~1-2 hours)

In this order:

1. **Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (Horace He et al., 2025)** -
   https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ - the idea this work
   reproduces: kernels are run-to-run deterministic but not *batch-invariant*, and a server's batch size
   changes with load. ~45 min.
2. **This repo's `README.md`** - what is reproduced, the kernel modes, the layout. ~10 min.
3. **`docs/EXPERIMENTS.md`** - what each experiment measures and what result to expect. ~20 min.
4. **`experiment_results/2026-10-08_gb10/README.md`** - what we measured on our GB10. ~10 min.
5. **vLLM docs, "Batch Invariance"** (beta feature) - how `VLLM_BATCH_INVARIANT=1` works and which
   models/hardware are supported: https://docs.vllm.ai (Features -> Batch Invariance; source
   `docs/features/batch_invariance.md` in vLLM v0.31.0).
6. Optional: **thinking-machines-lab/batch_invariant_ops** - https://github.com/thinking-machines-lab/batch_invariant_ops
   - the reference batch-invariant kernels (matmul, RMSNorm, log-softmax), used here for the op-level tests.

Concepts worth knowing beforehand: floating-point addition is not associative; temperature 0 = greedy
decoding; continuous batching in an inference server; "logprobs" (per-token log-probabilities, compared
bit for bit to detect differences that do not show in the text).

## 2. To run it

**Hardware.** One GB10 (DGX Spark class). CPU and GPU share the 128 GB memory, so the OS, the file page cache
and the model compete for one pool - the runbook's "drop the page cache" step matters (§7 of the runbook).
Larger models do not fit: the blog's Qwen3-235B needs ~470 GB. Models used: Qwen3-8B (dense; main) and
Qwen3-30B-A3B-Instruct-2507 (mixture of experts). Hybrid linear-attention / Mamba models (Qwen3.5, 3.6,
3.8, Nemotron-H) cannot run in vLLM 0.31's batch-invariant mode.

**Software and access.**
- DGX OS 7.x with NVIDIA driver >= 580 (CUDA 13), Docker and the NVIDIA Container Toolkit (preinstalled on DGX OS).
- One-time sudo: `sudo usermod -aG docker $USER`; dropping the page cache before runs also needs root
  (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`, or a privileged container).
- Network: Docker Hub (`vllm/vllm-openai:v0.31.0`), huggingface.co (models), GitHub (this repo - private,
  needs a token with read access; write access to push results).
- Everything else runs inside the container: vLLM 0.31.0, PyTorch 2.13, Triton 3.7, transformers 5.17.

**Steps** (details in `docs/GB10_RUNBOOK.md`):

```bash
git clone --recurse-submodules -b deterministic_llm_poc https://github.com/pratyay-roy_ZSinc/Deterministic_llm_poc.git
cd Deterministic_llm_poc
bash scripts/gb10/setup.sh --suite smoke --suite qwen3-8b-mine      # checks, image, Qwen3-1.7B + Qwen3-8B (~15 min)
bash scripts/gb10/run_suite.sh smoke                                # sanity check (~5 min)
bash scripts/gb10/run_suite.sh configs/suites/qwen3-8b-mine.yaml --exclude 'sweep-*'   # the experiment (~1 h)
watch -n 10 python3 scripts/gb10/progress.py                        # live progress
```

**Measured device time (2026-10-08).**

| step | time |
|---|---|
| Container image pull + build | ~10 min |
| Model download: Qwen3-8B / Qwen3-30B-A3B | ~3 min / ~10 min |
| Smoke suite (Qwen3-1.7B, every experiment once) | ~4 min |
| Qwen3-8B: 8 prompts x 200 requests, default vLLM | 27 min (incl. ~4 min server start) |
| Qwen3-8B: same, batch-invariant vLLM | 23 min (incl. ~2 min server start) |
| Qwen3-30B-A3B: 8 prompts x 100 requests, default vLLM | 33 min (incl. ~10 min server start) |
| Qwen3-30B-A3B: same, batch-invariant vLLM | 31 min (incl. ~7 min server start) |
| CPU and op-level tests (floating point, 13 ops x 6 kernel modes) | ~1 min |

## 3. To use the result day to day

A ready-to-run deterministic Qwen3-8B (pinned weights, frozen vLLM image, launcher, self-check) needs
~40 GB of the GB10's memory and ~25 GB of disk for the package (16.4 GB weights + 8.4 GB compressed image;
+24 GB once the image is loaded into Docker):

```bash
bash scripts/gb10/serve_deterministic.sh       # OpenAI-compatible API on http://127.0.0.1:8000/v1
python3 scripts/gb10/check_determinism.py      # same prompt N times at once -> one answer
```

## 4. Risks and lessons from the first run

- **Disk** was the main constraint (the machine started 97% full): plan ~125 GB free for image + both models.
- **Page cache** shares the unified memory and CUDA's free-memory query does not count it; vLLM 0.31 compensates on
  GB10 (no run needed a memory retry), but dropping it before long runs (needs root) keeps headroom for every GPU program.
- **Docker group** membership needs sudo once; without it nothing runs.
- **Mixed load matters:** identical requests in one server produced far fewer variants (7-12 of 200) than
  the realistic mix of different prompts (up to 172 of 200) - test with a mix.
- **Scope of the guarantee:** deterministic for the same weights, vLLM build, GPU type and settings; a
  different GPU or vLLM version can give different (but again deterministic) answers.
