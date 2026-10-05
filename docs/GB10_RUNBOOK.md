# GB10 runbook

Step-by-step instructions for running the reproduction on a **Dell Pro Max with NVIDIA GB10**
(DGX OS). The same steps work on a DGX Spark. Every command below runs **on the GB10**.

* [1. Before you start](#1-before-you-start)
* [2. Get the code](#2-get-the-code)
* [3. One-time setup](#3-one-time-setup)
* [4. Plan your device time](#4-plan-your-device-time)
* [5. Run the suites](#5-run-the-suites)
* [6. Read the results](#6-read-the-results)
* [7. Memory on a unified-memory GPU](#7-memory-on-a-unified-memory-gpu)
* [8. Choosing the container image](#8-choosing-the-container-image)
* [9. Troubleshooting](#9-troubleshooting)
* [10. Take the results home](#10-take-the-results-home)
* [11. Tests and manual runs](#11-tests-and-manual-runs)

---

## 1. Before you start

| Need | Check |
|---|---|
| DGX OS with the NVIDIA driver | `nvidia-smi` shows `NVIDIA GB10`, compute capability `12.1` |
| Docker + NVIDIA Container Toolkit (preinstalled on DGX OS) | `docker run --rm --gpus all ubuntu nvidia-smi` |
| Your user in the `docker` group | `docker ps` works without sudo; otherwise `sudo usermod -aG docker $USER && newgrp docker` |
| Disk space | ~20 GB for the image, ~20 GB of models for `smoke` + `parallel`, ~110 GB more for `blog` |
| Network access | Docker Hub (`vllm/vllm-openai` image; `nvcr.io` only if you switch to the NGC image), `huggingface.co` (models), `github.com` (code) |
| Optional: Hugging Face token | Qwen models are public; a token only raises rate limits: `export HF_TOKEN=hf_...` |
| `tmux` (recommended) | long runs survive an SSH disconnect: `tmux new -s detllm` |

Facts about the platform that shaped this setup:

* **One GPU, ~122 GB unified memory.** CPU and GPU share the same LPDDR5x, so the OS, page cache,
  Python processes and every model compete for one pool. `nvidia-smi` shows "Memory-Usage: Not
  Supported" on this iGPU platform; that is expected.
* **Compute capability 12.1 (sm_121), consumer-Blackwell-class.** About 99 KB of shared memory per
  thread block, compared with 227 KB on H100. Triton tiles tuned for data-center GPUs can fail to
  launch (see §9).
* **aarch64 + CUDA 13.** Run vLLM/PyTorch from a container built for arm64 and CUDA 13: the
  default is the upstream `vllm/vllm-openai:v0.31.0`, with NVIDIA's NGC build as the alternative
  (§8).

## 2. Get the code

```bash
git clone --recurse-submodules -b deterministic_llm_poc \
    https://github.com/pratyay-roy_ZSinc/Deterministic_llm_poc.git
cd Deterministic_llm_poc
```

`--recurse-submodules` fetches `batch_invariant_ops/` (thinking-machines-lab/batch_invariant_ops,
pinned to `f22b1fb`). If you forgot it, `setup.sh` runs `git submodule update --init` for you.

## 3. One-time setup

```bash
bash scripts/gb10/setup.sh
```

What it does (safe to re-run):

1. Host checks: architecture, `/etc/dgx-release`, driver, Docker + NVIDIA runtime, free disk/memory.
   Saves them to `results/host_info.txt`.
2. Initialises the `batch_invariant_ops` submodule.
3. Pulls `vllm/vllm-openai:v0.31.0` from Docker Hub, which is large. This is the slowest step the
   first time.
4. Builds `detllm:latest`, a thin layer that adds only `pytest` and environment variables. Torch,
   Triton, vLLM and transformers come from the base image untouched.
5. Runs `python3 -m detllm env` inside the container. Check that it reports the GB10 and `cuda_available: true`.
6. Downloads the models used by the `smoke` and `parallel` suites (Qwen3-1.7B,
   Qwen3-4B-Instruct-2507) into `~/.cache/huggingface`.

Options: `--suite blog` (download the blog-suite models too, ~80 GB), `--skip-download`,
`--skip-build`. The cache directories are `~/.cache/huggingface` (models) and `~/.cache/detllm`
(Triton / torch.compile / vLLM compile caches, kept between runs so later runs start faster).

## 4. Plan your device time

Jobs run **side by side** on the GPU, packed into a memory budget (default 100 GB of the ~122 GB).
Running concurrently does not change the numerics any job computes: kernel choice depends on
shapes, not on GPU load. Only the throughput jobs are marked `exclusive` and run alone at the end.

| Suite | What | Rough wall time* | Models to download |
|---|---|---|---|
| `smoke` | every experiment once, tiny workloads | 10-20 min | Qwen3-1.7B (4 GB) |
| `parallel` | the full reproduction with small models (main suite) | 1-1.5 h | + Qwen3-4B-Instruct-2507 (8 GB) |
| `blog` | Qwen3-8B completions + throughput (blog's perf model), 4B sweeps, optional 30B-A3B MoE | 2-3 h (1-1.5 h without MoE) | + Qwen3-8B (16 GB), Qwen3-30B-A3B (61 GB) |

\* First runs are slower: Triton kernels, torch.compile graphs and CUDA graphs are compiled and
cached. These are estimates; the job log shows real progress.

Recommended order: `smoke` → `parallel` → (if time allows) `blog`. Every suite is **resumable**,
so you can stop at any point and continue in a later session.

## 5. Run the suites

```bash
bash scripts/gb10/run_suite.sh smoke
bash scripts/gb10/run_suite.sh parallel
bash scripts/gb10/run_suite.sh blog --exclude 'moe-*'      # without the 80 GB MoE jobs
```

`run_suite.sh` first flushes the page cache (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`,
see §7), then runs `python3 -m detllm suite <suite> --resume-latest` in the container. The console
shows one line per job start/finish:

```
10:02:11 INFO detllm.runner: START completions-qwen3-1.7b-baseline   mem  17.1 GB | running 5/8, planned 61/100 GB, MemAvailable 70 GB
10:09:40 INFO detllm.runner: DONE  ops-official_full                 3m12s
```

**Disconnect-proof runs.** Use `tmux`, or run detached and follow the logs:

```bash
DETACH=1 bash scripts/gb10/run_suite.sh parallel
docker logs -f detllm-parallel
```

**Stopping and resuming.** `Ctrl+C` (or `docker stop detllm-parallel`) terminates the running jobs
cleanly. Running the same command again continues the newest unfinished run of that suite: jobs
whose `result.json` says `ok` are skipped and failed or cancelled jobs are re-run. Pass
`--no-resume` to start over, or `--run-dir results/<dir>` to resume a specific run.

**Subsets and tuning.**

```bash
bash scripts/gb10/run_suite.sh parallel --dry-run                    # show the job plan
bash scripts/gb10/run_suite.sh parallel --only 'completions-*' 'sweep-*'
bash scripts/gb10/run_suite.sh parallel --exclude 'ablate-*' 'perf-*'
bash scripts/gb10/run_suite.sh parallel --memory-budget-gb 90 --max-parallel 6
```

**Monitoring.** `results/<run>/status.json` holds the state of every job. Each job writes
`results/<run>/<job>/job.log`, and server jobs also write `vllm_server.log`. On the host, `free -g`
shows how much of the unified memory is in use.

## 6. Read the results

```
results/parallel-20261006-101500/
  report.md            <- start here
  summary.json         machine-readable summaries of all jobs
  status.json          job states, durations, attempts
  suite.json           the expanded suite that was run
  <job-id>/
    job.json  result.json  env.json  job.log
    completions.jsonl  needle_outputs.jsonl  vllm_server.log  ...   (experiment specific)
```

What a successful reproduction looks like in `report.md`:

* **Headline.** Baseline vLLM gives *many* unique completions out of 1000, together with the token
  position where they first diverge. Batch-invariant vLLM gives **1 unique**, and the logprobs of
  all 1000 completions are bitwise identical ("logprob variants of modal = 1").
* **Op table.** `baseline` and `torch_flags` show "varies" for matmul-like ops while every op is
  run-to-run deterministic: this is the blog's central point. `official_full` and `custom` show
  "invariant". `official` covers mm/linear/RMSNorm/log_softmax but not bmm/softmax.
  `cublaslt_ws1` tests, on your GB10, the assumption vLLM makes for GEMMs on this architecture.
  SDPA attention is not overridden by any mode, so it shows whether the stock attention kernels on
  sm_121 are batch-invariant.
* **HF model table.** `baseline` / `baseline_eager` may differ across batch sizes; `official_full`
  and `custom` should be bitwise invariant for both the logits and greedy decoding.
* **Needle sweep.** Baseline mismatches (at least in logprobs) at some batch sizes;
  batch-invariant mode is bitwise identical everywhere.
* **Throughput.** The slowdown of batch-invariant mode. The blog measured 26 s → 55 s → 42 s on
  its GPU with Qwen3-8B; numbers on GB10 differ in absolute terms.

What the blog measured, for comparison: Qwen3-235B-A22B-Instruct-2507 gave 80 unique completions
out of 1000 (the most common one 78 times), identical for the first 102 tokens and diverging at
token 103 ("Queens, New York" vs "New York City"); with batch-invariant kernels, 1 unique.
Smaller models diverge at different points. Some may produce few unique completions in baseline
mode, which is why the suites also compare logprobs bitwise.

## 7. Memory on a unified-memory GPU

* **Budget.** Each job declares the memory it needs (`mem_gb` in `status.json`). For vLLM jobs this
  is weights + KV cache + overhead (see `configs/models.yaml`) and is passed to vLLM as both
  `gpu_memory_utilization` and an explicit `kv_cache_memory_bytes`. Several engines can therefore
  share the GPU predictably. The scheduler starts jobs until the suite's `memory_budget_gb` is used.
* **Page cache.** On GB10, file pages cached by the OS sit in the same memory as GPU allocations,
  and `cudaMemGetInfo` (which vLLM checks at start-up) does not count them as free. This is a
  documented DGX Spark issue. The harness handles it in two ways:
  `run_suite.sh` drops the page cache before a suite (needs sudo), and every job evicts its own
  model files from the cache right after loading them (`posix_fadvise`, no root needed).
* **If an engine cannot get memory at start-up,** the job exits with "retry later" and the
  scheduler re-queues it (up to 3 times, with back-off) instead of failing.
* **Tuning.** If you see repeated retries or OOM kills, lower the budget
  (`--memory-budget-gb 85`) or the parallelism (`--max-parallel 4`). To give a job more KV cache,
  set `kv_cache_gb` in its suite entry; its planned memory grows to match.

## 8. Choosing the container image

| Image | vLLM | Notes |
|---|---|---|
| `vllm/vllm-openai:v0.31.0` (default) | 0.31.0 | upstream release (2026-10-04, arm64 included) with the newest batch-invariance code, including Triton matmul tiles tuned for 12.x GPUs (GB10 class) for Qwen3-1.7B/4B/8B. CUDA 13.0, kernels built for compute capability 12.0 (runs on 12.1), Python 3.12. NVIDIA's Spark playbook also uses upstream vLLM images |
| `nvcr.io/nvidia/vllm:26.09-py3` | 0.29.0 | NVIDIA's NGC build, validated by NVIDIA for DGX Spark / GB10; PyTorch 2.14 pre-release build, CUDA 13.4.1, Python 3.12, Transformers 5.15. No GB10-tuned batch-invariant tiles yet (correct, but slower in batch-invariant mode) |

Switching to the NGC image (start a fresh run so results from the two images are not mixed):

```bash
BASE_IMAGE=nvcr.io/nvidia/vllm:26.09-py3 bash scripts/gb10/setup.sh --skip-download
BASE_IMAGE=nvcr.io/nvidia/vllm:26.09-py3 bash scripts/gb10/run_suite.sh parallel --no-resume
```

The image actually used is recorded in every `env.json` and in the report's environment table.
`python3 -m detllm run preflight` shows whether the installed vLLM has a tuned batch-invariant table
for your GPU (`tuned_matmul_table_for_this_gpu`; expected `true` with v0.31.0).

Driver compatibility: DGX OS 7 ships driver 580, which supports CUDA 13.0 natively, matching the
default image. If `detllm env` reports no GPU or a driver/CUDA mismatch, update DGX OS, or use the
CUDA 12.9 build of the same release: `BASE_IMAGE=vllm/vllm-openai:v0.31.0-cu129`. The NGC image
uses CUDA 13.4 and relies on CUDA minor-version compatibility with driver 580; NVIDIA's own Spark
playbooks use NGC images this way.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `permission denied ... docker.sock` | `sudo usermod -aG docker $USER && newgrp docker` |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing/not configured: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `detllm env` shows `cuda_available: false` | driver too old for the image's CUDA → update DGX OS or use an older image tag (§8) |
| `ptxas fatal: Value 'sm_121a' is not defined` (Triton) | the Triton build ships an older `ptxas`. Use the image's CUDA 13 one: `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas bash scripts/gb10/run_suite.sh smoke` (the variable is passed into the container), or use a newer image |
| `OutOfResources: shared memory` in `ops-official` for `float16` | expected: the library's fp16 tile needs ~144 KB of shared memory per block, GB10 has ~99 KB. It shows up as an `error` cell in the op table; models run in bf16 |
| vLLM job retries with "could not reserve memory yet" | other jobs or the page cache hold the memory. It recovers by itself; if it repeats, drop caches (§7) and lower `--memory-budget-gb` |
| `Free memory on device ... less than desired GPU memory utilization` in `vllm_server.log` | same as above |
| vLLM fails with an attention-backend error | set the backend explicitly in the suite entry, e.g. `params: {attention_backend: FLASH_ATTN}` (or `bi_attention_backend` for batch-invariant jobs; the default there is `TRITON_ATTN`) |
| `ops-custom` / `hf-...-custom` are slow | expected: the CUSTOM mode routes every op through Python and pads chunks. It exists for correctness, not speed |
| model download fails / rate limited | `export HF_TOKEN=...` before `setup.sh`; for an offline machine, copy `~/.cache/huggingface` over and set `HF_HUB_OFFLINE=1` |
| a job hits its timeout (default 4 h) | raise `timeout_min` in the suite entry or reduce `n` / `max_tokens` |
| first run very slow | compilation (Triton, torch.compile, CUDA graphs). Caches persist in `~/.cache/detllm`; later runs are faster |

The full traceback of any failed job is in `results/<run>/<job>/result.json` and `job.log`.

## 10. Take the results home

```bash
bash scripts/gb10/collect_results.sh            # newest run -> results/<run>.tar.gz
LIGHT=1 bash scripts/gb10/collect_results.sh    # without per-completion files and server logs
```

The archive contains `report.md`, all summaries, logs, environment snapshots and
`results/host_info.txt`. You can regenerate the report anywhere with Python and PyYAML:
`PYTHONPATH=src python3 -m detllm report results/<run>`.

## 11. Tests and manual runs

```bash
bash scripts/gb10/run.sh python3 -m pytest -q                 # unit tests (CPU only)
bash scripts/gb10/run.sh python3 -m pytest -q -m gpu tests/gpu   # kernel tests on the GB10
bash scripts/gb10/run.sh python3 -m detllm run preflight      # ~1 min environment check
bash scripts/gb10/run.sh python3 -m detllm list               # experiments, modes, models, suites
bash scripts/gb10/run.sh                                      # interactive shell in the container

# one experiment by hand (still goes through the scheduler, results land in results/run-*)
bash scripts/gb10/run.sh python3 -m detllm run vllm_completions --model qwen3-1.7b \
    --mode batch_invariant --param n=200 --param max_tokens=300
```

Without Docker (fallback, not recommended): `bash scripts/gb10/setup_native.sh` creates `.venv`
with `pip install vllm`. This only works if an aarch64 CUDA wheel exists that supports sm_121 with
your driver.
