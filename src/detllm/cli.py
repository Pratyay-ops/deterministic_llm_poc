"""Command line interface: ``python -m detllm <command>``.

  env        print the environment snapshot (GPU, driver, CUDA, library versions)
  list       list experiments, modes, model presets and suites
  suite      run a suite of jobs in parallel (memory-aware, resumable)
  run        run a single experiment (as a one-job suite)
  report     (re)generate report.md for a run directory
  download   pre-download model weights for suites / models
  job        internal: execute one job (used by the scheduler)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from detllm.utils import configure_logging, parse_key_values, resolve_path

log = logging.getLogger("detllm")


def _cmd_env(args: argparse.Namespace) -> int:
    from detllm.env import collect_env

    print(json.dumps(collect_env(gpu=not args.no_gpu), indent=2, default=str))
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    from detllm.config import list_suites, load_models, load_suite
    from detllm.experiments.registry import EXPERIMENTS
    from detllm.modes import MODE_DESCRIPTIONS

    print("Experiments:")
    for info in EXPERIMENTS.values():
        print(f"  {info.name:<18} [{info.family}] {info.description}")
        print(f"  {'':<18} modes: {', '.join(info.modes)}")
    print("\nModes:")
    for name, text in MODE_DESCRIPTIONS.items():
        if name != "default":
            print(f"  {name:<16} {text}")
    print("\nModels (configs/models.yaml):")
    for key, preset in load_models().items():
        print(f"  {key:<30} {preset.hf_id:<38} vLLM ~{preset.vllm_mem_gb():.0f} GB")
    print("\nSuites (configs/suites):")
    for path in list_suites():
        try:
            suite = load_suite(str(path))
            print(f"  {path.stem:<12} {len(suite.jobs):>3} jobs  {suite.description.splitlines()[0] if suite.description else ''}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.stem:<12} INVALID: {exc}")
    return 0


def _run_suite(suite, args: argparse.Namespace) -> int:
    from detllm.analysis.report import write_report
    from detllm.runner import SuiteRunner, default_run_dir, latest_run_dir

    results_root = resolve_path(args.results_root)
    if args.run_dir:
        run_dir = resolve_path(args.run_dir)
    elif getattr(args, "resume_latest", False):
        run_dir = latest_run_dir(results_root, suite.name) or default_run_dir(results_root, suite.name)
    else:
        run_dir = default_run_dir(results_root, suite.name)

    if args.dry_run:
        total = 0.0
        print(f"Suite {suite.name}: {len(suite.jobs)} jobs -> {run_dir}")
        for job in suite.jobs:
            total += job.mem_gb
            flags = " exclusive" if job.exclusive else ""
            print(f"  {job.id:<48} {job.experiment:<17} mem {job.mem_gb:5.1f} GB{flags}")
        print(f"Sum of planned memory: {total:.0f} GB (budget {args.memory_budget_gb or suite.memory_budget_gb:.0f} GB)")
        return 0

    runner = SuiteRunner(
        suite,
        run_dir,
        resume=not args.no_resume,
        only=args.only,
        exclude=args.exclude,
        max_parallel=args.max_parallel,
        budget_gb=args.memory_budget_gb,
    )
    code = runner.run()
    if not args.no_report:
        try:
            path = write_report(run_dir)
            print(f"\nReport: {path}")
        except Exception as exc:  # noqa: BLE001
            log.error("report generation failed: %s", exc)
    return code


def _cmd_suite(args: argparse.Namespace) -> int:
    from detllm.config import load_suite

    return _run_suite(load_suite(args.suite), args)


def _cmd_run(args: argparse.Namespace) -> int:
    from detllm.config import SuiteSpec, expand_jobs, load_models

    models = load_models()
    entry = {"experiment": args.experiment, "params": parse_key_values(args.param)}
    if args.model:
        entry["model"] = args.model
    if args.mode:
        entry["mode"] = args.mode
    if args.mem_gb:
        entry["mem_gb"] = args.mem_gb
    if args.id:
        entry["id"] = args.id
    jobs = expand_jobs([entry], defaults={"seed": 0}, models=models)
    suite = SuiteSpec(name=f"run-{args.experiment}", memory_budget_gb=1e9, max_parallel=1, jobs=jobs)
    return _run_suite(suite, args)


def _cmd_job(args: argparse.Namespace) -> int:
    from detllm.jobs import run_job

    return run_job(Path(args.spec), Path(args.out))


def _cmd_report(args: argparse.Namespace) -> int:
    from detllm.analysis.report import write_report

    print(write_report(resolve_path(args.run_dir)))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import snapshot_download

    from detllm.config import load_models, load_suite

    models = load_models()
    wanted: list[str] = []
    for suite_name in args.suite or []:
        for job in load_suite(suite_name, models).jobs:
            if job.model and models[job.model].hf_id not in wanted:
                wanted.append(models[job.model].hf_id)
    for key in args.model or []:
        hf_id = models[key].hf_id if key in models else key
        if hf_id not in wanted:
            wanted.append(hf_id)
    if not wanted:
        print("nothing to download (use --suite and/or --model)")
        return 1

    def fetch(hf_id: str) -> str:
        log.info("downloading %s ...", hf_id)
        path = snapshot_download(
            hf_id, ignore_patterns=["*.pth", "*.pt", "*.bin", "*.gguf", "*.onnx", "original/*"]
        )
        log.info("ready: %s -> %s", hf_id, path)
        return path

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, hf_id): hf_id for hf_id in wanted}
        for fut, hf_id in futures.items():
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log.error("download failed for %s: %s", hf_id, exc)
    return 1 if failures else 0


def _add_suite_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", help="explicit results directory (re-using one resumes it)")
    p.add_argument("--results-root", default="results", help="parent folder for new run directories")
    p.add_argument("--no-resume", action="store_true", help="re-run jobs even if result.json says ok")
    p.add_argument("--only", nargs="+", metavar="GLOB", help="run only job ids matching these globs")
    p.add_argument("--exclude", nargs="+", metavar="GLOB", help="skip job ids matching these globs")
    p.add_argument("--max-parallel", type=int, help="override the suite's max_parallel")
    p.add_argument("--memory-budget-gb", type=float, help="override the suite's memory budget")
    p.add_argument("--dry-run", action="store_true", help="print the job plan and exit")
    p.add_argument("--no-report", action="store_true", help="do not write report.md at the end")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detllm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("env", help="print environment snapshot")
    p.add_argument("--no-gpu", action="store_true", help="skip CUDA queries")
    p.set_defaults(func=_cmd_env)

    p = sub.add_parser("list", help="list experiments, modes, models, suites")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("suite", help="run a suite (smoke | parallel | blog | path/to/suite.yaml)")
    p.add_argument("suite")
    p.add_argument("--resume-latest", action="store_true", help="continue the newest unfinished run of this suite")
    _add_suite_options(p)
    p.set_defaults(func=_cmd_suite)

    p = sub.add_parser("run", help="run one experiment")
    p.add_argument("experiment")
    p.add_argument("--model", help="model preset key (configs/models.yaml)")
    p.add_argument("--mode", help="mode (see `detllm list`)")
    p.add_argument("--param", action="append", metavar="KEY=VALUE", help="experiment parameter (YAML value)")
    p.add_argument("--mem-gb", type=float, help="memory to plan for this job")
    p.add_argument("--id", help="job id")
    _add_suite_options(p)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("report", help="write report.md for a run directory")
    p.add_argument("run_dir")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("download", help="pre-download model weights into the HF cache")
    p.add_argument("--suite", action="append", help="download every model used by this suite")
    p.add_argument("--model", action="append", help="model preset key or HF repo id")
    p.add_argument("--workers", type=int, default=3)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("job", help=argparse.SUPPRESS)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_job)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "job":  # job processes configure their own logging
        configure_logging()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
