#!/usr/bin/env python3
"""Live job table of a suite run, for a terminal next to the run (host side, Python 3 stdlib only).

    watch -n 10 python3 scripts/gb10/progress.py                  # newest run in results/
    watch -n 10 python3 scripts/gb10/progress.py 'qwen3-8b-mine-*'
"""

import datetime
import glob
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
MARKS = {"ok": "[ok]  ", "running": "[run] ", "pending": "[wait]", "failed": "[FAIL]", "error": "[FAIL]",
         "timeout": "[FAIL]", "cancelled": "[stop]", "skipped": "[skip]", "unsatisfiable": "[skip]"}


def newest_run(pattern: str) -> str | None:
    runs = [d for d in glob.glob(os.path.join(RESULTS, pattern)) if os.path.isfile(os.path.join(d, "status.json"))]
    return max(runs, key=lambda d: os.path.getmtime(os.path.join(d, "status.json"))) if runs else None


def elapsed(run: str, job_id: str) -> float | None:
    """Seconds since the latest attempt started (from '===== attempt N at <iso> =====' in job.log)."""
    try:
        with open(os.path.join(run, job_id, "job.log"), errors="replace") as fh:
            starts = [line for line in fh if line.startswith("===== attempt")]
        stamp = starts[-1].split(" at ", 1)[1].split(" =====")[0].strip()
        started = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
        return (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()
    except (OSError, IndexError, ValueError):
        return None


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    run = newest_run(pattern)
    if run is None:
        sys.exit(f"no run matching results/{pattern} yet")
    try:
        with open(os.path.join(run, "status.json")) as fh:
            st = json.load(fh)
    except (OSError, ValueError) as exc:
        sys.exit(f"{os.path.basename(run)}: status.json not readable yet ({exc})")

    jobs = st.get("jobs", [])
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print(f"{os.path.basename(run)}   (status updated {st.get('updated', '?')})")
    print("   ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f"   total: {len(jobs)}")
    print()
    for j in jobs:
        dur = elapsed(run, j["id"]) if j["status"] == "running" else j.get("duration_s")
        took = f"{int(dur // 60)}m{int(dur % 60):02d}s" if dur else ""
        tries = f"try {j['attempts']}" if j.get("attempts", 0) > 1 else ""
        msg = (j.get("message") or "").replace("\n", " ")[:70]
        print(f"{MARKS.get(j['status'], '[ ]   ')} {j['id']:48s} {j['status']:9s} {took:>7s} {tries:6s} {msg}")

    mem: dict[str, int] = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, value = line.split(":", 1)
            mem[key] = int(value.split()[0])
    vfs = os.statvfs(RESULTS)
    print()
    print(
        f"memory: {mem['MemAvailable'] * 1024 / 1e9:.0f} GB available of {mem['MemTotal'] * 1024 / 1e9:.0f} GB "
        f"(page cache {mem['Cached'] * 1024 / 1e9:.0f} GB)   disk free: {vfs.f_bavail * vfs.f_frsize / 1e9:.0f} GB"
    )


if __name__ == "__main__":
    main()
