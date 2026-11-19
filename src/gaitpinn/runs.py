"""Run records: one folder per run with run.json, plus runs.csv and log.md."""

import csv
import json
import math
import os
import subprocess
from pathlib import Path

FIELDS = ["run_id", "kind", "model", "person", "seed", "commit", "slurm_job", "gpu_hours"]


def run_id(kind: str, model: str, person: str, seed: int) -> str:
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    job = f"{job}_{task}" if task else job
    return f"{kind}-{model}-{person}-s{seed}-j{job}"


def slurm_job() -> str | None:
    job = os.environ.get("SLURM_JOB_ID")
    return job


def code_commit(code_dir: Path) -> str:
    """Commit of the code copy (COMMIT file written by jobs/sync.sh) or of a local git tree."""
    f = code_dir / "COMMIT"
    if f.exists():
        return f.read_text().strip()
    try:
        out = subprocess.run(
            ["git", "-C", str(code_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run(runs_dir: Path, record: dict) -> Path:
    d = runs_dir / record["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return d


def load_runs(runs_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*/run.json"))]


def write_index(runs_dir: Path) -> Path:
    """Rebuild runs/runs.csv from all run.json files."""
    runs = load_runs(runs_dir)
    path = Path(runs_dir) / "runs.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    return path


def billed_gpu_hours(elapsed_s: float, gpus: int, cpus: int, mem_gb: float) -> float:
    """LUMI small-g/dev-g billing: 0.5 per slot-hour, slots = max(GCDs, cpus/8, mem/64GB)."""
    slots = max(gpus, math.ceil(cpus / 8), math.ceil(mem_gb / 64))
    return slots * elapsed_s / 3600 * 0.5


def parse_tres(tres: str) -> dict:
    """'billing=8,cpu=7,gres/gpu=1,mem=60G,node=1' -> {'cpu': 7, 'gpu': 1, 'mem_gb': 60.0}."""
    kv = dict(p.split("=", 1) for p in tres.split(",") if "=" in p)
    mem = kv.get("mem", "0M")
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}
    mem_gb = float(mem[:-1]) * scale[mem[-1]] if mem[-1] in scale else float(mem) / 1024
    return {"cpu": int(kv.get("cpu", 0)), "gpu": int(kv.get("gres/gpu", 0)), "mem_gb": mem_gb}


def fill_gpu_hours(runs_dir: Path, sacct_text: str) -> int:
    """Set gpu_hours in run.json files from `sacct -X -n -P` output; returns how many."""
    acct = {}
    for line in sacct_text.strip().splitlines():
        job, state, elapsed, tres = line.split("|")
        if state.startswith(("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY")):
            t = parse_tres(tres)
            acct[job] = billed_gpu_hours(int(elapsed), t["gpu"], t["cpu"], t["mem_gb"])
    n = 0
    for p in Path(runs_dir).glob("*/run.json"):
        r = json.loads(p.read_text())
        if r.get("gpu_hours") is None and str(r.get("slurm_job")) in acct:
            r["gpu_hours"] = round(acct[str(r["slurm_job"])], 6)
            p.write_text(json.dumps(r, indent=2, sort_keys=True))
            n += 1
    return n


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pending", "fill", "index"])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--sacct", type=Path)
    a = ap.parse_args()
    if a.action == "pending":
        jobs = {
            str(r["slurm_job"])
            for r in load_runs(a.runs)
            if r.get("gpu_hours") is None and r.get("slurm_job")
        }
        print(",".join(sorted(jobs)))
    elif a.action == "fill":
        print(f"filled gpu_hours for {fill_gpu_hours(a.runs, a.sacct.read_text())} runs")
    else:
        print(f"wrote {write_index(a.runs)}")


if __name__ == "__main__":
    main()
