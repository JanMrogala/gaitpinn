"""Run records and GPU-hour accounting.

- `runs/<run_id>/run.json`: one per run. Written when the run starts (status `running`) and
  again when it ends (status `done` with metrics).
- `runs/billing.csv`: local ledger of every `gp-*` Slurm job seen in `sacct`, failed ones
  included. It is the source of `gpu_hours` in run.json and of the running total.
- `runs/runs.csv`: index of all runs. `runs/log.md`: short dated notes, written by hand.
"""

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

FIELDS = ["run_id", "kind", "status", "model", "person", "seed", "commit", "slurm_job", "gpu_hours"]
LEDGER = ["job", "name", "state", "elapsed_s", "gpu_hours"]
FINAL_STATES = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL")


def run_id(kind: str, model: str, person: str, seed: int) -> str:
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    job = f"{job}_{task}" if task else job
    return f"{kind}-{model}-{person}-s{seed}-j{job}"


def slurm_job() -> str | None:
    return os.environ.get("SLURM_JOB_ID")


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
    d = Path(runs_dir) / record["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return d


def load_runs(runs_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*/run.json"))]


def write_index(runs_dir: Path) -> Path:
    """Rebuild runs/runs.csv from all run.json files."""
    path = Path(runs_dir) / "runs.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in load_runs(runs_dir):
            w.writerow(r)
    return path


def billed_gpu_hours(elapsed_s: float, gpus: int, cpus: int, mem_gb: float) -> float:
    """LUMI small-g/dev-g billing: 0.5 per slot-hour, slots = max(GCDs, cpus/8, mem/64GB)."""
    slots = max(gpus, math.ceil(cpus / 8), math.ceil(mem_gb / 64))
    return slots * elapsed_s / 3600 * 0.5


def parse_tres(tres: str) -> dict:
    """'cpu=8,gres/gpu:mi250=1,mem=60G' -> {'cpu': 8, 'gpu': 1, 'mem_gb': 60.0}.

    `cpu` is counted as Slurm reports it (hardware threads on LUMI-G), which can only
    over-estimate the billed slots, never under-estimate them.
    """
    kv = dict(p.split("=", 1) for p in tres.split(",") if "=" in p)
    mem = kv.get("mem", "0M")
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}
    mem_gb = float(mem[:-1]) * scale[mem[-1]] if mem[-1] in scale else float(mem) / 1024
    # LUMI reports the GPU as e.g. 'gres/gpu:mi250=1' and sometimes also 'gres/gpu=1'.
    gpu = max([int(v) for k, v in kv.items() if k.startswith("gres/gpu")] or [0])
    return {"cpu": int(kv.get("cpu", 0)), "gpu": gpu, "mem_gb": mem_gb}


def read_ledger(runs_dir: Path) -> dict[str, dict]:
    p = Path(runs_dir) / "billing.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["job"]: r for r in csv.DictReader(f)}


def update_ledger(runs_dir: Path, sacct_text: str) -> dict[str, dict]:
    """Add finished gp-* jobs from `sacct -X -n -P` (JobIDRaw|JobName|State|ElapsedRaw|AllocTRES)
    to runs/billing.csv. Entries already in the ledger are kept as they are."""
    ledger = read_ledger(runs_dir)
    for line in sacct_text.strip().splitlines():
        job, name, state, elapsed, tres = line.split("|")
        if not name.startswith("gp-") or not state.startswith(FINAL_STATES) or job in ledger:
            continue
        t = parse_tres(tres)
        gh = billed_gpu_hours(int(elapsed), t["gpu"], t["cpu"], t["mem_gb"])
        ledger[job] = {
            "job": job,
            "name": name,
            "state": state.split()[0],
            "elapsed_s": elapsed,
            "gpu_hours": f"{gh:.6f}",
        }
    p = Path(runs_dir) / "billing.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER)
        w.writeheader()
        for job in sorted(ledger, key=int):
            w.writerow(ledger[job])
    return ledger


def fill_gpu_hours(runs_dir: Path, ledger: dict[str, dict]) -> int:
    """Copy gpu_hours from the ledger into run.json files; returns how many were set."""
    n = 0
    for p in Path(runs_dir).glob("*/run.json"):
        r = json.loads(p.read_text())
        job = str(r.get("slurm_job"))
        if job in ledger and r.get("gpu_hours") is None:
            r["gpu_hours"] = float(ledger[job]["gpu_hours"])
            p.write_text(json.dumps(r, indent=2, sort_keys=True))
            n += 1
    return n


def total_gpu_hours(ledger: dict[str, dict]) -> float:
    return sum(float(r["gpu_hours"]) for r in ledger.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["ledger", "index"])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--sacct", type=Path)
    a = ap.parse_args()
    if a.action == "ledger":
        ledger = update_ledger(a.runs, a.sacct.read_text())
        n = fill_gpu_hours(a.runs, ledger)
        print(
            f"ledger: {len(ledger)} jobs, {total_gpu_hours(ledger):.3f} GPU-hours in total; "
            f"filled {n} run.json files"
        )
    else:
        print(f"wrote {write_index(a.runs)}")


if __name__ == "__main__":
    main()
