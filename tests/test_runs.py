import json

from gaitpinn import runs


def test_billing_matches_lumi_examples():
    # LUMI docs: 1 GCD, 24 h, 128 GB -> 24 GPU-hours; 2 GCDs, 24 h -> 24 GPU-hours.
    assert runs.billed_gpu_hours(24 * 3600, gpus=1, cpus=7, mem_gb=128) == 24
    assert runs.billed_gpu_hours(24 * 3600, gpus=2, cpus=14, mem_gb=60) == 24
    assert runs.billed_gpu_hours(3600, gpus=1, cpus=8, mem_gb=64) == 0.5


def test_parse_lumi_tres():
    t = runs.parse_tres("billing=14,cpu=14,gres/gpu:mi250=1,mem=60G,node=1")
    assert t == {"cpu": 14, "gpu": 1, "mem_gb": 60.0}


SACCT = (
    "123|gp-smoke|COMPLETED|720|billing=8,cpu=8,gres/gpu:mi250=1,mem=60G,node=1\n"
    "124|gp-final|FAILED|360|billing=8,cpu=8,gres/gpu:mi250=1,mem=60G,node=1\n"
    "125|gp-final|RUNNING|100|billing=8,cpu=8,gres/gpu:mi250=1,mem=60G,node=1\n"
    "126|other|COMPLETED|3600|billing=8,cpu=8,gres/gpu:mi250=1,mem=60G,node=1\n"
)


def test_ledger_counts_failed_jobs_and_survives_rsync(tmp_path):
    rec = {"run_id": "r1", "kind": "smoke", "slurm_job": "123", "gpu_hours": None}
    runs.write_run(tmp_path, rec)
    ledger = runs.update_ledger(tmp_path, SACCT)
    assert set(ledger) == {"123", "124"}  # running and non-gp jobs are skipped
    assert abs(runs.total_gpu_hours(ledger) - 0.15) < 1e-9
    assert runs.fill_gpu_hours(tmp_path, ledger) == 1
    # rsync puts back the remote run.json (gpu_hours null); the ledger refills it offline
    runs.write_run(tmp_path, rec)
    ledger = runs.update_ledger(tmp_path, "")
    assert runs.fill_gpu_hours(tmp_path, ledger) == 1
    assert json.loads((tmp_path / "r1" / "run.json").read_text())["gpu_hours"] == 0.1


def test_index_round_trip(tmp_path):
    rec = {
        "run_id": "smoke-base-none-s0-jlocal",
        "kind": "smoke",
        "status": "done",
        "model": "base",
        "person": "none",
        "seed": 0,
        "commit": "abc",
        "slurm_job": None,
        "gpu_hours": None,
    }
    runs.write_run(tmp_path, rec)
    lines = runs.write_index(tmp_path).read_text().splitlines()
    assert lines[0].startswith("run_id,") and lines[1].startswith(rec["run_id"])


def test_smoke_run_writes_record(tmp_path):
    import sys

    from gaitpinn import train

    argv = [
        "train",
        "--config",
        "configs/base.yaml",
        "--smoke",
        "--steps",
        "3",
        "--runs",
        str(tmp_path),
    ]
    old, sys.argv = sys.argv, argv
    try:
        train.main()
    finally:
        sys.argv = old
    (rec,) = runs.load_runs(tmp_path)
    assert rec["status"] == "done" and rec["kind"] == "smoke"
    assert rec["metrics"]["synthetic"] is True
