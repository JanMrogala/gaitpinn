import json

from gaitpinn import runs


def test_billing_matches_lumi_examples():
    # LUMI docs: 1 GCD, 24 h, 128 GB -> 24 GPU-hours; 2 GCDs, 24 h -> 24 GPU-hours.
    assert runs.billed_gpu_hours(24 * 3600, gpus=1, cpus=7, mem_gb=128) == 24
    assert runs.billed_gpu_hours(24 * 3600, gpus=2, cpus=14, mem_gb=60) == 24
    assert runs.billed_gpu_hours(3600, gpus=1, cpus=8, mem_gb=64) == 0.5


def test_index_round_trip(tmp_path):
    rec = {
        "run_id": "smoke-base-none-s0-jlocal",
        "kind": "smoke",
        "model": "base",
        "person": "none",
        "seed": 0,
        "commit": "abc",
        "slurm_job": None,
        "gpu_hours": None,
    }
    runs.write_run(tmp_path, rec)
    assert json.loads((tmp_path / rec["run_id"] / "run.json").read_text()) == rec
    lines = runs.write_index(tmp_path).read_text().splitlines()
    assert lines[0].startswith("run_id,") and lines[1].startswith(rec["run_id"])


def test_fill_gpu_hours_from_sacct(tmp_path):
    rec = {"run_id": "r1", "kind": "smoke", "slurm_job": "123", "gpu_hours": None}
    runs.write_run(tmp_path, rec)
    n = runs.fill_gpu_hours(
        tmp_path, "123|COMPLETED|720|billing=8,cpu=7,gres/gpu=1,mem=60G,node=1\n"
    )
    assert n == 1
    assert json.loads((tmp_path / "r1" / "run.json").read_text())["gpu_hours"] == 0.1
