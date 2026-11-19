"""Train one model (base or pinn) for one person and seed, and write its run record.

`--smoke` trains on synthetic signals for a few steps to check the GPU, the container and the
run-record round trip. Smoke runs are marked `kind: smoke` and never appear in results.
"""

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from gaitpinn import runs
from gaitpinn.model import Net

N_IN = 35  # 11 EMG + 24 IMU channels (Camargo, right leg)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthetic(n: int, t: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random sensor-like inputs and targets that are a causal function of them."""
    x = torch.randn(n, N_IN, t, generator=gen)
    kernel = torch.ones(1, 1, 20) / 20
    smooth = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x[:, :3].reshape(-1, 1, t), (19, 0)), kernel
    ).reshape(n, 3, t)
    return x, 10 * smooth


def smoke(cfg: dict, seed: int, steps: int = 200) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = pick_device()
    net = Net(N_IN, 3, cfg["width"], cfg["levels"], cfg["kernel"], cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    x, y = synthetic(256, 400, gen)
    xv, yv = synthetic(64, 400, gen)
    rf = net.receptive_field
    t0 = time.time()
    for i in range(steps):
        idx = torch.randint(0, len(x), (cfg["batch_size"],), generator=gen)
        xb, yb = x[idx].to(dev), y[idx].to(dev)
        loss = ((net(xb) - yb)[..., rf:] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    net.eval()
    with torch.no_grad():
        val = ((net(xv.to(dev)) - yv.to(dev))[..., rf:] ** 2).mean().sqrt().item()
    return {
        "synthetic": True,
        "steps": steps,
        "train_seconds": round(time.time() - t0, 2),
        "first_loss": first,
        "last_loss": loss.item(),
        "val_rmse": val,
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--person", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=Path, required=True, help="folder for run records")
    ap.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    kind = "smoke" if a.smoke else "final"
    if not a.smoke:
        raise NotImplementedError("real training is added in step 6 (see PLAN.md)")
    metrics = smoke(cfg, a.seed)
    rid = runs.run_id(kind, cfg["model"], a.person, a.seed)
    record = {
        "run_id": rid,
        "kind": kind,
        "model": cfg["model"],
        "person": a.person,
        "seed": a.seed,
        "commit": runs.code_commit(a.code),
        "config": cfg,
        "slurm_job": runs.slurm_job(),
        "gpu_hours": None,  # filled from sacct by jobs/fetch.sh
        "metrics": metrics,
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "host": platform.node(),
        },
    }
    out = runs.write_run(a.runs, record)
    print(f"wrote {out / 'run.json'}")


if __name__ == "__main__":
    main()
