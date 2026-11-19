"""Train `base` or `pinn` for one person, and write one run record per (config, seed).

Kinds of run:
- `smoke`: synthetic signals, a few steps; checks GPU, container and the run-record round trip.
- `tune`: every configuration in the model's grid, seed 100, scored on walking validation only.
  Tuning runs never load, predict or score stair files.
- `final`: the chosen configuration (from --chosen), given seeds; predicts the walking test
  files and the test stream and stores the per-run metrics (PLAN.md).

All calibration (normalization, gyro alignment, swing threshold, EMG scaling) is fitted on the
person's walking training files only.
"""

import argparse
import copy
import itertools
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from gaitpinn import data, gyro, runs, swing
from gaitpinn import test as metrics
from gaitpinn.model import Net

N_IN = len(data.EMG) + len(data.IMU)
TUNE_SEED = 100


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- per-person preparation (walking training data only) ----------


class Person:
    def __init__(self, path: Path):
        self.h = h5py.File(path, "r")
        self.name = self.h.attrs["person"]
        self.modes = data.file_modes(self.h)
        self.split = data.split_files(self.modes)
        self.dropped = [k for k in self.h["files"] if k not in self.modes]
        train = self.split["train"]
        data.check_no_stairs(train + self.split["val"] + self.split["test"], self.modes)
        tr = [self.load(k) for k in train]
        self.x_mean, self.x_std = data.norm_stats(self.h, train)
        y = np.concatenate([f["y"] for f in tr])
        self.y_mean, self.y_std = y.mean(0), y.std(0)
        pairs = [(f["x"], f["y"]) for f in tr]
        self.align = gyro.fit_alignment(pairs)
        self.gyro_scale = gyro.scales(pairs)
        self.gyro_check = gyro.check(
            [(f["x"], f["y"]) for f in map(self.load, self.split["val"])], self.align
        )
        self.gyro_ok = gyro.passes(self.gyro_check)
        # thigh angular speed sign: match d(hip)/dt on training files
        thigh = np.concatenate(
            [gyro.segment_speed(f["x"], "thigh", self.align["axes"]["thigh"]) for f in tr]
        )
        hip_speed = np.radians(np.concatenate([gyro.angle_speed(f["y"])[:, 0] for f in tr]))
        self.thigh_sign = 1.0 if np.corrcoef(thigh, hip_speed)[0, 1] >= 0 else -1.0
        self.swing_thr = swing.swing_threshold([f["x"] for f in tr])
        mi = swing.muscle_index()
        self.emg_peak = np.percentile(np.concatenate([f["x"][:, mi] for f in tr]), 95, axis=0)
        self.body = swing.body(self.h.attrs["height_m"], self.h.attrs["mass_kg"])
        self.tau_scale = self.body["m"] * swing.G * self.body["d"]  # N m
        self._cache = {}
        # unit-free scales of the swing residuals, from true knee motion during swing
        self.swing_scale = self._swing_scale(tr)

    def load(self, k: str) -> dict:
        g = self.h["files"][k]
        return {"x": g["x"][:].astype(np.float64), "y": g["y"][:].astype(np.float64)}

    def extras(self, k: str) -> dict:
        """Physics inputs for one file (computed once): gyro reference, thigh speed, swing
        mask, EMG activations."""
        if k not in self._cache:
            f = self.load(k)
            x = f["x"]
            self._cache[k] = {
                "x": x.astype(np.float32),
                "y": f["y"].astype(np.float32),
                "ref": gyro.reference(x, self.align).astype(np.float32),
                "phd": (
                    self.thigh_sign * gyro.segment_speed(x, "thigh", self.align["axes"]["thigh"])
                ).astype(np.float32),
                "swing": swing.swing_mask(x, self.swing_thr),
                "act": swing.activations(x[:, swing.muscle_index()], self.emg_peak).astype(
                    np.float32
                ),
            }
        return self._cache[k]

    def _swing_scale(self, tr) -> float:
        lhs = swing.lhs_function()
        vals = []
        for f in tr:
            q = -np.radians(f["y"][:, 1])
            ph = np.radians(f["y"][:, 0])
            qd = np.gradient(q) * data.RATE
            qdd = np.gradient(qd) * data.RATE
            phd = np.gradient(ph) * data.RATE
            phdd = np.gradient(phd) * data.RATE
            m = swing.swing_mask(f["x"], self.swing_thr)
            p = self.body
            v = lhs(
                *(torch.tensor(a[m]) for a in (q, qd, qdd, ph, phd, phdd)),
                p["m"],
                p["d"],
                p["I_c"],
                p["l_t"],
                swing.G,
            )
            vals.append(v.numpy())
        return float(np.concatenate(vals).std())

    def speed_check(self) -> dict:
        """Validation/test treadmill speeds must lie inside the training speed range."""

        def rng(part):
            a = [self.h["files"][k].attrs for k in self.split[part] if self.modes[k] == "treadmill"]
            return min(x["speed_min"] for x in a), max(x["speed_max"] for x in a)

        tr, va, te = rng("train"), rng("val"), rng("test")
        ok = all(tr[0] <= r[0] and r[1] <= tr[1] for r in (va, te))
        return {"train": tr, "val": va, "test": te, "ok": bool(ok)}

    def info(self) -> dict:
        return {
            "speed_check": self.speed_check(),
            "files": {p: len(v) for p, v in self.split.items()},
            "dropped_files": self.dropped,
            "gyro_check_val": self.gyro_check,
            "gyro_ok": bool(self.gyro_ok),
            "swing_threshold": self.swing_thr,
        }


# ---------- model with normalization ----------


class Model(torch.nn.Module):
    def __init__(self, cfg: dict, p: Person):
        super().__init__()
        n_out = 4 if cfg["model"] == "pinn" else 3
        self.net = Net(N_IN, n_out, cfg["width"], cfg["levels"], cfg["kernel"], cfg["dropout"])
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.register_buffer("x_mean", f32(p.x_mean)[None, :, None])
        self.register_buffer("x_std", f32(p.x_std)[None, :, None])
        self.register_buffer("y_mean", f32(p.y_mean)[None, :, None])
        self.register_buffer("y_std", f32(p.y_std)[None, :, None])
        self.tau_scale = p.tau_scale

    def forward(self, x_raw: torch.Tensor):
        """x_raw (B, C, T) -> angles (B, 3, T) deg, torque (B, T) N m or None."""
        out = self.net((x_raw - self.x_mean) / self.x_std)
        angles = out[:, :3] * self.y_std + self.y_mean
        tau = out[:, 3] * self.tau_scale if out.shape[1] > 3 else None
        return angles, tau, out[:, :3]


# ---------- training ----------


def batches(p: Person, ids: list[str], cfg: dict, rng: np.random.Generator, dev):
    """Random chunks of chunk_s seconds from within single files (lengths-weighted)."""
    n = int(round(cfg["chunk_s"] * data.RATE))
    lens = np.array([len(p.extras(k)["x"]) for k in ids], float)
    prob = lens / lens.sum()
    while True:
        cols = {k: [] for k in ("x", "y", "ref", "phd", "swing", "act", "score")}
        for i in rng.choice(len(ids), size=cfg["batch_size"], p=prob):
            e = p.extras(ids[i])
            T = len(e["x"])
            if T >= n:
                s = int(rng.integers(0, T - n + 1))
                sl = slice(s, s + n)
            else:
                s, sl = 0, slice(0, T)
            score = np.ones(n, bool)
            pad = n - (sl.stop - sl.start)
            # first 0.5 s of a file: context only
            first = np.arange(sl.start, sl.stop) < int(data.SKIP_S * data.RATE)
            sc = ~first
            for key in ("x", "y", "ref", "phd", "act"):
                a = e[key][sl]
                cols[key].append(np.pad(a, [(pad, 0)] + [(0, 0)] * (a.ndim - 1)))
            cols["swing"].append(np.pad(e["swing"][sl], (pad, 0)))
            score[pad:] = sc
            score[:pad] = False
            cols["score"].append(score)
        t = {k: torch.tensor(np.stack(v)).to(dev) for k, v in cols.items()}
        # (B, T, C) -> (B, C, T)
        for k in ("x", "y", "ref", "act"):
            t[k] = t[k].transpose(1, 2)
        yield t


def predict(model: Model, x: np.ndarray, dev) -> np.ndarray:
    """Whole-sequence causal prediction (T, C) -> (T, 3) deg."""
    with torch.no_grad():
        a, _, _ = model(torch.tensor(x.T[None], dtype=torch.float32, device=dev))
    return a[0].T.cpu().numpy().astype(np.float64)


def evaluate_files(model, p: Person, ids: list[str], dev) -> tuple[np.ndarray, np.ndarray]:
    """Errors (pred - true) and score masks over files, concatenated."""
    errs, masks = [], []
    for k in ids:
        e = p.extras(k)
        errs.append(predict(model, e["x"], dev) - e["y"])
        m = np.ones(len(e["x"]), bool)
        m[: int(data.SKIP_S * data.RATE)] = False
        masks.append(m)
    return np.concatenate(errs), np.concatenate(masks)


def val_rmse(model, p: Person, dev) -> float:
    model.eval()
    e, m = evaluate_files(model, p, p.split["val"], dev)
    model.train()
    return float(metrics.rmse(e, m).mean())


def fit(cfg: dict, p: Person, seed: int, dev) -> tuple[Model, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Model(cfg, p).to(dev)
    params = list(model.parameters())
    gterm = sterm = None
    if cfg["model"] == "pinn":
        gterm = gyro.GyroTerm(p.gyro_scale).to(dev)
        sterm = swing.SwingTerm(p.body, p.swing_scale, p.tau_scale, cfg["swing_init"]).to(dev)
        params += list(sterm.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = model.net.receptive_field - 1
    best, best_state, bad, hist = np.inf, None, 0, []
    gen = batches(p, p.split["train"], cfg, rng, dev)
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        tot = {"angle": 0.0, "gyro": 0.0, "swing": 0.0}
        for _ in range(cfg["steps_per_epoch"]):
            b = next(gen)
            angles, tau, out_n = model(b["x"])
            m = b["score"].clone()
            m[:, :warm] = False
            yn = (b["y"] - model.y_mean) / model.y_std
            mf = m[:, None, :].float()
            loss_a = (((out_n - yn) ** 2) * mf).sum() / (mf.sum() * 3).clamp(min=1)
            loss = loss_a
            tot["angle"] += loss_a.item()
            if gterm is not None:
                lg = gterm(angles, b["ref"], m)
                ls = sterm(angles, tau, b["phd"], b["act"], b["swing"] & m)
                loss = loss + cfg["lam"] * (lg + ls)
                tot["gyro"] += lg.item()
                tot["swing"] += ls.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        v = val_rmse(model, p, dev)
        hist.append(
            {
                "epoch": epoch,
                "val_rmse": v,
                **{k: x / cfg["steps_per_epoch"] for k, x in tot.items()},
            }
        )
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    out = {
        "val_rmse": best,
        "epochs": len(hist),
        "train_seconds": round(time.time() - t0, 1),
        "history": hist,
    }
    if sterm is not None:
        out["swing_params"] = {
            "w": dict(
                zip(
                    swing.MUSCLES,
                    (torch.nn.functional.softplus(sterm.w_raw) * sterm.sign).tolist(),
                    strict=True,
                )
            ),
            "k": torch.nn.functional.softplus(sterm.k_raw).item(),
            "b": torch.nn.functional.softplus(sterm.b_raw).item(),
            "q0": sterm.q0.item(),
        }
    return model, out


def test_run(model: Model, p: Person, dev) -> tuple[dict, dict]:
    """Final runs only: walking test errors and the test stream."""
    e_all, m_all = evaluate_files(model, p, p.split["test"], dev)
    lg = [k for k in p.split["test"] if p.modes[k] == "levelground"]
    e_lg, m_lg = evaluate_files(model, p, lg, dev)
    stream = data.build_stream(p.h, p.split)
    pred = predict(model, stream["x"].astype(np.float64), dev)
    met = metrics.run_metrics(stream, pred, (e_all, m_all, {"levelground": (e_lg, m_lg)}))
    keep = {
        "t": stream["t"].astype(np.float32),
        "y": stream["y"].astype(np.float32),
        "pred": pred.astype(np.float32),
        "score": stream["score"],
        "label": stream["label"].astype("S"),
        "file_idx": stream["file"].astype(np.int16),
    }
    return met, keep


# ---------- configurations ----------


def grid(cfg: dict) -> list[dict]:
    keys = list(cfg["grid"])
    out = []
    for vals in itertools.product(*(cfg["grid"][k] for k in keys)):
        c = {k: v for k, v in cfg.items() if k != "grid"}
        c.update(dict(zip(keys, vals, strict=True)))
        out.append(c)
    return out


def record(kind, cfg, person, seed, code, extra) -> dict:
    tag = cfg.get("tag", "")
    rid = runs.run_id(kind, cfg["model"] + (f"-{tag}" if tag else ""), person, seed)
    return {
        "run_id": rid,
        "kind": kind,
        "status": "running",
        "model": cfg["model"],
        "person": person,
        "seed": seed,
        "commit": runs.code_commit(code),
        "config": {k: v for k, v in cfg.items() if k != "grid"},
        "slurm_job": runs.slurm_job(),
        "gpu_hours": None,
        "metrics": None,
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
        },
        **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--person", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--kind", choices=["smoke", "check", "tune", "final"], default="final")
    ap.add_argument("--chosen", type=Path, help="final runs: JSON with the chosen config")
    ap.add_argument("--h5", type=Path, help="folder with <person>.h5")
    ap.add_argument("--runs", type=Path, required=True, help="folder for run records")
    ap.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=200, help="smoke run only")
    ap.add_argument("--max-epochs", type=int, help="override (for quick local checks)")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    if a.smoke or a.kind == "smoke":
        return smoke_main(cfg, a)
    dev = pick_device()
    p = Person(a.h5 / f"{a.person}.h5")
    if a.kind == "check":  # walking data only: split, dropped files, speed and gyro checks
        print(json.dumps({"person": p.name, **p.info()}), flush=True)
        return None
    if a.kind == "tune":
        todo = [(i, c, TUNE_SEED) for i, c in enumerate(grid(cfg))]
    else:
        chosen = json.loads(a.chosen.read_text()) if a.chosen else {}
        c = {k: v for k, v in cfg.items() if k != "grid"} | chosen
        todo = [(None, c, s) for s in (a.seeds if a.seeds else [a.seed])]
    for i, c, seed in todo:
        if a.max_epochs:
            c = c | {"max_epochs": a.max_epochs}
        if i is not None:
            c = c | {"tag": f"c{i:02d}"}
        rec = record(a.kind, c, p.name, seed, a.code, {"person_info": p.info()})
        out_dir = runs.write_run(a.runs, rec)
        model, fit_out = fit(c, p, seed, dev)
        rec["metrics"] = {"val_rmse": fit_out["val_rmse"]}
        rec["training"] = {k: v for k, v in fit_out.items() if k != "val_rmse"}
        if a.kind == "final":
            met, keep = test_run(model, p, dev)
            rec["metrics"].update(met)
            np.savez_compressed(out_dir / "stream.npz", **keep)
        rec["status"] = "done"
        runs.write_run(a.runs, rec)
        print(
            f"{rec['run_id']}: val {fit_out['val_rmse']:.2f} deg, "
            f"{fit_out['epochs']} epochs, {fit_out['train_seconds']} s",
            flush=True,
        )


# ---------- smoke ----------


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
    first = None
    for _ in range(steps):
        idx = torch.randint(0, len(x), (cfg["batch_size"],), generator=gen)
        loss = ((net(x[idx].to(dev)) - y[idx].to(dev))[..., rf:] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
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


def smoke_main(cfg: dict, a) -> None:
    rec = record("smoke", cfg, a.person, a.seed, a.code, {})
    out = runs.write_run(a.runs, rec)
    rec["metrics"] = smoke(cfg, a.seed, a.steps)
    rec["status"] = "done"
    runs.write_run(a.runs, rec)
    print(f"wrote {out / 'run.json'}")


if __name__ == "__main__":
    main()
