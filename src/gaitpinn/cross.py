"""Study 2 (PLAN2.md): one model for many people, tested on people it never saw.

Run kinds (one Slurm array task each):
- `xtune --fold F --model M --cfg I`: one grid configuration, seed 100, scored on the fold's
  validation people.
- `xcal --fold F --chosen J`: trains the fold's chosen `base` and `pinn` (seed 100), scores every
  self-calibration setting on the fold's validation people (label-free path) and records the
  calibration diagnostic.
- `xfinal --fold F --model M --seed S --chosen J`: trains the fold's chosen configuration and
  writes one run record per test person for M and for its calibrated version (`pinn_cal` or
  `base_cal`).

All choices are nested in a fold. Test people's labels are read only by `score_stream`.
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
from gaitpinn.train import Person

N_IN = len(data.EMG) + len(data.IMU)
TUNE_SEED = 100
N_FOLDS = 4
N_VAL = 3
CAL_WALK_S, CAL_SKIP_S = 120.0, 10.0
CAL_LR, CAL_BATCH = 1e-4, 64
CAL_GRID = {"steps": [100, 300], "mu": [1.0, 10.0]}
CAL_TERMS = {"pinn": ["gyro", "gyro+swing"], "base": ["gyro"]}
GATE_R = 0.8
ALL_MODES = [*data.WALKING, data.RAMP, data.STAIRS]
SKIP = int(data.SKIP_S * data.RATE)
PAR_KEYS = ("m", "d", "I_c", "l_t", "scale_dyn", "scale_tau", "tau_scale")


# ---------- people and folds ----------


def folds(people: list[str], seed: int = 0, n_val: int = N_VAL) -> list[dict]:
    """4 folds of test people; per fold n_val validation people and the rest for training."""
    people = sorted(people)
    rng = np.random.default_rng(seed)
    order = [people[i] for i in rng.permutation(len(people))]
    sizes = [len(order) // N_FOLDS + (1 if i < len(order) % N_FOLDS else 0) for i in range(N_FOLDS)]
    out, start = [], 0
    for f, n in enumerate(sizes):
        test = sorted(order[start : start + n])
        start += n
        rest = sorted(p for p in people if p not in test)
        r = np.random.default_rng(seed + 1 + f)
        val = sorted(rest[i] for i in r.permutation(len(rest))[:n_val])
        train = sorted(p for p in rest if p not in val)
        out.append({"fold": f, "test": test, "val": val, "train": train})
    return out


def calibration_ids(split: dict, modes: dict) -> list[str]:
    tread = sorted(k for k in split["train"] if modes[k] == "treadmill")
    return tread or sorted(split["train"])


def calibration_walk(h: h5py.File, split: dict, modes: dict, key: str = "x") -> list[np.ndarray]:
    """120 s of the first treadmill walking train file from 10 s in; level-ground train files if
    there is no treadmill train file. key='x' (sensors) everywhere; key='y' only for the
    validation-people diagnostic."""
    out, left = [], int(CAL_WALK_S * data.RATE)
    for k in calibration_ids(split, modes):
        if left <= 0:
            break
        a = h["files"][k][key][int(CAL_SKIP_S * data.RATE) :][:left].astype(np.float64)
        out.append(a)
        left -= len(a)
    return out


def walk_stats(walk: list[np.ndarray]) -> dict:
    """Label-free per-person statistics from the calibration walk (one recipe for everyone)."""
    xall = np.concatenate(walk)
    return {
        "emg_peak": np.percentile(xall[:, swing.muscle_index()], 95, axis=0),
        "swing_thr": swing.swing_threshold(walk),
    }


class Labelled(Person):
    """A training or validation person. Study 1's label-fitted gyro alignment is kept (labels
    are allowed for them); EMG scaling, swing threshold and gyro scales follow the shared
    label-free recipe."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.path = path
        walk = calibration_walk(self.h, self.split, self.modes)
        st = walk_stats(walk)
        self.emg_peak, self.swing_thr = st["emg_peak"], st["swing_thr"]
        ref = np.concatenate([gyro.reference(x, self.align) for x in walk])
        self.gyro_scale = {j: float(ref[:, i].std()) for i, j in enumerate(gyro.JOINTS)}
        self._cache = {}  # extras must use the recipe above

    def all_ids(self) -> list[str]:
        return sorted(k for k, m in self.modes.items() if m in ALL_MODES)

    def norm(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def physics(self, ratio: float) -> dict | None:
        """Per-person loss parameters, or None for people without physics (gyro check)."""
        if not self.gyro_ok:
            return None
        return {
            "inv": [1 / self.gyro_scale[j] for j in gyro.JOINTS],
            **{k: self.body[k] for k in ("m", "d", "I_c", "l_t")},
            "scale_dyn": self.tau_scale * ratio,
            "scale_tau": self.tau_scale,
            "tau_scale": self.tau_scale,
        }


class Unlabelled:
    """A person seen only through sensor signals (test people; validation people during the
    calibration choice). Targets are read only by `score_stream`."""

    def __init__(self, path: Path):
        self.path = path
        self.h = h5py.File(path, "r")
        self.name = self.h.attrs["person"]
        self.modes = data.file_modes(self.h)
        self.split = data.split_files(self.modes)
        self.x_mean, self.x_std = data.norm_stats(self.h, self.split["train"])
        self.body = swing.body(self.h.attrs["height_m"], self.h.attrs["mass_kg"])
        self.tau_scale = self.body["m"] * swing.G * self.body["d"]

    def norm(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)


# ---------- model ----------


class XModel(torch.nn.Module):
    """Network on per-person normalized inputs; one global target normalization."""

    def __init__(self, cfg: dict, y_mean, y_std):
        super().__init__()
        n_out = 4 if cfg["model"].startswith("pinn") else 3
        self.net = Net(N_IN, n_out, cfg["width"], cfg["levels"], cfg["kernel"], cfg["dropout"])
        f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None, :, None]  # noqa: E731
        self.register_buffer("y_mean", f(y_mean))
        self.register_buffer("y_std", f(y_std))

    def forward(self, xn: torch.Tensor):
        out = self.net(xn)
        tau_n = out[:, 3] if out.shape[1] > 3 else None
        return out[:, :3] * self.y_std + self.y_mean, tau_n, out[:, :3]


def predict(model: XModel, xn: np.ndarray, dev) -> np.ndarray:
    with torch.no_grad():
        a, _, _ = model(torch.tensor(xn.T[None], device=dev))
    return a[0].T.cpu().numpy().astype(np.float64)


def par_tensors(par: dict, B: int, dev) -> dict:
    out = {"inv": torch.tensor([par["inv"]] * B, dtype=torch.float32, device=dev)}
    for k in PAR_KEYS:
        out[k] = torch.full((B, 1), float(par[k]), device=dev)
    return out


# ---------- training on many people ----------


def batches(people: list[Labelled], cfg: dict, ratio: float, rng, dev):
    """Chunks: a person in proportion to their data, then a file by length, then a chunk."""
    n = int(round(cfg["chunk_s"] * data.RATE))
    files = {p.name: p.all_ids() for p in people}
    lens = {
        p.name: np.array([len(p.h["files"][k]["t"]) for k in files[p.name]], float) for p in people
    }
    pw = np.array([lens[p.name].sum() for p in people])
    phys = {p.name: p.physics(ratio) for p in people}
    while True:
        cols = {k: [] for k in ("x", "y", "ref", "phd", "act", "swing", "score", "has")}
        par = {k: [] for k in ("inv", *PAR_KEYS)}
        for pi in rng.choice(len(people), size=cfg["batch_size"], p=pw / pw.sum()):
            p = people[pi]
            ln = lens[p.name]
            k = files[p.name][rng.choice(len(ln), p=ln / ln.sum())]
            e = p.extras(k)
            T = len(e["x"])
            s = int(rng.integers(0, T - n + 1)) if T >= n else 0
            sl = slice(s, min(s + n, T))
            pad = n - (sl.stop - sl.start)

            def take(a, pad=pad):
                return np.pad(a, [(pad, 0)] + [(0, 0)] * (a.ndim - 1))

            cols["x"].append(take(p.norm(e["x"][sl])))
            for key in ("y", "ref", "phd", "act", "swing"):
                cols[key].append(take(e[key][sl]))
            sc = np.zeros(n, bool)
            sc[pad:] = np.arange(sl.start, sl.stop) >= SKIP
            cols["score"].append(sc)
            ph = phys[p.name]
            cols["has"].append(ph is not None)
            for key in par:
                par[key].append(ph[key] if ph else ([1.0, 1.0] if key == "inv" else 1.0))
        t = {k: torch.tensor(np.stack(v)).to(dev) for k, v in cols.items()}
        for k in ("x", "y", "ref", "act"):
            t[k] = t[k].transpose(1, 2)
        t["par"] = {
            k: torch.tensor(np.array(v), dtype=torch.float32, device=dev) for k, v in par.items()
        }
        for k in PAR_KEYS:
            t["par"][k] = t["par"][k][:, None]
        yield t


def physics_terms(gterm, sterm, angles, tau_n, b, mask, par, has):
    """(gyro, swing) losses over samples of people with physics (has); per-sample parameters."""
    m = mask & has[:, None]
    zero = torch.zeros((), device=angles.device)
    if not m.any():
        return zero, zero
    lg = gterm(angles, b["ref"], m, inv=par["inv"])
    if sterm is None or tau_n is None:
        return lg, zero
    ls = sterm(angles, tau_n * par["tau_scale"], b["phd"], b["act"], b["swing"] & m, body=par)
    return lg, ls


def files_rmse(model, people, dev) -> float:
    """Mean over people of each person's angle RMSE over all their files (mean of 3 joints)."""
    model.eval()
    per = []
    for p in people:
        if not hasattr(p, "_files"):  # cache normalized inputs and targets once
            ids = sorted(k for k, m in p.modes.items() if m in ALL_MODES)
            p._files = [
                (
                    p.norm(p.h["files"][k]["x"][:].astype(np.float64)),
                    p.h["files"][k]["y"][:].astype(np.float64),
                )
                for k in ids
            ]
        errs, masks = [], []
        for xn, y in p._files:
            errs.append(predict(model, xn, dev) - y)
            m = np.ones(len(y), bool)
            m[:SKIP] = False
            masks.append(m)
        per.append(metrics.rmse(np.concatenate(errs), np.concatenate(masks)).mean())
    return float(np.mean(per))


def swing_ratio(train_p: list[Labelled]) -> float:
    """R = median over training people (with physics) of true-angle swing scale / tau_scale."""
    r = [p.swing_scale / p.tau_scale for p in train_p if p.gyro_ok]
    return float(np.median(r)) if r else 1.0


def fit(cfg, train_p, val_p, seed, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.concatenate([p.load(k)["y"] for p in train_p for k in p.all_ids()])
    model = XModel(cfg, y.mean(0), y.std(0)).to(dev)
    ratio = swing_ratio(train_p)
    params = list(model.parameters())
    gterm = sterm = None
    if cfg["model"].startswith("pinn"):
        gterm = gyro.GyroTerm({j: 1.0 for j in gyro.JOINTS}).to(dev)
        sterm = swing.SwingTerm(train_p[0].body, 1.0, 1.0, cfg["swing_init"]).to(dev)
        params += list(sterm.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = model.net.receptive_field - 1
    best, bad, hist, best_state, best_swing = np.inf, 0, [], None, None
    gen = batches(train_p, cfg, ratio, rng, dev)
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot = {"angle": 0.0, "gyro": 0.0, "swing": 0.0}
        for _ in range(cfg["steps_per_epoch"]):
            b = next(gen)
            angles, tau_n, out_n = model(b["x"])
            m = b["score"].clone()
            m[:, :warm] = False
            yn = (b["y"] - model.y_mean) / model.y_std
            mf = m[:, None, :].float()
            loss = (((out_n - yn) ** 2) * mf).sum() / (mf.sum() * 3).clamp(min=1)
            tot["angle"] += loss.item()
            if gterm is not None:
                lg, ls = physics_terms(gterm, sterm, angles, tau_n, b, m, b["par"], b["has"])
                loss = loss + cfg["lam"] * (lg + ls)
                tot["gyro"] += lg.item()
                tot["swing"] += ls.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        v = files_rmse(model, val_p, dev)
        hist.append(
            {
                "epoch": epoch,
                "val_rmse": v,
                **{k: x / cfg["steps_per_epoch"] for k, x in tot.items()},
            }
        )
        if v < best - 1e-4:
            best, bad, best_state = v, 0, copy.deepcopy(model.state_dict())
            best_swing = copy.deepcopy(sterm.state_dict()) if sterm is not None else None
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    if sterm is not None:
        sterm.load_state_dict(best_swing)
    model.eval()
    return (
        model,
        sterm,
        ratio,
        {
            "val_rmse": best,
            "epochs": len(hist),
            "train_seconds": round(time.time() - t0, 1),
            "history": hist,
        },
    )


# ---------- label-free self-calibration ----------


def label_free_setup(model, tp: Unlabelled, ratio: float, dev) -> dict:
    """Physics inputs for a person from sensor signals and the model's OWN predictions."""
    walk = calibration_walk(tp.h, tp.split, tp.modes)
    preds = [predict(model, tp.norm(x), dev) for x in walk]
    st = walk_stats(walk)
    xall = np.concatenate(walk)
    axes = {seg: gyro.principal_axis(xall[:, gyro.gyro_index(seg)]) for seg in gyro.SEGMENTS}
    sp = {
        seg: np.concatenate([gyro.segment_speed(x, seg, axes[seg]) for x in walk])
        for seg in gyro.SEGMENTS
    }
    pred_speed = np.concatenate([gyro.angle_speed(p) for p in preds])
    best = None
    for signs in itertools.product((1, -1), repeat=3):  # one sign per sensor, shared by joints
        s = dict(zip(gyro.SEGMENTS, signs, strict=True))
        corr = {}
        for joint, (i, prox, dist) in gyro.JOINTS.items():
            w = gyro.smooth(np.degrees(s[dist] * sp[dist] - s[prox] * sp[prox]))
            corr[joint] = float(np.corrcoef(w, pred_speed[:, i])[0, 1])
        if best is None or sum(corr.values()) > sum(best[1].values()):
            best = (s, corr)
    signs, corr = best
    align = {"axes": {k: v.tolist() for k, v in axes.items()}, "joints": {}}
    for joint, (_, prox, dist) in gyro.JOINTS.items():
        align["joints"][joint] = {
            "axis_prox": axes[prox].tolist(),
            "axis_dist": axes[dist].tolist(),
            "sign_prox": signs[prox],
            "sign_dist": signs[dist],
        }
    gate = {j: bool(corr[j] >= GATE_R) for j in gyro.JOINTS}
    refs = [gyro.reference(x, align) for x in walk]
    scale = np.concatenate(refs).std(0)
    inv = [(1 / scale[i]) if gate[j] else 0.0 for i, j in enumerate(gyro.JOINTS)]
    hip = np.radians(pred_speed[:, 0])
    tsign = 1.0 if np.corrcoef(sp["thigh"], hip)[0, 1] >= 0 else -1.0
    mi = swing.muscle_index()
    segs = [
        {
            "x": tp.norm(x),
            "ref": r.astype(np.float32),
            "phd": (tsign * gyro.segment_speed(x, "thigh", axes["thigh"])).astype(np.float32),
            "act": swing.activations(x[:, mi], st["emg_peak"]).astype(np.float32),
            "swing": swing.swing_mask(x, st["swing_thr"]),
        }
        for x, r in zip(walk, refs, strict=True)
    ]
    par = {
        "inv": inv,
        **{k: tp.body[k] for k in ("m", "d", "I_c", "l_t")},
        "scale_dyn": tp.tau_scale * ratio,
        "scale_tau": tp.tau_scale,
        "tau_scale": tp.tau_scale,
    }
    return {"segs": segs, "par": par, "gate": gate, "corr": corr, "signs": signs}


def _cal_batch(segs, rng, dev):
    n = int(4.0 * data.RATE)
    lens = np.array([len(s["x"]) for s in segs], float)
    cols = {k: [] for k in ("x", "ref", "phd", "act", "swing")}
    for i in rng.choice(len(segs), size=CAL_BATCH, p=lens / lens.sum()):
        s = segs[i]
        a = int(rng.integers(0, max(len(s["x"]) - n, 0) + 1))
        for k in cols:
            cols[k].append(s[k][a : a + n])
    L = min(len(c) for c in cols["x"])
    t = {k: torch.tensor(np.stack([c[:L] for c in v])).to(dev) for k, v in cols.items()}
    for k in ("x", "ref", "act"):
        t[k] = t[k].transpose(1, 2)
    return t


def calibrate(model, sterm, setup, terms: str, steps: int, mu: float, seed: int, dev):
    """Fine-tune a copy on the calibration walk:
    sum over terms of L_t / L_t(start) + mu * anchor / anchor_ref. Physics parameters frozen."""
    use_swing = terms == "gyro+swing" and sterm is not None
    if not any(setup["gate"].values()) and not use_swing:
        return model  # the gate removed every gyro term: nothing to calibrate with
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cal = copy.deepcopy(model).train()
    model.eval()
    gterm = gyro.GyroTerm({j: 1.0 for j in gyro.JOINTS}).to(dev)
    if sterm is not None:
        for q in sterm.parameters():
            q.requires_grad_(False)
    opt = torch.optim.Adam(cal.parameters(), lr=CAL_LR)
    warm = cal.net.receptive_field - 1
    anchor_ref = float((1.0 / model.y_std.squeeze()).pow(2).mean())

    def losses(net, t):
        angles, tau_n, out_n = net(t["x"])
        B, L = t["x"].shape[0], t["x"].shape[2]
        m = torch.ones(B, L, dtype=torch.bool, device=dev)
        m[:, :warm] = False
        lg, ls = physics_terms(
            gterm,
            sterm if use_swing else None,
            angles,
            tau_n,
            t,
            m,
            par_tensors(setup["par"], B, dev),
            torch.ones(B, dtype=torch.bool, device=dev),
        )
        return lg, ls, out_n, m

    with torch.no_grad():  # start values of each term, on fixed batches
        r0 = np.random.default_rng(seed + 12345)
        l0 = [losses(model, _cal_batch(setup["segs"], r0, dev))[:2] for _ in range(5)]
        g0 = max(float(np.mean([a.item() for a, _ in l0])), 1e-8)
        s0 = max(float(np.mean([b.item() for _, b in l0])), 1e-8)
    for _ in range(steps):
        t = _cal_batch(setup["segs"], rng, dev)
        lg, ls, out_n, m = losses(cal, t)
        with torch.no_grad():
            _, _, ref_n = model(t["x"])
        mf = m[:, None, :].float()
        anchor = (((out_n - ref_n) ** 2) * mf).sum() / (mf.sum() * 3)
        loss = lg / g0 + mu * anchor / anchor_ref
        if use_swing:
            loss = loss + ls / s0
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cal.parameters(), 5.0)
        opt.step()
    return cal.eval()


def diagnostic(model, sterm, setup, vp: Labelled, dev) -> dict:
    """Validation people only: the calibration's physics terms at the TRUE angles and at the
    model's predictions on the calibration walk. Calibration can only help if truth scores
    lower."""
    walk = calibration_walk(vp.h, vp.split, vp.modes)
    ys = calibration_walk(vp.h, vp.split, vp.modes, key="y")
    gterm = gyro.GyroTerm({j: 1.0 for j in gyro.JOINTS}).to(dev)
    out = {}
    for name, ang in (("true", ys), ("pred", [predict(model, vp.norm(x), dev) for x in walk])):
        g, s = [], []
        for seg, a in zip(setup["segs"], ang, strict=True):
            t = {k: torch.tensor(seg[k][None]).to(dev) for k in ("ref", "phd", "act", "swing")}
            for k in ("ref", "act"):
                t[k] = t[k].transpose(1, 2)
            A = torch.tensor(a.T[None], dtype=torch.float32, device=dev)
            with torch.no_grad():
                _, tau_n, _ = model(torch.tensor(seg["x"].T[None], device=dev))
            m = torch.ones(1, A.shape[2], dtype=torch.bool, device=dev)
            m[:, :SKIP] = False
            lg, ls = physics_terms(
                gterm,
                sterm,
                A,
                tau_n,
                t,
                m,
                par_tensors(setup["par"], 1, dev),
                torch.ones(1, dtype=torch.bool, device=dev),
            )
            g.append(lg.item())
            s.append(ls.item())
        out[name] = {"gyro": float(np.mean(g)), "swing": float(np.mean(s))}
    return out


# ---------- scoring ----------


def score_stream(model, tp: Unlabelled, dev) -> tuple[dict, dict]:
    """Study-1 metrics on the person's test stream. The only place test labels are read."""
    errs, masks, lg_e, lg_m = [], [], [], []
    for k in tp.split["test"]:
        g = tp.h["files"][k]
        x, y = g["x"][:].astype(np.float64), g["y"][:].astype(np.float64)
        e = predict(model, tp.norm(x), dev) - y
        m = np.ones(len(y), bool)
        m[:SKIP] = False
        errs.append(e)
        masks.append(m)
        if tp.modes[k] == "levelground":
            lg_e.append(e)
            lg_m.append(m)
    walk = (
        np.concatenate(errs),
        np.concatenate(masks),
        {"levelground": (np.concatenate(lg_e), np.concatenate(lg_m))},
    )
    stream = data.build_stream(tp.h, tp.split)
    pred = predict(model, tp.norm(stream["x"].astype(np.float64)), dev)
    met = metrics.run_metrics(stream, pred, walk)
    keep = {
        "t": stream["t"].astype(np.float32),
        "y": stream["y"].astype(np.float32),
        "pred": pred.astype(np.float32),
        "score": stream["score"],
        "label": stream["label"].astype("S"),
        "file_idx": stream["file"].astype(np.int16),
    }
    return met, keep


# ---------- runs ----------


def grid(cfg: dict) -> list[dict]:
    keys = list(cfg["grid"])
    out = []
    for vals in itertools.product(*(cfg["grid"][k] for k in keys)):
        c = {k: v for k, v in cfg.items() if k != "grid"}
        c.update(dict(zip(keys, vals, strict=True)))
        out.append(c)
    return out


def cal_settings(model_name: str) -> list[dict]:
    out = [{"name": "none"}]
    for terms, steps, mu in itertools.product(
        CAL_TERMS[model_name], CAL_GRID["steps"], CAL_GRID["mu"]
    ):
        out.append({"name": f"{terms}-s{steps}-mu{mu:g}", "terms": terms, "steps": steps, "mu": mu})
    return out


def apply_cal(model, sterm, setup, s: dict, seed: int, dev):
    if s["name"] == "none":
        return model
    return calibrate(model, sterm, setup, s["terms"], s["steps"], s["mu"], seed, dev)


def record(kind, model_name, person, seed, code, extra) -> dict:
    return {
        "run_id": runs.run_id(kind, model_name, person, seed),
        "kind": kind,
        "status": "running",
        "model": model_name,
        "person": person,
        "seed": seed,
        "commit": runs.code_commit(code),
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
    ap.add_argument("kind", choices=["xtune", "xcal", "xfinal", "folds"])
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--model", choices=["base", "pinn"])
    ap.add_argument("--cfg", type=int, help="xtune: grid index")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chosen", type=Path, help="per-fold chosen configs and cal settings")
    ap.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--people", nargs="*", help="restrict to these people (local tests)")
    ap.add_argument("--n-val", type=int, default=N_VAL, help="validation people per fold")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    people = a.people or sorted(p.stem for p in a.h5.glob("AB*.h5"))
    fl = folds(people, n_val=a.n_val)
    if a.kind == "folds":
        print(json.dumps(fl, indent=1))
        return
    f = fl[a.fold]
    cdir = a.code / "configs"
    chosen = json.loads(a.chosen.read_text())[f"fold{a.fold}"] if a.chosen else {}
    t0 = time.time()
    train_p = [Labelled(a.h5 / f"{p}.h5") for p in f["train"]]
    val_p = [Labelled(a.h5 / f"{p}.h5") for p in f["val"]]
    extra = {
        "fold": f,
        "prep_seconds": round(time.time() - t0, 1),
        "physics_people": [p.name for p in train_p if p.gyro_ok],
    }

    def cfg_for(m):
        c = yaml.safe_load((cdir / f"x{m}.yaml").read_text())
        c = {k: v for k, v in c.items() if k != "grid"} | chosen.get(m, {})
        return c | ({"max_epochs": a.max_epochs} if a.max_epochs else {})

    if a.kind == "xtune":
        c = grid(yaml.safe_load((cdir / f"x{a.model}.yaml").read_text()))[a.cfg]
        c = c | {"tag": f"c{a.cfg:02d}"} | ({"max_epochs": a.max_epochs} if a.max_epochs else {})
        rec = record(
            "xtune",
            f"{a.model}-c{a.cfg:02d}",
            f"fold{a.fold}",
            TUNE_SEED,
            a.code,
            extra | {"config": c},
        )
        runs.write_run(a.runs, rec)
        _, _, _, out = fit(c, train_p, val_p, TUNE_SEED, dev)
        rec |= {"metrics": {"val_rmse": out["val_rmse"]}, "training": out, "status": "done"}
        runs.write_run(a.runs, rec)
        return
    if a.kind == "xcal":
        for m in ("base", "pinn"):
            c = cfg_for(m)
            rec = record("xcal", m, f"fold{a.fold}", TUNE_SEED, a.code, extra | {"config": c})
            runs.write_run(a.runs, rec)
            model, sterm, ratio, out = fit(c, train_p, val_p, TUNE_SEED, dev)
            setups = {
                vp.name: label_free_setup(model, Unlabelled(vp.path), ratio, dev) for vp in val_p
            }
            gates = {
                vp.name: {
                    "gate": setups[vp.name]["gate"],
                    "corr": setups[vp.name]["corr"],
                    "true_check_ok": bool(vp.gyro_ok),
                }
                for vp in val_p
            }
            diag = (
                {vp.name: diagnostic(model, sterm, setups[vp.name], vp, dev) for vp in val_p}
                if m == "pinn"
                else {}
            )
            res = {}
            for s in cal_settings(m):
                errs = [
                    files_rmse(
                        apply_cal(model, sterm, setups[vp.name], s, TUNE_SEED, dev), [vp], dev
                    )
                    for vp in val_p
                ]
                res[s["name"]] = float(np.mean(errs))
            rec |= {
                "metrics": {"val_rmse": out["val_rmse"], "cal": res},
                "gates": gates,
                "diagnostic": diag,
                "training": out,
                "status": "done",
            }
            runs.write_run(a.runs, rec)
        return
    # xfinal
    c = cfg_for(a.model)
    model, sterm, ratio, out = fit(c, train_p, val_p, a.seed, dev)
    cal_name = "pinn_cal" if a.model == "pinn" else "base_cal"
    cal_set = chosen.get(f"cal_{a.model}", {"name": "none"})
    for name in f["test"]:
        tp = Unlabelled(a.h5 / f"{name}.h5")
        setup = label_free_setup(model, tp, ratio, dev)
        variants = [
            (a.model, model, None),
            (cal_name, apply_cal(model, sterm, setup, cal_set, a.seed, dev), cal_set),
        ]
        for mname, mdl, cs in variants:
            info = {
                "config": c,
                "calibration": cs,
                "gate": setup["gate"] if cs else None,
                "gate_corr": setup["corr"],
            }
            rec = record("xfinal", mname, name, a.seed, a.code, extra | info)
            d = runs.write_run(a.runs, rec)
            met, keep = score_stream(mdl, tp, dev)
            rec |= {
                "metrics": {"val_rmse": out["val_rmse"], **met},
                "status": "done",
                "training": {k: v for k, v in out.items() if k != "history"},
                "person_info": {"gyro_ok": True},
            }
            np.savez_compressed(d / "stream.npz", **keep)
            runs.write_run(a.runs, rec)
            print(rec["run_id"], "done", flush=True)  # no stair numbers in logs


if __name__ == "__main__":
    main()
