"""Study 3 (PLAN3.md): gaps in the true angles of training and validation people.

Run kinds (one Slurm array task each):
- `gmasks`: mask statistics per share and draw (reads no angles) and every person's
  calibration-trial setup (reads only the always-kept calibration trial).
- `gtune --share S --fold F --model M --cfg I`: one grid configuration, seed (= mask draw) 100.
- `gfinal --share S --fold F --model M --seed s --chosen J`: the chosen configuration with
  seed (= mask draw) s; one run record per test person. M is base, pinn or pinn_lab.

True angles of training and validation people are read only through `GapPerson.y`, which
blanks the lost samples (NaN) right after reading. Test people are handled exactly as in study 2
(`cross.Unlabelled`, scored by `cross.score_stream`).
"""

import argparse
import copy
import json
import time
import zlib
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from gaitpinn import cross, data, gyro, runs, swing
from gaitpinn.train import Person

BLOCK = 2 * data.RATE  # samples per gap block
CAL_TRIAL_S = 30.0  # always-labelled calibration trial
SHARES = [0.0, 0.5, 0.9, 0.99]
TUNE_SEED = cross.TUNE_SEED
SKIP = cross.SKIP
MIN_PIECE = 20  # samples; shorter calibration pieces are dropped (gradient/filter need length)


# ---------- masks ----------


def block_uniform(n: int, person: str, key: str, draw: int) -> np.ndarray:
    """One uniform number per 2 s block of a file (the last, shorter block counts as one)."""
    ss = np.random.SeedSequence([int(draw), zlib.crc32(f"{person}/{key}".encode())])
    return np.random.default_rng(ss).random(-(-n // BLOCK))


def keep_mask(n: int, person: str, key: str, share: float, draw: int) -> np.ndarray:
    """True where the true angles are kept. Nested in the share for a fixed draw."""
    kb = block_uniform(n, person, key, draw) < 1.0 - share
    return np.repeat(kb, BLOCK)[:n]


def trial_segments(h: h5py.File, split: dict, modes: dict) -> list[tuple[str, int, int]]:
    """The 30 s calibration trial: the first 30 s of study 2's calibration walk."""
    out, left = [], int(CAL_TRIAL_S * data.RATE)
    s0 = int(cross.CAL_SKIP_S * data.RATE)
    for k in cross.calibration_ids(split, modes):
        if left <= 0:
            break
        T = len(h["files"][k]["t"])
        e = min(T, s0 + left)
        if e > s0:
            out.append((k, s0, e))
            left -= e - s0
    return out


def split_pieces(pieces, n_first: int):
    """Cut a list of (x, y) pieces into the first n_first samples and the rest."""
    first, rest, left = [], [], n_first
    for x, y in pieces:
        if left >= len(y):
            first.append((x, y))
        elif left > 0:
            first.append((x[:left], y[:left]))
            rest.append((x[left:], y[left:]))
        else:
            rest.append((x, y))
        left -= len(y)
    keep = lambda ps: [p for p in ps if len(p[1]) >= MIN_PIECE]  # noqa: E731
    return keep(first), keep(rest)


# ---------- people ----------


class GapPerson:
    """A training or validation person whose true angles have gaps (share, draw)."""

    def __init__(self, path: Path, share: float, draw: int, setup: bool = True):
        self.path = path
        self.h = h5py.File(path, "r")
        self.name = str(self.h.attrs["person"])
        self.modes = data.file_modes(self.h)
        self.split = data.split_files(self.modes)
        self.share, self.draw = float(share), int(draw)
        self.x_mean, self.x_std = data.norm_stats(self.h, self.split["train"])
        self.body = swing.body(self.h.attrs["height_m"], self.h.attrs["mass_kg"])
        self.tau_scale = self.body["m"] * swing.G * self.body["d"]
        self.trial = trial_segments(self.h, self.split, self.modes)
        self.keep = {}
        for k in self.all_ids():
            self.keep[k] = keep_mask(len(self.h["files"][k]["t"]), self.name, k, share, draw)
        for k, s, e in self.trial:
            self.keep[k][s:e] = True
        self._cache, self._val = {}, None
        self.fill = None  # study 4 `distill`: teacher predictions for the lost angles
        self.gyro_ok, self.setup_info = False, None
        if setup:
            self._setup()

    def all_ids(self) -> list[str]:
        return sorted(k for k, m in self.modes.items() if m in cross.ALL_MODES)

    def x(self, k: str) -> np.ndarray:
        return self.h["files"][k]["x"][:].astype(np.float64)

    def y(self, k: str) -> np.ndarray:
        """True angles with lost samples set to NaN: the only place angles are read."""
        y = self.h["files"][k]["y"][:].astype(np.float64)
        y[~self.keep[k]] = np.nan
        return y

    def norm(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def loss_keep(self, k: str) -> np.ndarray:
        """Samples with a training target: kept angles, or every sample once a teacher fills
        the lost ones (`distill`)."""
        return self.keep[k] if self.fill is None else np.ones_like(self.keep[k])

    def mask_hash(self) -> str:
        bits = np.concatenate([self.keep[k] for k in self.all_ids()])
        return f"{zlib.crc32(np.packbits(bits).tobytes()):08x}"

    def kept_seconds(self) -> dict:
        out = {}
        for k in self.all_ids():
            m = self.modes[k]
            out[m] = out.get(m, 0.0) + float(self.keep[k].sum()) / data.RATE
        return out

    def _setup(self) -> None:
        """Physics setup from the calibration trial only (PLAN3.md)."""
        walk = cross.calibration_walk(self.h, self.split, self.modes)  # sensors, 120 s
        st = cross.walk_stats(walk)
        self.emg_peak, self.swing_thr = st["emg_peak"], st["swing_thr"]
        pieces = [(self.x(k)[s:e], self.y(k)[s:e]) for k, s, e in self.trial]
        assert all(np.isfinite(y).all() for _, y in pieces), "calibration trial must be kept"
        pieces = [p for p in pieces if len(p[1]) >= MIN_PIECE]
        fit, chk = split_pieces(pieces, int(CAL_TRIAL_S / 2 * data.RATE))
        info = {"trial_s": sum(len(y) for _, y in pieces) / data.RATE}
        if not fit or not chk:
            self.setup_info = info | {"gyro_ok": False, "reason": "calibration trial too short"}
            return
        pca_x = np.concatenate([self.x(k) for k in self.split["train"]])
        self.align = gyro.fit_alignment(fit, chk, pca_x=pca_x)
        self.gyro_check = gyro.check(chk, self.align)
        ok = gyro.passes(self.gyro_check)
        thigh = np.concatenate(
            [gyro.segment_speed(x, "thigh", self.align["axes"]["thigh"]) for x, _ in pieces]
        )
        hip = np.radians(np.concatenate([gyro.angle_speed(y)[:, 0] for _, y in pieces]))
        c = float(np.corrcoef(thigh, hip)[0, 1])
        self.thigh_sign = 1.0 if c >= 0 else -1.0
        self.swing_scale = Person._swing_scale(self, [{"x": x, "y": y} for x, y in pieces])
        ref = np.concatenate([gyro.reference(x, self.align) for x in walk])
        self.gyro_scale = {j: float(ref[:, i].std()) for i, j in enumerate(gyro.JOINTS)}
        finite = np.isfinite([c, self.swing_scale, *self.gyro_scale.values()]).all()
        self.gyro_ok = bool(ok and finite)
        self.setup_info = info | {
            "gyro_ok": self.gyro_ok,
            "check": self.gyro_check,
            "method": {j: v["method"] for j, v in self.align["joints"].items()},
            "val_corr": {j: v["val_corr"] for j, v in self.align["joints"].items()},
            "thigh_corr": c,
            "swing_scale": self.swing_scale,
        }

    def physics(self, ratio: float) -> dict | None:
        if not self.gyro_ok:
            return None
        return {
            "inv": [1 / self.gyro_scale[j] for j in gyro.JOINTS],
            **{k: self.body[k] for k in ("m", "d", "I_c", "l_t")},
            "scale_dyn": self.tau_scale * ratio,
            "scale_tau": self.tau_scale,
            "tau_scale": self.tau_scale,
        }

    def extras(self, k: str) -> dict:
        """Sensors, physics inputs, and the angles with gaps (NaN) for one file."""
        if k not in self._cache:
            x = self.x(k)
            y = self.y(k)
            if self.fill is not None:  # lost angles -> teacher predictions (never true angles)
                y = np.where(self.keep[k][:, None], y, self.fill[k])
            e = {"x": self.norm(x), "y": y.astype(np.float32), "keep": self.loss_keep(k)}
            if self.gyro_ok:
                e |= {
                    "ref": gyro.reference(x, self.align).astype(np.float32),
                    "phd": (
                        self.thigh_sign
                        * gyro.segment_speed(x, "thigh", self.align["axes"]["thigh"])
                    ).astype(np.float32),
                    "swing": swing.swing_mask(x, self.swing_thr),
                    "act": swing.activations(x[:, swing.muscle_index()], self.emg_peak).astype(
                        np.float32
                    ),
                }
            self._cache[k] = e
        return self._cache[k]

    def val_files(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(normalized x, angles with gaps, scored kept mask) for files with kept samples."""
        if self._val is None:
            self._val = []
            for k in self.all_ids():
                m = self.keep[k].copy()
                m[:SKIP] = False
                if m.any():
                    self._val.append((self.norm(self.x(k)), self.y(k), m))
        return self._val


def swing_ratio(train_p: list[GapPerson]) -> float:
    r = [p.swing_scale / p.tau_scale for p in train_p if p.gyro_ok]
    return float(np.median(r)) if r else 1.0


# ---------- batches ----------

N_IMU = len(gyro.JOINTS)


class Sampler:
    """Labelled chunks around kept samples, and (pinn) unlabelled chunks from physics people.
    Two separate random streams, so base and pinn see the same labelled batches per seed."""

    def __init__(self, people: list[GapPerson], cfg: dict, ratio: float, seed: int, warm: int):
        self.people, self.B, self.warm = people, cfg["batch_size"], warm
        self.n = int(round(cfg["chunk_s"] * data.RATE))
        self.phys = [p.physics(ratio) for p in people]
        self.files = [p.all_ids() for p in people]
        P, F, idx = [], [], []
        for pi, p in enumerate(people):
            for fi, k in enumerate(self.files[pi]):
                ii = np.flatnonzero(p.loss_keep(k))
                ii = ii[ii >= SKIP]
                P.append(np.full(len(ii), pi, np.int32))
                F.append(np.full(len(ii), fi, np.int32))
                idx.append(ii.astype(np.int64))
        self.kP, self.kF, self.kI = map(np.concatenate, (P, F, idx))
        self.lens = [
            np.array([len(p.h["files"][k]["t"]) for k in self.files[pi]], float)
            for pi, p in enumerate(people)
        ]
        self.upeople = [pi for pi in range(len(people)) if self.phys[pi] is not None]
        w = np.array([self.lens[pi].sum() for pi in self.upeople])
        self.uw = w / w.sum() if len(w) else w
        self.rl = np.random.default_rng([seed, 0])
        self.ru = np.random.default_rng([seed, 1])

    def _chunk(self, pi: int, fi: int, s: int, labelled: bool) -> dict:
        p = self.people[pi]
        e = p.extras(self.files[pi][fi])
        T = len(e["x"])
        sl = slice(s, min(s + self.n, T))
        pad = self.n - (sl.stop - sl.start)

        def take(a):
            return np.pad(a[sl], [(pad, 0)] + [(0, 0)] * (a.ndim - 1))

        sc = np.zeros(self.n, bool)
        sc[pad:] = np.arange(sl.start, sl.stop) >= SKIP
        ph = self.phys[pi]
        out = {"x": take(e["x"]), "score": sc, "has": ph is not None, "ph": ph}
        if labelled:
            out["y"] = take(e["y"])
            out["keep"] = take(e["keep"])
        else:
            out["y"] = np.full((self.n, 3), np.nan, np.float32)
            out["keep"] = np.zeros(self.n, bool)
        if ph is not None:
            for k in ("ref", "phd", "act", "swing"):
                out[k] = take(e[k])
        else:
            out["ref"] = np.zeros((self.n, 2), np.float32)
            out["phd"] = np.zeros(self.n, np.float32)
            out["act"] = np.zeros((self.n, len(swing.muscle_index())), np.float32)
            out["swing"] = np.zeros(self.n, bool)
        return out

    def labelled(self) -> list[dict]:
        out = []
        for _ in range(self.B):
            j = int(self.rl.integers(len(self.kI)))
            pi, fi, i = int(self.kP[j]), int(self.kF[j]), int(self.kI[j])
            T = int(self.lens[pi][fi])
            s = int(self.rl.integers(i - self.n + 1, i - self.warm + 1))
            s = min(max(s, 0), max(T - self.n, 0))
            out.append(self._chunk(pi, fi, s, True))
        return out

    def unlabelled(self) -> list[dict]:
        out = []
        for _ in range(self.B):
            pi = self.upeople[int(self.ru.choice(len(self.upeople), p=self.uw))]
            ln = self.lens[pi]
            fi = int(self.ru.choice(len(ln), p=ln / ln.sum()))
            T = int(ln[fi])
            s = int(self.ru.integers(0, T - self.n + 1)) if T >= self.n else 0
            out.append(self._chunk(pi, fi, s, False))
        return out


def to_tensors(chunks: list[dict], dev) -> dict:
    t = {
        k: torch.tensor(np.stack([c[k] for c in chunks])).to(dev)
        for k in ("x", "y", "keep", "ref", "phd", "act", "swing", "score")
    }
    t["has"] = torch.tensor([c["has"] for c in chunks], device=dev)
    for k in ("x", "y", "ref", "act"):
        t[k] = t[k].transpose(1, 2)
    par = {k: [] for k in ("inv", *cross.PAR_KEYS)}
    for c in chunks:
        for k in par:
            ph = c["ph"]
            par[k].append(ph[k] if ph else ([1.0, 1.0] if k == "inv" else 1.0))
    t["par"] = {
        k: torch.tensor(np.array(v), dtype=torch.float32, device=dev) for k, v in par.items()
    }
    for k in cross.PAR_KEYS:
        t["par"][k] = t["par"][k][:, None]
    return t


# ---------- training ----------


def val_score(model, val_p: list[GapPerson], dev) -> float:
    """RMSE pooled over all kept, scored samples of the validation people (mean of 3 joints)."""
    model.eval()
    se, n = np.zeros(3), 0
    for p in val_p:
        for xn, y, m in p.val_files():
            e = cross.predict(model, xn, dev)[m] - y[m]
            se += (e**2).sum(0)
            n += int(m.sum())
    if n == 0:
        raise RuntimeError("no kept validation samples in this fold")
    v = float(np.sqrt(se / n).mean())
    assert np.isfinite(v)
    return v


def fit(cfg, train_p, val_p, seed, dev, unlabelled=True, trace=False):
    """Train one model. pinn uses the unlabelled batch unless unlabelled=False (pinn_lab)."""
    torch.manual_seed(seed)
    y = np.concatenate([p.y(k) for p in train_p for k in p.all_ids()])
    model = cross.XModel(cfg, np.nanmean(y, 0), np.nanstd(y, 0)).to(dev)
    del y
    ratio = swing_ratio(train_p)
    pinn = cfg["model"].startswith("pinn")
    params = list(model.parameters())
    gterm = sterm = None
    if pinn:
        gterm = gyro.GyroTerm({j: 1.0 for j in gyro.JOINTS}).to(dev)
        if cfg.get("swing", True):  # study 4: gyro term only
            sterm = swing.SwingTerm(train_p[0].body, 1.0, 1.0, cfg["swing_init"]).to(dev)
            params += list(sterm.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = model.net.receptive_field - 1
    smp = Sampler(train_p, cfg, ratio, seed, warm)
    use_unl = pinn and unlabelled and len(smp.upeople) > 0
    best, bad, hist, best_state, best_swing = np.inf, 0, [], None, None
    steps = []
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot = {"angle": 0.0, "gyro": 0.0, "swing": 0.0, "angle_samples": 0.0}
        for _ in range(cfg["steps_per_epoch"]):
            chunks = smp.labelled()
            nl = len(chunks)
            if use_unl:
                chunks = chunks + smp.unlabelled()
            b = to_tensors(chunks, dev)
            angles, tau_n, out_n = model(b["x"])
            m = b["score"].clone()
            m[:, :warm] = False
            sel = m[:nl] & b["keep"][:nl]
            yn = (b["y"][:nl] - model.y_mean) / model.y_std
            o = out_n[:nl].permute(0, 2, 1)[sel]
            tgt = yn.permute(0, 2, 1)[sel]
            assert torch.isfinite(tgt).all(), "a lost angle reached the loss"
            loss = ((o - tgt) ** 2).mean() if len(tgt) else out_n.sum() * 0.0
            tot["angle"] += loss.item()
            tot["angle_samples"] += float(sel.sum().item())
            if pinn:
                lg, ls = cross.physics_terms(gterm, sterm, angles, tau_n, b, m, b["par"], b["has"])
                loss = loss + cfg["lam"] * (lg + ls)
                tot["gyro"] += lg.item()
                tot["swing"] += ls.item()
            assert torch.isfinite(loss), "non-finite loss"
            if trace:
                steps.append(
                    (
                        loss.item(),
                        float(tgt.double().sum().item()),
                        float(sel.sum().item()),
                        float(b["x"].double().sum().item()),
                    )
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        v = val_score(model, val_p, dev)
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
    out = {
        "val_rmse": best,
        "epochs": len(hist),
        "train_seconds": round(time.time() - t0, 1),
        "unlabelled_batch": use_unl,
        "ratio": ratio,
        "y_mean": model.y_mean.flatten().tolist(),
        "y_std": model.y_std.flatten().tolist(),
        "history": hist,
    }
    if trace:
        out["steps"] = steps
    return model, out


# ---------- runs ----------


def share_tag(share: float) -> str:
    return f"s{round(share * 100):02d}"


def people_info(train_p, val_p) -> dict:
    return {
        "physics_people": [p.name for p in train_p if p.gyro_ok],
        "mask_hash": {p.name: p.mask_hash() for p in train_p + val_p},
        "kept_seconds": {p.name: p.kept_seconds() for p in train_p + val_p},
        "setup": {p.name: p.setup_info for p in train_p},
    }


def masks_summary(h5: Path, people: list[str], fl: list[dict]) -> dict:
    """Mask statistics (no angles) and every person's calibration-trial setup."""
    out = {"shares": SHARES, "draws": [TUNE_SEED, 0, 1, 2, 3, 4], "people": {}, "folds": {}}
    for name in people:
        p = GapPerson(h5 / f"{name}.h5", 0.0, 0, setup=True)
        total = {k: len(p.h["files"][k]["t"]) for k in p.all_ids()}
        trial = {k: (s, e) for k, s, e in p.trial}
        rows = {}
        for share in SHARES:
            for draw in out["draws"]:
                kept = {}
                outside_kept = outside = 0
                for k, T in total.items():
                    m = keep_mask(T, name, k, share, draw)
                    o = np.ones(T, bool)
                    if k in trial:
                        o[trial[k][0] : trial[k][1]] = False
                    outside += int(o.sum())
                    outside_kept += int((m & o).sum())
                    m[~o] = True
                    mode = p.modes[k]
                    kept[mode] = kept.get(mode, 0.0) + float(m.sum()) / data.RATE
                rows[f"{share_tag(share)}/d{draw}"] = {
                    "kept_seconds": kept,
                    "realised_share": 1 - outside_kept / max(outside, 1),
                }
        out["people"][name] = {
            "total_seconds": sum(total.values()) / data.RATE,
            "masks": rows,
            "setup": p.setup_info,
        }
    for f in fl:
        out["folds"][f"fold{f['fold']}"] = {
            "physics_people": [n for n in f["train"] if out["people"][n]["setup"]["gyro_ok"]],
            "val_kept_seconds": {
                key: sum(
                    sum(out["people"][n]["masks"][key]["kept_seconds"].values()) for n in f["val"]
                )
                for key in out["people"][f["val"][0]]["masks"]
            },
        }
    return out


def gselect(runs_dir: Path, prefix: str = "g") -> dict:
    """Per share and fold: the configuration with the lowest validation score (nested).
    prefix: g (study 3), g4 (study 4, EMG only), g4a (study 4, EMG + accelerometers)."""
    recs = [r for r in runs.load_runs(runs_dir) if r.get("kind") == f"{prefix}tune"]
    recs = [r for r in recs if r.get("status") == "done"]
    out, n = {}, {}
    keys = ["lr", "width", "weight_decay", "lam"]
    for r in recs:
        key = f"{share_tag(r['share'])}/fold{r['fold']['fold']}"
        m = r["model"].split("-")[0]
        n[(key, m)] = n.get((key, m), 0) + 1
        cur = out.setdefault(key, {}).get(f"_{m}")
        if cur is None or r["metrics"]["val_rmse"] < cur[0]:
            out[key][f"_{m}"] = (r["metrics"]["val_rmse"], r["run_id"])
            out[key][m] = {k: r["config"][k] for k in keys if k in r["config"]}
    for (key, m), c in n.items():
        if c != 16:
            raise SystemExit(f"{key} {m}: {c} tuning runs, expected 16")
    for key, v in out.items():
        if not {"base", "pinn"} <= set(v):
            raise SystemExit(f"{key}: missing tuning result")
    (runs_dir / f"chosen_{prefix}.json").write_text(json.dumps(out, indent=2))
    return out


# Study 4 feasibility pilot (PLAN4.md): fold 0 only, validation people only, before any
# study-4 plan is frozen. lr 3e-3 and width 128 throughout.
PILOT4 = (
    [
        {"share": sh, "model": "base", "inputs": "emg", "weight_decay": wd, "lam": 0.0}
        for sh in (0.0, 0.9, 0.99)
        for wd in (0.0, 1e-4)
    ]
    + [
        {"share": sh, "model": "pinn", "inputs": "emg", "weight_decay": 0.0, "lam": lam}
        for sh in (0.0, 0.9, 0.99)
        for lam in (0.01, 0.1, 1.0, 10.0)
    ]
    + [
        {"share": 0.99, "model": "base", "inputs": "emg_acc", "weight_decay": 0.0, "lam": 0.0},
        {"share": 0.99, "model": "pinn", "inputs": "emg_acc", "weight_decay": 0.0, "lam": 1.0},
    ]
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["gmasks", "gtune", "gfinal", "gselect", "g4pilot"])
    ap.add_argument("--index", type=int, help="g4pilot: entry of PILOT4")
    ap.add_argument("--prefix", default="g", help="configs/<prefix><model>.yaml, record kinds")
    ap.add_argument("--teacher-chosen", type=Path, help="distill: study 3 choices (all sensors)")
    ap.add_argument("--h5", type=Path)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--share", type=float)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--model", choices=["base", "pinn", "pinn_lab", "distill"])
    ap.add_argument("--cfg", type=int, help="gtune: grid index")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chosen", type=Path)
    ap.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--people", nargs="*", help="restrict to these people (local tests)")
    ap.add_argument("--n-val", type=int, default=cross.N_VAL)
    a = ap.parse_args()
    if a.kind == "gselect":
        for k, v in sorted(gselect(a.runs, a.prefix).items()):
            print(k, {m: v[m] for m in ("base", "pinn")})
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    people = a.people or sorted(p.stem for p in a.h5.glob("AB*.h5"))
    fl = cross.folds(people, n_val=a.n_val)
    if not a.people and [f["test"] for f in fl] != cross.PLAN2_TEST:
        raise SystemExit(f"folds differ from PLAN2.md (people found: {len(people)})")
    if a.kind == "gmasks":
        rec = cross.record("gmasks", "all", "all", 0, a.code, {})
        rec |= {"metrics": masks_summary(a.h5, people, fl), "status": "done"}
        runs.write_run(a.runs, rec)
        return
    if a.kind == "g4pilot":
        e = PILOT4[a.index]
        a.share, a.fold, a.model = e["share"], 0, e["model"]
    if a.share not in SHARES:
        raise SystemExit(f"share must be one of {SHARES}")
    f = fl[a.fold]
    cdir = a.code / "configs"
    seed = TUNE_SEED if a.kind in ("gtune", "g4pilot") else a.seed
    t0 = time.time()
    train_p = [GapPerson(a.h5 / f"{p}.h5", a.share, seed) for p in f["train"]]
    val_p = [GapPerson(a.h5 / f"{p}.h5", a.share, seed, setup=False) for p in f["val"]]
    extra = {"fold": f, "share": a.share, "draw": seed, "prep_seconds": round(time.time() - t0, 1)}
    extra |= people_info(train_p, val_p)
    mx = {"max_epochs": a.max_epochs} if a.max_epochs else {}
    if a.kind == "g4pilot":
        base_cfg = yaml.safe_load((cdir / f"g{e['model']}.yaml").read_text())
        c = {k: v for k, v in base_cfg.items() if k != "grid"} | mx
        c |= {"lr": 3e-3, "width": 128, "swing": False, "study": 4}
        c |= {k: e[k] for k in ("inputs", "weight_decay", "lam")}
        name = f"{e['model']}-{e['inputs']}-{share_tag(a.share)}-p{a.index:02d}"
        rec = cross.record("g4pilot", name, "fold0", seed, a.code, extra | {"config": c})
        runs.write_run(a.runs, rec)
        _, out = fit(c, train_p, val_p, seed, dev)
        rec |= {"metrics": {"val_rmse": out["val_rmse"]}, "training": out, "status": "done"}
        runs.write_run(a.runs, rec)
        return
    if a.kind == "gtune":
        if a.model in ("pinn_lab", "distill"):
            raise SystemExit(f"{a.model} is not tuned")
        c = cross.grid(yaml.safe_load((cdir / f"{a.prefix}{a.model}.yaml").read_text()))[a.cfg]
        c = c | {"tag": f"c{a.cfg:02d}"} | mx
        name = f"{a.model}-{share_tag(a.share)}-c{a.cfg:02d}"
        kind = f"{a.prefix}tune"
        rec = cross.record(kind, name, f"fold{a.fold}", seed, a.code, extra | {"config": c})
        runs.write_run(a.runs, rec)
        _, out = fit(c, train_p, val_p, seed, dev)
        rec |= {"metrics": {"val_rmse": out["val_rmse"]}, "training": out, "status": "done"}
        runs.write_run(a.runs, rec)
        return
    # gfinal
    chosen = json.loads(a.chosen.read_text())[f"{share_tag(a.share)}/fold{a.fold}"]
    src = {"pinn_lab": "pinn", "distill": "base"}.get(a.model, a.model)
    base_cfg = yaml.safe_load((cdir / f"{a.prefix}{src}.yaml").read_text())
    c = {k: v for k, v in base_cfg.items() if k != "grid"} | chosen[src] | mx
    if a.model == "distill" and (a.prefix == "g" or not a.teacher_chosen):
        raise SystemExit("distill is a study-4 model: needs --prefix g4/g4a and --teacher-chosen")
    if a.model == "distill":  # teacher: study 3's all-sensor base, same share, fold and mask
        tch = json.loads(a.teacher_chosen.read_text())[f"{share_tag(a.share)}/fold{a.fold}"]
        tbase = yaml.safe_load((cdir / "gbase.yaml").read_text())
        tc = {k: v for k, v in tbase.items() if k != "grid"} | tch["base"] | mx
        teacher, tout = fit(tc, train_p, val_p, seed, dev)
        for p in train_p:
            p.fill = {k: cross.predict(teacher, p.norm(p.x(k)), dev) for k in p.all_ids()}
            p._cache = {}
        extra |= {"teacher": {"config": tc, "val_rmse": tout["val_rmse"]}}
        del teacher
    model, out = fit(c, train_p, val_p, seed, dev, unlabelled=a.model != "pinn_lab")
    for name in f["test"]:
        tp = cross.Unlabelled(a.h5 / f"{name}.h5")
        rec = cross.record(
            f"{a.prefix}final",
            f"{a.model}-{share_tag(a.share)}",
            name,
            seed,
            a.code,
            extra | {"config": c},
        )
        rec["model"] = a.model
        d = runs.write_run(a.runs, rec)
        met, keep = cross.score_stream(model, tp, dev)
        rec |= {
            "metrics": {"val_rmse": out["val_rmse"], **met},
            "status": "done",
            "training": {k: v for k, v in out.items() if k != "history"},
        }
        np.savez_compressed(d / "stream.npz", **keep)
        runs.write_run(a.runs, rec)
        print(rec["run_id"], "done", flush=True)  # no stair numbers in logs


if __name__ == "__main__":
    main()
