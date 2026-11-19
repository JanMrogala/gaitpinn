"""Camargo et al. 2021 -> one HDF5 file per person, plus the split and the test stream.

Processing (PLAN.md, "Data"): every file is processed on its own with causal filters only,
then sampled onto a 100 Hz grid. Each sample on the grid takes the newest source sample at or
before its time, so nothing from the future leaks in.

HDF5 layout (`<person>.h5`):
    attrs: person, height_m, mass_kg, emg, imu, targets (channel names), rate_hz
    files/<mode>__<name>/x      (T, 35) float32   EMG envelopes (11) + IMU (24)
    files/<mode>__<name>/y      (T, 3)  float32   hip, knee, ankle angle (deg)
    files/<mode>__<name>/t      (T,)    float64   recording time (s)
    files/<mode>__<name>/label  (T,)    bytes     activity label (scoring only, never an input)
    files/<mode>__<name>.attrs: mode, name, parent, trial_start, speed_min, speed_max
"""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, lfilter

RATE = 100
EMG = [
    "gastrocmed",
    "tibialisanterior",
    "soleus",
    "vastusmedialis",
    "vastuslateralis",
    "rectusfemoris",
    "bicepsfemoris",
    "semitendinosus",
    "gracilis",
    "gluteusmedius",
    "rightexternaloblique",
]
IMU = [
    f"{seg}_{kind}_{ax}"
    for seg in ["foot", "shank", "thigh", "trunk"]
    for kind in ["Accel", "Gyro"]
    for ax in "XYZ"
]
TARGETS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
WALKING = ["treadmill", "levelground"]
STAIRS = "stair"
RAMP = "ramp"  # used only by study 2 (cross-person); study 1 never reads ramp files
SKIP_S = 0.5  # first 0.5 s of each file: model context only, never scored or in a loss


# ---------- causal preprocessing ----------


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    """Causal Butterworth low-pass along axis 0, starting from zero state."""
    b, a = butter(order, fc / (fs / 2))
    return lfilter(b, a, x, axis=0)


def emg_envelope(emg: np.ndarray, fs: float, fc: float = 6.0) -> np.ndarray:
    return lowpass(np.abs(emg), fs, fc)


def sample_at(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Newest source sample at or before each destination time (causal, zero-order hold)."""
    idx = np.searchsorted(t_src, t_dst + 1e-9, side="right") - 1
    return x[np.clip(idx, 0, len(t_src) - 1)]


def grid(*starts_ends: tuple[float, float]) -> np.ndarray:
    """100 Hz time grid covering the span where all streams have data."""
    t0 = max(s for s, _ in starts_ends)
    t1 = min(e for _, e in starts_ends)
    n = int(math.floor((t1 - t0) * RATE + 1e-6)) + 1
    return t0 + np.arange(n) / RATE


def process_file(ik: dict, imu: dict, emg: dict, labels: dict | None) -> dict:
    """Turn one trial's tables into x (T, 35), y (T, 3), t (T,), label (T,) at 100 Hz."""
    t_ik = np.asarray(ik["Header"], float)
    t_imu = np.asarray(imu["Header"], float)
    t_emg = np.asarray(emg["Header"], float)
    fs_imu = 1 / np.median(np.diff(t_imu))
    fs_emg = 1 / np.median(np.diff(t_emg))
    t = grid((t_ik[0], t_ik[-1]), (t_imu[0], t_imu[-1]), (t_emg[0], t_emg[-1]))
    e = emg_envelope(np.column_stack([np.asarray(emg[c], float) for c in EMG]), fs_emg)
    i = lowpass(np.column_stack([np.asarray(imu[c], float) for c in IMU]), fs_imu, 40.0)
    x = np.hstack([sample_at(t_emg, e, t), sample_at(t_imu, i, t)])
    y = sample_at(t_ik, np.column_stack([np.asarray(ik[c], float) for c in TARGETS]), t)
    if labels is not None:
        lab = np.asarray(labels["Label"], dtype=object)
        label = sample_at(np.asarray(labels["Header"], float), lab, t).astype(str)
    else:
        label = np.full(len(t), "", dtype=str)
    return {"x": x.astype(np.float32), "y": y.astype(np.float32), "t": t, "label": label}


# ---------- target quality ----------

# Joint limits (deg) in the IK model's conventions; generous, only to catch broken IK solutions.
LIMITS = [(-60.0, 130.0), (-150.0, 30.0), (-60.0, 60.0)]


def target_ok(y: np.ndarray) -> tuple[bool, str]:
    """A file is unusable if any target angle is flat (std < 0.5 deg) or more than 1% of its
    samples lie outside the joint limits. Looks at the target only, never at predictions."""
    for j, (lo, hi) in enumerate(LIMITS):
        if y[:, j].std() < 0.5:
            return False, f"{TARGETS[j]} is flat"
        if np.mean((y[:, j] < lo) | (y[:, j] > hi)) > 0.01:
            return False, f"{TARGETS[j]} outside [{lo}, {hi}] for > 1% of samples"
    return True, ""


# ---------- conversion of one person ----------


def convert_person(person_dir: Path, info: dict, out: Path, ramps: bool = False) -> Path:
    """Convert <person>/<date>/{treadmill,levelground,stair}/... into <out>/<person>.h5."""
    from gaitpinn import matfile

    (date_dir,) = [d for d in person_dir.iterdir() if d.is_dir() and d.name != "osimxml"]
    person = person_dir.name
    path = out / f"{person}.h5"
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h:
        h.attrs.update(
            person=person,
            height_m=info["height_m"],
            mass_kg=info["mass_kg"],
            rate_hz=RATE,
            emg=EMG,
            imu=IMU,
            targets=TARGETS,
        )
        for mode in [*WALKING, STAIRS] + ([RAMP] if ramps else []):
            for f in sorted((date_dir / mode / "ik").glob("*.mat")):
                name = f.stem
                tabs = {
                    k: matfile.load_table(date_dir / mode / k / f.name)
                    for k in ["ik", "imu", "emg"]
                }
                cond = date_dir / mode / "conditions" / f.name
                ctabs = matfile.load_tables(cond)
                cvars = matfile.load_struct(cond)
                labels = next((tb for tb in ctabs if "Label" in tb), None)
                speed = next((tb for tb in ctabs if "Speed" in tb), None)
                d = process_file(tabs["ik"], tabs["imu"], tabs["emg"], labels)
                if np.isnan(d["x"]).any() or np.isnan(d["y"]).any():
                    raise ValueError(f"{f}: NaN after processing")
                g = h.create_group(f"files/{mode}__{name}")
                g["x"], g["y"], g["t"] = d["x"], d["y"], d["t"]
                g["label"] = d["label"].astype("S")
                ok, why = target_ok(d["y"])
                g.attrs.update(
                    valid=ok,
                    invalid_reason=why,
                    mode=mode,
                    name=name,
                    parent=str(cvars.get("file", name)).replace("\\", "/").split("/")[-1],
                    trial_start=float(np.ravel(cvars.get("trialStarts", [np.nan]))[0]),
                    speed_min=speed_range(speed)[0],
                    speed_max=speed_range(speed)[1],
                )
    tmp.replace(path)
    return path


def speed_range(speed: dict | None) -> tuple[float, float]:
    """Treadmill belt speed (m/s): 5th percentile while moving (> 0.1 m/s) and maximum."""
    if speed is None:
        return np.nan, np.nan
    v = np.asarray(speed["Speed"], float)
    moving = v[v > 0.1]
    return float(np.percentile(moving, 5)), float(v.max())


def subject_info(path: Path) -> dict[str, dict]:
    """SubjectInfo.mat -> {person: {height_m, mass_kg}}."""
    from gaitpinn import matfile

    t = matfile.load_table(path, header="Subject")
    return {
        s: {"height_m": float(h), "mass_kg": float(w)}
        for s, h, w in zip(t["Subject"], t["Height"], t["Weight"], strict=True)
    }


# ---------- split, normalization, stream ----------


def file_modes(h: h5py.File) -> dict[str, str]:
    """Mode of every file that passed the target-quality check (others are left out)."""
    return {
        k: h["files"][k].attrs["mode"] for k in h["files"] if h["files"][k].attrs.get("valid", True)
    }


def split_files(modes: dict[str, str], seed: int = 0) -> dict[str, list[str]]:
    """Per-mode split of walking files by file: val and test get max(1, floor(0.15 n + 0.5))
    files each, train the rest. Stair files are test only; other modes are dropped."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": [], "stairs": []}
    for mode in WALKING:
        ids = sorted(k for k, m in modes.items() if m == mode)
        if len(ids) < 3:
            raise ValueError(f"only {len(ids)} {mode} files")
        ids = [ids[i] for i in rng.permutation(len(ids))]
        k = max(1, math.floor(0.15 * len(ids) + 0.5))
        out["val"] += ids[:k]
        out["test"] += ids[k : 2 * k]
        out["train"] += ids[2 * k :]
    out["stairs"] = sorted(k for k, m in modes.items() if m == STAIRS)
    for part in ("train", "val", "test"):
        out[part].sort()
    return out


def check_no_stairs(ids: list[str], modes: dict[str, str]) -> None:
    bad = [i for i in ids if modes[i] == STAIRS]
    if bad:
        raise AssertionError(f"stair files used where only walking is allowed: {bad[:3]}")


def norm_stats(h: h5py.File, train_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of the inputs over walking training files only."""
    check_no_stairs(train_ids, file_modes(h))
    x = np.concatenate([h["files"][k]["x"][:] for k in train_ids])
    return x.mean(0), x.std(0) + 1e-8


def stair_order(h: h5py.File, ids: list[str]) -> list[str]:
    """Recorded order: by parent recording (file name), then by start time within it."""

    def key(k):
        a = h["files"][k].attrs
        return (str(a["parent"]), float(a["trial_start"]), str(a["name"]))

    return sorted(ids, key=key)


def build_stream(h: h5py.File, split: dict[str, list[str]], walk_s: float = 30.0) -> dict:
    """Test stream: first `walk_s` s of level-ground walking test files, then all stair files.

    Returns x, y, t (s since the switch), file (index per sample), score (bool: scored),
    label, and the file list. The first SKIP_S of every file is context only (score False).
    """
    modes = file_modes(h)
    walk = sorted(k for k in split["test"] if modes[k] == "levelground")
    parts, n_walk = [], int(round(walk_s * RATE))
    taken = 0
    for k in walk:
        if taken >= n_walk:
            break
        n = min(len(h["files"][k]["t"]), n_walk - taken)
        parts.append((k, n))
        taken += n
    stairs = stair_order(h, split["stairs"])
    parts += [(k, len(h["files"][k]["t"])) for k in stairs]
    xs, ys, fs, sc, lb = [], [], [], [], []
    for i, (k, n) in enumerate(parts):
        g = h["files"][k]
        xs.append(g["x"][:n])
        ys.append(g["y"][:n])
        fs.append(np.full(n, i))
        s = np.ones(n, bool)
        s[: int(SKIP_S * RATE)] = False
        sc.append(s)
        lb.append(g["label"][:n].astype(str))
    x, y = np.concatenate(xs), np.concatenate(ys)
    t = (np.arange(len(x)) - taken) / RATE
    return {
        "x": x,
        "y": y,
        "t": t,
        "file": np.concatenate(fs),
        "score": np.concatenate(sc),
        "label": np.concatenate(lb),
        "files": [k for k, _ in parts],
        "walk_s": taken / RATE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert one or more Camargo people to HDF5.")
    ap.add_argument("--extract", type=Path, required=True, help="folder with AB06/, AB07/, ...")
    ap.add_argument("--info", type=Path, required=True, help="SubjectInfo.mat")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ramps", action="store_true", help="also convert ramp files (study 2)")
    ap.add_argument("people", nargs="+")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    info = subject_info(a.info)
    for p in a.people:
        print(convert_person(a.extract / p, info[p], a.out, a.ramps), flush=True)


if __name__ == "__main__":
    main()
