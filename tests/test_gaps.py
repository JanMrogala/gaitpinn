"""Study 3: gap masks, the calibration trial, the sampler, and that lost angles are never read."""

import shutil

import h5py
import numpy as np
import pytest
import torch

from gaitpinn import data, gaps, gyro

CFG = {
    "model": "pinn",
    "width": 8,
    "levels": 3,
    "kernel": 3,
    "dropout": 0.0,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 4,
    "chunk_s": 4.0,
    "steps_per_epoch": 3,
    "max_epochs": 2,
    "patience": 8,
    "lam": 0.1,
    "swing_init": {"w": 0.0, "k": -2.0, "b": -2.0, "q0": 0.1},
}


def make_gyro_person(path, name, seed):
    """A person whose gyros match the derivatives of their angles, so the gyro check passes."""
    rng = np.random.default_rng(seed)
    gi = {s: gyro.gyro_index(s) for s in gyro.SEGMENTS}
    with h5py.File(path, "w") as h:
        h.attrs.update(person=name, height_m=1.7, mass_kg=70.0, rate_hz=data.RATE)
        spec = [("treadmill", 6, 6000), ("levelground", 8, 1500), ("stair", 3, 1500)]
        spec += [("ramp", 2, 1500)]
        for mode, n, T in spec:
            for i in range(n):
                t = np.arange(T) / data.RATE
                ph = rng.uniform(0, 2 * np.pi)
                w = 2 * np.pi * rng.uniform(0.9, 1.1)
                y = np.column_stack(
                    [
                        30 * np.sin(w * t + ph),
                        -30 * (1 + np.sin(w * t + ph + 1.0)),
                        10 * np.sin(w * t + ph + 2.0),
                    ]
                )
                dy = np.radians(np.gradient(y, axis=0) * data.RATE)
                x = rng.normal(size=(T, 35)) * 0.05
                x[:, : len(data.EMG)] = np.abs(rng.normal(size=(T, len(data.EMG))))
                thigh = dy[:, 0]
                shank = thigh + dy[:, 1]
                foot = shank + dy[:, 2]
                for seg, v in (("thigh", thigh), ("shank", shank), ("foot", foot)):
                    x[:, gi[seg][0]] = v + rng.normal(size=T) * 0.02
                g = h.create_group(f"files/{mode}__{mode}_{i:02d}")
                g["x"] = x.astype(np.float32)
                g["y"] = y.astype(np.float32)
                g["t"] = t
                g["label"] = np.array([b"walk"] * T)
                g.attrs.update(
                    mode=mode,
                    name=f"{mode}_{i:02d}",
                    parent=f"{mode}_parent_{i % 2}",
                    trial_start=float(100 - i),
                    speed_min=np.nan,
                    speed_max=np.nan,
                )
    return path


@pytest.fixture(scope="module")
def people(tmp_path_factory):
    d = tmp_path_factory.mktemp("h5")
    for i, n in enumerate(["P1", "P2", "P3"]):
        make_gyro_person(d / f"{n}.h5", n, i)
    return d


def test_masks_are_nested_deterministic_and_draw_dependent():
    T = 20_000
    m = {s: gaps.keep_mask(T, "AB10", "f1", s, 3) for s in gaps.SHARES}
    assert m[0.0].all()
    assert (m[0.99] <= m[0.9]).all() and (m[0.9] <= m[0.5]).all()
    assert abs(1 - m[0.5].mean() - 0.5) < 0.1
    assert np.array_equal(m[0.9], gaps.keep_mask(T, "AB10", "f1", 0.9, 3))
    assert not np.array_equal(m[0.5], gaps.keep_mask(T, "AB10", "f1", 0.5, 4))
    # whole 2 s blocks, the last shorter block counts as one
    b = m[0.5][: (T // gaps.BLOCK) * gaps.BLOCK].reshape(-1, gaps.BLOCK)
    assert (b.all(1) | ~b.any(1)).all()


def test_calibration_trial_is_kept_and_setup_finds_physics(people):
    p = gaps.GapPerson(people / "P1.h5", 0.99, 0)
    assert p.trial and sum(e - s for _, s, e in p.trial) == 30 * data.RATE
    for k, s, e in p.trial:
        assert p.keep[k][s:e].all() and data.file_modes(p.h)[k] == "treadmill"
    assert p.gyro_ok, p.setup_info
    q = gaps.GapPerson(people / "P1.h5", 0.5, 0)
    assert q.setup_info == p.setup_info  # the setup does not depend on the share
    assert 1 - np.mean(np.concatenate([p.keep[k] for k in p.all_ids()])) > 0.8


def test_labelled_chunks_score_their_kept_sample_and_streams_are_separate(people):
    tr = [gaps.GapPerson(people / f"{n}.h5", 0.9, 1) for n in ("P1", "P2")]
    a = gaps.Sampler(tr, CFG, 1.0, 5, warm=126)
    b = gaps.Sampler(tr, CFG, 1.0, 5, warm=126)
    for _ in range(5):
        la = a.labelled()
        a.unlabelled()  # pinn draws an extra batch: base's labelled batches must not change
        lb = b.labelled()
        for ca, cb in zip(la, lb, strict=True):
            assert np.array_equal(ca["x"], cb["x"])
            sc = ca["score"].copy()
            sc[:126] = False
            assert (sc & ca["keep"]).any()  # some kept sample is in the scored part
    u = a.unlabelled()
    assert all(not c["keep"].any() and np.isnan(c["y"]).all() for c in u)


def _train(h5dir, model, share):
    torch.set_num_threads(1)
    tr = [gaps.GapPerson(h5dir / f"{n}.h5", share, 7) for n in ("P1", "P2")]
    va = [gaps.GapPerson(h5dir / "P3.h5", share, 7, setup=False)]
    cfg = CFG | {"model": model}
    net, out = gaps.fit(cfg, tr, va, 7, torch.device("cpu"), trace=True)
    state = {k: v.clone() for k, v in net.state_dict().items()}
    setups = [p.setup_info for p in tr]
    return setups, out, state, tr + va


def _corrupt(src, dst, people, how):
    shutil.copytree(src, dst)
    rng = np.random.default_rng(0)
    for p in people:
        with h5py.File(dst / f"{p.name}.h5", "r+") as h:
            for k in p.all_ids():
                y = h["files"][k]["y"][:]
                lost = ~p.keep[k]
                y[lost] = rng.normal(0, 50, (lost.sum(), 3)) if how == "random" else 1e6
                h["files"][k]["y"][...] = y


@pytest.mark.parametrize("model", ["base", "pinn"])
@pytest.mark.parametrize("share", [0.5, 0.9])
def test_lost_angles_never_change_anything(people, tmp_path, model, share):
    ref = _train(people, model, share)
    assert ref[1]["unlabelled_batch"] == (model == "pinn")
    assert any(s["gyro_ok"] for s in ref[0])
    for how in ("random", "huge"):
        d = tmp_path / how
        _corrupt(people, d, ref[3], how)
        got = _train(d, model, share)
        assert got[0] == ref[0]
        for key in ("y_mean", "y_std", "steps", "history", "val_rmse", "ratio"):
            assert got[1][key] == ref[1][key], key
        for k in ref[2]:
            assert torch.equal(got[2][k], ref[2][k]), k


def test_study4_network_never_sees_the_gyros():
    from gaitpinn import cross

    gyro_cols = {len(data.EMG) + i for i, c in enumerate(data.IMU) if "_Gyro_" in c}
    for name in ("emg", "emg_acc"):
        assert not gyro_cols & set(cross.input_channels(name))
    assert len(cross.input_channels("emg")) == len(data.EMG)
    assert len(cross.input_channels("emg_acc")) == len(data.EMG) + 12
    torch.manual_seed(0)
    for inputs in ("emg", "emg_acc"):
        cfg = CFG | {"inputs": inputs, "swing": False}
        net = cross.XModel(cfg, np.zeros(3), np.ones(3)).eval()
        x = torch.randn(2, 35, 400)
        x2 = x.clone()
        x2[:, sorted(gyro_cols)] = torch.randn(2, len(gyro_cols), 400) * 100
        with torch.no_grad():
            assert torch.equal(net(x)[0], net(x2)[0])
        assert net(x)[1] is None  # no torque output without the swing term


def test_study4_pinn_trains_with_gyro_term_only(people):
    torch.set_num_threads(1)
    tr = [gaps.GapPerson(people / f"{n}.h5", 0.9, 7) for n in ("P1", "P2")]
    va = [gaps.GapPerson(people / "P3.h5", 0.9, 7, setup=False)]
    cfg = CFG | {"inputs": "emg", "swing": False, "lam": 1.0}
    _, out = gaps.fit(cfg, tr, va, 7, torch.device("cpu"))
    h = out["history"]
    assert all(e["gyro"] > 0 and e["swing"] == 0 for e in h)
