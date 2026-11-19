"""Stairs data must never touch training, tuning, normalization or calibration."""

import h5py
import numpy as np
import pytest

from gaitpinn import data


def test_split_keeps_stairs_out_of_walking_parts(person):
    with h5py.File(person) as h:
        modes = data.file_modes(h)
        s = data.split_files(modes)
    for part in ("train", "val", "test"):
        assert s[part], part
        assert all(modes[k] in data.WALKING for k in s[part]), part
    assert all(modes[k] == data.STAIRS for k in s["stairs"])
    assert not any(modes[k] == "ramp" for p in s.values() for k in p)
    # the three walking parts don't overlap, and both modes are in each part
    assert not set(s["train"]) & set(s["val"]) and not set(s["val"]) & set(s["test"])
    for part in ("train", "val", "test"):
        assert {modes[k] for k in s[part]} == set(data.WALKING)


def test_normalization_ignores_stairs(person, tmp_path):
    """Replacing every stair file with garbage must not change the statistics."""
    with h5py.File(person) as h:
        s = data.split_files(data.file_modes(h))
        mean1, std1 = data.norm_stats(h, s["train"])
    with h5py.File(person, "r+") as h:
        for k in s["stairs"]:
            h["files"][k]["x"][...] = 1e6
    with h5py.File(person) as h:
        mean2, std2 = data.norm_stats(h, s["train"])
    np.testing.assert_array_equal(mean1, mean2)
    np.testing.assert_array_equal(std1, std2)


def test_calibration_refuses_stair_files(person):
    with h5py.File(person) as h:
        s = data.split_files(data.file_modes(h))
        with pytest.raises(AssertionError):
            data.norm_stats(h, s["train"] + s["stairs"][:1])


def test_split_rule_sizes():
    modes = {f"t{i}": "treadmill" for i in range(7)} | {f"l{i}": "levelground" for i in range(30)}
    s = data.split_files(modes)
    # treadmill: 7 -> 1/1/5; level ground: 30 -> floor(4.5 + 0.5) = 5 -> 5/5/20
    assert len(s["val"]) == 6 and len(s["test"]) == 6 and len(s["train"]) == 25


def test_invalid_files_are_left_out(person):
    with h5py.File(person, "r+") as h:
        h["files"]["levelground__levelground_00"].attrs["valid"] = False
        h["files"]["stair__stair_03"].attrs["valid"] = False
    with h5py.File(person) as h:
        s = data.split_files(data.file_modes(h))
    ids = [k for p in s.values() for k in p]
    assert "levelground__levelground_00" not in ids and "stair__stair_03" not in ids


def test_target_quality_rule():
    rng = np.random.default_rng(0)
    y = np.column_stack([rng.normal(0, 15, 500), rng.normal(-30, 20, 500), rng.normal(0, 8, 500)])
    assert data.target_ok(y)[0]
    flat = y.copy()
    flat[:, 0] = 120.0
    assert not data.target_ok(flat)[0]
    wild = y.copy()
    wild[:20, 1] = 90.0
    assert not data.target_ok(wild)[0]


def test_all_calibration_and_training_ignore_stairs(person):
    """Garbage in every stair file must not change any calibration or the first training batch."""
    import torch
    import yaml

    from gaitpinn import train

    cfg = yaml.safe_load(open("configs/pinn.yaml"))

    def snapshot():
        p = train.Person(person)
        b = next(
            train.batches(p, p.split["train"], cfg, np.random.default_rng(0), torch.device("cpu"))
        )
        out = {
            "align": p.align,
            "gyro_scale": p.gyro_scale,
            "swing_thr": p.swing_thr,
            "emg_peak": p.emg_peak.tolist(),
            "y_mean": p.y_mean.tolist(),
            "y_std": p.y_std.tolist(),
            "x_mean": p.x_mean.tolist(),
            "swing_scale": p.swing_scale,
            "thigh_sign": p.thigh_sign,
            "batch": {k: v.numpy().tolist() for k, v in b.items()},
        }
        p.h.close()
        return out

    before = snapshot()
    with h5py.File(person, "r+") as h:
        for k in h["files"]:
            if h["files"][k].attrs["mode"] == data.STAIRS:
                h["files"][k]["x"][...] = 1e3
                h["files"][k]["y"][...] = -1e3
    assert snapshot() == before
