"""A small synthetic person file with the same layout as the real ones."""

import h5py
import numpy as np
import pytest

from gaitpinn import data


def make_person(path, n_tread=7, n_level=12, n_stair=10, n_ramp=4, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as h:
        h.attrs.update(person="ABxx", height_m=1.7, mass_kg=70.0, rate_hz=data.RATE)
        spec = (
            [("treadmill", n_tread, 3000)]
            + [("levelground", n_level, 1500)]
            + [("stair", n_stair, 1500)]
            + [("ramp", n_ramp, 1500)]
        )
        for mode, n, T in spec:
            for i in range(n):
                g = h.create_group(f"files/{mode}__{mode}_{i:02d}")
                g["x"] = rng.normal(size=(T, 35)).astype(np.float32)
                g["y"] = rng.normal(size=(T, 3)).astype(np.float32)
                g["t"] = np.arange(T) / data.RATE
                g["label"] = np.array([b"walk"] * T)
                # stair recordings come from 2 parents, listed out of order on purpose
                parent = f"{mode}_parent_{i % 2}"
                g.attrs.update(
                    mode=mode,
                    name=f"{mode}_{i:02d}",
                    parent=parent,
                    trial_start=float(100 - i),
                    speed_min=np.nan,
                    speed_max=np.nan,
                )
    return path


@pytest.fixture
def person(tmp_path):
    return make_person(tmp_path / "ABxx.h5")
