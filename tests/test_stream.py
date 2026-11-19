import h5py
import numpy as np

from gaitpinn import data


def test_stream_layout(person):
    with h5py.File(person) as h:
        s = data.split_files(data.file_modes(h))
        st = data.build_stream(h, s, walk_s=30.0)
        modes = data.file_modes(h)
        order = data.stair_order(h, s["stairs"])
        parents = [h["files"][k].attrs["parent"] for k in order]
        starts = [h["files"][k].attrs["trial_start"] for k in order]
    # 30 s of level-ground walking test data before t = 0, then all stair files in order
    assert st["walk_s"] == 30.0 and st["t"][0] == -30.0
    pre = {st["files"][i] for i in np.unique(st["file"][st["t"] < 0])}
    assert all(modes[k] == "levelground" and k in s["test"] for k in pre)
    post = [st["files"][i] for i in np.unique(st["file"][st["t"] >= 0])]
    assert post == order and sorted(post) == sorted(s["stairs"])
    # recorded order: by parent, then by start time within a parent
    assert parents == sorted(parents)
    for p in set(parents):
        st_p = [t for t, q in zip(starts, parents, strict=True) if q == p]
        assert st_p == sorted(st_p)
    # the first 0.5 s of each file is context only
    for i in range(len(st["files"])):
        idx = np.flatnonzero(st["file"] == i)
        assert not st["score"][idx[:50]].any() and st["score"][idx[50:]].all()
