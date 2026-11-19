"""Study 2: folds, Holm, nested selection, and label-free self-calibration."""

import shutil

import h5py
import numpy as np
import torch

from gaitpinn import cross, runs
from gaitpinn import test as metrics

PEOPLE = [f"AB{i:02d}" for i in range(6, 28)]


def test_folds_cover_everyone_once_and_never_mix_roles():
    fl = cross.folds(PEOPLE)
    tested = [p for f in fl for p in f["test"]]
    assert sorted(tested) == sorted(PEOPLE) and len(tested) == len(set(tested))
    assert [len(f["test"]) for f in fl] == [6, 6, 5, 5]
    for f in fl:
        assert len(f["val"]) == 3
        assert not set(f["test"]) & set(f["val"]) and not set(f["test"]) & set(f["train"])
        assert sorted(f["test"] + f["val"] + f["train"]) == sorted(PEOPLE)


def test_holm():
    np.testing.assert_allclose(metrics.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_xselect_is_nested_per_fold(tmp_path):
    for fold, vals in [(0, [3.0, 2.0]), (1, [1.0, 5.0])]:
        for i, v in enumerate(vals):
            runs.write_run(
                tmp_path,
                {
                    "run_id": f"xtune-base-c{i:02d}-fold{fold}",
                    "kind": "xtune",
                    "status": "done",
                    "model": f"base-c{i:02d}",
                    "fold": {"fold": fold},
                    "config": {"lr": 0.001 * (i + 1), "width": 64, "weight_decay": 0.0},
                    "metrics": {"val_rmse": v},
                },
            )
    out = metrics.xselect(tmp_path)
    assert out["fold0"]["base"]["lr"] == 0.002 and out["fold1"]["base"]["lr"] == 0.001


def test_self_calibration_never_reads_labels(person, tmp_path):
    """The same calibrated weights whether or not the person's true angles exist."""
    nan = tmp_path / "nan.h5"
    shutil.copy(person, nan)
    with h5py.File(nan, "r+") as h:
        for k in h["files"]:
            h["files"][k]["y"][...] = np.nan
    cfg = {"model": "pinn", "width": 8, "levels": 3, "kernel": 3, "dropout": 0.0}
    torch.manual_seed(0)
    model = cross.XModel(cfg, [0.0, -30.0, 0.0], [15.0, 20.0, 10.0]).eval()
    from gaitpinn import swing

    sterm = swing.SwingTerm(swing.body(1.7, 70), 1.0, 1.0, {"w": 0.0, "k": -2, "b": -2, "q0": 0})
    weights = []
    for path in (person, nan):
        tp = cross.Unlabelled(path)
        setup = cross.label_free_setup(model, tp, 1.0, torch.device("cpu"))
        setup["gate"] = {"knee": True, "ankle": True}  # random data: force both terms on
        cal = cross.calibrate(model, sterm, setup, "gyro+swing", 3, 1.0, 0, torch.device("cpu"))
        weights.append([p.detach().clone() for p in cal.parameters()])
    for a, b in zip(*weights, strict=True):
        assert torch.equal(a, b)
    assert not all(
        torch.equal(a, b) for a, b in zip(weights[0], model.parameters(), strict=True)
    )  # it did change the model
