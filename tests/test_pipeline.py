"""End to end on fake final runs: summary -> figures -> website (no real data)."""

import json

import numpy as np

from gaitpinn import plot, runs, site
from gaitpinn import test as metrics


def fake_runs(d, people=tuple(f"AB{i:02d}" for i in range(1, 10)), bad=("AB06",)):
    rng = np.random.default_rng(0)
    t = (np.arange(9000) - 3000) / 100
    lab = np.where(t < 0, "walk", "idle").astype(object)
    for s0 in range(0, 6000, 1500):
        lab[3000 + s0 + 100 : 3000 + s0 + 300] = "walk-stairascent"
        lab[3000 + s0 + 300 : 3000 + s0 + 500] = "stairascent"
        lab[3000 + s0 + 800 : 3000 + s0 + 1000] = "walk-stairdescent"
    for p in people:
        for m, extra in [("base", 3.0), ("pinn", 2.0), ("pinn_strong", 3.5)]:
            for seed in range(5):
                y = rng.normal(0, 10, (9000, 3))
                e = rng.normal(0, 1, (9000, 3)) * np.where(t[:, None] < 0, 1, extra)
                score = np.ones(9000, bool)
                stream = {
                    "t": t,
                    "y": y,
                    "score": score,
                    "label": lab.astype(str),
                    "file": np.zeros(9000, int),
                }
                walk = (e[:3000], score[:3000], {"levelground": (e[:3000], score[:3000])})
                met = metrics.run_metrics(stream, y + e, walk) | {"val_rmse": 2.0}
                rid = f"final-{m}-{p}-s{seed}-jx"
                rec = {
                    "run_id": rid,
                    "kind": "final",
                    "status": "done",
                    "model": m,
                    "person": p,
                    "seed": seed,
                    "person_info": {"gyro_ok": p not in bad},
                    "config": {"lr": 1e-3, "width": 64, "lam": 0.1 if m == "pinn_strong" else 0},
                    "metrics": met,
                }
                out = runs.write_run(d, rec)
                np.savez_compressed(
                    out / "stream.npz",
                    t=t,
                    y=y,
                    pred=y + e,
                    score=score,
                    label=lab.astype("S"),
                    file_idx=np.zeros(9000, np.int16),
                )


def test_summary_figures_site(tmp_path):
    fake_runs(tmp_path)
    s = metrics.summary(tmp_path)
    assert s["excluded"] == ["AB06"] and len(s["people"]) == 8
    assert s["verdict"] == "pinn is better"  # pinn errors are lower by construction
    assert len(s["sensitivity_all_people"]["people"]) == 9
    assert s["extra"]["pinn_strong"]["verdict"] == "base is better"
    s = json.loads((tmp_path / "summary" / "results.json").read_text())
    figs = {"error_over_time": plot.error_over_time(s), "error_by_bout": plot.error_by_bout(s)}
    figs["error_over_time_pinn_strong"] = plot.error_over_time(s, models=["base", "pinn_strong"])
    figs["error_by_bout_pinn_strong"] = plot.error_by_bout(s, models=["base", "pinn_strong"])
    (tmp_path / "summary" / "figures.json").write_text(json.dumps(figs))
    (tmp_path / "summary" / "people.json").write_text(json.dumps(plot.people(tmp_path, s)))
    out = tmp_path / "site"
    site.build(tmp_path, out)
    page = (out / "index.html").read_text()
    assert "Yes: pinn kept a lower" in page and "fig1" in page
    assert "AB06" in (out / "people.html").read_text()
    assert "Added after the main result" in page and "error_over_time_pinn_strong" in page


def _core(s):
    keep = ["people", "H", "primary", "verdict", "secondary"]
    out = {k: s[k] for k in keep}
    out["numbers"] = {m: s["numbers"][m] for m in ("base", "pinn")}
    out["curves"] = {k: s["curves"][k] for k in ("t", "base", "pinn")}
    return json.dumps(out, sort_keys=True)


def test_extra_model_never_changes_the_preregistered_result(tmp_path):
    import shutil

    fake_runs(tmp_path)
    ref = None
    for d in sorted(tmp_path.glob("final-pinn_strong-*")):  # 1) without any pinn_strong run
        shutil.move(d, tmp_path.parent / ("aside_" + d.name))
    ref = _core(metrics.summary(tmp_path))
    for d in sorted(tmp_path.parent.glob("aside_final-pinn_strong-*")):  # 2) all present
        shutil.move(d, tmp_path / d.name.removeprefix("aside_"))
    assert _core(metrics.summary(tmp_path)) == ref
    shutil.rmtree(tmp_path / "final-pinn_strong-AB01-s4-jx")  # 3) one strong run missing
    s = metrics.summary(tmp_path)
    assert _core(s) == ref and "pinn_strong" not in s["models"]


def fake_xruns(d, people=tuple(f"AB{i:02d}" for i in range(1, 11))):
    rng = np.random.default_rng(1)
    t = (np.arange(6000) - 3000) / 100
    lab = np.where(t < 0, "walk", "stairascent").astype(object)
    lab[3000:3200] = "walk-stairascent"
    for i, p in enumerate(people):
        for m, extra in [("base", 3.0), ("pinn", 3.0), ("pinn_cal", 2.0), ("base_cal", 3.0)]:
            for seed in range(5):
                y = rng.normal(0, 10, (6000, 3))
                e = rng.normal(0, 1, (6000, 3)) * np.where(t[:, None] < 0, 1, extra)
                score = np.ones(6000, bool)
                stream = {
                    "t": t,
                    "y": y,
                    "score": score,
                    "label": lab.astype(str),
                    "file": np.zeros(6000, int),
                }
                walk = (e[:3000], score[:3000], {"levelground": (e[:3000], score[:3000])})
                met = metrics.run_metrics(stream, y + e, walk) | {"val_rmse": 2.0}
                cal = {"name": "gyro-s100-mu1"} if m.endswith("_cal") else None
                rec = {
                    "run_id": f"xfinal-{m}-{p}-s{seed}-jx",
                    "kind": "xfinal",
                    "status": "done",
                    "model": m,
                    "person": p,
                    "seed": seed,
                    "fold": {"fold": i % 4},
                    "calibration": cal,
                    "gate": {"knee": True, "ankle": i % 3 > 0} if cal else None,
                    "metrics": met,
                }
                out = runs.write_run(d, rec)
                np.savez_compressed(
                    out / "stream.npz",
                    t=t,
                    y=y,
                    pred=y + e,
                    score=score,
                    label=lab.astype("S"),
                    file_idx=np.zeros(6000, np.int16),
                )


def test_study2_summary_and_page(tmp_path):
    fake_xruns(tmp_path)
    s = metrics.xsummary(tmp_path, require=10)
    assert set(s["tests"]) == {"base vs pinn", "base vs pinn_cal", "pinn vs pinn_cal"}
    assert s["tests"]["base vs pinn_cal"]["verdict"] == "pinn_cal is better"
    assert s["tests"]["base vs pinn"]["verdict"] == "no clear difference"
    assert all(w["p_holm"] >= w["p"] for w in s["tests"].values())
    assert s["calibration"]["pinn_cal"]["joint_checks"] == 2 * 10 * 5
    s = json.loads((tmp_path / "summary" / "results2.json").read_text())
    figs = {
        "error_over_time": plot.error_over_time(s, models=["base", "pinn", "pinn_cal"]),
        "error_by_bout": plot.error_by_bout(s, models=["base", "pinn", "pinn_cal"]),
        "control": plot.error_over_time(s, models=["base", "base_cal", "pinn_cal"]),
    }
    (tmp_path / "summary" / "figures2.json").write_text(json.dumps(figs))
    site.build(tmp_path, tmp_path / "site")
    page = (tmp_path / "site" / "study2.html").read_text()
    assert "pinn_cal is better" in page and "Holm-adjusted" in page and "fold 3" in page


def fake_gruns(d, people=tuple(f"AB{i:02d}" for i in range(1, 13))):
    rng = np.random.default_rng(2)
    t = (np.arange(4000) - 2000) / 100
    lab = np.where(t < 0, "walk", "stairascent").astype(object)
    lab[2000:2200] = "walk-stairascent"
    for i, p in enumerate(people):
        for share in (0.0, 0.5, 0.9, 0.99):
            models = [("base", 3.0), ("pinn", 2.0 if share == 0.99 else 3.0)]
            if share > 0:
                models.append(("pinn_lab", 3.0))
            for m, extra in models:
                for seed in range(5):
                    y = rng.normal(0, 10, (4000, 3))
                    e = rng.normal(0, 1, (4000, 3)) * np.where(t[:, None] < 0, 1, extra)
                    score = np.ones(4000, bool)
                    stream = {"t": t, "y": y, "score": score, "label": lab.astype(str)}
                    stream["file"] = np.zeros(4000, int)
                    walk = (e[:2000], score[:2000], {"levelground": (e[:2000], score[:2000])})
                    met = metrics.run_metrics(stream, y + e, walk) | {"val_rmse": 2.0}
                    rec = {
                        "run_id": f"gfinal-{m}-s{round(share * 100):02d}-{p}-s{seed}-jx",
                        "kind": "gfinal",
                        "status": "done",
                        "model": m,
                        "person": p,
                        "seed": seed,
                        "share": share,
                        "fold": {"fold": i % 4},
                        "config": {"lr": 1e-3, "width": 64, "lam": 0.01},
                        "metrics": met,
                    }
                    out = runs.write_run(d, rec)
                    np.savez_compressed(
                        out / "stream.npz",
                        t=t,
                        y=y,
                        pred=y + e,
                        score=score,
                        label=lab.astype("S"),
                        file_idx=np.zeros(4000, np.int16),
                    )


def test_study3_summary_and_page(tmp_path):
    fake_gruns(tmp_path)
    s = metrics.gsummary(tmp_path, require=12)
    assert list(s["shares"]) == ["s00", "s50", "s90", "s99"]
    assert "p_holm" not in s["shares"]["s00"]["test"]  # 0% is not in the primary family
    assert s["shares"]["s99"]["test"]["verdict"] == "pinn is better"
    assert s["shares"]["s50"]["test"]["verdict"] in ("inconclusive", "equivalent within ±0.5°")
    assert "pinn_lab vs pinn/primary" in s["shares"]["s99"]["secondary"]
    assert s["trend"]["n"] == 12
    s = json.loads((tmp_path / "summary" / "results3.json").read_text())
    figs = {"error_vs_share": plot.error_vs_share(s)}
    for t, b in s["shares"].items():
        figs[f"error_over_time_{t}"] = plot.error_over_time(b, models=["base", "pinn"])
    (tmp_path / "summary" / "figures3.json").write_text(json.dumps(figs))
    site.build(tmp_path, tmp_path / "site")
    page = (tmp_path / "site" / "study3.html").read_text()
    assert "99% missing" in page and "pinn is better" in page and "error_vs_share" in page


def test_study3_verdict_rules():
    w = {"p_holm": 0.01, "rank_biserial_r": 0.6, "hl_ci95": [0.2, 0.9]}
    assert metrics.gverdict(w, {0: 0.3, 1: 0.2, 2: 0.1, 3: -0.1}) == "pinn is better"
    assert metrics.gverdict(w, {0: 0.3, 1: -0.2, 2: 0.1, 3: -0.1}) == "pinn is better (pooled only)"
    w = {"p_holm": 0.3, "rank_biserial_r": 0.1, "hl_ci95": [-0.2, 0.3]}
    assert metrics.gverdict(w, {}) == "equivalent within ±0.5°"
    w = {"p_holm": 0.3, "rank_biserial_r": 0.1, "hl_ci95": [-0.2, 0.7]}
    assert metrics.gverdict(w, {}) == "inconclusive"
