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
                met = metrics.run_metrics(stream, y + e, walk)
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
    (tmp_path / "summary" / "figures.json").write_text(json.dumps(figs))
    (tmp_path / "summary" / "people.json").write_text(json.dumps(plot.people(tmp_path, s)))
    out = tmp_path / "site"
    site.build(tmp_path, out)
    page = (out / "index.html").read_text()
    assert "Yes: pinn kept a lower" in page and "fig1" in page
    assert "AB06" in (out / "people.html").read_text()
    assert "Strongly physics-constrained model (pinn_strong" in page
