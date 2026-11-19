"""Metrics (PLAN.md, "Metrics") and the across-people comparison.

Per run (called by train.py on its own predictions):
    rolling error, walking error W, error after switch, time to fail, ascent/descent errors.
Across runs (`python -m gaitpinn.test`): seed averaging, censoring at H, Wilcoxon test,
effect sizes, curves for the plots. Writes runs/summary/*.json for plot.py and site.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gaitpinn import data

JOINTS = ["hip", "knee", "ankle"]
WINDOW_S = 2.0
HOLD_S = 2.0
FACTOR = 1.5
ASCENT = ("stairascent",)
DESCENT = ("stairdescent",)


def rmse(e: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """RMSE per joint over samples where mask is True. e: (T, 3)."""
    if mask is not None:
        e = e[mask]
    return np.sqrt(np.mean(e**2, axis=0))


def rolling(e: np.ndarray, score: np.ndarray, window_s: float = WINDOW_S) -> np.ndarray:
    """Trailing-window RMSE per joint at every sample, over scored samples only. (T, 3)."""
    n = int(round(window_s * data.RATE))
    sq = np.where(score[:, None], e**2, 0.0)
    cnt = score.astype(float)
    csq = np.cumsum(np.vstack([np.zeros((1, 3)), sq]), axis=0)
    ccnt = np.cumsum(np.concatenate([[0.0], cnt]))
    hi = np.arange(1, len(e) + 1)
    lo = np.maximum(hi - n, 0)
    num = csq[hi] - csq[lo]
    den = (ccnt[hi] - ccnt[lo])[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(den > 0, num / den, np.nan))


def time_to_fail(
    t: np.ndarray, roll: np.ndarray, w: float, hold_s: float = HOLD_S, factor: float = FACTOR
) -> float | None:
    """First t >= 0 where roll > factor * w and stays above for hold_s. None if never.
    A hold that would run past the end of the stream counts as not failed. roll: (T,)."""
    above = roll > factor * w
    n = int(round(hold_s * data.RATE))
    start = np.searchsorted(t, 0.0)
    run = 0
    for i in range(start, len(t)):
        run = run + 1 if above[i] else 0
        if run > n:  # above for the sample at i - n through i: a full hold_s
            return float(t[i - n])
    return None


def run_metrics(
    stream: dict, pred: np.ndarray, walk_test: tuple[np.ndarray, np.ndarray, dict]
) -> dict:
    """All per-run numbers. stream: data.build_stream output; pred: (T, 3) on the stream.
    walk_test: (errors on all walking test files, mask, {'levelground': (err, mask)})."""
    e = pred - stream["y"]
    score = stream["score"]
    t = stream["t"]
    post = score & (t >= 0)
    roll = rolling(e, score)
    lg_err, lg_mask = walk_test[2]["levelground"]
    w = rmse(lg_err, lg_mask)
    ttf = [time_to_fail(t, roll[:, j], w[j]) for j in range(3)]
    lab = stream["label"]
    asc = post & np.isin(lab, ASCENT)
    desc = post & np.isin(lab, DESCENT)
    return {
        "walk_error": dict(zip(JOINTS, w.tolist(), strict=True)),
        "walk_error_all_test": dict(
            zip(JOINTS, rmse(walk_test[0], walk_test[1]).tolist(), strict=True)
        ),
        "error_after_switch": dict(zip(JOINTS, rmse(e, post).tolist(), strict=True)),
        "error_ascent": dict(zip(JOINTS, rmse(e, asc).tolist(), strict=True)),
        "error_descent": dict(zip(JOINTS, rmse(e, desc).tolist(), strict=True)),
        "time_to_fail": dict(zip(JOINTS, ttf, strict=True)),
        "stairs_seconds": float(t[-1]),
    }


# ---------- across runs ----------

MODELS = ["base", "pinn"]  # the pre-registered comparison
EXTRA = ["pinn_strong"]  # added later on request, compared with base the same way
BOUT_WINDOW = (-2.0, 6.0)
BOUT_ENTRY = {"ascent": "walk-stairascent", "descent": "walk-stairdescent"}


def load(runs_dir: Path, kind: str) -> list[dict]:
    from gaitpinn import runs as runrec

    return [r for r in runrec.load_runs(runs_dir) if r["kind"] == kind and r["status"] == "done"]


def select(runs_dir: Path) -> dict:
    """Tuning: per model, the grid configuration with the lowest mean walking-validation
    angle RMSE across people. Writes runs/chosen_<model>.json and summary/tuning.json."""
    tune = [r for r in load(runs_dir, "tune") if r["person_info"]["gyro_ok"]]
    last = max(r["config"].get("round", 1) for r in tune)  # only the latest tuning round
    tune = [r for r in tune if r["config"].get("round", 1) == last]
    out = {}
    for m in MODELS:
        by_tag: dict[str, list] = {}
        cfgs = {}
        for r in tune:
            if r["model"] != m:
                continue
            tag = r["config"]["tag"]
            by_tag.setdefault(tag, []).append(r["metrics"]["val_rmse"])
            cfgs[tag] = r["config"]
        if not by_tag:
            continue
        n = max(len(v) for v in by_tag.values())
        table = {
            t: {"mean_val_rmse": float(np.mean(v)), "people": len(v)} for t, v in by_tag.items()
        }
        full = {t: s for t, s in table.items() if s["people"] == n}
        best = min(full, key=lambda t: full[t]["mean_val_rmse"])
        grid_keys = ["lr", "width", "weight_decay", "lam"]
        chosen = {k: cfgs[best][k] for k in grid_keys if k in cfgs[best]}
        (runs_dir / f"chosen_{m}.json").write_text(json.dumps(chosen, indent=2))
        out[m] = {
            "chosen_tag": best,
            "chosen": chosen,
            "table": table,
            "run_ids": sorted(r["run_id"] for r in tune if r["model"] == m),
        }
    (runs_dir / "summary").mkdir(exist_ok=True)
    (runs_dir / "summary" / "tuning.json").write_text(json.dumps(out, indent=2))
    return out


def wilcoxon(base: np.ndarray, pinn: np.ndarray, seed: int = 0) -> dict:
    """Paired two-sided Wilcoxon on base - pinn, rank-biserial r, Hodges-Lehmann + bootstrap."""
    from scipy import stats

    d = np.asarray(base) - np.asarray(pinn)
    if not np.any(d != 0):  # identical for everyone (e.g. both never fail): no evidence
        return {
            "n": int(len(d)),
            "p": 1.0,
            "statistic": float("nan"),
            "rank_biserial_r": 0.0,
            "hodges_lehmann": 0.0,
            "hl_ci95": [0.0, 0.0],
            "median_diff": 0.0,
        }
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    r = float((w_pos - w_neg) / (w_pos + w_neg)) if len(nz) else 0.0

    def hl(x):
        i, j = np.triu_indices(len(x))
        return float(np.median((x[i] + x[j]) / 2))

    rng = np.random.default_rng(seed)
    boots = [hl(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
    return {
        "n": int(len(d)),
        "p": float(res.pvalue),
        "statistic": float(res.statistic),
        "rank_biserial_r": r,
        "hodges_lehmann": hl(d),
        "hl_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "median_diff": float(np.median(d)),
    }


def verdict(w: dict, name: str = "pinn") -> str:
    if w["p"] < 0.05 and w["rank_biserial_r"] > 0:
        return f"{name} is better"
    if w["p"] < 0.05 and w["rank_biserial_r"] < 0:
        return "base is better"
    return "no clear difference"


def summary(runs_dir: Path) -> dict:
    """Final runs -> summary/results.json (numbers, test, curves) for plot.py and site.py."""
    final_all = load(runs_dir, "final")
    excluded = sorted({r["person"] for r in final_all if not r["person_info"]["gyro_ok"]})
    final = [r for r in final_all if r["person"] not in excluded]
    people = sorted({r["person"] for r in final})
    models = MODELS + [m for m in EXTRA if any(r["model"] == m for r in final)]
    by = {
        (m, p): [r for r in final if r["model"] == m and r["person"] == p]
        for m in models
        for p in people
    }
    # the pre-registered comparison depends only on base and pinn runs
    complete = [p for p in people if all(len(by[(m, p)]) == 5 for m in MODELS)]
    models = MODELS + [m for m in models[2:] if all(len(by[(m, p)]) == 5 for p in complete)]
    H = min(
        r["metrics"]["stairs_seconds"]
        for r in final
        if r["person"] in complete and r["model"] in MODELS
    )

    def seed_mean(m, p, key):
        return {j: float(np.mean([r["metrics"][key][j] for r in by[(m, p)]])) for j in JOINTS}

    def ttf(m, p):
        out = {}
        for j in JOINTS:
            v = [
                min(r["metrics"]["time_to_fail"][j], H)
                if r["metrics"]["time_to_fail"][j] is not None
                else H
                for r in by[(m, p)]
            ]
            out[j] = float(np.median(v))
        return out

    per = {
        p: {
            m: {
                "error_after_switch": seed_mean(m, p, "error_after_switch"),
                "walk_error": seed_mean(m, p, "walk_error"),
                "walk_error_all_test": seed_mean(m, p, "walk_error_all_test"),
                "error_ascent": seed_mean(m, p, "error_ascent"),
                "error_descent": seed_mean(m, p, "error_descent"),
                "time_to_fail": ttf(m, p),
                "run_ids": sorted(r["run_id"] for r in by[(m, p)]),
            }
            for m in models
        }
        for p in complete
    }

    def vec(m, key, j=None):
        if j is None:
            return np.array([np.mean([per[p][m][key][k] for k in JOINTS]) for p in complete])
        return np.array([per[p][m][key][j] for p in complete])

    primary = wilcoxon(vec("base", "error_after_switch"), vec("pinn", "error_after_switch"))
    secondary = {}
    for key in ["error_after_switch", "error_ascent", "error_descent", "time_to_fail"]:
        for j in JOINTS:
            secondary[f"{key}/{j}"] = wilcoxon(vec("base", key, j), vec("pinn", key, j))
    numbers = {
        m: {
            "error_after_switch": {
                j: float(vec(m, "error_after_switch", j).mean()) for j in JOINTS
            },
            "error_after_switch_mean3": float(vec(m, "error_after_switch").mean()),
            "walk_error": {j: float(vec(m, "walk_error", j).mean()) for j in JOINTS},
            "error_ascent": {j: float(vec(m, "error_ascent", j).mean()) for j in JOINTS},
            "error_descent": {j: float(vec(m, "error_descent", j).mean()) for j in JOINTS},
            "time_to_fail_median": {j: float(np.median(vec(m, "time_to_fail", j))) for j in JOINTS},
            "never_failed": {j: int(np.sum(vec(m, "time_to_fail", j) >= H)) for j in JOINTS},
        }
        for m in models
    }
    extra = {}
    for m in models[2:]:
        prim = wilcoxon(vec("base", "error_after_switch"), vec(m, "error_after_switch"))
        sec = {
            f"{key}/{j}": wilcoxon(vec("base", key, j), vec(m, key, j))
            for key in ["error_after_switch", "error_ascent", "error_descent", "time_to_fail"]
            for j in JOINTS
        }
        extra[m] = {
            "primary": prim,
            "verdict": verdict(prim, m),
            "secondary": sec,
            "val_rmse": {
                mm: float(
                    np.mean([r["metrics"]["val_rmse"] for p in complete for r in by[(mm, p)]])
                )
                for mm in ("base", m)
            },
        }
    curves = error_curves(runs_dir, by, complete, H, models)
    bouts = bout_curves(runs_dir, by, complete, models)
    # sensitivity only (not the verdict): all people, including those excluded by the gyro check
    allp = sorted({r["person"] for r in final_all})
    sens = {
        m: np.array(
            [
                np.mean(
                    [
                        np.mean(list(r["metrics"]["error_after_switch"].values()))
                        for r in final_all
                        if r["person"] == p and r["model"] == m
                    ]
                )
                for p in allp
            ]
        )
        for m in models
    }
    configs = {
        m: {k: by[(m, complete[0])][0]["config"].get(k) for k in ("lr", "width", "lam")}
        for m in models
    }
    out = {
        "models": models,
        "configs": configs,
        "extra": extra,
        "sensitivity_all_people": wilcoxon(sens["base"], sens["pinn"]) | {"people": allp},
        "people": complete,
        "excluded": excluded,
        "H": H,
        "per_person": per,
        "numbers": numbers,
        "primary": primary,
        "verdict": verdict(primary),
        "secondary": secondary,
        "curves": curves,
        "bouts": bouts,
    }
    (runs_dir / "summary").mkdir(exist_ok=True)
    (runs_dir / "summary" / "results.json").write_text(json.dumps(out))
    return out


def _stream(runs_dir: Path, run_id: str) -> dict:
    z = np.load(runs_dir / run_id / "stream.npz")
    return {k: z[k] for k in z.files}


def error_curves(runs_dir, by, people, H, models=MODELS) -> dict:
    """Rolling error on t in [-30, H]: per person seed-mean, then mean and t-based 95% CI."""
    from scipy import stats

    t_grid = np.round(np.arange(-30.0, H + 1e-9, 1 / data.RATE), 2)
    out = {"t": t_grid[::10].tolist()}
    for m in models:
        per_person = []
        for p in people:
            seeds = []
            for r in by[(m, p)]:
                s = _stream(runs_dir, r["run_id"])
                roll = rolling(s["pred"].astype(float) - s["y"], s["score"])
                i0 = int(np.sum(s["t"] < 0))  # index of t = 0; samples are 1/RATE apart
                idx = i0 + np.round(t_grid * data.RATE).astype(int)
                seeds.append(roll[idx])
            per_person.append(np.nanmean(seeds, axis=0))
        a = np.array(per_person)  # (people, T, 3)
        mean = np.nanmean(a, axis=0)
        se = np.nanstd(a, axis=0, ddof=1) / np.sqrt(a.shape[0])
        half = stats.t.ppf(0.975, a.shape[0] - 1) * se
        out[m] = {
            j: {
                "mean": mean[::10, k].tolist(),
                "lo": (mean - half)[::10, k].tolist(),
                "hi": (mean + half)[::10, k].tolist(),
            }
            for k, j in enumerate(JOINTS)
        }
    return out


def bout_curves(runs_dir, by, people, models=MODELS) -> dict:
    """Error vs time since each stair entry: sqrt(mean over bouts of e^2) per person, then
    mean over people. Ascent and descent separately."""
    lo, hi = (int(round(v * data.RATE)) for v in BOUT_WINDOW)
    offs = np.arange(lo, hi + 1)
    out = {"t": (offs / data.RATE).tolist()}
    for m in models:
        out[m] = {}
        for kind, entry in BOUT_ENTRY.items():
            per_person = []
            for p in people:
                sq = []
                for r in by[(m, p)]:
                    s = _stream(runs_dir, r["run_id"])
                    lab = s["label"].astype(str)
                    e2 = (s["pred"].astype(float) - s["y"]) ** 2
                    starts = np.flatnonzero(
                        (lab == entry) & (np.roll(lab, 1) != entry) & (s["t"] >= 0)
                    )
                    for st in starts:
                        idx = st + offs
                        if idx[0] < 0 or idx[-1] >= len(lab):
                            continue
                        sq.append(np.where(s["score"][idx, None], e2[idx], np.nan))
                per_person.append(np.sqrt(np.nanmean(sq, axis=0)))
            a = np.nanmean(np.array(per_person), axis=0)
            out[m][kind] = {j: a[:, k].tolist() for k, j in enumerate(JOINTS)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["select", "summary", "xselect", "xsummary"])
    ap.add_argument("--runs", type=Path, required=True)
    a = ap.parse_args()
    if a.action == "xselect":
        for f, v in xselect(a.runs).items():
            print(f, {k: v[k] for k in v if not k.startswith("_")})
        return
    if a.action == "xsummary":
        s = xsummary(a.runs)
        print(f"study 2: people {len(s['people'])}, H {s['H']:.1f} s")
        for k, w in s["tests"].items():
            print(k, w["verdict"], f"p={w['p']:.4f} p_holm={w['p_holm']:.4f}")
        return
    if a.action == "select":
        for m, s in select(a.runs).items():
            print(m, s["chosen_tag"], s["chosen"], f"{s['table'][s['chosen_tag']]}")
    else:
        s = summary(a.runs)
        print(f"people {len(s['people'])}, excluded {s['excluded']}, H {s['H']:.1f} s")
        print("verdict:", s["verdict"], s["primary"])


if __name__ == "__main__":
    main()


# ---------- study 2 (PLAN2.md) ----------

XMODELS = ["base", "pinn", "pinn_cal", "base_cal"]
XTESTS = [("base", "pinn"), ("base", "pinn_cal"), ("pinn", "pinn_cal")]


def xselect(runs_dir: Path) -> dict:
    """Per fold: the configuration with the lowest validation score among that fold's tuning
    runs (nested); with --cal, also the calibration setting from that fold's xcal runs."""
    from gaitpinn import runs as runrec

    recs = [r for r in runrec.load_runs(runs_dir) if r["status"] == "done"]
    out = {}
    for r in recs:
        if r["kind"] != "xtune":
            continue
        f = f"fold{r['fold']['fold']}"
        m = r["model"].split("-")[0]
        cur = out.setdefault(f, {}).get(f"_{m}")
        if cur is None or r["metrics"]["val_rmse"] < cur[0]:
            keys = ["lr", "width", "weight_decay", "lam"]
            out[f][f"_{m}"] = (r["metrics"]["val_rmse"], r["run_id"])
            out[f][m] = {k: r["config"][k] for k in keys if k in r["config"]}
    for r in recs:
        if r["kind"] != "xcal":
            continue
        f = f"fold{r['fold']['fold']}"
        res = r["metrics"]["cal"]
        best = min(res, key=res.get)
        from gaitpinn import cross

        s = next(s for s in cross.cal_settings(r["model"]) if s["name"] == best)
        out[f][f"cal_{r['model']}"] = s
        out[f][f"_cal_{r['model']}"] = (res[best], r["run_id"])
    (runs_dir / "chosen_x.json").write_text(json.dumps(out, indent=2))
    return out


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def xsummary(runs_dir: Path) -> dict:
    """Final study-2 runs -> summary/results2.json."""
    from gaitpinn import runs as runrec

    final = [
        r for r in runrec.load_runs(runs_dir) if r["kind"] == "xfinal" and r["status"] == "done"
    ]
    models = [m for m in XMODELS if any(r["model"] == m for r in final)]
    people = sorted({r["person"] for r in final})
    by = {
        (m, p): [r for r in final if r["model"] == m and r["person"] == p]
        for m in models
        for p in people
    }
    complete = [p for p in people if all(len(by[(m, p)]) == 5 for m in models)]
    H = min(r["metrics"]["stairs_seconds"] for r in final if r["person"] in complete)
    fold_of = {r["person"]: r["fold"]["fold"] for r in final}

    def seed_mean(m, p, key):
        return {j: float(np.mean([r["metrics"][key][j] for r in by[(m, p)]])) for j in JOINTS}

    def ttf(m, p):
        return {
            j: float(
                np.median(
                    [
                        min(v, H) if (v := r["metrics"]["time_to_fail"][j]) is not None else H
                        for r in by[(m, p)]
                    ]
                )
            )
            for j in JOINTS
        }

    keys = [
        "error_after_switch",
        "walk_error",
        "walk_error_all_test",
        "error_ascent",
        "error_descent",
    ]
    per = {
        p: {
            m: {
                **{k: seed_mean(m, p, k) for k in keys},
                "time_to_fail": ttf(m, p),
                "run_ids": sorted(r["run_id"] for r in by[(m, p)]),
            }
            for m in models
        }
        for p in complete
    }

    def vec(m, key, j=None, ppl=None):
        ppl = ppl or complete
        if j is None:
            return np.array([np.mean([per[p][m][key][k] for k in JOINTS]) for p in ppl])
        return np.array([per[p][m][key][j] for p in ppl])

    tests = {}
    raw = []
    for a, b in XTESTS:
        if a in models and b in models:
            w = wilcoxon(vec(a, "error_after_switch"), vec(b, "error_after_switch"))
            tests[f"{a} vs {b}"] = w
            raw.append(w["p"])
    for (name, w), padj in zip(tests.items(), holm(raw), strict=True):
        a, b = name.split(" vs ")
        w["p_holm"] = padj
        w["verdict"] = (
            f"{b} is better"
            if padj < 0.05 and w["rank_biserial_r"] > 0
            else f"{a} is better"
            if padj < 0.05 and w["rank_biserial_r"] < 0
            else "no clear difference"
        )
        w["per_fold_median"] = {
            f: float(
                np.median(
                    [
                        per[p][a]["error_after_switch"][j] - per[p][b]["error_after_switch"][j]
                        for p in complete
                        if fold_of[p] == f
                        for j in JOINTS
                    ]
                )
            )
            for f in sorted(set(fold_of.values()))
        }
    secondary = {}
    pairs = [("base", "base_cal"), ("base_cal", "pinn_cal")] + XTESTS
    for a, b in pairs:
        if a not in models or b not in models:
            continue
        for key in ["error_after_switch", "error_ascent", "error_descent", "time_to_fail"]:
            for j in JOINTS:
                secondary[f"{a} vs {b}/{key}/{j}"] = wilcoxon(vec(a, key, j), vec(b, key, j))
        if (a, b) in [("base", "base_cal"), ("base_cal", "pinn_cal")]:
            secondary[f"{a} vs {b}/primary"] = wilcoxon(
                vec(a, "error_after_switch"), vec(b, "error_after_switch")
            )
    gyro_ok = []
    checks = runs_dir / "checks" / "checks.jsonl"
    if checks.exists():
        ok = {
            json.loads(line)["person"]: json.loads(line)["gyro_ok"]
            for line in checks.read_text().splitlines()
        }
        gyro_ok = [p for p in complete if ok.get(p)]
        for a, b in XTESTS:
            if a in models and b in models and len(gyro_ok) > 5:
                secondary[f"{a} vs {b}/primary/gyro-check people"] = wilcoxon(
                    vec(a, "error_after_switch", ppl=gyro_ok),
                    vec(b, "error_after_switch", ppl=gyro_ok),
                )
    numbers = {
        m: {
            "error_after_switch": {
                j: float(vec(m, "error_after_switch", j).mean()) for j in JOINTS
            },
            "error_after_switch_mean3": float(vec(m, "error_after_switch").mean()),
            "walk_error": {j: float(vec(m, "walk_error", j).mean()) for j in JOINTS},
            "error_ascent": {j: float(vec(m, "error_ascent", j).mean()) for j in JOINTS},
            "error_descent": {j: float(vec(m, "error_descent", j).mean()) for j in JOINTS},
            "time_to_fail_median": {j: float(np.median(vec(m, "time_to_fail", j))) for j in JOINTS},
        }
        for m in models
    }
    cal_info = {}
    for m in ("pinn_cal", "base_cal"):
        rs = [r for r in final if r["model"] == m]
        if rs:
            cal_info[m] = {
                "settings_by_fold": {str(r["fold"]["fold"]): r["calibration"]["name"] for r in rs},
                "joints_gated_off": int(sum(not v for r in rs for v in r["gate"].values())),
                "joint_checks": int(sum(len(r["gate"]) for r in rs)),
            }
    curves = error_curves(runs_dir, by, complete, H, models)
    bouts = bout_curves(runs_dir, by, complete, models)
    out = {
        "study": 2,
        "models": models,
        "people": complete,
        "H": H,
        "folds": fold_of,
        "per_person": per,
        "numbers": numbers,
        "tests": tests,
        "secondary": secondary,
        "gyro_check_people": gyro_ok,
        "calibration": cal_info,
        "curves": curves,
        "bouts": bouts,
    }
    (runs_dir / "summary").mkdir(exist_ok=True)
    (runs_dir / "summary" / "results2.json").write_text(json.dumps(out))
    return out
