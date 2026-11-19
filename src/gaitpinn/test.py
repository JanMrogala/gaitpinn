"""Metrics (PLAN.md, "Metrics") and the across-people comparison.

Per run (called by train.py on its own predictions):
    rolling error, walking error W, error after switch, time to fail, ascent/descent errors.
Across runs (`python -m gaitpinn.test`): seed averaging, censoring at H, Wilcoxon test,
effect sizes, curves for the plots. Writes runs/summary/*.json for plot.py and site.py.
"""

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
