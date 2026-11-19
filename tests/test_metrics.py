import numpy as np

from gaitpinn import data
from gaitpinn import test as metrics


def test_rolling_is_trailing_and_skips_unscored():
    e = np.zeros((1000, 3))
    e[500:, 0] = 3.0
    score = np.ones(1000, bool)
    r = metrics.rolling(e, score)
    assert r[499, 0] == 0  # nothing from the future
    assert np.isclose(r[699, 0], 3.0)  # window fully inside the error region after 2 s
    score[:500] = False
    assert np.isclose(metrics.rolling(e, score)[520, 0], 3.0)  # only scored samples count


def test_time_to_fail_needs_a_full_hold():
    t = (np.arange(3000) - 1000) / data.RATE  # -10 s .. 20 s
    roll = np.ones(3000)
    roll[1500:1600] = 5  # 1 s above: not a failure
    roll[2000:2300] = 5  # 3 s above: failure at t = 10 s
    assert metrics.time_to_fail(t, roll, w=1.0) == 10.0
    roll2 = np.ones(3000)
    roll2[2900:] = 5  # above for the last 1 s only: the hold runs off the end
    assert metrics.time_to_fail(t, roll2, w=1.0) is None
    roll3 = np.full(3000, 5.0)  # above everywhere, including before the switch
    assert metrics.time_to_fail(t, roll3, w=1.0) == 0.0


def test_wilcoxon_direction_and_verdict():
    rng = np.random.default_rng(0)
    base = 10 + rng.normal(0, 1, 22)
    w = metrics.wilcoxon(base, base - 2)  # pinn 2 deg lower for everyone
    assert w["p"] < 0.05 and w["rank_biserial_r"] == 1.0 and np.isclose(w["hodges_lehmann"], 2)
    assert metrics.verdict(w) == "pinn is better"
    assert metrics.verdict(metrics.wilcoxon(base, base + 2)) == "base is better"
    noise = metrics.wilcoxon(base, base + rng.normal(0, 0.01, 22))
    assert metrics.verdict(noise) == "no clear difference" or noise["p"] < 0.05


def test_rmse_mask():
    e = np.array([[1.0, 2, 3], [100, 100, 100]])
    np.testing.assert_allclose(metrics.rmse(e, np.array([True, False])), [1, 2, 3])
