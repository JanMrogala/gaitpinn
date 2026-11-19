"""Required physics tests (PLAN.md): the swing term is ~0 on a simulated pendulum with known
torque, and the derived equation reduces to the textbook pendulum."""

import numpy as np
import sympy as sp
import torch
from scipy.integrate import solve_ivp

from gaitpinn import data, gyro, swing


def test_derived_equation_is_a_pendulum_when_thigh_is_still():
    lhs, s = swing.derive()
    q, qd, qdd, ph, phd, phdd, m, d, i_c, l_t, g = s
    still = sp.simplify(lhs.subs({ph: 0, phd: 0, phdd: 0}))
    assert sp.simplify(still - ((i_c + m * d**2) * qdd + m * g * d * sp.sin(q))) == 0


def simulate(seconds=1.0):
    """Knee pendulum under a known torque while the thigh swings; returns 100 Hz samples."""
    p = swing.body(1.75, 70.0)
    lhs, s = swing.derive()
    q, qd, qdd, ph, phd, phdd, m, d, i_c, l_t, g = s
    num = {m: p["m"], d: p["d"], i_c: p["I_c"], l_t: p["l_t"], g: swing.G}
    acc = sp.lambdify(
        (q, qd, ph, phd, phdd, sp.Symbol("rhs")),
        sp.solve(sp.Eq(lhs.subs(num), sp.Symbol("rhs")), qdd)[0],
        "numpy",
    )
    k, b, q0 = 2.0, 0.3, 0.2

    def thigh(t):  # thigh swings forward like in swing phase
        return (
            0.4 * np.sin(2 * np.pi * t),
            0.8 * np.pi * np.cos(2 * np.pi * t),
            -1.6 * np.pi**2 * np.sin(2 * np.pi * t),
        )

    def torque(t):
        return 3.0 * np.sin(3 * t) - 1.0

    def f(t, z):
        a, ad, add = thigh(t)
        rhs = torque(t) - k * (z[0] - q0) - b * z[1]
        return [z[1], acc(z[0], z[1], a, ad, add, rhs)]

    t = np.arange(0, seconds, 1 / data.RATE)
    sol = solve_ivp(f, (0, seconds), [0.3, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    a, ad, _ = thigh(t)
    return p, (k, b, q0), t, sol.y[0], a, ad, torque(t)


def test_swing_residual_is_zero_on_simulated_pendulum():
    p, (k, b, q0), t, q, ph, phd, tau = simulate()
    term = swing.SwingTerm(
        p, scale_dyn=1.0, scale_tau=1.0, init={"w": 0.0, "k": 0.0, "b": 0.0, "q0": q0}
    )
    term = term.double()
    # set softplus(k_raw) = k, softplus(b_raw) = b
    with torch.no_grad():
        term.k_raw.fill_(float(np.log(np.expm1(k))))
        term.b_raw.fill_(float(np.log(np.expm1(b))))
    angles = torch.zeros(1, 3, len(t), dtype=torch.float64)
    angles[0, 0] = torch.from_numpy(np.degrees(ph))
    angles[0, 1] = torch.from_numpy(-np.degrees(q))  # Camargo sign: negative in flexion
    r1, _ = term.residuals(
        angles,
        torch.from_numpy(tau)[None],
        torch.from_numpy(phd)[None],
        torch.zeros(1, len(swing.MUSCLES), len(t), dtype=torch.float64),
    )
    lhs_scale = np.abs(tau).max()
    assert r1.abs().max().item() < 0.02 * lhs_scale  # finite differences at 100 Hz
    # a wrong torque is clearly detected
    r1_bad, _ = term.residuals(
        angles,
        torch.from_numpy(tau + 1.0)[None],
        torch.from_numpy(phd)[None],
        torch.zeros(1, len(swing.MUSCLES), len(t), dtype=torch.float64),
    )
    assert r1_bad.abs().mean().item() > 0.9


def test_gyro_term_is_zero_when_prediction_matches_gyro():
    t = np.arange(0, 2, 0.01)
    ang = np.stack(
        [10 * np.sin(2 * np.pi * t), -30 + 25 * np.sin(2 * np.pi * t), 5 * np.cos(2 * np.pi * t)]
    )
    ref = np.gradient(ang[1:], axis=1) * data.RATE
    term = gyro.GyroTerm({"knee": 100.0, "ankle": 50.0})
    r = term(torch.tensor(ang[None]), torch.tensor(ref[None]))
    assert r.item() < 1e-3
    r_bad = term(torch.tensor(ang[None]), torch.tensor(ref[None]) + 50)
    assert r_bad.item() > 0.1


def test_swing_mask_needs_100ms():
    x = np.zeros((100, len(data.EMG) + len(data.IMU)))
    cols = gyro.gyro_index("foot")
    x[10:15, cols[0]] = 5.0  # 50 ms: too short
    x[40:60, cols[0]] = 5.0  # 200 ms: swing
    m = swing.swing_mask(x, threshold=1.0)
    assert not m[10:15].any() and m[40:60].all() and m.sum() == 20
