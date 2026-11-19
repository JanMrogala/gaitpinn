"""Swing physics term (knee): shank + foot as one pendulum hanging from the moving thigh.

Model (PLAN.md):
- Sagittal plane. Thigh angle from vertical `ph` (flexion forward positive), knee flexion `q`.
  The shank's angle from vertical is `ph - q`.
- The knee joint moves on a circle of radius `l_t` around the hip. Hip linear acceleration is
  neglected (it is not measured on this side).
- The shank + foot body has mass `m`, centre of mass at distance `d` below the knee and inertia
  `I_c` about that centre (Winter's body-segment tables, scaled by height and mass).
- Euler-Lagrange in `q` gives `lhs(q, qd, qdd, ph, phd, phdd)`. The residuals are
    1. lhs - (tau - k (q - q0) - b qd)    [tau: the network's knee torque output, flexion +]
    2. tau - sum_m w_m a_m                 [a_m: EMG activations]
  Both are used during swing only, and both are divided by a fixed scale (unit-free).

The equation is derived with sympy and turned into a torch function, never typed by hand.
"""

import functools

import numpy as np
import sympy as sp
import torch
from torch import nn

from gaitpinn import data

G = 9.81
# Winter (2009) segment parameters, as fractions of body mass or height.
SHANK_MASS, FOOT_MASS = 0.0465, 0.0145
THIGH_LEN, SHANK_LEN, FOOT_LEN = 0.245, 0.246, 0.152  # x height
SHANK_COM, SHANK_RG = 0.433, 0.302  # from knee, radius of gyration about COM (x shank length)
FOOT_RG = 0.475  # x foot length; foot COM placed at the ankle, a pendulum approximation

FLEXORS = ["bicepsfemoris", "semitendinosus", "gracilis", "gastrocmed"]
EXTENSORS = ["vastusmedialis", "vastuslateralis", "rectusfemoris"]
MUSCLES = FLEXORS + EXTENSORS
ACT_TAU = 0.040  # s, first-order activation filter


def body(height_m: float, mass_kg: float) -> dict:
    """Shank + foot pendulum parameters from body-segment tables."""
    ls, lf = SHANK_LEN * height_m, FOOT_LEN * height_m
    ms, mf = SHANK_MASS * mass_kg, FOOT_MASS * mass_kg
    m = ms + mf
    d = (ms * SHANK_COM * ls + mf * ls) / m
    i_c = (
        ms * (SHANK_RG * ls) ** 2
        + ms * (SHANK_COM * ls - d) ** 2
        + mf * (FOOT_RG * lf) ** 2
        + mf * (ls - d) ** 2
    )
    return {"m": m, "d": d, "I_c": i_c, "l_t": THIGH_LEN * height_m}


@functools.cache
def derive():
    """Return (sympy lhs expression, its symbols) of the Euler-Lagrange equation in q."""
    t = sp.symbols("t")
    m, d, i_c, l_t, g = sp.symbols("m d I_c l_t g", positive=True)
    q, ph = sp.Function("q")(t), sp.Function("ph")(t)
    knee = sp.Matrix([l_t * sp.sin(ph), -l_t * sp.cos(ph)])
    shank = ph - q
    com = knee + d * sp.Matrix([sp.sin(shank), -sp.cos(shank)])
    v = com.diff(t)
    kinetic = m * (v.dot(v)) / 2 + i_c * shank.diff(t) ** 2 / 2
    potential = m * g * com[1]
    lag = kinetic - potential
    lhs = (lag.diff(q.diff(t))).diff(t) - lag.diff(q)
    s = sp.symbols("q qd qdd ph phd phdd")
    subs = {
        q.diff(t, 2): s[2],
        ph.diff(t, 2): s[5],
        q.diff(t): s[1],
        ph.diff(t): s[4],
        q: s[0],
        ph: s[3],
    }
    lhs = sp.simplify(lhs.subs(subs))
    return lhs, (*s, m, d, i_c, l_t, g)


@functools.cache
def lhs_function():
    """Torch-callable lhs(q, qd, qdd, ph, phd, phdd, m, d, I_c, l_t, g)."""
    lhs, syms = derive()
    return sp.lambdify(syms, lhs, modules=[{"sin": torch.sin, "cos": torch.cos}, "math"])


# ---------- inputs computed per file (numpy) ----------


def activations(env: np.ndarray, peak: np.ndarray) -> np.ndarray:
    """EMG envelopes (T, M) -> first-order activation (T, M), causal, normalised by `peak`."""
    u = env / peak
    a = np.zeros_like(u)
    k = (1 / data.RATE) / ACT_TAU
    for i in range(1, len(u)):
        a[i] = a[i - 1] + k * (u[i] - a[i - 1])
    return a


def muscle_index() -> list[int]:
    return [data.EMG.index(mu) for mu in MUSCLES]


def foot_gyro_norm(x: np.ndarray) -> np.ndarray:
    from gaitpinn.gyro import gyro_index

    return np.linalg.norm(x[:, gyro_index("foot")], axis=1)


def swing_mask(x: np.ndarray, threshold: float, min_s: float = 0.1) -> np.ndarray:
    """Swing from IMU only: foot gyro norm above `threshold` for at least `min_s` seconds."""
    on = foot_gyro_norm(x) > threshold
    out = np.zeros_like(on)
    n = int(round(min_s * data.RATE))
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= n:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def swing_threshold(xs: list[np.ndarray], swing_fraction: float = 0.4) -> float:
    """Threshold such that `swing_fraction` of walking training samples are above it."""
    return float(np.quantile(np.concatenate([foot_gyro_norm(x) for x in xs]), 1 - swing_fraction))


# ---------- the loss term ----------


class SwingTerm(nn.Module):
    """Two residuals during swing; parameters w_m >= 0, k >= 0, b >= 0 and q0 are learned."""

    def __init__(self, body_params: dict, scale_dyn: float, scale_tau: float, init: dict):
        super().__init__()
        self.p = body_params
        self.scale_dyn, self.scale_tau = scale_dyn, scale_tau
        sign = [1.0] * len(FLEXORS) + [-1.0] * len(EXTENSORS)
        self.register_buffer("sign", torch.tensor(sign))
        self.w_raw = nn.Parameter(torch.full((len(MUSCLES),), float(init["w"])))
        self.k_raw = nn.Parameter(torch.tensor(float(init["k"])))
        self.b_raw = nn.Parameter(torch.tensor(float(init["b"])))
        self.q0 = nn.Parameter(torch.tensor(float(init["q0"])))
        self.lhs = lhs_function()

    def muscle_torque(self, act: torch.Tensor) -> torch.Tensor:
        """act (B, M, T) -> (B, T) N m, flexion positive."""
        w = nn.functional.softplus(self.w_raw) * self.sign
        return torch.einsum("m,bmt->bt", w, act)

    def residuals(self, angles, tau, phd, act):
        """angles (B,3,T) deg; tau (B,T) N m; phd thigh speed (B,T) rad/s; act (B,M,T).

        Returns the two unit-free residuals, each (B, T-2)."""
        rate = data.RATE
        q = -torch.deg2rad(angles[:, 1])  # Camargo knee angle is negative in flexion
        ph = torch.deg2rad(angles[:, 0])  # thigh angle ~ hip flexion (pelvis tilt neglected)
        qd = (q[:, 2:] - q[:, :-2]) * (rate / 2)
        qdd = (q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]) * rate**2
        phdd = (phd[:, 2:] - phd[:, :-2]) * (rate / 2)
        qc, phc, phdc, tc = q[:, 1:-1], ph[:, 1:-1], phd[:, 1:-1], tau[:, 1:-1]
        p = self.p
        lhs = self.lhs(qc, qd, qdd, phc, phdc, phdd, p["m"], p["d"], p["I_c"], p["l_t"], G)
        k = nn.functional.softplus(self.k_raw)
        b = nn.functional.softplus(self.b_raw)
        r1 = (lhs - (tc - k * (qc - self.q0) - b * qd)) / self.scale_dyn
        r2 = (tc - self.muscle_torque(act)[:, 1:-1]) / self.scale_tau
        return r1, r2

    def forward(self, angles, tau, phd, act, swing) -> torch.Tensor:
        r1, r2 = self.residuals(angles, tau, phd, act)
        m = swing[:, 1:-1].float()
        n = m.sum().clamp(min=1)
        return ((r1**2 + r2**2) * m).sum() / n
