"""Gyro physics term: joint angle speed = distal segment gyro - proximal segment gyro.

Sensor alignment (PLAN.md): each sensor gets one unit sagittal axis, the first principal axis
of its gyro signal during walking training files. Per joint, the two signs are chosen so that
`s_d * (a_d . g_d) - s_p * (a_p . g_p)` correlates best with d(true angle)/dt. There is no
free scale. Hip has no term (Camargo has a trunk IMU, not a pelvis IMU).
Gyro data are in rad/s, angles in degrees.

The gyro joint speed is smoothed with a zero-phase 6 Hz low-pass, matching the 6 Hz filtering
of the motion-capture markers behind the target angles (without it, heel-strike spikes make
the ankle gyro speed ~1.5x larger than the IK ankle speed). This smoothing is used only inside
the training loss, never as a model input, so the model stays causal.
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from gaitpinn import data

JOINTS = {"knee": (1, "thigh", "shank"), "ankle": (2, "shank", "foot")}
SEGMENTS = ("thigh", "shank", "foot")


def gyro_index(seg: str) -> list[int]:
    """Columns of x holding the three gyro axes of a segment."""
    off = len(data.EMG)
    return [off + data.IMU.index(f"{seg}_Gyro_{a}") for a in "XYZ"]


def principal_axis(g: np.ndarray) -> np.ndarray:
    """Unit vector along which a (N, 3) angular velocity signal varies most."""
    w, v = np.linalg.eigh(g.T @ g)
    a = v[:, np.argmax(w)]
    return a / np.linalg.norm(a)


def angle_speed(y: np.ndarray) -> np.ndarray:
    """d(angle)/dt in deg/s from (T, 3) angles sampled at data.RATE."""
    return np.gradient(y, axis=0) * data.RATE


def segment_speed(x: np.ndarray, seg: str, axis) -> np.ndarray:
    """Sagittal angular speed of one segment (rad/s) from raw x (T, C)."""
    return x[:, gyro_index(seg)].astype(float) @ np.asarray(axis, float)


def joint_speed(x: np.ndarray, align: dict, joint: str) -> np.ndarray:
    """Joint angle speed from the gyros (deg/s), shape (T,)."""
    _, prox, dist = JOINTS[joint]
    s = align["joints"][joint]
    w = s["sign_dist"] * segment_speed(x, dist, align["axes"][dist]) - s[
        "sign_prox"
    ] * segment_speed(x, prox, align["axes"][prox])
    return np.degrees(w)


SMOOTH_HZ = 6.0


def smooth(v: np.ndarray, fc: float = SMOOTH_HZ) -> np.ndarray:
    b, a = butter(2, fc / (data.RATE / 2))
    return filtfilt(b, a, v, axis=0) if len(v) > 15 else v


def reference(x: np.ndarray, align: dict) -> np.ndarray:
    """Smoothed gyro joint speeds (T, 2) in deg/s for knee and ankle: the gyro term's target."""
    return np.column_stack([smooth(joint_speed(x, align, j)) for j in JOINTS])


def fit_alignment(files: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Fit axes and signs on walking training files [(x raw, y deg), ...]."""
    x = np.concatenate([f[0] for f in files])
    axes = {seg: principal_axis(x[:, gyro_index(seg)].astype(float)).tolist() for seg in SEGMENTS}
    align = {"axes": axes, "joints": {}}
    ref = {
        j: np.concatenate([angle_speed(fy)[:, i] for _, fy in files])
        for j, (i, _, _) in JOINTS.items()
    }
    for joint in JOINTS:
        best = None
        for sp in (1, -1):
            for sd in (1, -1):
                align["joints"][joint] = {"sign_prox": sp, "sign_dist": sd}
                w = np.concatenate([joint_speed(fx, align, joint) for fx, _ in files])
                r = np.corrcoef(w, ref[joint])[0, 1]
                if best is None or r > best[0]:
                    best = (r, sp, sd)
        align["joints"][joint] = {"sign_prox": best[1], "sign_dist": best[2]}
    return align


def check(files: list[tuple[np.ndarray, np.ndarray]], align: dict) -> dict:
    """The required check with TRUE angles: correlation and relative residual per joint."""
    w = np.concatenate([reference(fx, align) for fx, _ in files])
    ref = np.concatenate([angle_speed(fy) for _, fy in files])
    out = {}
    for k, (joint, (i, _, _)) in enumerate(JOINTS.items()):
        a, b = w[:, k], ref[:, i]
        out[joint] = {
            "corr": float(np.corrcoef(a, b)[0, 1]),
            "rel_residual": float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2))),
        }
    return out


def passes(result: dict, min_corr: float = 0.85, max_rel: float = 0.7) -> bool:
    return all(r["corr"] >= min_corr and r["rel_residual"] <= max_rel for r in result.values())


class GyroTerm(torch.nn.Module):
    """Loss term: mean over knee and ankle of ((d(pred angle)/dt - gyro speed) / scale)^2."""

    def __init__(self, scale: dict):
        super().__init__()
        self.idx = [JOINTS[j][0] for j in JOINTS]
        self.register_buffer(
            "inv", torch.tensor([1 / scale[j] for j in JOINTS], dtype=torch.float32)
        )

    def residual(self, angles: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """angles (B, 3, T) deg, ref (B, 2, T) deg/s -> unit-free residual (B, 2, T-2)."""
        speed = (angles[:, self.idx, 2:] - angles[:, self.idx, :-2]) * (data.RATE / 2)
        return (speed - ref[..., 1:-1]) * self.inv[None, :, None]

    def forward(self, angles: torch.Tensor, ref: torch.Tensor, mask=None) -> torch.Tensor:
        r = self.residual(angles, ref) ** 2
        if mask is None:
            return r.mean()
        m = mask[:, None, 1:-1].float().expand_as(r)
        return (r * m).sum() / m.sum().clamp(min=1)


def scales(files: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """std of d(true angle)/dt (deg/s) per joint on walking training files."""
    return {
        j: float(np.concatenate([angle_speed(fy)[:, i] for _, fy in files]).std())
        for j, (i, _, _) in JOINTS.items()
    }
