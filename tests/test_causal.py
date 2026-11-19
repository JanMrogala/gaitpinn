"""The network must be causal: output at t never depends on inputs after t."""

import numpy as np
import torch

from gaitpinn import data
from gaitpinn.model import Net


def test_future_input_does_not_change_past_output():
    torch.manual_seed(0)
    net = Net(n_in=5, n_out=3, width=16, levels=4).eval()
    x = torch.randn(2, 5, 200)
    y = net(x)
    x2 = x.clone()
    x2[..., 120:] = torch.randn(2, 5, 80) * 10
    y2 = net(x2)
    assert torch.allclose(y[..., :120], y2[..., :120])
    assert not torch.allclose(y[..., 120:], y2[..., 120:])


def test_receptive_field():
    net = Net(n_in=1, n_out=1, width=4, levels=6, kernel=3)
    assert net.receptive_field == 127


def test_preprocessing_is_causal():
    rng = np.random.default_rng(0)
    emg = rng.normal(size=(5000, 3))
    t_src = np.arange(5000) / 1000
    t_dst = np.arange(0, 4.99, 0.01)
    a = data.sample_at(t_src, data.emg_envelope(emg, 1000), t_dst)
    emg2 = emg.copy()
    emg2[3000:] = 50  # change the future after t = 3.0 s
    b = data.sample_at(t_src, data.emg_envelope(emg2, 1000), t_dst)
    past = t_dst < 3.0 - 1e-9
    np.testing.assert_array_equal(a[past], b[past])
    assert not np.allclose(a[~past], b[~past])


def test_sample_at_never_looks_ahead():
    t_src = np.array([0.0, 0.004, 0.01, 0.016])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    out = data.sample_at(t_src, x, np.array([0.0, 0.009, 0.01, 0.015]))
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0, 2.0])
