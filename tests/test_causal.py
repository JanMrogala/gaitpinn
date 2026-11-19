"""The network must be causal: output at t never depends on inputs after t."""

import torch

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
