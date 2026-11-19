"""One small causal 1D CNN, shared by `base` and `pinn`."""

import torch
from torch import nn


class CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class Net(nn.Module):
    """Maps sensor channels (B, C, T) to outputs (B, n_out, T).

    Output at sample t depends only on inputs at samples <= t.
    `pinn` uses n_out = 4 (hip, knee, ankle angles + knee torque); `base` uses 3.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 3,
        width: int = 64,
        levels: int = 6,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.inp = nn.Conv1d(n_in, width, 1)
        self.blocks = nn.ModuleList([CausalConv(width, width, kernel, 2**i) for i in range(levels)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(width, n_out, 1)
        self.receptive_field = 1 + (kernel - 1) * sum(2**i for i in range(levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = h + self.drop(torch.relu(block(h)))
        return self.out(h)
