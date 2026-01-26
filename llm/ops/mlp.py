import torch
import torch.nn as nn

from .func import _silu


class _MLP:
    def __init__(self, gate, up, down):
        self._gate = gate
        self._up = up
        self._down = down

    def __call__(self, x):
        assert x.dim() == 3, "x size should be [batch_size, seq_len, dim]"

        y = torch.matmul(x, self._gate)
        y = _silu(y)
        y = y * torch.matmul(x, self._up)
        y = torch.matmul(y, self._down)

        return y
