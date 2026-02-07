import torch.nn as nn

from .func import _silu


class _MLP:
    def __init__(self, gate, up, down):
        self._gate = gate
        self._up = up
        self._down = down

    def __call__(self, x):
        assert x.dim() == 3, "x size should be [batch_size, seq_len, dim]"

        y = nn.functional.linear(x, self._gate)
        y = _silu(y)
        y = y * nn.functional.linear(x, self._up)
        y = nn.functional.linear(y, self._down)

        return y
