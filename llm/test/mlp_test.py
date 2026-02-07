import torch
import unittest
import torch.nn as nn

from copy import deepcopy
from dataclasses import dataclass
from llm.layer import _MLP


@dataclass
class MLPCONF:
    hidden_size = 64
    intermediate_size = 16


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MLPTestCase(unittest.TestCase):
    def test_mlp(self):
        mlp_conf = MLPCONF
        ref_mlp = Qwen2MLP(mlp_conf)

        batch_size = 40
        seq_len = 32
        dim = ref_mlp.hidden_size

        random_in = torch.randn((batch_size, seq_len, dim))

        gate_proj = deepcopy(ref_mlp.gate_proj.weight)
        up_proj = deepcopy(ref_mlp.up_proj.weight)
        down_proj = deepcopy(ref_mlp.down_proj.weight)

        ref = ref_mlp(random_in)

        my_mlp = _MLP(
            gate=gate_proj,
            up=up_proj,
            down=down_proj,
        )
        res = my_mlp(random_in)

        self.assertTrue(torch.allclose(ref, res, atol=1e-5), "_MLP impl failed")
        print("_MLP impl success")
