import torch
import torch.nn as nn
import unittest

from copy import deepcopy
from llm.ops import _RMSNorm


class TestRMSNorm(unittest.TestCase):
    def test_rmsnorm(self):

        eps = 1e-5
        batch_size = 1
        seq_len = 2
        num_head = 1
        sub_head_dim = 8

        random_in = torch.randn(
            (batch_size, seq_len, num_head, sub_head_dim), device="cuda"
        )

        ref_rmsnorm = nn.RMSNorm([num_head, sub_head_dim], eps=eps).cuda()
        weight = deepcopy(ref_rmsnorm.weight)
        ref = ref_rmsnorm(random_in)

        my_norm = _RMSNorm(weight=weight, eps=eps, device="cuda")
        res = my_norm(random_in)

        self.assertTrue(torch.allclose(ref, res, atol=1e-5))
        print("_RMSNorm impl success")
