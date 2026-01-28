import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from llm.ops import _scaled_dot_product_attention
from llm.ops import _SimpleMultiHeadAttention


class TestScaleDotAttention(unittest.TestCase):
    def test_my_scale_dot_attention(self):
        torch.manual_seed(42)

        batch_size = 18

        q = torch.randn((batch_size, 16, 16, 8)).cuda()
        k = torch.randn((batch_size, 4, 32, 8)).cuda()
        v = torch.randn((batch_size, 4, 32, 8)).cuda()

        res = _scaled_dot_product_attention(q, k, v, is_causal=False)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        self.assertTrue(torch.allclose(res, ref, atol=1e-5))

        res = _scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        self.assertTrue(torch.allclose(res, ref, atol=1e-5))

        print("_scaled_dot_product_attention run success")


# class TestMultiHeadAttention(unittest.TestCase):
#     def test_my_multi_head_attntion(self):
#         torch.manual_seed(42)

#         batch_size = 128
#         length = 20
#         sub_head_dim = 32
#         num_head = 3

#         mha_torch = nn.MultiheadAttention(
#             sub_head_dim * num_head, num_head, bias=False, dropout=0.0, batch_first=True
#         ).cuda()
#         W = mha_torch.in_proj_weight
#         W_q, W_k, W_v = W.chunk(3, dim=0)
#         W_o = mha_torch.out_proj.weight  # [E, E]

#         x = torch.randn(batch_size, length, sub_head_dim * num_head).cuda()
#         with torch.no_grad():
#             ref, _ = mha_torch(x, x, x)
#         mha_my = _SimpleMultiHeadAttention(
#             sub_head_dim * num_head, num_head, W_q, W_k, W_v, W_o
#         )
#         res = mha_my(x, x, x)

#         self.assertTrue(torch.allclose(res, ref, atol=1e-5))
#         print("_SimpleMultiHeadAttention run success")
