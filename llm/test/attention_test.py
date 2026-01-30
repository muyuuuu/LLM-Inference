import torch
import torchtune
import torch.nn as nn
import torch.nn.functional as F
import unittest

from llm.ops import _scaled_dot_product_attention
from llm.ops import _SimpleMultiHeadAttention, _GroupedMultiHeadAttention


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


class TestMultiHeadAttention(unittest.TestCase):
    def test_my_multi_head_attntion(self):
        torch.manual_seed(42)

        batch_size = 128
        length = 20
        sub_head_dim = 32
        num_head = 3

        mha_torch = nn.MultiheadAttention(
            sub_head_dim * num_head, num_head, bias=False, dropout=0.0, batch_first=True
        ).cuda()
        W = mha_torch.in_proj_weight
        W_q, W_k, W_v = W.chunk(3, dim=0)
        W_o = mha_torch.out_proj.weight  # [E, E]

        x = torch.randn(batch_size, length, sub_head_dim * num_head).cuda()
        with torch.no_grad():
            ref, _ = mha_torch(x, x, x)
        mha_my = _SimpleMultiHeadAttention(
            sub_head_dim * num_head, num_head, W_q, W_k, W_v, W_o
        )
        res = mha_my(x, x, x)

        self.assertTrue(torch.allclose(res, ref, atol=1e-5))
        print("_SimpleMultiHeadAttention run success")


class TestGroupedMultiHeadAttention(unittest.TestCase):
    def test_grouped_numti_head_attention(self):
        torch.manual_seed(42)

        batch_size = 1
        q_length = 1
        q_num_head = 1
        kv_num_head = 1
        dim_out = 1
        dim_in = 1
        head_dim = dim_out // q_num_head

        q_proj = nn.Linear(dim_in, q_num_head * head_dim, bias=False)
        k_proj = nn.Linear(dim_in, kv_num_head * head_dim, bias=False)
        v_proj = nn.Linear(dim_in, kv_num_head * head_dim, bias=False)
        o_proj = nn.Linear(q_num_head * head_dim, dim_out, bias=False)

        gmha_torch = torchtune.modules.MultiHeadAttention(
            embed_dim=dim_out,
            num_heads=q_num_head,
            num_kv_heads=kv_num_head,
            head_dim=dim_out // q_num_head,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=o_proj,
            is_causal=False,
        ).cuda()

        x = torch.randn(batch_size, q_length, dim_in).cuda()
        with torch.no_grad():
            ref = gmha_torch(x, y=x)

        gmha_my = _GroupedMultiHeadAttention(
            q_num_head=q_num_head,
            kv_num_head=kv_num_head,
            dim=head_dim,
            q_weight=q_proj.weight,
            k_weight=k_proj.weight,
            v_weight=v_proj.weight,
            o_weight=o_proj.weight,
        )
        res = gmha_my(x, x, x)

        self.assertTrue(torch.allclose(res, ref, atol=1e-5))
        print("_GroupedMultiHeadAttention run success")
