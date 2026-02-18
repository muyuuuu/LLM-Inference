import torch
import torch.nn as nn
import unittest
import random
import time

from llm.ops import (
    flash_attention_forward_cpu,
    flash_attention_tile_forward_triton,
    flash_attention_forward_triton,
)

from llm.model.utils import get_causal_mask


class FlashAttnTestCase(unittest.TestCase):

    def _test_flash_cpu_impl(self):
        for i in range(5):
            batch = 2
            q_seq_len = random.randint(10, 20)
            q_head_num = 2
            k_head_num = 1
            v_head_num = 1
            k_seq_len = random.randint(10, 20)
            head_dim = 16

            q = torch.randn((batch, q_head_num, q_seq_len, head_dim)).cuda()
            k = torch.randn((batch, k_head_num, k_seq_len, head_dim)).cuda()
            v = torch.randn((batch, v_head_num, k_seq_len, head_dim)).cuda()

            is_causal = True if i > 2 else False
            mask = None
            if not is_causal:
                mask = get_causal_mask(q_seq_len, k_seq_len).cuda()
                mask = mask.unsqueeze(0).expand(batch, -1, -1)

            ref = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, enable_gqa=True, attn_mask=mask
            )
            out = torch.zeros_like(ref)
            flash_attention_forward_cpu(q, k, v, out, is_causal=is_causal, mask=mask)
            print(
                f" >>> flash_attention_forward_cpu",
                f"max diff: {(ref - out).abs().max():.4e}",
                sep=", ",
            )

    def _test_flash_triton_impl(self):
        for i in range(6):
            batch = random.randint(10, 18)
            q_seq_len = random.randint(400, 500)
            k_seq_len = random.randint(400, 500)
            q_head_num = 16
            k_head_num = 4
            v_head_num = 4
            head_dim = 64

            q = torch.randn((batch, q_head_num, q_seq_len, head_dim)).cuda()
            k = torch.randn((batch, k_head_num, k_seq_len, head_dim)).cuda()
            v = torch.randn((batch, v_head_num, k_seq_len, head_dim)).cuda()

            is_causal = True if i > 2 else False

            mask = None
            if not is_causal:
                mask = get_causal_mask(q_seq_len, k_seq_len).cuda()
                mask = mask.unsqueeze(0).expand(batch, -1, -1)
                mask = mask.unsqueeze(1)

            bench_time = time.time()
            ref = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, enable_gqa=True, attn_mask=mask
            )
            torch.cuda.synchronize()
            bench_time = time.time() - bench_time

            test_time = time.time()
            out = flash_attention_forward_triton(
                q, k, v, is_causal=is_causal, mask=mask
            )
            torch.cuda.synchronize()
            test_time = time.time() - test_time
            if i >= 1:
                print(
                    f" >>> flash_attention_forward_triton",
                    f"bench_time: {bench_time:.4e}, my_time:{test_time:.4e}",
                    f"max diff: {(ref - out).abs().max():.4e}",
                    sep=", ",
                )

    def _test_flash_triton_tile_impl(self):
        for i in range(6):
            batch = random.randint(10, 18)
            q_seq_len = random.randint(100, 400)
            k_seq_len = random.randint(400, 500)
            q_head_num = 16
            k_head_num = 4
            v_head_num = 4
            head_dim = 64
            head_dim = 128

            q = torch.randn((batch, q_head_num, q_seq_len, head_dim)).cuda()
            k = torch.randn((batch, k_head_num, k_seq_len, head_dim)).cuda()
            v = torch.randn((batch, v_head_num, k_seq_len, head_dim)).cuda()

            is_causal = True if i > 2 else False

            mask = None
            if not is_causal:
                mask = get_causal_mask(q_seq_len, k_seq_len).cuda()
                mask = mask.unsqueeze(0).expand(batch, -1, -1)
                mask = mask.unsqueeze(1)

            bench_time = time.time()
            ref = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, enable_gqa=True, attn_mask=mask
            )
            torch.cuda.synchronize()
            bench_time = time.time() - bench_time

            test_time = time.time()
            out = flash_attention_tile_forward_triton(
                q, k, v, is_causal=is_causal, mask=mask
            )
            torch.cuda.synchronize()
            test_time = time.time() - test_time
            if i >= 1:
                print(
                    f" >>> flash_attention_tile_forward_triton",
                    f"bench_time: {bench_time:.4e}, my_time:{test_time:.4e}",
                    f"max diff: {(ref - out).abs().max():.4e}",
                    sep=", ",
                )

    def test_flash(self):
        self._test_flash_cpu_impl()
        self._test_flash_triton_impl()
        self._test_flash_triton_tile_impl()
