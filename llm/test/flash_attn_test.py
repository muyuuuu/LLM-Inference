import torch
import torch.nn as nn
import unittest
import random
import time
import math

from llm.ops import (
    flash_attention_forward_cpu,
    flash_attention_tile_forward_triton,
    flash_attention_forward_triton,
)


class FlashAttnTestCase(unittest.TestCase):

    def _test_flash_cpu_impl(self):
        for i in range(3):
            batch = 2
            seq_len = random.randint(10, 20)
            head_num = 1
            head_dim = 16

            q = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            k = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            v = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            is_causal = True if random.randint(0, 2) >= 1 else False
            ref = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )
            out = torch.zeros_like(ref)
            flash_attention_forward_cpu(q, k, v, out, is_causal=is_causal)
            print(
                f" >>> {[batch, head_num, seq_len, head_dim]}, flash_attention_forward_cpu",
                f"max diff: {(ref - out).abs().max():.4e}",
                sep=", ",
            )

    def _test_flash_triton_impl(self):
        for i in range(6):
            batch = random.randint(10, 18)
            seq_len = random.randint(400, 500)
            head_num = 16
            head_dim = 64

            q = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            k = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            v = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            # is_causal = True if random.randint(0, 2) >= 1 else False
            is_causal = False

            bench_time = time.time()
            ref = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )
            torch.cuda.synchronize()
            bench_time = time.time() - bench_time

            test_time = time.time()
            out = flash_attention_forward_triton(q, k, v)
            torch.cuda.synchronize()
            test_time = time.time() - test_time
            if i >= 1:
                print(
                    f" >>> {[batch, head_num, seq_len, head_dim]}",
                    f"flash_attention_forward_triton, bench_time: {bench_time:.4e}, my_time:{test_time:.4e}",
                    f"max diff: {(ref - out).abs().max():.4e}",
                    sep=", ",
                )

    def _test_flash_triton_tile_impl(self):
        for i in range(6):
            batch = random.randint(10, 18)
            seq_len = random.randint(400, 500)
            head_num = 16
            head_dim = 128

            q = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            k = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            v = torch.randn((batch, head_num, seq_len, head_dim)).cuda()
            # is_causal = True if random.randint(0, 2) >= 1 else False
            is_causal = False

            bench_time = time.time()
            ref = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )
            torch.cuda.synchronize()
            bench_time = time.time() - bench_time

            test_time = time.time()
            out = flash_attention_tile_forward_triton(q, k, v)
            torch.cuda.synchronize()
            test_time = time.time() - test_time
            if i >= 1:
                print(
                    f" >>> {[batch, head_num, seq_len, head_dim]}",
                    f"flash_attention_tile_forward_triton, bench_time: {bench_time:.4e}, my_time:{test_time:.4e}",
                    f"max diff: {(ref - out).abs().max():.4e}",
                    sep=", ",
                )

    def test_flash(self):
        self._test_flash_cpu_impl()
        self._test_flash_triton_impl()
        self._test_flash_triton_tile_impl()
