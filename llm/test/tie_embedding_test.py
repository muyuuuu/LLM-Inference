import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from copy import deepcopy
from llm.ops import _TiedEmbedding


class TestTieEmbedding(unittest.TestCase):
    def test_tie_embedding(self):
        vocab_size = 1024
        dim = 128

        ref_emb = nn.Embedding(vocab_size, dim).cuda()
        weight = deepcopy(ref_emb.weight)

        my_emb = _TiedEmbedding(vocab_size, dim, weight)

        random_in = torch.randint(0, vocab_size, (dim, 1)).cuda()

        ref = ref_emb(random_in)
        res = my_emb(random_in)

        self.assertTrue(torch.allclose(ref, res)), "_TiedEmbedding impl failed"

        hidden_out = torch.randn((128, dim)).cuda()
        res = my_emb.as_linear(hidden_out)
        ref = F.linear(hidden_out, weight)
        self.assertTrue(torch.allclose(ref, res)), "_TiedEmbedding impl failed"

        print("_TiedEmbedding impl success")
