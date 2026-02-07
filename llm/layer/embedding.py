import torch


class _TiedEmbedding:
    def __init__(self, vocab_size, dim, weight):
        self._vocab_size = vocab_size
        self._dim = dim
        self._weight = weight

    def __call__(self, input_ids):
        return self._weight[input_ids]

    def as_linear(self, h):
        return torch.matmul(h, self._weight.transpose(0, 1))
