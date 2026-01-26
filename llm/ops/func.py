import torch


def _silu(x):
    return x / (1 + torch.exp(-x))
