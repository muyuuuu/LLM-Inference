import torch


def get_causal_mask(q_length, k_length):
    ones = torch.ones((q_length, k_length), dtype=torch.float32)
    tri = torch.tril(ones, diagonal=k_length - q_length)
    out = torch.where(tri.bool(), torch.zeros_like(tri), torch.full_like(tri, -1e5))
    return out
