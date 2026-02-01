import torch
import math


class _Rope:
    def __init__(self, dim, max_seq_len, base=10000, traditional=False, device="cpu"):

        assert dim % 2 == 0, "dim % 2 !=0"

        self._dim = dim
        self._max_seq_len = max_seq_len
        self._base = base
        self._device = device

        theta = [1.0 / math.pow(base, i / dim) for i in range(0, dim, 2)]

        self._theta = torch.tensor(theta).reshape(1, -1)
        self._idx = torch.arange(max_seq_len, dtype=self._theta.dtype).reshape(-1, 1)

        # in position m, rotate 2i and 2i + 1
        self._cos_value = torch.cos(torch.matmul(self._idx, self._theta)).to(device)
        self._sin_value = torch.sin(torch.matmul(self._idx, self._theta)).to(device)

        self._traditional = traditional

    def __call__(self, x, offset=None):
        assert (
            x.dim() == 4
        ), "input dimention should be [batch, seq_len, num_head, head_dim]"

        origin_dtype = x.dtype
        seq_len = x.size(1)
        _cos = self._cos_value[:seq_len,]

        _sin = self._sin_value[:seq_len,]
        if offset is not None:
            _cos = self._cos_value[offset]
            _sin = self._sin_value[offset]

        # sin and cos: [1, seq_len, 1, head_dim / 2]
        _cos = _cos.reshape(1, seq_len, 1, -1)
        _sin = _sin.reshape(1, seq_len, 1, -1)

        if self._traditional:
            # x_shaped: [batch, seq_len, num_head, head_dim / 2, 2]
            x_shaped = x.reshape(*x.shape[:-1], -1, 2)

            x1 = x_shaped[..., 0]
            x2 = x_shaped[..., 1]
        else:
            x1 = x[..., 0 : self._dim // 2]
            x2 = x[..., self._dim // 2 :]

        # out_1 and out_2: [B, seq_len, num_head, head_dim / 2]
        out_1 = x1 * _cos - x2 * _sin
        out_2 = x1 * _sin + x2 * _cos

        res = torch.stack([out_1, out_2], dim=-1)
        res = res.reshape(*res.shape[:-2], -1).to(origin_dtype)

        return res


# def apply_rotary_emb(
#     x: torch.Tensor,
#     cos: torch.Tensor,
#     sin: torch.Tensor,
# ) -> torch.Tensor:
#     x1, x2 = torch.chunk(x.float(), 2, dim=-1)
#     y1 = x1 * cos - x2 * sin
#     y2 = x2 * cos + x1 * sin
#     return torch.cat((y1, y2), dim=-1).to(x.dtype)


# class _Rope:
#     def __init__(self, dim, max_seq_len, base=10000, traditional=False, device="cuda"):
#         self.head_size = dim
#         rotary_dim = dim
#         inv_freq = 1.0 / (
#             base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
#         )
#         t = torch.arange(max_seq_len, dtype=torch.float)
#         freqs = torch.einsum("i,j -> ij", t, inv_freq)
#         cos = freqs.cos()
#         sin = freqs.sin()
#         cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
#         self.cos_sin_cache = cache.cuda()

#     def __call__(
#         self,
#         x: torch.Tensor,
#         offset: torch.Tensor,
#     ):
#         cos_sin = self.cos_sin_cache[offset]
#         cos, sin = cos_sin.chunk(2, dim=-1)
#         x = apply_rotary_emb(x, cos, sin)
#         return x
