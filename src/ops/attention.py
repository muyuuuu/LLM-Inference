import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_dot_product_attention(query, key, value, mask=None, is_causal=False):
    assert query.shape == key.shape == value.shape, "QKV's shape is not equal"

    batch_size = query.size(0)
    length = 0
    hidden_dim = 0
    num_heads = 1

    if query.dim() == 3:
        length = query.size(1)
        hidden_dim = query.size(2)

        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

    elif query.dim() == 4:
        num_heads = query.size(1)
        length = query.size(2)
        hidden_dim = query.size(3)
    else:
        raise RuntimeError(f"not support for dim {query.dim()}")

    assert batch_size > 0, "bad batch size"
    assert length > 0, "bad length"
    assert hidden_dim > 0, "bad hidden_dim"
    assert num_heads > 0, "bad num heads"

    device = query.device
    factor = 1 / math.sqrt(hidden_dim)

    attn = torch.matmul(query, torch.transpose(key, 2, 3))
    attn *= factor

    if is_causal:
        assert mask is None
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )
        mask_values = torch.zeros(length, length, device=device).masked_fill(
            causal_mask, float("-inf")
        )
        attn += mask_values
    else:
        if mask is not None:
            if mask.dtype == torch.bool:
                mask_values = torch.where(mask_values, float("-inf"), 0.0)
                attn += mask_values
            else:
                attn += mask

    attn = F.softmax(attn, dim=-1)
    attn = torch.matmul(attn, value)

    return attn


class _SimpleMultiHeadAttention:
    def __init__(self, head_dim, num_heads, q_weight, k_weight, v_weight, o_weight):

        assert (
            o_weight.shape == q_weight.shape == k_weight.shape == v_weight.shape
        ), "QKVO weight shape not equal"

        self._head_dim = head_dim
        self._num_heads = num_heads

        assert head_dim % num_heads == 0, "head_dim % num_heads != 0"
        self._sub_head_dim = head_dim // num_heads

        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight

    def __call__(
        self,
        query,
        key,
        value,
        mask=None,
        is_causal=False,
    ):
        assert query.shape == key.shape == value.shape

        batch = query.size(0)
        length = query.size(1)

        _q = (
            F.linear(query, self.q_weight)
            .view(batch, length, self._num_heads, self._sub_head_dim)
            .transpose(1, 2)
        )
        _k = (
            F.linear(key, self.k_weight)
            .view(batch, length, self._num_heads, self._sub_head_dim)
            .transpose(1, 2)
        )
        _v = (
            F.linear(value, self.v_weight)
            .view(batch, length, self._num_heads, self._sub_head_dim)
            .transpose(1, 2)
        )

        out = (
            _scaled_dot_product_attention(_q, _k, _v, mask=mask, is_causal=is_causal)
            .transpose(1, 2)
            .reshape(batch, length, self._head_dim)
        )

        return F.linear(out, self.o_weight)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 128
    length = 20
    sub_head_dim = 32

    q = torch.randn((batch_size, 1, length, sub_head_dim)).cuda()
    k = torch.randn((batch_size, 1, length, sub_head_dim)).cuda()
    v = torch.randn((batch_size, 1, length, sub_head_dim)).cuda()

    res = _scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(
        res, ref, atol=1e-5
    ), "_scaled_dot_product_attention impl wrong"
    print("_scaled_dot_product_attention run success")

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

    assert torch.allclose(res, ref, atol=1e-5), "_SimpleMultiHeadAttention impl wrong"
    print("_SimpleMultiHeadAttention run success")
