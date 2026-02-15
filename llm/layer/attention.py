import math
import torch
import torch.nn.functional as F


def _scaled_dot_product_attention(query, key, value, mask=None, is_causal=True):
    assert query.size(0) == key.size(0) == value.size(0), "QKV's shape is not equal"

    origin_dtype = query.dtype
    batch_size = query.size(0)

    if query.dim() == 3:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

    assert (
        query.dim() == key.dim() == value.dim() == 4
    ), "_scaled_dot_product_attention only support for 4D tensor"

    assert query.size(3) == key.size(3) == value.size(3), "QKV must has same dimension"


    q_dim = query.size(3)
    q_length = query.size(2)
    k_length = key.size(2)

    device = query.device
    factor = 1 / math.sqrt(q_dim)

    # [..., q_length, kv_length]
    attn = torch.matmul(query, torch.transpose(key, 2, 3))
    attn *= factor

    if is_causal:
        assert mask is None
        i = torch.arange(q_length).view(-1, 1)
        j = torch.arange(k_length).view(1, -1)
        mask = torch.where(
            j > i,
            torch.tensor(float("-inf")),
            torch.tensor(0.0),
        ).to(origin_dtype)
        attn += mask.to(device)
    else:
        if mask is not None:
            if mask.dtype == torch.bool:
                mask_values = torch.where(mask, float("-inf"), 0.0).to(
                    origin_dtype
                )
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
        is_causal=True,
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


class _GroupedMultiHeadAttention:
    def __init__(
        self,
        q_num_head,
        kv_num_head,
        dim,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
    ):

        assert q_num_head % kv_num_head == 0, "q_num_head % kv_num_head != 0"
        assert q_num_head >= kv_num_head, "q_num_head < kv_num_head"

        self._dim = dim
        self._q_num_head = q_num_head
        self._kv_num_head = kv_num_head

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
        is_causal=True,
    ):

        assert (
            query.size() == key.size() == value.size()
        ), "now only support for self-attention"
        batch_size = query.size(0)

        q_length = query.size(1)
        k_length = key.size(1)
        v_length = value.size(1)

        _q = (
            F.linear(query, self.q_weight)
            .view(batch_size, q_length, self._q_num_head, self._dim)
            .transpose(1, 2)
        )
        _k = (
            F.linear(key, self.k_weight)
            .view(batch_size, k_length, self._kv_num_head, self._dim)
            .transpose(1, 2)
        )
        _v = (
            F.linear(value, self.v_weight)
            .view(batch_size, v_length, self._kv_num_head, self._dim)
            .transpose(1, 2)
        )

        out = (
            _scaled_dot_product_attention(_q, _k, _v, mask=mask, is_causal=is_causal)
            .transpose(1, 2)
            .reshape(batch_size, q_length, self._dim * self._q_num_head)
        )

        return F.linear(out, self.o_weight)
