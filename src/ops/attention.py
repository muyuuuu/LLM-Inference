import torch
import math
import torch.nn.functional as F

def __create_upper_inf_mask(batch_size, dim, device):
    mask = torch.triu(torch.ones(dim, dim), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask.to(device)

def _scaled_dot_product_attention(
    query, key, value, mask=None, is_causal=False
):
    assert query.dim() == 3, f"expect query is 3D tensor, but got{query.dim()}"
    assert key.dim() == 3, f"expect query is 3D tensor, but got{query.dim()}"
    assert value.dim() == 3, f"expect query is 3D tensor, but got{query.dim()}"

    assert query.shape == key.shape == value.shape, "QKV's shape is not equal"

    batch_size = query.size(0)
    length = query.size(1)
    hidden_dim = query.size(2)

    assert batch_size > 0, "bad batch size"
    assert length > 0, "bad length"
    assert hidden_dim > 0, "bad hidden_dim"

    device = query.device
    factor = 1 / math.sqrt(hidden_dim)

    attn = torch.matmul(query, torch.transpose(key, 1, 2))
    attn *= factor

    if is_causal:
        assert mask is None
        mask_values = __create_upper_inf_mask(batch_size, hidden_dim, device)
        attn += mask_values
    else:
        if mask is not None:
            if mask.dtype == torch.bool:
                mask_values = torch.where(mask_values, float('-inf'), 0.0)
                attn += mask_values
            else:
                mask_values = mask
                attn += mask_values

    attn = F.softmax(attn, dim=-1)
    attn = torch.matmul(attn, value)

    return attn

if __name__ == "__main__":
    batch_size = 512
    length = 20
    hidden_dim = 30

    q = torch.randn((batch_size, length, hidden_dim)).cuda()
    k = torch.randn((batch_size, length, hidden_dim)).cuda()
    v = torch.randn((batch_size, length, hidden_dim)).cuda()

    res = _scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)

    assert torch.allclose(res, ref, atol=1e-5), "_scaled_dot_product_attention impl wrong"
    print("_scaled_dot_product_attention run success")
