import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_q_b: tl.int64,
    stride_q_s: tl.int64,
    stride_q_h: tl.int64,
    stride_k_b: tl.int64,
    stride_k_s: tl.int64,
    stride_k_h: tl.int64,
    stride_v_b: tl.int64,
    stride_v_s: tl.int64,
    stride_v_h: tl.int64,
    stride_o_b: tl.int64,
    stride_o_s: tl.int64,
    stride_o_h: tl.int64,
    seq_len: tl.int64,
    head_num: tl.int64,
    head_dim: tl.int64,
    scale: tl.float32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Flash Attention kernel. Q, K, V layout: (batch, seq_len, head_num, head_dim).
    Each program handles one (batch, head) and one block of Q rows (BLOCK_M).
    """
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(seq_len, BLOCK_M)
    batch_head_idx = pid // num_blocks_m
    block_m_idx = pid % num_blocks_m
    batch_idx = batch_head_idx // head_num
    head_idx = batch_head_idx % head_num

    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        q_ptr
        + batch_idx * stride_q_b
        + offs_m[:, None] * stride_q_s
        + head_idx * stride_q_h
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    block_n_end = (
        seq_len if not IS_CAUSAL else tl.minimum((block_m_idx + 1) * BLOCK_M, seq_len)
    )
    for start_n in range(0, block_n_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (
            k_ptr
            + batch_idx * stride_k_b
            + offs_n[:, None] * stride_k_s
            + head_idx * stride_k_h
            + offs_d[None, :]
        )
        v_ptrs = (
            v_ptr
            + batch_idx * stride_v_b
            + offs_n[:, None] * stride_v_s
            + head_idx * stride_v_h
            + offs_d[None, :]
        )
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        qk = tl.where(
            (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(
            (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
            p,
            0.0,
        )

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]

    out_ptrs = (
        out_ptr
        + batch_idx * stride_o_b
        + offs_m[:, None] * stride_o_s
        + head_idx * stride_o_h
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)


def flash_attention_forward(q, k, v, causal=False, BLOCK_M=64, BLOCK_N=64):
    """
    Flash Attention forward. Q, K, V: (batch, seq_len, head_num, head_dim).
    """
    batch_size, seq_len, head_num, head_dim = q.shape
    assert head_dim <= 128 and head_dim % 8 == 0
    scale = head_dim ** (-0.5)

    out = torch.zeros_like(q)

    BLOCK_D = head_dim
    grid = (batch_size * head_num * triton.cdiv(seq_len, BLOCK_M),)

    flash_attention_kernel[grid](
        q,
        k,
        v,
        out,
        stride_q_b=seq_len * head_num * head_dim,
        stride_q_s=head_num * head_dim,
        stride_q_h=head_dim,
        stride_k_b=seq_len * head_num * head_dim,
        stride_k_s=head_num * head_dim,
        stride_k_h=head_dim,
        stride_v_b=seq_len * head_num * head_dim,
        stride_v_s=head_num * head_dim,
        stride_v_h=head_dim,
        stride_o_b=seq_len * head_num * head_dim,
        stride_o_s=head_num * head_dim,
        stride_o_h=head_dim,
        seq_len=seq_len,
        head_num=head_num,
        head_dim=head_dim,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
    )
    return out


def flash_attention():
    batch_size = 1
    seq_len = 128
    head_num = 1
    head_dim = 64
    q = torch.randn(
        batch_size, seq_len, head_num, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, head_num, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, head_num, head_dim, device="cuda", dtype=torch.float16
    )

    out = flash_attention_forward(q, k, v, causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    print("max diff:", (out - ref).abs().max().item())
    print("flash attention test done.")


flash_attention()
