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
    q_seq_len: tl.int64,
    k_seq_len: tl.int64,
    q_total_size: tl.int64,
    k_total_size: tl.int64,
    v_total_size: tl.int64,
    head_num: tl.int64,
    head_dim: tl.constexpr,
    scale: tl.float32,
    is_causal: tl.constexpr,
):
    """
    flash attention forward kernel
    """
    pid_x = tl.program_id(0)

    batch_idx = pid_x // head_num
    head_idx = pid_x % head_num

    for i in range(0, q_seq_len):
        q_off = (
            batch_idx * stride_q_b
            + head_idx * stride_q_h
            + i * stride_q_s
            + tl.arange(0, head_dim)
        )
        q_block = tl.load(
            q_ptr + q_off,
            mask=q_off < q_total_size,
            other=0.0,
        )

        m = -float("inf")
        d = 0.0
        o = tl.zeros((head_dim,), dtype=tl.float32)
        for j in range(0, k_seq_len):
            k_off = (
                batch_idx * stride_k_b
                + head_idx * stride_k_h
                + j * stride_k_s
                + tl.arange(0, head_dim)
            )
            k_block = tl.load(
                k_ptr + k_off,
                mask=k_off < k_total_size,
                other=0.0,
            )
            v_off = (
                batch_idx * stride_v_b
                + head_idx * stride_v_h
                + j * stride_v_s
                + tl.arange(0, head_dim)
            )
            v_block = tl.load(
                v_ptr + v_off,
                mask=v_off < v_total_size,
                other=0.0,
            )

            mask_n = j < k_seq_len
            if is_causal:
                mask_n = mask_n & (j <= i)

            tmp = tl.sum(q_block * k_block) * scale
            tmp = tl.where(mask_n, tmp, -float("inf"))

            m_pre = m
            m = max(m, tmp)
            d_pre = d
            d = d_pre * tl.exp(m_pre - m) + tl.exp(tmp - m)
            o = o * (d_pre * tl.exp(m_pre - m)) / d + v_block * tl.exp(tmp - m) / d

        o_off = (
            batch_idx * stride_o_b
            + head_idx * stride_o_h
            + i * stride_o_s
            + tl.arange(0, head_dim)
        )
        tl.store(
            out_ptr + o_off,
            o,
            mask=o_off < q_total_size,
        )


def flash_attention_forward_triton(q, k, v, is_causal=False):
    assert q.dim() == k.dim() == v.dim(), "bad dim"
    assert k.size() == v.size(), "bad size"

    batch_size = q.size(0)
    q_head_num = q.size(1)
    q_seq_len = q.size(2)
    head_dim = q.size(3)

    k_head_num = k.size(1)
    k_seq_len = k.size(2)

    if q_head_num != k_head_num:
        assert q_head_num % k_head_num == 0, "q_head_num must be divisible by k_head_num"
        k = k.repeat_interleave(q_head_num // k_head_num, dim=1)
        v = v.repeat_interleave(q_head_num // k_head_num, dim=1)

    out = torch.zeros_like(q)

    flash_attention_kernel[(batch_size * q_head_num,)](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        q_seq_len,
        k_seq_len,
        batch_size * q_head_num * q_seq_len * head_dim,
        batch_size * q_head_num * k_seq_len * head_dim,
        batch_size * q_head_num * k_seq_len * head_dim,
        q_head_num,
        head_dim,
        1.0 / head_dim**0.5,
        is_causal=is_causal,
    )
    return out


@triton.jit
def flash_attention_kernel_tile(
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
    q_seq_len: tl.int64,
    k_seq_len: tl.int64,
    q_total_size: tl.int64,
    k_total_size: tl.int64,
    v_total_size: tl.int64,
    q_head_num: tl.int64,
    head_dim: tl.constexpr,
    scale: tl.float32,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    flash attention forward kernel
    """
    pid_x = tl.program_id(0)
    batch_idx = pid_x // q_head_num
    head_idx = pid_x % q_head_num

    pid_y = tl.program_id(1)
    block_start = pid_y * BLOCK_M
    offs_m = block_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    q_off = (
        batch_idx * stride_q_b
        + head_idx * stride_q_h
        + offs_m[:, None] * stride_q_s
        + offs_d[None, :]
    )
    q_block = tl.load(
        q_ptr + q_off,
        mask=offs_m[:, None] < q_seq_len,
        other=0.0,
    )

    m = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    d = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    for j in range(0, k_seq_len, BLOCK_N):
        offs_n = j + tl.arange(0, BLOCK_N)
        k_off = (
            batch_idx * stride_k_b
            + head_idx * stride_k_h
            + offs_n[:, None] * stride_k_s
            + offs_d[None, :]
        )
        k_block = tl.load(
            k_ptr + k_off,
            mask=(offs_n[:, None] < k_seq_len),
            other=0.0,
        )
        v_off = (
            batch_idx * stride_v_b
            + head_idx * stride_v_h
            + offs_n[:, None] * stride_v_s
            + offs_d[None, :]
        )
        v_block = tl.load(
            v_ptr + v_off,
            mask=(offs_n[:, None] < k_seq_len),
            other=0.0,
        )

        mask_n = offs_n < k_seq_len
        if is_causal:
            mask_n = mask_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
        else:
            mask_n = mask_n[None, :]

        s = tl.dot(q_block, tl.trans(k_block)) * scale
        s = tl.where(mask_n, s, -float("inf"))

        m_ij = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_ij[:, None])
        p = tl.where(mask_n, tl.exp(s - m_ij[:, None]), 0.0)

        d_new = d * tl.exp(m - m_ij) + tl.sum(p, axis=1)
        o = (
            o * (d * tl.exp(m - m_ij))[:, None] / d_new[:, None]
            + tl.dot(p, v_block) / d_new[:, None]
        )
        m = m_ij
        d = d_new

    o_off = (
        batch_idx * stride_o_b
        + head_idx * stride_o_h
        + offs_m[:, None] * stride_o_s
        + offs_d[None, :]
    )
    tl.store(
        out_ptr + o_off,
        o,
        mask=(offs_m[:, None] < q_seq_len),
    )


def flash_attention_tile_forward_triton(q, k, v, is_causal=False):
    assert q.dim() == k.dim() == v.dim(), "bad dim"

    batch_size, q_head_num, q_seq_len, head_dim = q.size()
    k_head_num = k.size(1)
    k_seq_len = k.size(2)

    out = torch.zeros_like(q)
    BLOCK_M = 32
    BLOCK_N = 32

    if q_head_num != k_head_num:
        assert q_head_num % k_head_num == 0, "q_head_num must be divisible by k_head_num"
        k = k.repeat_interleave(q_head_num // k_head_num, dim=1)
        v = v.repeat_interleave(q_head_num // k_head_num, dim=1)

    flash_attention_kernel_tile[(batch_size * q_head_num, triton.cdiv(q_seq_len, BLOCK_M))](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        q_seq_len,
        k_seq_len,
        batch_size * q_head_num * q_seq_len * head_dim,
        batch_size * q_head_num * k_seq_len * head_dim,
        batch_size * q_head_num * k_seq_len * head_dim,
        q_head_num,
        head_dim,
        1.0 / head_dim**0.5,
        is_causal,
        BLOCK_M,
        BLOCK_N,
    )

    return out


def flash_attention_forward_cpu(q_ptr, k_ptr, v_ptr, out_ptr, is_causal=False):
    """
    demo for flash attention forward impl cpu version
    not run, very slow, just for demo
    """
    batch_size, q_head_num, q_seq_len, head_dim = q_ptr.shape
    k_head_num = k_ptr.size(1)

    if q_head_num != k_head_num:
        v_head_num = v_ptr.size(1)
        assert q_head_num % k_head_num == 0, "q_head_num must be divisible by k_head_num"
        k_ptr = k_ptr.repeat_interleave(q_head_num // k_head_num, dim=1)
        v_ptr = v_ptr.repeat_interleave(q_head_num // v_head_num, dim=1)

    k_seq_len = k_ptr.size(2)

    scale = head_dim ** (-0.5)

    k_ptr = k_ptr.transpose(2, 3)

    for b in range(batch_size):
        for h in range(q_head_num):
            for i in range(q_seq_len):
                m = float("-inf")
                d = 0
                o = torch.zeros(head_dim, device=q_ptr.device, dtype=q_ptr.dtype)
                for j in range(k_seq_len):
                    if is_causal and j > i:
                        continue

                    tmp = 0
                    for k in range(head_dim):
                        tmp += q_ptr[b, h, i, k] * k_ptr[b, h, k, j]
                    tmp *= scale

                    m_pre = m
                    m = max(m, tmp)
                    d_pre = d
                    d = d_pre * torch.exp(m_pre - m) + torch.exp(tmp - m)
                    o = (
                        o * (d_pre * torch.exp(m_pre - m)) / d
                        + v_ptr[b, h, j, :] * torch.exp(tmp - m) / d
                    )
                out_ptr[b, h, i, :] = o
