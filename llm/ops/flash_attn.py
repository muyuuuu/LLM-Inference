import torch
import triton
import triton.language as tl
import time


@triton.jit
def online_softmax_kernel(
    src_ptr,
    dst_ptr,
    cols,
    src_stride: tl.int64,
    dst_stride: tl.int64,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    p_src = src_ptr + row_idx * src_stride
    p_dst = dst_ptr + row_idx * dst_stride

    m = -float("inf")
    d = 0.0

    for off in range(0, cols, BLOCK_SIZE):
        col = off + tl.arange(0, BLOCK_SIZE)
        mask = col < cols
        x = tl.load(p_src + col, mask=mask, other=-float("inf"))

        m_new = tl.max(x)
        m_new = tl.maximum(m, m_new)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new

    for off in range(0, cols, BLOCK_SIZE):
        col = off + tl.arange(0, BLOCK_SIZE)
        mask = col < cols
        x = tl.load(p_src + col, mask=mask, other=0.0)
        y = tl.exp(x - m) / d
        tl.store(p_dst + col, y, mask=mask)


# x: torch.Tensor on CUDA, 2D [M, N] and axis == -1
x = torch.randn((256, 10240)).cuda()
y = torch.empty_like(x)

M = x.size(0)
grid = (M,)

online_softmax_kernel[grid](
    x,
    y,
    cols=x.size(1),
    src_stride=x.stride(0),
    dst_stride=y.stride(0),
    BLOCK_SIZE=1024,
    num_warps=16,
)

x = torch.randn((256, 10240)).cuda()
y = torch.empty_like(x)

s = time.time()
online_softmax_kernel[grid](
    x,
    y,
    cols=x.size(1),
    src_stride=x.stride(0),
    dst_stride=y.stride(0),
    BLOCK_SIZE=1024,
    num_warps=16,
)
print(time.time() - s)

standard = torch.nn.functional.softmax(x, dim=1)

s = time.time()
standard = torch.nn.functional.softmax(x, dim=1)
torch.cuda.synchronize()
print(time.time() - s)

print(torch.max(torch.abs(standard - y)))
