# import torch
# import triton
# import triton.language as tl


# @triton.jit
# def add_kernel(
#     src1_pt1,
#     src2_pt1,
#     dst_ptr,
#     size,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     block_start = pid * BLOCK_SIZE
#     offset = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offset < size
#     x = tl.load(src1_pt1 + offset, mask=mask)
#     y = tl.load(src2_pt1 + offset, mask=mask)
#     out = x + y
#     tl.store(dst_ptr + offset, out, mask=mask)


# def add(x: torch.Tensor, y: torch.Tensor):
#     output = torch.empty_like(x)
#     assert x.is_cuda and y.is_cuda and output.is_cuda
#     n_elements = output.numel()
#     grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

#     add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
#     return output


# torch.manual_seed(0)
# size = 98432
# x = torch.rand(size, device="cuda")
# y = torch.rand(size, device="cuda")
# output_torch = x + y
# output_triton = add(x, y)

# print(torch.max(torch.abs(output_torch - output_triton)))

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    input_row_stride: tl.int64,
    output_row_stride: tl.int64,
    n_cols,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator * 1.0

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        y,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


torch.manual_seed(0)
x = torch.randn(1823, 7810, device="cuda")
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
