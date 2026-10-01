"""Baseline: hand-written fused row-softmax in Triton (one program per row)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * in_stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_fn(x):
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 8 if BLOCK >= 2048 else 4
    _softmax_kernel[(M,)](out, x, x.stride(0), out.stride(0), N, BLOCK=BLOCK, num_warps=num_warps)
    return out
