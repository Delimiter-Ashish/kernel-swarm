"""Baseline: fused bias-add + GELU (tanh approximation) in Triton.

Uses the identity 0.5*z*(1 + tanh(a)) == z * sigmoid(2a), with
a = sqrt(2/pi) * (z + 0.044715 z^3), so we only need tl.sigmoid.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_elements, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs % n_cols, mask=mask).to(tl.float32)
    z = x + b
    y = z * tl.sigmoid(1.5957691216057308 * (z + 0.044715 * z * z * z))
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_fn(x, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    _bias_gelu_kernel[(triton.cdiv(n, BLOCK),)](x, b, out, n, x.shape[-1], BLOCK=BLOCK, num_warps=4)
    return out
