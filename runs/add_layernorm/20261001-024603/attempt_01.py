import torch
import triton
import triton.language as tl


@triton.jit
def _add_layernorm_kernel_single_pass(
    x_ptr,
    r_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_x_row,
    stride_r_row,
    stride_y_row,
    D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    row_x_ptr = x_ptr + row_idx * stride_x_row + cols
    row_r_ptr = r_ptr + row_idx * stride_r_row + cols
    row_y_ptr = y_ptr + row_idx * stride_y_row + cols

    x = tl.load(row_x_ptr, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(row_r_ptr, mask=mask, other=0.0).to(tl.float32)
    h = x + r

    # Compute mean
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / D

    # Compute variance
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = tl.rsqrt(var + eps)

    # Affine transform
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * w + b

    tl.store(row_y_ptr, y.to(tl.float16), mask=mask)


@triton.jit
def _add_layernorm_kernel_multi_pass(
    x_ptr,
    r_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_x_row,
    stride_r_row,
    stride_y_row,
    D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Pass 1: compute mean
    mean_acc = 0.0
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row_idx * stride_r_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean_acc += tl.sum(tl.where(mask, x + r, 0.0), axis=0)
    mean = mean_acc / D

    # Pass 2: compute variance
    var_acc = 0.0
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row_idx * stride_r_row + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, (x + r) - mean, 0.0)
        var_acc += tl.sum(diff * diff, axis=0)
    var = var_acc / D
    rstd = tl.rsqrt(var + eps)

    # Pass 3: normalize and store
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row_idx * stride_r_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = (x + r) - mean
        y = diff * rstd * w + b
        tl.store(y_ptr + row_idx * stride_y_row + cols, y.to(tl.float16), mask=mask)


def kernel_fn(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, D = x.shape
    y = torch.empty_like(x)

    if D <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(D)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4
        _add_layernorm_kernel_single_pass[(M,)](
            x,
            r,
            w,
            b,
            y,
            x.stride(0),
            r.stride(0),
            y.stride(0),
            D,
            1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        BLOCK_SIZE = 2048
        _add_layernorm_kernel_multi_pass[(M,)](
            x,
            r,
            w,
            b,
            y,
            x.stride(0),
            r.stride(0),
            y.stride(0),
            D,
            1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

    return y
