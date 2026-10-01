import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=8),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=16),
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=8),
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=16),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=8),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=16),
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_warps=8),
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_warps=16),
    ],
    key=["D"],
)
@triton.jit
def _add_layernorm_fast_kernel(
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
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    inv_D = 1.0 / D

    w = tl.load(w_ptr + cols).to(tl.float32)
    b = tl.load(b_ptr + cols).to(tl.float32)

    base_row = pid * ROWS_PER_PROGRAM

    if ROWS_PER_PROGRAM == 1:
        row_x = x_ptr + base_row * stride_x_row + cols
        row_r = r_ptr + base_row * stride_r_row + cols
        row_y = y_ptr + base_row * stride_y_row + cols

        x = tl.load(row_x).to(tl.float32)
        r = tl.load(row_r).to(tl.float32)
        h = x + r
        mean = tl.sum(h, axis=0) * inv_D
        diff = h - mean
        var = tl.sum(diff * diff, axis=0) * inv_D
        rstd = tl.rsqrt(var + eps)
        scale = rstd * w
        y = diff * scale + b
        tl.store(row_y, y.to(tl.float16))
    else:
        for step in range(ROWS_PER_PROGRAM // 2):
            row0 = base_row + step * 2
            row1 = row0 + 1

            r0_x = x_ptr + row0 * stride_x_row + cols
            r0_r = r_ptr + row0 * stride_r_row + cols
            r0_y = y_ptr + row0 * stride_y_row + cols

            r1_x = x_ptr + row1 * stride_x_row + cols
            r1_r = r_ptr + row1 * stride_r_row + cols
            r1_y = y_ptr + row1 * stride_y_row + cols

            # Concurrently issue loads for two consecutive rows
            x0 = tl.load(r0_x).to(tl.float32)
            r0 = tl.load(r0_r).to(tl.float32)
            x1 = tl.load(r1_x).to(tl.float32)
            r1 = tl.load(r1_r).to(tl.float32)

            # Process row 0
            h0 = x0 + r0
            mean0 = tl.sum(h0, axis=0) * inv_D
            diff0 = h0 - mean0
            var0 = tl.sum(diff0 * diff0, axis=0) * inv_D
            rstd0 = tl.rsqrt(var0 + eps)
            scale0 = rstd0 * w
            y0 = diff0 * scale0 + b
            tl.store(r0_y, y0.to(tl.float16))

            # Process row 1
            h1 = x1 + r1
            mean1 = tl.sum(h1, axis=0) * inv_D
            diff1 = h1 - mean1
            var1 = tl.sum(diff1 * diff1, axis=0) * inv_D
            rstd1 = tl.rsqrt(var1 + eps)
            scale1 = rstd1 * w
            y1 = diff1 * scale1 + b
            tl.store(r1_y, y1.to(tl.float16))


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=4),
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=8),
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=8),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _add_layernorm_general_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_x_row,
    stride_r_row,
    stride_y_row,
    M,
    D,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    if HAS_MASK:
        mask = cols < D
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        w = tl.load(w_ptr + cols).to(tl.float32)
        b = tl.load(b_ptr + cols).to(tl.float32)

    inv_D = 1.0 / D

    for i in range(ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + i
        if row < M:
            row_x = x_ptr + row * stride_x_row + cols
            row_r = r_ptr + row * stride_r_row + cols
            row_y = y_ptr + row * stride_y_row + cols

            if HAS_MASK:
                mask = cols < D
                x = tl.load(row_x, mask=mask, other=0.0).to(tl.float32)
                r = tl.load(row_r, mask=mask, other=0.0).to(tl.float32)
                h = x + r
                mean = tl.sum(tl.where(mask, h, 0.0), axis=0) * inv_D
                diff = tl.where(mask, h - mean, 0.0)
                var = tl.sum(diff * diff, axis=0) * inv_D
                rstd = tl.rsqrt(var + eps)
                scale = rstd * w
                y = diff * scale + b
                tl.store(row_y, y.to(tl.float16), mask=mask)
            else:
                x = tl.load(row_x).to(tl.float32)
                r = tl.load(row_r).to(tl.float32)
                h = x + r
                mean = tl.sum(h, axis=0) * inv_D
                diff = h - mean
                var = tl.sum(diff * diff, axis=0) * inv_D
                rstd = tl.rsqrt(var + eps)
                scale = rstd * w
                y = diff * scale + b
                tl.store(row_y, y.to(tl.float16))


@triton.jit
def _add_layernorm_multi_pass_kernel(
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

    mean_acc = 0.0
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row_idx * stride_r_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean_acc += tl.sum(tl.where(mask, x + r, 0.0), axis=0)
    mean = mean_acc / D

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

    if D <= 8192:
        BLOCK_SIZE = triton.next_power_of_2(D)
        if BLOCK_SIZE == D and (M % 8 == 0):
            grid = lambda META: (M // META["ROWS_PER_PROGRAM"],)
            _add_layernorm_fast_kernel[grid](
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
            )
        else:
            HAS_MASK = BLOCK_SIZE != D
            grid = lambda META: (triton.cdiv(M, META["ROWS_PER_PROGRAM"]),)
            _add_layernorm_general_kernel[grid](
                x,
                r,
                w,
                b,
                y,
                x.stride(0),
                r.stride(0),
                y.stride(0),
                M,
                D,
                1e-5,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_MASK=HAS_MASK,
            )
    else:
        BLOCK_SIZE = 2048
        _add_layernorm_multi_pass_kernel[(M,)](
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
