import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=3),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    output_ptr,
    stride_lm,
    stride_lv,
    stride_t,
    stride_o,
    M,
    V,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Load target logit early to overlap with subsequent memory loads
    target_idx = tl.load(target_ptr + row_idx * stride_t)
    target_logit = tl.load(
        logits_ptr + row_idx * stride_lm + target_idx * stride_lv
    ).to(tl.float32)

    row_logits_ptr = logits_ptr + row_idx * stride_lm
    cols = tl.arange(0, BLOCK_SIZE)

    # Thread-local running max and sum of exponentials (no block reductions in loop)
    m = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    d = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    num_full_blocks = V // BLOCK_SIZE

    # Pipelined unmasked loop for all full blocks
    for i in range(num_full_blocks):
        offset = i * BLOCK_SIZE
        chunk = tl.load(row_logits_ptr + (offset + cols) * stride_lv).to(tl.float32)

        # Single-exp online update identity: exp(-|chunk - m|)
        diff = -tl.abs(chunk - m)
        scale = tl.exp(diff)
        d = tl.where(chunk > m, d * scale + 1.0, d + scale)
        m = tl.maximum(m, chunk)

    # Remainder tail handling with masking
    tail_offset = num_full_blocks * BLOCK_SIZE
    if tail_offset < V:
        col_offsets = tail_offset + cols
        mask = col_offsets < V
        chunk = tl.load(
            row_logits_ptr + col_offsets * stride_lv,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        diff = -tl.abs(chunk - m)
        scale = tl.exp(diff)
        d = tl.where(mask, tl.where(chunk > m, d * scale + 1.0, d + scale), d)
        m = tl.where(mask, tl.maximum(m, chunk), m)

    # Single final block-wide reduction across threads
    global_max = tl.max(m, axis=0)
    d_scaled = tl.where(m == -float("inf"), 0.0, d * tl.exp(m - global_max))
    d_total = tl.sum(d_scaled, axis=0)

    loss = global_max + tl.log(d_total) - target_logit
    tl.store(output_ptr + row_idx * stride_o, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    output = torch.empty((M,), device=logits.device, dtype=torch.float32)

    grid = (M,)
    _cross_entropy_kernel[grid](
        logits,
        target,
        output,
        logits.stride(0),
        logits.stride(1),
        target.stride(0),
        output.stride(0),
        M,
        V,
    )
    return output
