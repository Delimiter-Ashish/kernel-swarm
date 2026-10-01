import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    out_ptr,
    stride_lm,
    stride_t,
    stride_o,
    M,
    V,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_offset = row_idx * stride_lm

    # Early asynchronous target logit load to hide DRAM latency
    target_idx = tl.load(target_ptr + row_idx * stride_t)
    target_logit = tl.load(logits_ptr + row_offset + target_idx).to(tl.float32)

    # Align row start to 16 bytes (8 fp16 elements) to prevent cache-line splitting
    shift = row_offset % 8
    aligned_offset = tl.multiple_of(row_offset - shift, 8)
    aligned_ptr = logits_ptr + aligned_offset

    total_elements = shift + V
    num_full_blocks = total_elements // BLOCK_SIZE

    if num_full_blocks > 0:
        # Peel Block 0 with unmasked 16-byte aligned load; filter out-of-row head in registers
        cols = tl.arange(0, BLOCK_SIZE)
        vals = tl.load(aligned_ptr + cols).to(tl.float32)
        vals = tl.where(cols >= shift, vals, -float("inf"))

        m = tl.max(vals, axis=0)
        d = tl.sum(tl.exp(vals - m), axis=0)

        # Main reduction loop with unmasked, 16-byte aligned vectorized loads
        for i in range(1, num_full_blocks):
            block_cols = i * BLOCK_SIZE + cols
            vals = tl.load(aligned_ptr + block_cols).to(tl.float32)

            chunk_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m, chunk_max)

            d = d * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m), axis=0)
            m = new_m

        # Tail block for remainder elements
        tail_offset = num_full_blocks * BLOCK_SIZE
        if tail_offset < total_elements:
            block_cols = tail_offset + cols
            mask = block_cols < total_elements
            vals = tl.load(aligned_ptr + block_cols, mask=mask, other=-float("inf")).to(tl.float32)

            chunk_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m, chunk_max)

            d = d * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m), axis=0)
            m = new_m
    else:
        # Fallback for vocab sizes smaller than BLOCK_SIZE
        cols = tl.arange(0, BLOCK_SIZE)
        mask = (cols >= shift) & (cols < total_elements)
        vals = tl.load(aligned_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(vals, axis=0)
        d = tl.sum(tl.exp(vals - m), axis=0)

    lse = m + tl.log(d)
    loss = lse - target_logit
    tl.store(out_ptr + row_idx * stride_o, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor, output: torch.Tensor = None) -> torch.Tensor:
    if not logits.is_contiguous():
        logits = logits.contiguous()

    M, V = logits.shape
    if output is None:
        output = torch.empty(M, device=logits.device, dtype=torch.float32)

    grid = (M,)
    _cross_entropy_kernel[grid](
        logits,
        target,
        output,
        logits.stride(0),
        target.stride(0),
        output.stride(0),
        M,
        V,
    )
    return output
