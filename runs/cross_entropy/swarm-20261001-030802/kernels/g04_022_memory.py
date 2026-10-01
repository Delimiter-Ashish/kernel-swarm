import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_V": 16384}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 16384}, num_warps=16, num_stages=3),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    stride_lm,
    stride_tm,
    stride_om,
    V,
    STRIDE_LV: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Prefetch target index and target logit early to hide memory latency
    target_idx = tl.load(target_ptr + row_idx * stride_tm)
    target_logit = tl.load(
        logits_ptr + row_idx * stride_lm + target_idx * STRIDE_LV
    ).to(tl.float32)

    # Align base pointer to 16 bytes (8 fp16 elements) for smem/vectorized loads
    if STRIDE_LV == 1:
        shift = (row_idx * stride_lm) % 8
        aligned_base = tl.multiple_of(row_idx * stride_lm - shift, 8)
        aligned_ptr = logits_ptr + aligned_base
    else:
        shift = 0
        aligned_ptr = logits_ptr + row_idx * stride_lm

    total_len = shift + V
    v_full = (total_len // BLOCK_V) * BLOCK_V

    if v_full >= BLOCK_V:
        # Chunk 0: peel leading shift elements and initialize accumulators directly
        cols0 = tl.arange(0, BLOCK_V)
        vals0 = tl.load(
            aligned_ptr + cols0 * STRIDE_LV,
            mask=cols0 >= shift,
            other=-float("inf"),
        ).to(tl.float32)
        m = tl.max(vals0, axis=0)
        d = tl.sum(tl.exp(vals0 - m), axis=0)

        # Main reduction loop over 100% unmasked, 16-byte aligned chunks
        for col_offset in range(BLOCK_V, v_full, BLOCK_V):
            cols = col_offset + tl.arange(0, BLOCK_V)
            vals = tl.load(aligned_ptr + cols * STRIDE_LV).to(tl.float32)

            m_chunk = tl.max(vals, axis=0)
            m_new = tl.maximum(m, m_chunk)
            scale = tl.exp(m - m_new)
            d = d * scale + tl.sum(tl.exp(vals - m_new), axis=0)
            m = m_new

        # Remainder chunk with tail masking
        if v_full < total_len:
            cols_rem = v_full + tl.arange(0, BLOCK_V)
            vals_rem = tl.load(
                aligned_ptr + cols_rem * STRIDE_LV,
                mask=cols_rem < total_len,
                other=-float("inf"),
            ).to(tl.float32)

            m_chunk = tl.max(vals_rem, axis=0)
            m_new = tl.maximum(m, m_chunk)
            scale = tl.exp(m - m_new)
            d = d * scale + tl.sum(tl.exp(vals_rem - m_new), axis=0)
            m = m_new
    else:
        # Small vocabulary fallback: entire row fits into a single chunk
        cols = tl.arange(0, BLOCK_V)
        mask = (cols >= shift) & (cols < total_len)
        vals = tl.load(
            aligned_ptr + cols * STRIDE_LV, mask=mask, other=-float("inf")
        ).to(tl.float32)
        m = tl.max(vals, axis=0)
        d = tl.sum(tl.exp(vals - m), axis=0)

    lse = m + tl.log(d)
    loss = lse - target_logit
    tl.store(loss_ptr + row_idx * stride_om, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    loss = torch.empty((M,), dtype=torch.float32, device=logits.device)

    grid = (M,)
    _cross_entropy_kernel[grid](
        logits,
        target,
        loss,
        logits.stride(0),
        target.stride(0),
        loss.stride(0),
        V,
        STRIDE_LV=logits.stride(1),
    )
    return loss
