import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=5),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=6),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_V": 2048}, num_warps=4, num_stages=3),
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

    # 1. Asynchronously issue target index load early
    target_idx = tl.load(target_ptr + row_idx * stride_tm)

    # 2. Compute 16-byte aligned base pointer for vectorized 128-bit memory transactions
    start_idx = row_idx * stride_lm
    pad_front = start_idx % 8
    aligned_start = tl.multiple_of(start_idx - pad_front, 8)
    row_base = tl.multiple_of(logits_ptr + aligned_start, 8)
    total_elems = V + pad_front

    cols = tl.arange(0, BLOCK_V)
    v_full = (total_elems // BLOCK_V) * BLOCK_V

    if v_full >= BLOCK_V:
        # Issue chunk 0 load concurrently while target_idx is in-flight
        vals = tl.load(row_base + cols)

        # Issue target logit load; target_idx has arrived, overlapping DRAM latency
        target_logit = tl.load(
            logits_ptr + start_idx + target_idx * STRIDE_LV
        ).to(tl.float32)

        vals = vals.to(tl.float32)
        if pad_front != 0:
            vals = tl.where(cols >= pad_front, vals, -float("inf"))

        m = tl.max(vals, axis=0)
        d = tl.sum(tl.exp(vals - m), axis=0)

        # Main reduction loop: full 16-byte aligned unmasked loads
        for col_offset in range(BLOCK_V, v_full, BLOCK_V):
            col_offset = tl.multiple_of(col_offset, 8)
            vals = tl.load(row_base + col_offset + cols).to(tl.float32)

            m_chunk = tl.max(vals, axis=0)
            m_new = tl.maximum(m, m_chunk)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
            m = m_new

        # Remainder chunk with boundary masking
        if v_full < total_elems:
            v_full_aligned = tl.multiple_of(v_full, 8)
            rem_mask = (v_full + cols) < total_elems
            vals = tl.load(
                row_base + v_full_aligned + cols,
                mask=rem_mask,
                other=-float("inf"),
            ).to(tl.float32)

            m_chunk = tl.max(vals, axis=0)
            m_new = tl.maximum(m, m_chunk)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
            m = m_new
    else:
        target_logit = tl.load(
            logits_ptr + start_idx + target_idx * STRIDE_LV
        ).to(tl.float32)
        mask = (cols >= pad_front) & (cols < total_elems)
        vals = tl.load(row_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
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
