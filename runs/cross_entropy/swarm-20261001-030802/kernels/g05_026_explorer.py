import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_V": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=3),
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
    start_idx = row_idx * stride_lm

    # Issue target index load early to initiate memory controller transaction
    target_idx = tl.load(target_ptr + row_idx * stride_tm)

    # Align starting element to a 64-element (128-byte) cache-line boundary
    pad_front = start_idx % 64
    aligned_start = tl.multiple_of(start_idx - pad_front, 64)
    total_elems = V + pad_front

    cols = tl.arange(0, BLOCK_V)
    v_full = (total_elems // BLOCK_V) * BLOCK_V
    base_ptr = logits_ptr + aligned_start

    if v_full >= BLOCK_V:
        # Issue chunk 0 load before dependent target_logit to keep memory pipeline full
        vals = tl.load(base_ptr + cols).to(tl.float32)

        # Issue target logit load now; target_idx is already in flight / ready
        target_logit = tl.load(
            logits_ptr + start_idx + target_idx * STRIDE_LV
        ).to(tl.float32)

        # Neutralize prepended padding elements from the previous row
        vals = tl.where(cols >= pad_front, vals, -float("inf"))
        m = tl.max(vals, axis=0)
        d = tl.sum(tl.exp(vals - m), axis=0)

        # Full chunks: 128-byte aligned, unmasked, with decoupled chunk reduction
        for col_offset in range(BLOCK_V, v_full, BLOCK_V):
            col_offset = tl.multiple_of(col_offset, 64)
            vals = tl.load(base_ptr + col_offset + cols).to(tl.float32)

            # Decoupled reduction: local max and sum are independent of running state
            m_chunk = tl.max(vals, axis=0)
            d_chunk = tl.sum(tl.exp(vals - m_chunk), axis=0)

            # Associative combination step (scalar only)
            m_new = tl.maximum(m, m_chunk)
            d = d * tl.exp(m - m_new) + d_chunk * tl.exp(m_chunk - m_new)
            m = m_new

        # Remainder chunk with boundary masking
        if v_full < total_elems:
            rem_cols = v_full + cols
            rem_mask = rem_cols < total_elems
            vals = tl.load(
                base_ptr + v_full + cols,
                mask=rem_mask,
                other=-float("inf"),
            ).to(tl.float32)

            m_chunk = tl.max(vals, axis=0)
            d_chunk = tl.sum(tl.exp(vals - m_chunk), axis=0)

            m_new = tl.maximum(m, m_chunk)
            d = d * tl.exp(m - m_new) + d_chunk * tl.exp(m_chunk - m_new)
            m = m_new
    else:
        # Fallback for small vocabularies where total_elems < BLOCK_V
        target_logit = tl.load(
            logits_ptr + start_idx + target_idx * STRIDE_LV
        ).to(tl.float32)
        mask = (cols >= pad_front) & (cols < total_elems)
        vals = tl.load(base_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
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
