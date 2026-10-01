import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_V = 8192
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=5),
        # BLOCK_V = 4096
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_V": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=3),
        # BLOCK_V = 16384
        triton.Config({"BLOCK_V": 16384}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 16384}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 16384}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_V": 16384}, num_warps=16, num_stages=3),
        # Fallback configs
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

    # 1. Issue target index load immediately to overlap DRAM latency with pointer setup
    target_idx = tl.load(target_ptr + row_idx * stride_tm)

    # 2. Align row start to 8-element (16-byte) boundary for vectorized memory transactions
    start_idx = row_idx * stride_lm
    pad_front = start_idx % 8
    aligned_start = tl.multiple_of(start_idx - pad_front, 8)
    row_base = tl.multiple_of(logits_ptr + aligned_start, 8)
    total_elems = V + pad_front

    cols = tl.arange(0, BLOCK_V)
    v_full = (total_elems // BLOCK_V) * BLOCK_V

    if v_full >= BLOCK_V:
        # Issue chunk 0 unmasked load
        vals_0 = tl.load(row_base + cols).to(tl.float32)

        # Prefetch target logit early: target_idx load latency has completed during chunk 0 setup
        target_logit = tl.load(
            logits_ptr + start_idx + target_idx * STRIDE_LV
        ).to(tl.float32)

        # Mask front padding elements in registers
        vals = tl.where(cols >= pad_front, vals_0, -float("inf"))

        m = tl.max(vals, axis=0)
        d = tl.sum(tl.exp(vals - m), axis=0)

        # Full chunks: completely unmasked, 16-byte aligned, branchless online logsumexp
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
        # Fallback for small vocabularies where total_elems < BLOCK_V
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
