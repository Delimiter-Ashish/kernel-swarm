import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=2),
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
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)

    # Early asynchronous load of target index and corresponding logit
    row_start = row_idx * stride_lm
    target_idx = tl.load(target_ptr + row_idx * stride_tm)
    target_logit = tl.load(logits_ptr + row_start + target_idx).to(tl.float32)

    # Decompose row start into 128-byte (64 fp16 elements) aligned base and shift offset
    shift = (row_start & 63).to(tl.int32)
    aligned_start = row_start - shift
    base_ptr = logits_ptr + aligned_start

    # Chunk 0: Head elements, absorbs the misalignment shift
    cols0 = tl.arange(0, BLOCK_V)
    idx0 = cols0 - shift
    mask0 = (idx0 >= 0) & (idx0 < V)
    vals0 = tl.load(base_ptr + cols0, mask=mask0, other=-float("inf"))

    m = tl.max(vals0, axis=0).to(tl.float32)
    safe_vals0 = tl.where(mask0, vals0.to(tl.float32), m)
    exp0 = tl.where(mask0, tl.exp(safe_vals0 - m), 0.0)
    d = tl.sum(exp0, axis=0)

    # Middle chunks: 100% within row bounds, 100% 128-byte aligned, 100% unmasked
    K_UNMASKED: tl.constexpr = (V // BLOCK_V) - 1 if (V >= BLOCK_V) else -1

    if K_UNMASKED >= 0:
        for k in range(1, K_UNMASKED + 1):
            col_k = k * BLOCK_V
            vals = tl.load(base_ptr + col_k + tl.arange(0, BLOCK_V))
            m_chunk = tl.max(vals, axis=0).to(tl.float32)
            m_new = tl.maximum(m, m_chunk)

            scale = tl.exp(m - m_new)
            exp_vals = tl.exp(vals.to(tl.float32) - m_new)
            d = d * scale + tl.sum(exp_vals, axis=0)
            m = m_new

    # Tail chunk(s): process any remaining elements up to V
    tail_start: tl.constexpr = (
        (K_UNMASKED + 1) * BLOCK_V if K_UNMASKED >= 0 else BLOCK_V
    )
    for col_tail in range(tail_start, V + 64, BLOCK_V):
        cols_t = tl.arange(0, BLOCK_V)
        idx_t = col_tail - shift + cols_t
        mask_t = idx_t < V
        vals_t = tl.load(
            base_ptr + col_tail + cols_t, mask=mask_t, other=-float("inf")
        )
        m_chunk = tl.max(vals_t, axis=0).to(tl.float32)
        m_new = tl.maximum(m, m_chunk)

        scale = tl.exp(m - m_new)
        safe_vals_t = tl.where(mask_t, vals_t.to(tl.float32), m_new)
        exp_t = tl.where(mask_t, tl.exp(safe_vals_t - m_new), 0.0)
        d = d * scale + tl.sum(exp_t, axis=0)
        m = m_new

    lse = m + tl.log(d)
    loss = lse - target_logit
    tl.store(loss_ptr + row_idx * stride_om, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    loss = torch.empty((M,), dtype=torch.float32, device=logits.device)

    assert logits.stride(1) == 1, "logits must be contiguous in vocab dimension"
    grid = (M,)
    _cross_entropy_kernel[grid](
        logits,
        target,
        loss,
        logits.stride(0),
        target.stride(0),
        loss.stride(0),
        V=V,
    )
    return loss
