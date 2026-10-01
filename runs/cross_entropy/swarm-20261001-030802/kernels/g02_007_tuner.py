import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=4),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    stride_lm,
    stride_lv,
    stride_tm,
    stride_om,
    V,
    BLOCK_V: tl.constexpr,
    HAS_UNIT_STRIDE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Pointer to current row in logits
    logits_row_ptr = logits_ptr + row_idx * stride_lm

    # Load target index and corresponding logit
    target_idx = tl.load(target_ptr + row_idx * stride_tm)
    if HAS_UNIT_STRIDE:
        target_logit = tl.load(logits_row_ptr + target_idx).to(tl.float32)
    else:
        target_logit = tl.load(logits_row_ptr + target_idx * stride_lv).to(tl.float32)

    # Online logsumexp accumulators
    m = -float("inf")
    d = 0.0

    for col_offset in range(0, V, BLOCK_V):
        cols = col_offset + tl.arange(0, BLOCK_V)
        mask = cols < V
        if HAS_UNIT_STRIDE:
            vals = tl.load(logits_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        else:
            vals = tl.load(
                logits_row_ptr + cols * stride_lv, mask=mask, other=-float("inf")
            ).to(tl.float32)

        m_chunk = tl.max(vals, axis=0)
        m_new = tl.maximum(m, m_chunk)

        # Scale previous denominator; handle initial m = -inf cleanly
        scale = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_new))
        d = d * scale + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

    lse = m + tl.log(d)
    loss = lse - target_logit

    tl.store(loss_ptr + row_idx * stride_om, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    loss = torch.empty((M,), dtype=torch.float32, device=logits.device)
    has_unit_stride = logits.stride(1) == 1

    grid = (M,)
    _cross_entropy_kernel[grid](
        logits,
        target,
        loss,
        logits.stride(0),
        logits.stride(1),
        target.stride(0),
        loss.stride(0),
        V,
        HAS_UNIT_STRIDE=has_unit_stride,
    )
    return loss
