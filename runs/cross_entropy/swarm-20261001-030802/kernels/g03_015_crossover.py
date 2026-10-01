import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
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
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    logits_row_ptr = logits_ptr + row_idx * stride_lm

    # Issue target load early to overlap DRAM latency across the streaming loop
    target_idx = tl.load(target_ptr + row_idx * stride_tm)
    target_logit = tl.load(logits_row_ptr + target_idx * stride_lv).to(tl.float32)

    cols = tl.arange(0, BLOCK_SIZE)
    m = -float("inf")
    d = 0.0

    # Separate unmasked full blocks from remainder to enable unconditional vector loads
    num_full = V // BLOCK_SIZE
    full_end = num_full * BLOCK_SIZE

    for offset in range(0, full_end, BLOCK_SIZE):
        vals = tl.load(logits_row_ptr + (offset + cols) * stride_lv).to(tl.float32)
        chunk_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m, chunk_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

    # Process remaining elements with boundary mask
    if full_end < V:
        rem_cols = full_end + cols
        mask = rem_cols < V
        vals = tl.load(
            logits_row_ptr + rem_cols * stride_lv, mask=mask, other=-float("inf")
        ).to(tl.float32)
        chunk_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m, chunk_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

    loss = m + tl.log(d) - target_logit
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
        logits.stride(1),
        target.stride(0),
        loss.stride(0),
        V,
    )
    return loss
