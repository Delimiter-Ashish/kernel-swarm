import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=3),
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
):
    row_idx = tl.program_id(0)

    logits_row_ptr = logits_ptr + row_idx * stride_lm

    # Load target index and corresponding logit
    target_idx = tl.load(target_ptr + row_idx * stride_tm)
    target_logit = tl.load(logits_row_ptr + target_idx * stride_lv).to(tl.float32)

    # First chunk: initialize online logsumexp accumulators without branching
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V
    vals = tl.load(
        logits_row_ptr + cols * stride_lv, mask=mask, other=-float("inf")
    ).to(tl.float32)

    m = tl.max(vals, axis=0)
    d = tl.sum(tl.exp(vals - m), axis=0)

    # Remaining chunks
    for col_offset in range(BLOCK_V, V, BLOCK_V):
        cols = col_offset + tl.arange(0, BLOCK_V)
        mask = cols < V
        vals = tl.load(
            logits_row_ptr + cols * stride_lv, mask=mask, other=-float("inf")
        ).to(tl.float32)

        m_chunk = tl.max(vals, axis=0)
        m_new = tl.maximum(m, m_chunk)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

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
        logits.stride(1),
        target.stride(0),
        loss.stride(0),
        V,
    )
    return loss
