import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_V": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
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
    logits_row_ptr = logits_ptr + row_idx * stride_lm

    # Online logsumexp accumulators in fp32
    m = -float("inf")
    d = 0.0

    # Unmasked loop over full vocabulary chunks
    v_full = (V // BLOCK_V) * BLOCK_V
    for col_offset in range(0, v_full, BLOCK_V):
        cols = col_offset + tl.arange(0, BLOCK_V)
        # Load in native fp16 to conserve registers and enable 128-bit memory transactions
        vals = tl.load(logits_row_ptr + cols * STRIDE_LV)

        m_chunk = tl.max(vals, axis=0).to(tl.float32)
        m_new = tl.maximum(m, m_chunk)
        scale = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_new))
        d = d * scale + tl.sum(tl.exp(vals.to(tl.float32) - m_new), axis=0)
        m = m_new

    # Remainder chunk with boundary masking
    if v_full < V:
        cols = v_full + tl.arange(0, BLOCK_V)
        mask = cols < V
        vals = tl.load(
            logits_row_ptr + cols * STRIDE_LV, mask=mask, other=-float("inf")
        )

        m_chunk = tl.max(vals, axis=0).to(tl.float32)
        m_new = tl.maximum(m, m_chunk)
        scale = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_new))
        d = d * scale + tl.sum(tl.exp(vals.to(tl.float32) - m_new), axis=0)
        m = m_new

    lse = m + tl.log(d)

    # Load target logit (hits L1/L2 cache since row was just streamed)
    target_idx = tl.load(target_ptr + row_idx * stride_tm)
    target_logit = tl.load(logits_row_ptr + target_idx * STRIDE_LV).to(tl.float32)
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
