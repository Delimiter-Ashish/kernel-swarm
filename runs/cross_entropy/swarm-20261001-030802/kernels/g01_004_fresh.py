import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=16, num_stages=2),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    output_ptr,
    stride_lm,
    stride_lv,
    stride_t,
    stride_o,
    M,
    V,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Load target and corresponding target logit
    target_val = tl.load(target_ptr + row_idx * stride_t)
    target_logit_ptr = logits_ptr + row_idx * stride_lm + target_val * stride_lv
    target_logit = tl.load(target_logit_ptr).to(tl.float32)

    # Online logsumexp over row chunks
    m_prev = -float("inf")
    s_prev = 0.0

    cols = tl.arange(0, BLOCK_V)
    row_logits_ptr = logits_ptr + row_idx * stride_lm

    for offset in range(0, V, BLOCK_V):
        col_idx = offset + cols
        mask = col_idx < V
        vals = tl.load(row_logits_ptr + col_idx * stride_lv, mask=mask, other=-float("inf")).to(tl.float32)

        m_curr = tl.max(vals, axis=0)
        m_new = tl.maximum(m_prev, m_curr)

        # Handle initial m_prev = -inf cleanly
        alpha = tl.where(m_prev == -float("inf"), 0.0, tl.exp(m_prev - m_new))
        s_prev = s_prev * alpha + tl.sum(tl.exp(vals - m_new), axis=0)
        m_prev = m_new

    lse = m_prev + tl.log(s_prev)
    loss = lse - target_logit

    tl.store(output_ptr + row_idx * stride_o, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)

    grid = (M,)
    _cross_entropy_kernel[grid](
        logits,
        target,
        output,
        logits.stride(0),
        logits.stride(1),
        target.stride(0),
        output.stride(0),
        M=M,
        V=V,
    )
    return output
