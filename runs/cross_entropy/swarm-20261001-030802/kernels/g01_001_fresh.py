import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096}, num_warps=16, num_stages=3),
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
    stride_t,
    stride_loss,
    M,
    V,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Load target for this row and the corresponding target logit
    target_idx = tl.load(target_ptr + row_idx * stride_t)
    target_logit = tl.load(logits_ptr + row_idx * stride_lm + target_idx * stride_lv).to(tl.float32)

    row_logits_ptr = logits_ptr + row_idx * stride_lm

    m = -float("inf")
    d = 0.0

    for start_v in range(0, V, BLOCK_V):
        cols = start_v + tl.arange(0, BLOCK_V)
        mask = cols < V
        logits_chunk = tl.load(row_logits_ptr + cols * stride_lv, mask=mask, other=-float("inf")).to(tl.float32)

        chunk_m = tl.max(logits_chunk, axis=0)
        new_m = tl.maximum(m, chunk_m)

        scale = tl.where(m == -float("inf"), 0.0, tl.exp(m - new_m))
        d = d * scale + tl.sum(tl.exp(logits_chunk - new_m), axis=0)
        m = new_m

    lse = m + tl.log(d)
    loss = lse - target_logit

    tl.store(loss_ptr + row_idx * stride_loss, loss)


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
        M,
        V,
    )
    return output
