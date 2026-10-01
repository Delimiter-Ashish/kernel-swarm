import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    output_ptr,
    stride_lm,
    stride_ln,
    stride_t,
    stride_o,
    M,
    V,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    logits_row_ptr = logits_ptr + row_idx * stride_lm

    m = -float("inf")
    d = 0.0

    for col_offset in range(0, V, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        logits_chunk = tl.load(
            logits_row_ptr + cols * stride_ln,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        chunk_max = tl.max(logits_chunk, axis=0)
        new_m = tl.maximum(m, chunk_max)

        scale_prev = tl.exp(m - new_m)
        scale_chunk = tl.exp(logits_chunk - new_m)

        # Elements outside mask were -inf, so exp(-inf) = 0
        chunk_sum = tl.sum(scale_chunk, axis=0)
        d = d * scale_prev + chunk_sum
        m = new_m

    lse = m + tl.log(d)

    target_val = tl.load(target_ptr + row_idx * stride_t)
    target_logit = tl.load(logits_row_ptr + target_val * stride_ln).to(tl.float32)

    loss = lse - target_logit
    tl.store(output_ptr + row_idx * stride_o, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor, output: torch.Tensor = None) -> torch.Tensor:
    M, V = logits.shape
    if output is None:
        output = torch.empty(M, device=logits.device, dtype=torch.float32)

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
