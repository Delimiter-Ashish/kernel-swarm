import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048, "num_warps": 8}),
        triton.Config({"BLOCK_SIZE": 4096, "num_warps": 8}),
        triton.Config({"BLOCK_SIZE": 4096, "num_warps": 16}),
        triton.Config({"BLOCK_SIZE": 8192, "num_warps": 16}),
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
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Load target and corresponding logit
    target_idx = tl.load(target_ptr + row_idx * stride_t)
    target_logit = tl.load(logits_ptr + row_idx * stride_lm + target_idx * stride_lv).to(tl.float32)

    row_logits_ptr = logits_ptr + row_idx * stride_lm

    m = -float("inf")
    d = 0.0

    cols = tl.arange(0, BLOCK_SIZE)

    for offset in range(0, V, BLOCK_SIZE):
        col_offsets = offset + cols
        mask = col_offsets < V
        logits = tl.load(
            row_logits_ptr + col_offsets * stride_lv,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        chunk_max = tl.max(logits, axis=0)
        m_new = tl.maximum(m, chunk_max)

        # Scale previous sum and add new exp contributions
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new), axis=0)
        m = m_new

    loss = m + tl.log(d) - target_logit
    tl.store(output_ptr + row_idx * stride_o, loss)


def kernel_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    output = torch.empty((M,), device=logits.device, dtype=torch.float32)

    grid = lambda meta: (M,)
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
