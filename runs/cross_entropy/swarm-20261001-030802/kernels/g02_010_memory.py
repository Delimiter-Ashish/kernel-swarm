import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    out_ptr,
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

    row_logits_ptr = logits_ptr + row_idx * stride_lm

    # Asynchronously load target logit
    target_idx = tl.load(target_ptr + row_idx * stride_t)
    target_logit = tl.load(row_logits_ptr + target_idx * stride_lv).to(tl.float32)

    # Online logsumexp accumulators
    m = -float("inf")
    d = 0.0

    for off in range(0, V, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        vals = tl.load(
            row_logits_ptr + cols * stride_lv,
            mask=mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)

        chunk_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m, chunk_max)

        alpha = tl.exp(m - new_m)
        d = d * alpha + tl.sum(tl.exp(vals - new_m), axis=0)
        m = new_m

    lse = m + tl.log(d)
    loss = lse - target_logit
    tl.store(out_ptr + row_idx * stride_o, loss)


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
