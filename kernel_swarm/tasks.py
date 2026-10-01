"""Task definitions.

A task is a PyTorch reference op plus an input generator. Every candidate
kernel must define `kernel_fn(*inputs)` that returns the same result as
`reference(*inputs)`.

We target *fused* ops on purpose: PyTorch eager runs them as several
separate GPU launches, so a single fused Triton kernel can really win.
"""
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class Task:
    name: str
    description: str
    make_inputs: Callable[..., tuple]
    reference: Callable[..., torch.Tensor]
    atol: float = 1e-2
    rtol: float = 1e-2
    shape_info: dict = field(default_factory=dict)


# ---------- softmax (row-wise) ----------
def _softmax_inputs(M=8192, N=4096, dtype=torch.float16, device="cuda"):
    return (torch.randn(M, N, device=device, dtype=dtype),)


def _softmax_ref(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


# ---------- bias + GELU (tanh approximation) ----------
def _bias_gelu_inputs(M=8192, N=4096, dtype=torch.float16, device="cuda"):
    x = torch.randn(M, N, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)
    return (x, b)


def _bias_gelu_ref(x, b):
    return F.gelu(x.float() + b.float(), approximate="tanh").to(x.dtype)


# ---------- residual add + LayerNorm ----------
def _add_layernorm_inputs(M=8192, N=4096, dtype=torch.float16, device="cuda"):
    x = torch.randn(M, N, device=device, dtype=dtype)
    r = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)
    return (x, r, w, b)


def _add_layernorm_ref(x, r, w, b):
    h = x.float() + r.float()
    return F.layer_norm(h, (h.shape[-1],), w.float(), b.float(), eps=1e-5).to(x.dtype)


TASKS = {
    "softmax": Task(
        name="softmax",
        description="Row-wise softmax over the last dim of an (M, N) fp16 tensor.",
        make_inputs=_softmax_inputs,
        reference=_softmax_ref,
        atol=1e-3,
        rtol=1e-2,
        shape_info={"x": "(8192, 4096) fp16"},
    ),
    "bias_gelu": Task(
        name="bias_gelu",
        description="y = gelu_tanh(x + b), x (M, N) fp16, b (N,) fp16 broadcast over rows.",
        make_inputs=_bias_gelu_inputs,
        reference=_bias_gelu_ref,
        shape_info={"x": "(8192, 4096) fp16", "b": "(4096,) fp16"},
    ),
    "add_layernorm": Task(
        name="add_layernorm",
        description="y = layer_norm(x + r) * w + b over the last dim, eps=1e-5, all fp16.",
        make_inputs=_add_layernorm_inputs,
        reference=_add_layernorm_ref,
        atol=2e-2,
        rtol=2e-2,
        shape_info={"x": "(8192, 4096)", "r": "(8192, 4096)", "w": "(4096,)", "b": "(4096,)"},
    ),
}
