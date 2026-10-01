# Kernel Swarm 🧬⚡

A multi-agent system that **evolves GPU kernels**. LLM agents write Triton kernels, a cluster of GPUs benchmarks them in parallel, and the fastest *correct* ones survive and get mutated further, in the spirit of DeepMind's AlphaEvolve.

> Status: **Phase 1, foundation.** Benchmark harness + hand-written baselines.

## Why fused ops?
PyTorch eager runs ops like `bias + GELU` or `residual + LayerNorm` as several separate GPU launches, each reading and writing memory. A single fused kernel does it in one pass. That is where real, measurable speedups live, and where the swarm hunts.

Every result is reported against **two** baselines: PyTorch eager, and `torch.compile` (which also fuses). Beating the second one is the real bar.

## Tasks
| Task | Op | Baseline kernel |
|---|---|---|
| `softmax` | row-wise softmax, (8192, 4096) fp16 | `kernels/baseline/softmax.py` |
| `bias_gelu` | `gelu_tanh(x + b)` | `kernels/baseline/bias_gelu.py` |
| `add_layernorm` | `layer_norm(x + r) * w + b` | none yet, left for the swarm |

## Quickstart (BU SCC)
```bash
bash setup.sh                       # creates kernel-swarm-venv in the project, installs torch
source kernel-swarm-venv/bin/activate
python -m kernel_swarm.bench --all-baselines --compile
```

A candidate kernel is any Python file that defines `kernel_fn(*inputs)`. The harness checks it against the PyTorch reference on two random seeds, then times it. Candidates run in an isolated subprocess with a timeout, so a broken kernel can't crash the swarm.

## Roadmap
- [x] Phase 1: benchmark harness, baselines
- [ ] Phase 2: single agent loop (write → test → feedback → retry)
- [ ] Phase 3: swarm: population, Proposer / Mutator / Evaluator / Selector / Analyst, parallel SGE job arrays
- [ ] Phase 4: leaderboard, evolution-tree visualization, results write-up
