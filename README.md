# Kernel Swarm 🧬⚡

A multi-agent system that **evolves GPU kernels**. LLM agents write Triton kernels, a cluster of GPUs benchmarks them in parallel, and the fastest *correct* ones survive and get mutated further, in the spirit of DeepMind's AlphaEvolve.

> Status: **Phase 3, the swarm.** An evolutionary population of specialist LLM agents breeds faster Triton kernels.

## Results so far (NVIDIA A100-SXM4-80GB)
| Task | Agent kernel | PyTorch eager | vs eager | vs `torch.compile` |
|---|---|---|---|---|
| `add_layernorm` (8192×4096 fp16) | 0.125 ms | 0.932 ms | 7.47x | **1.04x** (3/3 runs) |

The add_layernorm kernel was written by a single Gemini Flash agent in 8 attempts and runs at roughly 79% of the A100's peak memory bandwidth.

## Why fused ops?
PyTorch eager runs ops like `bias + GELU` or `residual + LayerNorm` as several separate GPU launches, each reading and writing memory. A single fused kernel does it in one pass. That is where real, measurable speedups live, and where the swarm hunts.

Every result is reported against **two** baselines: PyTorch eager, and `torch.compile` (which also fuses). Beating the second one is the real bar.

## Tasks
| Task | Op | Baseline kernel |
|---|---|---|
| `softmax` | row-wise softmax, (8192, 4096) fp16 | `kernels/baseline/softmax.py` |
| `bias_gelu` | `gelu_tanh(x + b)` | `kernels/baseline/bias_gelu.py` |
| `add_layernorm` | `layer_norm(x + r) * w + b` | none, left for the agents |
| `cross_entropy` | per-row loss over a 50,257-word vocab (needs a streaming logsumexp) | none, the hard one |

## Quickstart (BU SCC)
```bash
bash setup.sh                       # creates kernel-swarm-venv in the project, installs torch
source kernel-swarm-venv/bin/activate
python -m kernel_swarm.bench --all-baselines --compile
```

A candidate kernel is any Python file that defines `kernel_fn(*inputs)`. The harness checks it against the PyTorch reference on two random seeds, then times it. Candidates run in an isolated subprocess with a timeout, so a broken kernel can't crash the swarm.

## Run the agent
```bash
cp .env.example .env      # then put your API key in .env
python -m kernel_swarm.agent --task add_layernorm --iters 8
```
Every attempt is saved in `runs/<task>/<timestamp>/`, and the fastest correct kernel goes to `kernels/best/<task>.py`. An anti-cheat check rejects kernels that just call the PyTorch op they are supposed to replace.

## The swarm
```bash
python -m kernel_swarm.evolve --task cross_entropy --generations 5 --children 6
```
Each generation, specialist agents write children from parents picked by tournament selection:

| Agent | Job |
|---|---|
| Memory | cut memory traffic: fusion, single pass, wide loads |
| Tuner | launch config: block sizes, warps, rows per program, autotune |
| Explorer | a structurally different algorithm |
| Crossover | merge the best ideas of two parents |
| Repair | one shot at fixing each failed child using its error |
| Analyst | explains why the champion is fast (`analysis.md`) |

All agents read a shared **blackboard** of recent ideas and their results, so the swarm learns from its own failures. Fitness is speedup vs PyTorch eager measured on the same GPU, and every kernel also reports **% of peak memory bandwidth** (roofline). `--eval sge` benchmarks each generation in parallel as an SGE job array on BU SCC (experimental). The full family tree is saved in `runs/<task>/swarm-*/lineage.json`.

## Roadmap
- [x] Phase 1: benchmark harness, baselines
- [x] Phase 2: single agent loop (write → anti-cheat check → benchmark → feedback → retry)
- [x] Phase 3: evolutionary swarm: specialist agents, blackboard, repair, roofline metrics, SGE job arrays
- [ ] Phase 4: leaderboard, evolution-tree visualization, results write-up
