# Why the winner is fast: `cross_entropy`

## Lineage
- gen 1 #3 [fresh] 0.2792 ms (10.69x eager, 72% peak BW): We implement a single-pass online log-sum-exp kernel where each Triton program processes one row of logits across chunked column blocks, maintaining numerical stability via running maximums. The targe
- gen 2 #6 [memory] 0.2765 ms (10.79x eager, 73% peak BW): We optimize memory throughput by separating the vocabulary loop into unmasked full-block loads and a single masked remainder, enabling unconditional 128-bit vectorized transfers. Setting the innermost
- gen 3 #14 [explorer] 0.2606 ms (11.45x eager, 77% peak BW): We structurally transform the row-streaming memory access pattern to achieve full 16-byte alignment across all rows regardless of non-power-of-2 vocab size $V=50257$. By prepending a small aligned off
- gen 4 #21 [crossover] 0.2600 ms (11.49x eager, 78% peak BW): We combine the 16-byte row alignment technique from Parent 1 with early asynchronous target logit prefetching to completely hide target DRAM latency behind the streaming reduction. We also propagate a
- gen 5 #28 [memory] 0.2584 ms (11.55x eager, 78% peak BW): We optimize the memory pipeline by overlapping DRAM requests at the kernel prologue: the target index load is issued first, followed immediately by the aligned Chunk 0 load, allowing the dependent tar

## Analysis
### Technical Analysis

#### What the Kernel Does
The kernel computes row-wise cross-entropy loss $\text{loss}_i = \text{logsumexp}(\text{logits}_i) - \text{logits}_{i, \text{target}_i}$ for a non-power-of-2 vocabulary ($V = 50257$). Because an entire row cannot fit within shared memory / register limits, it streams across the row in chunks using online log-sum-exp with running maximums and running normalizers, performing reductions across warps while reading the single target logit per row.

#### Lineage & Key Optimizations
1. **Streaming Online Log-Sum-Exp (Gen 1, 0.2792 ms):** Avoided two-pass memory traffic (one for max, one for exp-sum) by updating running maximums and scaling accumulators on the fly, immediately reaching ~72% DRAM peak bandwidth.
2. **Loop Peeling & Vectorized Transfers (Gen 2, 0.2765 ms):** Separated loop iterations into unconditional full chunks and a single masked remainder tail. This eliminated predicated load overhead and enabled compiler generation of 128-bit memory instructions.
3. **16-byte Row Alignment via Prepending (Gen 3, 0.2606 ms):** Since $V = 50257$ is odd, row start offsets alternate alignment. Backtracking row pointers by `(start_idx % 8)` elements and masking chunk 0 ensured every program started on a strict 16-byte boundary, enabling coalesced 128-bit vector loads across all inner iterations.
4. **Latency Hiding & Memory Overlap (Gens 4–5, 0.2584 ms):** Pipelined independent DRAM requests during kernel prologue: `target_idx` is requested first, chunk 0 vector load is issued concurrently, and the dependent target logit is fetched before starting the reduction loop, hiding pointer-chasing round-trips.

#### Performance Limits & Theoretical Ceiling
At 0.2584 ms, streaming $4096 \times 50257 \times 2$ bytes ($\approx 411.7$ MB) yields $\approx 1.59 \text{ TB/s}$, representing **~78% of A100 SXM4’s 2.039 TB/s theoretical HBM bandwidth**. Further gains are strictly bandwidth-bounded: remainder padding adds slight load amplification, small scatter/gather overhead remains for target logits, and DRAM burst efficiency is bounded near 80–85% for short unaligned row strides.
