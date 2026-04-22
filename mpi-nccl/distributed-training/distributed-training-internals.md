# Distributed Training Internals

What actually happens from `loss.backward()` to a weight update — and why DDP, FSDP, and MoE each make different choices about which collective to use and when.

---

## Start With the Mental Model

Before diving into each framework, here is the skeleton of every distributed training step:

```
Forward pass  →  compute activations (pure CUDA, no collectives)
loss = criterion(output, target)
loss.backward()  →  compute gradients (pure CUDA, no collectives yet)
[framework hook fires]  →  gradient collective (NCCL)
optimizer.step()  →  update parameters
checkpoint (optionally)  →  save state to disk
```

The thing most people get wrong early on: `loss.backward()` does **not** call NCCL. It computes gradients using autograd on the local GPU. The collective — the moment data moves across the network — happens when the framework (DDP, FSDP, or your MoE router) fires a hook that says "okay, now sync." That decision is framework-level, not PyTorch-core.

This matters because it means you can have a hang at `ncclAllReduce` whose root cause is actually a Python exception during data loading on one rank. The collective is just where the symptom surfaces.

---

## DDP: The Simple Case That Works Until It Doesn't

DistributedDataParallel is the conceptually clean approach. Every GPU has a complete copy of the model. You split your batch across GPUs (rank 0 gets items 0–15, rank 1 gets 16–31, and so on). Each GPU independently runs forward and backward on its own data shard.

After backward, DDP hooks fire and all-reduce the gradients:

```
GPU0 grad: [dW_from_shard_0]
GPU1 grad: [dW_from_shard_1]
...
GPU7 grad: [dW_from_shard_7]

After all-reduce: every GPU holds [sum_of_all_dW / N]
```

Every GPU then calls `optimizer.step()` with the same averaged gradient and produces identical updated weights. If you think about it, there is something wasteful here — 8 GPUs all computing the exact same Adam step on the exact same weight tensor. That is redundant optimizer work. But DDP accepts that tradeoff because the simplicity is worth it when the model fits in memory.

**On modern hardware,** the all-reduce goes over NVLink for intra-node GPU pairs (fast, in the ~TB/s range) and over InfiniBand for cross-node communication. DDP runs well on H100, H200, B200, and GB200 nodes. The hard ceiling is per-GPU HBM: 80 GB on H100, 141 GB on H200, and 192 GB on B200/GB200. Once your model — weights, optimizer state, activations, and gradients combined — exceeds that, DDP stops being an option and you need FSDP.

**Why it is still the right choice for smaller models:** the full model fits on one GPU, implementation is straightforward, and debugging is much simpler than FSDP. For anything under roughly 40–60B parameters (depending on optimizer state and precision), DDP is hard to beat on operational simplicity.

**Collective used:** `ncclAllReduce`

---

## FSDP: Sharding Your Way Past the HBM Ceiling

When your model does not fit on a single GPU, you need a different strategy. Fully Sharded Data Parallel takes the model and distributes it — rank 0 holds parameter shard 0, rank 1 holds shard 1, and so on. No single GPU ever holds the full model in memory at the same time.

The tradeoff is that you now need two collective operations per layer instead of one collective per step. Let's walk through why.

### Why the all-gather before forward?

Before FSDP can compute a layer's forward pass, it needs the full weights for that layer — but right now those weights are scattered across GPUs. So it runs an **all-gather**: every GPU broadcasts its shard to all other GPUs, and the result is that all GPUs temporarily hold the complete layer weights.

```
GPU0: shard_0  →  all-gather  →  full_layer_weights (temporary)
GPU1: shard_1  →  all-gather  →  full_layer_weights (temporary)
...
```

The key word is "temporarily." After the forward computation for that layer finishes, FSDP **discards the reconstructed weights**. They can be re-fetched during backward when needed. This is what lets FSDP keep peak HBM usage at roughly 1/N of the full model — you pay for one layer's worth of temporary expansion at a time, not the whole model.

### Why the reduce-scatter after backward?

After computing gradients for a layer, each GPU has computed gradient contributions from all of its data. But you only need the summed gradient for your own parameter shard. So FSDP runs a **reduce-scatter**: gradients are summed across all GPUs, and each GPU ends up holding the summed gradient for its own shard only.

```
GPU0 ends up with: dW_shard_0 (summed contributions from all ranks)
GPU1 ends up with: dW_shard_1
...
```

Each GPU then runs `optimizer.step()` only on its own shard. No redundant work. HBM usage scales as 1/N.

### Why FSDP scales better on GB200

The all-gather — reconstructing a full layer's worth of parameters before each forward pass — is the performance-sensitive operation to understand for large models. You are moving potentially hundreds of gigabytes of parameter data on every training step, once per layer.

On H100 and H200, when FSDP spans multiple nodes, the all-gather has to travel over InfiniBand (typically 400 Gb/s per link). That is the bottleneck.

On GB200 systems within the same rack, NVLink extends across nodes through MNNVL (Multi-Node NVLink), delivering approximately 1.8 TB/s of bandwidth. If the MNNVL fabric is healthy, the all-gather for same-rack GPUs goes over NVLink instead of IB — roughly 30x more bandwidth. This is why GB200 clusters scale considerably better for FSDP across many nodes: the most expensive collective in FSDP's hot path is running on much faster interconnect.

**Collective sequence:** `ncclAllGather` (before each layer's forward) + `ncclReduceScatter` (after each layer's backward)

**When to use over DDP:** when the model does not fit on a single GPU. FSDP makes 70B+ parameter training practical.

---

## MoE: Token Routing Is a Collective Problem

Mixture of Experts adds a fundamentally different collective pattern. The idea is that instead of running every token through the same dense feedforward layer, a router selects 1–2 expert sub-networks per token, and those experts are distributed across GPUs. GPU 0 might hold expert 0 and expert 4, GPU 1 holds expert 1 and expert 5, and so on.

The challenge: after the router decides which expert each token should go to, the tokens need to physically move to the GPU that holds the target expert. This is **dispatch**. After the expert processes the tokens, results need to come back to the original GPU. This is **combine**.

Both operations are all-to-all: every GPU sends data to every other GPU, but the volumes are unequal and not known in advance — they depend on how many tokens the router sent to each expert at this particular step.

```
GPU0 dispatches: 32 tokens to GPU1's expert, 0 tokens to GPU2's expert, 8 tokens to GPU3's expert
GPU1 dispatches: 7 tokens to GPU0's expert, 40 tokens to GPU2's expert, ...
```

This dynamic, variable-volume nature of the all-to-all is worth understanding. Standard NCCL all-to-all is synchronous — all GPUs must call it at the same time and the volumes have to be communicated upfront. NVSHMEM is more natural for MoE because each GPU can issue puts directly without requiring all-ranks synchronization before knowing the send volumes. (That is covered in detail in the NVSHMEM topic.)

### The GB200 advantage for MoE

On H100/H200 clusters, the dispatch all-to-all for cross-node token routing goes over IB. On GB200 with MNNVL healthy, same-rack token dispatch can travel over NVLink instead — significantly higher throughput. For token-heavy MoE models with large batch sizes, this difference is substantial: you are doing this all-to-all on every forward pass, at every MoE layer.

**Collective used:** `ncclAllToAll`

---

## Checkpointing: Saving State Without Losing Your Mind

After `optimizer.step()`, the weights are updated. Checkpointing saves enough state that if a process dies or a GPU goes down, training can resume from that point rather than from scratch. On a large cluster running for weeks, checkpoint strategy matters.

### DDP: rank 0 writes, others wait

In DDP, every GPU has identical weights. Rank 0 writes the checkpoint; the others wait at a barrier.

```python
if dist.get_rank() == 0:
    torch.save(model.state_dict(), "checkpoint.pt")
dist.barrier()  # ranks 1–7 wait here
```

The other ranks are not doing wasted work — their gradients were the training signal. They just have nothing to write because rank 0 already captured everything. Single file, simple to restore.

### FSDP: every rank must write its shard

In FSDP, each rank holds a different piece of the model. If only rank 0 checkpointed, you would lose the rest of the model entirely. Every rank writes its own shard:

```python
from torch.distributed.checkpoint import save
save({"model": model}, storage_writer=..., planner=...)
```

This produces N files on disk (one per rank) but writes them in parallel, so the wall-clock time to checkpoint can be comparable to DDP's single-file write. The distributed checkpoint format in PyTorch is also flexible — you can reload a 512-rank checkpoint onto a 256-rank job if needed.

**A practical scaling concern:** on large clusters with hundreds of GPUs, all ranks writing simultaneously to a shared filesystem (Lustre, GPFS, or similar) can create a real I/O bottleneck. If 512 ranks all open files and write at the same time, you can saturate the metadata server or the aggregate write bandwidth. Teams running at that scale often checkpoint to node-local NVMe first and then asynchronously copy to the shared filesystem, or stagger writes across ranks.

### What the checkpoint must include

Model weights alone are not enough to resume correctly. You need:

- Model weights (obviously)
- Optimizer state (Adam's first and second moment estimates, or equivalent)
- LR scheduler state
- Step number / epoch
- RNG state (so data sampling is reproducible)

Missing any of these and your resumed run will diverge from the original trajectory.

### How often to checkpoint

You cannot checkpoint every step — the I/O cost is too high. Typical approaches:

- Every 100–1000 steps for long runs
- Time-based (every N minutes) to guarantee a maximum loss window
- Before any planned maintenance window

---

## The Full Workflow, Step by Step

```
1. mpirun launches 8 processes (one per GPU)

2. Each process calls:
   dist.init_process_group("nccl")
   → NCCL bootstrap TCP handshake
   → NCCL builds communicator and selects transport (NVLink, IB, etc.)

3. Data loading (rank 0 loads and broadcasts, or each rank loads its own shard)

4. For each training step:

   a. Forward pass:
      - (FSDP only: all-gather to reconstruct each layer before computing it)
      - Pure CUDA: matmul, activations, attention
      - (MoE only: all-to-all dispatch during forward)
      - No NCCL for DDP during this phase

   b. loss = criterion(output, labels)

   c. loss.backward()
      - Pure CUDA: autograd computes gradients layer by layer
      - (FSDP: reduce-scatter fires per-layer as gradients are computed)
      - No NCCL for DDP during this phase

   d. Gradient sync (DDP):
      - ncclAllReduce sums all gradients across 8 GPUs
      - FSDP has already done this per-layer in step c

   e. optimizer.step()
      - Updates parameters using the synced gradient
      - Pure CPU + CUDA, no NCCL

   f. (Optionally) checkpoint to distributed filesystem

5. Repeat until convergence
```

---

## Where Each Collective Appears

| Framework | When | Collective | Data |
|---|---|---|---|
| DDP | After backward | AllReduce | Gradients |
| FSDP | Before each layer forward | AllGather | Parameters |
| FSDP | After each layer backward | ReduceScatter | Gradients |
| MoE dispatch | During forward | AllToAll | Tokens |
| MoE combine | During forward | AllToAll | Expert outputs |
| Gradient clipping | Before optimizer.step | AllReduce | Gradient norms |

This table is worth internalizing. When you are looking at a Nsight Systems trace and see NCCL kernels firing, this tells you which phase of training you are looking at and which framework operation triggered it.

---

## What Goes Wrong and Where

**Collective hangs.** One rank fails to call the collective — maybe an exception during data loading, maybe an OOM on one GPU, maybe a network timeout. All other ranks wait forever at `ncclAllReduce`. The hang manifests at the collective, but the root cause is elsewhere. Always check rank logs, not just the hanging rank.

```bash
NCCL_TIMEOUT=120    # seconds NCCL waits before failing rather than hanging silently
```

**OOM during FSDP all-gather.** Reconstructing a full layer temporarily requires peak memory equal to the full layer weight, not just your shard. If the all-gathered weight tensor plus your in-flight activations plus gradients exceeds HBM, you get OOM even though your steady-state shard fits fine. The standard fix is activation checkpointing: instead of storing activations during forward for use in backward, you recompute them during backward on demand. You trade compute for memory.

**Gradient explosion.** `loss.backward()` produces NaN or Inf gradients on one or more GPUs. The all-reduce propagates NaN to every GPU. Every subsequent computation is corrupted. The fix is gradient clipping — `torch.nn.utils.clip_grad_norm_()` — applied before the collective fires. Note that gradient clipping itself requires an all-reduce to compute the global gradient norm, which is why it appears in the collectives table above.
