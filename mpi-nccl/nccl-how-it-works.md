# NCCL — How It Actually Works

If you've spent any time debugging distributed training jobs, you've probably stared at NCCL debug output and wondered what it's actually telling you. This post breaks down how NCCL works under the hood — the algorithms, the transport selection logic, the bootstrap process, and how to read the debug output when things go wrong. Once you understand the internals, the behavior stops being mysterious and starts being predictable.

---

## What NCCL Actually Is (and Isn't)

Let's start by clearing up a common misconception: NCCL is not a networking library. It doesn't implement TCP or InfiniBand protocols. What it does is sit between your training code and the hardware transport layer — NVLink, InfiniBand, shared memory — and orchestrate how tensors move through whatever hardware is already there.

When your training code calls `ncclAllReduce(sendbuf, recvbuf, count, ...)`, here's what NCCL does:

1. Selects an algorithm (Ring or Tree)
2. Selects a transport for each GPU pair (NVLink, IB, SHM)
3. Breaks the buffer into chunks and assigns them to channels
4. Executes the algorithm using CUDA kernels that drive the chosen transport

That's the whole job. NCCL is the orchestration layer that decides how to move tensors through your hardware. Understanding each of those four steps is what this post is about.

---

## Ring All-Reduce: The Workhorse

Ring all-reduce is the default algorithm for large messages (anything over roughly 4 MB). The concept is elegant: arrange every GPU in a ring where each has exactly one left neighbor and one right neighbor, then execute two phases.

### Phase 1: Reduce-Scatter

Each GPU sends a chunk of its data to the next GPU in the ring, accumulates the incoming chunk into its own, then forwards. After N-1 steps, each GPU holds exactly one chunk of the fully-reduced result — but only its assigned chunk, not the full tensor.

```
GPU0 → GPU1 → GPU2 → GPU3 → GPU0 (ring)

Step 1: GPU0 sends chunk0 to GPU1
        GPU1 adds chunk0 to its own chunk0, forwards to GPU2
        ...
After N-1 steps: each GPU has one fully-reduced chunk
```

### Phase 2: All-Gather

Now each GPU broadcasts its reduced chunk around the ring. After another N-1 steps, every GPU has the complete result.

The total data movement per GPU works out to `2 × (N-1)/N × message_size`. This is why the bus bandwidth formula looks the way it does:

```
busbw = (size / time) × 2(N-1)/N
```

The `2(N-1)/N` factor normalizes out the ring inefficiency — at large N it approaches 2. For N=8: `2 × 7/8 = 1.75`. So if algbw = 4.6 GB/s, then busbw ≈ 4.6 × 1.75 ≈ 8 GB/s. Still terrible in that example, but that's what the formula is telling you.

### The Slow GPU Problem

Here's the critical property of ring all-reduce that explains a lot of mysterious performance regressions: it's lockstep. Every GPU waits for its neighbor before proceeding to the next step. **One slow GPU stalls all N GPUs at every ring step.**

This is not an edge case. When MNNVL is broken on one node in a GB200 cluster and that node falls back to shared memory, the slowest link in the ring becomes that node's inter-GPU path. The entire job's bandwidth collapses to match it. A 15x regression from a single misconfigured node is not hypothetical — it's exactly what the math predicts.

---

## Tree All-Reduce: When Latency Beats Bandwidth

Tree all-reduce uses a binary tree structure instead of a ring. The number of steps is log2(N) instead of N-1, which means far fewer round trips for small tensors.

The tradeoff: tree is better for latency but worse for bandwidth efficiency compared to ring, because not all GPUs are communicating with all other GPUs simultaneously.

NCCL uses Tree by default for messages smaller than roughly 4 MB, and Ring for everything larger. You can override this if you have a reason to:

```bash
NCCL_ALGO=Ring    # force ring for all message sizes
NCCL_ALGO=Tree    # force tree for all message sizes
```

In practice, the auto-selection is good. You'd only force this during debugging or benchmarking.

---

## Channels: Why One Ring Isn't Enough

NCCL doesn't run a single ring. It runs multiple rings in parallel — these are called **channels**. Each channel is an independent ring that operates concurrently on different slices of your data.

Why does this matter? A single ring serializes all data movement through a single pipeline. Two rings in parallel halve that serialization. On hardware with multiple NICs per GPU (like GB200 nodes with 4 NICs), NCCL can saturate all NIC bandwidth simultaneously by running 8–16 channels at once.

More channels means better bandwidth utilization, but also more CUDA kernel overhead. NCCL auto-tunes this based on your hardware configuration, but you can set it manually:

```bash
NCCL_NCHANNELS=8    # set channel count explicitly
```

With 4 IB NICs per node (typical for GB200), expect NCCL to auto-configure 8–16 channels.

---

## Transport Selection: Where the Magic Happens

This is where everything comes together. For each GPU pair in a communicator, NCCL picks the best transport by querying what the hardware actually reports. The hierarchy from fastest to slowest:

```
P2P/NVL    NVLink — direct GPU↔GPU transfer, no CPU involvement, no NIC
           Requires: NVLink topology detected + valid fabric UUID (non-zero)

P2P/IPC    PCIe peer-to-peer — same node, no NVLink
           Fallback when NVLink is physically absent

SHM        Shared memory via /dev/shm — CPU-staged transfers
           Used when P2P is disabled (e.g., fabric UUID = 0.0 on GB200,
           or ACS blocking on H100/H200/B200)

NET/IB     InfiniBand + GPUDirect RDMA
           Primary transport for cross-node GPU pairs

NET/Socket TCP socket — last resort
           When IB is not found or NCCL_IB_HCA is not set
```

NCCL announces its transport choices at communicator init. With `NCCL_DEBUG=INFO` you get the full picture:

```bash
NCCL_DEBUG=INFO mpirun ... 2>&1 | grep "Ring.*via"

# What you want to see:
# [rank0] Ring 00 : 0[b0000] -> 1[b0000] via P2P/NVL        ← NVLink, same node
# [rank0] Ring 00 : 3[b0000] -> 4[b0000] via NET/IB/GDRDMA  ← IB cross-node

# What means something is wrong:
# [rank0] Ring 00 : 0[b0000] -> 1[b0000] via SHM            ← P2P disabled
# [rank0] Ring 00 : 3[b0000] -> 4[b0000] via NET/Socket     ← IB not found
```

### Transport Differences Across GPU Generations

The same transport hierarchy applies across all GPU generations, but the triggers for each path differ significantly depending on your hardware:

**H100 / H200 / B200 (single-node):** Within a single node, GPUs communicate via NVLink through the NVSwitch. NCCL uses `P2P/NVL` for all intra-node GPU pairs. If you see `SHM` here, it almost always means PCIe ACS (Access Control Services) is blocking peer-to-peer transfers — this is a BIOS or driver setting issue, not a fabric manager issue. Check your BIOS for ACS settings and verify that the NVIDIA driver has properly configured the PCIe topology.

**H100 / H200 / B200 (multi-node):** Cross-node traffic always goes over InfiniBand (`NET/IB`). There is no MNNVL option on these platforms — NVLink is physically scoped to a single node's NVSwitch. If you see `NET/Socket` here instead of `NET/IB`, NCCL can't find your IB HCAs (check `NCCL_IB_HCA`).

**GB200 (single-node):** Within a single GB200 node (4 GPUs), NCCL uses `P2P/NVL` via the local NVSwitch fabric.

**GB200 (multi-node with MNNVL):** This is what makes GB200 different from every prior generation. The NVLink fabric extends across nodes within a rack — called Multi-Node NVLink (MNNVL). When the fabric manager has correctly provisioned the clique, NCCL detects a valid fabric UUID and selects `P2P/NVL` for cross-node GPU pairs, achieving true NVLink bandwidth across nodes. When the fabric UUID is zero (`0.0`), the fabric wasn't provisioned and NCCL falls back. The `MNNVL` transport you see in debug output is GB200-specific — you won't see it on H100/H200/B200.

---

## Why NCCL Disables P2P: Very Different Causes on Different Hardware

You might see this warning in your NCCL output:

```
NCCL WARN P2P is disabled between NVLINK connected GPUs 1 and 0.
This should not be the case given their connectivity, and is probably due to a hardware issue.
```

The warning says "hardware issue," but that's misleading in both cases where it appears. Here's what's actually happening, and why the root cause is completely different depending on your GPU generation:

**On GB200:** This warning means fabric UUID = 0.0. NCCL can see the physical NVLink bonds in the topology (it detects NV18 or similar in the topo), but it cannot use them because the fabric manager hasn't programmed the NVSwitch to admit this GPU into the clique. The physical hardware is fine. The provisioning isn't. Check whether the fabric manager service is running and whether it has successfully established a valid clique ID. This is a fabric manager issue, full stop.

**On H100 / H200 / B200:** The same warning here means something different entirely. These platforms don't use MNNVL, so fabric UUID isn't relevant for intra-node P2P. Instead, P2P gets disabled when PCIe ACS is enabled in your BIOS or when the platform's PCIe topology prevents direct GPU-to-GPU transfers. This is typically a BIOS/driver configuration issue — check your BIOS ACS settings, verify that `nvidia-smi topo -m` shows NVLink connectivity, and ensure the NVIDIA driver isn't reporting any PCIe topology errors.

The consequence in both cases is that NCCL falls back to SHM, but the fix is completely different. Knowing your hardware generation is the first step in diagnosing this correctly.

---

## NCCL Bootstrap: The Part That's Not NVLink or IB

Before any collective operation can run, all ranks need to find each other. This is the bootstrap phase, and it runs on a completely separate path from NCCL's data transport.

The bootstrap flow:
1. Rank 0 opens a TCP socket on a well-known address
2. All other ranks connect to rank 0 and exchange their addresses
3. NCCL builds the full communicator with the complete rank-to-address map
4. From this point on, data collectives use NVLink/IB — not the bootstrap socket

The bootstrap socket is always TCP, regardless of what transport NCCL ends up using for data. This is why setting `NCCL_IB_HCA` doesn't affect bootstrap — IB is data-plane only.

When you see this error, the bootstrap connection failed:

```
NCCL WARN socketProgress: Connection closed by remote peer
```

This almost always means another rank crashed and dropped the bootstrap connection. The error you're looking at is a secondary symptom. Find the rank that crashed first — it will have a more informative error above this one in the logs.

---

## Key Environment Variables

These are the ones that matter day-to-day:

```bash
NCCL_DEBUG=INFO              # full init log — transport selection, ring topology, UUID
NCCL_DEBUG=WARN              # errors and warnings only (good for production)

NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4   # tell NCCL which NICs to use for IB
                                              # exclude mlx5_0 if that's your Ethernet NIC

NCCL_SOCKET_IFNAME=^mlx5_0  # exclude this interface from bootstrap socket search
                              # ^ prefix means "exclude matching interfaces"

NCCL_IGNORE_DISABLED_P2P=1  # proceed even when P2P is disabled
                              # NCCL falls back to SHM — degraded but functional

NCCL_ALGO=Ring               # force algorithm (Ring or Tree)
NCCL_NCHANNELS=8             # force channel count
NCCL_TIMEOUT=120             # seconds before NCCL gives up waiting (default: 30)
```

---

## Reading NCCL_DEBUG Output

When something is wrong, these are the grep patterns that get you to the answer fastest:

```bash
# 1. Fabric UUID — check this first when debugging GB200 P2P issues
grep -i "MNNVL\|fabric\|uuid" output.txt

# Healthy:
# NCCL INFO MNNVL busId 0xb0000 fabric UUID 7a4af5e1... cliqueId 0x7ffe
# Broken:
# NCCL INFO MNNVL busId 0xb0000 fabric UUID 0.0 cliqueId 0x0

# 2. P2P warnings — are any GPUs falling back?
grep "P2P is disabled" output.txt

# 3. Transport selection — the most important line for understanding what path is used
grep "Ring.*via" output.txt

# 4. IB device selection — verify NCCL found the right NICs
grep "IB.*mlx5" output.txt

# 5. Everything at once — good starting point for triage
grep -iE "MNNVL|fabric|uuid|P2P|Ring.*via|WARN" output.txt | head -40
```

---

## Performance Expectations by Hardware Generation

Now that you understand the transport hierarchy, here's what to expect from nccl-tests across the hardware generations you're likely to encounter. These are `busbw` numbers from `all_reduce_perf` at large message sizes (≥ 1 GB).

| Hardware | Scenario | Expected busbw | Transport Used |
|---|---|---|---|
| **H100 (8x, NVLink 4.0)** | Single-node | ~350–400 GB/s | P2P/NVL intra-node |
| **H100** | Multi-node (IB NDR) | ~25–40 GB/s | NET/IB cross-node |
| **H200 (8x)** | Single-node | ~350–400 GB/s | P2P/NVL intra-node (similar to H100) |
| **H200** | Multi-node (IB NDR) | ~25–40 GB/s | NET/IB cross-node |
| **B200 (8x, NVLink 5.0)** | Single-node | ~700–800 GB/s | P2P/NVL intra-node |
| **B200** | Multi-node (IB NDR) | ~25–40 GB/s | NET/IB cross-node |
| **GB200 (4x per node)** | Single-node | ~400–500 GB/s | P2P/NVL intra-node |
| **GB200** | Multi-node, MNNVL healthy | ~150–200 GB/s | P2P/NVL via MNNVL fabric |
| **GB200** | Multi-node, MNNVL broken (SHM fallback) | ~4–8 GB/s | SHM on affected node |
| **GB200** | Multi-node, IB only (no MNNVL) | ~25–40 GB/s | NET/IB cross-node |

A few things worth calling out in this table:

**B200 vs H100 intra-node:** NVLink 5.0 on B200 nearly doubles intra-node bandwidth compared to NVLink 4.0 on H100. If you're doing large gradient all-reduces within a node, this matters a lot.

**GB200 MNNVL healthy vs broken:** This is the most dramatic difference in the table — ~150–200 GB/s versus ~4–8 GB/s. A 15–40x regression from a fabric provisioning issue. This is why validating MNNVL health before running any multi-node job on GB200 is not optional. Run `nccl-tests` between nodes before you assume the fabric is good.

**H100/H200/B200 multi-node:** Cross-node bandwidth is bounded by InfiniBand regardless of generation — NDR (400 Gb/s per port) caps you at roughly 25–40 GB/s busbw in practice with typical rail configurations. GB200's MNNVL healthy path is 4–6x better than IB for cross-node transfers, which is the whole point of the architecture.

**GB200 single-node vs H100 single-node:** GB200 nodes have 4 GPUs (not 8), but those 4 GPUs achieve 400–500 GB/s busbw. The per-GPU NVLink bandwidth is higher, but comparing to H100's 8-GPU number requires accounting for the smaller GPU count and different node topology.

The ring locking to the slowest link is not hypothetical — whenever you see a dramatic performance regression in multi-node training, the first question is always: which node is the bottleneck, and what transport is it using?

---

## Putting It Together: A Debugging Workflow

When a multi-node job is running slower than expected, here's the order of operations:

1. **Enable `NCCL_DEBUG=INFO`** and capture output from all ranks
2. **Check fabric UUIDs** — on GB200, `grep -i "uuid" output.txt` on each node. Any `0.0` means that node's MNNVL is broken
3. **Check transport selection** — `grep "Ring.*via" output.txt`. Any `SHM` on same-node pairs or `NET/Socket` on cross-node pairs is a problem
4. **Check P2P warnings** — `grep "P2P is disabled" output.txt`. Use your hardware generation to interpret the root cause (ACS on H100/B200, fabric manager on GB200)
5. **Run nccl-tests in isolation** — before adding your workload back, verify the fabric is healthy with `all_reduce_perf` between nodes. If nccl-tests is slow, your workload will be slow too

NCCL is transparent about what it's doing — you just need to know where to look.
