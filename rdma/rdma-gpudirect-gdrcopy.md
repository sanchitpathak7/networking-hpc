# RDMA, GPUDirect RDMA, and GDRCopy: Cutting the CPU Out of the Data Path

If you've spent any time tuning distributed GPU training, you've probably seen these three terms thrown around together — RDMA, GPUDirect RDMA, GDRCopy — and felt like they were describing variations of the same thing. They're not. They solve related but distinct problems, and understanding *which* bottleneck each one eliminates is what separates someone who can debug slow all-reduce from someone who just restarts the job and hopes for the best.

Let's start with the actual problem, then work our way through the solutions in the order they were invented.

---

## The Problem: Your CPU Is in the Way

Picture a standard GPU training job sending activations from one node to another. Without any of these technologies, here's what actually happens:

```
GPU HBM → PCIe → CPU → CPU RAM → CPU → NIC → network → NIC → CPU → CPU RAM → CPU → PCIe → GPU HBM
          ^^^^                   ^^^^                              ^^^^                   ^^^^
       PCIe copy               CPU copy                         CPU copy              PCIe copy
```

That's two CPU interventions and four PCIe bus crossings on every network transfer. Each one burns microseconds and burns memory bandwidth that your GPU could be using for actual compute.

The story of RDMA, GPUDirect RDMA, and GDRCopy is the story of systematically eliminating each of those boxes from the diagram.

---

## Step One: RDMA — Removing the CPU from CPU RAM Transfers

The first problem to solve was simpler than the GPU problem: even for pure CPU workloads, having the CPU manually copy data to and from the NIC is wasteful. Standard TCP/IP networking looks like this:

```
Without RDMA:
  Source CPU reads CPU RAM → hands to NIC → remote NIC → remote CPU writes to CPU RAM
```

The CPU on both sides has to stay awake for the whole transfer. For a 100 MB message, that's the CPU doing nothing useful while bytes move.

RDMA (Remote Direct Memory Access) flips this around. The NIC is given direct access to CPU RAM via DMA, and the remote NIC writes directly into the target's CPU RAM without involving the remote CPU at all:

```
With RDMA:
  Source NIC reads CPU RAM directly → network → remote NIC writes CPU RAM directly
  (CPUs not involved — but data must still be in CPU RAM)
```

The latency improvement is dramatic: TCP/IP sits at 50–100 µs per message. RDMA drops that to 1–2 µs. That difference is what makes InfiniBand the standard fabric for HPC and large-scale AI training.

InfiniBand provides RDMA natively at the protocol level. RoCE (RDMA over Converged Ethernet) brings the same capability to Ethernet networks by encapsulating IB transport in UDP/IP. Both give you the same fundamental property: the CPU is removed from the data path.

But here's the catch: RDMA still requires your data to be in CPU RAM. For GPU workloads, you've just pushed the problem one level deeper.

---

## Step Two: GPUDirect RDMA — Also Removing CPU RAM from the Path

Even with RDMA in place, a GPU training job doing cross-node communication has this data path:

```
GPU HBM → PCIe → CPU RAM → NIC → network → NIC → CPU RAM → PCIe → GPU HBM
```

Two PCIe crossings. On H100 systems with PCIe Gen4, you're looking at a ~64 GB/s ceiling on each of those crossings — and that ceiling is shared across everything else talking over PCIe on that host. On B200 and GB200 systems with PCIe Gen5, the ceiling doubles to ~128 GB/s, but the fundamental problem remains: you're paying the PCIe toll twice, and CPU RAM is an unnecessary middleman.

GPUDirect RDMA solves this. It lets the NIC DMA directly to and from GPU HBM, completely bypassing CPU RAM:

```
GPU HBM → PCIe → NIC → network → NIC → PCIe → GPU HBM
```

The CPU and CPU RAM are both out of the picture. For a 1 GB transfer, you've gone from four PCIe crossings to two.

### What actually makes this work: PCIe Peer-to-Peer

GPUDirect RDMA rests on a capability called PCIe Peer-to-Peer (P2P). Normally, a PCIe device like a NIC can only DMA to system RAM. PCIe P2P allows one PCIe device — the NIC — to DMA directly to another PCIe device's memory — the GPU.

For this to work, the GPU has to expose its HBM through a BAR (Base Address Register): a window of PCIe-addressable space that other devices on the bus can target. The NIC uses that BAR address to reach GPU memory directly without routing through RAM.

There's an important topology requirement here: the NIC and GPU generally need to be on the same PCIe root complex, or at least connected through a PCIe bridge that supports P2P transactions. If your NIC is hanging off a different root complex than your GPU, PCIe P2P won't work, and GPUDirect RDMA will either silently fall back or fail. NCCL's topology detection handles this, which is one reason `NCCL_DEBUG=INFO` output is so valuable for debugging transfer performance.

### The kernel-level plumbing: nvidia_peermem vs DMA-BUF

For the IB driver to create Memory Regions (MRs) against GPU HBM — which is what allows a NIC to target that memory for RDMA operations — there needs to be a kernel-level mechanism that tells the IB stack "this physical address range belongs to GPU memory, and the GPU driver vouches for it."

Historically, this was handled by `nvidia_peermem`, a separate kernel module that sits between the NVIDIA driver and the IB/RDMA stack:

```bash
lsmod | grep nvidia_peermem   # legacy path — still works on H100, H200, B200, GB200
```

`nvidia_peermem` works across GPU generations and is still widely deployed. But it has a maintenance story — it's an out-of-tree module that has to track both kernel and driver versions.

The modern path is **DMA-BUF**, a standard Linux kernel interface originally designed for sharing buffers between display drivers and V4L2. NVIDIA added support for exporting GPU memory as DMA-BUF objects, which lets the IB stack register them as MRs through a fully in-kernel, vendor-neutral interface. This is the direction the ecosystem is moving:

```bash
# Check whether DMA-BUF path is active during NCCL initialization:
NCCL_DEBUG=INFO mpirun ... 2>&1 | grep "DMA-BUF"
# NCCL INFO DMA-BUF is available on GPU device 0   ← this is what you want to see
```

DMA-BUF requires a recent-enough kernel (5.15+ with appropriate patches, though the exact requirements vary by distro) and NVIDIA driver support. On systems running modern software stacks — H100, H200, B200, or GB200 clusters with current drivers — you'll typically see DMA-BUF active. On older installs, you'll still be on `nvidia_peermem`, which is fine functionally.

> **IB vs GPUDirect RDMA — they are not the same thing.**
> InfiniBand is the network fabric. GPUDirect RDMA is a *capability* that runs on top of IB (or RoCE). IB alone gives you RDMA to CPU RAM. IB + DMA-BUF (or `nvidia_peermem`) gives you GPUDirect RDMA — RDMA directly to GPU HBM. A cluster can have IB without GPUDirect RDMA if the kernel plumbing isn't set up. Always verify independently.

### A note on GB200 and the Grace-Blackwell fabric

On GB200 NVL72 systems specifically, there's an additional path worth knowing about. The Grace CPU and Blackwell GPU in GB200 are connected via NVLink-C2C, a coherent chip-to-chip interconnect. This means CPU memory and GPU memory are part of a unified, coherent address space — and certain CPU-to-GPU transfers can bypass PCIe entirely, happening over the NVLink-C2C fabric at much higher bandwidth than any PCIe generation can offer.

This doesn't replace GPUDirect RDMA for cross-node transfers — those still go NIC → network → NIC, with the PCIe Gen5 path between GPU and NIC. But it changes the calculus for intra-node operations and for GDRCopy (more on that below). The GB200's coherent fabric is a qualitative architectural difference from H100/H200/B200, not just a bandwidth upgrade.

---

## Step Three: GDRCopy — Letting the CPU Write Directly to GPU Memory

GDRCopy solves a different problem entirely. It's not about the network path at all. It's about what happens when the CPU itself needs to write a small piece of data into GPU memory — say, a 64-byte completion flag.

Without GDRCopy, the only way for the CPU to write to GPU HBM is through the CUDA driver:

```
Without GDRCopy: CPU wants to write 8 bytes to GPU → cudaMemcpy() → CUDA driver → ~5–10 µs
```

The CUDA driver path has real overhead — it's designed for correctness and generality across all copy sizes and synchronization scenarios, not for 8-byte flag writes that need to land in under a microsecond.

GDRCopy takes a different approach. It maps GPU BAR pages directly into the CPU's virtual address space, so the CPU can do a plain `memcpy()` to an address that physically resolves to GPU HBM:

```
With GDRCopy:    CPU wants to write 8 bytes to GPU → memcpy() to mapped address → ~1–2 µs
```

No CUDA API call. No kernel launch overhead. No driver round-trip. Just a CPU store instruction that PCIe delivers to GPU memory.

The kernel module that enables this is `gdrdrv`:

```bash
lsmod | grep gdrdrv
ls /dev/gdrdrv   # should exist if gdrdrv is loaded
```

GDRCopy works across GPU generations — H100, H200, B200, and GB200 all support it, since it relies on the same BAR-mapping mechanism as GPUDirect RDMA. On GB200, the coherent Grace-Blackwell fabric means the CPU already has a lower-latency path to GPU memory for some operations, but `gdrdrv` remains useful for explicit small-write scenarios where you want the PCIe-mapped path.

### Why NCCL uses gdrdrv

NCCL's LL (Low Latency) protocol — used for small messages where latency matters more than bandwidth — involves ring-based all-reduce where each GPU writes a flag into the next GPU's memory to signal that it's done with a step. Historically this was done with a CUDA kernel, but launching a kernel just to write a completion flag adds overhead.

With `gdrdrv` loaded, NCCL can have the CPU write those flags directly into GPU memory using mapped BAR addresses. This shaves microseconds off the latency of each ring step. For small all-reduces (the kind that dominate gradient synchronization for small batch sizes or small models), this matters.

For large all-reduces where you're bandwidth-bound and not latency-bound, `gdrdrv` is less relevant — you're spending most of your time in the IB transport, not in flag writes.

---

## The Full Picture, Side by Side

It helps to see all three mechanisms against each other:

| Mechanism | Who accesses GPU HBM | Purpose | Kernel plumbing |
|---|---|---|---|
| DMA-BUF / nvidia_peermem | IB NIC (via RDMA) | Cross-node data transfers | IB MR registration against GPU memory |
| gdrdrv (GDRCopy) | CPU (via mapped BAR) | Low-latency small CPU→GPU writes | BAR pages mmap'd into CPU VA space |
| NVLink / NVSwitch | Another GPU | Intra-node GPU-to-GPU transfers | NVLink hardware, no PCIe involved |
| CUDA (cudaMemcpy) | CPU-initiated DMA | General CPU↔GPU copies | CUDA driver, PCIe |

And here's the progression of what each technology eliminates:

```
Standard networking:    GPU HBM → PCIe → CPU → CPU RAM → CPU → NIC → network
                                  ^^^^          ^^^^
                              PCIe crossing    CPU copies

RDMA:                   CPU RAM → NIC → network → NIC → CPU RAM
                        (CPU removed, but data must live in CPU RAM)

GPUDirect RDMA:         GPU HBM → PCIe → NIC → network → NIC → PCIe → GPU HBM
                        (CPU + CPU RAM removed — only PCIe and network remain)
```

Each step eliminates one class of bottleneck. GPUDirect RDMA is as far as you can go on the network path — what's left is PCIe between the GPU and NIC, and the network itself.

---

## Verifying GPUDirect RDMA Is Actually Working

Knowing the theory is one thing. Here's how to confirm the plumbing is active on your nodes:

```bash
# Legacy kernel path (works on H100, H200, B200, GB200)
lsmod | grep nvidia_peermem

# Modern kernel path — look for this line in NCCL init output
NCCL_DEBUG=INFO mpirun ... 2>&1 | grep "DMA-BUF"
# NCCL INFO DMA-BUF is available on GPU device 0   ← you want to see this

# Validate end-to-end IB bandwidth — this confirms the NIC can reach GPU memory
# If GPUDirect RDMA is active, these numbers reflect GPU HBM bandwidth, not CPU RAM bandwidth
ib_write_bw -d mlx5_1 -p 18515 --report_gbits          # run on the server
ib_write_bw -d mlx5_1 -p 18515 <server-host>           # run on the client
# Expected on NDR 400 Gb/s fabric: >390 Gb/s

# Verify GDRCopy
lsmod | grep gdrdrv
ls /dev/gdrdrv   # should be present if gdrdrv is loaded
```

If `DMA-BUF` doesn't appear in NCCL output and `nvidia_peermem` isn't loaded, your cross-node transfers are bouncing through CPU RAM — and you'll see it in your all-reduce bandwidth numbers.

---

## What This Means in Practice

If you're running on H100s with PCIe Gen4, your effective ceiling for GPU-to-NIC data movement is around 64 GB/s per direction. That number is the hard limit for cross-node bandwidth, no matter how fast your network fabric is.

Move to B200 or GB200 with PCIe Gen5, and that ceiling roughly doubles to ~128 GB/s. The network can now be the bottleneck rather than the PCIe bus — which is the right problem to have.

On GB200 specifically, the coherent Grace-Blackwell fabric opens up additional paths for intra-node CPU↔GPU communication that don't touch PCIe at all. For workloads where the CPU is actively feeding data into the GPU (large-scale data preprocessing pipelines, online serving with CPU-side tokenization, etc.), this architectural difference can be significant.

But across all these generations, the fundamental hierarchy holds: GPUDirect RDMA removes CPU and CPU RAM from the cross-node path, and GDRCopy removes the CUDA driver overhead for small CPU-to-GPU writes. These are not the same thing, and you need both for a properly tuned multi-node training environment.
