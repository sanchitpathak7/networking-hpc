# InfiniBand

If you've spent any time around GPU clusters, you've heard "InfiniBand" thrown around as the reason one cluster is faster than another. But what actually *is* it, why does it matter for AI training, and how does it compare to just using Ethernet? This post breaks it all down — including how the network topology differs between DGX H100 systems, newer B200 nodes, and the GB200 NVL72 rack architecture.

---

## Why Not Just Use Ethernet?

Here's the thing: Ethernet is everywhere, it's cheap, and your laptop talks to the internet over it just fine. So why would you spend significantly more on a completely different network technology for GPU clusters?

The answer comes down to what AI training actually does at the network layer.

During distributed training, every GPU needs to synchronize gradients with every other GPU — thousands of times per training step. The dominant primitive here is all-reduce: every node sends its local gradients, and everyone ends up with the global sum. If you're running 1,000 training steps per second, that's 1,000 synchronization rounds per second. The latency of each round directly limits how fast you can train.

Standard Ethernet was designed for general-purpose networking where millisecond latency is perfectly acceptable — web requests, file transfers, database queries. GPU training all-reduce needs *microsecond* latency. That's a 1,000x gap, and it's not just a minor inconvenience; it's the difference between your GPUs being busy doing useful work versus sitting idle waiting for network synchronization.

Here's the full picture side by side:

| | Ethernet | InfiniBand |
|---|---|---|
| Addressing | IP addresses | LIDs (16-bit Local IDs) |
| Routing | Each router decides per-hop | Subnet Manager computes all routes centrally |
| Congestion control | TCP backoff (drops packets, retransmits) | Credit-based flow control (no drops) |
| Latency | ~50–100 µs | ~1–2 µs |
| Loss | Packets can drop | Lossless by design |
| Management | ifconfig, ip, iptables | ibstat, ibdiagnet, UFM |

The **lossless** property deserves special attention. In a ring all-reduce, you've got a chain of GPUs where every node depends on the previous one. Drop a single packet anywhere in that ring and *every* GPU in the ring stalls, waiting for the retransmit. InfiniBand's credit-based flow control prevents drops entirely — a sender only transmits when the receiver has confirmed it has buffer space. No drops, no retransmits, no stalls.

This is also why NCCL (NVIDIA's collective communication library) is specifically tuned to love InfiniBand. NCCL uses `libibverbs` to talk directly to the IB kernel subsystem, bypassing TCP/IP entirely. There's no socket overhead, no kernel networking stack to traverse — just direct memory operations between GPUs.

---

## InfiniBand Generations: NDR, HDR, and What's Coming

Not all InfiniBand is equal. The technology has evolved through several generations, and you'll encounter different speeds depending on which cluster you're working with:

| Generation | Speed per Port | Status |
|---|---|---|
| HDR | 200 Gb/s | Older H100 clusters, still common |
| NDR | 400 Gb/s | Current standard (H100, H200, B200, GB200) |
| XDR | 800 Gb/s | Next generation, emerging |

Most current-generation systems — including DGX H100, H200, B200, and GB200 nodes — run NDR at 400 Gb/s per port. If you're benchmarking or troubleshooting a cluster, it's worth checking which generation you're on, since the expected bandwidth numbers differ significantly. Some older H100 deployments still run HDR at 200 Gb/s, so don't assume NDR just because the GPUs are H100s.

XDR at 800 Gb/s is coming and will matter a lot as model sizes continue to grow and inter-node communication becomes even more of a bottleneck.

---

## How Different GPU Platforms Wire Up InfiniBand

This is where it gets interesting, and where a lot of confusion comes from. The *number* of NICs per node and how they map to GPUs varies significantly across hardware generations.

### DGX H100 / H200 / B200: 8 NICs, One Per GPU

The DGX H100, H200, and B200 all follow the same network topology: 8 GPUs per node, 8 ConnectX-7 NICs at 400 Gb/s NDR, one NIC per GPU.

```
DGX H100/H200/B200 node:
  GPU 0  →  mlx5_0   400 Gb/s NDR IB
  GPU 1  →  mlx5_1   400 Gb/s NDR IB
  GPU 2  →  mlx5_2   400 Gb/s NDR IB
  GPU 3  →  mlx5_3   400 Gb/s NDR IB
  GPU 4  →  mlx5_4   400 Gb/s NDR IB
  GPU 5  →  mlx5_5   400 Gb/s NDR IB
  GPU 6  →  mlx5_6   400 Gb/s NDR IB
  GPU 7  →  mlx5_7   400 Gb/s NDR IB

Total IB bandwidth per node: 8 × 400 Gb/s = 400 GB/s bidirectional
All NICs: native InfiniBand, all show Active in ibstat
```

Every NIC is pure InfiniBand — there's no Ethernet NIC mixed in here. When you run `ibstat` on a DGX H100, you should see 8 ports all in Active state. If any are down, that's a real problem, not normal behavior.

Device names on DGX H100 systems use the pattern `ibp*s0` (e.g., `ibp12s0`, `ibp46s0`) rather than `mlx5_*` directly, though the mlx5 devices still appear in the RDMA subsystem.

### GB200 NVL4: 4 NICs, Mixed Ethernet + IB

The GB200 NVL4 node pairs 4 GPUs with 4 IB NICs, but it also has a dedicated Ethernet NIC for storage and management — and the device naming reflects a different physical layout:

```
GB200 NVL4 node:
  mlx5_0   200 GbE    enp50s0 (Up)    ← Ethernet (storage, management)
  mlx5_1   400 Gb/s NDR IB  ibp* (Down at IP layer — normal)
  mlx5_2   400 Gb/s NDR IB  ibp* (Down at IP layer — normal)
  mlx5_3   400 Gb/s NDR IB  ibp* (Down at IP layer — normal)
  mlx5_4   400 Gb/s NDR IB  ibp* (Down at IP layer — normal)

Total IB bandwidth per node: 4 × 400 Gb/s = 200 GB/s bidirectional
```

Lower per-node IB bandwidth than the H100 — 200 GB/s vs 400 GB/s — but this is intentional. In the GB200 architecture, GPUs within the same rack talk to each other over NVSwitch via Multi-Node NVLink (MNNVL), which is dramatically faster than InfiniBand. InfiniBand in the GB200 NVL4 topology is primarily for rack-to-rack communication.

### GB200 NVL72: InfiniBand Only for Rack-to-Rack

The GB200 NVL72 rack takes this to its logical extreme: 72 GPUs in a single rack, all connected by NVSwitch fabric. Within the rack, you have incredibly fast GPU-to-GPU communication without touching the network at all.

InfiniBand in the NVL72 context is almost entirely about communicating *between* racks. If your job fits within a single NVL72 rack, IB barely matters. If you're scaling across racks — which you will be for large frontier models — IB is still your inter-rack interconnect.

This is a fundamentally different utilization pattern compared to H100 clusters, where IB carries heavy inter-node traffic even within a job that spans a single rack.

---

## Your Cluster's Network Architecture (GB200 NVL4 Specific)

If you're on a GB200 NVL4 system, here's the full breakdown of what you're looking at:

```
TAN   mlx5_0    200 GbE    enp50s0 (Up)       storage, management
CIN   mlx5_1–4  400 Gb IB  ibp*   (Down*)     GPU compute — NCCL, MPI
                             *Down at IP layer only. IB fabric is Active.
NVLink NVSwitch  —           —                 intra-node GPU traffic
SMN   BMC/IPMI  1 GbE       —                 out-of-band mgmt (not in VM)
```

One thing that trips people up: your IB NICs (`mlx5_1` through `mlx5_4`) will show as "Down" when you run `ip link` or `ibdev2netdev`. This is completely normal and expected — they have no IP address configured and don't need one. NCCL uses `libibverbs` to talk directly to the IB kernel subsystem, bypassing TCP/IP entirely. "Down" at the IP layer has zero effect on NCCL or any RDMA application.

Contrast this with a DGX H100 where `mlx5_0` is also an IB device (not Ethernet), so the "all NICs should show Active" expectation applies differently. On GB200 NVL4, `mlx5_0` being Up (Ethernet) and `mlx5_1–4` being Down at IP layer (IB, no IP) is the correct healthy state.

---

## The Subnet Manager: The Brain of the Fabric

InfiniBand doesn't use distributed routing like Ethernet does. Instead, a single **Subnet Manager (SM)** is the authoritative controller for the entire fabric. It's worth understanding what it actually does, because it affects how the fabric behaves under failures.

When the SM starts (or when a new device comes online), it:

1. Discovers every NIC and switch in the fabric
2. Assigns a unique 16-bit LID (Local ID) to every port
3. Computes full routing tables — every path through the fabric
4. Programs those routing tables into every switch

Once routing is programmed into the switches, data flows without the SM's involvement. The SM is in the critical path for *setup*, not for *data movement*.

```bash
ibstat | grep "SM lid"    # all IB NICs should point to the same SM
                          # SM lid: 913 → fabric is up
                          # SM lid: 0   → fabric is down
```

Here's the practical implication: if the SM goes down after routing is programmed, existing connections continue working — the routes are already in the switches. But no *new* connections can form, and link failures can't be rerouted around. For production, run a standby SM. For your learning cluster, a single SM is fine but keep this behavior in mind when you're debugging connectivity issues.

---

## One NIC Per GPU: Why It Matters

Whether you're on an H100 node with 8 NICs or a GB200 NVL4 with 4, the same principle applies: one NIC per GPU. This is deliberate.

If all GPUs on a node shared a single NIC, they'd contend for bandwidth at the NIC level. With dedicated NICs, each GPU has an uncontended path to the fabric. On a DGX H100, that's 8 × 400 Gb/s = 400 GB/s total. On a GB200 NVL4, it's 4 × 400 Gb/s = 200 GB/s total.

You can verify this yourself with `ibdev2netdev` — each NIC should be on a different PCIe bus. On GB200 NVL4, you'll see different bus numbers (e.g., 167, 185, 212, 230 for `mlx5_1` through `mlx5_4`). Physically isolated paths, no shared bottleneck between GPUs.

NCCL's `NCCL_SOCKET_NTHREADS` and channel tuning parameters ultimately trace back to this topology — NCCL is trying to keep all those NIC-to-GPU paths saturated simultaneously.

---

## InfiniBand vs RoCE: The Infrastructure Choice

There's another option in the RDMA world: **RoCE** (RDMA over Converged Ethernet). RoCE gives you RDMA semantics — direct memory operations between hosts — over standard Ethernet hardware.

On a GB200 NVL4 node, `mlx5_0` (the 200 GbE port) is actually RoCE-capable hardware. Your `mlx5_1–4` are native InfiniBand. The comparison:

| | InfiniBand | RoCE |
|---|---|---|
| Lossless | Native, built-in | Requires PFC + ECN tuning |
| Latency | Lower | Slightly higher |
| Cost | Higher | Lower (reuses Ethernet infrastructure) |
| Tuning burden | Minimal | Significant |
| Failure modes | Well-understood | More complex (PFC pause storms) |

One important clarification: the IB vs RoCE tradeoff is about **network infrastructure**, not GPU generation. An H100 cluster can run either IB or RoCE depending on how the network was built. A B200 cluster can run either. The GPU doesn't determine this choice — your switch fabric and NIC configuration do.

That said, for AI training at scale where reliability matters, native InfiniBand is strongly preferred. RoCE is more common in cloud environments (AWS EFA, Azure InfiniBand variants, GCP) where Ethernet infrastructure already exists and the economics favor it. Getting RoCE to perform reliably at scale requires careful Priority Flow Control (PFC) and ECN tuning — getting it wrong leads to PFC pause storms that can be worse than packet drops.

---

## Diagnostics

Here are the commands you'll reach for when things aren't working — or when you want to verify a healthy cluster:

```bash
# Port state and fabric info
ibstat
ibstat | grep -E "State|Physical|Rate|LID|SM lid"

# Map IB devices to interface names
ibdev2netdev
# DGX H100 example:
# mlx5_0 → ibp12s0 (Active)    ← IB (all 8 are IB)
# mlx5_1 → ibp46s0 (Active)
# ...
#
# GB200 NVL4 example:
# mlx5_0 → enp50s0 (Up)        ← Ethernet
# mlx5_1 → ibp230s0 (Down)     ← IB (Down at IP layer only — normal)

# Validate data-plane bandwidth between nodes
# Run on server first:
ib_write_bw -d mlx5_1 -p 18515 --report_gbits
# Run on client:
ib_write_bw -d mlx5_1 -p 18515 <server-ib-hostname> --report_gbits
# Expected: >390 Gb/s on NDR 400 Gb/s links

# Full fabric diagnostic scan
ibdiagnet

# Discover all nodes and switches in the fabric
ibnetdiscover

# Query port performance counters
perfquery

# NOTE: ibping requires an explicit device flag
ibping -C mlx5_1 -P 1 <lid>    # correct — specify the IB device
ibping <lid>                    # wrong — defaults to mlx5_0 (Ethernet on GB200 NVL4), fails with "Invalid argument"
```

The `ib_write_bw` bandwidth test is your best friend for validating a new cluster or diagnosing a performance regression. If you're seeing less than ~390 Gb/s on an NDR link, something is wrong — misconfigured MTU, faulty cable, speed negotiation issue, or a switch port running at HDR instead of NDR.

---

## /dev/infiniband/ Device Files

Understanding what lives in `/dev/infiniband/` helps demystify how NCCL and other RDMA applications actually interact with the hardware:

```bash
ls /dev/infiniband/

uverbs0–4    # user-space verbs — data plane, one per NIC
             # what NCCL and RDMA apps use directly (libibverbs)

umad0–4      # user-space MAD — management plane, one per NIC
             # used by ibstat, ibping, ibdiagnet

rdma_cm      # RDMA Connection Manager — handles connection setup

issm1–4      # in-band Subnet Manager capability on this HCA

by-ibdev/    # symlinks organized by IB device name
by-path/     # symlinks organized by PCIe path
```

The key distinction: `uverbs*` is the data plane — this is what NCCL talks to when it's moving gradients. `umad*` is the management plane — this is what diagnostic tools like `ibstat` and `ibping` use. When NCCL says it found 4 IB devices, it found 4 `uverbs` entries. When `ibstat` says a port is Active, it's talking through `umad`.
