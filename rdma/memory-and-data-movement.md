# Memory and Data Movement in GPU Infrastructure

If you want to understand why a distributed training job runs at 40% of theoretical peak instead of 90%, the answer almost always lives in one place: data movement. Not the model architecture, not the optimizer, not the learning rate schedule — where the data is, and what path it has to take to get somewhere useful.

Every GPU infrastructure decision you make — how you place tensors, which collective algorithm you choose, whether you enable GPUDirect RDMA, how you configure NCCL — is ultimately a decision about data movement. Getting this right is the difference between a cluster that earns its hardware cost and one that spends most of its time waiting.

This post walks through the full picture: the four memory pools that matter, the interconnects that connect them, how the hardware story has evolved from H100 to GB200, and the practical tools you use to verify that everything is actually routing the way you think it is.

---

## The Four Memory Pools

Before getting into bandwidth numbers, it helps to have a clear mental model of where data actually lives. There are four distinct memory pools in a GPU cluster, and data does not flow between them automatically — every crossing is an explicit hardware transfer with its own cost.

| Pool | Hardware | Capacity per generation | Who can access it |
|---|---|---|---|
| GPU HBM | HBM stack on-chip | H100: 80 GB · H200: 141 GB · B200/GB200: 192 GB | GPU compute engines only |
| CPU RAM | DRAM/LPDDR5 on host | Varies by server config; GB200 Grace: ~880 GB | CPU, OS, Python process |
| `/dev/shm` | tmpfs (carved out of CPU RAM) | Typically ~half of CPU RAM | CPU processes + NCCL SHM transport |
| Remote GPU HBM | HBM on a different node | Same per-GPU capacities as above | Via RDMA or NVLink fabric |

The consequence of this layout is that your dataloader, your model weights, your optimizer state, and the tensors being communicated between GPUs all potentially live in different pools — and moving them between pools is where training time gets eaten.

---

## The Interconnects: Four Very Different Speed Tiers

Think of GPU interconnects as a four-lane highway system where each lane has a wildly different speed limit. Understanding which traffic goes on which lane is the whole game.

### PCIe — The Necessary Bottleneck

PCIe is the bridge between your CPU and GPU. On current systems it tops out around 64 GB/s, which sounds reasonable until you compare it to GPU HBM bandwidth (~8 TB/s internally) or NVLink (900–1800 GB/s depending on generation). PCIe is roughly 15–100x slower than what the GPU can consume internally.

This isn't a problem you can engineer away entirely — your dataloader has to stage data from storage through CPU RAM to GPU HBM, and that path crosses PCIe. The goal is to make sure PCIe only sees that unavoidable traffic, and that your hot-path training collectives (all-reduce, all-gather, etc.) never touch it.

On H100, H200, and B200 servers, all CPU-to-GPU data movement goes over PCIe. This is standard and expected. Where it becomes a problem is if NCCL misconfigures and starts routing *intra-node* GPU-to-GPU traffic through PCIe instead of NVLink — more on that below.

### NVLink — The Fast Intra-Node Fabric

NVLink is what makes a multi-GPU node feel like a single large GPU rather than a collection of separate devices. It bypasses PCIe and CPU entirely, connecting GPUs through NVSwitch chips that act as a non-blocking crossbar: any GPU can talk to any other GPU at full bandwidth simultaneously.

The bandwidth has scaled significantly across generations:

- **H100 (NVLink 4.0):** 900 GB/s bidirectional per GPU with NVSwitch. Each DGX H100 has 8 GPUs and 4 NVSwitch chips. NVLink is strictly intra-node — cross-node traffic must use InfiniBand.
- **H200 (NVLink 4.0):** Same 900 GB/s as H100. H200 is a memory upgrade (HBM3e 141 GB vs HBM3 80 GB), not a connectivity upgrade. Same intra-node-only constraint.
- **B200 in DGX B200 (NVLink 5.0):** 1800 GB/s bidirectional per GPU — double H100/H200. DGX B200 has 8 GPUs per node. Still intra-node only; cross-node still requires InfiniBand.
- **GB200 in NVL72 rack (NVLink 5.0 + MNNVL):** Same 1800 GB/s per GPU, but MNNVL (multi-node NVLink) extends the fabric across nodes within the same rack. The NVSwitch fabric manager and IMEX daemon handle routing. This is what makes an NVL72 rack behave like a single 72-GPU system rather than 18 separate 4-GPU nodes.

```bash
nvidia-smi topo -m     # NV18 = 18 NVLink bonds between each GPU pair (GB200)
                       # NV18 on H100/H200 as well; NVLink 5.0 on B200/GB200 doubles per-bond speed
nvidia-smi nvlink -s   # per-link utilization and speed
```

### InfiniBand — Cross-Node RDMA

When you need to move tensors between nodes, InfiniBand with GPUDirect RDMA is how it's done in production. GPUDirect lets the NIC DMA data directly into and out of GPU HBM — no CPU involvement, no PCIe staging through system memory. The NIC is on its own PCIe lane, but the GPU HBM is the source/destination, bypassing the CPU bottleneck.

The IB configuration differs between H100 and GB200 systems:

- **H100 DGX:** 8 GPUs per node, 8 ConnectX-7 NICs at 400 Gb/s NDR each. Each NIC is dedicated to one GPU on its own PCIe bus — no shared contention. Total aggregate: ~400 GB/s per node.
- **GB200 NVL4 nodes:** 4 GPUs per node, 4 NICs at 400 Gb/s NDR each. Total aggregate: ~200 GB/s per node. Note that per-node IB bandwidth is lower on GB200 nodes than DGX H100, but GB200's MNNVL handles a lot of the cross-node communication within a rack, so IB is primarily for rack-to-rack or cluster-scale traffic.

```bash
ibdev2netdev           # shows each IB device and its PCIe bus address
                       # on H100 DGX you'll see 8 devices on 8 different buses
```

For H100 and H200 clusters, all cross-node traffic is InfiniBand. There is no cross-node NVLink. This is worth internalizing: when you run a multi-node all-reduce on an H100 cluster, every tensor that crosses a node boundary is going over IB at ~400 GB/s total per node, not over NVLink.

### MNNVL — Cross-Node NVLink (GB200 Only, Same Rack)

MNNVL deserves its own callout because it fundamentally changes the topology available on GB200. In a standard NVL72 rack, 18 GB200 nodes (72 GPUs total) are connected via NVLink fabric that the NVSwitch fabric manager treats as a single flat crossbar. Cross-node bandwidth within the rack approaches intra-node speeds.

This matters enormously for tensor parallelism and pipeline parallelism, where the communication pattern is dense and low-latency. On H100, you'd have to keep TP groups within a single node (8 GPUs) to get NVLink bandwidth. On GB200 NVL72, you can span TP across the entire rack with NVLink bandwidth.

When MNNVL is healthy, NCCL routes intra-rack communication over NVL. When the fabric has issues (link flap, NVSwitch problem, IMEX misconfiguration), NCCL falls back to IB or SHM — and you'll see a sharp drop in collective throughput that's the first sign something is wrong.

**Important:** MNNVL is exclusive to GB200. H100, H200, and B200 servers in separate chassis must use InfiniBand for all cross-node communication, full stop.

---

## GB200's Unique Memory Architecture: HBM as a NUMA Node

This section is specific to Grace-Blackwell and has no equivalent on H100, H200, or B200. It's worth understanding because it changes how you think about dataloader design and CPU-GPU data staging.

On H100, H200, and B200, if your CPU-side code (dataloader, preprocessing, Python process) touches data that needs to go to GPU, it crosses PCIe. There's no way around this — the GPU HBM is not addressable by the CPU without an explicit copy.

On GB200, the Grace CPU and Blackwell GPU are connected via a coherent NVLink-C2C fabric. This means GPU HBM appears as a NUMA node from the CPU's perspective. The OS memory allocator can place buffers directly in GPU HBM, and the CPU can read and write them without a PCIe copy.

```bash
numactl --hardware
# On a GB200 node you'll see something like:
# NUMA nodes 2, 3, 4, 9 → GPU HBM (GPU2, GPU0, GPU3, GPU1), ~184 GB each

numactl --membind=3 python dataloader.py
# allocates dataloader output buffers in GPU0's HBM (NUMA node 3)
# GPU0 can consume that data without any PCIe copy
# this is not possible on H100, H200, or B200 — they require explicit cudaMemcpy
```

The practical implication: on GB200 you can design dataloaders that write preprocessed tensors directly into GPU HBM from the CPU side, eliminating the PCIe staging step. This is especially valuable for high-throughput data pipelines where the dataloader is the bottleneck.

---

## Bandwidth Reference Across Generations

Here's the full picture of what each path actually costs, organized so you can compare across hardware generations:

| Path | Technology | H100 | H200 | B200 | GB200 |
|---|---|---|---|---|---|
| Same-node GPU↔GPU | NVLink + NVSwitch | 900 GB/s | 900 GB/s | 1800 GB/s | 1800 GB/s |
| Cross-node, same rack | MNNVL | N/A (IB only) | N/A (IB only) | N/A (IB only) | ~1800 GB/s |
| Cross-node (IB) | GPUDirect RDMA | ~400 GB/s/node | ~400 GB/s/node | ~400 GB/s/node | ~200 GB/s/node |
| Same-node fallback | SHM (PCIe-staged) | ~60 GB/s | ~60 GB/s | ~60 GB/s | ~60 GB/s |
| GPU↔CPU RAM | PCIe | ~64 GB/s | ~64 GB/s | ~64 GB/s | N/A (coherent fabric) |
| GPU↔CPU RAM | NVLink-C2C | N/A | N/A | N/A | ~900 GB/s |
| Cross-node fallback | TCP socket | ~10 GB/s | ~10 GB/s | ~10 GB/s | ~10 GB/s |
| GPU HBM bandwidth | Internal | ~3.35 TB/s | ~4.8 TB/s | ~8 TB/s | ~8 TB/s |
| GPU HBM capacity | per GPU | 80 GB | 141 GB | 192 GB | 192 GB |

The numbers that matter most in practice are the same-node NVLink bandwidth and the cross-node IB bandwidth per node. A 15x regression from accidentally routing intra-node traffic over IB instead of NVLink is not a theoretical risk — it's a real operational mistake that happens when NCCL isn't configured correctly or when GPUDirect is broken.

> **The most common expensive mistake:** Misconfiguring NCCL such that intra-node GPU pairs communicate over IB instead of NVLink. On H100 this is a ~4.5x regression (900 GB/s → 400 GB/s node-aggregate). On GB200 within a rack it's a ~9x regression (1800 GB/s → 200 GB/s). Always verify transport selection before trusting benchmark numbers.

---

## Verifying NCCL Transport Selection

Knowing the hardware is only half the battle. The other half is confirming that NCCL is actually using the right transport for each communication pattern. `NCCL_DEBUG=INFO` gives you exactly this visibility.

```bash
NCCL_DEBUG=INFO mpirun ... 2>&1 | grep "Ring.*via"
# What you want to see:
# P2P/NVL  → NVLink is being used for this GPU pair (correct for same-node)
# NET/IB   → InfiniBand for cross-node (correct)

# What you don't want to see:
# SHM      → CPU-staged via shared memory (P2P disabled or NVLink broken)
# NET/Socket → TCP fallback (IB not found or GPUDirect RDMA broken — severely degraded)
```

If you see SHM on intra-node pairs, the first things to check are whether NVLink is enabled (`nvidia-smi nvlink -s`), whether peer-to-peer access is enabled between GPU pairs (`nvidia-smi topo -p2p r`), and whether there's an IOMMU or virtualization layer blocking P2P.

If you see NET/Socket instead of NET/IB on cross-node traffic, GPUDirect RDMA is likely not working — check that the IB driver is loaded (`ibstat`), that GPUDirect is enabled in the driver (`cat /proc/driver/nvidia/params | grep EnableGpuDirect`), and that the IB devices are on the expected PCIe buses.

The general principle: always instrument before you optimize. A cluster that looks correctly configured can still have subtle routing issues that cost you 30–50% of throughput, and the only way to know is to look at what NCCL is actually doing, not what you think it should be doing.
