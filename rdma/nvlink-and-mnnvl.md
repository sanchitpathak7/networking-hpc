# NVLink, NVSwitch, and MNNVL

If you've ever looked at a slow NCCL all-reduce and wondered why your GPUs aren't talking to each other as fast as they should be, the answer almost always lives in the interconnect layer. NVLink is what makes intra-node GPU communication absurdly fast. MNNVL extends that story across nodes — but only on certain hardware, and only when a chain of services are all working correctly.

This is a full walkthrough of how NVLink has evolved across H100, H200, B200, and GB200, why MNNVL is a GB200-exclusive feature, and what breaks when any of this goes wrong.

---

## How NVLink Has Evolved Across GPU Generations

It helps to understand what you're working with before debugging it. NVLink is not a static technology — it has changed meaningfully across the H100, B200, and GB200 generations.

### H100 and H200: NVLink 4.0

On H100 and H200, each GPU has 18 NVLink lanes, and they're connected through NVSwitch 3.0 inside the DGX H100 node. The result is 900 GB/s bidirectional bandwidth per GPU (450 GB/s per direction). That's fast enough to make PCIe look like a bicycle lane next to a highway.

The 8 GPUs in a DGX H100 are connected all-to-all through the NVSwitch fabric — every GPU can talk to every other GPU at full bandwidth simultaneously, without any contention.

**Cross-node on H100/H200:** There is no NVLink between separate DGX H100 servers. Zero. When your H100 job spans multiple nodes, every byte of cross-node collective traffic goes through InfiniBand. There's no Multi-Node NVLink here, no fabric manager, no fabric UUID concept. The NVSwitch programming happens at driver initialization time, not through a separate fabric manager service.

```bash
nvidia-smi topo -m
# NV18 between every intra-node GPU pair
# SYS or PHB between GPUs on different nodes = pure IB path
```

### B200 in DGX B200: NVLink 5.0

B200 doubles the bandwidth. Each GPU still has 18 NVLink lanes, but NVLink 5.0 with NVSwitch 4.0 pushes to 1800 GB/s bidirectional per GPU (900 GB/s per direction) — double the H100.

Same topology as before: 8 GPUs per DGX B200, all-to-all NVSwitch fabric intra-node. And the same constraint: **cross-node on B200 still uses InfiniBand exclusively**. If you're running a multi-node B200 job, your collective communication between nodes goes through IB. There's no NVLink extending outside the chassis. No fabric manager. No fabric UUID.

This is a point of confusion worth stating directly: B200 has NVLink 5.0 and is significantly faster than H100 within a node, but it shares the same cross-node architecture as H100 — scale out via InfiniBand.

### GB200 NVL72: NVLink 5.0 + MNNVL

GB200 is where the architecture changes fundamentally. The NVL72 form factor fits 36 Grace-Blackwell superchips (72 GPUs) in a single rack. The "NVL72" name is telling you something important: all 72 GPUs are connected via NVLink fabric, even though they span multiple physical nodes.

Each GPU has 18 NVLink lanes at NVLink 5.0 speeds — same per-GPU bandwidth as B200. But the NVSwitch chips don't just live inside a single chassis. They extend across the rack, connecting nodes together through the same NVLink fabric. This is Multi-Node NVLink (MNNVL).

For GB200 NVL72, InfiniBand is only needed for rack-to-rack communication. Within a rack, NVLink handles everything. That's the architectural leap: you're not scaling up with IB anymore until you cross rack boundaries.

**Summary of what you're working with:**

| Platform | NVLink Version | Intra-Node BW (per GPU) | Cross-Node Fabric | Fabric Manager? |
|---|---|---|---|---|
| H100 / H200 (DGX) | NVLink 4.0 | 900 GB/s bidir | InfiniBand only | No |
| B200 (DGX B200) | NVLink 5.0 | 1800 GB/s bidir | InfiniBand only | No |
| GB200 NVL72 | NVLink 5.0 | 1800 GB/s bidir | NVLink (intra-rack) + IB (rack-to-rack) | Yes |

---

## The Fabric UUID — GB200-Specific

Everything that follows is GB200-specific. If you're running H100 or B200, you won't encounter fabric UUIDs, IMEX, or Insufficient Resources errors. Your NVSwitch is programmed at driver init time, and cross-node is just InfiniBand.

For GB200, the OS tracks which GPUs belong to the same NVLink clique via a **fabric UUID**. NCCL reads this UUID during initialization to decide which transport to use for each GPU pair.

```
NCCL INFO MNNVL busId 0xb0000 fabric UUID 7a4af5e106e46349.c4fe63d71b7d9aa1 cliqueId 0x7ffe
                                           ──────────────────────────────────
                                           non-zero = fabric manager provisioned this GPU ✓

NCCL INFO MNNVL busId 0xb0000 fabric UUID 0.0  cliqueId 0x0
                                           ───
                                           zero = GPU not in any NVLink clique ✗
```

UUID `0.0` is the signal that something went wrong in the provisioning chain. NVLink physically exists — `nvidia-smi topo` will still show NV18 between GPUs — but the fabric isn't initialized, so NCCL disables P2P between those GPUs even though the wires are there.

This distinction matters: a non-zero topo entry doesn't mean the fabric is working. The UUID check is what NCCL actually uses.

---

## Who Assigns the Fabric UUID (GB200)

Three components form a chain. If any one of them breaks, the GPUs end up with UUID 0.0.

**NVSwitch Fabric Manager** runs at the hypervisor or rack level, not inside your VMs. It programs the NVSwitch routing tables and assigns clique IDs based on which GPU Fabric GUIDs have been registered. This is the authoritative source. If the fabric manager hasn't provisioned a node's GPU GUIDs, those GPUs get UUID 0.0 regardless of anything else you do inside the VM.

**IMEX (Inter-Multinode Exchange)** handles the inter-node authentication and channel management. It manages secure channels between VMs on different nodes and is necessary for cross-node NVLink to work. But IMEX being connected doesn't mean the fabric manager has provisioned your GPUs — these are separate concerns. A healthy IMEX is required but not sufficient.

**CUDA driver** reads the fabric UUID from the IMEX channel and exposes it to applications. NCCL and `nvidia-smi` both read it from here.

```
Fabric Manager provisions GUIDs
  → NVSwitch routing tables programmed
  → IMEX distributes UUID across nodes
  → CUDA driver reads UUID
  → NCCL selects NVLink transport
```

Break any link in that chain and you get UUID 0.0 at the end.

---

## Checking Fabric State (GB200)

```bash
# Per-GPU fabric state — this is your first check
nvidia-smi -q | grep -A 5 "Fabric"

# Healthy output:
# State:       Completed
# Status:      Success
# CliqueId:    32766          ← non-zero, GPU is in a clique
# ClusterUUID: 7a4af5e1-...  ← non-zero, fabric UUID is assigned

# Broken output (what you might see on a misconfigured node):
# State:       Completed
# Status:      Insufficient Resources   ← fabric manager rejected this node
# CliqueId:    0
# ClusterUUID: 00000000-0000-0000-0000-000000000000

# IMEX connectivity — check that all nodes show Connected
nvidia-imex-ctl -N
# Node #0  * 172.27.49.42 *  READY  C   ← C = Connected
# Node #1  - 172.27.51.54   READY   C

# IMEX logs — look for errors about resources, clique IDs, or UUIDs
grep -iE "error|clique|uuid|resource" /var/log/nvidia-imex.log | tail -20

# GPU Fabric GUIDs — you'll need these if escalating to the platform operator
nvidia-smi -q | grep "GPU Fabric GUID"
```

---

## Common Failure: Insufficient Resources (GB200)

`Status: Insufficient Resources` means the fabric manager received the GPU's registration request and explicitly rejected it. The GPU GUIDs are not in the fabric manager's clique configuration. This is one of the more frustrating failures because the hardware looks fine from the inside — topo shows NV18, IMEX shows Connected — but the fabric isn't functional.

**You cannot fix this from inside the VM.** The platform operator needs to:
1. Add the node's GPU Fabric GUIDs to the fabric manager's clique configuration
2. Restart the fabric manager so it reprograms the NVSwitch routing tables

Get the GUIDs from the affected node and hand them to the operator:
```bash
nvidia-smi -q | grep "GPU Fabric GUID"
# GPU Fabric GUID : 0x034c4931ff7d1c4f
# GPU Fabric GUID : 0x955bdb0e73dbd4cf
# GPU Fabric GUID : 0x2adb8e75720cd097
# GPU Fabric GUID : 0x1f49aef20863a95e
```

---

## Common Failure: IMEX Can't Find Network Interface (GB200)

```
ERROR: No matching network interface found for any IP addresses in nodes_config.cfg
```

IMEX starts during boot and reads its peer IP list from `/etc/nvidia-imex/nodes_config.cfg`. If IMEX starts before DHCP finishes assigning the management IP, the network interface exists but has no address — IMEX can't bind, and GPU fabric registration fails silently downstream.

```bash
cat /etc/nvidia-imex/nodes_config.cfg     # should contain both node IPs
ip addr show enp50s0                      # confirm the IP is actually assigned
sudo systemctl restart nvidia-imex        # restart after confirming IP is up
grep -i "No matching" /var/log/nvidia-imex.log  # confirm the error is gone
```

> [!WARNING]
> Fixing the IMEX startup issue is necessary but not sufficient. If `Status: Insufficient Resources` persists after IMEX comes up healthy, you still have the fabric manager provisioning problem described above — those are two separate issues that can co-exist.

---

## What Happens to NCCL When MNNVL Is Broken (GB200)

When one node in a GB200 NVL72 job has UUID 0.0 and another has a valid UUID, NCCL falls back to different transports per node:

```
node-001: P2P/NVL  — NVLink direct through NVSwitch fabric (~956 GB/s per direction, ~1.8 TB/s bidirectional)
node-000: SHM      — data staged through CPU RAM via /dev/shm (~60 GB/s)
```

Ring all-reduce locks to the slowest link. node-001 is roughly 15x faster than node-000, but it must wait at every ring step. The result is overall bus bandwidth in the 4–5 GB/s range instead of the 150–200 GB/s you'd expect on a healthy fabric.

At large message sizes (16 GB and above), it gets worse: node-000's SHM transport stages data through `/dev/shm` (tmpfs). With 4 GPUs each staging 16+ GB simultaneously, `/dev/shm` exhausts and the test hangs rather than completing slowly.

If you want to confirm the fallback is happening, look for this in NCCL output:
```
NCCL INFO Channel 00/32 : 0[b0000] -> 1[b8000] via P2P/NVL    ← working
NCCL INFO Channel 00/32 : 0[b0000] -> 1[b8000] via SHM        ← fallback
```

**Workaround (not a fix):**
```bash
export NCCL_IGNORE_DISABLED_P2P=1   # lets NCCL proceed with SHM fallback
```

Setting this gets your job running but doesn't restore NVLink. Use it to unblock testing while you work the actual problem. For production workloads, you need the fabric provisioned correctly.

On H100 or B200, if intra-node NVLink is broken you'll see a similar SHM fallback for intra-node traffic, and all cross-node traffic was already going through IB. The fallback behavior looks similar but the cause and fix are completely different — there's no fabric manager or UUID chain to debug.
