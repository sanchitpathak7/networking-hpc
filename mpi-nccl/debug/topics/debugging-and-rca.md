# Debugging GPU Cluster Issues — Methodology and RCA

If you've ever stared at a NCCL collective that's running at 4% of expected bandwidth and wondered where to even start, this post is for you. GPU cluster debugging is one of those skills that looks opaque from the outside but has a very learnable structure once you understand the layered nature of the stack.

The key insight is this: **symptoms surface at higher layers, but root causes live at lower ones.** A NCCL hang is almost never a NCCL bug. A slow collective is almost never a NCCL configuration problem. Start at the bottom and work up.

---

## The Debugging Stack

Here's the mental model you want to internalize before you touch a single log file:

```
Application        loss.backward(), checkpoint, OOM
NCCL               transport selection, collective hang, slow busbw
MPI / PMIx         bootstrap failure, rank → node mapping
CUDA driver        fabric UUID, UVM, device access
Fabric (IMEX)      channel auth, IMEX connectivity
NVSwitch FM        clique provisioning, UUID assignment
IB fabric          SM, LID assignment, link state
Hardware           XID errors, NVLink errors, temperature
```

When something breaks, the first question you should ask is: **at which layer did the first symptom appear?** Then trace down one layer at a time. Don't skip layers. The temptation to jump straight to Googling NCCL environment variables is real, but if the problem is a misconfigured fabric manager, no amount of `NCCL_*` tuning will help you.

---

## Postmortem: MNNVL UUID=0.0 on node-000

*This is a GB200-specific incident. The MNNVL fabric (Multi-Node NVLink) and IMEX daemon are architectural features of the GB200 NVL72 platform. H100, H200, and B200 systems do not have MNNVL, IMEX, or a fabric manager — skip ahead to the failure patterns section for how similar symptoms manifest on those platforms.*

This is a real incident. Here's exactly what we saw, what we checked, and what it turned out to be.

### What We Saw

You kick off an all-reduce benchmark across two GB200 nodes. Expected busbw is somewhere in the 150–200 GB/s range. What you get:

```
nccl-tests all_reduce_perf: busbw ~4.6 GB/s
Test hangs at 16G message size
```

4.6 GB/s. That's not slow — that's broken. And the hang at 16G is a separate, additional signal that something is very wrong. Let's trace it down.

### Layer 1 — Check NCCL First: What Transport Is Being Used?

Before you touch any hardware tooling, look at what NCCL itself is doing. Set `NCCL_DEBUG=INFO`, rerun, and grep the output:

```bash
grep "Ring.*via" allreduce_debug.txt

# node-001 GPUs: P2P/NVL   ← NVLink working
# node-000 GPUs: SHM        ← P2P disabled, CPU-staged

grep "P2P is disabled" allreduce_debug.txt
# NCCL WARN P2P is disabled between NVLINK connected GPUs 1 and 0.
```

This is already extremely informative. node-001 is using NVLink (fast, correct). node-000 is falling back to SHM — shared memory routed through CPU RAM. That's why it's slow, and that's why it hangs at large sizes (more on the hang mechanics below).

The question shifts from "why is NCCL slow" to "why is P2P disabled on node-000?" NCCL disables P2P when it can't verify that GPUs share a valid fabric domain. That means we need to check the fabric UUID.

### Layer 2 — CUDA / Fabric UUID: The Telltale Zero

```bash
# On node-000:
nvidia-smi -q | grep -A 5 "Fabric"
# Status: Insufficient Resources
# CliqueId: 0
# ClusterUUID: 00000000-0000-0000-0000-000000000000

# On node-001:
# Status: Success
# CliqueId: 32766
# ClusterUUID: 7a4af5e1-06e4-6349-c4fe-63d71b7d9aa1
```

There it is. UUID `0.0` on node-000. All zeros means the fabric manager never assigned this node a valid fabric identity. node-001 has a real UUID and a real clique ID. node-000 doesn't exist from the fabric's perspective.

The status `Insufficient Resources` is the fabric manager's way of saying: "I know you're asking to join a clique, but I have no record of you." But why? This is where it gets interesting.

### Layer 3 — IMEX: A Race Condition at Boot Time

On GB200 systems, IMEX (the NVIDIA inter-node fabric exchange daemon) is what registers each node's GPUs with the fabric manager. If IMEX has a problem at startup, the node never gets a fabric UUID.

```bash
nvidia-imex-ctl -N
# Node #0  * 172.27.49.42 *  READY  C
# Node #1  - 172.27.51.54   READY   C
```

At first glance, this looks fine — both nodes show `READY` and `C` (Connected). But don't be fooled by the current state. Always check the logs for what happened at startup:

```bash
grep -i "No matching" /var/log/nvidia-imex.log
# ERROR: No matching network interface found for any IP addresses in nodes_config.cfg
```

There it is. IMEX logged this error at boot. Here's what happened: the IMEX systemd service started before DHCP finished assigning an IP address to `enp50s0`. The interface existed, but had no IP yet. IMEX tried to bind to the address in its config file, found nothing, and initialized in a degraded state.

You can verify this by checking the config against the current interface state:

```bash
cat /etc/nvidia-imex/nodes_config.cfg
# 172.27.49.42
# 172.27.51.54

ip addr show enp50s0 | grep inet
# Now shows DHCP-assigned IP — but IMEX already failed its init
```

The IP is there now, but the damage was done at boot. Restarting IMEX clears the error:

```bash
sudo systemctl restart nvidia-imex
grep "No matching" /var/log/nvidia-imex.log  # error gone after restart
```

Problem solved, right? Not quite.

### Layer 4 — Fabric Manager: The Part You Can't Fix Yourself

After restarting IMEX, the daemon shows healthy. But check the fabric UUID again:

```bash
nvidia-smi -q | grep -A 5 "Fabric"
# Status: Insufficient Resources   ← still broken
```

Still broken. This is the most important lesson from this incident: **IMEX being healthy is necessary but not sufficient.**

Here's why. The NVSwitch Fabric Manager runs at the hypervisor or rack level — it's not something you can see or touch from inside the VM. It maintains a clique configuration: an explicit list of GPU GUIDs that belong to each NVLink fabric domain. When a node's GPUs try to register, the fabric manager checks their GUIDs against this list. If they're not there, registration fails with "Insufficient Resources" — every single time, regardless of IMEX state.

On node-000, the GPU GUIDs were simply never added to the fabric manager's clique config. This isn't a VM-level fix. You need to escalate to whoever manages the physical infrastructure.

### Root Cause and Resolution

The fabric manager was never configured to include node-000's GPU GUIDs in the clique. This is the actual root cause. To fix it, the operator needs to:

1. Pull the GPU Fabric GUIDs from node-000:
   ```bash
   nvidia-smi -q | grep "GPU Fabric GUID"
   # GPU Fabric GUID : 0x034c4931ff7d1c4f
   # GPU Fabric GUID : 0x955bdb0e73dbd4cf
   # GPU Fabric GUID : 0x2adb8e75720cd097
   # GPU Fabric GUID : 0x1f49aef20863a95e
   ```
2. Add those GUIDs to the fabric manager clique configuration
3. Restart the fabric manager so it reprograms the NVSwitch with the updated config

### Why 4.6 GB/s Specifically

This is worth understanding because the number isn't random — it's predictable from first principles.

In a ring all-reduce with 8 GPUs across 2 nodes, the ring is only as fast as its slowest link. node-001 is using NVLink (~956 GB/s ceiling). node-000 is using SHM (~60 GB/s ceiling). The ring locks to the slowest participant.

Effective algorithm bandwidth ≈ 2.6 GB/s. Bus bandwidth = algbw × 2(N-1)/N = 2.6 × 1.75 ≈ **4.6 GB/s**.

When you see a number this specific in a benchmark, it's a clue. Work backwards from it.

### Why the Hang at 16G

The SHM transport stages data through `/dev/shm`, which is a tmpfs filesystem backed by CPU RAM. On a typical Linux system, `/dev/shm` defaults to 50% of available RAM. With 4 GPUs on node-000 each trying to stage 16GB of data simultaneously:

```bash
df -h /dev/shm
# 4 GPUs × 16GB = 64GB → /dev/shm exhausted → kernel blocks writes
```

The test just... waits. It's not a deadlock in the traditional sense — it's the kernel blocking on a full tmpfs. If you need a temporary workaround while you wait for the operator fix:

```bash
# Allow the test to proceed with degraded SHM performance:
NCCL_IGNORE_DISABLED_P2P=1 mpirun ...

# If you need to handle larger message sizes, expand shm:
sudo mount -o remount,size=128G /dev/shm
```

Be clear that these are workarounds, not fixes. They don't restore NVLink — they just stop the test from hanging.

---

## General NCCL Debugging Workflow

Once you've internalized the layered approach, most debugging sessions follow the same pattern. Here's a repeatable workflow:

```bash
# Step 1: Run with full debug output — always start here
NCCL_DEBUG=INFO mpirun ... 2>&1 | tee nccl_debug.txt

# Step 2: Check fabric UUIDs on all nodes
grep -i "MNNVL\|fabric uuid" nccl_debug.txt

# Step 3: Check P2P status — is NVLink actually being used?
grep "P2P is disabled\|P2P.*WARN" nccl_debug.txt

# Step 4: Check transport selection — this tells you exactly what's happening
grep "Ring.*via" nccl_debug.txt
# Healthy: P2P/NVL for intra-node, NET/IB for cross-node
# Broken:  SHM for intra-node, NET/Socket for cross-node

# Step 5: Confirm IB transport is found and being used
grep "NET/IB\|mlx5" nccl_debug.txt

# Step 6: Find the first error, not the loudest one
grep "WARN\|error\|fail" nccl_debug.txt | head -30

# All-in-one triage grep:
grep -iE "MNNVL|fabric|uuid|P2P|Ring.*via|WARN|error" nccl_debug.txt | head -50
```

The most important discipline here is Step 6: find the **first** error. NCCL failures are often cascading — one node crashes and takes three others down with it, filling your logs with secondary noise. The root cause is buried in the first few lines of errors, not the loudest ones.

---

## Common Failure Patterns

### P2P Disabled — NCCL Falls Back to SHM

```
NCCL WARN P2P is disabled between NVLINK connected GPUs 1 and 0.
Ring 00 : 0[b0000] -> 1[b0000] via SHM
```

**On GB200 (MNNVL platform):** This means the MNNVL fabric is broken for this node. Run:

```bash
nvidia-smi -q | grep -A 5 "Fabric"
```

- `Status: Insufficient Resources` → fabric manager provisioning issue; GPU GUIDs not in clique config (operator fix, can't resolve from inside the VM)
- `Status: Uninitialized` → IMEX daemon not running or initialized in degraded state; check `/var/log/nvidia-imex.log` for boot-time errors
- `Status: Success` but UUID still 0.0 → CUDA driver not reading UUID correctly; rare, try driver reinstall

**On H100/H200/B200 (no MNNVL, no IMEX, no fabric manager):** These platforms don't have a fabric manager — NVSwitch is programmed directly by the driver at init time. If you see SHM fallback on these systems, the cause is almost always a **PCIe ACS (Access Control Services) issue** — a BIOS setting that blocks peer-to-peer DMA between GPUs.

Check this with:

```bash
# Look for ACS enabled on PCIe bridges — this blocks P2P
lspci -vvv | grep -i "ACS"

# Check nvidia-smi topology to confirm P2P is blocked
nvidia-smi topo -m
```

The fix on H100/H200/B200 is a BIOS setting (disable PCIe ACS) or a kernel boot parameter, not a fabric manager operation. If intra-node NVLink appears broken on these platforms, check XID errors and `nvidia-smi topo` output — NVSwitch issues here surface as XID errors at driver init, not as IMEX failures.

---

### Bootstrap Failure — Ranks Can't Find Each Other

```
NCCL WARN socketProgress: Connection closed by remote peer
NCCL failure: 'remote process exited or there was a network error'
```

This one looks scary but is almost always a secondary error — one rank died and the rest are failing because they can't communicate with it. Don't debug the NCCL error; find what killed that rank:

```bash
grep "Segmentation\|Killed\|Exception\|Error" nccl_debug.txt | head -10
```

---

### IB Not Found — NCCL Uses TCP Fallback

```
Ring 00 : 3[b0000] -> 4[b0000] via NET/Socket
```

`NET/Socket` instead of `NET/IB` means NCCL is routing cross-node traffic over Ethernet instead of InfiniBand. You'll be running at maybe 10% of expected bandwidth.

This applies across all GPU generations — H100, H200, B200, GB200. The IB debugging approach is the same regardless of what GPUs you're running.

```bash
# Check if NCCL found any IB interfaces at all
grep "IB\|mlx5" nccl_debug.txt | head -20

# If NCCL auto-detected the wrong NICs, force the right ones:
NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4

# Verify the IB NICs are actually up and have an SM lid assigned
ibstat | grep -E "State|SM lid"
```

If `ibstat` shows `State: Down` or SM lid of 0, your IB fabric has a problem that NCCL can't work around.

---

### UCX UAR Segfault — MPI Crashes Before NCCL Starts

```
mlx5dv_devx_alloc_uar(device=mlx5_1, flags=0x0) type=WC failed: Cannot allocate memory
Segmentation fault (signal 11)
```

This happens when UCX tries to allocate a User Access Region in the IB device memory but can't — typically because Write Combining memory mappings aren't available. This is a VM/cloud-specific issue and can affect any GPU generation (H100, H200, B200, GB200) when running in a virtualized environment. The hypervisor may not expose WC mappings properly.

The fix is straightforward — tell UCX to skip the IB path for its transport layer and use TCP instead:

```bash
mpirun -x UCX_TLS=tcp,self ...
```

Note that this only affects the MPI bootstrap/control plane. NCCL will still use IB for data transport.

---

### Hang at Large Message Sizes

The test runs fine at 1G, 2G, 4G, 8G — then stalls completely at 16G+. This is the `/dev/shm` exhaustion pattern described in the postmortem above. The diagnostic steps:

```bash
df -h /dev/shm                    # is shm actually full?
grep "Ring.*via" nccl_debug.txt   # confirm SHM transport is being used
NCCL_TIMEOUT=120 mpirun ...       # extend timeout — is it slow or truly hung?
dmesg | grep -i xid               # check for GPU hardware errors underneath
```

---

## Hardware Health Checks

Before you spend an hour debugging NCCL configuration, spend five minutes verifying your hardware is actually healthy. A surprising number of "NCCL problems" are hardware problems in disguise.

XID errors are your first stop. These are logged to the kernel ring buffer and tell you directly when the GPU driver detected something wrong:

```bash
dmesg | grep -i xid
```

The XID codes that matter most, across all GPU generations (H100, H200, B200, GB200):

| XID | Meaning | Severity |
|-----|---------|----------|
| 45  | Pre-emptive channel termination | Normal under heavy load; investigate if frequent |
| 79  | GPU has fallen off the bus | Serious hardware fault — escalate immediately |
| 92  | High single-bit ECC errors | Memory degradation — plan for replacement |
| 94  | Contained ECC error | Field-recoverable; monitor closely |

XID 79 is the one that should make your heart rate go up. If you see it, that GPU is effectively unavailable and you need to get the hardware team involved.

Beyond XID errors, the standard health check battery:

```bash
# GPU health via DCGM — quick check first, deep diagnostic second
dcgmi health -g 0 -c
dcgmi diag -r 1

# Temperature and power — thermal throttling will hurt bandwidth
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv

# NVLink error counters — even one non-zero counter is worth investigating
nvidia-smi nvlink -e -i 0

# IB port error counters
perfquery -x 0 mlx5_1 1
```

Don't skip the NVLink counters. A single flapping NVLink lane won't necessarily show up as a NCCL error immediately, but it will hurt bandwidth in ways that are hard to trace if you don't check here first.

---

## Systematic Isolation Strategy

When a multi-node test fails, the fastest path to root cause is methodical isolation. Don't try to debug everything at once.

**Step 1: Test each node in isolation first.** This is the most important step and the one people most often skip. Single-node tests isolate intra-node NVLink from everything else:

```bash
ssh node-000 "mpirun -np 4 -npernode 4 ./all_reduce_perf -b 1G -e 8G -f 2 -g 1"
ssh node-001 "mpirun -np 4 -npernode 4 ./all_reduce_perf -b 1G -e 8G -f 2 -g 1"
```

Compare the busbw numbers. If node-000 is 3x slower than node-001 on single-node, you found your culprit before touching any multi-node tooling.

**Step 2: Test IB separately from NCCL.** Before blaming NCCL for cross-node performance, validate the network directly. This is hardware-agnostic and applies to all GPU generations:

```bash
# On node-001 (server):
ib_write_bw -d mlx5_1 -p 18515 --report_gbits

# On node-000 (client):
ib_write_bw -d mlx5_1 -p 18515 node-001 --report_gbits
# Expected on NDR 400Gb/s: >390 Gb/s
```

If `ib_write_bw` shows the right numbers, your IB fabric is healthy. If NCCL is still slow after that, the problem is in NCCL's transport selection — almost certainly `NCCL_IB_HCA` not being set or a fabric UUID issue.

**Step 3: Only run two-node NCCL after single-node and IB both pass.** If both individual tests look good but the combined test is slow, the issue is specifically in the interaction between nodes at the NCCL layer — most likely MNNVL/fabric UUID (on GB200), IB HCA selection, or a NCCL topology detection failure.

This order matters because each test narrows the search space in a way that the next test can exploit. Skipping steps doesn't save time — it usually costs you an hour of chasing the wrong thing.
