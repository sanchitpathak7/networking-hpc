# MPI and Process Launch: What's Actually Happening When You Start a Multi-GPU Job

Here's a misconception that trips up almost everyone getting started with distributed GPU training: MPI is a communication library, so it must be doing the GPU communication, right? Not even close. In a NCCL-based training job, MPI's only job is to launch your processes and help them find each other. The moment NCCL takes over, MPI is basically done. All the interesting GPU data movement — the all-reduces, all-gathers, ring collectives — that's entirely NCCL's domain.

Understanding this distinction matters because it changes how you debug. If your collective is slow, don't look at MPI. If your job won't start, don't look at NCCL. The two stacks are largely independent, and confusing them wastes a lot of time.

---

## What MPI Actually Does in a NCCL Job

When you run `mpirun`, here's the full extent of what happens:

1. MPI reads your hostfile to figure out which nodes to use
2. It SSH's into each node and launches your binary
3. It sets environment variables that your ranks need — global rank, world size, local rank
4. It provides a bootstrap key-value store so NCCL ranks can exchange their addresses during `ncclCommInitRank`

That's the entire MPI story. Once NCCL has initialized its communicator and all ranks have found each other, MPI is essentially idle. No GPU data ever touches the MPI stack.

This bootstrap KV store is actually the most important thing MPI contributes. Before any collective can run, every rank needs to know the IP address and port of every other rank's NCCL listener. Someone has to coordinate that exchange. MPI provides that coordination layer. Once it's done, NCCL runs the show.

---

## The mpirun Command, Explained Flag by Flag

Here's the command you'd run for a 2-node GB200 cluster, where each node has 4 GPUs:

```bash
mpirun --allow-run-as-root \
  --hostfile /etc/hostfile \
  -np 8 -npernode 4 \
  --map-by ppr:4:node --bind-to none \
  -x UCX_TLS=tcp,self \
  -x NCCL_DEBUG=INFO \
  --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --mca coll ^cuda \
  /opt/nccl-tests/build/all_reduce_perf -b 1G -e 8G -f 2 -g 1
```

Let's go through each flag, because each one exists for a specific reason — most of them exist because something broke without it.

---

### `-np 8 -npernode 4`

This tells MPI to launch 8 total ranks, 4 per node. For a 2-node GB200 cluster (4 GPUs per node), this is the right shape: 2 nodes × 4 GPUs = 8 ranks total.

**This is hardware-specific.** On a DGX H100, H200, or B200, you have 8 GPUs per node. For 2 nodes of those, you'd use `-np 16 -npernode 8`. Getting this wrong means some GPUs sit idle or you try to assign two ranks to the same GPU, neither of which will give you useful benchmark results.

---

### `--map-by ppr:4:node --bind-to none`

`ppr:4:node` means "4 Processes Per Resource per node" — it's telling OpenMPI's process mapper how to distribute ranks across the system topology. Combined with `--bind-to none`, this says: lay out 4 ranks per node, but don't pin them to specific CPU cores.

Why does the CPU binding matter? CUDA handles GPU assignment based on rank number, not CPU affinity. If you accidentally bind MPI ranks to CPU cores that are NUMA-distant from the GPUs they end up using, you introduce unnecessary PCIe hop latency for host-side operations. Pinning is tempting but often causes more problems than it solves unless you're doing explicit NUMA-aware placement.

---

### `-x UCX_TLS=tcp,self`

The `-x` flag propagates an environment variable to all ranks. This particular one is forcing UCX (more on UCX below) to use TCP for its internal control messages instead of InfiniBand.

**Why does this exist? Because on GB200 VMs (and cloud/virtualized environments generally), it prevents a segfault.** Without this flag, UCX sees your InfiniBand NICs and tries to use them via `rc_x` (IB Reliable Connected transport). That requires allocating UAR (User Access Region) memory with WC (Write Combining) semantics. In a VM, the VF BAR space is limited, and that allocation fails — hard:

```
mlx5dv_devx_alloc_uar(device=mlx5_1, flags=0x0) type=WC failed: Cannot allocate memory
Segmentation fault (signal 11) in libucs.so
```

`tcp,self` tells UCX: use TCP for cross-node control messages, use shared memory for same-node loopback. No IB UAR allocation, no segfault.

**The important caveat:** this workaround is only necessary in virtualized environments. On bare metal DGX H100, H200, or B200 systems, VF BAR space is sufficient and UCX can happily use IB for its control path. On bare metal, forcing TCP is actually a minor performance regression for the MPI control plane (though since MPI isn't on the data path, it rarely matters). The flag is safe to include everywhere, but know that it's doing real work on VMs and is essentially a no-op guard on bare metal.

---

### `--mca coll ^cuda`

This one is subtle but important. OpenMPI has a CUDA-aware collective plugin that can perform reductions directly on GPU memory. Sounds useful, right? The problem is that when NCCL is also trying to run collectives, these two systems conflict. OpenMPI's CUDA collective plugin attempts to intercept `MPI_Allreduce` calls over GPU buffers and run its own reduction — which steps on NCCL's feet.

The `^` prefix means "exclude this component." So `--mca coll ^cuda` says: disable the CUDA collective module, let NCCL handle all GPU collectives without interference.

This flag is relevant regardless of GPU generation — GB200, H100, H200, B200, it doesn't matter. Any time you're using NCCL for your collectives (which is essentially always in modern training workloads), you want OpenMPI's CUDA collectives out of the picture. The symptoms of forgetting this flag can be subtle: wrong answers from collectives, unexpected hangs, or performance that looks right but is mysteriously worse than expected.

---

### `--mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"`

MPI launches remote processes via SSH. The `StrictHostKeyChecking=no` prevents SSH from refusing to connect when it hasn't seen a host before. In a fresh cluster setup or after nodes are reprovisioned, this is almost always necessary — otherwise your first `mpirun` fails because the compute nodes aren't in `~/.ssh/known_hosts` yet.

`UserKnownHostsFile=/dev/null` discards any host key it learns during this session, so you don't accumulate stale entries across node reprovisioning. In a production cluster with stable node identities this is less important, but during initial setup it saves a lot of friction.

---

## The Hostfile

```
# /etc/hostfile
gb200-compute-node-001 slots=4
gb200-compute-node-000 slots=4
```

The `slots=N` value tells MPI how many process slots to allocate on this node — it should match your GPU count. MPI assigns ranks in order: ranks 0–3 go to node-001, ranks 4–7 go to node-000.

The ordering isn't arbitrary. It matters for debugging. If you see NCCL `WARN P2P disabled` or a fabric error appearing on ranks 4–7, you immediately know the issue is on node-000. If it's ranks 0–3, it's node-001. That mapping from rank number to physical node is one of the first things you should internalize when diagnosing multi-node failures.

---

## The PMIx Problem: Why You Should Use mpirun Even Under Slurm

This is one of those things that's painful to discover through experience rather than documentation. When you run jobs under Slurm, the natural instinct is to use `srun --mpi=pmix` for everything. For single-node or non-NCCL jobs, this often works fine. For multi-node NCCL jobs, it can silently fail in a maddening way: the job appears to launch, but it just hangs at NCCL init and never prints any error.

Here's what's happening under the hood. Before NCCL can start doing collectives, all ranks need to exchange their network addresses. This exchange happens through a process management interface (PMI or PMIx). With `mpirun`, MPI acts as both the launcher and the PMIx server — rank 0 publishes its address to mpirun's PMIx daemon, every other rank looks it up, and it works reliably because there's one source of truth.

With `srun --mpi=pmix`, Slurm runs a PMIx daemon on each node. Cross-node KV exchange has to traverse the Slurm daemon network. If those daemons don't correctly forward KV entries across node boundaries — and this depends heavily on your specific Slurm and PMIx version combination — ranks on different nodes can't find each other. NCCL init hangs. No error, no timeout, just silence.

**This isn't a GB200-specific or H100-specific issue. It's an OpenMPI + Slurm integration problem that exists across all GPU generations.** The fix is straightforward: use `mpirun` inside your `sbatch` script instead of `srun`.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

mpirun --allow-run-as-root \
  --hostfile $SLURM_JOB_NODELIST \
  -np 8 -npernode 4 \
  --map-by ppr:4:node --bind-to none \
  -x UCX_TLS=tcp,self \
  -x NCCL_DEBUG=INFO \
  --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no" \
  --mca coll ^cuda \
  /opt/nccl-tests/build/all_reduce_perf -b 1G -e 8G -f 2 -g 1 \
  2>&1 | tee allreduce_${SLURM_JOB_ID}.txt
```

Slurm still manages resource allocation (nodes, GPUs, scheduling). But you hand the actual process launch and PMIx coordination back to `mpirun`, which handles it reliably. The key insight is that these two concerns — cluster scheduling and process launch — don't have to be done by the same tool.

**Avoid `srun --mpi=pmix` for multi-node NCCL jobs.** The silent hang at NCCL init is notoriously hard to distinguish from a real NCCL fabric problem, and you'll waste hours debugging the wrong layer.

---

## UCX: The Hidden Transport Layer

UCX (Unified Communication X) is the transport abstraction that OpenMPI uses for its own control-plane messages. It's not the GPU data path — NCCL handles that — but it is responsible for the MPI-level coordination messages your job needs to start.

UCX auto-detects available transports and picks the best one. On a node with InfiniBand, it naturally prefers `rc_x` (IB Reliable Connected). On a node with only TCP, it uses that. The transport selection happens at startup based on what hardware UCX can probe.

The segfault scenario described above happens specifically because UCX's IB transport probe tries to allocate UAR memory, and that allocation fails in VMs before UCX has a chance to fall back gracefully. `UCX_TLS=tcp,self` sidesteps the probe entirely.

If you want to understand what UCX is doing on a given system:

```bash
# See what transports UCX would select:
UCX_LOG_LEVEL=info mpirun ... 2>&1 | grep "UCX  INFO"

# Confirm which transport is active:
UCX_LOG_LEVEL=diag mpirun ... 2>&1 | grep "tcp\|rc_x\|mlx5"
```

On a healthy bare metal system, you'd see UCX selecting `rc_x` or `dc_x` for cross-node traffic. On a VM with the override, you'd see `tcp`. Either is fine for MPI's control messages — the control plane latency difference is irrelevant since NCCL owns all the hot paths.

---

## How Ranks Map to GPUs

Each MPI rank corresponds to one GPU. CUDA figures out which GPU to use based on environment variables your training code reads. OpenMPI sets `OMPI_COMM_WORLD_LOCAL_RANK` — the rank's index within its node (0 through N-1). Training frameworks use this to call `torch.cuda.set_device()`:

```python
local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

You're working with two rank spaces simultaneously:

- **Global rank**: the rank's index across the entire job (0 through world_size-1). NCCL uses global ranks to construct collective rings and determine communication order.
- **Local rank**: the rank's index on its specific node (0 through GPUs_per_node-1). Used for GPU assignment and to determine whether two ranks are on the same physical node (intra-node = NVLink, cross-node = IB).

When you're reading NCCL debug logs and trying to correlate a warning to a physical device, local rank tells you which GPU on a node is involved. Global rank tells you where that rank sits in the collective topology.

---

## Passing Environment Variables Through mpirun

Every variable you want available on all ranks needs to be explicitly forwarded with `-x`:

```bash
mpirun \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
  -x NCCL_SOCKET_IFNAME=^mlx5_0 \
  -x UCX_TLS=tcp,self \
  -x LD_LIBRARY_PATH \
  ...
```

Variables specified as `-x VAR=value` set the value explicitly on all ranks. Variables specified as `-x VAR` (no `=`) propagate whatever value `VAR` has in your current shell. This distinction matters when you want the same value everywhere (use `=`) versus when you want each node to use its own locally-set value (omit `=`).

`LD_LIBRARY_PATH` is typically passed without `=` so that if nodes have slightly different library paths (different local install locations, for example), each node uses its own correctly.

---

## Diagnosing Launch Failures

When a multi-node job won't start, work through the layers in order. Start with the simplest possible thing that can fail, not with the most complex.

```bash
# Layer 1: Can you even SSH to the compute nodes?
ssh gb200-compute-node-000 hostname

# Layer 2: Can MPI launch processes at all? (no NCCL involved)
mpirun --allow-run-as-root \
  --hostfile /etc/hostfile -np 8 -npernode 4 \
  --map-by ppr:4:node --bind-to none \
  -x UCX_TLS=tcp,self \
  --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no" \
  hostname

# Expected: 8 lines, 4 hostnames from each node
# If you see fewer than 8, MPI can't reach all slots

# Layer 3: Confirm rank-to-node mapping
mpirun ... \
  bash -c 'echo "rank $OMPI_COMM_WORLD_RANK on $(hostname), local_rank $OMPI_COMM_WORLD_LOCAL_RANK"'
```

The hostname test is especially useful. If you get 4 lines instead of 8, you know exactly which node MPI can't reach before you ever involve NCCL. If you get 8 lines but the collectives hang, now you're looking at NCCL bootstrap or fabric issues — a completely different problem space.

This layered approach — MPI process launch first, then NCCL bootstrap, then actual collectives — saves a lot of time by preventing you from debugging the wrong layer.
