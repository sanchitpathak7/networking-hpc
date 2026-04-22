#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gb200

test_bin="/opt/nccl-tests/build"

srun --mpi=pmix --unbuffered \
  -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
  -x NCCL_DEBUG=INFO \
  $test_bin/all_reduce_perf -b 2G -e 32G -f 2 -g 1 \
  2>&1 | tee all_reduce.txt


test_bin="/opt/nccl-tests/build/"
tests=("all_reduce_perf")
hostfile="hostfile"
gpus=$(( $(wc -l < "$hostfile") * 4 ))

for test in "${tests[@]}"; do
mpirun  --allow-run-as-root \
  --hostfile $hostfile \
  -np $gpus --map-by ppr:4:node  \
  -x UCX_TLS=tcp,self \
  -x NCCL_DEBUG=INFO \
  --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --mca coll ^cuda \
  -x LD_LIBRARY_PATH \
  $test_bin/$test -b 2G -e 32G -f 2 -g 1 \
  2>&1 | tee ${test}.txt
done