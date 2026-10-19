# bwUniCluster 3.0 profile (sourced by submit.sh and env.sh).
# https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs
#
# Unlike Helix, there is no gpumem_per_gpu gres; the partition selects the GPU type:
#   gpu_h100     4x H100, max 72h   (default here; enough memory for every job we run)
#   gpu_a100_il  4x A100, max 48h
#   gpu_h100_il  4x H100, max 48h
#   gpu_mi300    AMD MI300 -- do NOT use, our PyTorch env is a CUDA build
#   dev_* / *_short -- short test queues (check `sinfo -s` for the limits)
# Use --partition on submit.sh to pick a different one. Check `sinfo -s` on first use.

AG_USE_SLURM=1

# sbatch args for one GPU. $1 (GPU memory) is ignored here, since every UC3 NVIDIA
# partition has at least as much memory as our largest job asks for.
cluster_gpu_args() {
    local partition=${2:-gpu_h100}
    echo "--partition=${partition} --gres=gpu:1"
}

# sbatch args for a CPU-only job. TODO(first use): confirm the partition with `sinfo -s`.
cluster_cpu_args() {
    local partition=${1:-cpu}
    echo "--partition=${partition}"
}

# Runs inside the job, before conda is activated.
# TODO(first use): confirm the module name with `module avail cuda`.
cluster_env() {
    module load devel/cuda
}
