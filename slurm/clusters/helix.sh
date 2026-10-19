# Helix profile (sourced by submit.sh and env.sh).
# https://wiki.bwhpc.de/e/Helix/Slurm
#
# GPUs on gpu-single: A40 (48 GB), A100 (40/80 GB), H200 (141 GB). Rather than pinning a
# type, request the GPU memory you need and let Slurm pick a card that has it.
# Max walltime on gpu-single: 120h.

AG_USE_SLURM=1

# sbatch args for one GPU with at least $1 GB of memory.
cluster_gpu_args() {
    local gpu_mem=$1 partition=${2:-gpu-single}
    echo "--partition=${partition} --gres=gpu:1,gpumem_per_gpu:${gpu_mem}GB"
}

# sbatch args for a CPU-only job.
cluster_cpu_args() {
    local partition=${1:-cpu-single}
    echo "--partition=${partition}"
}

# Runs inside the job, before conda is activated.
cluster_env() {
    module load devel/cuda
}
