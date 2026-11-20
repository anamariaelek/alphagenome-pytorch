# HoreKa 2 (KIT / NHR) profile (sourced by submit.sh and env.sh).
# https://docs.nhr.kit.edu/usage/slurm/#horeka-2-partitions
# Login: hk2-x86.scc.kit.edu (x86), hk2-a64.scc.kit.edu (ARM, once available).
#
# GPU partitions (shared nodes, --gres=gpu:N, max walltime 2 days):
#   gpu-h100     4x H100  94 GB   x86   (HoreKa Teal)
#   gpu-h200     4x H200 141 GB   x86   (HoreKa Ruby)
#   gpu-h200-8   8x H200 141 GB   x86   (HoreKa Teal, a single node)
#   gpu-b200     4x B200 186 GB   ARM / Grace (HoreKa Jade)
#   dev-gpu-h100, dev-gpu-b200    1 h test queues, one job at a time (no H200 dev queue)
# CPU partitions: cpu (max 3 days), dev-cpu (1 h), large (4 TB nodes, 2 days).
# --partition and --time are mandatory on HoreKa 2; submit.sh always passes both.
#
# --gpu-mem only picks the default partition: <= 94 GB -> gpu-h100, more -> gpu-h200.
# gpu-b200 is never picked automatically: its nodes are aarch64, so it needs a conda env
# built on the ARM login node from env_alphagenome_pytorch_genomicsxai_aarch64.yml under a
# separate name, e.g. AG_CONDA_ENV=alphagenome_pytorch_genomicsxai_aarch64
# slurm/submit.sh --partition gpu-b200 ... (an x86 env in the shared $HOME won't run there).
# If your user belongs to more than one project, set AG_SLURM_ACCOUNT (or pass --account).

AG_USE_SLURM=1

# sbatch args for one GPU with at least $1 GB of memory.
cluster_gpu_args() {
    local gpu_mem=$1 partition=${2:-}
    if [[ -z "${partition}" ]]; then
        if (( gpu_mem <= 94 )); then
            partition=gpu-h100
        else
            partition=gpu-h200
        fi
    fi
    echo "--partition=${partition} --gres=gpu:1"
}

# sbatch args for a CPU-only job.
cluster_cpu_args() {
    local partition=${1:-cpu}
    echo "--partition=${partition}"
}

# sds is reached over rclone exactly as on HoreKa 1 (workspace copy for training, per-job
# mount otherwise); see clusters/sds_rclone.sh. Keep the code checkout outside ~/sds.
source "${AG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/slurm/clusters/sds_rclone.sh"

# Runs inside the job, before conda is activated. The pip torch wheels ship their own CUDA
# runtime, so the module is only a convenience. HoreKa 2 uses EasyBuild-style names
# (CUDA/<version>), not HoreKa 1's devel/cuda.
cluster_env() {
    module load CUDA 2>/dev/null || true
    ag_setup_sds
}
