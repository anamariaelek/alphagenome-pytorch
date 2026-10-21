# HoreKa (KIT / NHR) profile (sourced by submit.sh and env.sh).
# https://www.nhr.kit.edu/userdocs/horeka/batch/
#
# GPU partitions (shared nodes, --gres=gpu:N, max walltime 2 days):
#   accelerated          4x A100        (HoreKa Green)
#   accelerated-h100     4x H100        (HoreKa Teal)
#   accelerated-h200     4x H200        (HoreKa Ruby)
#   accelerated-h200-8   8x H200        (HoreKa Teal)
#   dev_accelerated, dev_accelerated-h100   1h test queues, one job at a time
# CPU partitions: cpuonly (max 3 days), dev_cpuonly (4h).
# As on UC3, the partition selects the GPU: --gpu-mem only picks the default partition
# (<= 40 GB: accelerated, <= 80 GB: accelerated-h100, more: accelerated-h200).
# Use --partition on submit.sh to pick one explicitly. If your user belongs to more than
# one project, set AG_SLURM_ACCOUNT (or pass --account).

AG_USE_SLURM=1

# sbatch args for one GPU with at least $1 GB of memory.
cluster_gpu_args() {
    local gpu_mem=$1 partition=${2:-}
    if [[ -z "${partition}" ]]; then
        if (( gpu_mem <= 40 )); then
            partition=accelerated
        elif (( gpu_mem <= 80 )); then
            partition=accelerated-h100
        else
            partition=accelerated-h200
        fi
    fi
    echo "--partition=${partition} --gres=gpu:1"
}

# sbatch args for a CPU-only job.
cluster_cpu_args() {
    local partition=${1:-cpuonly}
    echo "--partition=${partition}"
}

# sds is rclone-mounted at ~/sds on the login node only; see clusters/sds_rclone.sh for
# how jobs reach it (workspace copy for training, per-job rclone mount otherwise). The
# code checkout (AG_ROOT) must NOT be under ~/sds: clone the repo into $HOME or a
# workspace (env.sh is read before sds is set up).
source "${AG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/slurm/clusters/sds_rclone.sh"

# Runs inside the job, before conda is activated.
cluster_env() {
    module load devel/cuda 2>/dev/null || true
    ag_setup_sds
}
