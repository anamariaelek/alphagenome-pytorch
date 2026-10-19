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

# Runs inside the job, before conda is activated.
#
# The sds share is rclone-mounted at ~/sds on the login node only; a FUSE mount is
# per node, so on a compute node ~/sds is an empty directory. Mount it again at the
# same path for the lifetime of the job, so the $HOME/sds/... paths in configs keep
# working. The code checkout (AG_ROOT) must therefore NOT be under ~/sds: clone the
# repo into $HOME or a workspace on HoreKa (env.sh is read before this mount exists).
#
# Overridable (export in the submitting shell; forwarded by submit.sh):
#   AG_SDS_REMOTE  rclone remote for sds (default sds-sftp:, see `rclone listremotes`)
#   AG_SDS_MOUNT   mount point            (default ~/sds)
cluster_env() {
    module load devel/cuda 2>/dev/null || true

    local remote=${AG_SDS_REMOTE:-sds-sftp:} mnt=${AG_SDS_MOUNT:-${HOME}/sds}
    if mountpoint -q "${mnt}"; then
        return 0
    fi
    mkdir -p "${mnt}"
    # Node-local VFS cache; checkpoints written under ~/sds are uploaded from here.
    rclone mount "${remote}" "${mnt}" \
        --vfs-cache-mode full \
        --cache-dir "${TMPDIR:-/tmp}/rclone-cache" \
        --daemon
    local i
    for i in $(seq 1 30); do
        mountpoint -q "${mnt}" && break
        sleep 1
    done
    if ! mountpoint -q "${mnt}"; then
        echo "ERROR: could not rclone-mount ${remote} at ${mnt}" >&2
        exit 1
    fi
    AG_SDS_MOUNTED=${mnt}
    trap _ag_unmount_sds EXIT
    trap 'exit 143' TERM   # walltime/scancel: run the EXIT trap too
    echo "Mounted ${remote} at ${mnt}"
}

# Unmount at job end, then wait for rclone to exit: it keeps uploading cached writes
# (e.g. checkpoints under ~/sds) after the unmount, and Slurm kills any process still
# running when the job script returns.
_ag_unmount_sds() {
    local mnt=${AG_SDS_MOUNTED:-}
    [[ -n "${mnt}" ]] || return 0
    fusermount -u "${mnt}" 2>/dev/null || fusermount3 -u "${mnt}" 2>/dev/null || true
    local i
    for i in $(seq 1 600); do
        pgrep -u "$(id -u)" -f "rclone mount .* ${mnt}" >/dev/null || return 0
        sleep 1
    done
    echo "WARNING: rclone still running 10 min after unmounting ${mnt}; uploads may be incomplete" >&2
}
