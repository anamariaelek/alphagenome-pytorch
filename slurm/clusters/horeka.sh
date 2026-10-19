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
# The sds share is rclone-mounted at ~/sds on the login node only (a FUSE mount is per
# node), so each job mounts it itself. A node can also hold a dead ~/sds mount left by
# an earlier job whose rclone was killed ("Transport endpoint is not connected"), and
# GPU nodes are shared, so another of your jobs may be using ~/sds there. Each job
# therefore mounts sds in its own $TMPDIR and points HOME at a per-job directory that
# links every entry of your real home except sds, which is the job's mount. The
# $HOME/sds/... paths in configs keep working, without touching the real ~/sds.
#
# The code checkout (AG_ROOT) must NOT be under ~/sds: clone the repo into $HOME or a
# workspace on HoreKa (env.sh is read before this mount exists).
#
# Overridable (export in the submitting shell; forwarded by submit.sh):
#   AG_SDS_REMOTE  rclone remote for sds (default sds-sftp:, see `rclone listremotes`)
#   AG_RCLONE      rclone executable     (default: rclone on PATH, else ~/bin/rclone)
cluster_env() {
    module load devel/cuda 2>/dev/null || true

    local remote=${AG_SDS_REMOTE:-sds-sftp:}
    local real_home=${HOME}
    local job_dir=${TMPDIR:-/tmp/${USER}-${SLURM_JOB_ID:-$$}}
    local job_home=${job_dir}/home
    local mnt=${job_home}/sds

    local rclone=${AG_RCLONE:-}
    if [[ -z "${rclone}" ]]; then
        rclone=$(command -v rclone || true)
        [[ -n "${rclone}" ]] || for c in "${real_home}/bin/rclone" "${real_home}/.local/bin/rclone"; do
            [[ -x "${c}" ]] && rclone=${c} && break
        done
    fi
    if [[ -z "${rclone}" ]]; then
        echo "ERROR: rclone not found; set AG_RCLONE=/path/to/rclone" >&2
        exit 1
    fi

    mkdir -p "${mnt}"
    local e name
    for e in "${real_home}"/.[!.]* "${real_home}"/*; do
        name=${e##*/}
        # Skip sds by name first: stat on a dead mount there would fail.
        [[ "${name}" == sds || "${name}" == "*" || "${name}" == ".[!.]*" ]] && continue
        ln -s "${e}" "${job_home}/${name}"
    done

    # Node-local VFS cache; checkpoints written under ~/sds are uploaded from here.
    "${rclone}" mount "${remote}" "${mnt}" \
        --vfs-cache-mode full \
        --cache-dir "${job_dir}/rclone-cache" \
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

    export HOME=${job_home}
    # env.sh derived AG_DATA from the real HOME before this ran.
    [[ "${AG_DATA:-}" == "${real_home}/sds/"* ]] && export AG_DATA="${HOME}/sds/${AG_DATA#"${real_home}/sds/"}"
    echo "Mounted ${remote} at ${mnt} (HOME=${HOME})"
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
