# sds access for clusters where it is only reachable over SFTP with rclone (HoreKa, UC3).
# Sourced by clusters/horeka.sh and clusters/uc3.sh; ag_setup_sds runs from their
# cluster_env, inside the job.
#
# The job's HOME becomes a per-job directory that links every entry of your real home
# except sds, so the $HOME/sds/... paths in configs resolve to whatever ~/sds is in that
# directory. That depends on AG_SDS_MODE:
#
#   workspace  ~/sds is $AG_WORKSPACE/sds, a copy of the inputs made beforehand on the
#              login node with `slurm/stage_inputs.sh CONFIG`. Reads are local and fast.
#              Outputs are written to the workspace too; job scripts copy them back to
#              sds with ag_sync_back (finetune_splice does every 10 min and at exit).
#              Default for finetune_splice.
#   mount      rclone-mount sds for the job (in its own TMPDIR, so dead or shared ~/sds
#              mounts on the node don't matter). Every read goes over SFTP: fine for
#              small jobs, too slow for training. Default for the other jobs.
#
# Overridable (export in the submitting shell; forwarded by submit.sh):
#   AG_SDS_MODE    workspace | mount
#   AG_WORKSPACE   workspace directory (default: `ws_find $AG_WS_NAME`)
#   AG_WS_NAME     workspace name      (default alphagenome)
#   AG_SDS_REMOTE  rclone remote for sds (default sds-sftp:, see `rclone listremotes`)
#   AG_RCLONE      rclone executable     (default: rclone on PATH, else ~/bin or ~/.local/bin)

ag_find_rclone() {
    local rclone=${AG_RCLONE:-} c
    if [[ -z "${rclone}" ]]; then
        rclone=$(command -v rclone || true)
        [[ -n "${rclone}" ]] || for c in "${HOME}/bin/rclone" "${HOME}/.local/bin/rclone"; do
            [[ -x "${c}" ]] && rclone=${c} && break
        done
    fi
    [[ -n "${rclone}" ]] || { echo "ERROR: rclone not found; set AG_RCLONE=/path/to/rclone" >&2; return 1; }
    echo "${rclone}"
}

ag_find_workspace() {
    local ws=${AG_WORKSPACE:-}
    if [[ -z "${ws}" ]] && command -v ws_find >/dev/null; then
        ws=$(ws_find "${AG_WS_NAME:-alphagenome}" 2>/dev/null || true)
    fi
    [[ -n "${ws}" && -d "${ws}" ]] || {
        echo "ERROR: no workspace; run 'ws_allocate ${AG_WS_NAME:-alphagenome} 60' and" \
             "slurm/stage_inputs.sh on the login node, or set AG_WORKSPACE" >&2
        return 1
    }
    echo "${ws}"
}

ag_setup_sds() {
    local mode=${AG_SDS_MODE:-mount}
    local remote=${AG_SDS_REMOTE:-sds-sftp:}
    local real_home=${HOME}
    local job_dir=${TMPDIR:-/tmp/${USER}-${SLURM_JOB_ID:-$$}}
    local job_home=${job_dir}/home
    AG_RCLONE_BIN=$(ag_find_rclone) || exit 1

    mkdir -p "${job_home}"
    local e name
    for e in "${real_home}"/.[!.]* "${real_home}"/*; do
        name=${e##*/}
        # Skip sds by name first: stat on a dead mount there would fail.
        [[ "${name}" == sds || "${name}" == "*" || "${name}" == ".[!.]*" ]] && continue
        ln -s "${e}" "${job_home}/${name}"
    done

    case "${mode}" in
        workspace)
            local ws
            ws=$(ag_find_workspace) || exit 1
            mkdir -p "${ws}/sds"
            ln -s "${ws}/sds" "${job_home}/sds"
            AG_SYNC_REMOTE=${remote}
            echo "sds: workspace copy ${ws}/sds (outputs synced back to ${remote})"
            ;;
        mount)
            local mnt=${job_home}/sds i
            mkdir -p "${mnt}"
            # Node-local VFS cache; files written under ~/sds are uploaded from here.
            "${AG_RCLONE_BIN}" mount "${remote}" "${mnt}" \
                --vfs-cache-mode full --cache-dir "${job_dir}/rclone-cache" --daemon
            for i in $(seq 1 30); do
                mountpoint -q "${mnt}" && break
                sleep 1
            done
            mountpoint -q "${mnt}" || { echo "ERROR: could not rclone-mount ${remote} at ${mnt}" >&2; exit 1; }
            AG_SDS_MOUNTED=${mnt}
            # Files only reach sds once closed, so job scripts keep long-open logs here
            # and copy them into the run dir (see slurm/jobs/finetune_splice.sh).
            export AG_LOCAL_LOG_DIR=${job_dir}/logs
            echo "sds: rclone mount of ${remote} at ${mnt}"
            ;;
        *)
            echo "ERROR: AG_SDS_MODE must be workspace or mount, got '${mode}'" >&2
            exit 1
            ;;
    esac

    trap _ag_sds_exit EXIT
    trap 'exit 143' TERM   # walltime/scancel: run the EXIT trap too
    export HOME=${job_home}
    # env.sh derived AG_DATA from the real HOME before this ran.
    [[ "${AG_DATA:-}" == "${real_home}/sds/"* ]] && export AG_DATA="${HOME}/sds/${AG_DATA#"${real_home}/sds/"}"
    echo "HOME=${HOME}"
}

# Copy a directory under the job's ~/sds back to the same place on sds (workspace mode;
# no-op otherwise). Only new or changed files are transferred.
ag_sync_back() {
    local dir=$1
    [[ -n "${AG_SYNC_REMOTE:-}" && -d "${dir}" ]] || return 0
    [[ "${dir}" == "${HOME}/sds/"* ]] || { echo "WARNING: ${dir} is not under ~/sds; not synced" >&2; return 0; }
    "${AG_RCLONE_BIN}" copy "${dir}" "${AG_SYNC_REMOTE}${dir#"${HOME}/sds/"}" --transfers 4 \
        || echo "WARNING: syncing ${dir} back to sds failed" >&2
}

# At job end: the job script's own hook (e.g. final log copy / sync back), then, in mount
# mode, unmount and wait for rclone to exit -- it keeps uploading cached writes after
# the unmount, and Slurm kills any process still running when the job script returns.
_ag_sds_exit() {
    declare -F _ag_job_exit_hook >/dev/null && _ag_job_exit_hook
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
