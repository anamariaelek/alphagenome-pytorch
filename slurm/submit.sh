#!/bin/bash
# Submit a job from slurm/jobs/ on whichever machine this runs on.
#
#   slurm/submit.sh [options] JOB [job args...]
#
#   JOB          name of a script in slurm/jobs/ (with or without .sh)
#
# Options (all optional):
#   --gpu-mem N  GPU memory in GB (default 40). On Helix it becomes gpumem_per_gpu:NGB;
#                on UC3 it is ignored, since the partition determines the GPU; on HoreKa
#                it only picks the default partition (<=40: A100, <=80: H100, else H200).
#   --cpu        CPU-only job: no GPU requested, the cluster's CPU partition is used
#   --mem M      host memory, e.g. 24G (default 32G)
#   --cpus N     cpus-per-task (default 2)
#   --time T     walltime, e.g. 36:00:00 (default 24:00:00)
#   --partition P  override the cluster's default GPU (or, with --cpu, CPU) partition
#   --name NAME  job name (default: ft-<config stem> for finetune_splice, else JOB-<arg1>)
#   --account A  Slurm account (default $AG_SLURM_ACCOUNT if set)
#   --sbatch-arg X  extra raw sbatch argument, repeatable (e.g. --sbatch-arg --dependency=afterok:123)
#   --dry-run    print the command instead of running it
#
# The cluster comes from $AG_CLUSTER (helix|uc3|horeka|sasselab3), or else from the hostname;
# see slurm/clusters/. On sasselab3 there is no Slurm: the job runs in the background
# with nohup, and the resource options are ignored. A --sbatch-arg --array=SPEC is still
# honored there: tasks run one after another (not concurrently, to avoid overcommitting a
# single shared workstation), each with its own SLURM_ARRAY_TASK_ID and log file.
#
# Jobs get a clean environment (--export=NONE) plus AG_ROOT/AG_CLUSTER/AG_REQUIRE_CUDA
# and a known allowlist of job-script overrides (DATA_TAG, RESUME, BATCH_SIZE, ...; see
# EXPORT_VARS below) -- NOT the submitting shell's full environment. On UC3, forwarding
# everything (--export=ALL) from a shell with conda already activated makes the job sit
# PENDING forever with reason "user env retrieval error".
#
# Examples:
#   slurm/submit.sh --gpu-mem 80 --mem 24G --time 36:00:00 finetune_splice configs/hqmrboc_132kb.yaml
#   slurm/submit.sh --gpu-mem 40 --mem 40G --time 12:00:00 evaluate_splice lora_64_emb_traj_hqmrboc___524kb
#   DATA_TAG=gtf slurm/submit.sh --dry-run evaluate_splice my_run
#   slurm/submit.sh --cpu --mem 128G --sbatch-arg --array=0-12 update_usage_parquet
set -euo pipefail

AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AG_ROOT

GPU_MEM=40 CPU_ONLY=0 MEM=32G CPUS=2 TIME=24:00:00 PARTITION="" NAME="" DRY_RUN=0
ACCOUNT=${AG_SLURM_ACCOUNT:-}
EXTRA=()

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-mem)    GPU_MEM=$2; shift 2 ;;
        --cpu)        CPU_ONLY=1; shift ;;
        --mem)        MEM=$2; shift 2 ;;
        --cpus)       CPUS=$2; shift 2 ;;
        --time)       TIME=$2; shift 2 ;;
        --partition)  PARTITION=$2; shift 2 ;;
        --name)       NAME=$2; shift 2 ;;
        --account)    ACCOUNT=$2; shift 2 ;;
        --sbatch-arg) EXTRA+=("$2"); shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage 0 ;;
        --*)          echo "ERROR: unknown option $1" >&2; usage 1 ;;
        *)            break ;;
    esac
done

[[ $# -ge 1 ]] || usage 1
JOB=${1%.sh}; shift
JOB_SCRIPT="${AG_ROOT}/slurm/jobs/${JOB}.sh"
if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "ERROR: no job script ${JOB_SCRIPT}" >&2
    echo "Available: $(cd "${AG_ROOT}/slurm/jobs" && ls *.sh | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    exit 1
fi

source "${AG_ROOT}/slurm/clusters/detect.sh"
source "${AG_ROOT}/slurm/clusters/${AG_CLUSTER}.sh"

if [[ -z "${NAME}" ]]; then
    if [[ "${JOB}" == finetune_splice && $# -ge 1 ]]; then
        NAME="ft-$(basename "${1%.yaml}")"
    else
        NAME="${JOB}${1:+-$(basename "$1")}"
    fi
fi

LOG_DIR="${AG_ROOT}/logs/slurm"
mkdir -p "${LOG_DIR}"

run() {
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' "$@"; echo
    else
        "$@"
    fi
}

# Expand a Slurm-style --array spec ("4", "0-12", "1,3,5-7", with an optional
# ":STEP" on a range and/or a trailing "%THROTTLE") into one index per line.
# THROTTLE is ignored here: local emulation always runs tasks one at a time.
expand_array_spec() {
    local spec=${1%%%*} part lo hi step idx
    IFS=',' read -ra parts <<< "${spec}"
    for part in "${parts[@]}"; do
        if [[ "${part}" == *-* ]]; then
            lo=${part%-*}; hi=${part#*-}; step=1
            if [[ "${hi}" == *:* ]]; then step=${hi#*:}; hi=${hi%%:*}; fi
            for ((idx = lo; idx <= hi; idx += step)); do echo "${idx}"; done
        else
            echo "${part}"
        fi
    done
}

if [[ "${AG_USE_SLURM}" == 1 ]]; then
    # shellcheck disable=SC2207  # cluster_*_args return space-separated flags without spaces inside
    if [[ "${CPU_ONLY}" == 1 ]]; then
        GPU_ARGS=($(cluster_cpu_args ${PARTITION:+"${PARTITION}"}))
    else
        GPU_ARGS=($(cluster_gpu_args "${GPU_MEM}" ${PARTITION:+"${PARTITION}"}))
    fi
    EXPORT_VARS=("AG_ROOT=${AG_ROOT}" "AG_CLUSTER=${AG_CLUSTER}" "AG_REQUIRE_CUDA=$((1 - CPU_ONLY))")
    for v in AG_CONDA_SH AG_CONDA_ENV DATA_TAG TRAIN_SPECIES HELDOUT_SPECIES BATCH_SIZE \
             HELDOUT_BATCH_SIZE MAX_WINDOWS CHECKPOINT_PATH PRED_DIR RESUME COMPILE MIN_COUNT \
             MIN_COVERAGE MIN_REPRODUCIBILITY ORGANISMS SPECIES SPLISER_DIR \
             AG_SDS_REMOTE AG_RCLONE AG_SDS_MODE AG_WORKSPACE AG_WS_NAME; do
        # ${!v+isset}, not ${!v:-}: RESUME= (set but empty) is a meaningful override
        # (disables auto-resume) distinct from RESUME being unset entirely.
        [[ "${!v+isset}" == "isset" ]] && EXPORT_VARS+=("${v}=${!v}")
    done
    CMD=(sbatch
        --job-name="${NAME}"
        "${GPU_ARGS[@]}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
        --output="${LOG_DIR}/%x_%j.log"
        --export="NONE,$(IFS=,; echo "${EXPORT_VARS[*]}")")
    [[ -n "${ACCOUNT}" ]] && CMD+=(--account="${ACCOUNT}")
    CMD+=(${EXTRA[@]+"${EXTRA[@]}"} "${JOB_SCRIPT}" "$@")
    echo "[${AG_CLUSTER}] submitting ${NAME}"
    run "${CMD[@]}"
else
    ARRAY_SPEC=""
    for e in ${EXTRA[@]+"${EXTRA[@]}"}; do
        [[ "${e}" == --array=* ]] && ARRAY_SPEC=${e#--array=}
    done

    if [[ -n "${ARRAY_SPEC}" ]]; then
        TASKS=$(expand_array_spec "${ARRAY_SPEC}" | tr '\n' ' ')
        if [[ "${DRY_RUN}" == 1 ]]; then
            echo "[${AG_CLUSTER}] would run array tasks locally, one after another: ${TASKS}"
        else
            run_array_tasks() {
                local idx tlog
                for idx in $(expand_array_spec "${ARRAY_SPEC}"); do
                    tlog="${LOG_DIR}/${NAME}_task${idx}_$(date +%Y%m%d_%H%M%S).log"
                    echo "[${AG_CLUSTER}] array task ${idx}: log ${tlog}"
                    SLURM_ARRAY_TASK_ID="${idx}" bash "${JOB_SCRIPT}" "$@" > "${tlog}" 2>&1
                done
            }
            export -f run_array_tasks expand_array_spec
            export ARRAY_SPEC LOG_DIR NAME AG_CLUSTER JOB_SCRIPT
            nohup bash -c 'run_array_tasks "$@"' _ "$@" \
                > "${LOG_DIR}/${NAME}_array.log" 2>&1 &
            echo "PID $!  (sequential array tasks: ${TASKS})"
            echo "Index log: ${LOG_DIR}/${NAME}_array.log  (per-task logs listed inside)"
        fi
    else
        LOG="${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log"
        echo "[${AG_CLUSTER}] running ${NAME} locally (no Slurm); log: ${LOG}"
        if [[ "${DRY_RUN}" == 1 ]]; then
            run nohup bash "${JOB_SCRIPT}" "$@"
        else
            nohup bash "${JOB_SCRIPT}" "$@" > "${LOG}" 2>&1 &
            echo "PID $!"
        fi
    fi
fi
