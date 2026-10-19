#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Cluster-specific options (partition, GPU, mem, time) are passed by slurm/submit.sh.
#
# Finetune on splice data with one config:
#   slurm/submit.sh --gpu-mem 80 --mem 24G --time 36:00:00 --cpus 2 \
#       finetune_splice configs/hqmrboc_132kb.yaml [extra finetune_splice.py args...]
#
# CONFIG may be absolute or relative to the repo root. Any extra arguments are passed on
# to finetune_splice.py. RESUME (env) defaults to "auto" (latest epoch_*.pth in the run
# dir); set RESUME=path/to/checkpoint.pth or RESUME= (empty) to override.

source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"

CONFIG=${1:?usage: finetune_splice.sh CONFIG [extra args...]}
shift
[[ "${CONFIG}" = /* ]] || CONFIG="${AG_ROOT}/${CONFIG}"

# Verify config file exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# Read output_dir and run_name from config YAML
# Use Python to read and expand environment variables in paths
OUTPUT_DIR=$(python -c "import yaml; import os; c=yaml.safe_load(open('${CONFIG}')); od=c.get('output_dir','').rstrip('/'); print(os.path.expandvars(os.path.expanduser(od)) if od else '')")
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('run_name',''))")

# Fallback to timestamp if run_name is empty
if [ -z "$RUN_NAME" ]; then
    RUN_NAME=$(date +%Y%m%d_%H%M%S)
fi

LOG_DIR="${OUTPUT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${LOG_TIMESTAMP}.log"

RESUME=${RESUME-auto}
RESUME_ARGS=()
[[ -n "${RESUME}" ]] && RESUME_ARGS=(--resume "${RESUME}")

echo "Starting finetuning at $(date)"
echo "Config: ${CONFIG}"
echo "Resume: ${RESUME:-<none>}"
echo "Log file: ${LOG_FILE}"
echo "---"

# Python's setup_output_logging will handle the tee to LOG_FILE
python -u "${AG_ROOT}/scripts/finetune_splice.py" \
    --config "${CONFIG}" \
    --compile \
    "${RESUME_ARGS[@]}" \
    --log-file "${LOG_FILE}" \
    "$@"

echo "---"
echo "Finetuning completed at $(date). Logs saved to ${LOG_FILE}"
