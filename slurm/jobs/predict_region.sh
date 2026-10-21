#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Cluster-specific options (partition, GPU, mem, time) are passed by slurm/submit.sh.
#
# Predict and plot one genomic region for each organism:
#   slurm/submit.sh --gpu-mem 20 --mem 60G --time 8:00:00 \
#       predict_region lora_32_human_mouse_rat_rabbit_opossum chr17:41196312-41277500
#
# Positional: RUN (directory under ${AG_DATA}/models/), REGION
# Env overrides: DATA_TAG (default intersect_usage), ORGANISMS, CHECKPOINT_PATH
#
# Output goes under ${AG_DATA}/predictions/${RUN}/ (not models/${RUN}/), to keep
# ad-hoc region-prediction output separate from the checkpoint/training-log directory.

source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"

RUN=${1:?usage: predict_region.sh RUN REGION}
REGION=${2:?usage: predict_region.sh RUN REGION}
DATA_TAG=${DATA_TAG:-intersect_usage}
ORGANISMS=${ORGANISMS:-human mouse rat rabbit opossum chicken}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-${AG_DATA}/models/${RUN}}
PRED_DIR=region_preds_$(date +%Y%m%d_%H%M%S)
DATA_CONFIG="${AG_DATA}/data/data_config_${DATA_TAG}.json"

for ORGANISM in ${ORGANISMS}; do
    OUT_DIR=${AG_DATA}/predictions/${RUN}/${PRED_DIR}/${ORGANISM}/
    mkdir -p "${OUT_DIR}"
    python "${AG_ROOT}/scripts/predict_region.py" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --data-config "${DATA_CONFIG}" \
        --organisms "${ORGANISM}" \
        --regions "${REGION}" \
        --observed-conditions-only \
        --cross-species-usage \
        --plot \
        --device cuda \
        --output-dir "${OUT_DIR}"
done
