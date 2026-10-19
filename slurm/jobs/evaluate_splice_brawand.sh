#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Cluster-specific options (partition, GPU, mem, time) are passed by slurm/submit.sh.
#
# Evaluate one run on the Brawand primate/mouse data (see WORKFLOW_TEST_SPLICING.md):
#   slurm/submit.sh --gpu-mem 40 --mem 40G --time 12:00:00 \
#       evaluate_splice_brawand lora_64_emb_traj_hqmrboc__
#
# Positional: RUN (directory under ${AG_DATA}/)
# Env overrides: DATA_TAG (default intersect_protein_coding), SPECIES, BATCH_SIZE (6),
#                MAX_WINDOWS (1000), CHECKPOINT_PATH

source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"

RUN=${1:?usage: evaluate_splice_brawand.sh RUN}
DATA_TAG=${DATA_TAG:-intersect_protein_coding}
SPECIES=${SPECIES:-human gorilla chimpanzee bonobo macaque mouse}
BATCH_SIZE=${BATCH_SIZE:-6}
MAX_WINDOWS=${MAX_WINDOWS:-1000}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-${AG_DATA}/${RUN}}
PRED_DIR=${PRED_DIR:-preds_${DATA_TAG}}
DATA_CONFIG="${AG_DATA}/data/brawand/data_config_${DATA_TAG}.json"

for EVAL_SPECIES in ${SPECIES}; do
    OUT_DIR=${AG_DATA}/${RUN}/${PRED_DIR}/${EVAL_SPECIES}_brawand/
    mkdir -p "${OUT_DIR}"
    python "${AG_ROOT}/scripts/evaluate_splice.py" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --data-config "${DATA_CONFIG}" \
        --eval-species "${EVAL_SPECIES}" \
        --per-tissue \
        --observed-conditions-only \
        --overwrite \
        --batch-size "${BATCH_SIZE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device cuda \
        --seed 1950 \
        --output-dir "${OUT_DIR}"
done
