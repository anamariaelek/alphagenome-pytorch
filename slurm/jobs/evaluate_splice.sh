#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Cluster-specific options (partition, GPU, mem, time) are passed by slurm/submit.sh.
#
# Evaluate one finetuned run on every species:
#   slurm/submit.sh --gpu-mem 40 --mem 40G --time 12:00:00 \
#       evaluate_splice lora_64_emb_traj_hqmrboc___524kb
#
# Memory guide: 132kb models ~25 GB GPU for batch 4; 524kb models ~95 GB for batch 4.
#
# Positional: RUN (directory under ${AG_DATA}/models/)
# Env overrides:
#   DATA_TAG         data_config_<DATA_TAG>.json and preds_<DATA_TAG>/ (default intersect_protein_coding;
#                    others: gtf usage intersect union intersect_usage union_protein_coding)
#   TRAIN_SPECIES    species the model was trained on      (default: all 7)
#   HELDOUT_SPECIES  left-out species, evaluated with --cross-species-usage (default: none)
#   BATCH_SIZE       (default 3)   HELDOUT_BATCH_SIZE (default 6)
#   MAX_WINDOWS      (default 1000)
#   CHECKPOINT_PATH  override the checkpoint (e.g. ${AG_ROOT}/checkpoints/pretrained.pth for
#                    a sanity check; then also set PRED_DIR=preds_pretrained_<tag>)

source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"

RUN=${1:?usage: evaluate_splice.sh RUN}
DATA_TAG=${DATA_TAG:-intersect_protein_coding}
TRAIN_SPECIES=${TRAIN_SPECIES:-human mouse macaque rat rabbit opossum chicken}
HELDOUT_SPECIES=${HELDOUT_SPECIES:-}
BATCH_SIZE=${BATCH_SIZE:-3}
HELDOUT_BATCH_SIZE=${HELDOUT_BATCH_SIZE:-6}
MAX_WINDOWS=${MAX_WINDOWS:-1000}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-${AG_DATA}/models/${RUN}}
PRED_DIR=${PRED_DIR:-preds_${DATA_TAG}}
DATA_CONFIG="${AG_DATA}/data/data_config_${DATA_TAG}.json"

echo "Run: ${RUN}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data config: ${DATA_CONFIG}"
echo "Trained species: ${TRAIN_SPECIES}"
echo "Held-out species: ${HELDOUT_SPECIES:-<none>}"

# Evaluation on trained species
for EVAL_SPECIES in ${TRAIN_SPECIES}; do
    OUT_DIR=${AG_DATA}/evals/${RUN}/${PRED_DIR}/${EVAL_SPECIES}/
    mkdir -p "${OUT_DIR}"
    python "${AG_ROOT}/scripts/evaluate_splice.py" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --data-config "${DATA_CONFIG}" \
        --eval-species "${EVAL_SPECIES}" \
        --per-tissue \
        --observed-conditions-only \
        --trajectory-corr \
        --tissue-contrast \
        --overwrite \
        --batch-size "${BATCH_SIZE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device cuda \
        --seed 1950 \
        --output-dir "${OUT_DIR}"
done

# Evaluation on left-out species
for EVAL_SPECIES in ${HELDOUT_SPECIES}; do
    OUT_DIR=${AG_DATA}/evals/${RUN}/${PRED_DIR}/${EVAL_SPECIES}/
    mkdir -p "${OUT_DIR}"
    python "${AG_ROOT}/scripts/evaluate_splice.py" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --data-config "${DATA_CONFIG}" \
        --eval-species "${EVAL_SPECIES}" \
        --per-tissue \
        --cross-species-usage \
        --observed-conditions-only \
        --trajectory-corr \
        --tissue-contrast \
        --overwrite \
        --batch-size "${HELDOUT_BATCH_SIZE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device cuda \
        --seed 1950 \
        --output-dir "${OUT_DIR}"
done
