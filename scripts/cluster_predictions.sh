#!/bin/bash
# GP-cluster predicted splice-usage trajectories for every (model, species) pair returned by
# list_prediction_dirs.sh, across the tissues with a reference clustering
# (Brain/Cerebellum/Liver/Testis -- matches examples/notebooks/splice_prediction_clustering.ipynb's
# TISSUE_LIST). Always runs that script first (rather than assuming a stale jobs file), so this
# always picks up newly-evaluated predictions -- just rerun this script.
#
# Writes to <preds_dir>/<species>/pred_gp_splice_usage/<tissue>/, the exact nested layout the
# notebook's pred_clusters_path() expects. Species with no reference clustering (chicken) are
# skipped automatically. cluster_predictions.py itself is idempotent (skips existing output
# unless --overwrite), so this is safe to re-run any time.
#
# Run on the cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai
ROOT=${HOME}/sds/sd17d003/Anamaria
WORK_DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code
JOBS_FILE=${JOBS_FILE:-/tmp/prediction_dirs.txt}
REF_DIR=${ROOT}/alphagenome_genomicsxai_code/devgp/test/
USAGE_TEMPLATE=${DIR}/data/combined_usage_data_{species}.parquet
TISSUES=(Brain Cerebellum Heart Kidney Liver Ovary Testis)

# Total parallel cluster_predictions.py processes, and GP-fitting threads *within* each one
# (--n-jobs). Their product should stay under your core count -- these run CPU-bound GP fits
# per site, so oversubscribing both dimensions at once just thrashes.
PARALLEL=${PARALLEL:-6}
GP_NJOBS=${GP_NJOBS:-2}
# Set OVERWRITE=1 to recompute existing outputs (e.g. after a reference relabel, to pick up
# the new per-site ShapeSite / same_shape_site columns). Default 0 = skip existing.
OVERWRITE=${OVERWRITE:-0}
OV_FLAG=""; [[ "${OVERWRITE}" == "1" ]] && OV_FLAG="--overwrite"

bash "${SCRIPT_DIR}/list_prediction_dirs.sh"

TISSUE_JOBS_FILE=/tmp/cluster_predictions_jobs.txt
: > "${TISSUE_JOBS_FILE}"

while IFS=$'\t' read -r CKPT DCFG SP OUT_DIR; do
    PREDS_DIR=$(dirname "${OUT_DIR}")
    if [[ ! -d "${REF_DIR}/${SP}" ]]; then
        continue  # no reference clustering for this species (e.g. chicken)
    fi
    for TIS in "${TISSUES[@]}"; do
        echo -e "${PREDS_DIR}\t${SP}\t${TIS}" >> "${TISSUE_JOBS_FILE}"
    done
done < "${JOBS_FILE}"

# Dedup: multiple jobs-file rows can share the same (preds_dir, species) if the same run
# appears under more than one metrics.json family (e.g. preds_intersect_protein_coding and
# an _epoch_NN variant) -- cluster once per unique (preds_dir, species, tissue).
sort -u -o "${TISSUE_JOBS_FILE}" "${TISSUE_JOBS_FILE}"

N=$(wc -l < "${TISSUE_JOBS_FILE}")
echo "Clustering predictions for ${N} (model, species, tissue) combinations (parallel=${PARALLEL}, gp-n-jobs=${GP_NJOBS})..."

xargs -a "${TISSUE_JOBS_FILE}" -P "${PARALLEL}" -I {} -d '\n' bash -c '
    IFS=$'"'"'\t'"'"' read -r PREDS_DIR SP TIS <<< "{}"
    OUT="${PREDS_DIR}/${SP}/pred_gp_splice_usage/${TIS}"
    mkdir -p "${OUT}"
    python "'"${WORK_DIR}"'/scripts/cluster_predictions.py" \
        --species "${SP}" \
        --tissue "${TIS}" \
        --ref-dir "'"${REF_DIR}"'" \
        --preds-dir "${PREDS_DIR}" \
        --usage-template "'"${USAGE_TEMPLATE}"'" \
        --n-jobs "'"${GP_NJOBS}"'" \
        --output "${OUT}" '"${OV_FLAG}"' \
        > "${OUT}/cluster_predictions.log" 2>&1
    echo "done: ${OUT}"
'

echo "All done."
