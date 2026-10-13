#!/bin/bash
# Re-calculates tissue_contrast (mean_true_gap, mean_pred_gap, compression_ratio, rmse,
# pearson_r, sign_correct_frac) for every prediction directory returned by
# list_prediction_dirs.sh -- always runs that script first, so this covers every directory
# under DIR, not just ones that already had --tissue-contrast run before: first-time and
# refresh runs are handled identically, since --skip-predictions --overwrite recomputes the
# whole per-species metrics.json block regardless of what (if anything) was there already.
# Run on the cluster (needs torch); --device cpu since --skip-predictions never touches the
# GPU/model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code
JOBS_FILE=${JOBS_FILE:-/tmp/prediction_dirs.txt}
PARALLEL=${PARALLEL:-8}

bash "${SCRIPT_DIR}/list_prediction_dirs.sh"

N=$(wc -l < "${JOBS_FILE}")
echo "Recalculating tissue_contrast metrics in ${N} directories (parallel=${PARALLEL})..."

xargs -a "${JOBS_FILE}" -P "${PARALLEL}" -I {} -d '\n' bash -c '
    IFS=$'"'"'\t'"'"' read -r CKPT DCFG SP OUT <<< "{}"
    if python "'"${WORK_DIR}"'/scripts/evaluate_splice.py" \
        --checkpoint "${CKPT}" \
        --data-config "${DCFG}" \
        --eval-species "${SP}" \
        --per-tissue \
        --observed-conditions-only \
        --tissue-contrast \
        --skip-predictions \
        --overwrite \
        --device cpu \
        --output-dir "${OUT}" \
        > "${OUT}/refresh_tissue_contrast.log" 2>&1
    then
        echo "done: ${OUT}"
    else
        echo "FAILED: ${OUT} (see ${OUT}/refresh_tissue_contrast.log)"
    fi
'

echo "All done. Check for any FAILED lines above, or grep -L tissue_contrast */metrics.json."
