#!/bin/bash
# Re-calculates trajectory_correlation / trajectory_magnitude (mean_pearson_r, rmse,
# amplitude_r2, trajectory_r2) for every prediction directory returned by
# list_prediction_dirs.sh -- always runs that script first, so this covers every directory
# under DIR, not just ones that already had --trajectory-corr run before: first-time and
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
echo "Recalculating trajectory metrics in ${N} directories (parallel=${PARALLEL})..."

xargs -a "${JOBS_FILE}" -P "${PARALLEL}" -I {} -d '\n' bash -c '
    IFS=$'"'"'\t'"'"' read -r CKPT DCFG SP OUT <<< "{}"
    python "'"${WORK_DIR}"'/scripts/evaluate_splice.py" \
        --checkpoint "${CKPT}" \
        --data-config "${DCFG}" \
        --eval-species "${SP}" \
        --per-tissue \
        --observed-conditions-only \
        --trajectory-corr \
        --skip-predictions \
        --overwrite \
        --device cpu \
        --output-dir "${OUT}" \
        > "${OUT}/refresh_traj_r2.log" 2>&1
    echo "done: ${OUT}"
'

echo "All done. Check each directory's refresh_traj_r2.log if a job failed."
