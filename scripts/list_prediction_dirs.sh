#!/bin/bash
# General lister: scans every model run directory under DIR for per-species prediction
# output directories (identified by a metrics.json inside them, under any
# */preds_*/<species>/ path), and writes one tab-separated (checkpoint, data_config, species,
# output_dir) line per match to JOBS_FILE (default /tmp/prediction_dirs.txt) -- re-derived
# from each directory's own eval.log, so no hardcoded model list. Unconstrained on both the
# model run directory's own name (lora_*, 132kb_encode_intersect_usage_pc*, ...) and the
# preds-subfolder's name beyond the preds_ prefix (preds_intersect_protein_coding,
# preds_intersect, preds_gtf, preds_union, ...). Lists every matching directory regardless of
# what's already been computed inside it -- consumers below decide what to do with each one.
#
# Consumed by:
#   - backfill_trajectory_metrics.sh   (re-run evaluate_splice.py --trajectory-corr for every dir)
#   - cluster_predictions.sh (GP-cluster predictions for every dir)
# Both always run this script first, so they cover the same set of directories without
# needing to be chained manually.
#
# Run on the cluster (reads metrics.json/eval.log from prediction output directories).
# Cheap and side-effect-free -- rerun any time new predictions/evaluations appear.
set -euo pipefail

DIR=${DIR:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai}
JOBS_FILE=${JOBS_FILE:-/tmp/prediction_dirs.txt}

cd "${DIR}"

python3 - <<'PYEOF' > "${JOBS_FILE}"
import glob, os

PATTERN = "**/preds_*/*/metrics.json"
paths = sorted(glob.glob(PATTERN, recursive=True))

for p in paths:
    out_dir = os.path.dirname(p)
    species = os.path.basename(out_dir)
    log_path = os.path.join(out_dir, "eval.log")
    if not os.path.exists(log_path):
        continue
    checkpoint = data_config = None
    for line in open(log_path):
        if "Checkpoint   :" in line:
            checkpoint = line.split("Checkpoint   :")[1].strip()
        elif "Data config  :" in line:
            data_config = line.split("Data config  :")[1].strip()
    if checkpoint and data_config:
        print(f"{checkpoint}\t{data_config}\t{species}\t{os.path.abspath(out_dir)}")
PYEOF

N=$(wc -l < "${JOBS_FILE}")
echo "${N} prediction directories under ${DIR} -> ${JOBS_FILE}" >&2
