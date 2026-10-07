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
# Checkpoint fallback: if eval.log has no usable "Checkpoint   :" line (missing/truncated
# log) or the recorded path no longer exists (e.g. checkpoints later reorganised under
# models/), falls back to models/<RUN>/ -- the same directory-style --checkpoint path the
# submit scripts already use -- where <RUN> is found by walking up from the species folder
# past any number of preds_*-named ancestors (.../<RUN>/preds_*/<species>/, or
# .../<RUN>/preds_*/preds_*/<species>/ for an older nested layout a few nested runs used).
# Only used if models/<RUN>/best_model.pth actually exists, so a wrong guess never produces
# a bad job line, just no fallback.
#
# data_config fallback: same idea, guessing data/data_config_<X>.json from the pred_dir
# name (preds_<X> -- matching the PRED_DIR=preds_${TIMESTAMP} / DATA_CONFIG=data_config_
# ${TIMESTAMP}.json convention the submit scripts already use), also trying with a leading
# "all_" or trailing "_all" token stripped (a few historical runs' pred_dir names don't
# exactly match their data_config's). Only used if the guessed file actually exists.
# Neither fallback is needed for a directory whose eval.log is intact -- both only kick in
# when the recorded/expected value is missing or stale.
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

def find_run_name(out_dir: str) -> str:
    """Walk up from a species dir past any preds_*-named ancestors to find <RUN>."""
    d = os.path.dirname(out_dir)  # immediate pred_dir
    while os.path.basename(d).startswith("preds_"):
        d = os.path.dirname(d)
    return os.path.basename(d)

for p in paths:
    out_dir = os.path.dirname(p)
    species = os.path.basename(out_dir)
    run_name = find_run_name(out_dir)
    pred_dir_name = os.path.basename(os.path.dirname(out_dir))

    checkpoint = data_config = None
    log_path = os.path.join(out_dir, "eval.log")
    if os.path.exists(log_path):
        for line in open(log_path):
            if "Checkpoint   :" in line:
                checkpoint = line.split("Checkpoint   :")[1].strip()
            elif "Data config  :" in line:
                data_config = line.split("Data config  :")[1].strip()

    # Fall back to models/<RUN>/ when eval.log had no checkpoint line, or the
    # line it had points at a path that no longer exists -- see header comment.
    if not checkpoint or not os.path.exists(checkpoint):
        fallback = os.path.join("models", run_name)
        if os.path.isfile(os.path.join(fallback, "best_model.pth")):
            checkpoint = os.path.abspath(fallback)

    # Fall back to data/data_config_<X>.json, guessed from the pred_dir name,
    # when eval.log had no data_config line or it no longer exists.
    if (not data_config or not os.path.exists(data_config)) and pred_dir_name.startswith("preds_"):
        token = pred_dir_name[len("preds_"):]
        candidates = [token]
        if token.endswith("_all"):
            candidates.append(token[: -len("_all")])
        if token.startswith("all_"):
            candidates.append(token[len("all_"):])
        for cand in candidates:
            guess = os.path.join("data", f"data_config_{cand}.json")
            if os.path.isfile(guess):
                data_config = os.path.abspath(guess)
                break

    if checkpoint and data_config:
        print(f"{checkpoint}\t{data_config}\t{species}\t{os.path.abspath(out_dir)}")
PYEOF

N=$(wc -l < "${JOBS_FILE}")
echo "${N} prediction directories under ${DIR} -> ${JOBS_FILE}" >&2
