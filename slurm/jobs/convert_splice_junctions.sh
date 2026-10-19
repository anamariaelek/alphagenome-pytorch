#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Cluster-specific options (partition, mem, time) are passed by slurm/submit.sh.
#
# Build junctions.parquet (+ .json, .check.json) next to each dataset's usage.parquet
# from the Spliser .combined.tsv files, reusing usage.json's condition labels.
#
# One dataset per array task (test on Homo_sapiens first with --array=0):
#   slurm/submit.sh --cpu --mem 64G --time 12:00:00 --sbatch-arg --array=0 convert_splice_junctions
#   slurm/submit.sh --cpu --mem 64G --time 12:00:00 --sbatch-arg --array=1-12 convert_splice_junctions
# Or name datasets explicitly (run one after another):
#   slurm/submit.sh --cpu --mem 64G convert_splice_junctions Homo_sapiens brawand/Homo_sapiens
#
# Positional: DATASET... (output dir under ${AG_DATA}/data/; default: all, or the
#             SLURM_ARRAY_TASK_ID-th entry of DATASETS for array jobs)
# Env overrides: SPLISER_DIR, MIN_REPRODUCIBILITY (0.5), MIN_COUNT (1)

export AG_REQUIRE_CUDA=0
source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"
set -euo pipefail

SPLISER_DIR=${SPLISER_DIR:-${HOME}/sds/sd17d003/Anamaria/spliser/data}
MIN_REPRODUCIBILITY=${MIN_REPRODUCIBILITY:-0.5}
MIN_COUNT=${MIN_COUNT:-1}

# Same dataset list as update_usage_parquet.sh.
DATASETS=(
    Homo_sapiens Mus_musculus Rattus_norvegicus Oryctolagus_cuniculus
    Monodelphis_domestica Macaca_mulatta Gallus_gallus
    brawand/Homo_sapiens brawand/Mus_musculus brawand/Macaca_mulatta
    brawand/Pan_troglodytes brawand/Pan_paniscus brawand/Gorilla_gorilla
)

if [[ $# -gt 0 ]]; then
    SELECTED=("$@")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SELECTED=("${DATASETS[${SLURM_ARRAY_TASK_ID}]}")
else
    SELECTED=("${DATASETS[@]}")
fi

STATUS=0
for dataset in "${SELECTED[@]}"; do
    species=${dataset#brawand/}
    in_dir=${SPLISER_DIR}/${species}
    [[ "${dataset}" == brawand/* ]] && in_dir=${SPLISER_DIR}/${species}_brawand
    out_dir=${AG_DATA}/data/${dataset}
    echo "=== ${dataset}: ${in_dir} -> ${out_dir}/junctions.parquet"
    if [[ ! -f "${out_dir}/usage.json" ]]; then
        echo "No ${out_dir}/usage.json; skipping."
        continue
    fi
    python -u "${AG_ROOT}/scripts/convert_splice_junctions_to_parquet.py" \
        --input-dir "${in_dir}" \
        --usage-json "${out_dir}/usage.json" \
        --output "${out_dir}/junctions.parquet" \
        --min-reproducibility "${MIN_REPRODUCIBILITY}" \
        --min-count "${MIN_COUNT}" \
        --strip-chr-names || STATUS=1
done
exit ${STATUS}
