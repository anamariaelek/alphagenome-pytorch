#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Cluster-specific options (partition, mem, time) are passed by slurm/submit.sh.
#
# Regenerate existing usage.parquet files with the Sample column kept, so that
# replicates with identical values in the same condition are no longer collapsed.
# For each dataset:
#   1. convert Spliser .combined.tsv -> usage.new.parquet (same flags as the originals)
#   2. compare against the old usage.parquet (scripts/compare_usage_parquets.py),
#      writing usage.compare.json next to it
#   3. only if the comparison passes: back up the old files to
#      usage.collapsed_bak.{parquet,json} (an existing backup is never overwritten) and
#      move the new files into place. On failure the old files stay untouched, the new
#      ones are left next to them, and the job exits 1.
#
# One dataset per array task (test on Homo_sapiens first with --array=0):
#   slurm/submit.sh --cpu --mem 128G --time 24:00:00 --sbatch-arg --array=0 update_usage_parquet
#   slurm/submit.sh --cpu --mem 128G --time 24:00:00 --sbatch-arg --array=1-12 update_usage_parquet
# Or name datasets explicitly (run one after another):
#   slurm/submit.sh --cpu --mem 128G update_usage_parquet Homo_sapiens brawand/Homo_sapiens
#
# Positional: DATASET... (output dir under ${AG_DATA}/data/; default: all, or the
#             SLURM_ARRAY_TASK_ID-th entry of DATASETS for array jobs)
# Env overrides: SPLISER_DIR, MIN_COVERAGE (10), MIN_REPRODUCIBILITY (0.5)

export AG_REQUIRE_CUDA=0
source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"
set -euo pipefail

SPLISER_DIR=${SPLISER_DIR:-${HOME}/sds/sd17d003/Anamaria/spliser/data}
MIN_COVERAGE=${MIN_COVERAGE:-10}
MIN_REPRODUCIBILITY=${MIN_REPRODUCIBILITY:-0.5}

# Output dir under ${AG_DATA}/data/. The Spliser input dir is the species name,
# with a _brawand suffix for the Brawand datasets.
DATASETS=(
    # Cardoso-Moreira (training)
    Homo_sapiens Mus_musculus Rattus_norvegicus Oryctolagus_cuniculus
    Monodelphis_domestica Macaca_mulatta Gallus_gallus
    # Brawand (test)
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

update_one() {
    local dataset=$1
    local species=${dataset#brawand/}
    local in_dir=${SPLISER_DIR}/${species}
    [[ "${dataset}" == brawand/* ]] && in_dir=${SPLISER_DIR}/${species}_brawand
    local out_dir=${AG_DATA}/data/${dataset}
    local old=${out_dir}/usage.parquet
    local new=${out_dir}/usage.new.parquet
    local bak=${out_dir}/usage.collapsed_bak.parquet

    echo "=== ${dataset}: ${in_dir} -> ${out_dir}"
    if [[ ! -f "${old}" ]]; then
        echo "No existing ${old}; skipping."
        return 0
    fi

    python -u "${AG_ROOT}/scripts/convert_splice_usage_to_parquet.py" \
        --input-dir "${in_dir}" \
        --output "${new}" \
        --min-reproducibility "${MIN_REPRODUCIBILITY}" \
        --min-coverage "${MIN_COVERAGE}" \
        --strip-chr-names

    if ! python -u "${AG_ROOT}/scripts/compare_usage_parquets.py" \
        --old "${old}" \
        --new "${new}" \
        --min-coverage "${MIN_COVERAGE}" \
        --report "${out_dir}/usage.compare.json"
    then
        echo "Comparison FAILED for ${dataset}; ${old} unchanged, new files left at ${new}."
        return 1
    fi

    if [[ -e "${bak}" ]]; then
        echo "Backup ${bak} already exists; leaving it untouched."
    else
        cp -p "${old}" "${bak}"
        cp -p "${old%.parquet}.json" "${bak%.parquet}.json"
        echo "Backed up old files to ${bak%.parquet}.{parquet,json}"
    fi
    mv "${new}" "${old}"
    mv "${new%.parquet}.json" "${old%.parquet}.json"
    echo "Updated ${old}"
}

STATUS=0
for dataset in "${SELECTED[@]}"; do
    update_one "${dataset}" || STATUS=1
done
exit ${STATUS}
