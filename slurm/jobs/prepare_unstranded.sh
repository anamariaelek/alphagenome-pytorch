#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Cluster-specific options (partition, mem, time) are passed by slurm/submit.sh.
#
# Finetuning/evaluation inputs for an unstranded Spliser dataset (Strand '?'):
#   DATASET=khrameeva  brain regions, spliser/data/<Species>_khrameeva (default)
#   DATASET=brawand    6 organs,      spliser/data/<Species>_brawand
# written to ${AG_DATA}/data/${DATASET}/<Species>/:
#   gene_annotation.parquet, gene_annotation_protein_coding.parquet
#                                   copied from ${GTF_DIR} (same Ensembl 115 GTFs)
#   usage.parquet (+ .json, usage.unshifted.parquet)
#                                   min coverage/reproducibility filters; left-end shift and
#                                   Labels from intron motifs (shift_unstranded_usage.py)
#   usage.unfiltered.parquet (+ .json, usage.unfiltered.unshifted.parquet)
#                                   same, without filters
#   junctions.parquet (+ .json, .check.json)  motif-strand junctions
#   splice_sites_gtf.parquet                   GTF sites (all genes)
#   splice_sites_intersect_protein_coding.parquet
#   splice_sites_union_protein_coding.parquet
#   splice_sites_intersect_usage_protein_coding.parquet
# and, when all species are run in one job (no array), the dataset-level
#   ${AG_DATA}/data/${DATASET}/data_config_{intersect,intersect_usage}_protein_coding.json
#
# Genomes are Ensembl 115 (${GENOME_DIR}/<Species>.fa), also for mouse (GRCm39): the
# Brawand mouse Spliser runs are on GRCm39, not the GRCm38 "mazin" genome. The mouse fold
# BEDs are still GRCm38, so the data configs list mouse but it should not be trained on.
#
# Condition names are <Tissue>_14 with underscores dropped from the tissue (e.g.
# CerebellarGreyMatter_14), so code reading the tissue as the text before the first "_"
# sees one tissue per region.
#
# Each step is skipped if its output exists (delete to redo).
#
# Submit through the per-dataset wrappers (submit.sh does not forward DATASET):
# one species per array task, then the data configs:
#   slurm/submit.sh --cpu --mem 96G --time 08:00:00 --sbatch-arg --array=0-3 prepare_khrameeva
#   slurm/submit.sh --cpu --mem 8G --time 00:10:00 prepare_khrameeva --configs-only
#   slurm/submit.sh --cpu --mem 96G --time 08:00:00 --sbatch-arg --array=0-6 prepare_brawand
#   slurm/submit.sh --cpu --mem 8G --time 00:10:00 prepare_brawand --configs-only
# Or everything in one job (species one after another):
#   slurm/submit.sh --cpu --mem 96G --time 24:00:00 prepare_brawand
# Or name species explicitly:
#   slurm/submit.sh --cpu --mem 96G --time 08:00:00 prepare_brawand Pongo_abelii
#
# Positional: SPECIES... (default: all, or the SLURM_ARRAY_TASK_ID-th entry of SPECIES_LIST),
#             or --configs-only
# Env overrides: DATASET, SPLISER_DIR, GENOME_DIR, GTF_DIR, MIN_REPRODUCIBILITY (0.5),
#                MIN_COVERAGE (10)

export AG_REQUIRE_CUDA=0
source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"
set -euo pipefail

DATASET=${DATASET:-khrameeva}
SPLISER_DIR=${SPLISER_DIR:-${HOME}/sds/sd17d003/Anamaria/spliser/data}
GENOME_DIR=${GENOME_DIR:-${HOME}/sds/sd17d003/Anamaria/genomes/ensembl115/fasta}
GTF_DIR=${GTF_DIR:-${AG_DATA}/data/brawand}   # Ensembl 115 gene_annotation*.parquet
MIN_REPRODUCIBILITY=${MIN_REPRODUCIBILITY:-0.5}
MIN_COVERAGE=${MIN_COVERAGE:-10}
DATA_ROOT=${AG_DATA}/data/${DATASET}

case "${DATASET}" in
    khrameeva) SPECIES_LIST=(Homo_sapiens Macaca_mulatta Pan_troglodytes Pan_paniscus) ;;
    brawand)   SPECIES_LIST=(Homo_sapiens Macaca_mulatta Pan_troglodytes Pan_paniscus
                             Gorilla_gorilla Pongo_abelii Mus_musculus) ;;
    *) echo "Unknown DATASET '${DATASET}' (khrameeva | brawand)" >&2; exit 1 ;;
esac
echo "Dataset: ${DATASET} -> ${DATA_ROOT}"

CONFIGS=1
if [[ "${1:-}" == "--configs-only" ]]; then
    SELECTED=()
elif [[ $# -gt 0 ]]; then
    SELECTED=("$@")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SELECTED=("${SPECIES_LIST[${SLURM_ARRAY_TASK_ID}]}")
    CONFIGS=0   # parallel tasks would race on the shared config files
else
    SELECTED=("${SPECIES_LIST[@]}")
fi

# make_usage OUT_STEM [converter filter args...]: usage parquet + shift, unless done
make_usage() {
    local stem=$1; shift
    if [[ ! -f "${out_dir}/${stem}.unshifted.parquet" ]]; then
        python -u "${AG_ROOT}/scripts/convert_splice_usage_to_parquet.py" \
            --input-dir "${in_dir}" \
            --output "${out_dir}/${stem}.parquet" \
            --strip-chr-names "$@"
        # Unstranded: shift left intron ends by -1 and assign Labels from intron motifs.
        python -u "${AG_ROOT}/scripts/shift_unstranded_usage.py" \
            --input-dir "${in_dir}" \
            --usage "${out_dir}/${stem}.parquet" \
            --backup "${out_dir}/${stem}.unshifted.parquet" \
            --genome "${genome}"
    fi
}

# make_sites NAME ANNOTATION [usage args...]
make_sites() {
    local name=$1 annotation=$2; shift 2
    if [[ ! -f "${out_dir}/${name}.parquet" ]]; then
        python -u "${AG_ROOT}/scripts/convert_splice_sites_to_parquet.py" \
            --gtf "${out_dir}/${annotation}" \
            --output "${out_dir}/${name}.parquet" "$@"
    fi
}

for species in "${SELECTED[@]}"; do
    in_dir=${SPLISER_DIR}/${species}_${DATASET}
    out_dir=${DATA_ROOT}/${species}
    genome=${GENOME_DIR}/${species}.fa
    mkdir -p "${out_dir}"
    echo "=== ${species}: ${in_dir} -> ${out_dir}"

    for f in gene_annotation.parquet gene_annotation_protein_coding.parquet; do
        [[ -f "${out_dir}/${f}" ]] || cp "${GTF_DIR}/${species}/${f}" "${out_dir}/${f}"
    done

    make_usage usage \
        --min-reproducibility "${MIN_REPRODUCIBILITY}" \
        --min-coverage "${MIN_COVERAGE}"
    make_usage usage.unfiltered

    if [[ ! -f "${out_dir}/junctions.parquet" ]]; then
        python -u "${AG_ROOT}/scripts/convert_splice_junctions_to_parquet.py" \
            --input-dir "${in_dir}" \
            --usage-json "${out_dir}/usage.json" \
            --output "${out_dir}/junctions.parquet" \
            --min-reproducibility "${MIN_REPRODUCIBILITY}" \
            --strip-chr-names \
            --genome "${genome}"
    fi

    make_sites splice_sites_gtf gene_annotation.parquet
    usage_args=(--usage-parquet "${out_dir}/usage.parquet" --min-alpha 5)
    make_sites splice_sites_intersect_protein_coding gene_annotation_protein_coding.parquet \
        "${usage_args[@]}" --usage-mode intersect
    make_sites splice_sites_union_protein_coding gene_annotation_protein_coding.parquet \
        "${usage_args[@]}" --usage-mode union
    make_sites splice_sites_intersect_usage_protein_coding gene_annotation_protein_coding.parquet \
        "${usage_args[@]}" --usage-mode 'intersect+usage'

    python -u "${AG_ROOT}/scripts/check_splice_positions.py" \
        --annotation "${out_dir}/splice_sites_intersect_usage_protein_coding.parquet" \
        --genome "${genome}" \
        --n-sites 5000
done

if [[ "${CONFIGS}" == "1" ]]; then
    # Dataset-level data configs, in the layout of the original data/brawand/data_config_*.json,
    # for every species whose files exist. Paths use "~" like the Brawand configs.
    python - "${DATA_ROOT}" <<'EOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
home = str(Path.home())
NAMES = {"Homo_sapiens": "human", "Pan_troglodytes": "chimpanzee",
         "Pan_paniscus": "bonobo", "Gorilla_gorilla": "gorilla",
         "Pongo_abelii": "orangutan", "Macaca_mulatta": "macaque",
         "Mus_musculus": "mouse"}
folds = "~/sds/sd17d003/Anamaria/borzoi_folds/results_orthologs_132kb_protein_coding"
tilde = lambda p: str(p).replace(home, "~", 1)

for tag in ["intersect_protein_coding", "intersect_usage_protein_coding"]:
    cfg = {}
    for species, name in NAMES.items():
        d = root / species
        if not (d / "usage.json").exists():
            continue
        labels = json.loads((d / "usage.json").read_text())["condition_labels"]
        cfg[name] = {
            "name": name,
            "genome": f"~/sds/sd17d003/Anamaria/genomes/ensembl115/fasta/{species}.fa",
            "gene_annotation": tilde(d / "gene_annotation.parquet"),
            "annotation_parquet": tilde(d / f"splice_sites_{tag}.parquet"),
            "usage_parquet": tilde(d / "usage.parquet"),
            "junction_parquet": tilde(d / "junctions.parquet"),
            "train_bed": f"{folds}/{name}/fold0/train.bed",
            "val_bed": f"{folds}/{name}/fold0/valid.bed",
            "test_bed": f"{folds}/{name}/fold0/test.bed",
            "n_conditions": len(labels),
            "condition_labels": labels,
        }
    out = root / f"data_config_{tag}.json"
    out.write_text(json.dumps(cfg, indent=4) + "\n")
    print(f"Wrote {out} ({', '.join(cfg)})")
EOF
fi
