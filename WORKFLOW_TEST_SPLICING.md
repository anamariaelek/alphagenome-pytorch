# Testing Fine-Tuned Splice Model on Brawand et al. Dataset

This workflow describes how to prepare and evaluate fine-tuned splice-site classification and usage-prediction models on the new Brawand et al. multi-species developmental dataset.

## Dataset Overview

The Brawand dataset includes:
- **Known species** (seen during finetuning): `Homo_sapiens`, `Mus_musculus`, `Macaca_mulatta`
- **New species**: `Pan_troglodytes` (chimpanzee), `Pan_paniscus` (bonobo), `Gorilla_gorilla` (gorilla), `Pongo_abelii` (orangutan, *not yet processed*)

All genomes/annotations are from Ensembl 115.

Before running anything, activate the conda environment:

```bash
conda activate alphagenome_pytorch_genomicsxai
```

## Set Up Paths

```bash
# Data sources (Ensembl 115)
gtf_dir=${HOME}/sds/sd17d003/Anamaria/genomes/ensembl115/gtf/
fa_dir=${HOME}/sds/sd17d003/Anamaria/genomes/ensembl115/fasta/

# Spliser Brawand data
spliser_brawand_dir=${HOME}/sds/sd17d003/Anamaria/spliser/data/

# Output directory for test data
test_out_dir=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/brawand

# Predictions/results directory
pred_out_dir=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/predictions/brawand

mkdir -p ${test_out_dir}
mkdir -p ${pred_out_dir}
mkdir -p logs/
```

## Species List

```bash
# Note: Pongo abelii is excluded until processing is complete
species_list=(
    "Homo_sapiens"
    "Mus_musculus"
    "Macaca_mulatta"
    "Pan_troglodytes"
    "Pan_paniscus"
    "Gorilla_gorilla"
)
```

## Data Preparation

### 1. Convert GTF to Parquet

For each species, convert the Ensembl 115 GTF to a compact parquet annotation.

```bash
for species in "${species_list[@]}"; do
    mkdir -p ${test_out_dir}/${species}
    python scripts/convert_gtf_to_parquet.py \
        --input ${gtf_dir}/${species}.gtf.gz \
        --output ${test_out_dir}/${species}/gene_annotation.parquet \
        > logs/gtf_to_parquet_brawand_${species}.log 2>&1
    echo "Converted GTF: ${species}"
done
```

Optionally, keep only protein-coding genes (matches finetuning strategy):

```bash
for species in "${species_list[@]}"; do
    python scripts/convert_gtf_to_parquet.py \
        --input ${gtf_dir}/${species}.gtf.gz \
        --biotype-filter protein_coding \
        --output ${test_out_dir}/${species}/gene_annotation_protein_coding.parquet \
        > logs/gtf_to_parquet_brawand_${species}_protein_coding.log 2>&1
    echo "Converted GTF (protein-coding): ${species}"
done
```

### 2. Prepare Splice Usage Data

Convert Spliser `.combined.tsv` files from the Brawand data directory to compact parquet format.

```bash
for species in "${species_list[@]}"; do
    python -u scripts/convert_splice_usage_to_parquet.py \
        --input-dir ${spliser_brawand_dir}/${species}_brawand \
        --output ${test_out_dir}/${species}/usage.parquet \
        --min-coverage 10 \
        --min-reproducibility 0.5 \
        --strip-chr-names \
        > logs/usage_to_parquet_brawand_${species}.log 2>&1
    echo "Prepared splice usage: ${species}"
done
```

### 3. Prepare Splice Site Annotations

Build splice-site annotation parquets from GTF, Spliser usage data, or both. The choice depends on whether you want:
- **GTF-only sites**: Canonical splice junctions from annotation
- **Usage-only sites**: All sites observed in the developmental data
- **Union/Intersect**: Combination of both sources

For testing, we recommend using the same configuration used during finetuning. If finetuning used `splice_sites_intersect_protein_coding.parquet`, use the same here for consistency:

```bash
# Option A: GTF-based (canonical sites only)
for species in "${species_list[@]}"; do
    python -u scripts/convert_splice_sites_to_parquet.py \
        --gtf ${test_out_dir}/${species}/gene_annotation.parquet \
        --output ${test_out_dir}/${species}/splice_sites_gtf.parquet \
        > logs/splice_sites_brawand_${species}_gtf.log 2>&1
done

# Option B: GTF-based, protein-coding only
for species in "${species_list[@]}"; do
    python -u scripts/convert_splice_sites_to_parquet.py \
        --gtf ${test_out_dir}/${species}/gene_annotation_protein_coding.parquet \
        --output ${test_out_dir}/${species}/splice_sites_gtf_protein_coding.parquet \
        > logs/splice_sites_brawand_${species}_gtf_protein_coding.log 2>&1
done

# Option C: Union of GTF and usage sites, protein-coding genes
for species in "${species_list[@]}"; do
    python -u scripts/convert_splice_sites_to_parquet.py \
        --gtf ${test_out_dir}/${species}/gene_annotation_protein_coding.parquet \
        --usage-parquet ${test_out_dir}/${species}/usage.parquet \
        --usage-mode union \
        --min-alpha 5 \
        --output ${test_out_dir}/${species}/splice_sites_union_protein_coding.parquet \
        > logs/splice_sites_brawand_${species}_union_protein_coding.log 2>&1
done

# Option D: Intersection of GTF and usage sites, protein-coding genes (
for species in "${species_list[@]}"; do
    python -u scripts/convert_splice_sites_to_parquet.py \
        --gtf ${test_out_dir}/${species}/gene_annotation_protein_coding.parquet \
        --usage-parquet ${test_out_dir}/${species}/usage.parquet \
        --usage-mode intersect \
        --min-alpha 5 \
        --output ${test_out_dir}/${species}/splice_sites_intersect_protein_coding.parquet \
        > logs/splice_sites_brawand_${species}_intersect_protein_coding.log 2>&1
done

# Option E: Intersection + all usage sites, protein-coding genes
for species in "${species_list[@]}"; do
    python -u scripts/convert_splice_sites_to_parquet.py \
        --gtf ${test_out_dir}/${species}/gene_annotation_protein_coding.parquet \
        --usage-parquet ${test_out_dir}/${species}/usage.parquet \
        --usage-mode 'intersect+usage' \
        --min-alpha 5 \
        --output ${test_out_dir}/${species}/splice_sites_intersect_usage_protein_coding.parquet \
        > logs/splice_sites_brawand_${species}_intersect_usage_protein_coding.log 2>&1
done
```

### 4. Sanity-Check Splice Positions

Optionally verify that splice-site coordinates match canonical GT/AG motifs in the genome:

```bash
# Pick one annotation variant (e.g., intersect_usage_protein_coding)
for species in "${species_list[@]}"; do
    python scripts/check_splice_positions.py \
        --annotation ${test_out_dir}/${species}/splice_sites_intersect_usage_protein_coding.parquet \
        --genome ${fa_dir}/${species}.fa.gz \
        --n-sites 5000 \
        > logs/check_splice_positions_brawand_${species}.log 2>&1
    echo "Verified splice positions: ${species}"
done
```

## Prepare Test Folds

Since Brawand is a new dataset, you need to define train/validation/test splits. 
If the goal is to generate predictions on the entire dataset without train/valid splits:

```bash
for species in "${species_list[@]}"; do
    mkdir -p ${test_out_dir}/${species}/folds/FOLD_0
    
    # Create a BED file with all genomic regions covered by usage data
    # This is a placeholder; adjust based on your data structure
    # Example: extract all intervals from the parquet
    python -c "
import pandas as pd
df = pd.read_parquet('${test_out_dir}/${species}/usage.parquet')
# Assume 'chr', 'start', 'end', 'strand' columns exist
regions = df[['chr', 'start', 'end', 'strand']].drop_duplicates().sort_values(['chr', 'start'])
regions.to_csv('${test_out_dir}/${species}/folds/FOLD_0/test.bed', sep='\t', header=False, index=False)
    "
    echo "Created test.bed: ${species}"
done
```

Alternatively, if you generate a reference fold structure using scripts in `/home/elek/sds/sd17d003/Anamaria/fold_splits`.

## Generate Predictions

Run predictions for a single species:

```bash
species="Homo_sapiens"

python -u scripts/evaluate_splice.py \
    --checkpoint "/path/to/finetuned/model" \
    --bed "${test_out_dir}/${species}/folds/FOLD_0/test.bed" \
    --gtf-sites "${test_out_dir}/${species}/splice_sites_intersect_protein_coding.parquet" \
    --batch-size 4 \
    --device cuda \
    --output-dir "${pred_out_dir}/${species}"
```

Create a YAML config file for multi-species evaluation:

```yaml
# config_test_brawand.yaml
species:
  - Homo_sapiens
  - Mus_musculus
  - Macaca_mulatta
  - Pan_troglodytes
  - Pan_paniscus
  - Gorilla_gorilla

checkpoint: /path/to/finetuned/model

data_config:
  test_out_dir: ${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data_brawand
  
output_dir: ${pred_out_dir}

batch_size: 4
device: cuda
```

Then run a wrapper script `scripts/evaluate_splice_brawand_submit.sh` to evaluate predictions for all species.


## Post-Prediction Analysis

### 1. Trajectory Clustering (Optional)

If the finetuned model includes a usage head, cluster predicted developmental trajectories:

```bash
# First, extract reference trajectories from Brawand usage data
for species in "${species_list[@]}"; do
    OUT=${HOME}/sds/sd17d003/Anamaria/gp_splice_usage_brawand/${species}
    mkdir -p ${OUT}
    
    python scripts/cluster_trajectories.py \
        --parquet-path ${test_out_dir}/${species}/usage.parquet \
        --species ${species} \
        --output ${OUT} \
        --save-plots \
        > ${OUT}/cluster_trajectories.log 2>&1
    echo "Clustered observed trajectories: ${species}"
done
```

Then cluster predictions against these reference clusters:

```bash
REF=${HOME}/sds/sd17d003/Anamaria/gp_splice_usage_brawand/
USAGE_TEMPLATE=${test_out_dir}/{species}/usage.parquet

for species in "${species_list[@]}"; do
    for TISSUE in Brain Cerebellum Liver Testis; do
        OUT=${pred_out_dir}/${species}/pred_gp_splice_usage/${TISSUE}
        mkdir -p ${OUT}
        
        python scripts/cluster_predictions.py \
            --species ${species} \
            --tissue ${TISSUE} \
            --ref-dir ${REF} \
            --preds-dir ${pred_out_dir} \
            --usage-template ${USAGE_TEMPLATE} \
            --output ${OUT} \
            > ${OUT}/cluster_predictions.log 2>&1
        echo "Clustered predictions: ${species} ${TISSUE}"
    done
done
```

### 2. Cross-Species Comparison

Compare predictions across species (e.g., human → primate lineage):

```bash
# Liftover human splice sites to other primates via HAL/multiz alignment
python scripts/liftover_splice_hal.py \
    > logs/liftover_splice_brawand.log 2>&1
```

Then compute concordance/conservation metrics across the primate species.

### 3. Summary Metrics

Generate a summary report comparing known vs. novel species predictions:

```bash
python -c "
import os
import json
import pandas as pd

results = {}
for species in ['Homo_sapiens', 'Mus_musculus', 'Macaca_mulatta', 'Pan_troglodytes', 'Pan_paniscus', 'Gorilla_gorilla']:
    metrics_file = os.path.join('${pred_out_dir}', species, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            results[species] = json.load(f)

# Print summary
for species, metrics in results.items():
    print(f'{species}:')
    for key, val in metrics.items():
        print(f'  {key}: {val}')
    print()
"
```

## Notes

### Known vs. Novel Species

- **Known species** (Homo_sapiens, Mus_musculus, Macaca_mulatta) should produce predictions consistent with finetuning-set validation metrics, since the model has seen these organisms during training.
- **Novel species** (Pan_troglodytes, Pan_paniscus, Gorilla_gorilla) test out-of-distribution generalization. Performance may be lower, but the model can still produce meaningful predictions via sequence homology to human/macaque.

### Protein-Coding Filter

The finetuning was likely done on protein-coding genes only (see `WORKFLOW_FINETUNING_SPLICING.md`). For consistency, prepare test data the same way (`splice_sites_intersect_protein_coding.parquet` or similar).

### Missing Data: Orangutan (Pongo abelii)

Orangutan is excluded from this workflow until the genome/annotation processing is complete. Once available:

1. Download Ensembl 115 GTF and FASTA for `Pongo_abelii`.
2. Run GTF→parquet conversion.
3. Prepare splice usage from Spliser Brawand data (if available).
4. Generate predictions following the same steps as above.

## Troubleshooting

### No Usage Data for a Species

If a species has no Spliser Brawand data, `convert_splice_usage_to_parquet.py` will fail or produce an empty parquet. In that case:
- Use GTF-only splice sites: `--usage-mode` is not needed.
- Generate predictions without usage trajectories (classification head only).

### Coordinate Mismatches

If splice sites don't align to GT/AG motifs (check via `check_splice_positions.py`):
- Verify GTF and FASTA are from the same Ensembl release (both 115).
- Check for strand/coordinate convention mismatches (off-by-one errors are common).

### Memory Issues During Evaluation

If the model or batch size causes OOM errors:
- Reduce `--batch-size` (default 4).
- Reduce `INPUT_SEQ_LEN` if evaluation script supports it.
- Use a GPU with more memory or run on CPU (slower).

## References

- `WORKFLOW_FINETUNING_SPLICING.md` – Original finetuning pipeline (data prep, training, evaluation).
- `scripts/evaluate_splice.py` – Main evaluation script.
- `scripts/cluster_trajectories.py` – Reference trajectory clustering.
- `scripts/cluster_predictions.py` – Prediction clustering against reference.
