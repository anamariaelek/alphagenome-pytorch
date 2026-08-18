Before running anything, make sure to activate the conda environment with all dependencies installed:
```bash
conda activate alphagenome_pytorch_genomicsxai
```

```bash
species=Homo_sapiens
species=Mus_musculus
species=Rattus_norvegicus
species=Oryctolagus_cuniculus
species=Monodelphis_domestica
species=Macaca_mulatta
species=Gallus_gallus

# Use new genome for human and macaque
if [[ "$species" == "Homo_sapiens" ]] || [[ "$species" == "Macaca_mulatta" ]]; then
    gtf_dir=${HOME}/sds/sd17d003/Anamaria/genomes/ensembl115/gtf/
    fa_dir=${HOME}/sds/sd17d003/Anamaria/genomes/ensembl115/fasta/
else
    gtf_dir=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/gtf_ensembl/
    fa_dir=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/fasta/
fi

out_dir=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/
data_dir=${out_dir}/data
mkdir -p ${data_dir}/${species}
```

# Convert gtf to parquet

```bash
python scripts/convert_gtf_to_parquet.py \
    --input ${gtf_dir}/${species}.gtf.gz \
    --output ${data_dir}/${species}/gene_annotation.parquet > logs/gtf_to_parquet_${species}.log
```

Only keep protein-coding genes:

```bash
python scripts/convert_gtf_to_parquet.py \
    --input ${gtf_dir}/${species}.gtf.gz \
    --biotype-filter protein_coding \
    --output ${data_dir}/${species}/gene_annotation_protein_coding.parquet > logs/gtf_to_parquet_${species}_protein_coding.log
```

# Prepare splice usage for training

`convert_splice_usage_to_parquet.py` converts Spliser `.combined.tsv` files to a compact `_usage.parquet` + `_usage.json` (per-site-per-condition SSE/Alpha/Beta, plus condition/class label metadata).

```bash
python -u scripts/convert_splice_usage_to_parquet.py \
    --input-dir ${HOME}/sds/sd17d003/Anamaria/spliser/data/${species} \
    --output ${data_dir}/${species}/usage.parquet \
    --min-reproducibility 0.5 \
    --min-coverage 10 \
    --strip-chr-names > logs/usage_to_parquet_${species}.log
```

# Prepare splice site annotation for training

To save parquet file with splice site annotations, run `convert_splice_sites_to_parquet.py`. It builds a splice-site annotation Parquet from a GTF (donor/acceptor positions per strand), optionally unioned or intersected with sites found in a Spliser usage file.

To save only those sites found in gtf file, you only pass the gtf file as input. Alternatively, you can also keep all sites with usage above a certain threshold, or take the union/intersection of the gtf and usage-based sites.

```bash
python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --output ${data_dir}/${species}/splice_sites_gtf.parquet > logs/splice_sites_to_parquet_${species}_gtf.log

python -u scripts/convert_splice_sites_to_parquet.py \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --usage-mode usage-only \
    --min-alpha 5 \
    --output ${data_dir}/${species}/splice_sites_usage.parquet > logs/splice_sites_to_parquet_${species}_usage.log

python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --usage-mode union \
    --min-alpha 5 \
    --output ${data_dir}/${species}/splice_sites_union.parquet > logs/splice_sites_to_parquet_${species}_union.log

python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --usage-mode intersect \
    --min-alpha 5 \
    --output ${data_dir}/${species}/splice_sites_intersect.parquet > logs/splice_sites_to_parquet_${species}_intersect.log
```

Here all usage sites are kept, in addition to the sites that are present both in the gtf and usage.

```bash
python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --usage-mode 'intersect+usage' \
    --min-alpha 5 \
    --output ${data_dir}/${species}/splice_sites_intersect_usage.parquet > logs/splice_sites_to_parquet_${species}_intersect_usage.log
```

To additionally keep splice sites only for protein-coding genes, use pre-filtered gtf annotation file:

```bash
python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation_protein_coding.parquet \
    --output ${data_dir}/${species}/splice_sites_gtf_protein_coding.parquet > logs/splice_sites_to_parquet_${species}_gtf_protein_coding.log

python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation_protein_coding.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --usage-mode 'intersect' \
    --min-alpha 5 \
    --output ${data_dir}/${species}/splice_sites_intersect_protein_coding.parquet > logs/splice_sites_to_parquet_${species}_intersect_protein_coding.log

python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation_protein_coding.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --usage-mode 'union' \
    --min-alpha 5 \
    --output ${data_dir}/${species}/splice_sites_union_protein_coding.parquet > logs/splice_sites_to_parquet_${species}_union_protein_coding.log

python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation_protein_coding.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --usage-mode 'intersect+usage' \
    --min-alpha 5 \
    --output ${data_dir}/${species}/splice_sites_intersect_usage_protein_coding.parquet > logs/splice_sites_to_parquet_${species}_intersect_usage_protein_coding.log
```

Optionally, `check_splice_positions.py` can be used to sanity-check an annotation Parquet against canonical GT/AG splice motifs in the genome FASTA — a fast way to catch off-by-one coordinate errors before training.

```bash
python scripts/check_splice_positions.py \
    --annotation ${data_dir}/${species}/splice_sites_intersect_usage_protein_coding.parquet \
    --genome ${fa_dir}/${species}.fa.gz \
    --n-sites 5000  > logs/check_splice_positions_${species}_intersect_usage_protein_coding.log
```

# Prepare orthology-based folds

For all species, download orthologs from Ensembl and run the code in `~/sds/sd17d003/Anamaria/borzoi_folds/orthologs.ipynb` to make orthogroup-aware splits for each species.  
For a given target input sequence length `INPUT_SEQ_LEN`, results for each `SPECIES` are saved in: `/home/elek/sds/sd17d003/Anamaria/borzoi_folds/results_orthologs_{INPUT_SEQ_LEN}/{SPECIES}/fold{FOLD}/`.


# Finetune

`finetune_splice.py` trains 5-class splice-site classification (Donor+/Acceptor+/Donor-/Acceptor-/Background) and optionally usage prediction on top of a pretrained AlphaGenome trunk. Supports linear-probe, LoRA, and full fine-tuning modes.

To finetune heads only for single species:

```bash
python -u scripts/finetune_splice.py --mode linear-probe \
    --genome ${fa_dir}/${species}.fa \
    --annotation-parquet ${data_dir}/${species}/splice_sites.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --train-bed ${data_dir}/${species}/folds/FOLD_0/train.bed \
    --val-bed ${data_dir}/${species}/folds/FOLD_0/valid.bed \
    --pretrained-weights ${HOME}/sds/sd17d003/Anamaria/alphagenome_ft_pytorch/checkpoints/model_fold_0.safetensors
```

Finetune model for multiple species:

```bash
python -u scripts/finetune_splice.py --config scripts/configs/finetune_splice_example.yaml
```

`finetune_splice_submit.sh` / `finetune_splice_submit_*.sh` are SLURM submission wrappers around `finetune_splice.py`. The suffix encodes which species are included in that training run (`h`=human, `m`=mouse, `r`=rat, `b`=rabbit, `o`=opossum, `c`=chicken).

```bash
sbatch scripts/finetune_splice_submit_hmrro.sh
```

`finetune_splice_synthetic.py` reproduces the splice fine-tuning GPU workload (forward → loss → backward → optimizer step) with random weights and synthetic batches, for isolating training-loop performance from data loading.

```bash
python scripts/finetune_splice_synthetic.py --hours 6
```

## Classification loss

The classification head uses standard cross-entropy over the 5 classes
(Donor+/Acceptor+/Donor-/Acceptor-/Background) — unchanged from the original
AlphaGenome classification loss. `splice_classification_loss` accepts an optional
`class_weights` tensor for median-frequency class balancing
(`compute_splice_class_weights`), since Background positions typically outnumber
real splice sites ~10,000:1 — but `scripts/finetune_splice.py` never computes or
passes it (`class_weights` stays `None` throughout), so current training runs use
**plain, unweighted** cross-entropy despite that imbalance.

## Usage loss

The usage head is trained with a composite loss, configured via `usage_loss_weights`
in the YAML config:

- `bce`: binary cross-entropy on absolute usage (SSE) — every observed (position,
  condition) pair. The baseline "get the level right" term.
- `delta_mse`: per-(site,tissue) centered MSE — teaches trajectory *magnitude*.
- `trajectory_pearson`: `1 - PearsonR` per (site,tissue) — teaches trajectory
  *direction*, independent of amplitude.

`delta_mse` and `trajectory_pearson` are restricted to sites whose true trajectory
clears an excursion filter (5-point median-filtered deviation from its own baseline,
"exc5") — this excludes flat sites, noisy wobble, and measurement outliers, so the 
trajectory terms only spend gradient on genuinely dynamic developmental sites 
instead of being swamped by the (much larger) flat majority.

`usage_traj_warmup_epochs` linearly ramps `delta_mse`/`trajectory_pearson` from 0 to
their configured weight over the first N epochs (`bce` is always at full weight), so
the level fit is established before the trajectory terms apply full pressure.

`tissue_cond_groups` (built automatically from each species' condition metadata)
splits a site's conditions by tissue, ordered by developmental timepoint, so the
trajectory terms measure *within-tissue* dynamics rather than between-tissue level
offsets.

See `usage_loss_weights` in any `configs/finetune_*_132kb.yaml` for the full
parameter reference and current defaults.


# Evaluate

`evaluate_splice.py` generates classification (AUPRC, binary + per-class) and usage (Pearson r) predictions for one or more species from a checkpoint, writing `predictions_<species>.npz` / `usage_<species>.npz` per species.

```bash
python -u scripts/evaluate_splice.py \
    --checkpoint "${out_dir}" \
    --bed "${data_dir}/${species}/folds/FOLD_0/test.bed" "${data_dir}/${species}/folds/FOLD_0/test.bed" \
    --gtf-sites "${data_dir}/${species}/splice_sites_gtf.parquet" "${data_dir}/${species}/splice_sites_gtf.parquet" \
    --batch-size 4 \
    --device cuda \
    --output-dir "${out_dir}/predictions"
```

Alternativelly, paths to data can be read from config: `--data-config /data/data_config.json`.

`evaluate_splice_submit.sh` is a SLURM wrapper that runs `evaluate_splice.py` for every species in a model's training set in turn, writing each species' output under `<run>/<pred_dir>/<species>/`.

```bash
sbatch scripts/evaluate_splice_submit.sh
```

`predict_splice_site.py` predicts splice-site classification probabilities for a single genomic region — a
lightweight, single-region alternative to `evaluate_splice.py` for spot-checks.

```bash
python scripts/predict_splice_site.py \
    --coords 1:1000:1500 \
    --genome /path/to/genome.fa \
    --annotation /path/to/annotation.parquet \
    --checkpoint /path/to/model.pth
```

### Developmental-trajectory clustering

#### cluster_trajectories.py
Clusters *observed* developmental SSE trajectories (GP smoothing + Ward hierarchical
clustering) and annotates each cluster with a shape label. Command-line driver for the
same pipeline used by `splice_trajectory_clustering.ipynb`; building blocks live in
[`alphagenome_pytorch.clustering`](../src/alphagenome_pytorch/clustering.py).

```bash
python scripts/cluster_trajectories.py \
    --species human --n-clusters 30 \
    --n-jobs 16 --output results/ --save-plots
```

#### cluster_predictions.py
Assigns *predicted* trajectories to the nearest centroid of an existing reference
(observed) clustering from `cluster_trajectories.py` — never re-clusters predictions
independently — and reports per-site true vs. predicted cluster/shape agreement.

For example, clustering trajectories from `<model>` predictions for 4 organs in 5 species against reference (observed) trajectories saved in `<ref>`.

```bash
ref=${HOME}/sds/sd17d003/Anamaria/gp_splice_usage/
model=lora_32_traj_human_mouse_rat_rabbit_opossum_sasse3
model_dir=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/${model}
preds_dir=${model_dir}/preds_intersect_protein_coding
usage_template=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/combined_usage_data_{species}.parquet
for species in human mouse rat rabbit opossum; do
    for tissue in Brain Cerebellum Liver Testis; do
        out=${preds_dir}/${species}/pred_gp_splice_usage/${tissue}
        mkdir -p ${out}
        python scripts/cluster_predictions.py \
            --species ${species} \
            --tissue ${tissue} \
            --ref-dir ${ref} \
            --preds-dir ${preds_dir} \
            --usage-template ${usage_template} \
            --output ${out} > ${out}/cluster_predictions.log 2>&1
        echo "Running clustering predictions for ${species} ${tissue}"
    done
done
```

```bash
python scripts/cluster_predictions.py --species human --tissue Brain \
    --ref-dir /path/to/gp_splice_usage --preds-dir /path/to/preds_root \
    --usage-template /path/to/data/combined_usage_data_{species}.parquet \
    --output /path/to/pred_clusters
```

#### plot_prediction_clusters.py
Heatmap + per-cluster mean-profile plots for predicted trajectories, grouped by the
cluster each was assigned to in `cluster_predictions.py`'s output.

```bash
python scripts/plot_prediction_clusters.py \
    --species human mouse rat rabbit opossum \
    --organ Brain Cerebellum Liver Testis
```

### Cross-species alignment

#### liftover_splice_hal.py
Lifts human splice sites over to other species via a HAL/multiz whole-genome alignment
and matches them to each target species' annotated sites (exact or within a max
distance), feeding the `splice_cross_species_*` notebooks. Configuration (HAL file,
per-species Parquet paths, distance tolerance) is edited at the top of the file rather
than passed as CLI flags.

```bash
python scripts/liftover_splice_hal.py
```
