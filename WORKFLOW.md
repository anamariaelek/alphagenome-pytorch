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

The usage head is trained with a composite loss (`splice_usage_loss` in
[`splice_losses.py`](../src/alphagenome_pytorch/extensions/finetuning/splice_losses.py)),
combining up to three components, each isolating a different, otherwise-conflatable
kind of error. `usage_loss_weights` in the YAML config supplies each component's
weight; a component with weight `0.0` (or an omitted key) is skipped entirely — its
loss and any per-batch metrics it would log are simply not computed.

- **`bce`** — binary cross-entropy on absolute usage (SSE) in `[0, 1]`, over every
  observed `(position, condition)` pair, no eligibility filter. The baseline "get the
  absolute level right" term — this is what actually calibrates predicted usage
  fraction against the true one; the two trajectory terms below are deliberately
  blind to absolute level.
- **`delta_mse`** — per-(site, tissue) centered MSE: both predicted and true
  trajectories are first mean-centered over their own observed timepoints
  (`_tissue_centered`), then squared error is computed between the two *centered*
  curves. Centering means a constant level offset scores zero error here — this term
  is purely about trajectory *magnitude* (how big the swing is), not where it sits.
  Eligible trajectories are weighted **equally** (not by amplitude), so a handful of
  extreme-swing sites can't dominate a batch's gradient.
- **`trajectory_pearson`** — `1 - PearsonR` per (site, tissue), computed on the same
  centered curves. Pearson r is scale- and shift-invariant, so this term is purely
  about trajectory *direction/shape* — a prediction that's the right shape but at
  half (or double) the true amplitude still scores well here (that failure mode is
  `delta_mse`'s job to catch, not this term's). The correlation denominator is
  regularized by `var_floor` (fixed at `1e-3`, not YAML-configurable) so a flat
  *prediction* (near-zero variance) yields a bounded gradient (`r → 0`) instead of
  the correlation blowing up, rather than being skipped as ineligible.

`delta_mse` and `trajectory_pearson` are both restricted to the same eligibility
gate: a trajectory must have `≥ min_tp` observed timepoints *and* its median-filtered
(`median_win`-point), self-recentered excursion — max deviation from its own
filtered-baseline mean, robust to a 1-2 point noisy outlier — must exceed
`exc_floor`. Current fixed values (`_per_tissue_delta_mse` / `_per_tissue_pearson_loss`
defaults, not exposed in the YAML config): `min_tp=3`, `exc_floor=0.10`,
`median_win=5` — nicknamed "exc5" elsewhere in this codebase (`evaluate_splice.py
--trajectory-corr`, the prediction-clustering notebooks). This excludes flat sites,
noisy wobble, and single-timepoint measurement outliers, so both trajectory terms
only spend gradient on genuinely dynamic developmental sites instead of being
swamped by the (much larger) flat majority — a plain, unfiltered version of either
loss over *all* sites never learns real trajectory behavior for the dynamic
minority.

`usage_traj_warmup_epochs` (integer, default `0` = off) linearly ramps
`delta_mse`/`trajectory_pearson`'s weights from `0` up to their configured value
over the first N epochs — `bce` is always at full weight from epoch 1. The scale
factor is `min(1, max(0, (epoch - 1) / usage_traj_warmup_epochs))`, so it's `0` at
epoch 1 and reaches `1.0` (full configured weight) at epoch
`usage_traj_warmup_epochs + 1`. The point is to let the level fit (`bce`) get
established before the trajectory terms start applying gradient pressure, rather
than all three competing from epoch 1.

`tissue_cond_groups` (built automatically from each species' condition metadata, not
a YAML key) splits a site's conditions by tissue, ordered by developmental
timepoint, so `delta_mse`/`trajectory_pearson` measure *within-tissue* dynamics
rather than between-tissue level offsets. Without it, both terms would collapse to
one all-conditions group per site, dominated by between-tissue level differences
rather than genuine developmental change within a tissue.

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

#### list_prediction_dirs.sh
General lister: scans every model run directory under `DIR` (default
`${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai`) for per-species prediction output
directories — identified by a `metrics.json` inside them, under any `*/preds_*/<species>/`
path, regardless of what the model run directory itself is named, or what follows the
`preds_` prefix (`preds_intersect_protein_coding`, `preds_gtf`, `preds_union`, `preds_usage`,
...) — and writes one tab-separated `(checkpoint, data_config, species, output_dir)` line per
match to `JOBS_FILE` (default `/tmp/prediction_dirs.txt`) — re-derived from each directory's
own `eval.log`, so no hardcoded model list. Lists every matching directory regardless of what's
already been computed inside it; its two consumers decide what to do with each one, and both
always run this script first, so they cover the same set of directories without needing to be
chained manually:

- **`backfill_trajectory_metrics.sh`** — reruns `evaluate_splice.py --skip-predictions
  --trajectory-corr --overwrite` for every directory, (re)computing `trajectory_correlation` /
  `trajectory_magnitude` whether or not that directory had them before.
- **`cluster_predictions.sh`** (Developmental-trajectory clustering, below).

```bash
bash scripts/list_prediction_dirs.sh   # -> /tmp/prediction_dirs.txt
bash scripts/backfill_trajectory_metrics.sh # PARALLEL=8 to tune
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

For example, clustering macaque trajectories for test sites, in brain, cerebellum, liver, and testis:

```bash
PARQUET=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/combined_usage_data_macaque.parquet
DATA_CONFIG=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/data_config.json
REF=${HOME}/sds/sd17d003/Anamaria/gp_splice_usage

for TISSUE in Brain Cerebellum Liver Testis; do
    OUT=${REF}/macaque/${TISSUE}
    mkdir -p ${OUT}
    python scripts/cluster_trajectories.py \
        --parquet-path ${PARQUET} \
        --species macaque \
        --tissue ${TISSUE} \
        --data-config ${DATA_CONFIG} \
        --split test \
        --output ${OUT} \
        --save-plots \
        > ${OUT}/cluster_trajectories.log 2>&1
    echo "Done: macaque ${TISSUE}"
done
```

#### cluster_predictions.py
Assigns *predicted* trajectories to the nearest centroid of an existing reference
(observed) clustering from `cluster_trajectories.py` — never re-clusters predictions
independently — and reports per-site true vs. predicted cluster/shape agreement.

For example, clustering trajectory predictions for multiple organs from multiple species against reference (observed) trajectories:

```bash
REF=${HOME}/sds/sd17d003/Anamaria/gp_splice_usage/
MODEL=lora_64_emb_traj_hm_
MODEL_DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/${MODEL}
PREDS_DIR=${MODEL_DIR}/preds_intersect_protein_coding
USAGE_TEMPLATE=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/combined_usage_data_{species}.parquet
for SPECIES in human mouse; do
    for TISSUE in Brain Cerebellum Liver Testis; do
        OUT=${PREDS_DIR}/${SPECIES}/pred_gp_splice_usage/${TISSUE}
        mkdir -p ${OUT}
        python scripts/cluster_predictions.py \
            --species ${SPECIES} \
            --tissue ${TISSUE} \
            --ref-dir ${REF} \
            --preds-dir ${PREDS_DIR} \
            --usage-template ${USAGE_TEMPLATE} \
            --output ${OUT} > ${OUT}/cluster_predictions.log 2>&1
        echo "Running clustering predictions for ${SPECIES} ${TISSUE}"
    done
done
```

```bash
python scripts/cluster_predictions.py --species human --tissue Brain \
    --ref-dir /path/to/gp_splice_usage --preds-dir /path/to/preds_root \
    --usage-template /path/to/data/combined_usage_data_{species}.parquet \
    --output /path/to/pred_clusters
```

#### cluster_predictions.sh
Automates the `cluster_predictions.py` loop above across every `(model, species)` pair from
`list_prediction_dirs.sh` (see the Evaluate section above), over the 4 tissues with a
reference clustering (Brain/Cerebellum/Liver/Testis — matches
`splice_prediction_clustering.ipynb`'s `TISSUE_LIST`). Writes to
`<preds_dir>/<species>/pred_gp_splice_usage/<tissue>/`, the exact
nested layout the notebook's `pred_clusters_path()` expects. Species with no reference
clustering (e.g. chicken) are skipped automatically; `cluster_predictions.py` itself is
idempotent (skips existing output unless `--overwrite`), so this is safe to rerun any time new
predictions show up — it only fills in what's missing.

```bash
bash scripts/cluster_predictions.sh
# tune parallelism: PARALLEL (concurrent cluster_predictions.py processes) x GP_NJOBS
# (GP-fitting threads within each) -- keep their product under your core count
PARALLEL=10 GP_NJOBS=2 bash scripts/cluster_predictions.sh
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
