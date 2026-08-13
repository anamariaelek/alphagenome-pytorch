# Scripts

Utility scripts for AlphaGenome-PyTorch model conversion, training, and validation,
plus a dedicated [Splicing](#splicing) section for the splice-site pipeline.

## Contents

- [Weight Conversion](#weight-conversion)
- [Fine-tuning](#fine-tuning)
- [Data Preprocessing](#data-preprocessing)
- [Validation & Benchmarking](#validation--benchmarking)
- [Demos](#demos)
- [Splicing](#splicing)
  - [Data preparation](#data-preparation)
  - [Fine-tuning](#fine-tuning-2)
  - [Evaluation](#evaluation)
  - [Developmental-trajectory clustering](#developmental-trajectory-clustering)
  - [Cross-species alignment](#cross-species-alignment)

## Weight Conversion

### convert_weights.py
Convert JAX AlphaGenome checkpoint to PyTorch format. Bundles track_means automatically.

```bash
python scripts/convert_weights.py /path/to/jax/checkpoint --output model.pth
```

### extract_track_means.py
Extract track_means from JAX model metadata (requires JAX dependencies).

```bash
python scripts/extract_track_means.py --output track_means.pt
```

### extract_track_metadata.py
Extract full track metadata (names, ontology, tissue info) from JAX model.

```bash
python scripts/extract_track_metadata.py --output-file track_metadata.parquet
```

### validate_weight_mapping.py
Audit all parameter mappings between JAX and PyTorch models.

```bash
python scripts/validate_weight_mapping.py --jax-checkpoint /path/to/checkpoint
python scripts/validate_weight_mapping.py --jax-checkpoint /path/to/checkpoint --verbose
```

## Fine-tuning

### finetune.py
Unified training script supporting linear probing, LoRA, and full fine-tuning.

```bash
# Linear probing
python scripts/finetune.py --mode linear-probe \
    --genome hg38.fa \
    --modality atac --bigwig *.bw \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights model.pth

# LoRA fine-tuning
python scripts/finetune.py --mode lora \
    --lora-rank 8 --lora-alpha 16 \
    --genome hg38.fa \
    --modality atac --bigwig *.bw \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights model.pth

# Multi-GPU with DDP
torchrun --nproc_per_node=4 scripts/finetune.py --mode lora ...
```

## Data Preprocessing

### convert_bigwigs_to_mmap.py
Convert BigWig files to memory-mapped numpy arrays for fast random access.

```bash
# Single file
python scripts/convert_bigwigs_to_mmap.py \
    --bigwig signal.bw \
    --output-dir mmap_signals/signal

# Multiple files in parallel
python scripts/convert_bigwigs_to_mmap.py \
    --bigwig *.bw \
    --output-dir mmap_signals/ \
    --workers 8
```

### convert_borzoi_folds.py
Convert Borzoi sequence folds (~196kb) to AlphaGenome format (1Mb regions).

```bash
python scripts/convert_borzoi_folds.py \
    --input sequences_human.bed.gz \
    --output-dir data/alphagenome_folds
```

### convert_gtf_to_parquet.py
Convert GTF annotation files to Parquet for fast loading.

```bash
python scripts/convert_gtf_to_parquet.py \
    --input gencode.v49.annotation.gtf \
    --output gencode.v49.parquet
```

### preprocess_polya.py
Convert GENCODE polyA metadata to Parquet with gene_id linking.

```bash
python scripts/preprocess_polya.py \
    --metadata gencode.v46.metadata.PolyA_feature \
    --gtf gencode.v46.annotation.parquet \
    --output gencode.v46.polyAs.linked.parquet
```

## Validation & Benchmarking

### verify_model.py
Verify model loading and run a basic forward pass.

```bash
python scripts/verify_model.py
```

### compare_models.py
Compare JAX and PyTorch model outputs for numerical validation.

```bash
python scripts/compare_models.py /path/to/jax/checkpoint \
    --torch_weights model.pth
```

### simple_compare.py
Simplified JAX vs PyTorch comparison script.

```bash
python scripts/simple_compare.py /path/to/jax/checkpoint
```

### benchmark_performance.py
Benchmark model inference performance (timing, memory, GPU profiling).

```bash
python scripts/benchmark_performance.py --weights model.pth
```

## Demos

### demo_manual_extraction.py
Demonstrates manual sequence extraction and variant handling with JAX model.

```bash
python scripts/demo_manual_extraction.py /path/to/jax/checkpoint
```

## Splicing

Scripts for the splice-site classification/usage pipeline: preparing data, fine-tuning,
generating and evaluating predictions, clustering developmental trajectories, and
cross-species lift-over. See [`examples/notebooks/README.md`](../examples/notebooks/README.md)
for the notebooks that consume these scripts' outputs, and
[`src/alphagenome_pytorch/evaluation/splicing.py`](../src/alphagenome_pytorch/evaluation/splicing.py) /
[`src/alphagenome_pytorch/clustering.py`](../src/alphagenome_pytorch/clustering.py) for the
shared loading/metric/clustering code these scripts and notebooks both import.

### Data preparation

#### convert_splice_sites_to_parquet.py
Build a splice-site annotation Parquet from a GTF (donor/acceptor positions per strand),
optionally unioned or intersected with sites found in a Spliser usage file.

```bash
python scripts/convert_splice_sites_to_parquet.py \
    --gtf gencode.v47.annotation.gtf \
    --usage-parquet /path/to/spliser_output/_usage.parquet \
    --min-coverage 10 \
    --output splice_sites.parquet
```

#### convert_splice_usage_to_parquet.py
Convert Spliser `.combined.tsv` files to a compact `_usage.parquet` + `_usage.json`
(per-site-per-condition SSE/Alpha/Beta, plus condition/class label metadata).

```bash
python scripts/convert_splice_usage_to_parquet.py \
    --input-dir /path/to/spliser/Homo_sapiens/ \
    --output /path/to/spliser/Homo_sapiens/_usage.parquet
```

#### check_splice_positions.py
Sanity-checks an annotation Parquet against canonical GT/AG splice motifs in the genome
FASTA — a fast way to catch off-by-one coordinate errors before training.

```bash
python scripts/check_splice_positions.py \
    --annotation splice_sites.parquet \
    --genome hg38.fa \
    --n-sites 5000
```

### Fine-tuning

#### finetune_splice.py
Trains 5-class splice-site classification (Donor+/Acceptor+/Donor-/Acceptor-/Background)
and optionally usage prediction on top of a pretrained AlphaGenome trunk. Supports
linear-probe, LoRA, and full fine-tuning modes.

```bash
python scripts/finetune_splice.py --mode lora \
    --genome hg38.fa \
    --annotation-parquet splice_annotation.parquet \
    --train-bed train.bed --val-bed val.bed \
    --pretrained-weights model.pth
```

#### finetune_splice_submit.sh / finetune_splice_submit_*.sh
SLURM submission wrappers around `finetune_splice.py`. The suffix encodes which species
are included in that training run (`h`=human, `m`=mouse, `r`=rat, `b`=rabbit, `o`=opossum, `c`=chicken), e.g. `finetune_splice_submit_hmrboc.sh` trains on
all six species; `finetune_splice_submit.sh` is the base/no-suffix variant.

```bash
sbatch scripts/finetune_splice_submit_hmrro.sh
```

#### finetune_full_splice_submit_hmrro.sh
Same idea but for full (non-LoRA) fine-tuning, human+mouse+rat+rabbit+opossum.

```bash
sbatch scripts/finetune_full_splice_submit_hmrro.sh
```

#### finetune_splice_synthetic.py
Reproduces the splice fine-tuning GPU workload (forward → loss → backward → optimizer
step) with random weights and synthetic batches, for isolating training-loop performance
from data loading.

```bash
python scripts/finetune_splice_synthetic.py --hours 6
```

### Evaluation

#### evaluate_splice.py
Generates classification (AUPRC, binary + per-class) and usage (Pearson r) predictions
for one or more species from a checkpoint, writing `predictions_<species>.npz` /
`usage_<species>.npz` per species.

```bash
python scripts/evaluate_splice.py \
    --checkpoint /path/to/132kb_human_lora \
    --data-config /data/data_config.json \
    --eval-species mouse rat \
    --output-dir /results/predictions \
    --device cuda
```

#### evaluate_splice_submit.sh
SLURM wrapper that runs `evaluate_splice.py` for every species in a model's training set
in turn, writing each species' output under `<run>/<pred_dir>/<species>/`.

```bash
sbatch scripts/evaluate_splice_submit.sh
```

#### predict_splice_site.py
Predicts splice-site classification probabilities for a single genomic region — a
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

```bash
python scripts/cluster_predictions.py --species human --tissue Brain \
    --ref-dir /path/to/gp_splice_usage --output /path/to/pred_clusters
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
