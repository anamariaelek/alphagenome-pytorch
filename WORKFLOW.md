```bash
species=Mus_musculus
gtf_dir=/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/
fa_dir=/home/elek/sds/sd17d003/Anamaria/genomes/mazin/fasta/
out_dir=/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/
```

# Convert gtf to parquet

```bash
python scripts/convert_gtf_to_parquet.py \
    --input ${gtf_dir}/${species}.gtf.gz \
    --output ${out_dir}/${species}/gene_annotation.parquet > logs/gtf_to_parquet_${species}.log
```

# Prepare splice usage for training

```bash
python scripts/convert_splice_usage_to_parquet.py \
    --input-dir /home/elek/sds/sd17d003/Anamaria/spliser_for_splicevo/${species} \
    --output ${out_dir}/${species}/usage.parquet > logs/usage_to_parquet_${species}.log
```

# Prepare splice site annotation for training

```bash
python scripts/convert_splice_sites_to_parquet.py \
    --gtf ${gtf_dir}/${species}.gtf.gz \
    --usage-parquet ${out_dir}/${species}/usage.parquet \
    --min-coverage 10 \
    --usage-mode union \
    --output ${out_dir}/${species}/splice_sites.parquet > logs/splice_sites_to_parquet_${species}.log
```

```bash
python scripts/convert_splice_sites_to_parquet.py \
    --gtf ${gtf_dir}/${species}.gtf.gz \
    --usage-parquet ${out_dir}/${species}/usage.parquet \
    --min-coverage 10 \
    --usage-mode intersect \
    --output ${out_dir}/${species}/splice_sites_intersect.parquet > logs/splice_sites_to_parquet_${species}_intersect.log
```

# Prepare folds

```bash
if [[ $species == 'Homo_sapiens' ]]; then
  folds=/home/elek/projects/splicing/data/liftover/sequences_human_hg19_sorted.bed
elif [[ $species == 'Mus_musculus' ]]; then
  folds=/home/elek/projects/splicing/data/liftover/sequences_mouse.bed
fi

python scripts/convert_borzoi_folds.py \
    --input ${folds} \
    --output-dir ${out_dir}/${species}/folds > logs/convert_borzoi_folds_${species}.log
```

# Finetune

Finetune heads only, single species

```bash
python scripts/finetune_splice.py --mode linear-probe \
    --genome ${fa_dir}/${species}.fa \
    --annotation-parquet ${out_dir}/${species}/splice_sites.parquet \
    --usage-parquet ${out_dir}/${species}/usage.parquet \
    --train-bed ${out_dir}/${species}/folds/FOLD_0/train.bed \
    --val-bed ${out_dir}/${species}/folds/FOLD_0/valid.bed \
    --pretrained-weights /home/elek/projects/alphagenome_ft_pytorch/checkpoints/model_fold_0.safetensors
```

Finetune model for multiple species

```bash
python scripts/finetune_splice.py --config scripts/configs/finetune_splice_example.yaml
```