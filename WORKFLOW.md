```bash
species=Homo_sapiens
gtf_dir=/home/elek/sds/sd17d003/Anamaria/genomes/ensembl115/gtf/
fa_dir=/home/elek/sds/sd17d003/Anamaria/genomes/ensembl115/fasta/

species=Mus_musculus
gtf_dir=/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/
fa_dir=/home/elek/sds/sd17d003/Anamaria/genomes/mazin/fasta/

out_dir=/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/
data_dir=${out_dir}/data
mkdir -p ${data_dir}/${species}
```

# Convert gtf to parquet

```bash
python scripts/convert_gtf_to_parquet.py \
    --input ${gtf_dir}/${species}.gtf.gz \
    --output ${data_dir}/${species}/gene_annotation.parquet > logs/gtf_to_parquet_${species}.log
```

# Prepare splice usage for training

```bash
python scripts/convert_splice_usage_to_parquet.py \
    --input-dir /home/elek/sds/sd17d003/Anamaria/spliser/${species} \
    --output ${data_dir}/${species}/usage.parquet \
    --min-alpha 5 \
    --strip-chr-names > logs/usage_to_parquet_${species}.log
```

# Prepare splice site annotation for training

Save parquet file with splice site annotations for all sies found in either gtf file or in usage file: `--usage-mode union`.

```bash
python scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --min-alpha 100 \
    --usage-mode union \
    --output ${data_dir}/${species}/splice_sites_union.parquet > logs/splice_sites_to_parquet_${species}_union.log
```

Alternativelly, save parquet file with splice site annotations for only those sies found in both gtf file and in usage file: `--usage-mode intersect`.

```bash
python scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --min-alpha 100 \
    --usage-mode intersect \
    --output ${data_dir}/${species}/splice_sites_intersect.parquet > logs/splice_sites_to_parquet_${species}_intersect.log
```

I settle for a version where all usage sites are kept, in addiition to the sites that are present both in the gtf and usage.

```bash
python scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --min-alpha 50 \
    --usage-mode 'intersect+usage' \
    --output ${data_dir}/${species}/splice_sites.parquet > logs/splice_sites_to_parquet_${species}.log
```

# Prepare folds

```bash
if [[ $species == 'Homo_sapiens' ]]; then
  folds=/home/elek/projects/splicing/data/folds/sequences_human_hg19_sorted.bed
  folds=/home/elek/projects/splicing/data/folds/sequences_human.bed
elif [[ $species == 'Mus_musculus' ]]; then
  folds=/home/elek/projects/splicing/data/folds/sequences_mouse.bed
fi

python scripts/convert_borzoi_folds.py \
    --seq-len 131072 \
    --input ${folds} \
    --strip-chr-names \
    --output-dir ${out_dir}/${species}/folds_100kb > logs/convert_borzoi_folds_100kb_${species}.log
```

# Finetune

Finetune heads only, single species

```bash
python scripts/finetune_splice.py --mode linear-probe \
    --genome ${fa_dir}/${species}.fa \
    --annotation-parquet ${data_dir}/${species}/splice_sites.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --train-bed ${data_dir}/${species}/folds/FOLD_0/train.bed \
    --val-bed ${data_dir}/${species}/folds/FOLD_0/valid.bed \
    --pretrained-weights /home/elek/projects/alphagenome_ft_pytorch/checkpoints/model_fold_0.safetensors
```

Finetune model for multiple species

```bash
python scripts/finetune_splice.py --config scripts/configs/finetune_splice_example.yaml
```

# Predict

```bash
pred_dir=${out_dir}/predict/
mkdir -p ${pred_dir}

python scripts/predict_splice_site.py \
    --coords "1:121030623-121225633" \
    --genome ${fa_dir}/${species}.fa \
    --annotation ${out_dir}/${species}/gene_annotation.parquet \
    --organism 0 \
    --checkpoint /home/elek/projects/alphagenome_ft_pytorch/checkpoints/model_fold_0.safetensors \
    --output ${pred_dir}/splice_predictions.tsv
```