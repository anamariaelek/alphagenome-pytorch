```bash
species=Homo_sapiens
gtf_dir=${HOME}/sds/sd17d003/Anamaria/genomes/ensembl115/gtf/
fa_dir=${HOME}/sds/sd17d003/Anamaria/genomes/ensembl115/fasta/

species=Mus_musculus
gtf_dir=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/gtf/
fa_dir=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/fasta/

species=Rattus_norvegicus
gtf_dir=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/gtf/
fa_dir=${HOME}/sds/sd17d003/Anamaria/genomes/mazin/fasta/

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

# Prepare splice usage for training

```bash
python -u scripts/convert_splice_usage_to_parquet.py \
    --input-dir ${HOME}/sds/sd17d003/Anamaria/spliser/${species} \
    --output ${data_dir}/${species}/usage.parquet \
    --min-alpha 5 \
    --strip-chr-names > logs/usage_to_parquet_${species}.log
```

# Prepare splice site annotation for training

Save parquet file with splice site annotations. 

To save only those sies found in gtf file:

```bash
python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --output ${data_dir}/${species}/splice_sites_gtf.parquet > logs/splice_sites_to_parquet_${species}_gtf.log
```


To save all sites found in either gtf file or in usage file: `--usage-mode union`.

```bash
python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --min-alpha 100 \
    --usage-mode union \
    --output ${data_dir}/${species}/splice_sites_union.parquet > logs/splice_sites_to_parquet_${species}_union.log
```

Alternativelly, to save annotations for only those sies found in both gtf file and in usage file: `--usage-mode intersect`.

```bash
python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --min-alpha 100 \
    --usage-mode intersect \
    --output ${data_dir}/${species}/splice_sites_intersect.parquet > logs/splice_sites_to_parquet_${species}_intersect.log
```

I settle for a version where all usage sites are kept, in addiition to the sites that are present both in the gtf and usage.

```bash
python -u scripts/convert_splice_sites_to_parquet.py \
    --gtf ${data_dir}/${species}/gene_annotation.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --min-alpha 50 \
    --usage-mode 'intersect+usage' \
    --output ${data_dir}/${species}/splice_sites.parquet > logs/splice_sites_to_parquet_${species}.log
```

# Prepare folds

```bash
if [[ $species == 'Homo_sapiens' ]]; then
  folds=${HOME}/sds/sd17d003/Anamaria/borzoi_folds/sequences_human_hg19_sorted.bed
  folds=${HOME}/sds/sd17d003/Anamaria/borzoi_folds/sequences_human.bed.gz
  organism="human"
elif [[ $species == 'Mus_musculus' ]]; then
  folds=${HOME}/sds/sd17d003/Anamaria/borzoi_folds/sequences_mouse.bed.gz
  organism="mouse"
elif [[ $species == 'Rattus_norvegicus' ]]; then
  folds=${HOME}/sds/sd17d003/Anamaria/borzoi_folds/fold_assignments/rn5_folds.bed
  organism="rat"
fi

# Prepare folds
python -u scripts/convert_borzoi_folds.py \
    --seq-len 131072 \
    --input ${folds} \
    --organism ${organism} \
    --output-dir ${out_dir}/data/${species}/folds_100kb \
    --resolve-overlap-conflicts \
    --strip-chr-names > logs/convert_borzoi_folds_100kb_${species}.log

# Subset folds with 50% overlap
mkdir -p ${out_dir}/data/${species}/folds_100kb/FOLD_0_subset
for fn in train test valid
do
    IN_BED=${out_dir}/data/${species}/folds_100kb/FOLD_0/${fn}.bed
    OUT_BED=${out_dir}/data/${species}/folds_100kb/FOLD_0_subset/${fn}.bed
    python scripts/subset_nonoverlapping_bed.py $IN_BED 50 > $OUT_BED
done
```

# Finetune

Finetune heads only, single species

```bash
python -u scripts/finetune_splice.py --mode linear-probe \
    --genome ${fa_dir}/${species}.fa \
    --annotation-parquet ${data_dir}/${species}/splice_sites.parquet \
    --usage-parquet ${data_dir}/${species}/usage.parquet \
    --train-bed ${data_dir}/${species}/folds/FOLD_0/train.bed \
    --val-bed ${data_dir}/${species}/folds/FOLD_0/valid.bed \
    --pretrained-weights ${HOME}/sds/sd17d003/Anamaria/alphagenome_ft_pytorch/checkpoints/model_fold_0.safetensors
```

Finetune model for multiple species

```bash
python -u scripts/finetune_splice.py --config scripts/configs/finetune_splice_example.yaml
```

# Evaluate

```bash
python -u scripts/evaluate_splice.py \
    --checkpoint "${out_dir}" \
    --bed "${data_dir}/${species}/folds/FOLD_0/test.bed" "${data_dir}/${species}/folds/FOLD_0/test.bed" \
    --gtf-sites "${data_dir}/${species}/splice_sites_gtf.parquet" "${data_dir}/${species}/splice_sites_gtf.parquet" \
    --batch-size 4 \
    --device cuda \
    --output-dir "${out_dir}/predictions"
```