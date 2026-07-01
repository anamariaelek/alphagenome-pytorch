```bash
species=Homo_sapiens
species=Mus_musculus
species=Rattus_norvegicus
species=Oryctolagus_cuniculus
species=Monodelphis_domestica
species=Macaca_mulatta
species=Gallus_gallus

if [[ "$species" == "Homo_sapiens" ]]; then
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

```bash
python -u scripts/convert_splice_usage_to_parquet.py \
    --input-dir ${HOME}/sds/sd17d003/Anamaria/spliser/data/${species} \
    --output ${data_dir}/${species}/usage.parquet \
    --min-reproducibility 0.5 \
    --min-coverage 10 \
    --strip-chr-names > logs/usage_to_parquet_${species}.log
```

# Prepare splice site annotation for training

Save parquet file with splice site annotations. 

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

# Prepare orthology-based folds

For all species, download orthologs from Ensembl and run the code in `~/sds/sd17d003/Anamaria/borzoi_folds/orthologs.ipynb` to make orthogroup-aware splits for each species.  
For a given target input sequence length `INPUT_SEQ_LEN`, results for each `SPECIES` are saved in: `/home/elek/sds/sd17d003/Anamaria/borzoi_folds/results_orthologs_{INPUT_SEQ_LEN}/{SPECIES}/fold{FOLD}/`.


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