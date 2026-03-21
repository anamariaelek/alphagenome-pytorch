!/bin/bash

DIR=/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai

for SPS in Homo_sapiens Mus_musculus 
do
    echo "Evaluating splice sites for ${SPS}"
    python scripts/evaluate_splice_by_source.py \
        --pretrained-weights checkpoints/model_fold_0.safetensors \
        --bed ${DIR}/${SPS}/folds/FOLD_0/valid.bed \
        --annotation ${DIR}/${SPS}/splice_sites.parquet \
        --gtf-annotation ${DIR}/${SPS}/splice_sites_gtf.parquet \
        --usage-parquet ${DIR}/${SPS}/usage.parquet \
        --genome ${DIR}/../genomes/mazin/fasta/${SPS}.fa \
        --output-dir tmp/by_source/${SPS}/ \
        --n-windows 10000 \
        --device cuda
done