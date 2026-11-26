#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Brawand inputs: prepare_unstranded.sh with DATASET=brawand (see that file).
#   slurm/submit.sh --cpu --mem 96G --time 08:00:00 --sbatch-arg --array=0-6 prepare_brawand
#   slurm/submit.sh --cpu --mem 8G --time 00:10:00 prepare_brawand --configs-only
export DATASET=brawand
source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/jobs/prepare_unstranded.sh" "$@"
