#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Khrameeva inputs: prepare_unstranded.sh with DATASET=khrameeva (see that file).
#   slurm/submit.sh --cpu --mem 96G --time 08:00:00 --sbatch-arg --array=0-3 prepare_khrameeva
export DATASET=khrameeva
source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/jobs/prepare_unstranded.sh" "$@"
