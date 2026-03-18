#!/bin/bash
#SBATCH --job-name=ft
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gpumem_per_gpu:80GB
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
# 
# Helix GPU options:
# - A40 (48 GB):   --gres=gpu:A40:1
# - A100 (40 GB):  --gres=gpu:A100:1
# - A100 (80 GB):  --gres=gpu:A100:1
# - H200 (141 GB): --gres=gpu:H200:1

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Load CUDA module before activating conda environment
module load devel/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Fix PyTorch memory fragmentation (reduces reserved-but-unallocated memory)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
conda activate alphagenome_pytorch_genomicsxai

# Verify CUDA setup
echo "CUDA setup verification:"
echo "  CUDA_HOME: ${CUDA_HOME}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Exit if CUDA is not available
python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available in PyTorch!"
    exit 1
}

# Create a timestamp for unique log file names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Work directory
WORK_DIR=${HOME}/projects/alphagenome_ft_pytorch/

# Config file
CONFIG="${WORK_DIR}/scripts/configs/finetune_splice_helix.yaml"
python ${WORK_DIR}/scripts/finetune_splice.py --config ${CONFIG} > ${WORK_DIR}/logs/finetune_splice_${TIMESTAMP}.log 2>&1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Finetuning completed at ${TIMESTAMP}. Logs saved to ${WORK_DIR}/logs/finetune_splice_${TIMESTAMP}.log"