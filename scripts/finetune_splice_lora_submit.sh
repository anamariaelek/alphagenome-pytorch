#!/bin/bash
#SBATCH --job-name=ft-lora
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gpumem_per_gpu:80GB
#SBATCH --mem=120gb
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

# Set OpenMP threads (fallback to 8 if not set by SLURM)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

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
CONFIG="${WORK_DIR}/scripts/configs/finetune_splice_lora_helix.yaml"

# Verify config file exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p ${WORK_DIR}/logs

# Log file
LOG_FILE="${WORK_DIR}/logs/finetune_splice_${TIMESTAMP}.log"

echo "Starting finetuning at $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "---" | tee -a "${LOG_FILE}"

# Run training (both stdout and stderr captured, plus displayed in real-time)
python ${WORK_DIR}/scripts/finetune_splice.py --config ${CONFIG} 2>&1 | tee -a "${LOG_FILE}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "---" | tee -a "${LOG_FILE}"
echo "Finetuning completed at ${TIMESTAMP}. Logs saved to ${LOG_FILE}" | tee -a "${LOG_FILE}"