#!/bin/bash
#SBATCH --job-name=ft-multigpu
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # torchrun manages worker processes internally
#SBATCH --cpus-per-task=32          # 4 GPUs × 8 workers each
#SBATCH --gres=gpu:4,gpumem_per_gpu:80GB
#SBATCH --mem=200gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
#
# Helix GPU options:
# - 4 × A100 (80 GB): --gres=gpu:A100:4
# - 4 × H200 (141 GB): --gres=gpu:H200:4
#
# To use a specific GPU type uncomment above and remove the generic gres line.

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Load CUDA module before activating conda environment
module load devel/cuda
export OMP_NUM_THREADS=8            # per process (cpus-per-task / n_gpus)

# Fix PyTorch memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL tuning for single-node multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Activate conda environment
conda activate alphagenome_pytorch_genomicsxai

# Verify CUDA setup
# Use conda env's libstdc++ instead of the (older) system /lib64/libstdc++.so.6
# Fixes: GLIBCXX_3.4.29 not found when numpy/torch C-extensions are loaded
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
echo "CUDA setup verification:"
echo "  CUDA_HOME: ${CUDA_HOME}"
python -c "import torch; print(f'  GPUs available: {torch.cuda.device_count()}'); import sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available in PyTorch!"
    exit 1
}

WORK_DIR=${HOME}/projects/alphagenome_ft_pytorch
N_GPUS=4

CONFIG="${WORK_DIR}/configs/finetune_splice_helix_multigpu.yaml"

# Read output_dir and run_name from config YAML
OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('output_dir','').rstrip('/'))")
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('run_name',''))")

# Fallback to timestamp if run_name is empty
if [ -z "$RUN_NAME" ]; then
    RUN_NAME=$(date +%Y%m%d_%H%M%S)
fi

LOG_DIR="${OUTPUT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${LOG_TIMESTAMP}.log"

echo "Starting finetuning at $(date)" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "---" | tee -a "${LOG_FILE}"

torchrun \
    --standalone \
    --nproc_per_node=${N_GPUS} \
    ${WORK_DIR}/scripts/finetune_splice.py \
    --config ${CONFIG} | tee -a "${LOG_FILE}"

echo "---" | tee -a "${LOG_FILE}"
echo "Finetuning completed at $(date). Logs saved to ${LOG_FILE}" | tee -a "${LOG_FILE}"
