#!/bin/bash
#SBATCH --job-name=ft-hqm-phase1-pilot
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,gpumem_per_gpu:40GB
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
#
# Small pilot run (human/mouse/macaque, 1 epoch) comparing the Phase 1
# retraining-objective changes (coverage-weighted per-replicate BCE +
# tissue_contrast) against the existing best config (hqmrboc.yaml). See
# configs/hqmrboc_phase1_pilot.yaml for details on scope and comparison.
#
# Resources: same batch_size/sequence_length/lora_rank as the validated
# production hqmrboc.yaml run (lora_64_emb_traj_hqmrboc), which used
# gpumem_per_gpu:40GB / cpus-per-task:2 / mem:24gb successfully — mem bumped
# slightly to 32gb here for margin. ~300 optimizer steps/epoch here (3
# species balanced to the largest, vs. ~693/epoch for the full 7-species
# run), so expect roughly 2-2.5h for the single epoch plus compile warmup
# and validation, well within the 6h requested.
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
# Use conda env's libstdc++ instead of the (older) system /lib64/libstdc++.so.6
# Fixes: GLIBCXX_3.4.29 not found when numpy/torch C-extensions are loaded
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
echo "CUDA setup verification:"
echo "  CUDA_HOME: ${CUDA_HOME}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Exit if CUDA is not available
python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available in PyTorch!"
    exit 1
}

# Work directory
WORK_DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code

# Config file
CONFIG="${WORK_DIR}/configs/hqmrboc_phase1_pilot.yaml"

# Verify config file exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# Read output_dir and run_name from config YAML
# Use Python to read and expand environment variables in paths
OUTPUT_DIR=$(python -c "import yaml; import os; c=yaml.safe_load(open('${CONFIG}')); od=c.get('output_dir','').rstrip('/'); print(os.path.expandvars(os.path.expanduser(od)) if od else '')")
RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('run_name',''))")

# Fallback to timestamp if run_name is empty
if [ -z "$RUN_NAME" ]; then
    RUN_NAME=$(date +%Y%m%d_%H%M%S)
fi

LOG_DIR="${OUTPUT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${LOG_TIMESTAMP}.log"

echo "Starting finetuning at $(date)"
echo "Config: ${CONFIG}"
echo "Log file: ${LOG_FILE}"
echo "---"

# Resume if checkpoint exists
RESUME="${OUTPUT_DIR}/${RUN_NAME}/best_model.pth"
if [ -f "${RESUME}" ]; then
    echo "Resuming from checkpoint: ${RESUME}"
else
    echo "No checkpoint found at ${RESUME}. Starting fresh training."
    RESUME="auto"
fi

# Run training with timestamped log file
# Python's setup_output_logging will handle the tee to LOG_FILE
python -u ${WORK_DIR}/scripts/finetune_splice.py \
    --config ${CONFIG} \
    --compile \
    --resume ${RESUME} \
    --log-file ${LOG_FILE}

echo "---"
echo "Finetuning completed at $(date). Logs saved to ${LOG_FILE}"
