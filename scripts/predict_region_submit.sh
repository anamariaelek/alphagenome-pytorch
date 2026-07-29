#!/bin/bash
#SBATCH --job-name=predict-region
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,gpumem_per_gpu:20GB
#SBATCH --mem=60gb
#SBATCH --time=8:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
# 
# Helix GPU options:
# - A40 (48 GB):   --gres=gpu:A40:1
# - A100 (40 GB):  --gres=gpu:A100:1
# - A100 (80 GB):  --gres=gpu:A100:1
# - H200 (141 GB): --gres=gpu:H200:1
# 
# 132kb models:
#  - inference: 25GB for batch 4
# 524kb models:
#  - inference: 95GB for batch 4

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Load CUDA module before activating conda environment
module load devel/cuda

# Set OpenMP threads (fallback to 8 if not set by SLURM)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Fix PyTorch memory fragmentation (reduces reserved-but-unallocated memory)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Avoid "Intel oneMKL FATAL ERROR: Cannot load .../libtorch_cpu.so" caused by a
# conflicting MKL runtime getting loaded (e.g. via pandas/scipy/sklearn) before
# torch's own MKL-linked libtorch_cpu.so.
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# Activate conda environment
conda activate alphagenome_pytorch_genomicsxai

# Ensure correct libstdc++ is used (fix GLIBCXX errors)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

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
WORK_DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code

# Models directory
DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai

# Pretrained model for sanity check
# CHECKPOINT_PATH=${WORK_DIR}/checkpoints/pretrained.pth
# PRED_DIR=preds_pretrained_${TIMESTAMP}

# Finetuned model
RUN=lora_32_human_mouse_rat_rabbit_opossum
CHECKPOINT_PATH="${DIR}/${RUN}"
PRED_DIR=region_preds_${TIMESTAMP}

# Data configuration
DATA_CONFIG="${DIR}/data/data_config_intersect_usage.json"

# Evaluation settings
for ORGANISM in human mouse rat rabbit opossum chicken; do
OUT_DIR=${DIR}/${RUN}/${PRED_DIR}/${ORGANISM}/
mkdir -p ${OUT_DIR}

python ${WORK_DIR}/scripts/predict_region.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --data-config "${DATA_CONFIG}" \
    --organisms "${ORGANISM}" \
    --regions chr17:41196312-41277500 \
    --observed-conditions-only \
    --cross-species-usage \
    --plot  \
    --device cuda \
    --output-dir  "${OUT_DIR}"

done
