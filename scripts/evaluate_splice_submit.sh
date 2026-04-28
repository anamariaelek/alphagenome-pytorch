#!/bin/bash
#SBATCH --job-name=eval-524-human
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gpumem_per_gpu:40GB
#SBATCH --mem=120gb
#SBATCH --time=10:00:00
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
WORK_DIR=${HOME}/projects/alphagenome_ft_pytorch/

# Model directory
DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai
RUN=132kb_human_mouse_rat

# Evaluation settings
EVAL_SPECIES="mouse"

for EVAL_SPECIES in mouse rat; do
DATA_CONFIG="${DIR}/data/data_config_orthologs.json"
OUT_DIR=${DIR}/${RUN}/preds_${EVAL_SPECIES}/
mkdir -p ${OUT_DIR}

python ${WORK_DIR}/scripts/evaluate_splice.py \
    --checkpoint "${DIR}/${RUN}" \
    --data-config "${DATA_CONFIG}" \
    --eval-species ${EVAL_SPECIES} \
    --per-tissue \
    --overwrite \
    --batch-size 4 \
    --device cuda \
    --max-windows 5000 \
    --seed 1950 \
    --output-dir ${OUT_DIR}

done
