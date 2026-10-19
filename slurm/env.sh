# Shared runtime setup for every job in slurm/jobs/. Sourced at the top of each job script:
#
#   source "${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}/slurm/env.sh"
#
# Afterwards these are set: AG_ROOT (code checkout), AG_DATA (models/data/evals root),
# AG_CLUSTER, and the conda env is active with CUDA verified.
#
# Overridable from the environment:
#   AG_CONDA_SH   path to conda.sh       (default ~/miniforge3/etc/profile.d/conda.sh)
#   AG_CONDA_ENV  conda env name         (default alphagenome_pytorch_genomicsxai)
#   AG_REQUIRE_CUDA=0                    skip the CUDA check (CPU-only jobs)

set -e

AG_ROOT=${AG_ROOT:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code}
AG_DATA=${AG_DATA:-${HOME}/sds/sd17d003/Anamaria/alphagenome_genomicsxai}
export AG_ROOT AG_DATA

source "${AG_ROOT}/slurm/clusters/detect.sh"
source "${AG_ROOT}/slurm/clusters/${AG_CLUSTER}.sh"

echo "Cluster: ${AG_CLUSTER}  host: $(hostname)  job: ${SLURM_JOB_ID:-local}"

# Initialize conda for bash shell
source "${AG_CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"

# Cluster-specific modules (e.g. CUDA) before activating the conda environment
cluster_env

# Raise the open-fd soft limit to the hard cap. DataLoader's default
# "file_descriptor" tensor-sharing strategy holds one fd per in-flight tensor
# shared from a worker process; with several usage tensors per batch and
# persistent_workers=True, a long run can otherwise exhaust a low default
# soft limit ("RuntimeError: received 0 items of ancdata"). Avoid switching to
# the "file_system" strategy instead -- on clusters where /dev/shm is counted
# against the job's --mem cgroup (e.g. Helix), that trades the fd leak for a
# host-memory OOM instead.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# Set OpenMP threads (fallback to 8 if not set by SLURM)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Fix PyTorch memory fragmentation (reduces reserved-but-unallocated memory)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Avoid "Intel oneMKL FATAL ERROR: Cannot load .../libtorch_cpu.so" caused by a
# conflicting MKL runtime getting loaded (e.g. via pandas/scipy/sklearn) before
# torch's own MKL-linked libtorch_cpu.so.
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

conda activate "${AG_CONDA_ENV:-alphagenome_pytorch_genomicsxai}"

# Use conda env's libstdc++ instead of the (older) system /lib64/libstdc++.so.6
# Fixes: GLIBCXX_3.4.29 not found when numpy/torch C-extensions are loaded
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [[ "${AG_REQUIRE_CUDA:-1}" == "1" ]]; then
    echo "CUDA setup verification:"
    echo "  CUDA_HOME: ${CUDA_HOME:-}"
    echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
        echo "ERROR: CUDA is not available in PyTorch!"
        exit 1
    }
fi
