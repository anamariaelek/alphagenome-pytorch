# sasselab3 workstation profile (sourced by submit.sh and env.sh).
#
# No Slurm: submit.sh runs the job script directly with nohup and writes the log to
# logs/. There is a single GPU shared with other users, so check `nvidia-smi` before
# starting anything big.

AG_USE_SLURM=0

cluster_gpu_args() { :; }

cluster_cpu_args() { :; }

# Runs inside the job, before conda is activated. CUDA comes from the conda env /
# system driver, so there is nothing to load.
cluster_env() {
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
}
