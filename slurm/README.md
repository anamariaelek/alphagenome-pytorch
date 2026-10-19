# Job submission (Helix, bwUniCluster3, HoreKa, sasselab3)

Python entry points live in `scripts/`. This directory holds the batch-job layer around them.
It is split by layer, not by cluster:

```
slurm/
  submit.sh            # the only command you run: picks the cluster, adds its sbatch flags
  env.sh               # shared runtime setup (conda, CUDA check, LD_LIBRARY_PATH, ...)
  clusters/
    detect.sh          # sets AG_CLUSTER from $AG_CLUSTER or the hostname
    helix.sh           # partition/gres syntax + modules for Helix
    uc3.sh             # ... for bwUniCluster 3.0
    horeka.sh          # ... for HoreKa (KIT), incl. per-job rclone mount of ~/sds
    sasselab3.sh       # no Slurm: runs the job directly with nohup
  jobs/                # cluster-agnostic job bodies, parameterised by arguments/env
    finetune_splice.sh  evaluate_splice.sh  evaluate_splice_brawand.sh  predict_region.sh
```

`sbatch` flags on the command line override `#SBATCH` headers, and headers cannot contain
variables. So the job scripts carry no cluster-specific directives, and `submit.sh` passes
them as flags.

## One-time setup per machine

Add this to `~/.bashrc` (if it's missing, `submit.sh` falls back to hostname matching):

```bash
export AG_CLUSTER=helix        # or uc3, horeka, sasselab3
# export AG_SLURM_ACCOUNT=...  # only if the cluster needs --account
```

The conda env must exist on each machine at `~/miniforge3`. To use a different location or
env name, set `AG_CONDA_SH` / `AG_CONDA_ENV`.

**bwUniCluster3, first use:**
- Run `sinfo -s` to check that the partitions in `clusters/uc3.sh` exist. The default is
  `gpu_h100`; `gpu_mi300` is AMD and will not work with our CUDA build.
- Run `module avail cuda` and fix `cluster_env` in `clusters/uc3.sh` if the module isn't
  `devel/cuda`.
- Check that `$HOME/sds/...` is mounted on the compute nodes (e.g. `srun ... ls ~/sds`).
- UC3 has its own `$HOME`, so create the conda env there with
  `conda env create -f env_alphagenome_pytorch_genomicsxai.yml`.
- Do a short smoke test first:
  `slurm/submit.sh --partition dev_gpu_h100 --time 00:20:00 finetune_splice configs/hqm_pilot.yaml`
  (check the dev partition name in `sinfo -s`).

**HoreKa, first use:**
- **Keep the code checkout outside `~/sds`.** sds is rclone-mounted at `~/sds` on the login
  node only, so on a compute node `~/sds` is an empty directory. `clusters/horeka.sh` mounts
  it again at the same path at the start of each job, so the `$HOME/sds/...` paths in
  configs keep working. The job reads `slurm/env.sh` before that mount exists, so clone the
  repo into `$HOME` or a workspace and submit from that clone.
- **rclone remote.** The job uses the `sds-sftp:` remote. Check the name with
  `rclone listremotes`; if yours differs, export `AG_SDS_REMOTE=<name>:` before submitting
  (`submit.sh` forwards it).
- **Mounting from a compute node.** Check that `rclone` and `fusermount` work there before a
  long run, e.g. `slurm/submit.sh --partition dev_accelerated --time 00:20:00 finetune_splice configs/hqm_pilot.yaml`.
  The dev queues allow one job at a time and at most 1 hour (4 hours for `dev_cpuonly`).
- **Writes to `~/sds`.** Checkpoints written there go through rclone's cache in `$TMPDIR` and
  are uploaded in the background. At job end the mount is released and the job waits up
  to 10 minutes for rclone to finish uploading. For long runs, an `output_dir` on a
  HoreKa workspace avoids this.
- **Conda env.** Create the env on HoreKa's own `$HOME`:
  `conda env create -f env_alphagenome_pytorch_genomicsxai.yml`.
- **Account.** If your user belongs to more than one project, set `AG_SLURM_ACCOUNT`.
- **Limits.** GPU partitions allow at most 2 days per job, and `cpuonly` 3 days.

## Usage

```bash
slurm/submit.sh [--gpu-mem N] [--mem M] [--cpus N] [--time T] [--partition P] \
                [--name NAME] [--account A] [--sbatch-arg X]... [--dry-run] JOB [args...]

slurm/submit.sh --gpu-mem 80 --mem 24G --time 36:00:00 finetune_splice configs/hqmrboc_132kb.yaml
slurm/submit.sh --gpu-mem 40 --mem 40G --time 12:00:00 evaluate_splice lora_64_emb_traj_hqmrboc___524kb
HELDOUT_SPECIES="rat rabbit" TRAIN_SPECIES="human mouse" slurm/submit.sh evaluate_splice my_run
slurm/submit.sh --dry-run ...    # print the command without running it
```

- On HoreKa, `--gpu-mem` only selects the default partition: up to 40 picks `accelerated`
  (A100), up to 80 picks `accelerated-h100`, and anything larger picks `accelerated-h200`.
- `--gpu-mem` maps to `gpumem_per_gpu:<N>GB` on Helix. UC3 ignores it, because there the
  partition selects the GPU.
- Logs go to `logs/slurm/<job-name>_<jobid>.log`, with stdout and stderr in one file.
  `finetune_splice` also writes `train_<timestamp>.log` into the run's `output_dir`.
- Each job script documents its arguments and env overrides in its header.

## Resources that worked on Helix (from the old per-config submit scripts)

| Config | GPU mem (GB) | mem | cpus | time |
|---|---|---|---|---|
| `configs/finetune_h_132kb.yaml` | 40 | 240G | 1 | 36:00:00 |
| `configs/finetune_m_132kb.yaml` | 40 | 200G | 1 | 48:00:00 |
| `configs/hm.yaml` (no longer exists) | 40 | 30G | 1 | 48:00:00 |
| `configs/finetune_hmr_132kb.yaml` | 40 | 300G | 1 | 48:00:00 |
| `configs/finetune_hmrb_132kb.yaml` | 40 | 300G | 1 | 48:00:00 |
| `configs/finetune_hmbo_132kb.yaml` | 40 | 280G | 1 | 48:00:00 |
| `configs/finetune_hmrbo_132kb.yaml` | 40 | 80G | 4 | 48:00:00 |
| `configs/finetune_hmrboc_132kb.yaml` | 40 | 80G | 4 | 36:00:00 |
| `configs/finetune_hrbo_132kb.yaml` | 40 | 240G | 4 | 36:00:00 |
| `configs/finetune_hrboc_132kb.yaml` | 40 | 240G | 4 | 36:00:00 |
| `configs/hqmo.yaml` (no longer exists) | 40 | 30G | 2 | 48:00:00 |
| `configs/hqmrboc_132kb.yaml` | 80 | 24G | 2 | 36:00:00 |
| `configs/hqm_pilot.yaml` | 40 | 32G | 2 | 06:00:00 |
| `configs/finetune_full_hmrbo_132kb.yaml` (no longer exists; full finetune) | 80 | 320G | 16 | 48:00:00 |
| `evaluate_splice` / `evaluate_splice_brawand` | 40 | 40G | 1 | 12:00:00 |
| `predict_region` | 20 | 60G | 1 | 8:00:00 |

Host memory varies a lot between configs (24G to 300G), so check before reusing a number.
For evaluation, 132kb models need about 25 GB of GPU memory at batch 4, and 524kb models about 95 GB.

The old `scripts/*_submit*.sh` files are in git history (removed in the commit that added
this directory).
