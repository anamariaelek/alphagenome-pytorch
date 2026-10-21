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
    horeka.sh          # ... for HoreKa (KIT)
    sds_rclone.sh      # sds via rclone (HoreKa, UC3): workspace copy or per-job mount
  stage_inputs.sh      # copy a config's inputs to the workspace (HoreKa, UC3 login node)
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

**HoreKa and bwUniCluster3: sds over rclone**

On these clusters sds is only reachable over SFTP with rclone (on the login node it is
mounted at `~/sds`). `clusters/sds_rclone.sh` sets it up inside each job, in one of two
modes (`AG_SDS_MODE`):

- **`workspace`** (default for `finetune_splice`): `~/sds` in the job is a copy of the
  inputs in a workspace, so training reads local files. Reading over rclone made training
  very slow. Outputs (run dir: checkpoints, logs, `epoch_log.csv`) are written to the
  workspace and copied back to the same place on sds every 10 minutes and at job end.
  Only new or changed files are transferred.
- **`mount`** (default for the other jobs): sds is rclone-mounted in the job's own
  `$TMPDIR`. Fine for small jobs; set `AG_SDS_MODE=mount` to use it for training anyway.

In both modes the job's `HOME` is a per-job directory linking everything in your real home
except `sds`, so the `$HOME/sds/...` paths in configs work unchanged, and a dead or shared
`~/sds` mount on the node doesn't matter.

One-time setup on the login node:
- **Clone the code outside `~/sds`**, into `$HOME` or a workspace, and submit from that
  clone (the job reads `slurm/env.sh` before sds is set up).
- **Create the conda env** on the cluster's own `$HOME`:
  `conda env create -f env_alphagenome_pytorch_genomicsxai.yml`.
- **rclone remote.** Jobs use the `sds-sftp:` remote; check with `rclone listremotes` and
  export `AG_SDS_REMOTE=<name>:` if yours differs (`submit.sh` forwards it). `rclone` is
  looked up on `PATH`, then `~/bin` and `~/.local/bin`; otherwise set `AG_RCLONE`.
- **Workspace.** `ws_allocate alphagenome 60` (extend with `ws_extend alphagenome 60` before
  it expires). Jobs find it with `ws_find alphagenome`; set `AG_WS_NAME` or `AG_WORKSPACE`
  to use another.

Before training with a config (again whenever its inputs change on sds):
```bash
conda activate alphagenome_pytorch_genomicsxai     # stage_inputs.sh needs PyYAML
slurm/stage_inputs.sh configs/hmrboc_132kb_junctions.yaml
```
This copies every `$HOME/sds/...` path in the config (genomes with `.fai`, parquets with
their `.json`, BEDs, pretrained weights, metadata CSVs) to `<workspace>/sds/<same path>`,
plus the run dir if it already exists on sds, so a run started elsewhere resumes from its
checkpoints. Already-copied, unchanged files are skipped.

**bwUniCluster3, first use:**
- Run `sinfo -s` to check that the partitions in `clusters/uc3.sh` exist. The default is
  `gpu_h100`; `gpu_mi300` is AMD and will not work with our CUDA build.
- Run `module avail cuda` and fix `cluster_env` in `clusters/uc3.sh` if the module isn't
  `devel/cuda`.
- Do a short smoke test first:
  `slurm/submit.sh --partition dev_gpu_h100 --time 00:20:00 finetune_splice configs/hqm_pilot.yaml`
  (check the dev partition name in `sinfo -s`).

**HoreKa, first use:**
- Short test (dev queues: 1 hour, one job at a time):
  `slurm/submit.sh --partition dev_accelerated --time 00:20:00 finetune_splice configs/hqm_pilot.yaml`.
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
