#!/bin/bash
# Copy everything a finetune config reads from sds into a workspace, for clusters where
# sds is only reachable over SFTP (HoreKa, UC3). Run on the login node; jobs then read
# the copy instead of an rclone mount (AG_SDS_MODE=workspace, see clusters/sds_rclone.sh).
#
#   ws_allocate alphagenome 60                 # once; extend with ws_extend before expiry
#   slurm/stage_inputs.sh configs/hmrboc_132kb_junctions.yaml [more configs...]
#
# Every path under $HOME/sds/ in the config is copied to $WORKSPACE/sds/<same path>, so
# configs need no changes. Also copied: the .fai of genomes, the .json sidecar of
# parquets, and the run directory (output_dir/run_name) if it already exists on sds,
# so a run started elsewhere resumes from its checkpoints. Only new or changed files are
# transferred, so re-running is cheap.
#
# Env: AG_WORKSPACE, AG_WS_NAME (default alphagenome), AG_SDS_REMOTE (default sds-sftp:),
#      AG_RCLONE -- as for jobs. Needs python with PyYAML (e.g. the project conda env).
set -euo pipefail

AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${AG_ROOT}/slurm/clusters/sds_rclone.sh"
[[ $# -ge 1 ]] || { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 1; }

RCLONE=$(ag_find_rclone)
WS=$(ag_find_workspace)
REMOTE=${AG_SDS_REMOTE:-sds-sftp:}
echo "Workspace: ${WS}   remote: ${REMOTE}"

for config in "$@"; do
    [[ "${config}" = /* ]] || config="${AG_ROOT}/${config}"
    echo "=== ${config}"
    # Lines: "file <rel>" (required), "opt <rel>" (copy if present), "dir <rel>" (run dir).
    items=()
    while IFS= read -r line; do items+=("${line}"); done < <(python - "${config}" <<'PY'
import os, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
prefix = os.path.join(os.path.expanduser("~"), "sds") + os.sep

def rel(value):
    p = os.path.expanduser(os.path.expandvars(str(value)))
    return p[len(prefix):] if p.startswith(prefix) else None

def walk(node, key=None):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v, key)
    elif isinstance(node, str) and key not in ("output_dir", "log_file", "run_name"):
        r = rel(node)
        if r:
            yield r

seen = set()
for r in walk(cfg):
    if r in seen:
        continue
    seen.add(r)
    print("file", r)
    if r.endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz")):
        print("opt", r + ".fai")
        print("opt", r + ".gzi")
    if r.endswith(".parquet"):
        print("opt", r[: -len(".parquet")] + ".json")
out = rel(cfg.get("output_dir", "")) if cfg.get("output_dir") else None
if out and cfg.get("run_name"):
    print("dir", os.path.join(out.rstrip("/"), str(cfg["run_name"])))
PY
)
    for item in "${items[@]}"; do
        kind=${item%% *}; rel=${item#* }
        case "${kind}" in
            file)
                "${RCLONE}" copyto "${REMOTE}${rel}" "${WS}/sds/${rel}" \
                    || echo "  ERROR: could not copy ${rel}" >&2
                ;;
            opt)
                if "${RCLONE}" lsf "${REMOTE}${rel}" >/dev/null 2>&1; then
                    "${RCLONE}" copyto "${REMOTE}${rel}" "${WS}/sds/${rel}" \
                        || echo "  ERROR: could not copy ${rel}" >&2
                fi
                ;;
            dir)
                if "${RCLONE}" lsf "${REMOTE}${rel}" >/dev/null 2>&1; then
                    echo "  run dir exists on sds, copying for resume: ${rel}"
                    "${RCLONE}" copy "${REMOTE}${rel}" "${WS}/sds/${rel}" --transfers 4
                fi
                ;;
        esac
    done
    echo "  staged $(printf '%s\n' "${items[@]}" | grep -c '^file ') files"
done
du -sh "${WS}/sds" 2>/dev/null | sed 's/^/Workspace sds copy: /'
