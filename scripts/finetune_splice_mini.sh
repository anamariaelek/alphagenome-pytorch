# Prepare files
for sp in human mouse rabbit rat opossum chicken; do

    dir="/home/elek/sds/sd17d003/Anamaria/borzoi_folds/results_orthologs_132kb_protein_coding/${sp}//fold0"

    for sub in train valid test; do
        in="${dir}/${sub}.bed"
        out="${dir}/${sub}_mini.bed"

        if [ "${sub}" = "train" ]; then
            n=500
        else
            n=100
        fi

        if [ ! -f "${in}" ]; then
            echo "Skipping ${sub}: input file not found at ${in}"
            continue
        fi

        echo "Creating ${out} with ${n} lines from ${in}"
        head -n "${n}" "${in}" > "${out}"
    done
done

# Work directory
WORK_DIR=${HOME}/projects/alphagenome_ft_pytorch/

for CONFIG in "${WORK_DIR}/configs/finetune_mini"*".yaml"; do
    # Read output_dir and run_name from config YAML
    OUTPUT_DIR=$(python -c "import yaml; import os; c=yaml.safe_load(open('${CONFIG}')); od=c.get('output_dir','').rstrip('/'); print(os.path.expandvars(os.path.expanduser(od)) if od else '')")
    RUN_NAME=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('run_name',''))")
    LOG_DIR="${OUTPUT_DIR}/${RUN_NAME}"
    mkdir -p "${LOG_DIR}"
    LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/train_${LOG_TIMESTAMP}.log"

    echo
    echo "---------------------------------------------------"
    echo "Starting finetuning at $(date)"
    echo "Config: ${CONFIG}"
    echo "Log file: ${LOG_FILE}"

    # Resume if checkpoint exists
    RESUME="${OUTPUT_DIR}/${RUN_NAME}/best_model.pth"
    if [ -f "${RESUME}" ]; then
        echo "Resuming from checkpoint: ${RESUME}"
    else
        echo "No checkpoint found at ${RESUME}. Starting fresh training."
        RESUME="auto"
    fi

    # Run training with timestamped log file
    python -u ${WORK_DIR}/scripts/finetune_splice.py \
        --config ${CONFIG} \
        --compile \
        --batch-size 4 \
        --resume ${RESUME} \
        --log-file ${LOG_FILE}

    echo "Finetuning completed at $(date). Logs saved to ${LOG_FILE}"
    echo "---------------------------------------------------"

done
echo
echo "All finetuning runs completed at $(date)."


