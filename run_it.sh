#!/bin/bash

if [[ "$#" != "1" ]]; then
  echo "Usage: run_it.sh [qat|full]"
  exit 1
fi

EXP_TYPE="$1"

if [[ "$EXP_TYPE" == "qat" ]]; then
    EXTRA_ARGS="qat_mode=8da4w"
fi

RUN_TAG=`date +%s`
LLAMA_DIR="/home/andrewor/local/checkpoints/Llama-2-7b-chat-hf"
LOG_DIR="/home/andrewor/logs/tune/"
EXP_DIR="${LOG_DIR}/${EXP_TYPE}_${RUN_TAG}"
BATCH_SIZE="${BATCH_SIZE:-12}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

echo "Running '${EXP_TYPE}', logging to ${EXP_DIR}/run.log"
mkdir -p "${EXP_DIR}"

set -x
tune run --nnodes 1 --nproc_per_node 4 full_finetune_distributed --config llama2/7B_full \
    batch_size="$BATCH_SIZE" \
    enable_activation_checkpointing=False \
    log_peak_memory_stats=True \
    tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
    checkpointer.checkpoint_dir="${LLAMA_DIR}" \
    checkpointer.output_dir="${EXP_DIR}" \
    output_dir="${EXP_DIR}/alpaca-llama2-finetune" \
    $EXTRA_ARGS > "${EXP_DIR}/run.log" 2>&1
set +x
