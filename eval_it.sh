#!/bin/bash

if [[ "$#" != "1" ]]; then
  echo "Usage: eval_it.sh [exp_name]"
  exit 1
fi

EXP_NAME="$1"

LLAMA_DIR="/home/andrewor/local/checkpoints/Llama-2-7b-chat-hf"
LOG_DIR="${LOG_DIR:-/home/andrewor/logs/tune}"
EXP_DIR="${LOG_DIR}/${EXP_NAME}"
CHECKPOINT_FILES="${CHECKPOINT_FILES:-[hf_model_0001_2.pt,hf_model_0002_2.pt]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-2,3}"

echo "Running eval on '${EXP_NAME}', logging to ${EXP_DIR}/eval.log"

set -x
tune run eleuther_eval --config eleuther_evaluation \
    tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
    checkpointer.checkpoint_dir="${EXP_DIR}" \
    checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
    checkpointer.output_dir="${EXP_DIR}/eval_output" > "${EXP_DIR}/eval.log" 2>&1
set +x
