#!/bin/bash

if [[ "$#" != "1" ]]; then
  echo "Usage: quantize_it.sh [exp_name]"
  exit 1
fi

EXP_NAME="$1"

LLAMA_DIR="/home/andrewor/local/checkpoints/Llama-2-7b-chat-hf"
LOG_DIR="${LOG_DIR:-/home/andrewor/logs/tune}"
EXP_DIR="${LOG_DIR}/${EXP_NAME}"
CHECKPOINT_FILES="${CHECKPOINT_FILES:-[hf_model_0001_2.pt,hf_model_0002_2.pt]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-2,3}"

echo "Running quantize on '${EXP_NAME}', logging to ${EXP_DIR}/quantize.log"

set -x
tune run quantize --config quantization \
    tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
    checkpointer.checkpoint_dir="${EXP_DIR}" \
    checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
    checkpointer.output_dir="${EXP_DIR}/quantize_output" \
    quantizer._component_="torchtune.utils.quantization.Int8DynActInt4WeightQuantizer" \
    quantizer.groupsize=256 > "${EXP_DIR}/quantize.log" 2>&1
set +x
