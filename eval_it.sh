#!/bin/bash

if [[ "$#" != "1" ]]; then
  echo "Usage: eval_it.sh [exp_name]"
  exit 1
fi

EXP_NAME="$1"

LLAMA_DIR="/home/andrewor/local/checkpoints/Llama-2-7b-chat-hf"
LOG_DIR="${LOG_DIR:-/home/andrewor/logs/tune}"
EXP_DIR="${LOG_DIR}/${EXP_NAME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-2,3}"

if [[ -n "$QUANTIZED" ]]; then
    EXP_DIR="${EXP_DIR}/quantize_output"
    CHECKPOINTER="torchtune.utils.FullModelTorchTuneCheckpointer"
    EXTRA_ARGS="quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer"
    EXTRA_ARGS="$EXTRA_ARGS quantizer.groupsize=256"
    CHECKPOINT_FILES="[hf_model_0001_2-8da4w.pt]"
else
    CHECKPOINTER="torchtune.utils.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[hf_model_0001_2.pt,hf_model_0002_2.pt]"
fi

echo "Running eval on '${EXP_NAME}', logging to ${EXP_DIR}/eval.log"

set -x
tune run eleuther_eval --config eleuther_evaluation \
    tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
    checkpointer._component_="${CHECKPOINTER}" \
    checkpointer.checkpoint_dir="${EXP_DIR}" \
    checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
    checkpointer.output_dir="${EXP_DIR}/eval_output" \
    $EXTRA_ARGS > "${EXP_DIR}/eval.log" 2>&1
set +x
