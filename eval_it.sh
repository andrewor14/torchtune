#!/bin/bash

if [[ "$#" != "1" ]]; then
  echo "Usage: eval_it.sh [exp_name]"
  exit 1
fi

EXP_NAME="$1"

if [[ "$EXP_NAME" == *"llama2"* ]]; then
    LLAMA_DIR="/home/andrewor/local/checkpoints/Llama-2-7b-chat-hf"
    MODEL_TYPE="LLAMA2"
    MODEL_COMPONENT="torchtune.models.llama2.llama2_7b"
    TOKENIZER_COMPONENT="torchtune.models.llama2.llama2_tokenizer"
    CHECKPOINTER="torchtune.utils.FullModelHFCheckpointer"
    if [[ -n "$QUANTIZED" ]]; then
        CHECKPOINT_FILES="[hf_model_0001_2-8da4w.pt]"
    else
        CHECKPOINT_FILES="[hf_model_0001_2.pt,hf_model_0002_2.pt]"
    fi
elif [[ "$EXP_NAME" == *"llama3"* ]]; then
    LLAMA_DIR="/home/andrewor/local/checkpoints/Meta-Llama-3-8B-Instruct/original"
    MODEL_TYPE="LLAMA3"
    MODEL_COMPONENT="torchtune.models.llama3.llama3_8b"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    CHECKPOINTER="torchtune.utils.FullModelMetaCheckpointer"
    if [[ -n "$QUANTIZED" ]]; then
        CHECKPOINT_FILES="[meta_model_2-8da4w.pt]"
    else
        CHECKPOINT_FILES="[meta_model_2.pt]"
    fi
else
    echo "Did not find llama version from experiment name '$EXP_NAME'"
    exit 1
fi

LOG_DIR="${LOG_DIR:-/home/andrewor/logs/tune}"
EXP_DIR="${LOG_DIR}/${EXP_NAME}"
TASKS="${TASKS:-[\"wikitext\", \"hellaswag\", \"truthfulqa_mc2\"]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

if [[ -n "$QUANTIZED" ]]; then
    EXP_DIR="${EXP_DIR}/quantize_output"
    CHECKPOINTER="torchtune.utils.FullModelTorchTuneCheckpointer"
    EXTRA_ARGS="quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer"
    EXTRA_ARGS="$EXTRA_ARGS quantizer.groupsize=256"
fi

echo "Running eval on '${EXP_NAME}', logging to ${EXP_DIR}/eval.log"

set -x
tune run eleuther_eval --config eleuther_evaluation \
    model._component_="${MODEL_COMPONENT}" \
    tokenizer._component_="${TOKENIZER_COMPONENT}" \
    tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
    checkpointer._component_="${CHECKPOINTER}" \
    checkpointer.checkpoint_dir="${EXP_DIR}" \
    checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
    checkpointer.output_dir="${EXP_DIR}/eval_output" \
    checkpointer.model_type="$MODEL_TYPE" \
    tasks="$TASKS" \
    $EXTRA_ARGS > "${EXP_DIR}/eval.log" 2>&1
set +x
