#!/bin/bash

if [[ "$#" != "1" ]]; then
  echo "Usage: quantize_it.sh [exp_name]"
  exit 1
fi

EXP_NAME="$1"

if [[ "$EXP_NAME" == *"llama2"* ]]; then
    MODEL_TYPE="LLAMA2"
    MODEL_COMPONENT="torchtune.models.llama2.llama2_7b"
    CHECKPOINTER="torchtune.utils.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[hf_model_0001_2.pt,hf_model_0002_2.pt]"
elif [[ "$EXP_NAME" == *"llama3"* ]]; then
    MODEL_TYPE="LLAMA3"
    MODEL_COMPONENT="torchtune.models.llama3.llama3_8b"
    CHECKPOINTER="torchtune.utils.FullModelMetaCheckpointer"
    CHECKPOINT_FILES="[meta_model_2.pt]"
else
    echo "Did not find llama version from experiment name '$EXP_NAME'"
    exit 1
fi

LOG_DIR="${LOG_DIR:-/home/andrewor/logs/tune}"
EXP_DIR="${LOG_DIR}/${EXP_NAME}"
CHECKPOINT_FILES="${CHECKPOINT_FILES:-[hf_model_0001_2.pt,hf_model_0002_2.pt]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-2,3}"

echo "Running quantize on '${EXP_NAME}', logging to ${EXP_DIR}/quantize.log"

set -x
tune run quantize --config quantization \
    model._component_="${MODEL_COMPONENT}" \
    checkpointer._component_="${CHECKPOINTER}" \
    checkpointer.checkpoint_dir="${EXP_DIR}" \
    checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
    checkpointer.output_dir="${EXP_DIR}/quantize_output" \
    checkpointer.model_type="${MODEL_TYPE}" \
    quantizer._component_="torchtune.utils.quantization.Int8DynActInt4WeightQuantizer" \
    quantizer.groupsize=256 > "${EXP_DIR}/quantize.log" 2>&1
set +x
