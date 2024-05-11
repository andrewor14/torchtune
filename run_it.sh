#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [[ "$#" != "1" ]]; then
  echo "Usage: run_it.sh [qat|full]"
  exit 1
fi

EXP_TYPE="$1"
if [[ "$EXP_TYPE" == "qat" ]]; then
    EXTRA_ARGS="qat_mode=8da4w"
fi

if [[ "$LLAMA_VERSION" == "2" ]]; then
    LLAMA_DIR="/home/andrewor/local/checkpoints/Llama-2-7b-chat-hf"
    CHECKPOINTER="torchtune.utils.FullModelHFCheckpointer"
    CHECKPOINT_FILES="${CHECKPOINT_FILES:-[pytorch_model-00001-of-00002.bin,pytorch_model-00002-of-00002.bin]}"
    CONFIG="llama2/7B_full"
else
    # default
    LLAMA_VERSION="3"
    LLAMA_DIR="/home/andrewor/local/checkpoints/Meta-Llama-3-8B-Instruct/original"
    CHECKPOINTER="torchtune.utils.FullModelMetaCheckpointer"
    CHECKPOINT_FILES="${CHECKPOINT_FILES:-[consolidated.00.pth]}"
    CONFIG="llama3/8B_full"
fi

if [[ -n "$MAX_STEPS_PER_EPOCH" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS max_steps_per_epoch=$MAX_STEPS_PER_EPOCH"
fi
if [[ -n "$ENABLE_FAKE_QUANT_STEP" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS qat_enable_fake_quant_step=$ENABLE_FAKE_QUANT_STEP"
fi

EXP_NAME="${EXP_TYPE}_llama${LLAMA_VERSION}_"`date +%s`
if [[ -n "$RUN_TAG" ]]; then
    EXP_NAME="${EXP_NAME}_${RUN_TAG}"
fi
LOG_DIR="${LOG_DIR:-/home/andrewor/local/logs/tune}"
EXP_DIR="${LOG_DIR}/${EXP_NAME}"
LOG_FILE="${EXP_DIR}/run.log"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$LLAMA_DIR}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
PORT="${PORT:-29500}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_DEVICES="$(echo $CUDA_VISIBLE_DEVICES | sed 's/,/\n/g' | wc -l)"

mkdir -p "${EXP_DIR}"
echo "Running '${EXP_TYPE}', logging to ${LOG_FILE}"

# Log commit hashes and outstanding code changes
for repo in "pytorch" "ao" "torchtune"; do
    REPO_PATH="/home/andrewor/local/${repo}"
    echo -e "===== Logging commit hash from ${REPO_PATH} =====\n" >> "${LOG_FILE}"
    cd "$REPO_PATH"
    git branch >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    git log --oneline | head -n 10 >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    git diff >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
done
echo "============================================================================" >> "${LOG_FILE}"
echo "Starting finetuning run..." >> "${LOG_FILE}"

set -x
tune run --nnodes 1 --nproc_per_node "$NUM_DEVICES" --rdzv_endpoint="localhost:$PORT" full_finetune_distributed --config "$CONFIG" \
    batch_size="$BATCH_SIZE" \
    epochs="$NUM_EPOCHS" \
    enable_activation_checkpointing=False \
    log_peak_memory_stats=True \
    tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
    checkpointer._component_="${CHECKPOINTER}" \
    checkpointer.checkpoint_dir="${CHECKPOINT_DIR}" \
    checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
    checkpointer.output_dir="${EXP_DIR}" \
    output_dir="${EXP_DIR}/alpaca-llama2-finetune" \
    $EXTRA_ARGS >> "${LOG_FILE}" 2>&1
set +x
