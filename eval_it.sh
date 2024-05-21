#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [[ "$#" != "1" ]]; then
  echo "Usage: eval_it.sh [exp_dir]"
  exit 1
fi

EXP_DIR="$1"

if [[ "$EXP_DIR" == *"llama2"* ]] || [[ "$EXP_DIR" == *"Llama-2"* ]]; then
    LLAMA_DIR="/home/andrewor/local/checkpoints/Llama-2-7b-chat-hf"
    MODEL_TYPE="LLAMA2"
    MODEL_COMPONENT="torchtune.models.llama2.llama2_7b"
    TOKENIZER_COMPONENT="torchtune.models.llama2.llama2_tokenizer"
    CHECKPOINTER="torchtune.utils.FullModelHFCheckpointer"
    if [[ -z "$CHECKPOINT_FILES" ]]; then
        if [[ -n "$QUANTIZED" ]]; then
            CHECKPOINT_FILES="[hf_model_0001_2-8da4w.pt]"
        else
            CHECKPOINT_FILES="[hf_model_0001_2.pt,hf_model_0002_2.pt]"
        fi
    fi
elif [[ "$EXP_DIR" == *"llama3"* ]] || [[ "$EXP_DIR" == *"Llama-3"* ]]; then
    LLAMA_DIR="/home/andrewor/local/checkpoints/Meta-Llama-3-8B-Instruct/original"
    MODEL_TYPE="LLAMA3"
    MODEL_COMPONENT="torchtune.models.llama3.llama3_8b"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    CHECKPOINTER="torchtune.utils.FullModelMetaCheckpointer"
    if [[ -z "$CHECKPOINT_FILES" ]]; then
        if [[ -n "$QUANTIZED" ]]; then
            CHECKPOINT_FILES="[meta_model_2-8da4w.pt]"
        else
            CHECKPOINT_FILES="[meta_model_2.pt]"
        fi
    fi
else
    echo "Did not find llama version from experiment dir '$EXP_DIR'"
    exit 1
fi

if [[ -n "$RUN_TAG" ]]; then
    RUN_TAG="_${RUN_TAG}"
fi

LOG_FILE="eval${RUN_TAG}.log"
EVAL_OUTPUT_DIR="eval_output${RUN_TAG}"
TASKS="${TASKS:-"[\"hellaswag\", \"wikitext\", \"anli_r1\", \"anli_r2\", \"anli_r3\", \"arc_challenge\", \"arc_easy\", \"openbookqa\", \"piqa\"]"}"
ALL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
if [[ "$ALL_CUDA_VISIBLE_DEVICES" != *","* ]] && [[ "$SKIP_QUANTIZE" != "true" ]] && [[ "$SKIP_FLOAT" != "true" ]]; then
    echo "Need at least two CUDA_VISIBLE_DEVICES"
    exit 1
fi

if [[ "$EXP_DIR" == *"qat"* ]]; then
    MY_QUANTIZE_MODE="qat"
else
    MY_QUANTIZE_MODE="full"
fi

# Evaluate bf16 first
if [[ "$SKIP_FLOAT" != "true" ]]; then
    echo "Running eval on '${EXP_DIR}', ${EXP_DIR}/${LOG_FILE}"
    export CUDA_VISIBLE_DEVICES="$(echo ${ALL_CUDA_VISIBLE_DEVICES} | awk -F',' '{print $1}')"
    set -x
    tune run eleuther_eval --config eleuther_evaluation \
        model._component_="${MODEL_COMPONENT}" \
        tokenizer._component_="${TOKENIZER_COMPONENT}" \
        tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
        checkpointer._component_="${CHECKPOINTER}" \
        checkpointer.checkpoint_dir="${EXP_DIR}" \
        checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
        checkpointer.output_dir="${EXP_DIR}/${EVAL_OUTPUT_DIR}" \
        checkpointer.model_type="$MODEL_TYPE" \
        tasks="$TASKS" \
        my_quantize_mode="$MY_QUANTIZE_MODE" > "${EXP_DIR}/${LOG_FILE}" 2>&1 &
    set +x
fi

# Now evaluate quantized
if [[ "$SKIP_QUANTIZE" != "true" ]]; then
    echo "Running eval quantized on '${EXP_DIR}', ${EXP_DIR}/${LOG_FILE}"
    MY_QUANTIZE_MODE="${MY_QUANTIZE_MODE}-quantized"
    LOG_FILE="eval_quantized${RUN_TAG}.log"
    EVAL_OUTPUT_DIR="eval_quantized_output${RUN_TAG}"
    if [[ "$SKIP_FLOAT" != "true" ]]; then
        export CUDA_VISIBLE_DEVICES="$(echo ${ALL_CUDA_VISIBLE_DEVICES} | awk -F',' '{print $2}')"
    fi
    set -x
    tune run eleuther_eval --config eleuther_evaluation \
        model._component_="${MODEL_COMPONENT}" \
        tokenizer._component_="${TOKENIZER_COMPONENT}" \
        tokenizer.path="${LLAMA_DIR}/tokenizer.model" \
        checkpointer._component_="${CHECKPOINTER}" \
        checkpointer.checkpoint_dir="${EXP_DIR}" \
        checkpointer.checkpoint_files="${CHECKPOINT_FILES}" \
        checkpointer.output_dir="${EXP_DIR}/${EVAL_OUTPUT_DIR}" \
        checkpointer.model_type="$MODEL_TYPE" \
        tasks="$TASKS" \
        my_quantize_mode="$MY_QUANTIZE_MODE" > "${EXP_DIR}/${LOG_FILE}" 2>&1 &
    set +x
fi
wait
