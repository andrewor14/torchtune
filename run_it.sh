# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ENABLE_FP8="${ENABLE_FP8:-True}"
FP8_RECIPE_NAME="${FP8_RECIPE_NAME:-null}"
TP_PLAN="${TP_PLAN:-null}"
TP_DIM="${TP_DIM:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS=1
LAST_EPOCH_INDEX=0
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-null}"
COMPILE="True"
export NCCL_SHM_DISABLE=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
FIRST_GPU="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | head -n 1)"


# Experiment type
if [[ "$ENABLE_FP8" == "true" ]] || [[ "$ENABLE_FP8" == "True" ]]; then
    DIR_NAME="fp8_quantized_training"
else
    DIR_NAME="full_finetune_distributed_baseline"
fi
LOG_DIR="/home/${USER}/local/logs/tune/${DIR_NAME}"
LAST_EPOCH_CHECKPOINT_DIR="${LOG_DIR}/epoch_${LAST_EPOCH_INDEX}"


# Model type
if [[ -z "$MODEL" ]]; then
    MODEL="Llama3-8B"
fi
if [[ "$MODEL" == "Llama3.2-3B" ]]; then
    CONFIG="llama3_2/3B_full"
    MODEL_COMPONENT="torchtune.models.llama3_2.llama3_2_3b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00002.safetensors,model-00002-of-00002.safetensors]"
elif [[ "$MODEL" == "Llama3.1-8B" ]]; then
    CONFIG="llama3_1/8B_full"
    MODEL_COMPONENT="torchtune.models.llama3_1.llama3_1_8b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00004.safetensors,model-00002-of-00004.safetensors,model-00003-of-00004.safetensors,model-00004-of-00004.safetensors]"
elif [[ "$MODEL" == "Llama3-8B" ]]; then
    CONFIG="llama3/8B_full"
    MODEL_COMPONENT="torchtune.models.llama3.llama3_8b"
    CHECKPOINTER="torchtune.training.FullModelMetaCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00001.bin]"
else
    echo "Unknown model $MODEL"
    exit 1
fi


# Finetune
if [[ "$SKIP_FINETUNE" != "true" ]]; then
    # Delete the old log dir if it exists
    if [[ -d "$LOG_DIR" ]]; then
        echo "Removing $LOG_DIR..."
        rm -rf "$LOG_DIR"
    fi
    mkdir -p "$LOG_DIR"

    echo "Running finetuning, logging to ${LOG_DIR}/run.log"
    tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" full_finetune_distributed --config "$CONFIG" \
        enable_fp8_training="$ENABLE_FP8" \
        fp8_recipe_name="$FP8_RECIPE_NAME" \
        epochs="$EPOCHS" \
        batch_size="$BATCH_SIZE" \
        compile="$COMPILE" \
        dataset._component_=torchtune.datasets.alpaca_cleaned_dataset \
        checkpointer.output_dir="$LOG_DIR" \
        output_dir="${LOG_DIR}/metrics" \
        metric_logger.log_dir="${LOG_DIR}/metrics" \
        max_steps_per_epoch="$MAX_STEPS_PER_EPOCH" \
        tensor_parallel_plan="$TP_PLAN" \
        tensor_parallel_dim="$TP_DIM" \
        > "${LOG_DIR}/run.log" 2>&1
fi


# Eval
if [[ "$SKIP_EVAL" != "true" ]]; then
    echo "Evaluating wikitext, logging to ${LOG_DIR}/eval_wikitext_float.log"
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
        batch_size=1 \
        tasks=[wikitext] \
        model._component_="$MODEL_COMPONENT" \
        checkpointer._component_="$CHECKPOINTER" \
        checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
        checkpointer.output_dir="${LAST_EPOCH_CHECKPOINT_DIR}_out" \
        checkpointer.checkpoint_files="$CHECKPOINT_FILES" \
        checkpointer.model_type=LLAMA3 \
        tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
        tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
        > "${LOG_DIR}/eval_wikitext_float.log" 2>&1 &
    echo "Evaluating hellaswag, logging to ${LOG_DIR}/eval_hellaswag_float.log"
    CUDA_VISIBLE_DEVICES="$((FIRST_GPU+1))" tune run eleuther_eval --config eleuther_evaluation \
        batch_size=1 \
        tasks=[hellaswag] \
        model._component_="$MODEL_COMPONENT" \
        checkpointer._component_="$CHECKPOINTER" \
        checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
        checkpointer.output_dir="${LAST_EPOCH_CHECKPOINT_DIR}_out" \
        checkpointer.checkpoint_files="$CHECKPOINT_FILES" \
        checkpointer.model_type=LLAMA3 \
        tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
        tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
        > "${LOG_DIR}/eval_hellaswag_float.log" 2>&1 &
    wait
fi
