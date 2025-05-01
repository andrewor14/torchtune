# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ENABLE_QAT="${ENABLE_QAT:-true}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-32}"
RANGE_LEARNING="${RANGE_LEARNING:-false}"
DATASET="${DATASET:-torchtune.datasets.alpaca_cleaned_dataset}"
EPOCHS=1
LAST_EPOCH_INDEX=0
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-null}"
export NCCL_SHM_DISABLE=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
FIRST_GPU="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | head -n 1)"


# Model type
if [[ -z "$MODEL" ]]; then
    MODEL="Llama3-8B"
fi
if [[ "$MODEL" == "Llama3.2-3B" ]]; then
    if [[ "$ENABLE_QAT" == "true" ]]; then
        CONFIG="llama3_2/3B_qat_full"
    else
        CONFIG="llama3_2/3B_full"
    fi
    MODEL_COMPONENT="torchtune.models.llama3_2.llama3_2_3b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00002.safetensors,model-00002-of-00002.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00002-8da4w.ckpt"
elif [[ "$MODEL" == "Llama3.1-8B" ]]; then
    if [[ "$ENABLE_QAT" == "true" ]]; then
        CONFIG="llama3_1/8B_qat_full"
    else
        CONFIG="llama3_1/8B_full"
    fi
    MODEL_COMPONENT="torchtune.models.llama3_1.llama3_1_8b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00004.safetensors,model-00002-of-00004.safetensors,model-00003-of-00004.safetensors,model-00004-of-00004.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00004-8da4w.ckpt"
elif [[ "$MODEL" == "Llama3-8B" ]]; then
    if [[ "$ENABLE_QAT" == "true" ]]; then
        CONFIG="llama3/8B_qat_full"
    else
        CONFIG="llama3/8B_full"
    fi
    MODEL_COMPONENT="torchtune.models.llama3.llama3_8b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00004.safetensors,model-00002-of-00004.safetensors,model-00003-of-00004.safetensors,model-00004-of-00004.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00004-8da4w.ckpt"
else
    echo "Unknown model $MODEL"
    exit 1
fi


# Experiment type
if [[ "$ENABLE_QAT" == "true" ]]; then
    if [[ "$RANGE_LEARNING" == "true" ]]; then
        DIR_NAME="${MODEL}_qat_range_learning"
    else
        DIR_NAME="${MODEL}_qat"
    fi
else
    DIR_NAME="${MODEL}_full"
fi
LOG_DIR="/home/${USER}/local/logs/tune/${DIR_NAME}"
LAST_EPOCH_CHECKPOINT_DIR="${LOG_DIR}/epoch_${LAST_EPOCH_INDEX}"


# Finetune
if [[ "$SKIP_FINETUNE" != "true" ]]; then
    # Delete the old log dir if it exists
    if [[ -d "$LOG_DIR" ]]; then
        echo "Removing $LOG_DIR..."
        rm -rf "$LOG_DIR"
    fi
    mkdir -p "$LOG_DIR"

    echo "Running finetuning, logging to ${LOG_DIR}/run.log"
    if [[ "$ENABLE_QAT" == "true" ]]; then
        tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" qat_distributed --config "$CONFIG" \
            epochs="$EPOCHS" \
            batch_size="$BATCH_SIZE" \
            dataset._component_="$DATASET" \
            checkpointer.output_dir="$LOG_DIR" \
            output_dir="${LOG_DIR}/metrics" \
            metric_logger.log_dir="${LOG_DIR}/metrics" \
            max_steps_per_epoch="$MAX_STEPS_PER_EPOCH" \
            quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer \
            quantizer.groupsize="$GROUP_SIZE" \
            quantizer.range_learning="$RANGE_LEARNING" \
            > "${LOG_DIR}/run.log" 2>&1
    else
        tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" full_finetune_distributed --config "$CONFIG" \
            epochs="$EPOCHS" \
            batch_size="$BATCH_SIZE" \
            dataset._component_="$DATASET" \
            checkpointer.output_dir="$LOG_DIR" \
            output_dir="${LOG_DIR}/metrics" \
            metric_logger.log_dir="${LOG_DIR}/metrics" \
            max_steps_per_epoch="$MAX_STEPS_PER_EPOCH" \
            > "${LOG_DIR}/run.log" 2>&1
    fi
fi


# Eval
if [[ "$SKIP_EVAL" != "true" ]]; then
    echo "Quantizing, logging to ${LOG_DIR}/quantize.log"
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run quantize --config quantization \
        model._component_="$MODEL_COMPONENT" \
        checkpointer._component_="$CHECKPOINTER" \
        checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
        checkpointer.output_dir="${LAST_EPOCH_CHECKPOINT_DIR}_out" \
        checkpointer.checkpoint_files="$CHECKPOINT_FILES" \
        checkpointer.model_type=LLAMA3 \
        quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
        quantizer.groupsize="$GROUP_SIZE" \
        > "${LOG_DIR}/quantize.log" 2>&1
    mv "${LAST_EPOCH_CHECKPOINT_DIR}_out/${QUANTIZED_CHECKPOINT_FILE}" "$LAST_EPOCH_CHECKPOINT_DIR"

    echo "Evaluating wikitext (float), logging to ${LOG_DIR}/eval_wikitext_float.log"
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

    echo "Evaluating wikitext (quantized), logging to ${LOG_DIR}/eval_wikitext_quantized.log"
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
        batch_size=1 \
        tasks=[wikitext] \
        model._component_="$MODEL_COMPONENT" \
        checkpointer._component_="torchtune.training.FullModelTorchTuneCheckpointer" \
        checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
        checkpointer.output_dir="${LAST_EPOCH_CHECKPOINT_DIR}_out" \
        checkpointer.checkpoint_files="[$QUANTIZED_CHECKPOINT_FILE]" \
        checkpointer.model_type=LLAMA3 \
        tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
        tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
        quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
        quantizer.groupsize="$GROUP_SIZE" \
        > "${LOG_DIR}/eval_wikitext_quantized.log" 2>&1 &

    echo "Evaluating hellaswag (float), logging to ${LOG_DIR}/eval_hellaswag_float.log"
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

    echo "Evaluating hellaswag (quantized), logging to ${LOG_DIR}/eval_hellaswag_quantized.log"
    CUDA_VISIBLE_DEVICES="$((FIRST_GPU+2))" tune run eleuther_eval --config eleuther_evaluation \
        batch_size=1 \
        tasks=[hellaswag] \
        model._component_="$MODEL_COMPONENT" \
        checkpointer._component_="torchtune.training.FullModelTorchTuneCheckpointer" \
        checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
        checkpointer.output_dir="${LAST_EPOCH_CHECKPOINT_DIR}_out" \
        checkpointer.checkpoint_files="[$QUANTIZED_CHECKPOINT_FILE]" \
        checkpointer.model_type=LLAMA3 \
        tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
        tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
        quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
        quantizer.groupsize="$GROUP_SIZE" \
        > "${LOG_DIR}/eval_hellaswag_quantized.log" 2>&1 &
    wait
fi
