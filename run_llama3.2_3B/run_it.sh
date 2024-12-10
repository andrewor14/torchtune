# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export NCCL_SHM_DISABLE=0
export USE_QAT="${USE_QAT:-False}"

echo "Running 8da4w LoRA! With QAT? $USE_QAT"

# Pick the right path
if [[ "$USE_QAT" == "True" ]] || [[ "$USE_QAT" == "true" ]]; then
    LOG_DIR="/home/andrewor/local/logs/tune/qat_lora"
else
    LOG_DIR="/home/andrewor/local/logs/tune/lora_baseline"
fi

# Delete the old log dir if it exists
if [[ -d "$LOG_DIR" ]]; then
    echo "Removing $LOG_DIR..."
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

# Common configs
EPOCHS=1
LAST_EPOCH_INDEX=0
BATCH_SIZE=8
LORA_RANK=16
LORA_ALPHA=32
GROUP_SIZE=256
LEARNING_RATE=2e-5
GRADIENT_ACCUMULATION_STEPS=1
QUANTIZER="torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer"
LAST_EPOCH_CHECKPOINT_DIR="${LOG_DIR}/epoch_${LAST_EPOCH_INDEX}"

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    NUM_GPUS="$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)"
else
    NUM_GPUS=8
fi


if [[ "$USE_QAT" == "True" ]] || [[ "$USE_QAT" == "true" ]]; then
    tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" qat_lora_finetune_distributed --config llama3_2/3B_qat_lora \
        batch_size="$BATCH_SIZE" \
        epochs="$EPOCHS" \
        model.lora_rank="$LORA_RANK" \
        model.lora_alpha="$LORA_ALPHA" \
        optimizer.lr="$LEARNING_RATE" \
        gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
        enable_activation_checkpointing="true" \
        quantizer._component_="$QUANTIZER" \
        quantizer.groupsize="$GROUP_SIZE" \
        checkpointer.output_dir="$LOG_DIR" \
        metric_logger.log_dir="${LOG_DIR}/metrics" \
        output_dir="${LOG_DIR}/metrics" \
        > "${LOG_DIR}/run.log" 2>&1
else
    tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" lora_finetune_distributed --config llama3_2/3B_lora \
        batch_size="$BATCH_SIZE" \
        epochs="$EPOCHS" \
        model.lora_rank="$LORA_RANK" \
        model.lora_alpha="$LORA_ALPHA" \
        optimizer.lr="$LEARNING_RATE" \
        gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
        enable_activation_checkpointing="true" \
        checkpointer.output_dir="$LOG_DIR" \
        metric_logger.log_dir="${LOG_DIR}/metrics" \
        output_dir="${LOG_DIR}/metrics" \
        > "${LOG_DIR}/run.log" 2>&1
fi

cp "${LOG_DIR}/base_model/config.json" "$LAST_EPOCH_CHECKPOINT_DIR"
