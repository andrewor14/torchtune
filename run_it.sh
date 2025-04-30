# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ENABLE_QAT="${ENABLE_QAT:-true}"
ENABLE_LORA="${ENABLE_LORA:-false}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-32}"
DATASET="${DATASET:-alpaca}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
EPOCHS="${EPOCHS:-1}"
LAST_EPOCH_INDEX=0
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-null}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
ENABLE_ACTIVATION_CHECKPOINTING="${ENABLE_ACTIVATION_CHECKPOINTING:-true}"

export NCCL_SHM_DISABLE=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
FIRST_GPU="$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | head -n 1)"


# Finetune type
if [[ "$ENABLE_QAT" == "true" ]]; then
    if [[ "$ENABLE_LORA" == "true" ]]; then
        FINETUNE_TYPE="qat_lora"
    else
        FINETUNE_TYPE="qat"
    fi
else
    if [[ "$ENABLE_LORA" == "true" ]]; then
        FINETUNE_TYPE="lora"
    else
        FINETUNE_TYPE="full"
    fi
fi


# Model type
if [[ -z "$MODEL" ]]; then
    MODEL="Llama3-8B"
fi
if [[ "$MODEL" == "Llama3.2-3B" ]]; then
    if [[ "$FINETUNE_TYPE" == "qat" ]]; then
        CONFIG="llama3_2/3B_qat_full"
    elif [[ "$FINETUNE_TYPE" == "qat_lora" ]]; then
        CONFIG="llama3_2/3B_qat_lora"
    elif [[ "$FINETUNE_TYPE" == "full" ]]; then
        CONFIG="llama3_2/3B_full"
    else
        CONFIG="llama3_2/3B_lora"
    fi
    MODEL_COMPONENT="torchtune.models.llama3_2.llama3_2_3b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00002.safetensors,model-00002-of-00002.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00002-8da4w.ckpt"
    MODEL_TYPE="LLAMA3"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    TOKENIZER_PATH="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model"
    TOKENIZER_MERGES_FILE="null"
elif [[ "$MODEL" == "Llama3.1-8B" ]]; then
    if [[ "$FINETUNE_TYPE" == "qat" ]]; then
        CONFIG="llama3_1/8B_qat_full"
    elif [[ "$FINETUNE_TYPE" == "qat_lora" ]]; then
        CONFIG="llama3_1/8B_qat_lora"
    elif [[ "$FINETUNE_TYPE" == "full" ]]; then
        CONFIG="llama3_1/8B_full"
    else
        CONFIG="llama3_1/8B_lora"
    fi
    MODEL_COMPONENT="torchtune.models.llama3_1.llama3_1_8b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00004.safetensors,model-00002-of-00004.safetensors,model-00003-of-00004.safetensors,model-00004-of-00004.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00004-8da4w.ckpt"
    MODEL_TYPE="LLAMA3"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    TOKENIZER_PATH="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model"
    TOKENIZER_MERGES_FILE="null"
elif [[ "$MODEL" == "Llama3-8B" ]]; then
    if [[ "$FINETUNE_TYPE" == "qat" ]]; then
        CONFIG="llama3/8B_qat_full"
    elif [[ "$FINETUNE_TYPE" == "qat_lora" ]]; then
        CONFIG="llama3/8B_qat_lora"
    elif [[ "$FINETUNE_TYPE" == "full" ]]; then
        CONFIG="llama3/8B_full"
    else
        CONFIG="llama3/8B_lora"
    fi
    MODEL_COMPONENT="torchtune.models.llama3.llama3_8b"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00004.safetensors,model-00002-of-00004.safetensors,model-00003-of-00004.safetensors,model-00004-of-00004.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00004-8da4w.ckpt"
    MODEL_TYPE="LLAMA3"
    TOKENIZER_COMPONENT="torchtune.models.llama3.llama3_tokenizer"
    TOKENIZER_PATH="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model"
    TOKENIZER_MERGES_FILE="null"
elif [[ "$MODEL" == "Qwen3-1.7B" ]]; then
    if [[ "$FINETUNE_TYPE" == "qat" ]]; then
        CONFIG="qwen3/1.7B_qat_full"
    elif [[ "$FINETUNE_TYPE" == "qat_lora" ]]; then
        CONFIG="qwen3/1.7B_qat_lora"
    elif [[ "$FINETUNE_TYPE" == "full" ]]; then
        CONFIG="qwen3/1.7B_full"
    else
        CONFIG="qwen3/1.7B_lora"
    fi
    MODEL_COMPONENT="torchtune.models.qwen3.qwen3_1_7b_instruct"
    CHECKPOINTER="torchtune.training.FullModelHFCheckpointer"
    CHECKPOINT_FILES="[model-00001-of-00002.safetensors,model-00002-of-00002.safetensors]"
    QUANTIZED_CHECKPOINT_FILE="model-00001-of-00002-8da4w.ckpt"
    MODEL_TYPE="QWEN3"
    TOKENIZER_COMPONENT="torchtune.models.qwen3.qwen3_tokenizer"
    TOKENIZER_PATH="/tmp/Qwen3-1.7B/vocab.json"
    TOKENIZER_MERGES_FILE="/tmp/Qwen3-1.7B/merges.txt"
else
    echo "Unknown model $MODEL"
    exit 1
fi


# Experiment type
DIR_NAME="${MODEL}_${DATASET}_${FINETUNE_TYPE}"
if [[ "$FINETUNE_TYPE" == "qat" ]]; then
    RECIPE="qat_distributed"
elif [[ "$FINETUNE_TYPE" == "qat_lora" ]]; then
    RECIPE="qat_lora_finetune_distributed"
elif [[ "$FINETUNE_TYPE" == "full" ]]; then
    RECIPE="full_finetune_distributed"
else
    RECIPE="lora_finetune_distributed"
fi
if [[ "$ENABLE_QAT" ]]; then
    QUANTIZER="torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer"
else
    QUANTIZER="null"
    GROUP_SIZE="null"
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
    if [[ "$DATASET" == "alpaca" ]]; then
        tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" "$RECIPE" --config "$CONFIG" \
            epochs="$EPOCHS" \
            batch_size="$BATCH_SIZE" \
            dataset._component_="torchtune.datasets.alpaca_cleaned_dataset" \
            checkpointer.output_dir="$LOG_DIR" \
            output_dir="${LOG_DIR}/metrics" \
            metric_logger.log_dir="${LOG_DIR}/metrics" \
            max_steps_per_epoch="$MAX_STEPS_PER_EPOCH" \
            gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
            enable_activation_checkpointing="$ENABLE_ACTIVATION_CHECKPOINTING" \
            quantizer._component_="$QUANTIZER" \
            quantizer.groupsize="$GROUP_SIZE" \
            > "${LOG_DIR}/run.log" 2>&1
    elif [[ "$DATASET" == "c4" ]]; then
        tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" "$RECIPE" --config "$CONFIG" \
            epochs="$EPOCHS" \
            batch_size="$BATCH_SIZE" \
            tokenizer.max_seq_len="$MAX_SEQ_LEN" \
            dataset._component_="torchtune.datasets.text_completion_dataset" \
            dataset.source="allenai/c4" \
            dataset.column="text" \
            dataset.data_dir="realnewslike" \
            dataset.split="train" \
            checkpointer.output_dir="$LOG_DIR" \
            output_dir="${LOG_DIR}/metrics" \
            metric_logger.log_dir="${LOG_DIR}/metrics" \
            max_steps_per_epoch="$MAX_STEPS_PER_EPOCH" \
            gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
            enable_activation_checkpointing="$ENABLE_ACTIVATION_CHECKPOINTING" \
            quantizer._component_="$QUANTIZER" \
            quantizer.groupsize="$GROUP_SIZE" \
            > "${LOG_DIR}/run.log" 2>&1
    elif [[ "$DATASET" == "oasst1" ]]; then
        tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" "$RECIPE" --config "$CONFIG" \
            epochs="$EPOCHS" \
            batch_size="$BATCH_SIZE" \
            tokenizer.max_seq_len="$MAX_SEQ_LEN" \
            dataset._component_="torchtune.datasets.text_completion_dataset" \
            dataset.source="OpenAssistant/oasst1" \
            dataset.column="text" \
            dataset.split="train" \
            checkpointer.output_dir="$LOG_DIR" \
            output_dir="${LOG_DIR}/metrics" \
            metric_logger.log_dir="${LOG_DIR}/metrics" \
            max_steps_per_epoch="$MAX_STEPS_PER_EPOCH" \
            gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
            enable_activation_checkpointing="$ENABLE_ACTIVATION_CHECKPOINTING" \
            quantizer._component_="$QUANTIZER" \
            quantizer.groupsize="$GROUP_SIZE" \
            > "${LOG_DIR}/run.log" 2>&1
    else
        echo "Unrecognized dataset type: $DATASET"
        exit 1
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
        checkpointer.model_type="$MODEL_TYPE" \
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
        checkpointer.model_type="$MODEL_TYPE" \
        tokenizer._component_="$TOKENIZER_COMPONENT" \
        tokenizer.path="$TOKENIZER_PATH" \
        tokenizer.merges_file="$TOKENIZER_MERGES_FILE" \
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
        checkpointer.model_type="$MODEL_TYPE" \
        tokenizer._component_="$TOKENIZER_COMPONENT" \
        tokenizer.path="$TOKENIZER_PATH" \
        tokenizer.merges_file="$TOKENIZER_MERGES_FILE" \
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
        checkpointer.model_type="$MODEL_TYPE" \
        tokenizer._component_="$TOKENIZER_COMPONENT" \
        tokenizer.path="$TOKENIZER_PATH" \
        tokenizer.merges_file="$TOKENIZER_MERGES_FILE" \
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
        checkpointer.model_type="$MODEL_TYPE" \
        tokenizer._component_="$TOKENIZER_COMPONENT" \
        tokenizer.path="$TOKENIZER_PATH" \
        tokenizer.merges_file="$TOKENIZER_MERGES_FILE" \
        quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
        quantizer.groupsize="$GROUP_SIZE" \
        > "${LOG_DIR}/eval_hellaswag_quantized.log" 2>&1 &
    wait
fi
