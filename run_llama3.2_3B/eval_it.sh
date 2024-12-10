# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CHECKPOINT_FILE="${CHECKPOINT_FILE:-ft-model-00001-of-00002.safetensors,ft-model-00002-of-00002.safetensors}"
QUANTIZED_CHECKPOINT_FILE="${QUANTIZED_CHECKPOINT_FILE:-ft-model-00001-of-00002-8da4w.pt}"
GROUP_SIZE="${GROUP_SIZE:=256}"
LOG_BASE_DIR="${LOG_BASE_DIR:-/home/andrewor/local/logs/tune}"

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    FIRST_GPU="$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | head -n 1)"
else
    FIRST_GPU=0
fi

# Pick the right path
if [[ "$USE_QAT" == "True" ]] || [[ "$USE_QAT" == "true" ]]; then
    LOG_DIR="${LOG_BASE_DIR}/qat_lora"
else
    LOG_DIR="${LOG_BASE_DIR}/lora_baseline"
fi
LAST_EPOCH_CHECKPOINT_DIR="${LOG_DIR}/epoch_0"

echo "Quantizing LoRA model! With QAT? $USE_QAT"

CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run quantize --config quantization \
    model._component_=torchtune.models.llama3_2.llama3_2_3b \
    checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=["$CHECKPOINT_FILE"] \
    checkpointer.model_type=LLAMA3 \
    quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
    quantizer.groupsize="$GROUP_SIZE" \
    > "$LOG_DIR"/quantize.log 2>&1

echo "Evaling LoRA model! With QAT? $USE_QAT"

CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
    batch_size=16 \
    model._component_=torchtune.models.llama3_2.llama3_2_3b \
    checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=["$QUANTIZED_CHECKPOINT_FILE"] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    tasks=[wikitext,hellaswag] \
    quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
    quantizer.groupsize="$GROUP_SIZE" \
    > "$LOG_DIR"/eval.log 2>&1 &

echo "Evaling LoRA model FLOAT! With QAT? $USE_QAT"

CUDA_VISIBLE_DEVICES="$((FIRST_GPU+1))" tune run eleuther_eval --config eleuther_evaluation \
    batch_size=1 \
    model._component_=torchtune.models.llama3_2.llama3_2_3b \
    checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=["$CHECKPOINT_FILE"] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    tasks=[wikitext] \
    > "$LOG_DIR"/eval_wikitext_float.log 2>&1 &

CUDA_VISIBLE_DEVICES="$((FIRST_GPU+2))" tune run eleuther_eval --config eleuther_evaluation \
    batch_size=1 \
    model._component_=torchtune.models.llama3_2.llama3_2_3b \
    checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=["$CHECKPOINT_FILE"] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    tasks=[hellaswag] \
    > "$LOG_DIR"/eval_hellaswag_float.log 2>&1 &

wait

cat "${LOG_DIR}/eval_wikitext_float.log" "${LOG_DIR}/eval_hellaswag_float.log" > "${LOG_DIR}/eval_float.log"
