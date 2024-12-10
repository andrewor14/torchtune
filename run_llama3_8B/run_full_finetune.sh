# Setup

export NCCL_SHM_DISABLE=0
EPOCHS=1
LAST_EPOCH_INDEX=0
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
GROUP_SIZE=256
USE_QAT="${USE_QAT:-true}"

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    NUM_GPUS="$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)"
    FIRST_GPU="$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | head -n 1)"
else
    NUM_GPUS=8
    FIRST_GPU=0
fi

# Pick the right path, recipe, and config
if [[ "$USE_QAT" == "True" ]] || [[ "$USE_QAT" == "true" ]]; then
    LOG_DIR="/home/andrewor/local/logs/tune/test_update_qat"
    RECIPE="qat_distributed"
    CONFIG="llama3/8B_qat_full"
else
    LOG_DIR="/home/andrewor/local/logs/tune/full_finetune_distributed_baseline"
    RECIPE="full_finetune_distributed"
    CONFIG="llama3/8B_full"
fi
LAST_EPOCH_CHECKPOINT_DIR="${LOG_DIR}/epoch_${LAST_EPOCH_INDEX}"

# Delete the old log dir if it exists
if [[ -d "$LOG_DIR" ]]; then
    echo "Removing $LOG_DIR..."
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"


# Run experiment

echo "Finetuning: logging to ${LOG_DIR}/run.log"

tune run --nnodes 1 --nproc_per_node "$NUM_GPUS" "$RECIPE" --config "$CONFIG" \
    epochs="$EPOCHS" \
    batch_size="$BATCH_SIZE" \
    gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    dataset._component_=torchtune.datasets.alpaca_cleaned_dataset \
    checkpointer.output_dir="$LOG_DIR" \
    output_dir="${LOG_DIR}/metrics" \
    metric_logger.log_dir="${LOG_DIR}/metrics" \
    quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer \
    > "${LOG_DIR}/run.log" 2>&1

echo "Quantizing: logging to ${LOG_DIR}/quantize.log"

CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run quantize --config quantization \
    model._component_=torchtune.models.llama3.llama3_8b \
    checkpointer._component_=torchtune.training.FullModelMetaCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=[ft-model-00001-of-00001.bin] \
    checkpointer.model_type=LLAMA3 \
    quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
    > "${LOG_DIR}/quantize.log" 2>&1

echo "Evaluating: logging to ${LOG_DIR}/eval.log"

CUDA_VISIBLE_DEVICES="$FIRST_GPU" tune run eleuther_eval --config eleuther_evaluation \
    batch_size=16 \
    tasks=[wikitext,hellaswag] \
    model._component_=torchtune.models.llama3.llama3_8b \
    checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=[ft-model-00001-of-00001-8da4w.pt] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer \
    > "${LOG_DIR}/eval.log" 2>&1 &

echo "Evaluating (float): logging to ${LOG_DIR}/eval_wikitext_float.log and ${LOG_DIR}/eval_hellaswag_float.log"

CUDA_VISIBLE_DEVICES="$((FIRST_GPU+1))" tune run eleuther_eval --config eleuther_evaluation \
    batch_size=1 \
    tasks=[wikitext] \
    model._component_=torchtune.models.llama3.llama3_8b \
    checkpointer._component_=torchtune.training.FullModelMetaCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=[ft-model-00001-of-00001.bin] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    > "${LOG_DIR}/eval_wikitext_float.log" 2>&1 &

CUDA_VISIBLE_DEVICES="$((FIRST_GPU+2))" tune run eleuther_eval --config eleuther_evaluation \
    batch_size=16 \
    tasks=[hellaswag] \
    model._component_=torchtune.models.llama3.llama3_8b \
    checkpointer._component_=torchtune.training.FullModelMetaCheckpointer \
    checkpointer.checkpoint_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.output_dir="$LAST_EPOCH_CHECKPOINT_DIR" \
    checkpointer.checkpoint_files=[ft-model-00001-of-00001.bin] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    > "${LOG_DIR}/eval_hellaswag_float.log" 2>&1 &

wait

cat "${LOG_DIR}/eval_wikitext_float.log" "${LOG_DIR}/eval_hellaswag_float.log" > "${LOG_DIR}/eval_float.log"
