LOG_DIR="/home/andrewor/local/logs/tune/qt_w8"

if [[ -d "$LOG_DIR" ]]; then
    echo "Removing $LOG_DIR..."
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

echo "Running int8 weight-only quantized training..."
CUDA_VISIBLE_DEVICES=4,5,6,7 tune run --nnodes 1 --nproc_per_node 4 qat_distributed --config llama3/8B_qat_full \
    batch_size=8 \
    epochs=1 \
    max_steps_per_epoch=10 \
    checkpointer.output_dir="$LOG_DIR" \
    metric_logger.output_dir="${LOG_DIR}/alpaca-llama3-finetune" \
    > "${LOG_DIR}/run.log" 2>&1
