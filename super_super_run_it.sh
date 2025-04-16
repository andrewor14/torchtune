LOG_DIR="/home/andrewor/local/logs/tune"

LLAMA3_LOG_DIR="${LOG_DIR}/4-16/full-experiments-llama3-8b"
LLAMA3_1_LOG_DIR="${LOG_DIR}/4-16/full-experiments-llama3.1-8b"
LLAMA3_2_LOG_DIR="${LOG_DIR}/4-16/full-experiments-llama3.2-3b"

mkdir -p "$LLAMA3_LOG_DIR"
mkdir -p "$LLAMA3_1_LOG_DIR"
mkdir -p "$LLAMA3_2_LOG_DIR"

echo "Running Llama3-8B experiments..."
MODEL="Llama3-8B" ./super_run_it.sh
mv "$LOG_DIR/fp8"* "$LLAMA3_LOG_DIR"
mv "$LOG_DIR/full"* "$LLAMA3_LOG_DIR"

echo "Running Llama3.1-8B experiments..."
MODEL="Llama3.1-8B" ./super_run_it.sh
mv "$LOG_DIR/fp8"* "$LLAMA3_1_LOG_DIR"
mv "$LOG_DIR/full"* "$LLAMA3_1_LOG_DIR"

echo "Running Llama3.2-3B experiments..."
MODEL="Llama3.2-3B" BATCH_SIZE="32" ./super_run_it.sh
mv "$LOG_DIR/fp8"* "$LLAMA3_2_LOG_DIR"
mv "$LOG_DIR/full"* "$LLAMA3_2_LOG_DIR"
