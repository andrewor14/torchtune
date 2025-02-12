LOG_DIR="${LOG_DIR:-/home/${USER}/local/logs/tune}"
LAMMA_3_1_LOG_DIR="${LOG_DIR}/full_3_1_trace"
LLAMA_3_LOG_DIR="${LOG_DIR}/full_3_trace"

LLAMA_3_METRICS_FILE="$(find "${LLAMA_3_LOG_DIR}/metrics" -name log*txt)"
LLAMA_3_TOKENS_PER_SECOND="$(cat "$LLAMA_3_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
LLAMA_3_PEAK_MEMORY_ACTIVE="$(cat "$LLAMA_3_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
LLAMA_3_PEAK_MEMORY_ALLOC="$(cat "$LLAMA_3_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
LLAMA_3_PEAK_MEMORY_RESERVED="$(cat "$LLAMA_3_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"

LAMMA_3_1_METRICS_FILE="$(find "${LAMMA_3_1_LOG_DIR}/metrics" -name log*txt)"
if [[ -f "$LAMMA_3_1_METRICS_FILE" ]]; then
    LAMMA_3_1_TOKENS_PER_SECOND="$(cat "$LAMMA_3_1_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    LAMMA_3_1_PEAK_MEMORY_ACTIVE="$(cat "$LAMMA_3_1_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    LAMMA_3_1_PEAK_MEMORY_ALLOC="$(cat "$LAMMA_3_1_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    LAMMA_3_1_PEAK_MEMORY_RESERVED="$(cat "$LAMMA_3_1_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    LAMMA_3_1_EVAL_FILE="${LAMMA_3_1_LOG_DIR}/eval.log"

    # LAMMA_3_1 vs full finetune
    LAMMA_3_1_VS_FULL_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($LAMMA_3_1_TOKENS_PER_SECOND - $LLAMA_3_TOKENS_PER_SECOND) / $LLAMA_3_TOKENS_PER_SECOND * 100)")"
    LAMMA_3_1_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($LLAMA_3_PEAK_MEMORY_ACTIVE - $LAMMA_3_1_PEAK_MEMORY_ACTIVE) / $LLAMA_3_PEAK_MEMORY_ACTIVE * 100)")"
    LAMMA_3_1_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($LLAMA_3_PEAK_MEMORY_ALLOC - $LAMMA_3_1_PEAK_MEMORY_ALLOC) / $LLAMA_3_PEAK_MEMORY_ALLOC * 100)")"
    LAMMA_3_1_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($LLAMA_3_PEAK_MEMORY_RESERVED - $LAMMA_3_1_PEAK_MEMORY_RESERVED) / $LLAMA_3_PEAK_MEMORY_RESERVED * 100)")"
fi

echo "Llama3 metrics: $LLAMA_3_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $LLAMA_3_TOKENS_PER_SECOND"
echo "  peak_memory_active: $LLAMA_3_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $LLAMA_3_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $LLAMA_3_PEAK_MEMORY_RESERVED"
echo ""

if [[ -f "$LAMMA_3_1_METRICS_FILE" ]]; then
    echo "Llama3.1 metrics: $LAMMA_3_1_METRICS_FILE"
    echo "  tokens_per_second_per_gpu: $LAMMA_3_1_TOKENS_PER_SECOND"
    echo "  peak_memory_active: $LAMMA_3_1_PEAK_MEMORY_ACTIVE"
    echo "  peak_memory_alloc: $LAMMA_3_1_PEAK_MEMORY_ALLOC"
    echo "  peak_memory_reserved: $LAMMA_3_1_PEAK_MEMORY_RESERVED"
    echo ""
    echo "Llama3.1 stats compared to raw full finetune:"
    echo "  tokens_per_second_per_gpu % increase: ${LAMMA_3_1_VS_FULL_TOKENS_PER_SECOND_INCREASE}"
    echo "  peak_memory_active % decrease: ${LAMMA_3_1_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE}"
    echo "  peak_memory_alloc % decrease: ${LAMMA_3_1_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE}"
    echo "  peak_memory_reserved % decrease: ${LAMMA_3_1_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE}"
    echo ""
else
    echo "Llama3.1 quantized finetuning metrics file not found: $LAMMA_3_1_METRICS_FILE"
fi

