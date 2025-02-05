LOG_DIR="${LOG_DIR:-/home/andrewor/local/logs/tune}"
FP8_LOG_DIR="${LOG_DIR}/fp8_quantized_training"
FULL_FINETUNE_LOG_DIR="${LOG_DIR}/full_finetune_distributed_baseline"

FP8_METRICS_FILE="$(find "${FP8_LOG_DIR}/metrics" -name log*txt)"
FP8_TOKENS_PER_SECOND="$(cat "$FP8_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FP8_PEAK_MEMORY_ACTIVE="$(cat "$FP8_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FP8_PEAK_MEMORY_ALLOC="$(cat "$FP8_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FP8_PEAK_MEMORY_RESERVED="$(cat "$FP8_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FP8_EVAL_FILE="${FP8_LOG_DIR}/eval.log"
FP8_PERPLEXITY=$(grep "|word_perplexity|" "${FP8_LOG_DIR}/eval_wikitext_float.log" | awk -F'|' '{print $8}' | tail -n 1)
FP8_ACC=$(grep "|acc_norm" "${FP8_LOG_DIR}/eval_hellaswag_float.log" | awk -F'|' '{print $8}' | tail -n 1)

FULL_FINETUNE_METRICS_FILE="$(find "${FULL_FINETUNE_LOG_DIR}/metrics" -name log*txt)"
FULL_FINETUNE_TOKENS_PER_SECOND="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_ACTIVE="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_ALLOC="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_RESERVED="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_PERPLEXITY=$(grep "|word_perplexity|" "${FULL_FINETUNE_LOG_DIR}/eval_wikitext_float.log" | awk -F'|' '{print $8}' | tail -n 1)
FULL_ACC=$(grep "|acc_norm" "${FULL_FINETUNE_LOG_DIR}/eval_hellaswag_float.log" | awk -F'|' '{print $8}' | tail -n 1)

# FP8 vs full finetune
FP8_VS_FULL_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($FP8_TOKENS_PER_SECOND - $FULL_FINETUNE_TOKENS_PER_SECOND) / $FULL_FINETUNE_TOKENS_PER_SECOND * 100)")"
FP8_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ACTIVE - $FP8_PEAK_MEMORY_ACTIVE) / $FULL_FINETUNE_PEAK_MEMORY_ACTIVE * 100)")"
FP8_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ALLOC - $FP8_PEAK_MEMORY_ALLOC) / $FULL_FINETUNE_PEAK_MEMORY_ALLOC * 100)")"
FP8_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_RESERVED - $FP8_PEAK_MEMORY_RESERVED) / $FULL_FINETUNE_PEAK_MEMORY_RESERVED * 100)")"

echo "fp8 metrics: $FP8_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $FP8_TOKENS_PER_SECOND"
echo "  peak_memory_active: $FP8_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $FP8_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $FP8_PEAK_MEMORY_RESERVED"
echo "  word_perplexity: $FP8_PERPLEXITY"
echo "  acc_norm: $FP8_ACC"
echo ""

echo "full_finetune metrics: $FULL_FINETUNE_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $FULL_FINETUNE_TOKENS_PER_SECOND"
echo "  peak_memory_active: $FULL_FINETUNE_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $FULL_FINETUNE_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $FULL_FINETUNE_PEAK_MEMORY_RESERVED"
echo "  word_perplexity: $FULL_PERPLEXITY"
echo "  acc_norm: $FULL_ACC"
echo ""

echo "fp8 stats compared to raw full finetune:"
echo "  tokens_per_second_per_gpu % increase: ${FP8_VS_FULL_TOKENS_PER_SECOND_INCREASE}"
echo "  peak_memory_active % decrease: ${FP8_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE}"
echo "  peak_memory_alloc % decrease: ${FP8_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE}"
echo "  peak_memory_reserved % decrease: ${FP8_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE}"
echo ""
