LOG_DIR="${LOG_DIR:-/home/${USER}/local/logs/tune}"
FP8_LOG_DIR="${LOG_DIR}/fp8_quantized_training"
INT8_LOG_DIR="${LOG_DIR}/int8_quantized_training"
INT8_MP_LOG_DIR="${LOG_DIR}/int8_mixed_precision_quantized_training"
FULL_FINETUNE_LOG_DIR="${LOG_DIR}/full_finetune_distributed_baseline"

FULL_FINETUNE_METRICS_FILE="$(find "${FULL_FINETUNE_LOG_DIR}/metrics" -name log*txt)"
FULL_FINETUNE_TOKENS_PER_SECOND="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_ACTIVE="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_ALLOC="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_RESERVED="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_PERPLEXITY=$(grep "|word_perplexity|" "${FULL_FINETUNE_LOG_DIR}/eval_wikitext_float.log" | awk -F'|' '{print $8}' | tail -n 1)
FULL_ACC=$(grep "|acc_norm" "${FULL_FINETUNE_LOG_DIR}/eval_hellaswag_float.log" | awk -F'|' '{print $8}' | tail -n 1)

FP8_METRICS_FILE="$(find "${FP8_LOG_DIR}/metrics" -name log*txt)"
if [[ -f "$FP8_METRICS_FILE" ]]; then
    FP8_TOKENS_PER_SECOND="$(cat "$FP8_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    FP8_PEAK_MEMORY_ACTIVE="$(cat "$FP8_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    FP8_PEAK_MEMORY_ALLOC="$(cat "$FP8_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    FP8_PEAK_MEMORY_RESERVED="$(cat "$FP8_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    FP8_EVAL_FILE="${FP8_LOG_DIR}/eval.log"
    FP8_PERPLEXITY=$(grep "|word_perplexity|" "${FP8_LOG_DIR}/eval_wikitext_float.log" | awk -F'|' '{print $8}' | tail -n 1)
    FP8_ACC=$(grep "|acc_norm" "${FP8_LOG_DIR}/eval_hellaswag_float.log" | awk -F'|' '{print $8}' | tail -n 1)

    # FP8 vs full finetune
    FP8_VS_FULL_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($FP8_TOKENS_PER_SECOND - $FULL_FINETUNE_TOKENS_PER_SECOND) / $FULL_FINETUNE_TOKENS_PER_SECOND * 100)")"
    FP8_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ACTIVE - $FP8_PEAK_MEMORY_ACTIVE) / $FULL_FINETUNE_PEAK_MEMORY_ACTIVE * 100)")"
    FP8_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ALLOC - $FP8_PEAK_MEMORY_ALLOC) / $FULL_FINETUNE_PEAK_MEMORY_ALLOC * 100)")"
    FP8_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_RESERVED - $FP8_PEAK_MEMORY_RESERVED) / $FULL_FINETUNE_PEAK_MEMORY_RESERVED * 100)")"
fi

INT8_METRICS_FILE="$(find "${INT8_LOG_DIR}/metrics" -name log*txt)"
if [[ -f "$INT8_METRICS_FILE" ]]; then
    INT8_TOKENS_PER_SECOND="$(cat "$INT8_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_PEAK_MEMORY_ACTIVE="$(cat "$INT8_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_PEAK_MEMORY_ALLOC="$(cat "$INT8_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_PEAK_MEMORY_RESERVED="$(cat "$INT8_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_EVAL_FILE="${INT8_LOG_DIR}/eval.log"
    INT8_PERPLEXITY=$(grep "|word_perplexity|" "${INT8_LOG_DIR}/eval_wikitext_float.log" | awk -F'|' '{print $8}' | tail -n 1)
    INT8_ACC=$(grep "|acc_norm" "${INT8_LOG_DIR}/eval_hellaswag_float.log" | awk -F'|' '{print $8}' | tail -n 1)

    # INT8 vs full finetune
    INT8_VS_FULL_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($INT8_TOKENS_PER_SECOND - $FULL_FINETUNE_TOKENS_PER_SECOND) / $FULL_FINETUNE_TOKENS_PER_SECOND * 100)")"
    INT8_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ACTIVE - $INT8_PEAK_MEMORY_ACTIVE) / $FULL_FINETUNE_PEAK_MEMORY_ACTIVE * 100)")"
    INT8_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ALLOC - $INT8_PEAK_MEMORY_ALLOC) / $FULL_FINETUNE_PEAK_MEMORY_ALLOC * 100)")"
    INT8_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_RESERVED - $INT8_PEAK_MEMORY_RESERVED) / $FULL_FINETUNE_PEAK_MEMORY_RESERVED * 100)")"
fi

INT8_MP_METRICS_FILE="$(find "${INT8_MP_LOG_DIR}/metrics" -name log*txt)"
if [[ -f "$INT8_MP_METRICS_FILE" ]]; then
    INT8_MP_TOKENS_PER_SECOND="$(cat "$INT8_MP_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_MP_PEAK_MEMORY_ACTIVE="$(cat "$INT8_MP_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_MP_PEAK_MEMORY_ALLOC="$(cat "$INT8_MP_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_MP_PEAK_MEMORY_RESERVED="$(cat "$INT8_MP_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
    INT8_MP_EVAL_FILE="${INT8_MP_LOG_DIR}/eval.log"
    INT8_MP_PERPLEXITY=$(grep "|word_perplexity|" "${INT8_MP_LOG_DIR}/eval_wikitext_float.log" | awk -F'|' '{print $8}' | tail -n 1)
    INT8_MP_ACC=$(grep "|acc_norm" "${INT8_MP_LOG_DIR}/eval_hellaswag_float.log" | awk -F'|' '{print $8}' | tail -n 1)

    # INT8_MP_ vs full finetune
    INT8_MP_VS_FULL_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($INT8_MP_TOKENS_PER_SECOND - $FULL_FINETUNE_TOKENS_PER_SECOND) / $FULL_FINETUNE_TOKENS_PER_SECOND * 100)")"
    INT8_MP_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ACTIVE - $INT8_MP_PEAK_MEMORY_ACTIVE) / $FULL_FINETUNE_PEAK_MEMORY_ACTIVE * 100)")"
    INT8_MP_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ALLOC - $INT8_MP_PEAK_MEMORY_ALLOC) / $FULL_FINETUNE_PEAK_MEMORY_ALLOC * 100)")"
    INT8_MP_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_RESERVED - $INT8_MP_PEAK_MEMORY_RESERVED) / $FULL_FINETUNE_PEAK_MEMORY_RESERVED * 100)")"
fi

echo "full_finetune metrics: $FULL_FINETUNE_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $FULL_FINETUNE_TOKENS_PER_SECOND"
echo "  peak_memory_active: $FULL_FINETUNE_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $FULL_FINETUNE_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $FULL_FINETUNE_PEAK_MEMORY_RESERVED"
echo "  word_perplexity: $FULL_PERPLEXITY"
echo "  acc_norm: $FULL_ACC"
echo ""

if [[ -f "$FP8_METRICS_FILE" ]]; then
    echo "fp8 metrics: $FP8_METRICS_FILE"
    echo "  tokens_per_second_per_gpu: $FP8_TOKENS_PER_SECOND"
    echo "  peak_memory_active: $FP8_PEAK_MEMORY_ACTIVE"
    echo "  peak_memory_alloc: $FP8_PEAK_MEMORY_ALLOC"
    echo "  peak_memory_reserved: $FP8_PEAK_MEMORY_RESERVED"
    echo "  word_perplexity: $FP8_PERPLEXITY"
    echo "  acc_norm: $FP8_ACC"
    echo ""
    echo "fp8 stats compared to raw full finetune:"
    echo "  tokens_per_second_per_gpu % increase: ${FP8_VS_FULL_TOKENS_PER_SECOND_INCREASE}"
    echo "  peak_memory_active % decrease: ${FP8_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE}"
    echo "  peak_memory_alloc % decrease: ${FP8_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE}"
    echo "  peak_memory_reserved % decrease: ${FP8_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE}"
    echo ""
else
    echo "fp8 quantized finetuning metrics file not found: ${FP8_LOG_DIR}/metrics"
fi

if [[ -f "$INT8_METRICS_FILE" ]]; then
    echo "int8 metrics: $INT8_METRICS_FILE"
    echo "  tokens_per_second_per_gpu: $INT8_TOKENS_PER_SECOND"
    echo "  peak_memory_active: $INT8_PEAK_MEMORY_ACTIVE"
    echo "  peak_memory_alloc: $INT8_PEAK_MEMORY_ALLOC"
    echo "  peak_memory_reserved: $INT8_PEAK_MEMORY_RESERVED"
    echo "  word_perplexity: $INT8_PERPLEXITY"
    echo "  acc_norm: $INT8_ACC"
    echo ""
    echo "int8 stats compared to raw full finetune:"
    echo "  tokens_per_second_per_gpu % increase: ${INT8_VS_FULL_TOKENS_PER_SECOND_INCREASE}"
    echo "  peak_memory_active % decrease: ${INT8_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE}"
    echo "  peak_memory_alloc % decrease: ${INT8_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE}"
    echo "  peak_memory_reserved % decrease: ${INT8_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE}"
    echo ""
else
    echo "int8 quantized finetuning metrics file not found in ${INT8_LOG_DIR}/metrics"
fi

if [[ -f "$INT8_MP_METRICS_FILE" ]]; then
    echo "int8 mixed precision metrics: $INT8_MP_METRICS_FILE"
    echo "  tokens_per_second_per_gpu: $INT8_MP_TOKENS_PER_SECOND"
    echo "  peak_memory_active: $INT8_MP_PEAK_MEMORY_ACTIVE"
    echo "  peak_memory_alloc: $INT8_MP_PEAK_MEMORY_ALLOC"
    echo "  peak_memory_reserved: $INT8_MP_PEAK_MEMORY_RESERVED"
    echo "  word_perplexity: $INT8_MP_PERPLEXITY"
    echo "  acc_norm: $INT8_MP_ACC"
    echo ""
    echo "int8 mixed precision stats compared to raw full finetune:"
    echo "  tokens_per_second_per_gpu % increase: ${INT8_MP_VS_FULL_TOKENS_PER_SECOND_INCREASE}"
    echo "  peak_memory_active % decrease: ${INT8_MP_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE}"
    echo "  peak_memory_alloc % decrease: ${INT8_MP_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE}"
    echo "  peak_memory_reserved % decrease: ${INT8_MP_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE}"
    echo ""
else
    echo "int8 mixed precision quantized finetuning metrics file not found in ${INT8_MP_LOG_DIR}/metrics"
fi
