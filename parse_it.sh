LOG_DIR="${LOG_DIR:-/home/andrewor/local/logs/tune}"
BASELINE_LORA_LOG_DIR="${LOG_DIR}/lora_baseline"
QAT_LORA_LOG_DIR="${LOG_DIR}/qat_lora"
QAT_LOG_DIR="${LOG_DIR}/test_update_qat"
FULL_FINETUNE_LOG_DIR="${LOG_DIR}/full_finetune_distributed_baseline"

BASELINE_LORA_METRICS_FILE="$(find "${BASELINE_LORA_LOG_DIR}/metrics" -name log*txt)"
BASELINE_LORA_TOKENS_PER_SECOND="$(cat "$BASELINE_LORA_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
BASELINE_LORA_PEAK_MEMORY_ACTIVE="$(cat "$BASELINE_LORA_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
BASELINE_LORA_PEAK_MEMORY_ALLOC="$(cat "$BASELINE_LORA_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
BASELINE_LORA_PEAK_MEMORY_RESERVED="$(cat "$BASELINE_LORA_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
BASELINE_LORA_EVAL_FILE="${BASELINE_LORA_LOG_DIR}/eval.log"
BASELINE_LORA_EVAL_FLOAT_FILE="${BASELINE_LORA_LOG_DIR}/eval_float.log"
BASELINE_LORA_PERPLEXITY=$(grep "|word_perplexity|" "$BASELINE_LORA_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)
BASELINE_LORA_FLOAT_PERPLEXITY=$(grep "|word_perplexity|" "$BASELINE_LORA_EVAL_FLOAT_FILE" | awk -F'|' '{print $8}' | tail -n 1)
BASELINE_LORA_ACC=$(grep "|acc_norm" "$BASELINE_LORA_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)
BASELINE_LORA_FLOAT_ACC=$(grep "|acc_norm" "$BASELINE_LORA_EVAL_FLOAT_FILE" | awk -F'|' '{print $8}' | tail -n 1)

QAT_LORA_METRICS_FILE="$(find "${QAT_LORA_LOG_DIR}/metrics" -name log*txt)"
QAT_LORA_TOKENS_PER_SECOND="$(cat "$QAT_LORA_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_LORA_PEAK_MEMORY_ACTIVE="$(cat "$QAT_LORA_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_LORA_PEAK_MEMORY_ALLOC="$(cat "$QAT_LORA_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_LORA_PEAK_MEMORY_RESERVED="$(cat "$QAT_LORA_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_LORA_EVAL_FILE="${QAT_LORA_LOG_DIR}/eval.log"
QAT_LORA_PERPLEXITY=$(grep "|word_perplexity|" "$QAT_LORA_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)
QAT_LORA_ACC=$(grep "|acc_norm" "$QAT_LORA_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)

QAT_METRICS_FILE="$(find "${QAT_LOG_DIR}/metrics" -name log*txt)"
QAT_TOKENS_PER_SECOND="$(cat "$QAT_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_PEAK_MEMORY_ACTIVE="$(cat "$QAT_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_PEAK_MEMORY_ALLOC="$(cat "$QAT_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_PEAK_MEMORY_RESERVED="$(cat "$QAT_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
QAT_EVAL_FILE="${QAT_LOG_DIR}/eval.log"
QAT_PERPLEXITY=$(grep "|word_perplexity|" "$QAT_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)
QAT_ACC=$(grep "|acc_norm" "$QAT_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)

FULL_FINETUNE_METRICS_FILE="$(find "${FULL_FINETUNE_LOG_DIR}/metrics" -name log*txt)"
FULL_FINETUNE_TOKENS_PER_SECOND="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $6}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_ACTIVE="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $7}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_ALLOC="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $8}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_FINETUNE_PEAK_MEMORY_RESERVED="$(cat "$FULL_FINETUNE_METRICS_FILE" | awk '{print $9}' | awk -F':' '{print $2}' | awk '{SUM += $1} END {print SUM/NR}')"
FULL_EVAL_FILE="${FULL_FINETUNE_LOG_DIR}/eval.log"
FULL_EVAL_FLOAT_FILE="${FULL_FINETUNE_LOG_DIR}/eval_float.log"
FULL_PERPLEXITY=$(grep "|word_perplexity|" "$FULL_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)
FULL_FLOAT_PERPLEXITY=$(grep "|word_perplexity|" "$FULL_EVAL_FLOAT_FILE" | awk -F'|' '{print $8}' | tail -n 1)
FULL_ACC=$(grep "|acc_norm" "$FULL_EVAL_FILE" | awk -F'|' '{print $8}' | tail -n 1)
FULL_FLOAT_ACC=$(grep "|acc_norm" "$FULL_EVAL_FLOAT_FILE" | awk -F'|' '{print $8}' | tail -n 1)

# QAT LoRA vs baseline LoRA
QAT_LORA_VS_LORA_TOKENS_PER_SECOND_DECREASE="$(python -c "print(($BASELINE_LORA_TOKENS_PER_SECOND - $QAT_LORA_TOKENS_PER_SECOND) / $BASELINE_LORA_TOKENS_PER_SECOND * 100)")"
QAT_LORA_VS_LORA_PEAK_MEMORY_ACTIVE_INCREASE="$(python -c "print(($QAT_LORA_PEAK_MEMORY_ACTIVE - $BASELINE_LORA_PEAK_MEMORY_ACTIVE) / $BASELINE_LORA_PEAK_MEMORY_ACTIVE * 100)")"
QAT_LORA_VS_LORA_PEAK_MEMORY_ALLOC_INCREASE="$(python -c "print(($QAT_LORA_PEAK_MEMORY_ALLOC - $BASELINE_LORA_PEAK_MEMORY_ALLOC) / $BASELINE_LORA_PEAK_MEMORY_ALLOC * 100)")"
QAT_LORA_VS_LORA_PEAK_MEMORY_RESERVED_INCREASE="$(python -c "print(($QAT_LORA_PEAK_MEMORY_RESERVED - $BASELINE_LORA_PEAK_MEMORY_RESERVED) / $BASELINE_LORA_PEAK_MEMORY_RESERVED * 100)")"
QAT_LORA_VS_LORA_PERPLEXITY_RECOVERED="$(python -c "print((1 - ($QAT_LORA_PERPLEXITY - $BASELINE_LORA_FLOAT_PERPLEXITY) / ($BASELINE_LORA_PERPLEXITY - $BASELINE_LORA_FLOAT_PERPLEXITY)) * 100)")"
QAT_LORA_VS_LORA_ACC_RECOVERED="$(python -c "print((1 - ($QAT_LORA_ACC - $BASELINE_LORA_FLOAT_ACC) / ($BASELINE_LORA_ACC - $BASELINE_LORA_FLOAT_ACC)) * 100)")"

# QAT LoRA vs regular QAT
QAT_LORA_VS_QAT_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($QAT_LORA_TOKENS_PER_SECOND - $QAT_TOKENS_PER_SECOND) / $QAT_TOKENS_PER_SECOND * 100)")"
QAT_LORA_VS_QAT_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($QAT_PEAK_MEMORY_ACTIVE - $QAT_LORA_PEAK_MEMORY_ACTIVE) / $QAT_PEAK_MEMORY_ACTIVE * 100)")"
QAT_LORA_VS_QAT_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($QAT_PEAK_MEMORY_ALLOC - $QAT_LORA_PEAK_MEMORY_ALLOC) / $QAT_PEAK_MEMORY_ALLOC * 100)")"
QAT_LORA_VS_QAT_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($QAT_PEAK_MEMORY_RESERVED - $QAT_LORA_PEAK_MEMORY_RESERVED) / $QAT_PEAK_MEMORY_RESERVED * 100)")"
QAT_LORA_VS_QAT_PERPLEXITY_DECREASE="$(python -c "print($QAT_PERPLEXITY - $QAT_LORA_PERPLEXITY)")"
QAT_LORA_VS_QAT_ACC_INCREASE="$(python -c "print(($QAT_LORA_ACC - $QAT_ACC) * 100)")"

# QAT LoRA vs full finetune
QAT_LORA_VS_FULL_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($QAT_LORA_TOKENS_PER_SECOND - $FULL_FINETUNE_TOKENS_PER_SECOND) / $FULL_FINETUNE_TOKENS_PER_SECOND * 100)")"
QAT_LORA_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ACTIVE - $QAT_LORA_PEAK_MEMORY_ACTIVE) / $FULL_FINETUNE_PEAK_MEMORY_ACTIVE * 100)")"
QAT_LORA_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ALLOC - $QAT_LORA_PEAK_MEMORY_ALLOC) / $FULL_FINETUNE_PEAK_MEMORY_ALLOC * 100)")"
QAT_LORA_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_RESERVED - $QAT_LORA_PEAK_MEMORY_RESERVED) / $FULL_FINETUNE_PEAK_MEMORY_RESERVED * 100)")"
QAT_LORA_VS_FULL_PERPLEXITY_RECOVERED="$(python -c "print((1 - ($QAT_LORA_PERPLEXITY - $FULL_FLOAT_PERPLEXITY) / ($FULL_PERPLEXITY - $FULL_FLOAT_PERPLEXITY)) * 100)")"
QAT_LORA_VS_FULL_ACC_RECOVERED="$(python -c "print((1 - ($QAT_LORA_ACC - $FULL_FLOAT_ACC) / ($FULL_ACC - $FULL_FLOAT_ACC)) * 100)")"

# LoRA vs full finetune
LORA_VS_FULL_TOKENS_PER_SECOND_INCREASE="$(python -c "print(($BASELINE_LORA_TOKENS_PER_SECOND - $FULL_FINETUNE_TOKENS_PER_SECOND) / $FULL_FINETUNE_TOKENS_PER_SECOND * 100)")"
LORA_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ACTIVE - $BASELINE_LORA_PEAK_MEMORY_ACTIVE) / $FULL_FINETUNE_PEAK_MEMORY_ACTIVE * 100)")"
LORA_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_ALLOC - $BASELINE_LORA_PEAK_MEMORY_ALLOC) / $FULL_FINETUNE_PEAK_MEMORY_ALLOC * 100)")"
LORA_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE="$(python -c "print(($FULL_FINETUNE_PEAK_MEMORY_RESERVED - $BASELINE_LORA_PEAK_MEMORY_RESERVED) / $FULL_FINETUNE_PEAK_MEMORY_RESERVED * 100)")"
LORA_VS_FULL_PERPLEXITY_DECREASE="$(python -c "print($FULL_PERPLEXITY - $BASELINE_LORA_PERPLEXITY)")"
LORA_VS_FULL_ACC_INCREASE="$(python -c "print(($BASELINE_LORA_ACC - $FULL_ACC) * 100)")"

echo "baseline_lora metrics: $BASELINE_LORA_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $BASELINE_LORA_TOKENS_PER_SECOND"
echo "  peak_memory_active: $BASELINE_LORA_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $BASELINE_LORA_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $BASELINE_LORA_PEAK_MEMORY_RESERVED"
echo "  word_perplexity (float): $BASELINE_LORA_FLOAT_PERPLEXITY"
echo "  word_perplexity: $BASELINE_LORA_PERPLEXITY"
echo "  acc_norm (float): $BASELINE_LORA_FLOAT_ACC"
echo "  acc_norm: $BASELINE_LORA_ACC"
echo ""

echo "qat_lora metrics: $QAT_LORA_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $QAT_LORA_TOKENS_PER_SECOND"
echo "  peak_memory_active: $QAT_LORA_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $QAT_LORA_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $QAT_LORA_PEAK_MEMORY_RESERVED"
echo "  word_perplexity: $QAT_LORA_PERPLEXITY"
echo "  acc_norm: $QAT_LORA_ACC"
echo ""

echo "raw qat metrics: $QAT_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $QAT_TOKENS_PER_SECOND"
echo "  peak_memory_active: $QAT_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $QAT_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $QAT_PEAK_MEMORY_RESERVED"
echo "  word_perplexity: $QAT_PERPLEXITY"
echo "  acc_norm: $QAT_ACC"
echo ""

echo "raw full_finetune metrics: $FULL_FINETUNE_METRICS_FILE"
echo "  tokens_per_second_per_gpu: $FULL_FINETUNE_TOKENS_PER_SECOND"
echo "  peak_memory_active: $FULL_FINETUNE_PEAK_MEMORY_ACTIVE"
echo "  peak_memory_alloc: $FULL_FINETUNE_PEAK_MEMORY_ALLOC"
echo "  peak_memory_reserved: $FULL_FINETUNE_PEAK_MEMORY_RESERVED"
echo "  word_perplexity (float): $FULL_FLOAT_PERPLEXITY"
echo "  word_perplexity: $FULL_PERPLEXITY"
echo "  acc_norm (float): $FULL_FLOAT_ACC"
echo "  acc_norm: $FULL_ACC"
echo ""

echo "qat_lora stats compared to baseline_lora:"
echo "  tokens_per_second_per_gpu % decrease: ${QAT_LORA_VS_LORA_TOKENS_PER_SECOND_DECREASE}"
echo "  peak_memory_active % increase: ${QAT_LORA_VS_LORA_PEAK_MEMORY_ACTIVE_INCREASE}"
echo "  peak_memory_alloc % increase: ${QAT_LORA_VS_LORA_PEAK_MEMORY_ALLOC_INCREASE}"
echo "  peak_memory_reserved % increase: ${QAT_LORA_VS_LORA_PEAK_MEMORY_RESERVED_INCREASE}"
echo "  % perplexity degredation recovered: ${QAT_LORA_VS_LORA_PERPLEXITY_RECOVERED}"
echo "  % acc_norm degredation recovered: ${QAT_LORA_VS_LORA_ACC_RECOVERED}"
echo ""

echo "qat_lora stats compared to raw qat:"
echo "  tokens_per_second_per_gpu % increase: ${QAT_LORA_VS_QAT_TOKENS_PER_SECOND_INCREASE}"
echo "  peak_memory_active % decrease: ${QAT_LORA_VS_QAT_PEAK_MEMORY_ACTIVE_DECREASE}"
echo "  peak_memory_alloc % decrease: ${QAT_LORA_VS_QAT_PEAK_MEMORY_ALLOC_DECREASE}"
echo "  peak_memory_reserved % decrease: ${QAT_LORA_VS_QAT_PEAK_MEMORY_RESERVED_DECREASE}"
echo "  perplexity decrease: ${QAT_LORA_VS_QAT_PERPLEXITY_DECREASE}"
echo "  acc_norm increase: ${QAT_LORA_VS_QAT_ACC_INCREASE} %"
echo ""

echo "qat_lora stats compared to raw full finetune:"
echo "  tokens_per_second_per_gpu % increase: ${QAT_LORA_VS_FULL_TOKENS_PER_SECOND_INCREASE}"
echo "  peak_memory_active % decrease: ${QAT_LORA_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE}"
echo "  peak_memory_alloc % decrease: ${QAT_LORA_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE}"
echo "  peak_memory_reserved % decrease: ${QAT_LORA_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE}"
echo "  % perplexity degredation recovered: ${QAT_LORA_VS_FULL_PERPLEXITY_RECOVERED}"
echo "  % acc_norm degredation recovered: ${QAT_LORA_VS_FULL_ACC_RECOVERED}"
echo ""

echo "baseline_lora stats compared to raw full finetune:"
echo "  tokens_per_second_per_gpu % increase: ${LORA_VS_FULL_TOKENS_PER_SECOND_INCREASE}"
echo "  peak_memory_active % decrease: ${LORA_VS_FULL_PEAK_MEMORY_ACTIVE_DECREASE}"
echo "  peak_memory_alloc % decrease: ${LORA_VS_FULL_PEAK_MEMORY_ALLOC_DECREASE}"
echo "  peak_memory_reserved % decrease: ${LORA_VS_FULL_PEAK_MEMORY_RESERVED_DECREASE}"
echo "  perplexity decrease: ${LORA_VS_FULL_PERPLEXITY_DECREASE}"
echo "  acc_norm increase: ${LORA_VS_FULL_ACC_INCREASE} %"
echo ""
