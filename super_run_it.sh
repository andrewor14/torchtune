LOG_DIR="/home/andrewor/local/logs/tune"

rm -rf "${LOG_DIR}/fp8_quantized_training"*
rm -rf "${LOG_DIR}/full_finetune_distributed_baseline"*

export SKIP_EVAL="true"
export MAX_STEPS_PER_EPOCH="100"

# No TP
FP8_RECIPE_NAME="tensorwise" ./run_it.sh
mv "${LOG_DIR}/fp8_quantized_training" "${LOG_DIR}/fp8_quantized_training_tensorwise"
FP8_RECIPE_NAME="rowwise" ./run_it.sh
mv "${LOG_DIR}/fp8_quantized_training" "${LOG_DIR}/fp8_quantized_training_rowwise"
FP8_RECIPE_NAME="rowwise_with_gw_hp" ./run_it.sh
mv "${LOG_DIR}/fp8_quantized_training" "${LOG_DIR}/fp8_quantized_training_rowwise_with_gw_hp"
./run_it.sh
mv "${LOG_DIR}/fp8_quantized_training" "${LOG_DIR}/fp8_quantized_training_noname"

# With TP
#export TP_PLAN="{_component_: 'torchtune.models.llama3.base_llama_tp_plan'}"
#export TP_DIM="2"
#FP8_RECIPE_NAME="tensorwise" ./run_it.sh
#mv "${LOG_DIR}/fp8_quantized_training" "${LOG_DIR}/fp8_quantized_training_tensorwise_tp"
#./run_it.sh
#mv "${LOG_DIR}/fp8_quantized_training" "${LOG_DIR}/fp8_quantized_training_noname_tp"

# Baseline
#ENABLE_FP8="false" TP_PLAN="{_component_: 'torchtune.models.llama3.base_llama_tp_plan'}" ./run_it.sh
#mv "${LOG_DIR}/full_finetune_distributed_baseline" "${LOG_DIR}/full_finetune_distributed_baseline_tp"
ENABLE_FP8="false" TP_PLAN="null" ./run_it.sh
