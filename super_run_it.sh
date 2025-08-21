LOG_DIR="/home/andrewor/local/logs/tune"

rm -rf "${LOG_DIR}/Llama3"*

#export QUANTIZER="torchtune.training.quantization.NVFP4QATQuantizer"
export QUANTIZER="torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL="Llama3.2-3B" ENABLE_QAT="true" ./run_it.sh &

export CUDA_VISIBLE_DEVICES="4,5,6,7"
MODEL="Llama3.2-3B" ENABLE_QAT="false" ./run_it.sh &
wait
