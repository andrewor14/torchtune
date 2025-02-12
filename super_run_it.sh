export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="Llama3-8B" QUANTIZED_TRAINING_TYPE="int8" ./run_it.sh
mv /home/andrewor/local/logs/tune/int8_quantized_training /home/andrewor/local/logs/tune/saved-2-5-with-compile
MODEL="Llama3.1-8B" QUANTIZED_TRAINING_TYPE="int8" ./run_it.sh
mv /home/andrewor/local/logs/tune/int8_quantized_training /home/andrewor/local/logs/tune/saved-2-6-llama3.1
