export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL=Llama3-8B QUANTIZED_TRAINING_TYPE=int8_mixed_precision ./run_it.sh
mv /home/andrewor/local/logs/tune/int8* /home/andrewor/local/logs/tune/saved-2-21-llama3

MODEL=Llama3.1-8B QUANTIZED_TRAINING_TYPE=int8_mixed_precision ./run_it.sh
mv /home/andrewor/local/logs/tune/int8* /home/andrewor/local/logs/tune/saved-2-21-llama3.1
