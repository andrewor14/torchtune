LOG_DIR="/home/andrewor/local/logs/tune"

rm -rf "${LOG_DIR}/Llama3"*

MODEL="Llama3-8B" ENABLE_QAT="false" ./run_it.sh
MODEL="Llama3-8B" ENABLE_QAT="true" ./run_it.sh
MODEL="Llama3.1-8B" ENABLE_QAT="false" ./run_it.sh
MODEL="Llama3.1-8B" ENABLE_QAT="true" ./run_it.sh
MODEL="Llama3.2-3B" ENABLE_QAT="false" ./run_it.sh
MODEL="Llama3.2-3B" ENABLE_QAT="true" ./run_it.sh
