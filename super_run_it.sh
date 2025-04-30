# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

LOG_DIR="/home/andrewor/local/logs/tune"

rm -rf "${LOG_DIR}/Llama3"*
rm -rf "${LOG_DIR}/Qwen3"*

MODEL="Llama3-8B" ENABLE_QAT="false" ./run_it.sh
MODEL="Llama3-8B" ENABLE_QAT="true" ENABLE_LORA="false" ./run_it.sh
MODEL="Llama3-8B" ENABLE_QAT="true" ENABLE_LORA="true" ./run_it.sh

MODEL="Llama3.1-8B" ENABLE_QAT="false" ./run_it.sh
MODEL="Llama3.1-8B" ENABLE_QAT="true" ENABLE_LORA="false" ./run_it.sh
MODEL="Llama3.1-8B" ENABLE_QAT="true" ENABLE_LORA="true" ./run_it.sh

MODEL="Llama3.2-3B" ENABLE_QAT="false" ./run_it.sh
MODEL="Llama3.2-3B" ENABLE_QAT="true" ENABLE_LORA="false" ./run_it.sh
MODEL="Llama3.2-3B" ENABLE_QAT="true" ENABLE_LORA="true" ./run_it.sh

MODEL="Qwen3-1.7B" ENABLE_QAT="false" ./run_it.sh
MODEL="Qwen3-1.7B" ENABLE_QAT="true" ENABLE_LORA="false" ./run_it.sh
MODEL="Qwen3-1.7B" ENABLE_QAT="true" ENABLE_LORA="true" ./run_it.sh
