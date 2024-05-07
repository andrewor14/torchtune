#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 5/7/24
# new eval script, quantize directly during eval
export TASKS="[\"wikitext\", \"truthfulqa_mc2\"]"
#CUDA_VISIBLE_DEVICES=2 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-5-6 ./eval_it.sh qat_llama3_1714919453 &
CUDA_VISIBLE_DEVICES=3 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-4-29 ./eval_it.sh full_llama3_1714318494 &
wait

# 5/6/24
# eval delayed QAT after 1000 steps
#export LOG_DIR=/home/andrewor/local/logs/tune/saved-5-6
#CUDA_VISIBLE_DEVICES=1 ./quantize_it.sh qat_llama3_1714919453
#CUDA_VISIBLE_DEVICES=3 QUANTIZED=true ./eval_it.sh qat_llama3_1714919453 &
#CUDA_VISIBLE_DEVICES=1 ./eval_it.sh qat_llama3_1714919453 &
#wait

# 5/5/24
# eval QAT after loading from finetune checkpoint
#export LOG_DIR=/home/andrewor/local/logs/tune/saved-5-5
#export CHECKPOINT_FILES="[meta_model_2.pt]"
#echo "-------------------------------------- llama3 qat after finetune quantize"
#CUDA_VISIBLE_DEVICES=2 ./quantize_it.sh qat_llama3_1714794223
#echo "-------------------------------------- llama3 qat after finetune full eval (quantized)"
#CUDA_VISIBLE_DEVICES=2 QUANTIZED=true CHECKPOINT_FILES="[meta_model_2-8da4w.pt]" ./eval_it.sh qat_llama3_1714794223 &
#echo "-------------------------------------- llama3 qat after finetune full eval (not quantized)"
#CUDA_VISIBLE_DEVICES=3 ./eval_it.sh qat_llama3_1714794223 &
#wait

# 5/4/24
#echo "-------------------------------------- llama3 qat enable after 1000 steps"
#BATCH_SIZE=2 ENABLE_FAKE_QUANT_STEP=1000 ./run_it.sh qat

# 5/3/24
#echo "-------------------------------------- llama3 qat after finetune quantize"
#CUDA_VISIBLE_DEVICES=2 LOG_DIR=/home/andrewor/local/logs/tune/saved-5-3 ./quantize_it.sh qat_llama3_1714689077
#echo "-------------------------------------- llama3 qat after finetune full eval (quantized)"
#CUDA_VISIBLE_DEVICES=2 LOG_DIR=/home/andrewor/local/logs/tune/saved-5-3 QUANTIZED=true ./eval_it.sh qat_llama3_1714689077
#echo "-------------------------------------- llama3 qat after finetune full eval (not quantized)"
#CUDA_VISIBLE_DEVICES=2 LOG_DIR=/home/andrewor/local/logs/tune/saved-5-3 ./eval_it.sh qat_llama3_1714689077

# Starting QAT from finetuned checkpoint
#BATCH_SIZE=2 CHECKPOINT_DIR="/home/andrewor/local/logs/tune/saved-4-29/full_llama3_1714318494" CHECKPOINT_FILES="[meta_model_2.pt]" ./run_it.sh qat

# 5/2/24
#echo "-------------------------------------- llama3 qat after finetune quantize"
#CUDA_VISIBLE_DEVICES=2 LOG_DIR=/home/andrewor/local/logs/tune CHECKPOINT_FILES="[meta_model_0.pt]" ./quantize_it.sh qat_llama3_1714607603
#echo "-------------------------------------- llama3 qat after finetune full eval (not quantized)"
#CUDA_VISIBLE_DEVICES=2 LOG_DIR=/home/andrewor/local/logs/tune CHECKPOINT_FILES="[meta_model_0.pt]" TASKS="[\"truthfulqa_mc2\",\"wikitext\"]" ./eval_it.sh qat_llama3_1714607603 &
#echo "-------------------------------------- llama3 qat after finetune full eval (quantized)"
#CUDA_VISIBLE_DEVICES=3 LOG_DIR=/home/andrewor/local/logs/tune CHECKPOINT_FILES="[meta_model_0-8da4w.pt]" TASKS="[\"truthfulqa_mc2\",\"wikitext\"]" QUANTIZED="true" ./eval_it.sh qat_llama3_1714607603 &
#wait

# 5/1/24
# Starting QAT from finetuned checkpoint
#NUM_EPOCHS=3 BATCH_SIZE=2 CHECKPOINT_DIR="/home/andrewor/local/logs/tune/saved-4-29/full_llama3_1714318494" CHECKPOINT_FILES="[meta_model_2.pt]" ./run_it.sh qat

# 4/30/24
#echo "-------------------------------------- llama3 full eval (not quantized)"
#CUDA_VISIBLE_DEVICES=2 LOG_DIR=/home/andrewor/local/logs/tune/saved-4-29 ./eval_it.sh full_llama3_1714318494 &
#echo "-------------------------------------- llama3 qat eval (not quantized)"
#CUDA_VISIBLE_DEVICES=3 LOG_DIR=/home/andrewor/local/logs/tune/saved-4-29 ./eval_it.sh qat_llama3_1714169626 &
