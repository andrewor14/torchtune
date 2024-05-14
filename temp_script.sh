#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 5/13/24

CUDA_VISIBLE_DEVICES=6,7 ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-13/qat_llama3_1715638190_delay_1000_disable_vproj_fq

# 5/12/24

# New baseline
#export LOG_DIR=/home/andrewor/local/logs/tune/saved-5-12
#export RUN_TAG="8da4w"
#CUDA_VISIBLE_DEVICES=0 ./eval_it.sh qat_llama2_1715557828_new_delay_1000 &
#CUDA_VISIBLE_DEVICES=1 SHOULD_QUANTIZE=true ./eval_it.sh qat_llama2_1715557828_new_delay_1000 &
#CUDA_VISIBLE_DEVICES=4 ./eval_it.sh full_llama3_1715537738_new_baseline &
#CUDA_VISIBLE_DEVICES=5 SHOULD_QUANTIZE=true ./eval_it.sh full_llama3_1715537738_new_baseline &
#CUDA_VISIBLE_DEVICES=6 ./eval_it.sh full_llama2_1715537915_new_baseline &
#CUDA_VISIBLE_DEVICES=7 SHOULD_QUANTIZE=true ./eval_it.sh full_llama2_1715537915_new_baseline &
#wait

# eval 8da8w for both llama2 and llama3
# Note: use this commit 0a52fafec31b9c4f4403e109913ad9dadb5187a6
# Make sure you have the 8da8w changes in torchao!
#export LOG_DIR=/home/andrewor/local/logs/tune/saved-5-11
#CUDA_VISIBLE_DEVICES=0 ./eval_it.sh qat_llama3_1715456628_8da8w &
#CUDA_VISIBLE_DEVICES=1 SHOULD_QUANTIZE=true ./eval_it.sh qat_llama3_1715456628_8da8w &
#CUDA_VISIBLE_DEVICES=2 ./eval_it.sh qat_llama2_1715465847_8da8w &
#CUDA_VISIBLE_DEVICES=3 SHOULD_QUANTIZE=true ./eval_it.sh qat_llama2_1715465847_8da8w &
## Baseline 8da8w
#export LOG_DIR=/home/andrewor/local/logs/tune/saved-4-29
#export RUN_TAG="8da8w"
#CUDA_VISIBLE_DEVICES=4 ./eval_it.sh full_llama3_1714318494 &
#CUDA_VISIBLE_DEVICES=5 SHOULD_QUANTIZE=true ./eval_it.sh full_llama3_1714318494 &
#wait

# Run these on devgpu023.odn1
# eval FQ with matching numerics for both llama2 and llama3 (8da4w)
#export LOG_DIR=/home/andrewor/local/logs/tune/saved-5-11
#CUDA_VISIBLE_DEVICES=0 ./eval_it.sh qat_llama3_1715387019 &
#CUDA_VISIBLE_DEVICES=1 SHOULD_QUANTIZE=true ./eval_it.sh qat_llama3_1715387019 &
#CUDA_VISIBLE_DEVICES=2 ./eval_it.sh qat_llama2_1715387047 &
#CUDA_VISIBLE_DEVICES=3 SHOULD_QUANTIZE=true ./eval_it.sh qat_llama2_1715387047 &
#wait
# eval 8da8w llama2 baseline
#export LOG_DIR=/home/andrewor/local/logs/tune/saved-5-7
#export RUN_TAG="8da8w"
#CUDA_VISIBLE_DEVICES=4 ./eval_it.sh full_llama2_1715035106 &
#CUDA_VISIBLE_DEVICES=5 SHOULD_QUANTIZE=true ./eval_it.sh full_llama2_1715035106 &
#wait

# 5/9/24
# eval qat llama3 with fq matching numerics with core ops
#CUDA_VISIBLE_DEVICES=2 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-5-9 ./eval_it.sh qat_llama3_1715195393 &
#CUDA_VISIBLE_DEVICES=3 LOG_DIR=/home/andrewor/local/logs/tune/saved-5-9 ./eval_it.sh qat_llama3_1715195393 &
#wait
# baseline full llama3
#CUDA_VISIBLE_DEVICES=2 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-4-29 ./eval_it.sh full_llama3_1714318494 &
#CUDA_VISIBLE_DEVICES=3 LOG_DIR=/home/andrewor/local/logs/tune/saved-4-29 ./eval_it.sh full_llama3_1714318494 &
#wait

# Run these on devgpu023.odn1
# eval llama3 with core fq ops (float zp)
#CUDA_VISIBLE_DEVICES=2 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-5-8 ./eval_it.sh qat_llama3_1715137070 &
#CUDA_VISIBLE_DEVICES=3 LOG_DIR=/home/andrewor/local/logs/tune/saved-5-8 ./eval_it.sh qat_llama3_1715137070 &
#wait

# 5/8/24
# eval llama2
#export TASKS="[\"anli_r1\", \"anli_r2\", \"anli_r3\", \"arc_challenge\", \"arc_easy\", \"truthfulqa_mc1\", \"truthfulqa_mc2\", \"winogrande\", \"openbookqa\", \"piqa\"]"
#CUDA_VISIBLE_DEVICES=2 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-5-7 ./eval_it.sh qat_llama2_1715029335 &
#CUDA_VISIBLE_DEVICES=3 LOG_DIR=/home/andrewor/local/logs/tune/saved-5-7 ./eval_it.sh qat_llama2_1715029335 &
#wait
#
# Run these on devgpu023.odn1
#CUDA_VISIBLE_DEVICES=2 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-5-7 ./eval_it.sh full_llama2_1715035106 &
#CUDA_VISIBLE_DEVICES=2 LOG_DIR=/home/andrewor/local/logs/tune/saved-5-7 ./eval_it.sh full_llama2_1715035106 &
#wait

# 5/7/24
# new eval script, quantize directly during eval
#export TASKS="[\"wikitext\", \"truthfulqa_mc2\"]"
#CUDA_VISIBLE_DEVICES=2 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-5-6 ./eval_it.sh qat_llama3_1714919453 &
#CUDA_VISIBLE_DEVICES=3 SHOULD_QUANTIZE=true LOG_DIR=/home/andrewor/local/logs/tune/saved-4-29 ./eval_it.sh full_llama3_1714318494 &
#wait

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
