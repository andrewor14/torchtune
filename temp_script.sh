#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 5/19/24

#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama3_1715537738_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait

#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-19/qat_llama3_1716156677_delay_1000_skip_first3_last2_mlp_only"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait

# 5/17/24

#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-17/qat_llama2_1715978336_2w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait

#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama2_1715537915_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="2w_e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="2w_e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="2w_e2" ./eval_it.sh "$EXP_DIR" &
#wait

# on devgpu023
#export SKIP_QUANTIZE="true"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh /home/andrewor/local/logs/tune/full_llama3_1715971006_wikitext_103 &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh /home/andrewor/local/logs/tune/full_llama3_1715971006_wikitext_103 &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh /home/andrewor/local/logs/tune/full_llama3_1715971006_wikitext_103 &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[meta_model_3.pt]" RUN_TAG="e3" ./eval_it.sh /home/andrewor/local/logs/tune/full_llama3_1715971006_wikitext_103 &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[meta_model_4.pt]" RUN_TAG="e4" ./eval_it.sh /home/andrewor/local/logs/tune/full_llama3_1715971006_wikitext_103 &
#wait

#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-17/qat_llama2_1715978854_3w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait

#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama2_1715537915_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="3w_e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="3w_e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="3w_e2" ./eval_it.sh "$EXP_DIR" &
#wait

# 5/16/24
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_0.pt]" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-16/full_llama3_1715785328_wikitext_raw_103

#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-16/full_llama3_1715885149_wikitext_2"
#export SKIP_QUANTIZE="true"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[meta_model_3.pt]" RUN_TAG="e3" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[meta_model_4.pt]" RUN_TAG="e4" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[meta_model_5.pt]" RUN_TAG="e5" ./eval_it.sh "$EXP_DIR" &
#wait
#CUDA_VISIBLE_DEVICES=6,7 ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-16/full_llama3_1715785328_wikitext_raw_103

# on devgpu023
#export SKIP_QUANTIZE="true"
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama2_1715537915_new_baseline"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-16/full_llama2_1715888083_continue_baseline"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e3" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e4" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e5" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_3.pt, hf_model_0002_3.pt]" RUN_TAG="e6" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_4.pt, hf_model_0002_4.pt]" RUN_TAG="e7" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_5.pt, hf_model_0002_5.pt]" RUN_TAG="e8" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=6 CHECKPOINT_FILES="[hf_model_0001_6.pt, hf_model_0002_6.pt]" RUN_TAG="e9" ./eval_it.sh "$EXP_DIR" &
#wait
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin]" RUN_TAG="pretrained" ./eval_it.sh "/home/andrewor/local/checkpoints/Llama-2-7b-hf"

# 5/14/24
#CUDA_VISIBLE_DEVICES=4,5 ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/qat_llama3_1715659092_delay_1000_disable_first3_last2

#export SKIP_QUANTIZE="true"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[meta_model_3.pt]" RUN_TAG="e3" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[meta_model_4.pt]" RUN_TAG="e4" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[meta_model_5.pt]" RUN_TAG="e5" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=6 CHECKPOINT_FILES="[meta_model_6.pt]" RUN_TAG="e6" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=7 CHECKPOINT_FILES="[meta_model_7.pt]" RUN_TAG="e7" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &

#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[meta_model_8.pt]" RUN_TAG="e8" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[meta_model_9.pt]" RUN_TAG="e9" ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-14/full_llama3_1715711293_to_convergence &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[consolidated.00.pth]" RUN_TAG="pretrained" ./eval_it.sh /home/andrewor/local/checkpoints/Meta-Llama-3-8B-Instruct/original &
#wait

# 5/13/24
#CUDA_VISIBLE_DEVICES=6,7 ./eval_it.sh /home/andrewor/local/logs/tune/saved-5-13/qat_llama3_1715638190_delay_1000_disable_vproj_fq

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
