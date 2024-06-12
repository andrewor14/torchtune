#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 6/12/24 (c4 llama3 2b eval)

cd /home/andrewor/local/ao
git checkout 2b-weight-only
cd /home/andrewor/local/torchtune

export SKIP_FLOAT="true"
export GROUP_SIZE="32"

EXP_DIR="/home/andrewor/local/logs/tune/saved-5-31/full_llama3_1716429012_c4"
CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="2w_gs32_s5000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="2w_gs32_s10000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="2w_gs32_e0" ./eval_it.sh $EXP_DIR &
export SKIP_QUANTIZE_FILTER="skip_first3_last2"
CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s5000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=6 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s10000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=7 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="2w_skip_first3_last2_gs32_e0" ./eval_it.sh $EXP_DIR &
wait

# Other less important settings
CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s1000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s2000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s3000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s4000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=6 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s6000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=7 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s7000" ./eval_it.sh $EXP_DIR &
wait
CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s8000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s9000" ./eval_it.sh $EXP_DIR &
unset SKIP_QUANTIZE_FILTER
CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="2w_gs32_s1000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="2w_gs32_s2000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=6 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="2w_gs32_s3000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="2w_gs32_s4000" ./eval_it.sh $EXP_DIR &
wait
CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="2w_gs32_s6000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="2w_gs32_s7000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="2w_gs32_s8000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="2w_gs32_s9000" ./eval_it.sh $EXP_DIR &



# 6/3/24 (c4 llama3 2e-6 lr)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=3
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#export ENABLE_ACTIVATION_CHECKPOINTING="true"
#export RUN_TAG="c4_lr_2e-6"
#export LEARNING_RATE="2e-6"
#
#echo -e "=== Run full ==="
#
#./run_it.sh full
#
#echo -e "\n\n\n=== Eval full ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "=== Run qat ==="
#
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval qat ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait


# 6/2/24 (3-bit weight only c4 for llama3 2e-6 lr)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=3
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#export ENABLE_ACTIVATION_CHECKPOINTING="true"

# ===================================================================
#  3-bit weight only + skip first 3 last 2 + group size 32 + lr 2e-6
# ===================================================================

#cd /home/andrewor/local/ao
#git checkout 3b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo -e "=== Run full ==="
#
#export RUN_TAG="c4_lr_2e-6"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2"
#export LEARNING_RATE="2e-6"
#export GROUP_SIZE=32
#./run_it.sh full
#
#echo -e "\n\n\n=== Eval full ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="3w_skip_first3_last2_gs32_e0" ./eval_it.sh $EXP_DIR &
#wait


# 5/31/24 (3-bit weight only c4 for llama3)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=3
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#export ENABLE_ACTIVATION_CHECKPOINTING="true"
#
## =======================================
##  3-bit weight only + group size 32
## =======================================
#
#cd /home/andrewor/local/ao
#git checkout 3b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo -e "\n\n\n=== Run QAT delay 1000 3-bit weight only ==="
#
#export ENABLE_FAKE_QUANT_STEP=1000
#export GROUP_SIZE="32"
#export RUN_TAG="3w_c4_delay_1000_gs32"
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval QAT delay 1000 3-bit weight only ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Eval full 3-bit weight only ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-31/full_llama3_${TIMESTAMP}_c4"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="3w_gs32_s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="3w_gs32_s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="3w_gs32_s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="3w_gs32_s4000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="3w_gs32_s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="3w_gs32_s6000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="3w_gs32_s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="3w_gs32_s8000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="3w_gs32_s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="3w_gs32_s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="3w_gs32_e0" ./eval_it.sh $EXP_DIR &
#wait
#
## =========================================================
##  3-bit weight only + group size 32 + skip first 3 last 2
## =========================================================
#
#echo -e "\n\n\n=== Run QAT delay 1000 3-bit weight only skip first 3 last 2 ==="
#
#export ENABLE_FAKE_QUANT_STEP=1000
#export GROUP_SIZE="32"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2"
#export RUN_TAG="3w_c4_delay_1000_skip_first3_last2_gs32"
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval QAT delay 1000 3-bit weight only skip first 3 last 2 ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Eval full 3-bit weight only skip first 3 last 2 ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-31/full_llama3_${TIMESTAMP}_c4"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s4000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s6000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s8000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="3w_skip_first3_last2_gs32_s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="3w_skip_first3_last2_gs32_e0" ./eval_it.sh $EXP_DIR &
#wait
#
## =============================================
##  2-bit weight only + group size 32 full eval
## =============================================
#
#cd /home/andrewor/local/ao
#git checkout 2b-weight-only
#cd /home/andrewor/local/torchtune
#
#unset SKIP_QUANTIZE_FILTER
#
#echo -e "\n\n\n=== Eval full 2-bit weight only ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-31/full_llama3_${TIMESTAMP}_c4"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="2w_gs32_s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="2w_gs32_s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="2w_gs32_s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="2w_gs32_s4000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="2w_gs32_s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="2w_gs32_s6000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="2w_gs32_s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="2w_gs32_s8000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="2w_gs32_s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="2w_gs32_s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="2w_gs32_e0" ./eval_it.sh $EXP_DIR &
#wait
#
## ===================================================================
##  2-bit weight only + group size 32 + skip first 3 last 2 full eval
## ===================================================================
#
#export SKIP_QUANTIZE_FILTER="skip_first3_last2"
#
#echo -e "\n\n\n=== Eval full 2-bit weight only skip first 3 last 2 ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-31/full_llama3_${TIMESTAMP}_c4"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s4000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s6000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s8000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="2w_skip_first3_last2_gs32_s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="2w_skip_first3_last2_gs32_e0" ./eval_it.sh $EXP_DIR &
#wait


# 5/31/24 (c4 for llama3 full)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=3
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#export ENABLE_ACTIVATION_CHECKPOINTING="true"
#
#echo -e "=== Run full ==="
#
#unset ENABLE_FAKE_QUANT_STEP
#export RUN_TAG="c4"
#./run_it.sh full
#
#echo -e "\n\n\n=== Eval full ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait

# 5/30/24 (c4 for llama3)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=3
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#export ENABLE_ACTIVATION_CHECKPOINTING="true"
#
#echo -e "\n\n\n=== Run QAT delay 1000 ==="
#
#export ENABLE_FAKE_QUANT_STEP=1000
#export RUN_TAG="c4_delay_1000"
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval QAT delay 1000 ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=6,7 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait

# 5/22/24 (c4 for real)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=2
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export RUN_TAG="c4"
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#
#echo -e "=== Run full ==="
#
#./run_it.sh full
#
#echo -e "\n\n\n=== Eval full ==="
#
#export SKIP_FLOAT="true"
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_10999.pt, hf_model_0002_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_11999.pt, hf_model_0002_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_12999.pt, hf_model_0002_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_13999.pt, hf_model_0002_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_14999.pt, hf_model_0002_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_15999.pt, hf_model_0002_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_16999.pt, hf_model_0002_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_17999.pt, hf_model_0002_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_18999.pt, hf_model_0002_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_19999.pt, hf_model_0002_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#unset SKIP_FLOAT
#unset SKIP_QUANTIZE
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait

#echo -e "\n\n\n=== Run QAT ==="
#
#BATCH_SIZE=1 ./run_it.sh qat

#echo -e "\n\n\n=== Eval QAT ==="
#
#export SKIP_FLOAT="true"
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_10999.pt, hf_model_0002_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_11999.pt, hf_model_0002_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_12999.pt, hf_model_0002_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_13999.pt, hf_model_0002_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_14999.pt, hf_model_0002_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_15999.pt, hf_model_0002_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_16999.pt, hf_model_0002_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_17999.pt, hf_model_0002_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_18999.pt, hf_model_0002_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_19999.pt, hf_model_0002_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#unset SKIP_FLOAT
#unset SKIP_QUANTIZE
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait

# 5/21/24 (wikitext-2-v1)

#export TIMESTAMP=`date +%s`
#export LLAMA_VERSION=3
#export ENABLE_FAKE_QUANT_STEP=1000
#export BATCH_SIZE=4
#export NUM_EPOCHS=3
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
#echo -e "\n\n\n=== Running full ==="
#RUN_TAG="8da4w" ./run_it.sh full
#echo -e "\n\n\n=== Running QAT ==="
#RUN_TAG="8da4w" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on vanilla 8da4w ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on vanilla 8da4w (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama3_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
## Try skip_vproj
#
#export SKIP_QUANTIZE_FILTER=skip_vproj
#
#echo -e "\n\n\n=== Running QAT skip vproj ==="
#RUN_TAG="skip_vproj" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on skip_vproj ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_skip_vproj"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on skip_vproj (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama3_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="skip_vproj_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="skip_vproj_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="skip_vproj_e2" ./eval_it.sh $EXP_DIR &
#wait
#
## Try skip_first3_last2_vproj
#
#export SKIP_QUANTIZE_FILTER=skip_first3_last2_vproj
#
#echo -e "\n\n\n=== Running QAT skip_first3_last2_vproj ==="
#RUN_TAG="skip_first3_last2_vproj" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on skip_first3_last2_vproj ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_skip_first3_last2_vproj"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on skip_first3_last2_vproj (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama3_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="skip_first3_last2_vproj_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="skip_first3_last2_vproj_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="skip_first3_last2_vproj_e2" ./eval_it.sh $EXP_DIR &
#wait
#
## Try gs32
#
#unset SKIP_QUANTIZE_FILTER
#export GROUP_SIZE=32
#
#echo -e "\n\n\n=== Running QAT gs32 ==="
#RUN_TAG="gs32" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on gs32 ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_gs32"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on gs32 (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama3_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="gs32_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="gs32_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="gs32_e2" ./eval_it.sh $EXP_DIR &
#wait


# 5/20/24 (overnight!)

#export TIMESTAMP="1716180963"
#export LLAMA_VERSION=3
#export ENABLE_FAKE_QUANT_STEP=1000
#export BATCH_SIZE=4
#echo "Running QAT on delay_1000_skip_first3_last2_vproj"
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_TAG=delay_1000_skip_first3_last2_vproj SKIP_QUANTIZE_FILTER=skip_first3_last2_vproj ./run_it.sh qat
#echo "Running QAT on delay_1000_skip_first3_last2_vproj_output"
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_TAG=delay_1000_skip_first3_last2_vproj_output SKIP_QUANTIZE_FILTER=skip_first3_last2_vproj_output ./run_it.sh qat
#echo "Running QAT on delay_1000_skip_first3_last2_vproj_gs32"
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_TAG=delay_1000_skip_first3_last2_vproj_gs32 SKIP_QUANTIZE_FILTER=skip_first3_last2_vproj GROUP_SIZE=32 ./run_it.sh qat
#
#echo "Running eval on delay_1000_skip_first3_last2_vproj"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj"
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_delay_1000_skip_first3_last2_vproj"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo "Running eval on delay_1000_skip_first3_last2_vproj (baseline)"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj"
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama3_1715537738_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="skip_first3_last2_vproj_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="skip_first3_last2_vproj_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="skip_first3_last2_vproj_e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo "Running eval on delay_1000_skip_first3_last2_vproj_output"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj_output"
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_delay_1000_skip_first3_last2_vproj_output"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo "Running eval on delay_1000_skip_first3_last2_vproj_output (baseline)"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj_output"
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama3_1715537738_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="skip_first3_last2_vproj_output_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="skip_first3_last2_vproj_output_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="skip_first3_last2_vproj_output_e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo "Running eval on delay_1000_skip_first3_last2_vproj_gs32"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj"
#export GROUP_SIZE=32
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_delay_1000_skip_first3_last2_vproj_gs32"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo "Running eval on delay_1000_skip_first3_last2_vproj_gs32 (baseline)"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj"
#export GROUP_SIZE=32
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama3_1715537738_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="skip_first3_last2_vproj_gs32_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_1.pt]" RUN_TAG="skip_first3_last2_vproj_gs32_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_2.pt]" RUN_TAG="skip_first3_last2_vproj_gs32_e2" ./eval_it.sh $EXP_DIR &
#wait


# ============== #
#  on devgpu023  #
# ============== #

#cd /home/andrewor/local/ao
#git checkout 2b-weight-only
#cd /home/andrewor/local/torchtune
#
#export TIMESTAMP="1716181959"
#export LLAMA_VERSION=2
#export ENABLE_FAKE_QUANT_STEP=1000
#export BATCH_SIZE=4
##echo "Running QAT on 2-bit weight only"
##CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_TAG=delay_1000_2w ./run_it.sh qat
##echo "Running QAT on 2-bit weight only (group size 32)"
##CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_TAG=delay_1000_2w_gs32 GROUP_SIZE=32 ./run_it.sh qat
#
#echo "Running eval on 2-bit weight only"
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_delay_1000_2w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait
#
#echo "Running eval on 2-bit weight only (baseline)"
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama2_1715537915_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="delay_1000_2w_e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="delay_1000_2w_e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="delay_1000_2w_e2" ./eval_it.sh "$EXP_DIR" &
#wait
#
#echo "Running eval on 2-bit weight only (group size 32)"
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_delay_1000_2w_gs32"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" GROUP_SIZE="32" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" GROUP_SIZE="32" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" GROUP_SIZE="32" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait
#
#echo "Running eval on 2-bit weight only (group size 32) (baseline)"
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama2_1715537915_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" GROUP_SIZE="32" RUN_TAG="delay_1000_2w_gs32_e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" GROUP_SIZE="32" RUN_TAG="delay_1000_2w_gs32_e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" GROUP_SIZE="32" RUN_TAG="delay_1000_2w_gs32_e2" ./eval_it.sh "$EXP_DIR" &
#wait
#
#cd /home/andrewor/local/ao
#git checkout 3b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo "Running QAT on 3-bit weight only (group size 32)"
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_TAG=delay_1000_3w_gs32 GROUP_SIZE=32 ./run_it.sh qat
#
#echo "Running eval on 3-bit weight only (group size 32)"
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_delay_1000_3w_gs32"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" GROUP_SIZE="32" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" GROUP_SIZE="32" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" GROUP_SIZE="32" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait
#
#echo "Running eval on 3-bit weight only (group size 32) (baseline)"
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-12/full_llama2_1715537915_new_baseline"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" GROUP_SIZE="32" RUN_TAG="delay_1000_3w_gs32_e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" GROUP_SIZE="32" RUN_TAG="delay_1000_3w_gs32_e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" GROUP_SIZE="32" RUN_TAG="delay_1000_3w_gs32_e2" ./eval_it.sh "$EXP_DIR" &
#wait


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
