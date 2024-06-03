# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ============== #
#  on devgpu023  #
# ============== #

# 6/3/24 (3-bit weight only c4 for llama3 2e-6 lr, don't skip)

export TIMESTAMP="1716429012"
export LLAMA_VERSION=3
export BATCH_SIZE=2
export NUM_EPOCHS=1
export MAX_STEPS_PER_EPOCH=10000
export CHECKPOINT_EVERY_N_STEPS=1000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
export ENABLE_ACTIVATION_CHECKPOINTING="true"

# ===================================================================
#  3-bit weight only + skip first 3 last 2 + group size 32 + lr 2e-6
# ===================================================================

cd /home/andrewor/local/ao
git checkout 3b-weight-only
cd /home/andrewor/local/torchtune

echo -e "=== Run QAT ==="

export RUN_TAG="3w_c4_delay_1000_lr_2e-6_gs32"
export ENABLE_FAKE_QUANT_STEP=1000
export LEARNING_RATE="2e-6"
export GROUP_SIZE=32
./run_it.sh qat

echo -e "\n\n\n=== Eval QAT ==="

EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_${RUN_TAG}"
CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
wait
CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
wait
CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
wait
CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
wait


# 6/2/24 (3-bit weight only c4 for llama3 2e-6 lr)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=3
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#export ENABLE_ACTIVATION_CHECKPOINTING="true"
#
## ===================================================================
##  3-bit weight only + skip first 3 last 2 + group size 32 + lr 2e-6
## ===================================================================
#
#cd /home/andrewor/local/ao
#git checkout 3b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo -e "=== Run QAT ==="
#
#export RUN_TAG="3w_c4_delay_1000_skip_first3_last2_lr_2e-6_gs32"
#export ENABLE_FAKE_QUANT_STEP=1000
#export SKIP_QUANTIZE_FILTER="skip_first3_last2"
#export LEARNING_RATE="2e-6"
#export GROUP_SIZE=32
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval QAT ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama3_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[meta_model_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[meta_model_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[meta_model_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait


# 5/31/24 (2-bit weight only c4 for llama3)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=3
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export EXTRA_ARGS="dataset._component_=torchtune.datasets.text_completion_dataset dataset.source=allenai/c4 dataset.column=text dataset.name=en dataset.split=train"
#export ENABLE_ACTIVATION_CHECKPOINTING="true"
#
## =======================================
##  2-bit weight only + group size 32
## =======================================
#
#cd /home/andrewor/local/ao
#git checkout 2b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo -e "\n\n\n=== Run QAT delay 1000 2-bit weight only ==="
#
#export ENABLE_FAKE_QUANT_STEP=1000
#export GROUP_SIZE="32"
#export RUN_TAG="2w_c4_delay_1000_gs32"
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval QAT delay 1000 2-bit weight only ==="
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
## =========================================================
##  2-bit weight only + group size 32 + skip first 3 last 2
## =========================================================
#
#echo -e "\n\n\n=== Run QAT delay 1000 2-bit weight only skip first 3 last 2 ==="
#
#export ENABLE_FAKE_QUANT_STEP=1000
#export GROUP_SIZE="32"
#export SKIP_QUANTIZE_FILTER="skip_first3_last2"
#export RUN_TAG="2w_c4_delay_1000_skip_first3_last2_gs32"
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval QAT delay 1000 2-bit weight only skip first 3 last 2 ==="
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


# 5/30/24 (C4 llama2)

#export TIMESTAMP="1716429012"
#export LLAMA_VERSION=2
#export BATCH_SIZE=2
#export NUM_EPOCHS=1
#export MAX_STEPS_PER_EPOCH=10000
#export CHECKPOINT_EVERY_N_STEPS=1000
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
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
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_10999.pt, hf_model_0002_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_11999.pt, hf_model_0002_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_12999.pt, hf_model_0002_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_13999.pt, hf_model_0002_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_14999.pt, hf_model_0002_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_15999.pt, hf_model_0002_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_16999.pt, hf_model_0002_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_17999.pt, hf_model_0002_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_18999.pt, hf_model_0002_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_19999.pt, hf_model_0002_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "=== Run full ==="
#
#unset ENABLE_FAKE_QUANT_STEP
#export RUN_TAG="c4"
#./run_it.sh full
#
#echo -e "\n\n\n=== Eval full ==="
#
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_10999.pt, hf_model_0002_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_11999.pt, hf_model_0002_11999.pt]" RUN_TAG="s2000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_12999.pt, hf_model_0002_12999.pt]" RUN_TAG="s3000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_13999.pt, hf_model_0002_13999.pt]" RUN_TAG="s4000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_14999.pt, hf_model_0002_14999.pt]" RUN_TAG="s5000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_15999.pt, hf_model_0002_15999.pt]" RUN_TAG="s6000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_16999.pt, hf_model_0002_16999.pt]" RUN_TAG="s7000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_17999.pt, hf_model_0002_17999.pt]" RUN_TAG="s8000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_18999.pt, hf_model_0002_18999.pt]" RUN_TAG="s9000" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_19999.pt, hf_model_0002_19999.pt]" RUN_TAG="s10000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait


# 5/29/24 (wikitext2 small LR varying group size)

#export TIMESTAMP=`date +%s`
#export LLAMA_VERSION=2
#export BATCH_SIZE=4
#export NUM_EPOCHS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export LEARNING_RATE="5e-7"
#
#export GROUP_SIZE=32
#export RUN_TAG="lr_5e-7_gs${GROUP_SIZE}"
#echo -e "=== Run qat gs${GROUP_SIZE} ==="
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval qat ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#wait
#
#export GROUP_SIZE=64
#export RUN_TAG="lr_5e-7_gs${GROUP_SIZE}"
#echo -e "=== Run qat gs${GROUP_SIZE} ==="
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval qat ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#wait
#
#export GROUP_SIZE=128
#export RUN_TAG="lr_5e-7_gs${GROUP_SIZE}"
#echo -e "=== Run qat gs${GROUP_SIZE} ==="
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval qat ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#wait
#
#export GROUP_SIZE=256
#export RUN_TAG="lr_5e-7_gs${GROUP_SIZE}"
#echo -e "=== Run qat gs${GROUP_SIZE} ==="
#./run_it.sh qat
#
#echo -e "\n\n\n=== Eval qat ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#wait

#echo -e "\n\n\n=== Eval full ==="
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-22/full_llama2_1716427307_lr_5e-7"
#CUDA_VISIBLE_DEVICES=0,1 GROUP_SIZE=32 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0_gs32" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 GROUP_SIZE=64 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0_gs64" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 GROUP_SIZE=128 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0_gs128" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=6,7 GROUP_SIZE=256 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0_gs256" ./eval_it.sh "$EXP_DIR" &
#wait

# 5/23/24

#export SKIP_FLOAT="true"
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-22/full_llama2_1716427307_lr_5e-7"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#
#EXP_DIR="/home/andrewor/local/logs/tune/saved-5-22/qat_llama2_1716427307_lr_5e-7"
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait


# 5/22/24 (wikitext-2 smaller LR)

#export TIMESTAMP=`date +%s`
#export LLAMA_VERSION=2
#export BATCH_SIZE=4
#export NUM_EPOCHS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export LEARNING_RATE="5e-7"
#export RUN_TAG="lr_5e-7_no_mask"
#
#./run_it.sh qat

#echo -e "\n\n\n=== Eval full ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait

#echo -e "\n\n\n=== Eval qat ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh "$EXP_DIR" &
#wait

# Just delay, keep the mask

#cd /home/andrewor/local/ao
#git stash
#cd /home/andrewor/local/torchtune
#
#export RUN_TAG="delay_500_lr_5e-7"
#ENABLE_FAKE_QUANT_STEP=500 ./run_it.sh qat
#
#echo -e "\n\n\n=== Eval qat ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_${RUN_TAG}"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh "$EXP_DIR" &
#wait

# 5/21/24 (llama3 full skip eval)

#export CHECKPOINT_FILES="[consolidated.00.pth]"
#EXP_DIR="/home/andrewor/local/checkpoints/Meta-Llama-3-8B-Instruct/original"
#CUDA_VISIBLE_DEVICES=0,1 RUN_TAG="2w_no_skip" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2 SKIP_FLOAT="true" RUN_TAG="2w_skip_vproj" SKIP_QUANTIZE_FILTER="skip_vproj" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=3 SKIP_FLOAT="true" RUN_TAG="2w_skip_first3_last2" SKIP_QUANTIZE_FILTER="skip_first3_last2" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4 SKIP_FLOAT="true" RUN_TAG="2w_skip_first3_last2_vproj" SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=5 SKIP_FLOAT="true" RUN_TAG="2w_skip_first3_last2_vproj_output" SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj_output" ./eval_it.sh "$EXP_DIR" &
#wait
#
#cd /home/andrewor/local/ao
#git checkout 3b-weight-only
#cd /home/andrewor/local/torchtune
#
#CUDA_VISIBLE_DEVICES=0,1 RUN_TAG="3w_no_skip" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=2 SKIP_FLOAT="true" RUN_TAG="3w_skip_vproj" SKIP_QUANTIZE_FILTER="skip_vproj" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=3 SKIP_FLOAT="true" RUN_TAG="3w_skip_first3_last2" SKIP_QUANTIZE_FILTER="skip_first3_last2" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=4 SKIP_FLOAT="true" RUN_TAG="3w_skip_first3_last2_vproj" SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj" ./eval_it.sh "$EXP_DIR" &
#CUDA_VISIBLE_DEVICES=5 SKIP_FLOAT="true" RUN_TAG="3w_skip_first3_last2_vproj_output" SKIP_QUANTIZE_FILTER="skip_first3_last2_vproj_output" ./eval_it.sh "$EXP_DIR" &
#wait

# 5/20/24 (overnight)

#export TIMESTAMP="1716310674"
#export LLAMA_VERSION=2
#export BATCH_SIZE=4
#export NUM_EPOCHS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export SKIP_QUANTIZE="true"
#export LEARNING_RATE="5e-7"
#
#RUN_TAG="checkpoint_every_100_steps_lr_5e-7" CHECKPOINT_EVERY_N_STEPS="100" ./run_it.sh full
#
#echo -e "=== Eval ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_checkpoint_every_100_steps_lr_5e-7"
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_10099.pt, hf_model_0002_10099.pt]" RUN_TAG="s100" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_10199.pt, hf_model_0002_10199.pt]" RUN_TAG="s200" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_10299.pt, hf_model_0002_10299.pt]" RUN_TAG="s300" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_10399.pt, hf_model_0002_10399.pt]" RUN_TAG="s400" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_10499.pt, hf_model_0002_10499.pt]" RUN_TAG="s500" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_10599.pt, hf_model_0002_10599.pt]" RUN_TAG="s600" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_10699.pt, hf_model_0002_10699.pt]" RUN_TAG="s700" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_10799.pt, hf_model_0002_10799.pt]" RUN_TAG="s800" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_10899.pt, hf_model_0002_10899.pt]" RUN_TAG="s900" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_10999.pt, hf_model_0002_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_11099.pt, hf_model_0002_11099.pt]" RUN_TAG="s1100" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_11199.pt, hf_model_0002_11199.pt]" RUN_TAG="s1200" ./eval_it.sh $EXP_DIR &
#wait
#CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_11299.pt, hf_model_0002_11299.pt]" RUN_TAG="s1300" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_11399.pt, hf_model_0002_11399.pt]" RUN_TAG="s1400" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_11499.pt, hf_model_0002_11499.pt]" RUN_TAG="s1500" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#wait


# 5/20/24 (wikitext-2-v1)

#cd /home/andrewor/local/ao
#git checkout qat-test
#cd /home/andrewor/local/torchtune

#export TIMESTAMP="1716249131"
#export LLAMA_VERSION=2
#export ENABLE_FAKE_QUANT_STEP=1000
#export BATCH_SIZE=4
#export NUM_EPOCHS=3
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

#echo -e "\n\n\n=== Running full ==="
#RUN_TAG="8da4w" ./run_it.sh full
#echo -e "\n\n\n=== Running QAT ==="
#RUN_TAG="8da4w" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on vanilla 8da4w ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on vanilla 8da4w (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="8da4w_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="8da4w_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="8da4w_e2" ./eval_it.sh $EXP_DIR &
#wait


# Try 3b weight only

#cd /home/andrewor/local/ao
#git checkout 3b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo -e "\n\n\n=== Running QAT 3w ==="
#RUN_TAG="3w" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on 3w ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_3w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on 3w (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="3w_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="3w_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="3w_e2" ./eval_it.sh $EXP_DIR &
#wait


# Try 2b weight only

#cd /home/andrewor/local/ao
#git checkout 2b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo -e "\n\n\n=== Running QAT 2w ==="
#RUN_TAG="2w" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on 2w ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_2w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on 2w (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="2w_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="2w_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="2w_e2" ./eval_it.sh $EXP_DIR &
#wait


# Try 2b weight only gs32

#cd /home/andrewor/local/ao
#git checkout 2b-weight-only
#cd /home/andrewor/local/torchtune
#
#echo -e "\n\n\n=== Running QAT 2w gs32 ==="
#RUN_TAG="2w_gs32" ./run_it.sh qat
#
#echo -e "\n\n\n=== Running eval on 2w gs32 ==="
#EXP_DIR="/home/andrewor/local/logs/tune/qat_llama2_${TIMESTAMP}_2w_gs32"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="e2" ./eval_it.sh $EXP_DIR &
#wait
#
#echo -e "\n\n\n=== Running eval on 2w gs32 (baseline) ==="
#EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_8da4w"
#CUDA_VISIBLE_DEVICES=0,1 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="2w_gs32_e0" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=2,3 CHECKPOINT_FILES="[hf_model_0001_1.pt, hf_model_0002_1.pt]" RUN_TAG="2w_gs32_e1" ./eval_it.sh $EXP_DIR &
#CUDA_VISIBLE_DEVICES=4,5 CHECKPOINT_FILES="[hf_model_0001_2.pt, hf_model_0002_2.pt]" RUN_TAG="2w_gs32_e2" ./eval_it.sh $EXP_DIR &
#wait
