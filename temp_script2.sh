# ============== #
#  on devgpu023  #
# ============== #

# 5/20/24 (overnight)

export TIMESTAMP="1716249131"
export LLAMA_VERSION=2
export BATCH_SIZE=4
export NUM_EPOCHS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export SKIP_QUANTIZE="true"

RUN_TAG="checkpoint_every_100_steps" CHECKPOINT_EVERY_N_STEPS="100" ./run_it.sh full

echo -e "=== Eval ==="
EXP_DIR="/home/andrewor/local/logs/tune/full_llama2_${TIMESTAMP}_checkpoint_every_100_steps"
CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_10099.pt, hf_model_0002_10099.pt]" RUN_TAG="s100" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_10199.pt, hf_model_0002_10199.pt]" RUN_TAG="s200" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_10299.pt, hf_model_0002_10299.pt]" RUN_TAG="s300" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_10399.pt, hf_model_0002_10399.pt]" RUN_TAG="s400" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_10499.pt, hf_model_0002_10499.pt]" RUN_TAG="s500" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_10599.pt, hf_model_0002_10599.pt]" RUN_TAG="s600" ./eval_it.sh $EXP_DIR &
wait
CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_10699.pt, hf_model_0002_10699.pt]" RUN_TAG="s700" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_10799.pt, hf_model_0002_10799.pt]" RUN_TAG="s800" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_10899.pt, hf_model_0002_10899.pt]" RUN_TAG="s900" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_10999.pt, hf_model_0002_10999.pt]" RUN_TAG="s1000" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=4 CHECKPOINT_FILES="[hf_model_0001_11099.pt, hf_model_0002_11099.pt]" RUN_TAG="s1100" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=5 CHECKPOINT_FILES="[hf_model_0001_11199.pt, hf_model_0002_11199.pt]" RUN_TAG="s1200" ./eval_it.sh $EXP_DIR &
wait
CUDA_VISIBLE_DEVICES=0 CHECKPOINT_FILES="[hf_model_0001_11299.pt, hf_model_0002_11299.pt]" RUN_TAG="s1300" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=1 CHECKPOINT_FILES="[hf_model_0001_11399.pt, hf_model_0002_11399.pt]" RUN_TAG="s1400" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=2 CHECKPOINT_FILES="[hf_model_0001_11499.pt, hf_model_0002_11499.pt]" RUN_TAG="s1500" ./eval_it.sh $EXP_DIR &
CUDA_VISIBLE_DEVICES=3 CHECKPOINT_FILES="[hf_model_0001_0.pt, hf_model_0002_0.pt]" RUN_TAG="e0" ./eval_it.sh $EXP_DIR &
wait


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

