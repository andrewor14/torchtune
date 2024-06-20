# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

TIMESTAMP="${TIMESTAMP:-`date +%s`}"
QAT_DIR="/home/andrewor/local/logs/tune/qat_llama3_4w_${TIMESTAMP}"
FULL_DIR="/home/andrewor/local/logs/tune/full_llama3_4w_${TIMESTAMP}"

mkdir -p "${QAT_DIR}"
mkdir -p "${FULL_DIR}"

# Fine-tune

echo "Running QAT"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 qat_distributed --config recipes/configs/my_8B_qat_full.yaml \
  checkpointer.output_dir="${QAT_DIR}" \
  output_dir="${QAT_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int4WeightOnlyQATQuantizer \
  > "${QAT_DIR}/run.log" 2>&1

echo "Running full finetune"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 full_finetune_distributed --config recipes/configs/my_8B_full.yaml \
  checkpointer.output_dir="${FULL_DIR}" \
  output_dir="${FULL_DIR}" \
  > "${FULL_DIR}/run.log" 2>&1

# Quantize

echo "Quantizing QAT"
CUDA_VISIBLE_DEVICES=0 tune run quantize --config recipes/configs/my_quantization.yaml \
  checkpointer.checkpoint_dir="${QAT_DIR}" \
  checkpointer.output_dir="${QAT_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int4WeightOnlyQATQuantizer \
  > "${QAT_DIR}/quantize.log" 2>&1 &

echo "Quantizing full finetune"
CUDA_VISIBLE_DEVICES=1 tune run quantize --config recipes/configs/my_quantization.yaml \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int4WeightOnlyQuantizer \
  > "${FULL_DIR}/quantize.log" 2>&1 &

wait

# Eval

echo "Evaluating QAT quantized"
CUDA_VISIBLE_DEVICES=0 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
  checkpointer.checkpoint_dir="${QAT_DIR}" \
  checkpointer.checkpoint_files=[meta_model_0-4w.pt] \
  checkpointer.output_dir="${QAT_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int4WeightOnlyQuantizer \
  quantizer.groupsize=256 \
  > "${QAT_DIR}/eval_quantized.log" 2>&1 &

echo "Evaluating full finetune quantized"
CUDA_VISIBLE_DEVICES=1 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=[meta_model_0-4w.pt] \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int4WeightOnlyQuantizer \
  quantizer.groupsize=256 \
  > "${FULL_DIR}/eval_quantized.log" 2>&1 &

echo "Evaluating full finetune float"
CUDA_VISIBLE_DEVICES=2 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelMetaCheckpointer \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=[meta_model_0.pt] \
  checkpointer.output_dir="${FULL_DIR}" \
  > "${FULL_DIR}/eval_float.log" 2>&1

wait
