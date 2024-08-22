# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FULL_DIR="/home/andrewor/local/logs/tune/full_llama3_8212024"

mkdir -p "${FULL_DIR}"

# Fine-tune

echo "Running full finetune"
export NCCL_DESYNC_DEBUG=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 full_finetune_distributed --config recipes/configs/my_8B_full.yaml \
  checkpointer.output_dir="${FULL_DIR}" \
  output_dir="${FULL_DIR}" \
  > "${FULL_DIR}/run.log" 2>&1

# Quantize

echo "Quantizing full finetune"
CUDA_VISIBLE_DEVICES=1 tune run quantize --config recipes/configs/my_quantization.yaml \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer \
  > "${FULL_DIR}/quantize.log" 2>&1

# Eval

echo "Evaluating full finetune quantized"
CUDA_VISIBLE_DEVICES=1 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=[meta_model_0-8da4w.pt] \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer \
  quantizer.groupsize=256 \
  > "${FULL_DIR}/eval_quantized.log" 2>&1

echo "Evaluating full finetune float"
CUDA_VISIBLE_DEVICES=2 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelMetaCheckpointer \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=[meta_model_0.pt] \
  checkpointer.output_dir="${FULL_DIR}" \
  > "${FULL_DIR}/eval_float.log" 2>&1

wait
