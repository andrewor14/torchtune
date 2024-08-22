# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FULL_DIR="/home/andrewor/local/logs/tune/full_eval_test_8222024"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-consolidated.00.pth}"
DEVICE0="${DEVICE0:-0}"
DEVICE1="${DEVICE1:-1}"

NEW_CHECKPOINT_NAME="$(echo "$CHECKPOINT_NAME" | awk -F'.' '{print $1}')"
RUN_TAG="_${NEW_CHECKPOINT_NAME}"

# Quantize

echo "Quantizing full finetune"
CUDA_VISIBLE_DEVICES="$DEVICE0" tune run quantize --config recipes/configs/my_quantization.yaml \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=["${CHECKPOINT_NAME}"] \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer \
  > "${FULL_DIR}/quantize${RUN_TAG}.log" 2>&1 &

CUDA_VISIBLE_DEVICES="$DEVICE1" tune run quantize --config recipes/configs/my_quantization.yaml \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=["${CHECKPOINT_NAME}"] \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.NewInt8DynActInt4WeightQuantizer \
  > "${FULL_DIR}/quantize_subclass${RUN_TAG}.log" 2>&1 &

wait


# Eval

echo "Evaluating full finetune quantized"
CUDA_VISIBLE_DEVICES="$DEVICE0" tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=["${NEW_CHECKPOINT_NAME}"-8da4w.pt] \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer \
  quantizer.groupsize=256 \
  > "${FULL_DIR}/eval_quantized${RUN_TAG}.log" 2>&1 &

CUDA_VISIBLE_DEVICES="$DEVICE1" tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
  checkpointer.checkpoint_dir="${FULL_DIR}" \
  checkpointer.checkpoint_files=["${NEW_CHECKPOINT_NAME}"-new-8da4w.pt] \
  checkpointer.output_dir="${FULL_DIR}" \
  quantizer._component_=torchtune.utils.quantization.NewInt8DynActInt4WeightQuantizer \
  quantizer.groupsize=256 \
  > "${FULL_DIR}/eval_quantized_subclass${RUN_TAG}.log" 2>&1

wait
