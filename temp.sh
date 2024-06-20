# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p /home/andrewor/local/logs/tune/c4-llama3-full
mkdir -p /home/andrewor/local/logs/tune/c4-llama3-qat

# Fine-tune

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 full_finetune_distributed --config recipes/configs/my_8B_full.yaml > /home/andrewor/local/logs/tune/c4-llama3-full/run.log 2>&1
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 qat_distributed --config recipes/configs/my_8B_qat_full.yaml > /home/andrewor/local/logs/tune/c4-llama3-qat/run.log 2>&1

# Quantize

CUDA_VISIBLE_DEVICES=0 tune run quantize --config recipes/configs/my_quantization.yaml \
  checkpointer.checkpoint_dir=/home/andrewor/local/logs/tune/c4-llama3-qat \
  checkpointer.output_dir=/home/andrewor/local/logs/tune/c4-llama3-qat \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQATQuantizer \
  > /home/andrewor/local/logs/tune/c4-llama3-qat/quantize.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 tune run quantize --config recipes/configs/my_quantization.yaml \
  checkpointer.checkpoint_dir=/home/andrewor/local/logs/tune/c4-llama3-full \
  checkpointer.output_dir=/home/andrewor/local/logs/tune/c4-llama3-full \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer \
  > /home/andrewor/local/logs/tune/c4-llama3-full/quantize.log 2>&1 &

wait

# Eval

CUDA_VISIBLE_DEVICES=0 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
  checkpointer.checkpoint_dir=/home/andrewor/local/logs/tune/c4-llama3-qat \
  checkpointer.checkpoint_files=[meta_model_0-8da4w-qat.pt] \
  checkpointer.output_dir=/home/andrewor/local/logs/tune/c4-llama3-qat \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer \
  quantizer.groupsize=256 \
  > /home/andrewor/local/logs/tune/c4-llama3-qat/eval_quantized.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
  checkpointer.checkpoint_dir=/home/andrewor/local/logs/tune/c4-llama3-full \
  checkpointer.checkpoint_files=[meta_model_0-8da4w.pt] \
  checkpointer.output_dir=/home/andrewor/local/logs/tune/c4-llama3-full \
  quantizer._component_=torchtune.utils.quantization.Int8DynActInt4WeightQuantizer \
  quantizer.groupsize=256 \
  > /home/andrewor/local/logs/tune/c4-llama3-full/eval_quantized.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 tune run eleuther_eval --config recipes/configs/my_eleuther_evaluation.yaml \
  checkpointer._component_=torchtune.utils.FullModelMetaCheckpointer \
  checkpointer.checkpoint_dir=/home/andrewor/local/logs/tune/c4-llama3-full \
  checkpointer.checkpoint_files=[meta_model_0.pt] \
  checkpointer.output_dir=/home/andrewor/local/logs/tune/c4-llama3-full \
  > /home/andrewor/local/logs/tune/c4-llama3-full/eval_float.log 2>&1

wait
