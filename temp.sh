# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

BASE_DIR="/home/andrewor/local/logs/tune"
FULL_DIR="${BASE_DIR}/full_finetune"
QAT_SUBCLASS_DIR="${BASE_DIR}/4w_qat_subclass"
QAT_MODULE_SWAP_DIR="${BASE_DIR}/4w_qat_module_swap"

#rm -rf "$FULL_DIR"
#rm -rf "$QAT_SUBCLASS_DIR"
#rm -rf "$QAT_MODULE_SWAP_DIR"
#mkdir -p "$FULL_DIR"
#mkdir -p "$QAT_SUBCLASS_DIR"
#mkdir -p "$QAT_MODULE_SWAP_DIR"

#echo "Running QAT fine-tuning..."
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 --rdzv_endpoint="localhost:8900" qat_distributed --config llama3/8B_qat_full \
#    batch_size=8 \
#    fake_quant_after_n_steps=1000 \
#    checkpointer.output_dir="$QAT_SUBCLASS_DIR" \
#    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQATQuantizer \
#    quantizer.groupsize=128 \
#    > "${QAT_SUBCLASS_DIR}/run.log" 2>&1
#
#echo "Running QAT fine-tuning (module swap)..."
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 --rdzv_endpoint="localhost:8901" qat_distributed --config llama3/8B_qat_full \
#    batch_size=8 \
#    fake_quant_after_n_steps=1000 \
#    checkpointer.output_dir="$QAT_MODULE_SWAP_DIR" \
#    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQATQuantizerModuleSwap \
#    quantizer.groupsize=128 \
#    > "${QAT_MODULE_SWAP_DIR}/run.log" 2>&1
#
#echo "Running full fine-tuning..."
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 tune run --nnodes 1 --nproc_per_node 6 --rdzv_endpoint="localhost:8899" full_finetune_distributed --config llama3/8B_qat_full \
#    batch_size=8 \
#    checkpointer.output_dir="$FULL_DIR" \
#    > "${FULL_DIR}/run.log" 2>&1
#
#echo "Running quantize..."
#
#CUDA_VISIBLE_DEVICES=2 tune run quantize --config recipes/configs/quantization.yaml \
#    model._component_=torchtune.models.llama3.llama3_8b \
#    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer \
#    quantizer.groupsize=128 \
#    checkpointer._component_=torchtune.training.FullModelMetaCheckpointer \
#    checkpointer.checkpoint_dir="$FULL_DIR" \
#    checkpointer.output_dir="$FULL_DIR" \
#    checkpointer.checkpoint_files=[meta_model_2.pt] \
#    checkpointer.model_type=LLAMA3 \
#    > "${FULL_DIR}/quantize.log" 2>&1 &
#
#CUDA_VISIBLE_DEVICES=3 tune run quantize --config recipes/configs/quantization.yaml \
#    model._component_=torchtune.models.llama3.llama3_8b \
#    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer \
#    quantizer.groupsize=128 \
#    checkpointer._component_=torchtune.training.FullModelMetaCheckpointer \
#    checkpointer.checkpoint_dir="$QAT_SUBCLASS_DIR" \
#    checkpointer.output_dir="$QAT_SUBCLASS_DIR" \
#    checkpointer.checkpoint_files=[meta_model_2.pt] \
#    checkpointer.model_type=LLAMA3 \
#    > "${QAT_SUBCLASS_DIR}/quantize.log" 2>&1 &
#
#CUDA_VISIBLE_DEVICES=4 tune run quantize --config recipes/configs/quantization.yaml \
#    model._component_=torchtune.models.llama3.llama3_8b \
#    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer \
#    quantizer.groupsize=128 \
#    checkpointer._component_=torchtune.training.FullModelMetaCheckpointer \
#    checkpointer.checkpoint_dir="$QAT_MODULE_SWAP_DIR" \
#    checkpointer.output_dir="$QAT_MODULE_SWAP_DIR" \
#    checkpointer.checkpoint_files=[meta_model_2.pt] \
#    checkpointer.model_type=LLAMA3 \
#    > "${QAT_MODULE_SWAP_DIR}/quantize.log" 2>&1 &
#
#wait
#
#echo "Running eval..."

CUDA_VISIBLE_DEVICES=2 tune run eleuther_eval --config eleuther_evaluation \
    tasks="[hellaswag, wikitext]" \
    model._component_=torchtune.models.llama3.llama3_8b \
    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer \
    quantizer.groupsize=128 \
    checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
    checkpointer.checkpoint_dir="$FULL_DIR" \
    checkpointer.output_dir="$FULL_DIR" \
    checkpointer.checkpoint_files=[meta_model_2-4w.pt] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    > "${FULL_DIR}/eval.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 tune run eleuther_eval --config eleuther_evaluation \
    tasks="[hellaswag, wikitext]" \
    model._component_=torchtune.models.llama3.llama3_8b \
    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer \
    quantizer.groupsize=128 \
    checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
    checkpointer.checkpoint_dir="$QAT_SUBCLASS_DIR" \
    checkpointer.output_dir="$QAT_SUBCLASS_DIR" \
    checkpointer.checkpoint_files=[meta_model_2-4w.pt] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    > "${QAT_SUBCLASS_DIR}/eval.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 tune run eleuther_eval --config eleuther_evaluation \
    tasks="[hellaswag, wikitext]" \
    model._component_=torchtune.models.llama3.llama3_8b \
    quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer \
    quantizer.groupsize=128 \
    checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
    checkpointer.checkpoint_dir="$QAT_MODULE_SWAP_DIR" \
    checkpointer.output_dir="$QAT_MODULE_SWAP_DIR" \
    checkpointer.checkpoint_files=[meta_model_2-4w.pt] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    > "${QAT_MODULE_SWAP_DIR}/eval.log" 2>&1 &

wait
