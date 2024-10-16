#LOG_DIR="/home/andrewor/local/logs/tune/qt_w8_saved"
#CHECKPOINT_FILE="meta_model_1.pt"
LOG_DIR="/tmp/Meta-Llama-3-8B-Instruct/original/"
CHECKPOINT_FILE="consolidated.00.pth"

CUDA_VISIBLE_DEVICES=1 tune run eleuther_eval --config eleuther_evaluation \
    batch_size=1 \
    model._component_=torchtune.models.llama3.llama3_8b \
    checkpointer._component_=torchtune.training.FullModelMetaCheckpointer \
    checkpointer.checkpoint_dir="$LOG_DIR" \
    checkpointer.output_dir="$LOG_DIR" \
    checkpointer.checkpoint_files=["$CHECKPOINT_FILE"] \
    checkpointer.model_type=LLAMA3 \
    tokenizer._component_=torchtune.models.llama3.llama3_tokenizer \
    tokenizer.path=/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    tasks=[wikitext] \
    > "$LOG_DIR"/eval.log 2>&1
