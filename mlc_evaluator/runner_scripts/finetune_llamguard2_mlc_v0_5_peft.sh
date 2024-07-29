#!/bin/bash

# Define variables
NNODES=1
NPROC_PER_NODE=8
MODEL_NAME="meta-llama/Meta-Llama-Guard-2-8B"
DATASET="mlc_dataset"
OUTPUT_DIR="trained_model_checkpoints"
PATH_TO_FINETUNE="model/mlc_llama_guard/meta_llama_recipes/recipes/quickstart/finetuning/"


# For multi GPU parameter efficient finetuning
torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE $PATH_TO_FINETUNE/finetuning.py \
  --enable_fsdp --enable_fsdp \
  --model_name $MODEL_NAME \
  --pure_bf16 \
  --use_fast_kernels \
  --use_peft \
  --peft_method lora \
  --dataset $DATASET \
  --output_dir $OUTPUT_DIR \
  --save_metrics \
  --save_model